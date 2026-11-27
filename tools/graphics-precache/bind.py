import argparse
import json
from collections import defaultdict
from pathlib import Path

from catalog import xxhash
from session import subset_store


RESOURCES = ("rb", "rs", "gg", "sb", "si", "ss")
VERTEX_DRAW = ("ucp", "iv", "ii", "idraw", "clip", "cscale", "coffset")
PIXEL_DRAW = ("pbmask", "pbtests", "pblend", "unorm", "uint", "sint", "round", "param", "dual",
              "primid", "asrc", "emap")
COMMON = ("code", "ud", "cur", "rb", "rs", "gg")
PS = ("pbmask pbtests pblend wave scratch lds sysbase custom center sample centroid smask unorm uint sint "
      "round omode posx posy posz posw face faceall anc noper param kill zexp mexp sshade dual primid "
      "asrc early emap in").split()
VS = ("wave scratch lds vsout ucp fa fb dyn iv ii idraw clip cscale coffset mt mlds mscratch msub mwave "
      "mprim mops mppg mvpg mmaxv mmaxp mprov mfast attr vbuf").split()


def read_recipes(path):
    recipes = defaultdict(list)
    for line in path.open():
        row = json.loads(line)
        if row["version"] != 1:
            raise ValueError("unsupported recipe schema")
        recipes[row["stage"], row["code"]].append(row)
    return recipes


def producer_key(outputs):
    return tuple((s["id"], s["vgpr"], s["half"]) for s in outputs)


def attachment_masks(exports, attachments, dual):
    if len(attachments) > 8:
        raise ValueError("too many color attachments")
    masks = {}
    for export in exports:
        slot = export["target"]
        index = export["index"]
        if slot is None or slot >= len(attachments) or attachments[slot] is None or (dual and index == 1):
            continue
        fmt = attachments[slot]
        if not isinstance(fmt, str) or not fmt:
            raise ValueError("expected the Vulkan format of the active color attachment")
        if "Uint" in fmt or "Sint" in fmt:
            key = "uint" if "Uint" in fmt else "sint"
            masks[key] = masks.get(key, 0) | 1 << index
        elif not any(t in fmt for t in ("Unorm", "Snorm", "Srgb", "Sfloat", "Ufloat", "Sscaled", "Uscaled")):
            raise ValueError("unknown color attachment numeric class")
    return {key: f"{mask:x}" for key, mask in masks.items()}


def interpolants(inputs, outputs, mapping):
    if mapping not in ("0100", "10000", "default"):
        raise ValueError("unknown interpolant mapping")
    sources = {}
    for item in outputs:
        sources.setdefault(item["id"], item)
    registers = []
    for item in inputs:
        source = sources.get(item["id"])
        half = item["half"]
        common = half & source["half"] if source else 0
        value = item["default"] << 8
        if half:
            value |= half << 24 | 1 << 19 | item["default_hi"] << 21
            if mapping == "10000" and half == 2:
                value &= ~0x300
                value |= item["default_hi"] << 8
                if not common & 2:
                    value |= 32
            else:
                if not common & 1:
                    value |= 32
                if not common & 2:
                    value |= 1 << 20
        elif item["custom"] or source is None:
            value |= 32
        if source is not None:
            value |= source["vgpr"] & 31
            if item["flat"] or item["custom"]:
                value |= 0x400
        if mapping == "default" and not half and item["flat"]:
            value |= 32
        registers.append(f"{value:x}")
    return ",".join(registers)


def vertex_attributes(recipe, draw):
    if "dyn" not in recipe["fixed"]:
        if "attr" in draw or "vbuf" in draw:
            raise ValueError("unexpected vertex tables")
        return {}
    attrs = [item.split(".") for item in draw["attr"].split(",")]
    buffers = [tuple(map(int, item.split("."))) for item in draw["vbuf"].split(",")]
    if len(attrs) != len(recipe["inputs"]) or not 0 < len(buffers) <= len(attrs):
        raise ValueError("vertex table size disagrees with semantics")
    for attr, semantic in zip(attrs, recipe["inputs"]):
        if len(attr) != 8:
            raise ValueError("incomplete vertex descriptor")
        start, size, attribute_id, fetch, index = map(int, attr[3:])
        if (start, size, attribute_id) != (semantic["vgpr"], semantic["elements"], semantic["id"]):
            raise ValueError("vertex destination disagrees with semantics")
        stride = int(attr[0], 16) >> 16 & 0x3fff
        if fetch not in (0, 1) or not 0 <= index < len(buffers) or buffers[index] != (stride, int(attr[1]), fetch):
            raise ValueError("vertex descriptor disagrees with binding")
    return {"attr": draw["attr"], "vbuf": draw["vbuf"]}


def bind_recipe(recipe, draw, resources, producer=None, mapping="0100"):
    if not resources.keys() <= set(RESOURCES):
        raise ValueError("unknown resource specialization")
    if "gg" in resources and resources != {"gg": ""}:
        raise ValueError("generic graphics requires runtime resources without exact specialization")
    stage = recipe["stage"]
    values = dict(recipe["fixed"], code=recipe["code"], cur=str(recipe["cursor"]), **resources)
    if stage == "ps":
        if not draw.keys() <= {*PIXEL_DRAW, "alpha_to_coverage", "attachments"}:
            raise ValueError("unknown pixel draw field")
        if draw.get("alpha_to_coverage") and recipe["allow_alpha_to_coverage"]:
            values.pop("early", None)
        values.update({k: draw[k] for k in PIXEL_DRAW if k in draw})
        if "attachments" in draw:
            if "uint" in draw or "sint" in draw:
                raise ValueError("supply either active color formats or resolved target classes")
            values.update(attachment_masks(recipe["exports"], draw["attachments"], "dual" in draw))
        if "dual" in draw and "omode" in values:
            modes = values["omode"].split(",")
            if len(modes) == 1:
                modes.append(modes[0])
            else:
                modes[1] = modes[0]
            values["omode"] = ",".join(modes)
        if recipe["inputs"]:
            if producer is None:
                raise ValueError("pixel inputs require producer semantics")
            values["in"] = interpolants(recipe["inputs"], producer, mapping)
        return values
    if not draw.keys() <= {*VERTEX_DRAW, "attr", "vbuf", "primitive", "provoking_last"}:
        raise ValueError("unknown vertex draw field")
    values.update({k: draw[k] for k in VERTEX_DRAW if k in draw})
    if stage == "vs":
        values.update(vertex_attributes(recipe, draw))
    else:
        shape = next((s for s in recipe["assembly"] if s["mprim"] == str(draw["primitive"])), None)
        if shape is None:
            raise ValueError("primitive does not satisfy mesh assembly")
        values.update(shape)
        size = int(shape.get("mops", 3))
        if draw.get("provoking_last") and size > 1:
            values["mprov"] = str(size - 1)
    return values


def module_line(stage, values, hash64):
    order = [*COMMON, *(PS if stage == "ps" else VS), "sb", "si", "ss"]
    if stage not in ("ps", "vs", "mesh") or values.keys() - set(order):
        raise ValueError("unknown module fields")
    words = [stage]
    for key in order:
        if key in values:
            if any(c.isspace() for c in values[key]):
                raise ValueError(f"unexpected whitespace in module field {key}")
            words.append(f"{key}={values[key]}" if values[key] != "" else key)
    shape = " ".join(words)
    raw = shape.encode()
    return f"M {hash64(raw, len(raw)) >> 16:012x} {shape}"


def write_batch(output, store, lines, pipelines=()):
    lines = sorted(set(lines))
    if not lines:
        raise ValueError("empty module batch")
    ids = {line.split()[1] for line in lines}
    if len(ids) != len(lines):
        raise ValueError("module identifier collision")
    codes = {line.split()[3].removeprefix("code=") for line in lines}
    output.mkdir()
    cache = output / "_PipelineCache"
    cache.mkdir()
    header = "# kyty pipeline manifest v1\n"
    manifest = header + "\n".join(lines) + "\n"
    (cache / "offline_1.kpm").write_text(manifest)
    (output / "evaluation.kpm").write_text(manifest + "".join(p + "\n" for p in pipelines))
    subset_store(store, cache / "offline_1.code", codes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipes", type=Path)
    parser.add_argument("bindings", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    recipes = read_recipes(args.recipes)
    hash64 = xxhash()
    lines = set()
    for line in args.bindings.open():
        request = json.loads(line)
        stage, code = request["stage"], request["code"]
        recipes_for_code = recipes[stage, code]
        if not recipes_for_code:
            raise ValueError(f"no recipe for {stage} {code}")
        accepted = set()
        errors = set()
        for recipe in recipes_for_code:
            try:
                values = bind_recipe(recipe, request["draw"], request["resources"],
                                     request.get("producer"), request.get("mapping", "0100"))
            except (KeyError, ValueError) as error:
                errors.add(str(error))
                continue
            accepted.add(module_line(stage, values, hash64))
        if not accepted:
            raise ValueError(f"no compatible recipe for {stage} {code}: {sorted(errors)}")
        lines.update(accepted)
    write_batch(args.output, args.store, lines)


if __name__ == "__main__":
    main()
