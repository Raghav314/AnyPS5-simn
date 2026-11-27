import argparse
import json
from itertools import pairwise
from pathlib import Path

from bind import module_line
from catalog import xxhash
from session import fields, pipeline_modules, read_manifest


HEADER = "# kyty pipeline manifest v1"


def sampler_neighbors(layout):
    slots = layout.split(",")
    roots = {}
    children = {}
    mapping_end = 0
    for index, slot in enumerate(slots):
        if slot == "-":
            if children:
                return
            continue
        values = {}
        for field in slot.split("/"):
            key, _, value = field.partition(":")
            if key not in ("ir", "im", "is") or key in values or not value.isascii() or not value.isdecimal():
                return
            if len(value) > 10 or int(value) > 0xFFFFFFFF:
                return
            values[key] = int(value)
        root = values.get("ir")
        if root == index:
            iterations = values.get("is", 0)
            mapping = values.get("im", 0)
            if children or not 1 <= iterations <= 32 or mapping < mapping_end:
                return
            roots[root] = 1 << (iterations - 1)
            mapping_end = mapping + 1 + 2 * roots[root]
            if mapping_end > 0xFFFFFFFF:
                return
        elif root in roots and set(values) == {"ir"}:
            if children and root < next(reversed(children)):
                return
            group = children.setdefault(root, [])
            group.append(index)
            if len(group) > roots[root]:
                return
        else:
            return
    if roots.keys() != children.keys():
        return

    for left, right in pairwise(children):
        first, second = children[left], children[right]
        if len(first) > 1 and len(second) < roots[right]:
            candidate = slots.copy()
            candidate[first[-1]] = f"ir:{right}"
            yield ",".join(candidate)
        if len(second) > 1 and len(first) < roots[left]:
            candidate = slots.copy()
            candidate[second[0]] = f"ir:{left}"
            yield ",".join(candidate)


def shipped_stage_pairs(inventory):
    return {tuple(shaders[stage]["code"] for stage in ("vs", "ps"))
            for bundle in inventory["bundles"] for effect in bundle["effects"] for row in effect["passes"]
            if set(shaders := row["shaders"]) == {"vs", "ps"}}


def pipeline_fields(line):
    return fields(line.split()[1:])


def pipeline_identity(line):
    return tuple(sorted(pipeline_fields(line).items()))


def build_candidates(modules, pipelines, pairs, hash64):
    known = {pipeline_identity(line) for line in pipelines}
    candidates = {}
    replacements = {}
    generated_modules = {}
    for row in pipelines:
        options = pipeline_fields(row)
        if "h" in options or "d" in options or not {"v", "p"} <= options.keys():
            continue
        vertex = modules[options["v"]]
        pixel = modules[options["p"]]
        if vertex[0] != "vs" or pixel[0] != "ps" or "back" in vertex[1] or "back" in pixel[1]:
            continue
        if (vertex[1]["code"], pixel[1]["code"]) not in pairs:
            continue

        for slot in ("v", "p"):
            ident = options[slot]
            if ident not in replacements:
                stage, values, source = modules[ident]
                variants = replacements[ident] = []
                if values.get("ss") and module_line(stage, values, hash64) == source:
                    for layout in sampler_neighbors(values["ss"]):
                        variant = module_line(stage, values | {"ss": layout}, hash64)
                        variant_id = variant.split()[1]
                        if ((variant_id in modules and modules[variant_id][2] != variant) or
                            generated_modules.setdefault(variant_id, variant) != variant):
                            raise ValueError("conflicting sampler variant module identity")
                        variants.append(variant_id)
            for variant_id in replacements[ident]:
                candidate = " ".join(f"{slot}={variant_id}" if word == f"{slot}={ident}" else word
                                     for word in row.split())
                identity = pipeline_identity(candidate)
                if identity not in known:
                    candidates.setdefault(identity, candidate)
    return generated_modules, candidates


def compare_holdout(path, modules, candidates):
    held_modules, held_pipelines = read_manifest(path)
    held_keys = {pipeline_identity(line) for line in held_pipelines}
    return sum(key in held_keys and all(ident in held_modules and modules[ident] == held_modules[ident][2]
                                       for ident in pipeline_modules(row))
               for key, row in candidates.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate offline graphics candidates for adjacent indirect sampler roots")
    parser.add_argument("inventory", type=Path, help="shipped graphics inventory written by bundle.py")
    parser.add_argument("manifest", type=Path, help="existing exact graphics recipes used for the other pipeline inputs")
    parser.add_argument("--output", required=True, type=Path, help="new candidate .kpm file using existing shader code")
    parser.add_argument("--limit", type=int, help="maximum number of candidate pipelines to emit")
    parser.add_argument("--held-out", action="append", default=[], type=Path,
                        help="compare exact recipes against a later capture after writing the candidates")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("expected a positive pipeline limit")
    try:
        modules, pipelines = read_manifest(args.manifest)
        pairs = shipped_stage_pairs(json.loads(args.inventory.read_text()))
        generated, candidates = build_candidates(modules, pipelines, pairs, xxhash())
        selected = {key: candidates[key] for key in sorted(candidates)[:args.limit]}
        referenced = {ident for row in selected.values() for ident in pipeline_modules(row)}
        selected_modules = {ident: generated[ident] if ident in generated else modules[ident][2] for ident in referenced}
        with args.output.open("x", encoding="utf-8") as output:
            print(HEADER, file=output)
            for ident in sorted(selected_modules):
                print(selected_modules[ident], file=output)
            for row in selected.values():
                print(row, file=output)
        print(f"candidate pipelines {len(selected)}/{len(candidates)}, new modules {len(referenced - modules.keys())}")
        for path in args.held_out:
            print(f"held-out exact matches {compare_holdout(path, selected_modules, selected)}/{len(selected)}")
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"sampler candidates: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
