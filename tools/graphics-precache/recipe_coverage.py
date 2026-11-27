import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bind import (PIXEL_DRAW, RESOURCES, VERTEX_DRAW, attachment_masks, bind_recipe, module_line,
                  producer_key, read_recipes, write_batch)
from catalog import xxhash
from interfaces import STAGE_CURSOR
from session import fields, read_manifest


def replay_draw(stage, observed):
    names = PIXEL_DRAW if stage == "ps" else (*VERTEX_DRAW, "attr", "vbuf")
    draw = {key: observed[key] for key in names if key in observed}
    if stage == "mesh":
        draw.update(primitive=int(observed.get("mprim", 0)), provoking_last="mprov" in observed)
    if stage == "ps":
        draw["alpha_to_coverage"] = "early" not in observed
    return draw


def resolve(recipes, observed, producers):
    for recipe in recipes:
        if (values := resolve_interface(recipe, observed, producers)) is not None:
            return values
    return None


def resolve_interface(recipe, observed, producers):
    stage = recipe["stage"]
    draw = replay_draw(stage, observed)
    resources = {k: observed[k] for k in RESOURCES if k in observed}
    for producer in producers if stage == "ps" and recipe["inputs"] else ((),):
        for mapping in ("0100", "10000", "default") if stage == "ps" and recipe["inputs"] else ("0100",):
            try:
                values = bind_recipe(recipe, draw, resources, producer, mapping)
            except (KeyError, ValueError):
                continue
            if values == observed:
                return values
    return None


def remap_pipeline(line, ids):
    parts = line.split()
    for i in range(1, len(parts)):
        key, sep, value = parts[i].partition("=")
        if sep and key in ("v", "h", "d", "p"):
            parts[i] = key + "=" + ids[value]
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipes", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and not args.store:
        parser.error("--output requires --store")
    catalog = read_recipes(args.recipes)
    programs, pipelines = read_manifest(args.reference)
    layouts = {}
    for (stage, _), recipes in catalog.items():
        if stage != "ps":
            for r in recipes:
                layouts[producer_key(r["outputs"])] = r["outputs"]
    partners = defaultdict(dict)
    paired, target_hits, link_hits = 0, 0, 0
    for line in pipelines:
        opts = fields(line.split()[1:])
        if "p" in opts:
            stage, values, _ = programs[opts.get("d", opts["v"])]
            linked = {}
            for r in catalog[stage, values["code"]]:
                linked[producer_key(r["outputs"])] = r["outputs"]
            partners[opts["p"]].update(linked)
            _, pixel, _ = programs[opts["p"]]
            observed = dict(pixel, cur=str(STAGE_CURSOR["ps"]))
            pix = catalog["ps", pixel["code"]]
            attachments = [opts.get(f"c{i}") if int(opts.get(f"m{i}", "0"), 16) else None
                           for i in range(int(opts.get("cc", 0)))]
            expected = {k: pixel[k] for k in ("uint", "sint") if k in pixel}
            target_hits += any(attachment_masks(r["exports"], attachments, "dual" in pixel) == expected for r in pix)
            link_hits += resolve(pix, observed, linked.values()) is not None
            paired += 1
    module_hits, code_hits = Counter(), set()
    ids, reference, bound = {}, set(), set()
    hash64 = xxhash()
    partner_hits, partner_total = 0, 0
    for ident, (stage, old, _) in programs.items():
        observed = dict(old, cur=str(STAGE_CURSOR[stage]))
        line = module_line(stage, observed, hash64)
        ids[ident] = line.split()[1]
        reference.add(line)
        recipes = catalog[stage, old["code"]]
        if not recipes:
            continue
        resolved = None
        if stage == "ps" and ident in partners:
            partner_total += 1
            resolved = resolve(recipes, observed, partners[ident].values())
            partner_hits += resolved is not None
        if resolved is None:
            resolved = resolve(recipes, observed, layouts.values())
        if resolved is not None:
            module_hits[stage] += 1
            code_hits.add((stage, old["code"]))
            bound.add(module_line(stage, resolved, hash64))
    print("recorded requests", len(programs), "programs", len({(s, d["code"]) for s, d, _ in programs.values()}))
    print("resolved recipe requests", dict(module_hits), "programs", dict(Counter(s for s, _ in code_hits)))
    print("pixel requests with matching recorded partner", partner_hits, "/", partner_total)
    print("pipeline pairs with reconstructed interpolation", link_hits, "/", paired,
          "numeric classes from active attachments", target_hits, "/", paired)
    print("canonical module requests", len(reference), "resolved", len(bound))
    if args.output:
        args.output.mkdir()
        plans = list(dict.fromkeys(remap_pipeline(line, ids) for line in pipelines))
        write_batch(args.output / "reference", args.store, reference, plans)
        write_batch(args.output / "bound", args.store, bound)
        evaluation = args.output / "reference/evaluation.kpm"
        (args.output / "bound/evaluation.kpm").write_bytes(evaluation.read_bytes())
        (args.output / "requests.json").write_text(json.dumps(ids, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
