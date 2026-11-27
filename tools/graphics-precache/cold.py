import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from bind import RESOURCES, bind_recipe, interpolants, module_line, producer_key, read_recipes, write_batch
from catalog import xxhash
from interfaces import STAGE_CURSOR
from session import read_manifest


def target_classes(exports):
    choices = []
    for target in exports:
        mode = target["format"]
        if mode.startswith("32_"):
            choices.append(("", "uint", "sint"))
        elif mode in ("UINT16_ABGR", "SINT16_ABGR"):
            choices.append(("uint" if mode == "UINT16_ABGR" else "sint",))
        else:
            choices.append(("",))
    for classes in product(*choices):
        masks = {}
        for target, cls in zip(exports, classes):
            if cls:
                masks[cls] = masks.get(cls, 0) | 1 << target["index"]
        yield {k: f"{v:x}" for k, v in masks.items()}


def cold_shapes(recipe, layouts, resources):
    stage = recipe["stage"]
    if stage != "ps":
        if "dyn" in recipe["fixed"]:
            return
        geometries = recipe.get("assembly", [None])
        for shape in geometries:
            for indirect in ({}, {"idraw": ""}):
                draw = dict(indirect)
                if shape is not None:
                    draw["primitive"] = shape["mprim"]
                yield bind_recipe(recipe, draw, resources)
                if shape is not None and int(shape.get("mops", 3)) > 1:
                    yield bind_recipe(recipe, dict(draw, provoking_last=True), resources)
        return
    links = {}
    for producer in layouts if recipe["inputs"] else ((),):
        for mapping in ("0100", "10000", "default"):
            value = interpolants(recipe["inputs"], producer, mapping)
            links.setdefault(value, (producer, mapping))
    for targets in target_classes(recipe["exports"]):
        for parameter in ({}, {"param": "Rectangle"}, {"param": "LastVertex"}):
            draw = dict(targets, **parameter)
            for producer, mapping in links.values():
                yield bind_recipe(recipe, draw, resources, producer, mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipes", type=Path)
    parser.add_argument("--use", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--descriptor-free", action="store_true")
    args = parser.parse_args()
    if args.output and not args.store:
        parser.error("--output requires --store")
    catalog = read_recipes(args.recipes)
    used = None
    if args.use:
        records = args.use.read_text().splitlines()
        if records[:2] != ["# kyty shader use order v1", "S"] or records.count("S") != 1:
            raise ValueError("expected a single use-order session")
        used = set(records[2:])
    layouts = {producer_key(r["outputs"]): r["outputs"] for (s, _), rows in catalog.items()
               if s != "ps" for r in rows}
    hash64 = xxhash()
    wanted = defaultdict(Counter)
    requests = {}
    if args.reference:
        requests, _ = read_manifest(args.reference)
        for stage, old, _ in requests.values():
            values = dict(old, cur=str(STAGE_CURSOR[stage]))
            if not args.descriptor_free:
                values = {k: v for k, v in values.items() if k not in RESOURCES}
                values["gg"] = ""
            wanted[stage, old["code"]][module_line(stage, values, hash64)] += 1
    generated, hits, code_hits = Counter(), Counter(), Counter()
    output = set()
    candidates = 0
    for (stage, code), recipes in catalog.items():
        if used is not None and code not in used:
            continue
        lines = set()
        for recipe in recipes:
            if args.descriptor_free and recipe["isa"]["resource_operations"]:
                continue
            resources = {} if args.descriptor_free else {"gg": ""}
            for values in cold_shapes(recipe, layouts.values(), resources):
                lines.add(module_line(stage, values, hash64))
        if lines:
            generated[stage] += 1
            candidates += len(lines)
        matched = lines & wanted[stage, code].keys()
        if matched:
            code_hits[stage] += 1
            hits[stage] += sum(wanted[stage, code][m] for m in matched)
        if args.output:
            output.update(lines)
    print("programs with cold shapes", dict(generated), "candidate modules", candidates)
    if args.reference:
        print("matched programs", dict(code_hits), "requests", dict(hits), "of", len(requests))
    if args.output:
        write_batch(args.output, args.store, output)


if __name__ == "__main__":
    main()
