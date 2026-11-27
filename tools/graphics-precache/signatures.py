import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from catalog import xxhash
from session import pipeline_modules, read_manifest, subset_store


def creation_classes(path):
    classes = defaultdict(set)
    for line in path.open():
        code, kind, cx, sh, enable, scratch = json.loads(line)[:6]
        if kind not in (1, 2):
            continue
        regs = dict(sh)
        rsrc = regs.get(11 if kind == 1 else 139)
        if rsrc is None:
            continue
        stage = "ps" if kind == 1 else "mesh" if enable is not None and (enable & 32 or rsrc >> 19 & 255) else "vs"
        key = (stage, tuple(map(tuple, cx)), rsrc, enable, scratch)
        classes[code].add(key)
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("headers", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("use", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("expected a positive limit")
    classes = creation_classes(args.headers)
    donors = defaultdict(lambda: defaultdict(set))
    training, pipelines = read_manifest(args.train)
    for stage, fields, line in training.values():
        code = fields["code"]
        if stage == "cs" or "back" in fields:
            continue
        parts = line.split()
        signature = " ".join(field for field in parts[3:] if not field.startswith("code="))
        for key in classes[code]:
            if key[0] == stage:
                donors[key][signature].add(code)
    used = set(args.use.read_text().splitlines()[2:])
    hash64 = xxhash()
    modules = set()
    codes = set()
    for target in sorted(used):
        ranked = defaultdict(set)
        for key in classes[target]:
            for signature, programs in donors[key].items():
                ranked[(key[0], signature)].update(programs - {target})
        counts = Counter({s: len(p) for s, p in ranked.items() if p})
        for (stage, signature), _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:args.limit]:
            text = f"{stage} code={target} {signature}".rstrip()
            raw = text.encode()
            modules.add(f"M {hash64(raw, len(raw)) >> 16:012x} {text}")
            codes.add(target)
    args.output.mkdir()
    cache = args.output / "_PipelineCache"
    cache.mkdir()
    rows = ["# kyty pipeline manifest v1", *sorted(modules)]
    (cache / "offline_1.kpm").write_text("\n".join(rows) + "\n")
    predicted = {line.split()[1] for line in modules}
    warm = [line for line in pipelines if set(pipeline_modules(line)) <= predicted]
    (args.output / "evaluation.kpm").write_text("\n".join([*rows, *warm]) + "\n")
    subset_store(args.store, cache / "offline_1.code", codes)
    print("cross-program templates", len(modules), "target programs", len(codes))


if __name__ == "__main__":
    main()
