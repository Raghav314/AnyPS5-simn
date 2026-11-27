import argparse
import ctypes
import ctypes.util
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


def fields(words):
    return {key: value for key, _, value in (word.partition("=") for word in words)}


def read_manifest(path):
    modules = {}
    pipelines = []
    for line in path.read_text().splitlines():
        if line.startswith("M "):
            _, ident, stage, *items = line.split()
            if ident in modules:
                raise ValueError(f"duplicate module {ident}")
            modules[ident] = (stage, fields(items), line)
        elif line.startswith("P "):
            pipelines.append(line)
    return modules, pipelines


def inventory(modules):
    stages = defaultdict(lambda: defaultdict(list))
    for stage, values, _ in modules.values():
        expanded = {}
        for key, value in values.items():
            if key in ("code",):
                continue
            if key in ("sb", "si", "ss"):
                slots = value.split(",")
                expanded[key + ".length"] = len(slots)
                for index, slot in enumerate(slots):
                    for item in slot.split("/"):
                        if item != "-":
                            name, _, attr = item.partition(":")
                            expanded.setdefault(key + "." + name, []).append((index, attr))
            else:
                expanded[key] = value
        stages[stage][values["code"]].append(expanded)
    result = {}
    for stage, by_code in stages.items():
        varied = Counter()
        for rows in by_code.values():
            for key in set().union(*(row.keys() for row in rows)):
                if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
                    varied[key] += 1
        result[stage] = dict(codes=len(by_code), modules=sum(map(len, by_code.values())),
                             multiple=sum(len(rows) > 1 for rows in by_code.values()),
                             max=max(map(len, by_code.values())), varying=dict(sorted(varied.items())))
    return result


def program_hash(values, hash64):
    if "back" not in values:
        return values["code"]
    pair = struct.pack("<QQ", int(values["code"], 16), int(values["back"], 16))
    return f"{hash64(pair, len(pair)):016x}"


def pipeline_modules(line):
    opts = fields(line.split()[1:])
    return [opts[key] for key in ("v", "h", "d", "p") if key in opts]


def subset_store(store, output, codes):
    remaining = set(codes)
    with store.open("rb") as src, output.open("xb") as dst:
        magic = src.read(8)
        if magic != struct.pack("<Q", 0x3145444F4359544B):
            raise ValueError("unsupported store")
        dst.write(magic)
        while header := src.read(16):
            digest, words, size = struct.unpack("<QII", header)
            data = src.read(size)
            if not size or not words or len(data) != size:
                raise ValueError("invalid record")
            digest = f"{digest:016x}"
            if digest in remaining:
                remaining.remove(digest)
                dst.write(header)
                dst.write(data)
    if remaining:
        raise ValueError(f"{len(remaining)} missing codes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--use", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modules, pipes = read_manifest(args.manifest)
    print(json.dumps(inventory(modules), sort_keys=True))
    print("pipelines", len(pipes), "manifest-sha256", hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    if args.output is None:
        return
    if not args.use or not args.store:
        parser.error("--output requires --use and --store")
    sessions = args.use.read_text().splitlines()
    if sessions[:2] != ["# kyty shader use order v1", "S"] or sessions.count("S") != 1:
        raise ValueError("expected a single use-order session")
    used = set(sessions[2:])
    xxh = ctypes.CDLL(ctypes.util.find_library("xxhash")).XXH3_64bits
    xxh.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    xxh.restype = ctypes.c_uint64
    observed = {ident: m for ident, m in modules.items() if program_hash(m[1], xxh) in used}
    selected = {ident: m for ident, m in observed.items() if m[0] != "cs"}
    matched = {program_hash(m[1], xxh) for m in observed.values()}
    print("used", len(used), "in-manifest", len(matched), "unmatched", sorted(used - matched))
    print("candidate-modules", json.dumps(inventory(selected), sort_keys=True))
    selected_pipes = [p for p in pipes if all(ident in selected for ident in pipeline_modules(p))]
    print("candidate-pipelines", len(selected_pipes))
    args.output.mkdir()
    cache = args.output / "_PipelineCache"
    cache.mkdir()
    rows = ["# kyty pipeline manifest v1"] + [m[2] for m in selected.values()]
    (args.output / "evaluation.kpm").write_text("\n".join(rows + selected_pipes) + "\n")
    (cache / "offline_1.kpm").write_text("\n".join(rows) + "\n")
    codes = {m[1][key] for m in selected.values() for key in ("code", "back") if key in m[1]}
    subset_store(args.store, cache / "offline_1.code", codes)


if __name__ == "__main__":
    main()
