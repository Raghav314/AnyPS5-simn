import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from catalog import xxhash
from session import subset_store


def eligible_codes(path):
    allowed = {}
    for line in path.open():
        row = json.loads(line)
        decoded = row.get("decoded")
        if not decoded:
            continue
        code = row["code"]
        isa = decoded["isa"]
        memory = ("BUFFER_", "TBUFFER_", "S_BUFFER_", "IMAGE_", "S_LOAD", "FLAT_", "GLOBAL_", "SCRATCH_")
        good = (not decoded["unsupported"] and not decoded["calls"] and not decoded["swap_pc"]
                and "S_ENDPGM" in isa and not any(op.startswith(memory) for op in isa))
        allowed[code] = good and allowed.get(code, True)
    return {code for code, good in allowed.items() if good}


def pixel_interface(cx, rsrc, scratch):
    common = []
    if not {0x8f, 0x1b3, 0x1b4, 0x1b6, 0x1b8, 0x1c5, 0x203} <= cx.keys():
        return
    active = cx[0x1b3]
    if active != cx[0x1b4] or active & ~0xffff:
        return
    count = cx[0x1b6] & 63
    if count > 32:
        return
    if cx[0x1b6] & 0x8000:
        common.append("wave=32")
    if scratch:
        common.append(f"scratch={scratch}")
    lds = ((rsrc >> 8) & 255) * 128
    if lds:
        common.append(f"lds={lds}")
    base = 2 * (active & 0x77).bit_count() + 3 * bool(active & 8) + bool(active & 128)
    if base:
        common.append(f"sysbase={base}")
    if active & 2:
        common.append(f"center={2 * bool(active & 1)}")
    if active & 1:
        common.append("sample=0")
    if active & 4:
        common.append(f"centroid={2 * (active & 3).bit_count()}")
    if cx[0x8f] != 0xffffffff:
        common.append(f"smask={cx[0x8f]:x}")
    modes = [(cx[0x1c5] >> (i * 4)) & 15 for i in range(8)]
    while modes and modes[-1] == 0:
        modes.pop()
    if modes:
        common.append("omode=" + ",".join(map(str, modes)))
    flags = [(active & 0x100, "posx"), (active & 0x200, "posy"), (active & 0x400, "posz"),
             (active & 0x800, "posw"), (active & 0x1000, "face"), (cx[0x1b8] & (1 << 24), "faceall"),
             (active & 0x2000, "anc"), (active & 0x20, "noper")]
    common.extend(name for present, name in flags if present)
    control = cx[0x203]
    flags = [(control & 64, "kill"), (control & 1, "zexp"), (control & 256, "mexp"),
             (active & 17, "sshade"), ((control >> 4) & 3 == 1 and not control & 0x141, "early")]
    post = [name for present, name in flags if present]
    return common, post, count


def module_shapes(row):
    code, kind, cx, sh, enable, scratch = row[:6]
    cx, sh = dict(cx), dict(sh)
    if kind not in (1, 2):
        return
    rsrc = sh.get(11 if kind == 1 else 139)
    if rsrc is None:
        return
    user = ((rsrc >> 1) & 31) | (((rsrc >> 27) & 1) << 5)
    common = [f"{'ps' if kind == 1 else 'vs'} code={code}"]
    if user:
        common.append(f"ud={user}")
    if kind == 2:
        if enable is None or enable & 0x20 or (rsrc >> 19) & 255 or 0x207 not in cx:
            return
        for cursor in range(33):
            parts = list(common)
            if cursor:
                parts.append(f"cur={cursor}")
            if enable & 0x400000:
                parts.append("wave=32")
            if scratch:
                parts.append(f"scratch={scratch}")
            if cx[0x207]:
                parts.append(f"vsout={cx[0x207]:x}")
            yield " ".join(parts)
            yield " ".join([*parts, "idraw"])
        return
    pixel = pixel_interface(cx, rsrc, scratch)
    if pixel is None:
        return
    static, post, count = pixel
    common.extend(static)
    for parameter in ([], ["param=Rectangle"], ["param=LastVertex"]):
        for flat in (0, 0x400) if count else (0,):
            parts = [*common, *parameter, *post]
            if count:
                parts.append("in=" + ",".join(f"{i | flat:x}" for i in range(count)))
            yield " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("headers", type=Path)
    parser.add_argument("decoded", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--use", type=Path)
    args = parser.parse_args()
    eligible = eligible_codes(args.decoded)
    selected = set(args.use.read_text().splitlines()[2:]) if args.use else eligible
    catalog = defaultdict(set)
    for line in args.headers.open():
        row = json.loads(line)
        if row[0] in eligible:
            catalog[row[0]].update(module_shapes(row))
    total = Counter()
    for shapes in catalog.values():
        if shapes:
            total[next(iter(shapes)).split()[0]] += 1
    print("eligible by ISA", len(eligible), "programs with recipes", dict(total))
    catalog = {code: shapes for code, shapes in catalog.items() if shapes and code in selected}
    hash64 = xxhash()
    lines = []
    for code, shapes in sorted(catalog.items()):
        for shape in sorted(shapes):
            raw = shape.encode()
            lines.append(f"M {hash64(raw, len(raw)) >> 16:012x} {shape}")
    args.output.mkdir()
    cache = args.output / "_PipelineCache"
    cache.mkdir()
    (cache / "offline_1.kpm").write_text("\n".join(["# kyty pipeline manifest v1", *lines]) + "\n")
    subset_store(args.store, cache / "offline_1.code", catalog)
    print("selected", len(catalog), "module recipes", len(lines))


if __name__ == "__main__":
    main()
