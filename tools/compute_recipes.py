import argparse
import ctypes
import ctypes.util
import mmap
import os
import struct
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def code_index(path):
    records = {}
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if stream.read(8) != struct.pack("<Q", 0x3145444F4359544B):
            raise ValueError("unsupported code store")
        while header := stream.read(16):
            if len(header) != 16:
                raise ValueError("truncated code record")
            digest, words, compressed = struct.unpack("<QII", header)
            if not words or not compressed or compressed > size - stream.tell():
                raise ValueError("invalid code length")
            stream.seek(compressed, os.SEEK_CUR)
            if digest and records.setdefault(digest, words) != words:
                raise ValueError("inconsistent code length")
    return records


def read_recipe(data, header, header_size):
    end = header + header_size
    registers = header + 32 + struct.unpack_from("<q", data, header + 32)[0]
    specials = header + 40 + struct.unpack_from("<q", data, header + 40)[0]
    count = data[header + 92]
    if not header + 96 <= registers <= end - 8 * count or not header + 96 <= specials <= end - 20:
        return None
    values = dict(struct.iter_unpack("<II", data[registers:registers + 8 * count]))
    required = (0x207, 0x208, 0x209, 0x212, 0x213)
    if not all(key in values for key in required) or values.get(0x20C) != 0 or values.get(0x20D, 1) & 0xff:
        return None
    group = tuple(values[key] for key in required[:3])
    if not all(group) or group[0] * group[1] * group[2] > 1024:
        return None
    scratch = struct.unpack_from("<H", data, header + 84)[0]
    modifier = struct.unpack_from("<I", data, specials + 16)[0]
    return (*group, values[0x212], values[0x213], modifier, scratch)


HEADER_MAGIC = struct.pack("<II", 0x34333231, 24)


def header_offsets(data):
    cursor = 0
    while (header := data.find(HEADER_MAGIC, cursor)) >= 0:
        yield header, None
        cursor = header + len(HEADER_MAGIC)


def shader_record(data, header, code, hash_code):
    if not 0 <= header <= len(data) - 96 or data[header:header + 8] != HEADER_MAGIC:
        return None
    header_size, size = struct.unpack_from("<II", data, header + 64)
    if header_size < 96 or header + header_size > len(data) or size < 48:
        return None
    if code is None:
        code = header + header_size
    if not 0 <= code <= len(data) - size:
        return None
    footer = code + size - 48
    if data[footer:footer + 8] != b"barefoot":
        return None
    version, length, constants, metadata = struct.unpack_from("<4I", data, footer + 16)
    words_size = ((length + 15) & ~15) + constants
    if version != 1 or not length or words_size % 4:
        return None
    if ((words_size + 15) & ~15) + ((metadata + 15) & ~15) != size - 48:
        return None
    payload = data[code:code + words_size]
    return hash_code(payload, words_size), data[header + 90], header_size, payload


def scan(path, index, hash_code, stages, recipes, pairs=None, discovered=None):
    with path.open("rb") as file:
        if not os.fstat(file.fileno()).st_size:
            if pairs:
                raise ValueError("binding map references an empty file")
            return
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for header, code in header_offsets(data) if pairs is None else pairs:
                record = shader_record(data, header, code, hash_code)
                if record is None:
                    if pairs is not None:
                        raise ValueError(f"invalid shader binding at header offset {header:#x}")
                    continue
                digest, stage, header_size, payload = record
                words = index.get(digest)
                if words is not None and words != len(payload) // 4:
                    raise ValueError("inconsistent code length")
                if words is None and (discovered is None or stage != 0):
                    continue
                recipe = read_recipe(data, header, header_size) if stage == 0 else None
                if words is None:
                    if recipe is None:
                        continue
                    if discovered.setdefault(digest, payload) != payload:
                        raise ValueError("inconsistent discovered code")
                stages[digest].add(stage)
                if recipe is not None:
                    recipes.add((digest, *recipe))


def write_discovered_store(path, discovered):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            out.write(struct.pack("<Q", 0x3145444F4359544B))
            for digest, code in sorted(discovered.items()):
                frame = subprocess.run(["zstd", "-q", "-c", f"--stream-size={len(code)}"],
                                       input=code, stdout=subprocess.PIPE, check=True).stdout
                out.write(struct.pack("<QII", digest, len(code) // 4, len(frame)))
                out.write(frame)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Index verified shader creation headers by program hash")
    parser.add_argument("store", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bindings", nargs=2, type=Path, action="append", default=[], metavar=("IMAGE", "MAP"),
                        help="read header and code file offsets from verified shader creation arguments (two integers per line)")
    parser.add_argument("--discovered-store", type=Path,
                        help="also catalog compute programs found outside the input store and save their code in a separate store")
    args = parser.parse_args()
    inputs = {p.resolve() for p in [args.store, *args.files, *(p for pair in args.bindings for p in pair)]}
    outputs = [p.resolve() for p in [args.output, args.discovered_store] if p is not None]
    if any(p in inputs for p in outputs) or len(set(outputs)) != len(outputs):
        parser.error("each output must be distinct from all other paths")
    lib = ctypes.CDLL(ctypes.util.find_library("xxhash") or "libxxhash.so.0")
    hash_code = lib.XXH3_64bits
    hash_code.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    hash_code.restype = ctypes.c_uint64
    index = code_index(args.store)
    stages = defaultdict(set)
    recipes = set()
    discovered = {} if args.discovered_store else None
    for path in args.files:
        scan(path, index, hash_code, stages, recipes, discovered=discovered)
    for image, mapping in args.bindings:
        pairs = []
        for row in mapping.read_text().splitlines():
            header, code = row.split()
            pairs.append((int(header, 0), int(code, 0)))
        scan(image, index, hash_code, stages, recipes, pairs, discovered)
    if args.discovered_store:
        write_discovered_store(args.discovered_store, discovered)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            print("kyty compute recipes 1", file=out)
            for digest, *values in sorted(recipes):
                print(f"{digest:016x}", *values, file=out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temp, args.output)
        finally:
            temp.unlink(missing_ok=True)
    counts = Counter(stage for found in stages.values() for stage in found)
    print(f"{len(stages.keys() & index.keys())}/{len(index)} store programs with creation headers, {counts[0]} compute, "
          f"{len({r[0] for r in recipes})} with complete inputs, {len(recipes)} recipes")


if __name__ == "__main__":
    main()
