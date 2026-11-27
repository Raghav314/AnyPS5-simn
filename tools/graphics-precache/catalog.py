import argparse
import ctypes
import ctypes.util
import json
import mmap
import struct
from collections import Counter, defaultdict
from pathlib import Path


def xxhash():
    function = ctypes.CDLL(ctypes.util.find_library("xxhash")).XXH3_64bits
    function.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    function.restype = ctypes.c_uint64
    return function


def store_index(path):
    records = {}
    with path.open("rb") as stream:
        if stream.read(8) != struct.pack("<Q", 0x3145444F4359544B):
            raise ValueError("unsupported store")
        while header := stream.read(16):
            digest, words, compressed = struct.unpack("<QII", header)
            if not words or not compressed or len(stream.read(compressed)) != compressed:
                raise ValueError("truncated code store")
            if digest and records.setdefault(digest, words) != words:
                raise ValueError("inconsistent code size")
    return records


def read_registers(data, header, end, pointer, count):
    pos = header + pointer + struct.unpack_from("<q", data, header + pointer)[0]
    if not count:
        return {}
    if not header + 96 <= pos <= end - 8 * count:
        raise ValueError("register table outside creation header")
    return dict(struct.iter_unpack("<II", data[pos:pos + 8 * count]))


def read_creation_metadata(data, header, end):
    def target(field, size):
        offset, = struct.unpack_from("<q", data, field)
        if offset == 0:
            return None
        pos = field + offset
        if pos < header + 96 or pos + size > end:
            raise ValueError("creation metadata outside header")
        return pos

    def array(field, count, fmt):
        if not count:
            return ()
        size = count * struct.calcsize(fmt)
        pos = target(field, size)
        if pos is None:
            raise ValueError("missing creation metadata array")
        return tuple(value for value, in struct.iter_unpack(fmt, data[pos:pos + size]))

    inputs, _, outputs, special_size = struct.unpack_from("<IHHH", data, header + 80)
    user = target(header + 8, 54)
    direct = None
    if user is not None:
        count, = struct.unpack_from("<H", data, user + 44)
        direct = array(user, count, "<H")
    specials = array(header + 40, special_size // 4, "<I") if special_size % 4 == 0 else ()
    return (array(header + 48, inputs, "<I"), array(header + 56, outputs, "<I"), direct, specials)


def creation_headers(path, index, hash64, metadata=False):
    magic = struct.pack("<II", 0x34333231, 24)
    if not path.stat().st_size:
        return
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        cursor = 0
        while (header := data.find(magic, cursor)) >= 0:
            cursor = header + len(magic)
            if header + 96 > len(data):
                continue
            header_size, size = struct.unpack_from("<II", data, header + 64)
            start = header + header_size
            footer = start + size - 48
            if header_size < 96 or size < 48 or start + size > len(data) or data[footer:footer + 8] != b"barefoot":
                continue
            version, length, constants, metadata_size = struct.unpack_from("<4I", data, footer + 16)
            length = ((length + 15) & ~15) + constants if length else 0
            if version != 1 or not length or length % 4 or ((length + 15) & ~15) + ((metadata_size + 15) & ~15) != size - 48:
                continue
            digest = hash64(data[start:start + length], length)
            if index.get(digest) != length // 4:
                continue
            kind = data[header + 90]
            cx = read_registers(data, header, start, 24, data[header + 91])
            sh = read_registers(data, header, start, 32, data[header + 92])
            special = header + 40 + struct.unpack_from("<q", data, header + 40)[0]
            stage_enable = None
            if header + 96 <= special <= start - 16:
                stage_enable = struct.unpack_from("<I", data, special + 12)[0]
            scratch = struct.unpack_from("<H", data, header + 84)[0]
            record = (kind, tuple(sorted(cx.items())), tuple(sorted(sh.items())), stage_enable, scratch)
            if metadata:
                record += read_creation_metadata(data, header, start)
            yield f"{digest:016x}", record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("assets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    records = store_index(args.store)
    headers = defaultdict(set)
    hash64 = xxhash()
    for path in args.assets:
        for digest, record in creation_headers(path, records, hash64, args.metadata):
            headers[digest].add(record)
    counts = Counter()
    paths = Counter()
    for records_for_code in headers.values():
        counts[tuple(sorted({r[0] for r in records_for_code}))] += 1
        routes = set()
        for kind, _, sh, stage_enable, *_ in records_for_code:
            if kind == 2:
                regs = dict(sh)
                if 0x8b in regs and stage_enable is not None:
                    routes.add("mesh" if (stage_enable & 0x20 or (regs[0x8b] >> 19) & 255) else "vs")
                else:
                    routes.add("unknown")
        if routes:
            paths[tuple(sorted(routes))] += 1
    with args.output.open("x") as file:
        for code, values in sorted(headers.items()):
            for record in sorted(values, key=json.dumps):
                print(json.dumps([code, *record], separators=(",", ":")), file=file)
    print("stored", len(records), "typed", len(headers), "types", counts, "GS routes at creation", paths)


if __name__ == "__main__":
    main()
