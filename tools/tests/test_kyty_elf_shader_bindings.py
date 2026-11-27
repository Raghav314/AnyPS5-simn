from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kyty_elf_shader_bindings import HEADER_MAGIC, discover


def hash_code(payload, _):
    return int.from_bytes(payload[:4], "little")


def make_image(path, corrupt_code=False):
    data = bytearray(8192)
    base = 0x10000
    dynamic_offset, rela_offset, jump_offset = 0x100, 0x300, 0x700
    symbols_offset, strings_offset, got_offset = 0x600, 0x680, 0x780
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, b"\x7fELF\x02\x01\x01", 3, 62, 1,
                     0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 7, 0, base, 0, len(data), len(data), 0x1000)
    strings = b"\0sceAgcCreateShader\0"
    data[strings_offset:strings_offset + len(strings)] = strings
    struct.pack_into("<I", data, symbols_offset + 24, 1)
    struct.pack_into("<QQq", data, jump_offset, base + got_offset, (1 << 32) | 7, 0)

    sizes = (0x900, 0xb00, 0xd00, 0xf00)
    for value, header in enumerate(sizes, 1):
        code = header + 0x100
        data[header:header + 8] = HEADER_MAGIC
        struct.pack_into("<II", data, header + 64, 96, 64)
        data[code:code + 16] = bytes([value]) * 16
        data[code + 16:code + 24] = b"barefoot"
        struct.pack_into("<IIII", data, code + 32, 1, 16, 0, 0)
    if corrupt_code:
        data[sizes[3] + 0x110] = 0

    relocations = []
    for element, (header, offset) in enumerate(zip(sizes[:3], (0x480, 0x498, 0x4c8), strict=True)):
        struct.pack_into("<I", data, offset, element)
        relocations.extend(((base + offset + 8, 8, base + header), (base + offset + 16, 8, base + header + 0x100)))
    for index, entry in enumerate(relocations):
        struct.pack_into("<QQq", data, rela_offset + 24 * index, *entry)

    call, stub, code = 0x200, 0x240, sizes[3] + 0x100
    data[stub:stub + 2] = b"\xff\x25"
    struct.pack_into("<i", data, stub + 2, got_offset - stub - 6)
    data[call:call + 3] = b"\x48\x8d\x35"
    struct.pack_into("<i", data, call + 3, sizes[3] - call - 7)
    data[call + 7:call + 10] = b"\x48\x8d\x15"
    struct.pack_into("<i", data, call + 10, code - call - 14)
    data[call + 14] = 0xe8
    struct.pack_into("<i", data, call + 15, stub - call - 19)

    dynamic = ((7, base + rela_offset), (8, 24 * len(relocations)), (9, 24), (20, 7),
               (23, base + jump_offset), (2, 24), (5, base + strings_offset), (10, len(strings)),
               (6, base + symbols_offset), (11, 24), (0, 0))
    for index, entry in enumerate(dynamic):
        struct.pack_into("<qQ", data, dynamic_offset + index * 16, *entry)
    length = len(dynamic) * 16
    struct.pack_into("<IIQQQQQQ", data, 120, 2, 4, dynamic_offset, base + dynamic_offset, 0, length, length, 8)
    path.write_bytes(data)
    return {value * 0x01010101: 4 for value in range(1, 5)}


class ElfShaderBindingTests(unittest.TestCase):
    def test_finds_indexed_records_and_verified_direct_call(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.elf"
            index = make_image(image)
            tables, calls = discover(image, index, hash_code)
        self.assertEqual(tables, {(0x900, 0xa00), (0xb00, 0xc00)})
        self.assertEqual(calls, {(0xf00, 0x1000)})

    def test_rejects_unverified_call_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.elf"
            index = make_image(image, corrupt_code=True)
            tables, calls = discover(image, index, hash_code)
        self.assertEqual(len(tables), 2)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
