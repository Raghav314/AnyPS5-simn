import argparse
import ctypes
import ctypes.util
import json
import mmap
from pathlib import Path
import re
import struct
import sys

from compute_recipes import HEADER_MAGIC, code_index, shader_record
from nid_names import compute_nid
from unwrap_self import PT_DYNAMIC, PT_LOAD, Refused, parse_elf_at


RELA = struct.Struct("<QQq")
DYNAMIC = struct.Struct("<qQ")
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_PLTRELSZ = 2
DT_STRTAB = 5
DT_SYMTAB = 6
DT_STRSZ = 10
DT_SYMENT = 11
DT_PLTREL = 20
DT_JMPREL = 23
R_X86_64_RELATIVE = 8
R_X86_64_JUMP_SLOT = 7
CREATE_SHADER = {b"sceAgcCreateShader", compute_nid("sceAgcCreateShader").encode()}
CREATE_CALL = re.compile(rb"\x48\x8d\x35....\x48\x8d\x15....\xe8....|\x48\x8d\x15....\x48\x8d\x35....\xe8....", re.S)


class LinkedImage:
    def __init__(self, data):
        if data[:4] != b"\x7fELF":
            raise ValueError("expected a locally relinked ELF")
        _, _, headers = parse_elf_at(data, 0)
        self.data = data
        self.loads = [row for row in headers if row["type"] == PT_LOAD]
        for load in self.loads:
            self.file_span(load["offset"], load["size"])
        self.tags = {}
        dynamic = [row for row in headers if row["type"] == PT_DYNAMIC]
        if len(dynamic) != 1:
            raise ValueError("expected one ELF dynamic segment")
        for tag, value in DYNAMIC.iter_unpack(self.file_span(dynamic[0]["offset"], dynamic[0]["size"])):
            if tag == 0:
                break
            self.tags[tag] = value
        else:
            raise ValueError("unterminated ELF dynamic segment")

    def file_span(self, offset, size):
        if not 0 <= offset <= len(self.data) or size > len(self.data) - offset:
            raise ValueError("ELF span lies outside the file")
        return self.data[offset:offset + size]

    def offset(self, address, size):
        for load in self.loads:
            delta = address - load["vaddr"]
            if 0 <= delta <= load["size"] - size:
                return load["offset"] + delta
        return None

    def relas(self, address_tag, size_tag):
        address, size = self.tags.get(address_tag), self.tags.get(size_tag, 0)
        if address is None or size == 0:
            return ()
        offset = self.offset(address, size)
        if offset is None or self.tags.get(DT_RELAENT, RELA.size) != RELA.size or size % RELA.size:
            raise ValueError("invalid ELF RELA table")
        return RELA.iter_unpack(self.file_span(offset, size))

    def verified_pair(self, header_address, code_address, index, hash_code):
        header = self.offset(header_address, 96)
        code = self.offset(code_address, 48)
        if header is None or code is None or self.data[header:header + 8] != HEADER_MAGIC:
            return None
        header_size, code_size = struct.unpack_from("<II", self.data, header + 64)
        if self.offset(header_address, header_size) is None or self.offset(code_address, code_size) is None:
            return None
        record = shader_record(self.data, header, code, hash_code)
        if record is None or index.get(record[0]) != len(record[3]) // 4:
            return None
        return header, code

    def table_pairs(self, index, hash_code):
        relatives = {slot: target for slot, kind, target in self.relas(DT_RELA, DT_RELASZ) if kind == R_X86_64_RELATIVE}
        candidates = {}
        for slot, header in relatives.items():
            code = relatives.get(slot + 8)
            if code is None:
                continue
            pair = self.verified_pair(header, code, index, hash_code)
            if pair is None:
                continue
            counter = self.offset(slot - 8, 4)
            if counter is None:
                continue
            candidates[slot] = struct.unpack_from("<I", self.data, counter)[0], pair

        for slot, (counter, pair) in candidates.items():
            previous = candidates.get(slot - 24)
            following = candidates.get(slot + 24)
            if (previous is not None and previous[0] == counter - 1
                    or following is not None and following[0] == counter + 1):
                yield pair

    def shader_import_slots(self):
        if self.tags.get(DT_PLTREL) != DT_RELA:
            return set()
        symbol_address = self.tags.get(DT_SYMTAB, 0)
        strings_size = self.tags.get(DT_STRSZ, 0)
        strings_offset = self.offset(self.tags.get(DT_STRTAB, 0), strings_size)
        if not strings_size or strings_offset is None or self.tags.get(DT_SYMENT) != 24:
            return set()
        strings = self.file_span(strings_offset, strings_size)
        slots = set()
        for slot, info, _ in self.relas(DT_JMPREL, DT_PLTRELSZ):
            if info & 0xffffffff != R_X86_64_JUMP_SLOT:
                continue
            symbol = self.offset(symbol_address + 24 * (info >> 32), 24)
            if symbol is None:
                continue
            name = struct.unpack_from("<I", self.data, symbol)[0]
            end = strings.find(b"\0", name)
            if end >= 0 and strings[name:end].split(b"#", 1)[0] in CREATE_SHADER:
                slots.add(slot)
        return slots

    def call_pairs(self, index, hash_code):
        slots = self.shader_import_slots()
        if not slots:
            return
        code_loads = [row for row in self.loads if row["flags"] & 1]
        stubs = set()
        for load in code_loads:
            start, end = load["offset"], load["offset"] + load["size"]
            pc = start
            while (pc := self.data.find(b"\xff\x25", pc, end - 4)) >= 0:
                va = load["vaddr"] + pc - start
                if va + 6 + struct.unpack_from("<i", self.data, pc + 2)[0] in slots:
                    stubs.add(va)
                pc += 2
        for load in code_loads:
            start, end = load["offset"], load["offset"] + load["size"]
            for match in CREATE_CALL.finditer(self.data, start, end):
                pc = match.start()
                va = load["vaddr"] + pc - start
                if va + 19 + struct.unpack_from("<i", self.data, pc + 15)[0] not in stubs:
                    continue
                first = va + 7 + struct.unpack_from("<i", self.data, pc + 3)[0]
                second = va + 14 + struct.unpack_from("<i", self.data, pc + 10)[0]
                header, code = (first, second) if self.data[pc + 2] == 0x35 else (second, first)
                if pair := self.verified_pair(header, code, index, hash_code):
                    yield pair


def discover(image, index, hash_code):
    with image.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        elf = LinkedImage(data)
        return set(elf.table_pairs(index, hash_code)), set(elf.call_pairs(index, hash_code))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find validated detached shader bindings in a locally relinked ELF.")
    parser.add_argument("image", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new binding map for compute_recipes.py --bindings")
    args = parser.parse_args(argv)
    try:
        hash_code = ctypes.CDLL(ctypes.util.find_library("xxhash") or "libxxhash.so.0").XXH3_64bits
        hash_code.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        hash_code.restype = ctypes.c_uint64
        tables, calls = discover(args.image, code_index(args.store), hash_code)
        pairs = sorted(tables | calls)
        with args.output.open("x", encoding="ascii") as output:
            output.writelines(f"{header:#x} {code:#x}\n" for header, code in pairs)
    except (OSError, Refused, ValueError) as error:
        print(f"kyty_elf_shader_bindings: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"table_bindings": len(tables), "call_bindings": len(calls), "total_bindings": len(pairs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
