import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def footer():
    trailer = bytearray(48)
    trailer[:8] = b'barefoot'
    struct.pack_into('<4I', trailer, 16, 1, 16, 0, 0)
    return trailer


def create_compute_shader():
    registers = [(0x207, 8), (0x208, 4), (0x209, 1), (0x212, 0), (0x213, 0), (0x20C, 0), (0x20D, 0)]
    header = bytearray(96 + len(registers) * 8 + 20)
    struct.pack_into('<2I', header, 0, 0x34333231, 24)
    struct.pack_into('<2q', header, 32, 96 - 32, 96 + len(registers) * 8 - 40)
    struct.pack_into('<2I', header, 64, len(header), 16 + len(footer()))
    header[92] = len(registers)
    for index, register in enumerate(registers):
        struct.pack_into('<2I', header, 96 + index * 8, *register)
    return bytes(header + struct.pack('<4I', 0xBF810000, 0, 0, 0) + footer())


def create_archive(shader):
    extra = len(shader) - 15
    compressed = b'\xf0' + b'\xff' * (extra // 255) + bytes([extra % 255]) + shader
    return (struct.pack('<5I', 2, 0, 0, 0, 1) + struct.pack('<QIIB', 0, len(compressed), len(shader), 0)
            + struct.pack('<2I', 0, 0) + compressed)


def create_pak(name, payload):
    name = name.encode() + b'\0'
    entry = struct.pack('<3QI', 0, len(payload), len(payload), 0) + bytes(20) + struct.pack('<BI', 0, 0)
    stored = entry + payload
    index = struct.pack('<I', 1) + b'\0' + struct.pack('<2I', 1, len(name)) + name + entry
    trailer = struct.pack('<2I2Q', 0x5A6F12E1, 3, len(stored), len(index)) + bytes(20)
    return stored + index + trailer


def run(binary, source):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        assets = work / 'assets'
        assets.mkdir()
        output = work / 'inventory'
        shader = create_compute_shader()
        (assets / 'pack.ushaderbytecode').write_bytes(create_archive(shader))
        (assets / 'duplicate.bin').write_bytes(shader[-64:])
        (assets / 'untyped.bin').write_bytes(struct.pack('<4I', 0xBF810000, 1, 0, 0) + footer())
        (assets / 'unreadable.ushaderbytecode').write_bytes(struct.pack('<I', 1))
        kpm = b'# kyty pipeline manifest v1\n'
        (assets / 'loose.kpm').write_bytes(kpm + b'M a cs code=0123456789abcdef t=8,4,1\n')
        (assets / 'recipes.pak').write_bytes(create_pak('packed.kpm', kpm + b'M a cs code=1234567890abcdef t=4,8,1\n'))
        (assets / 'invalid.kpm').write_bytes(b'other format\n')
        (assets / 'cache.upipelinecache').touch()
        wrapper = source / 'tools/precache/precache'
        image = work / 'image.bin'
        detached_code = struct.pack('<4I', 0xBF810000, 2, 0, 0)
        executable_header = shader[:-64] + bytes(64)
        image.write_bytes(executable_header + detached_code + footer())
        bindings = work / 'bindings.map'
        bindings.write_text(f'0 {len(executable_header)}\n')
        subprocess.run([sys.executable, wrapper, '--binary', binary, 'inventory', assets, output,
                        '--bindings', image, bindings], check=True)
        counts = json.loads((output / 'report.json').read_text())
        assert counts['shader_programs'] == 3, counts
        assert counts['creation_records'] == 1, counts
        assert counts['unresolved_containers'] == 1, counts
        assert counts['packaged_manifests'] == 2, counts
        assert counts['invalid_manifests'] == 1, counts
        assert counts['foreign_pipeline_caches'] == 1, counts
        assert counts['skipped_packaged_records'] == 0, counts
        marker, *shipped = (output / 'shipped.kpm').read_text().splitlines()
        assert marker == '# kyty pipeline manifest v1'
        assert {line.split()[3] for line in shipped} == {'code=0123456789abcdef', 'code=1234567890abcdef'}, shipped
        assert (output / 'creation-records.bin').read_bytes() == shader
        store = output / 'shaders.code'
        assert store.read_bytes().startswith(struct.pack('<Q', 0x3145444F4359544B))
        recipes = output / 'compute-recipes.txt'
        marker, *compute = recipes.read_text().splitlines()
        assert marker == 'kyty compute recipes 1' and len(compute) == 2
        digests = {recipe.split()[0] for recipe in compute}
        assert len(digests) == 2
        for recipe in compute:
            digest, *fields = recipe.split()
            assert fields == ['8', '4', '1', '0', '0', '0', '0'], recipe
            arguments = struct.pack('<Q8I', int(digest, 16), *map(int, fields), 4)
            for code in (shader[-64:-48], detached_code):
                prep = subprocess.run([binary, '--prepare-compute'], input=arguments + code, cwd=work)
                if prep.returncode == 0:
                    break
                assert prep.returncode == 64
            else:
                raise AssertionError(f'No known payload matched recipe {digest}')
            assert 'cs code=' + digest in (work / 'compute.kpm').read_text()
            assert (work / 'compute-0.spv').read_bytes().startswith(struct.pack('<I', 0x07230203))
        failed = subprocess.run([binary, '--inventory', assets, output], capture_output=True)
        assert failed.returncode != 0
        assert (output / 'creation-records.bin').read_bytes() == shader


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parents[2])
