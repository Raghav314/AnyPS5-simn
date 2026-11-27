import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graphics-precache"))
import bundle
import pass_coverage
import resource_coverage


def words(*values):
    return struct.pack(f"<{len(values)}I", *values)


def sized(data):
    return words(len(data)) + data


def reflect(resources=()):
    table = b"".join(struct.pack("<H4B5HII", kind, slot, 0, 0, flags, 0, 0, 0, 0, 0, 0, 0)
                     for kind, slot, flags in resources)
    return sized(struct.pack("<4H", len(resources), 0, 0, 8 + len(table)) + table)


def shader(kind, code, resources=(), *, metadata_bytes=152):
    header = bytearray(96)
    struct.pack_into("<2I", header, 0, 0x34333231, 24)
    struct.pack_into("<2I", header, 64, 96, 64)
    header[90] = kind
    footer = bytearray(48)
    footer[:8] = b"barefoot"
    struct.pack_into("<4I", footer, 16, 1, 4, 0, 0)
    program = b"sgaProg\0" + words(96, 64) + header + words(code, 0, 0, 0) + footer
    return b"\x02a\0\x32" + sized(program) + bytes(metadata_bytes) + reflect(resources)


def fixture(pixel_slot=1, vertex_resources=(), pixel_resources=(), *, embedded=False):
    header = b"SGD5" + bytes(8) + words(0) + reflect()
    metadata_bytes = 144 if embedded else 152
    vertex = shader(2, 0x100, vertex_resources, metadata_bytes=metadata_bytes)
    pixel = shader(1, 0x200, pixel_resources, metadata_bytes=metadata_bytes)
    shaders = words(1) + vertex + words(1) + pixel + words(0, 0, 0, 0)
    arrays = words(1, 0, 1, 0, 0, 0, 0, 0)
    raster = bytearray(16)
    raster[0] = 3
    raster[1] = 2
    depth = bytearray(24)
    depth[0] = 1
    blend = bytearray(68)
    blend[11] = 15
    states = words(1) + raster + words(1) + depth + words(1) + blend + words(0)
    row = bytearray(24)
    row[:2] = bytes([1, pixel_slot])
    row[9:11] = bytes([9, 15])
    row[18:22] = bytes([0, 255, 255, 255])
    struct.pack_into("<H", row, 16, 24)
    technique_and_pass = words(1, 1 << 24, 1) + row + words(4, 0, 0, 0, 0)
    binding_and_pool = words(0x80000000) + reflect() + reflect() + words(1, 0x80000000, 0) + sized(b"t\0")
    effect = bytes(80) + states + technique_and_pass + binding_and_pool
    if embedded:
        return shaders + effect
    return header + shaders + words(1) + b"\x02e\0" + words(0) + arrays + effect + bytes(10)


def fixture_hash(program, length):
    if length != 16 or program[4:] != bytes(12):
        raise ValueError("unexpected shader program extent")
    return struct.unpack_from("<I", program)[0]


class ShaderBundleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def read(self, data, include_resources=False):
        path = self.root / "fixture.bin"
        path.write_bytes(data)
        return list(bundle.read_bundles(path, fixture_hash, include_resources=include_resources))

    def test_reports_stable_references_and_selected_states(self):
        bundles = self.read(fixture())
        summary, pairs = bundle.summarize(bundles)
        self.assertEqual((summary["bundles"], summary["techniques"], summary["graphics_passes"]), (1, 1, 1))
        self.assertEqual(pairs, {("0000000000000100", "0000000000000200")})
        effect, = bundles[0]["effects"]
        row, = effect["passes"]
        self.assertEqual(row["shaders"]["vs"]["native_kind"], 2)
        self.assertEqual(row["states"], dict(rasterizer=0, depth_stencil=0, blend=0))
        self.assertEqual(row["bindings"], [0, None, None, None])
        self.assertEqual(row["stencil_reference"], 9)
        self.assertEqual(row["usage_offset"], 0)
        self.assertEqual(effect["rasterizer"][0]["cull_mode"], 2)
        self.assertEqual(effect["depth_stencil"][0]["depth_enable"], 1)
        self.assertEqual(effect["blend"][0]["targets"][0]["write_mask"], 15)
        manifest = self.root / "fixture.kpm"
        manifest.write_text("M a vs code=0000000000000100\nM b ps code=0000000000000200\n"
                            "M c ps code=0000000000000300\nP v=a p=b\nP v=a p=c\n")
        self.assertEqual(bundle.compare_manifest(manifest, pairs), dict(graphics_records=2, with_shipped_stage_pair=1))

    def test_finds_unaligned_embedded_effect_after_invalid_candidate(self):
        image = self.root / "fixture-image.bin"
        prefix = b"pre" + fixture(pixel_slot=2, embedded=True) + b"between"
        effect = fixture(embedded=True)
        image.write_bytes(prefix + effect + fixture(embedded=True)[:-12])
        found, = bundle.read_embedded_bundles(image, fixture_hash)
        self.assertTrue(found["embedded"])
        self.assertEqual((found["offset"], found["end"]), (len(prefix), len(prefix) + len(effect)))
        summary, pairs = bundle.summarize([found])
        self.assertEqual(summary["graphics_passes"], 1)
        self.assertEqual(pairs, {("0000000000000100", "0000000000000200")})

    def test_reads_two_aligned_bundles(self):
        data = bytearray(3 * bundle.ALIGNMENT)
        for offset in (bundle.ALIGNMENT, 2 * bundle.ALIGNMENT):
            member = fixture()
            data[offset:offset + len(member)] = member
        self.assertEqual([group["offset"] for group in self.read(data)], [4096, 8192])

    def test_compares_selected_states_without_guessing_missing_ones(self):
        manifests = self.root / "states.kpm"
        manifests.write_text("M a vs code=0000000000000100\nM b ps code=0000000000000200\n"
                             "M c ps code=0000000000000300\n"
                             "P v=a p=b c0=R8G8B8A8Unorm m0=f noclip\n"
                             "P v=a p=b c0=R8G8B8A8Unorm m0=3 poly=Line a2c b0=1.4.0.5\n"
                             "P v=a p=c c0=R8G8B8A8Unorm m0=f noclip\n")
        groups = self.read(fixture())
        states = pass_coverage.index_pass_states(groups)
        report = pass_coverage.compare_manifest_states(manifests, states)
        self.assertEqual(report["with_pair_candidates"], 2)
        self.assertEqual(report["with_explicit_selection"], dict(rasterizer=2, depth_stencil=2, blend=2))
        for name, counts in report["fields"].items():
            with self.subTest(name=name):
                self.assertEqual(counts, dict(agree=1, differ=1, unobserved=1))

        choices, = states.values()
        choices.append({})
        self.assertEqual(report, pass_coverage.compare_manifest_states(manifests, states))
        choices.append({"rasterizer": dict(fill_mode=2, depth_clip_enable=1)})
        mixed = pass_coverage.compare_manifest_states(manifests, states)["fields"]
        self.assertEqual(mixed["polygon_mode"], dict(mixed=2, unobserved=1))
        self.assertEqual(mixed["depth_clip"], dict(mixed=2, unobserved=1))

    def test_rejects_bad_shader_slot_and_truncated_table(self):
        with self.assertRaisesRegex(ValueError, "missing shader variant"):
            self.read(fixture(pixel_slot=2))
        with self.assertRaisesRegex(ValueError, "exceeds the input"):
            self.read(fixture()[:-12])

    def test_resource_declarations_constrain_dimensions_without_assigning_slots(self):
        vertex = [(0x1C2, 7, 64), (0x430, 9, 0)]
        pixel = [(0x302, 0, 0)]
        groups = self.read(fixture(vertex_resources=vertex, pixel_resources=pixel), include_resources=True)
        texture, buffer = groups[0]["stages"]["vs"][0]["resources"]
        self.assertEqual(texture["texture_dimension"], "2DArray")
        self.assertEqual(texture["serialized_slot"], 7)
        self.assertEqual(texture["flags"], 64)
        self.assertEqual(buffer["category"], 3)
        self.assertIsNone(buffer["texture_dimension"])
        manifests = self.root / "resources.kpm"
        manifests.write_text("M a vs code=0000000000000100 si=n:Float/d:2DArray,n:Uint/d:2D/heap\n"
                             "M b vs code=0000000000000100 si=n:Uint/d:2DArray\n"
                             "M c ps code=0000000000000200 si=n:Float/d:3D/ir:2\n"
                             "M d ps code=0000000000000300 si=n:Float/d:1D\n"
                             "P v=a p=c\nP v=b p=d\n")
        declarations = resource_coverage.index_declared_dimensions(groups)
        report = resource_coverage.compare_manifest_resources(manifests, declarations)
        self.assertEqual(report["shader_modules"], 4)
        self.assertEqual(report["modules_with_comparable_shader_record"], 3)
        self.assertEqual(report["image_modules_with_equal_dimension_set"], 2)
        self.assertEqual(report["image_modules_with_any_dimension_agreement"], 3)
        self.assertEqual(report["modules_with_indirect_table_or_heap_images"], 2)
        self.assertEqual(report["undeclared_dimensions"], {"2D": 1})
        self.assertEqual(report["runtime_numeric_classes"], {"Float": 3, "Uint": 2})
        self.assertEqual(report["code_digests_with_varying_image_signatures"], 1)


if __name__ == "__main__":
    unittest.main()
