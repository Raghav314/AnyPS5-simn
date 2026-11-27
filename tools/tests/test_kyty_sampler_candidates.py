import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "graphics-precache"
sys.path.insert(0, str(TOOLS))
from bind import module_line
from catalog import xxhash
from sampler_candidates import HEADER, sampler_neighbors
from session import read_manifest


VERTEX = "1111222233334444"
PIXEL = "5555666677778888"
LAYOUT = "ir:0/is:3,ir:1/im:9/is:3,ir:0,ir:0,ir:1,ir:1"
NEIGHBORS = (
    "ir:0/is:3,ir:1/im:9/is:3,ir:0,ir:1,ir:1,ir:1",
    "ir:0/is:3,ir:1/im:9/is:3,ir:0,ir:0,ir:0,ir:1",
)


class SamplerCandidatesTest(unittest.TestCase):
    def test_only_the_boundary_child_changes(self):
        self.assertEqual(tuple(sampler_neighbors(LAYOUT)), NEIGHBORS)
        self.assertEqual(tuple(sampler_neighbors("ir:0/is:1,ir:1/im:3/is:1,ir:0,ir:1")), ())

    def test_corrupt_or_unsupported_mapping_is_rejected(self):
        layouts = (
            LAYOUT.replace("im:9", "im:8"),
            LAYOUT.replace("im:9", "im:4294967296"),
            LAYOUT.replace("im:9", "im:-1"),
            LAYOUT.replace("im:9", "im:9/im:11"),
            LAYOUT.replace("im:9", "im:bad"),
            LAYOUT.replace("is:3", "is:0"),
            LAYOUT.replace("is:3", "is:33"),
            LAYOUT.replace("ir:0,ir:0,ir:1,ir:1", "ir:0,ir:1,ir:0,ir:1"),
            LAYOUT.replace("ir:0,ir:0,ir:1,ir:1", "ir:0,ir:0,ir:1,ir:2"),
            LAYOUT.replace("ir:0,ir:0,ir:1,ir:1", "ir:0,ir:0,ir:1,ir:1/im:20"),
        )
        for layout in layouts:
            with self.subTest(layout=layout):
                self.assertEqual(tuple(sampler_neighbors(layout)), ())

    def test_cli_writes_only_novel_recipes_and_holdout_leaves_output_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            inventory = directory / "shipped.json"
            effects = [{"passes": [{"shaders": {"vs": {"code": VERTEX}, "ps": {"code": PIXEL}}}]}]
            inventory.write_text(json.dumps({"bundles": [{"effects": effects}]}))
            hash64 = xxhash()
            vertex = module_line("vs", {"code": VERTEX}, hash64)
            variant = {"code": PIXEL, "cur": "19", "ss": LAYOUT}
            pixel = module_line("ps", variant, hash64)
            known = module_line("ps", variant | {"ss": NEIGHBORS[0]}, hash64)
            unknown = module_line("ps", variant | {"ss": NEIGHBORS[1]}, hash64)
            vertex_id = vertex.split()[1]
            pixel_id = pixel.split()[1]
            known_id = known.split()[1]
            unknown_id = unknown.split()[1]
            source_pipeline = f"P v={vertex_id} p={pixel_id} fp=c0.0.1 plain cc=1 c0=R8G8B8A8Unorm"
            known_pipeline = source_pipeline.replace(f"p={pixel_id}", f"p={known_id}")
            expected = source_pipeline.replace(f"p={pixel_id}", f"p={unknown_id}")
            seed = directory / "seed.kpm"
            seed.write_text("\n".join([HEADER, vertex, pixel, known, source_pipeline, known_pipeline]) + "\n")
            holdout = directory / "holdout.kpm"
            holdout.write_text("\n".join([HEADER, vertex, unknown, expected]) + "\n")
            output = directory / "generated.kpm"
            checked = directory / "checked.kpm"
            command = [sys.executable, TOOLS / "sampler_candidates.py", inventory, seed]
            for destination, evaluation in ((output, []), (checked, ["--held-out", holdout])):
                result = subprocess.run([*command, "--output", destination, *evaluation],
                                        capture_output=True, text=True, check=True)
                if evaluation:
                    self.assertIn("held-out exact matches 1/1", result.stdout)
            self.assertEqual(output.read_bytes(), checked.read_bytes())
            modules, pipelines = read_manifest(output)
            self.assertEqual(set(modules), {vertex_id, unknown_id})
            self.assertEqual(pipelines, [expected])


if __name__ == "__main__":
    unittest.main()
