import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COLLECT = Path(__file__).resolve().parents[1] / "graphics-precache" / "collect.py"
HEADER = "# kyty pipeline manifest v1"
VERTEX = "M 000000000001 vs code=1111222233334444"
PIXEL = "M 000000000002 ps code=5555666677778888"
PIPELINE = "P v=000000000001 p=000000000002 fv=c0.0.1 fp=c0.0.1 plain cc=1 c0=R8G8B8A8Unorm"
LAYOUT = "T 0000000000000001 icp=3"


def capture(variant, *rows):
    return [f"[kyty-queued-recipe] {variant:016x} {row}" for row in rows]


class CollectQueuedRecipesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.output = self.folder / "output.kpm"

    def write(self, name, rows, newline=True):
        path = self.folder / name
        path.write_text("\n".join(rows) + ("\n" if newline else ""), encoding="utf-8")
        return path

    def collect(self, *inputs):
        return subprocess.run([sys.executable, COLLECT, *inputs, "--output", self.output],
                              check=False, capture_output=True, text=True)

    def test_merge_interleaved_captures_and_existing_manifest(self):
        first = capture(1, HEADER, VERTEX, PIXEL, PIPELINE)
        float_variant = PIPELINE.replace("fp=c0.0.1", "fp=f0.0.1")
        second = capture(2, HEADER, VERTEX, PIXEL, float_variant)
        log = self.write("first.log", [first[0], second[0], *first[1:], *second[1:]])
        repeated = self.write("repeated.log", first)
        base = self.write("base.kpm", [HEADER, VERTEX, LAYOUT])
        result = self.collect(log, repeated, "--manifest", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 complete, 0 missing dependencies, 0 incomplete records", result.stderr)
        self.assertEqual(self.output.read_text().splitlines(),
                         [HEADER, VERTEX, PIXEL, PIPELINE, float_variant, LAYOUT])

    def test_missing_and_truncated_recipes_do_not_leak_into_manifest(self):
        lone_module = "M 000000000003 vs code=0000000000000003"
        log = self.write("runtime.log", capture(1, HEADER, VERTEX, PIXEL, PIPELINE)
                         + capture(2, HEADER, lone_module)
                         + capture(3, "missing stage=0 module=0000000000000004")
                         + capture(4, HEADER, VERTEX, PIPELINE)
                         + capture(5, HEADER, VERTEX, PIXEL, PIPELINE), newline=False)
        result = self.collect(log)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 complete, 1 missing dependencies, 3 incomplete records", result.stderr)
        self.assertEqual(self.output.read_text().splitlines(), [HEADER, VERTEX, PIXEL, PIPELINE])

    def test_empty_capture_preserves_destination(self):
        self.output.write_text("previous content\n", encoding="utf-8")
        result = self.collect(self.write("empty.log", ["unrelated log line"]))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("no complete queued recipe was found", result.stderr)
        self.assertEqual(self.output.read_text(), "previous content\n")

    def test_conflicting_module_preserves_destination(self):
        self.output.write_text("previous content\n", encoding="utf-8")
        base = self.write("base.kpm", [HEADER, VERTEX.replace("1111", "0000")])
        log = self.write("runtime.log", capture(1, HEADER, VERTEX, PIXEL, PIPELINE))
        result = self.collect(log, "--manifest", base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("conflicting M record", result.stderr)
        self.assertEqual(self.output.read_text(), "previous content\n")


if __name__ == "__main__":
    unittest.main()
