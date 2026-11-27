import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PRECACHE = Path(__file__).resolve().parents[1] / 'precache' / 'precache'


class PrecacheSelectionTest(unittest.TestCase):
    def test_selects_shipped_compute_gaps_and_ranks_complete_reflection(self):
        texture_free, legacy_free, textured, legacy_textured, unknown, existing, graphics, no_recipe = (
            f'{i:016x}' for i in range(1, 9)
        )
        passes = [
            {'cs': {'code': unknown, 'reflected_texture_count': 0}},
            {'cs': {'code': unknown}},
            {'cs': {'code': textured, 'reflected_texture_count': 0}},
            {'cs': {'code': textured, 'reflected_texture_count': 2}},
            {'cs': {'code': legacy_textured, 'resources': [{'kind': 2, 'category': 0, 'texture_dimension': None}]}},
            {'cs': {'code': texture_free, 'reflected_texture_count': 0}},
            {'cs': {'code': legacy_free, 'resources': []}},
            {'cs': {'code': existing, 'reflected_texture_count': 0}},
            {'vs': {'code': graphics, 'reflected_texture_count': 0}},
            {'cs': {'code': no_recipe, 'reflected_texture_count': 0}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / 'graphics-passes.json'
            recipes = root / 'compute-recipes.txt'
            manifest = root / 'existing.kpm'
            selection = root / 'selection.txt'
            inventory.write_text(json.dumps({'bundles': [{'effects': [{'passes': [
                {'shaders': stages} for stages in passes
            ]}]}]}))
            recipes.write_text('kyty compute recipes 1\n' + ''.join(
                f'{digest} 32 1 1 0 0 0 0\n' for digest in
                (unknown, textured, legacy_textured, texture_free, legacy_free, existing, graphics)
            ))
            manifest.write_text(f'# kyty pipeline manifest v1\nM g vs code={texture_free}\nM c cs code={existing}\n')
            command = [sys.executable, str(PRECACHE), '--binary', str(root / 'missing'), 'select',
                       str(inventory), str(recipes), '--existing', str(manifest)]
            all_gaps = subprocess.check_output(command, text=True)
            self.assertEqual(all_gaps.splitlines(), [texture_free, legacy_free, legacy_textured, textured, unknown])
            subprocess.run([*command, '--texture-free', '--limit', '1', '--output', str(selection)], check=True)
            self.assertEqual(selection.read_text(), f'{texture_free}\n')


if __name__ == '__main__':
    unittest.main()
