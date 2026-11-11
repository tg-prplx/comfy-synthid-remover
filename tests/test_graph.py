import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cli

ROOT = Path(__file__).resolve().parents[1]


class GraphTests(unittest.TestCase):
    def test_graphs_are_consistent(self):
        for path in (ROOT / 'workflows').glob('*.json'):
            with self.subTest(workflow=path.name):
                g = json.loads(path.read_text(encoding='utf-8'))
                nodes = {n['id']: n for n in g['nodes']}
                links = {link[0]: link for link in g['links']}
                self.assertEqual(len(nodes), len(g['nodes']))
                self.assertEqual(len(links), len(g['links']))
                self.assertLessEqual(max(nodes), g['last_node_id'])
                for link_id, src, slot, dst, idx, dtype in g['links']:
                    self.assertIn(src, nodes)
                    self.assertIn(dst, nodes)
                    self.assertLess(slot, len(nodes[src]['outputs']))
                    self.assertLess(idx, len(nodes[dst].get('inputs', [])))
                    inp = nodes[dst]['inputs'][idx]
                    self.assertEqual(inp['link'], link_id)
                    self.assertEqual(inp['type'], dtype)
                    self.assertIn(link_id, nodes[src]['outputs'][slot]['links'])

    def test_optimized_uses_only_core_nodes(self):
        g = json.loads((ROOT / 'workflows' / 'optimized.json').read_text())
        self.assertFalse(any(n['type'] == 'SaveImageAdvanced' for n in g['nodes']))
        self.assertEqual(sum(n['type'] == 'CLIPTextEncode' for n in g['nodes']), 1)
        self.assertEqual(len([n for n in g['nodes'] if n['type'] == 'SaveImage']), 1)

    def test_api_links_and_parameters(self):
        g = json.loads((ROOT / 'api' / 'prompt.json').read_text())
        self.assertEqual(g['13']['inputs']['positive'], g['13']['inputs']['negative'])
        self.assertEqual(g['14']['inputs']['shift'], 3)
        self.assertEqual(g['12']['inputs']['type'], 'lumina2')
        self.assertEqual(g['13']['inputs']['cfg'], 1)
        for node in g.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertEqual(len(value), 2)
                    self.assertIn(value[0], g)
                    self.assertIsInstance(value[1], int)

    def test_overrides_do_not_modify_template(self):
        g = json.loads((ROOT / 'api' / 'prompt.json').read_text())
        options = SimpleNamespace(model=None, clip=None, vae=None, prompt='',
                                  steps=4, denoise=0.4, seed=99, scheduler='simple', shift=2)
        p = cli.build_prompt(g, options, 'upload.png')
        self.assertEqual(p['13']['inputs']['steps'], 4)
        self.assertEqual(p['13']['inputs']['seed'], 99)
        self.assertEqual(p['1']['inputs']['image'], 'upload.png')
        self.assertEqual(p['14']['inputs']['shift'], 2)
        self.assertEqual(g['13']['inputs']['steps'], 6)

    def test_output_paths(self):
        x = Path('/tmp/image.webp')
        self.assertEqual(cli.target_for(x, None, False), Path('/tmp/image_processed.png'))
        self.assertEqual(cli.target_for(x, Path('results'), True), Path('results/image_processed.png'))

    def test_history(self):
        self.assertIsNone(cli.get_images_from_history({}, 'id'))
        data = {'id': {'outputs': {'8': {'images': [{'filename': 'a.png'}]}}}}
        self.assertEqual(cli.get_images_from_history(data, 'id')[0]['filename'], 'a.png')


if __name__ == '__main__':
    unittest.main()
