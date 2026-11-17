"""Exercise the actual Python/URL/JS path, including row-scale padding and rolling context.

OMP_NUM_THREADS=2 uv run python stories/test_stories.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for tools/
from tools.host import PREFIX
import atexit
import os
import shutil
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# Use a small, deterministic corpus even in a fresh clone; no download is
# needed to test packing, gradients, rolling caches or exact training resume.
ROOT = Path(__file__).resolve().parents[1]
fixture = tempfile.TemporaryDirectory(dir=ROOT / '.venv')
atexit.register(fixture.cleanup)
work = Path(fixture.name)
(work / 'data').mkdir(); (work / 'out').mkdir()
vocab = '\0 ' + 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
(work / 'data/vocab.json').write_text(json.dumps({'vocab': vocab}))
(work / 'data/train.u8').write_bytes(bytes(range(len(vocab))) * 19000)
(work / 'data/heldout.u8').write_bytes(bytes(range(len(vocab))) * 20)
(work / 'data/selection.u8').write_bytes(bytes(range(len(vocab))) * 20)

os.chdir(work)
import stories
atexit.register(os.chdir, ROOT)


class PackedRuntimeTest(unittest.TestCase):
    def test_clipped_ste_stops_saturated_gradients(self):
        m = stories.GPT(8, 1, 2, 8, 4, 'row', ste='clipped'); m.q = True
        w = torch.nn.Parameter(torch.tensor([[0., 0.1, -0.2, 8., 0., 0., 0., 0.]]))
        _, scale, _, _ = stories.quantize(w, 4, 'row')
        m.weight(w).sum().backward()
        expected = (w.detach().abs() <= 7 * scale).float()
        self.assertTrue(torch.equal(w.grad, expected))

    def test_packed_logits(self):
        for mode, bits in [('tensor', 4), ('row', 4), ('row', 5), ('row', 8)]:
            with self.subTest(mode=mode, bits=bits):
                torch.manual_seed(8)
                m = stories.GPT(8, 2, 2, 8, bits, mode)
                vals, codes, scales = [], [], []
                M = 2 ** (bits - 1) - 1
                for w in m.tensors():
                    q, s, base, rows = stories.quantize(w, bits, mode)
                    w.data.copy_(q * s)
                    vals.extend((q + M).flatten().int().tolist())
                    if rows is not None: codes.extend(rows.flatten().int().tolist())
                    scales.append('%.4g' % base)
                packed = (stories.pack_bits(codes, 8) if codes else '') + stories.pack_bits(vals, bits)
                tokens = [0] + list(range(1, 35))
                with torch.no_grad():
                    expected = m(torch.tensor([tokens]))[0].numpy()
                # Suppress the auto-story's timers, then reset and run the exported program.
                script = ('globalThis.document={body:{style:{}}};globalThis.setTimeout=()=>{};'
                          'globalThis.d=' + json.dumps(packed) + ';(0,eval)('
                          + json.dumps(stories.runtime(m, scales, .8))
                          + ');R();console.log(JSON.stringify('
                          + json.dumps(tokens) + '.map(t=>Array.from(S(t)))));')
                actual = np.array(json.loads(subprocess.check_output(['node', '-e', script], text=True)))
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_checkpoint_free_url_decoder(self):
        import base64
        from score_story import load_model
        torch.manual_seed(9)
        m = stories.GPT(8, 1, 2, 8, 4, 'row')
        vals, codes, scales = [], [], []
        for w in m.tensors():
            q, scale, base, rows = stories.quantize(w, 4, 'row')
            vals.extend((q + 7).flatten().int().tolist())
            codes.extend(rows.flatten().int().tolist())
            scales.append('%.4g' % base)
        program = stories.runtime(m, scales, .8)
        packed = stories.pack_bits(codes, 8) + stories.pack_bits(vals, 4)
        url = PREFIX + base64.urlsafe_b64encode(program.encode()).decode().rstrip('=') + '.' + packed
        path = Path('out/fixture_url.txt'); path.write_text(url)
        decoded = load_model(path)
        tokens = list(range(20))
        script = ('globalThis.document={body:{style:{}}};globalThis.setTimeout=()=>{};'
                  'globalThis.d=' + json.dumps(packed) + ';(0,eval)(' + json.dumps(program)
                  + ');R();console.log(JSON.stringify(' + json.dumps(tokens) + '.map(t=>Array.from(S(t)))));')
        actual = np.array(json.loads(subprocess.check_output(['node', '-e', script], text=True)))
        with torch.no_grad(): expected = decoded(torch.tensor([tokens]))[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_interrupted_training_resumes_exactly(self):
        cfg = stories.parse(['d=8', 'L=1', 'H=2', 'ctx=8', 'bs=2', 'steps=4',
                             'minutes=1', 'dev_chars=16', 'diag_chars=16',
                             'checkpoint_seconds=0', 'qat=0.5'])
        with tempfile.TemporaryDirectory(dir='out') as tmp:
            prefix = Path(tmp).name
            stories.train(prefix + '/full', cfg)
            save = stories.atomic_save

            class Interrupted(Exception): pass

            def interrupt(obj, path):
                save(obj, path)
                if obj['steps'] == 2: raise Interrupted()

            with patch.object(stories, 'atomic_save', interrupt):
                with self.assertRaises(Interrupted): stories.train(prefix + '/split', cfg)
            stories.train(prefix + '/split', stories.parse([f'resume={tmp}/split.last.pt']))
            full = torch.load(f'{tmp}/full.last.pt', weights_only=False)
            split = torch.load(f'{tmp}/split.last.pt', weights_only=False)
            self.assertEqual(full['steps'], split['steps'])
            for k in full['sd']:
                self.assertTrue(torch.equal(full['sd'][k], split['sd'][k]), k)
            self.assertEqual(full['history'][-1]['quant_dev_bpc'], split['history'][-1]['quant_dev_bpc'])


if __name__ == '__main__':
    unittest.main()
