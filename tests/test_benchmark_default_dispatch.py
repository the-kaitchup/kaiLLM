import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'benchmark_graphs', ROOT / 'runtime-source/vllm_gguf_plugin/benchmark_graphs.py')
graphs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(graphs)


class Flags(list):
    def all(self):
        return all(self)


class DefaultDispatchTest(unittest.TestCase):
    def test_mixed_and_unclassified_dispatch_remain_distinguishable(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = graphs.DefaultDispatchCounters(tmp, 'GPU-test', 'FULL_DECODE_ONLY', False)
            self.addCleanup(obs.counts.close)
            prefill = SimpleNamespace(has_prefill=True, is_prefilling_np=Flags([True]))
            decode = SimpleNamespace(has_prefill=False, is_prefilling_np=Flags([False]))
            mixed = SimpleNamespace(has_prefill=True, is_prefilling_np=Flags([True, False]))
            for state, mode in ((prefill, 'NONE'), (decode, 'FULL'), (mixed, 'NONE'), (decode, 'PIECEWISE')):
                phases = obs.completed_forward(state, mode)
                obs.add(phases, 3)
            self.assertIsNone(obs.completed_forward(None, 'FULL'))
            meta_file, = Path(tmp).glob('*.json')
            self.assertEqual(meta_file.stem, f'GPU-test.{graphs.os.getpid()}')
            self.assertEqual(json.loads(meta_file.read_bytes())['fields'],
                             ['full', 'direct', 'other', 'direct_head', 'piecewise'])
            self.assertEqual(struct.unpack('<10Q', meta_file.with_suffix('.dispatch-counts').read_bytes()),
                             (0, 2, 1, 2, 0, 1, 1, 1, 3, 1))
