import hashlib
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks/long_context'))
import hybrid_execution as hybrid
from benchmark import full_graph_runtime
from cuda_graphs import verify_interval, validate_kai_launch

spec = importlib.util.spec_from_file_location('graph_observer', ROOT / 'runtime-source/vllm_gguf_plugin/benchmark_graphs.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class DirectHeadTests(unittest.TestCase):
    def test_full_backbone_and_exact_terminal_calls_cover_each_phase(self):
        with TemporaryDirectory() as d:
            ids = ['GPU-first','GPU-last']
            ranks = [backend.TargetGraphCounters(d, uid, 'FULL', False) for uid in ids]
            try:
                head = ranks[-1]
                project = lambda h: SimpleNamespace(shape=(h.shape[0], 128))
                with patch.dict('os.environ', {hybrid.HEAD_ENV: 'direct'}):
                    head.install_head(None, 2, project)
                descriptor = SimpleNamespace(cg_mode=SimpleNamespace(name='FULL'),num_active_loras=0,
                                             num_tokens=2048,uniform_token_count=None,max_query_len=None)
                for rank in ranks:
                    rank.ready([descriptor],2048)
                before = hybrid.read_direct(d, ids)
                self.assertEqual(hybrid.direct_devices(before, ids, ['--pipeline-parallel-size','2']), ids[-1:])
                with self.assertRaises(ValueError):
                    hybrid.direct_devices(before, ids, [])
                for phase in (0,1):
                    for rank in ranks:
                        rank.add(phase, 0);rank.replayed([phase])
                    self.assertEqual(head.project_head(SimpleNamespace(shape=(1,16)),[phase]).shape, (1,128))
                head.project_head(SimpleNamespace(shape=(0,16)), [0])
                end = hybrid.read_direct(d, ids)
                target = verify_interval(before, end, ('prefill','decode'))
                per_rank = hybrid.direct_interval(before,end,('prefill','decode'),ids[-1:],target)
                proof = hybrid.direct_record([(w,per_rank) for w in ('prefill-1','prefill-2','prefill-3','full')])
                self.assertEqual(proof['prefill']['completed_calls'],4)
                self.assertEqual(proof['decode']['per_wave_by_gpu'],{'full':{'GPU-last':1}})
                self.assertNotIn('graph_replays',proof['decode'])
                # A head error records the attempt, never the completion.
                head.direct_head = lambda hidden: None
                with self.assertRaisesRegex(RuntimeError, 'scheduled vocabulary rows'):
                    head.project_head(SimpleNamespace(shape=(1,16)),[1])
                with self.assertRaisesRegex(ValueError,'Unmatched direct'):
                    hybrid.direct_interval(before, hybrid.read_direct(d,ids), ('prefill','decode'), ids[-1:], target)
                with self.assertRaisesRegex(ValueError,'Unmatched direct'):
                    hybrid.direct_interval(before,end,('decode',),ids[-1:],{'prefill':{ids[-1]:1},'decode':{ids[-1]:2}})
            finally:
                for r in ranks:
                    r.counts.close()
                    if r.head_counts is not None:
                        r.head_counts.close()

    def test_head_route_scope_default_isolation_and_real_weight_bytes(self):
        for key in ('inkling-udq2','inkling-loop-iq4'):
            self.assertNotEqual(full_graph_runtime(key,'plain'),full_graph_runtime(key,'plain', output_head='direct'))
        self.assertIsNone(full_graph_runtime('hy4-iq2m','plain',output_head='direct'))
        for engine,mode in [('kaiLLM','dspark5'),('llama.cpp','plain')]:
            with self.assertRaises(ValueError):
                hybrid.head_policy(engine,mode,'direct')
        with self.assertRaises(ValueError):
            validate_kai_launch(['--enforce-eager','--compilation-config','{"cudagraph_mode":"FULL"}'])
        with TemporaryDirectory() as d:
            path=Path(d)/'weight';payload=b'1234';path.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest();spec={'files':{path.name:{'sha256':digest}}}
            self.assertEqual(hybrid.verify_files([path], spec),[digest]);path.write_bytes(b'1235')
            with self.assertRaisesRegex(ValueError,'digest differs'):
                hybrid.verify_files([path],spec)
