import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks/long_context'))
from cuda_graphs import draft_proof_record, verify_draft_interval, verify_head_interval, proof_record, snapshot, validate_kai_launch, verify_interval

spec = importlib.util.spec_from_file_location('graph_backend', ROOT/'runtime-source/vllm_gguf_plugin/benchmark_graphs.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class Flags(list):
    def all(self):
        return all(self)


class TestGraphProof(unittest.TestCase):
    def test_preflight(self):
        for cmd in ([], ['--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}'],
                    ['--enforce-eager','--compilation-config','{"cudagraph_mode":"FULL"}']):
            with self.assertRaises(ValueError):
                validate_kai_launch(cmd)
        validate_kai_launch(['--compilation-config','{"cudagraph_mode":"FULL"}'])
        validate_kai_launch(['-cc={"cudagraph_mode":"FULL"}'])
        with self.assertRaises(ValueError):
            validate_kai_launch(['-cc={"cudagraph_mode":"FULL"}','--compilation-config.cudagraph_mode=NONE'])
        with self.assertRaises(ValueError):
            validate_kai_launch(['--enforce_eager=true','-cc={"cudagraph_mode":"FULL"}'])

    def test_ranked_wave_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            uuids = ['GPU-a','GPU-b']
            writers = [backend.TargetGraphCounters(directory, uid, 'FULL', False) for uid in uuids]
            try:
                desc = NS(cg_mode=NS(name='FULL'),num_active_loras=0,num_tokens=2048,uniform_token_count=None,max_query_len=None)
                for w in writers:
                    with self.assertRaises(RuntimeError):
                        w.ready([],2048)
                    w.ready([desc],2048)
                before = snapshot(directory,uuids)
                with self.assertRaises(ValueError):
                    snapshot(directory, [*uuids,'GPU-c'])
                state = NS(has_prefill=True,is_prefilling_np=Flags([True]))
                for w in writers:
                    w.replayed(w.require(state,'FULL'))
                middle = snapshot(directory,uuids)
                pre = verify_interval(before,middle,('prefill',))
                with self.assertRaises(ValueError):
                    verify_interval(before,middle,('decode',))
                state = NS(has_prefill=False,is_prefilling_np=Flags([False]))
                for w in writers:
                    w.replayed(w.require(state,'FULL'))
                end = snapshot(directory,uuids)
                full = verify_interval(before,end,('prefill','decode'))
                proof = proof_record([('prefill-1',pre),('prefill-2',pre),('prefill-3',pre),('full',full)])
                self.assertEqual(proof['prefill']['graph_replays'],8)
                self.assertEqual(proof['decode']['graph_replays'],2)
                self.assertEqual(proof['decode']['per_wave_by_gpu']['full'],{'GPU-a':1,'GPU-b':1})
                for w in writers:
                    mixed = NS(has_prefill=True,is_prefilling_np=Flags([True, False]))
                    w.replayed(w.require(mixed,'FULL'))
                overlap = verify_interval(end,snapshot(directory,uuids),('prefill','decode'))
                self.assertEqual(overlap['prefill'],overlap['decode'])
                with self.assertRaises(RuntimeError):
                    writers[0].require(state,'NONE')
                with self.assertRaises(ValueError):
                    verify_interval(end,snapshot(directory,uuids),('decode',))
            finally:
                for w in writers:
                    w.counts.close()


class TestDraftGraphProof(unittest.TestCase):
    def test_draft_rank_and_proposal_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            uuid = 'GPU-draft'
            w = backend.TargetGraphCounters(directory, uuid, 'FULL', False)
            desc = NS(cg_mode=NS(name='FULL'), num_active_loras=0, num_tokens=2048, uniform_token_count=None, max_query_len=None)
            try:
                with self.assertRaisesRegex(RuntimeError, 'implementation'):
                    w.install_drafter(object())
                with self.assertRaisesRegex(RuntimeError, 'No captured'):
                    w.install_drafter(NS(install_benchmark_graphs=lambda observer: 0))
                w.install_drafter(NS(install_benchmark_graphs=lambda observer: 3))
                w.ready([desc], 2048)
                before = snapshot(directory, [uuid])
                empty_probe = verify_draft_interval(before, before, [uuid], False)
                with self.assertRaises(ValueError):
                    verify_draft_interval(before, before, [uuid], True)
                with self.assertRaises(ValueError):
                    verify_draft_interval(before, before, ['GPU-other'], False)
                w.begin_proposal()
                w.draft_replayed()
                w.draft_replayed()
                with self.assertRaises(ValueError):
                    verify_draft_interval(before, snapshot(directory, [uuid]), [uuid], True)
                w.end_proposal()
                full = verify_draft_interval(before, snapshot(directory, [uuid]), [uuid], True)
                self.assertEqual(full[uuid], {'proposals': 1, 'completed_proposals': 1, 'graph_replays': 2})
                record = draft_proof_record([('prefill-1', empty_probe), ('prefill-2', empty_probe),
                                            ('prefill-3', empty_probe), ('full', full)])
                self.assertEqual(record['proposals'], 1)
                self.assertEqual(record['graph_replays'], 2)
                with self.assertRaisesRegex(RuntimeError, 'Unclassified'):
                    w.draft_replayed()
                with self.assertRaises(ValueError):
                    verify_draft_interval(before, snapshot(directory, [uuid]), [uuid], True)
            finally:
                w.counts.close()
                if w.draft_counts is not None:
                    w.draft_counts.close()


class TestOutputHeadProof(unittest.TestCase):
    def test_missing_terminal_rank_and_unmatched_launch(self):
        from struct import Struct
        with tempfile.TemporaryDirectory() as directory:
            w = backend.TargetGraphCounters(directory, 'GPU-head', 'FULL', False)
            peer = backend.TargetGraphCounters(directory, 'GPU-backbone', 'FULL', False)
            uuids = [peer.gpu_uuid, w.gpu_uuid]
            participants = [w.gpu_uuid]
            try:
                desc = NS(cg_mode=NS(name='FULL'), num_active_loras=0, num_tokens=2048, uniform_token_count=None, max_query_len=None)
                w.head = NS(sizes=[1, 2, 4])
                counter_path = Path(str(w.path)+'.head-counts')
                counter_path.write_bytes(Struct('<4Q').pack(0,0,0,0))
                for writer in (w, peer):
                    writer.ready([desc], 2048)
                before = snapshot(directory, uuids)
                counter_path.write_bytes(Struct('<4Q').pack(2,2,3,3))
                after = snapshot(directory, uuids)
                both = verify_head_interval(before, after, ('prefill','decode'), participants)
                self.assertEqual(both, {'prefill':{'GPU-head':2},'decode':{'GPU-head':3}})
                with self.assertRaises(ValueError):
                    verify_head_interval(before, after, ('prefill','decode'), uuids)
                counter_path.write_bytes(Struct('<4Q').pack(3,2,3,3))
                with self.assertRaises(ValueError):
                    verify_head_interval(before, snapshot(directory, uuids), ('prefill',), participants)
            finally:
                w.counts.close()
                peer.counts.close()
