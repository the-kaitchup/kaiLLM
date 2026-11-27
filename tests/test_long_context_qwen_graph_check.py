import copy
import json
import math
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks' / 'long_context'))
import check_qwen_graphs as check


def capture(tokens):
    return {'prompt_sha256': 'verified', 'tokens': tokens,
            'top_logprobs': [{str(i): -float(i) for i in range(8)} for _ in tokens]}


class Qualification(unittest.TestCase):
    def test_scores_exclude_positions_after_divergence(self):
        left = {'prose-64': [capture([0, 1, 0])]}
        right = {'prose-64': [capture([0, 2, 3])]}
        right['prose-64'][0]['top_logprobs'][1]['1'] = -2.5
        result = check.compare_scores(left, right)
        self.assertFalse(result['passed'])
        self.assertEqual((result['matching_greedy'], result['common_prefix_positions']), (1, 2))
        self.assertEqual(result['overlapping_top8_ids'], 16)
        self.assertEqual(result['max_abs_logprob_delta'], 1.5)
        self.assertEqual(result['first_difference']['output_position'], 1)
        self.assertTrue(check.compare_scores(left, left)['passed'])
        same_ids = copy.deepcopy(left)
        same_ids['prose-64'][0]['top_logprobs'][0]['7'] = -7.25
        self.assertTrue(check.compare_scores(left, same_ids, .25)['passed'])
        self.assertFalse(check.compare_scores(left, same_ids, .20)['passed'])
        right['prose-64'][0]['top_logprobs'][2]['3'] = math.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            check.compare_scores(left, right)
        with self.assertRaisesRegex(ValueError, 'Empty qualification'):
            check.compare_scores({'code': []}, {'code': []})

    def test_launch_verifies_last_effective_policy_and_exact_scope(self):
        launch = ['--compilation-config', '{"cudagraph_mode":"FULL"}', '--dtype', 'bfloat16', '--kv-cache-dtype', 'auto',
                  '--no-enable-prefix-caching', '--language-model-only', '--max-num-seqs', '2', '--max-num-batched-tokens', '2048']
        normal = {'method': 'mtp', 'num_speculative_tokens': 5}
        argv = launch + ['--speculative-config', json.dumps(normal)]
        check.validate_launch(argv, 2, 2048, 'mtp5', False, 1)
        forced = dict(normal, rejection_sample_method='synthetic', synthetic_acceptance_rates=[0.] * 5)
        argv += ['--speculative-config', json.dumps(forced)]
        self.assertEqual(check.validate_launch(argv, 2, 2048, 'mtp5', True, 1), forced)
        for mode, force, batch in [('plain', False, 2), ('dspine7', True, 2), ('mtp5', False, 2), ('mtp5', True, 4)]:
            with self.assertRaises(ValueError):
                check.validate_launch(argv, batch, 2048, mode, force, 1)
        with self.assertRaisesRegex(ValueError, 'one slot'):
            check.validate_mode(check.MODELS['qwen-chima'], 'dspine7', 2)
        with self.assertRaisesRegex(ValueError, 'no native MTP'):
            check.validate_mode(check.MODELS['qwen-chima'], 'mtp5', 1)

    def test_files_read_even_if_size_and_download_receipt_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'publisher/target/rev'
            base.mkdir(parents=True)
            file = base / 'target.gguf'
            file.write_bytes(b'abc')
            spec = {'repo': 'publisher/target', 'revision': 'rev', 'files': check.fingerprints([file])}
            (base / 'download.json').write_text(json.dumps(dict(spec, source='huggingface.co')))
            self.assertEqual(check.verified_files(root, 'target', spec), [file])
            file.write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                check.verified_files(root, 'target', spec)

    def test_receipt_keeps_scope_and_failed_common_prefix(self):
        key = 'qwen-udq2'
        model = check.MODELS[key]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            lock = check.full_graph_runtime(key, 'plain', revision=check.REVISION)[1]
            (directory / 'benchmark-source.sha256').write_text(lock)
            (directory / 'qwen.json').write_text('{"tokenizer_revision":"test"}')
            device = dict(uuid='GPU-test', compute_cap='12.0', name='Test GPU', driver_version='test')
            sources = ['direct', 'plain', 'native', 'forced']
            for final_token, success in [(0, True), (1, False)]:
                out = directory / str(final_token)
                args = SimpleNamespace(checkpoint=key, mode='mtp5', batch=2, kv_mib=128, timeout=30, max_logprob_delta=None,
                                       runtime=directory, weights=directory, banks=directory, devices=[0], out=out)
                def control(args, name, *parameters):
                    row = capture([0, final_token if name == 'forced' else 0])
                    return {'prose-64': [row, row]}, {'prepared_draft_files': None}
                with ExitStack() as stack:
                    for name, result in [('verified_files', [directory / 'm.gguf']), ('verify_prompt_bank', None),
                                         ('gpu_inventory', [device]), ('visible_cuda_uuids', ['GPU-test'])]:
                        stack.enter_context(patch.object(check, name, return_value=result))
                    runner = stack.enter_context(patch.object(check, 'run_control', side_effect=control))
                    if success:
                        check.main(args)
                    else:
                        with self.assertRaisesRegex(ValueError, 'score parity'):
                            check.main(args)
                receipt = json.loads((out / 'qualification.json').read_text())
                self.assertEqual([call.args[1] for call in runner.call_args_list], sources)
                self.assertEqual((receipt['passed'], receipt['mode'], receipt['batch']), (success, 'mtp5', 2))
                self.assertEqual((receipt['gpus'][0]['uuid'], receipt['target']['files']), ('GPU-test', model['files']))
                self.assertEqual(receipt['comparisons']['plain-forced']['passed'], success)

    def test_counters_exclude_earlier_accepts_and_require_actual_proposals(self):
        start = {'draft_tokens': 30, 'accepted_tokens': 12}
        self.assertEqual(check.token_delta(start, dict(start, draft_tokens=40), True),
                         {'proposed': 10, 'accepted': 0, 'rejected': 10})
        with self.assertRaises(ValueError):
            check.token_delta(start, start, True)
        with self.assertRaises(ValueError):
            check.token_delta(start, {'draft_tokens': 39, 'accepted_tokens': 13}, True)

    def test_mixed_step_requires_both_phases_in_flight_and_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            start = {'gpu': {'pid': 42, 'prefill': {'scheduled': 3, 'graph_replays': 3},
                             'decode': {'scheduled': 5, 'graph_replays': 5}}}
            file = path / 'gpu.42.counts'
            file.write_bytes(check.COUNTERS.pack(4, 3, 0, 0, 6, 5, 0, 0))
            with check.MixedSteps(path, start) as mixed:
                for _ in range(100):
                    if mixed.seen:
                        break
                    time.sleep(.001)
            finish = copy.deepcopy(start)
            with self.assertRaises(ValueError):
                mixed.finished(finish)
            finish['gpu']['prefill']['graph_replays'] = 4
            finish['gpu']['decode']['graph_replays'] = 6
            self.assertEqual(mixed.finished(finish), {'gpu': {'prefill': 4, 'decode': 6}})
            file.write_bytes(check.COUNTERS.pack(4, 4, 0, 0, 6, 5, 0, 0))
            with check.MixedSteps(path, start) as mixed:
                time.sleep(.003)
            with self.assertRaises(ValueError):
                mixed.finished(finish)

    def test_mixed_prompt_generation_lengths_follow_distinct_slots(self):
        def fake(engine, url, prompt, *, output_tokens, timeout):
            return prompt, output_tokens
        with patch.object(check, 'predict', side_effect=fake):
            self.assertEqual(check.capture_wave('unused', [[1], [2]], [12, 1], 1), [([1], 12), ([2], 1)])


if __name__ == '__main__':
    unittest.main()
