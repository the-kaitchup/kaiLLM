import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/long_context'))
import check_inkling_native_graphs as check


class NativeGraphCheckTest(unittest.TestCase):
    def test_overlap_requires_full_concurrent_request_slots_and_four_servers(self):
        for slots, full, repeated in ((1, True, True), (2, False, True), (2, True, False)):
            with self.subTest(slots=slots, full=full, repeat=repeated):
                with self.assertRaisesRegex(ValueError, 'two full-context slots'):
                    check.main(SimpleNamespace(check_overlap=True, slots=slots, full_context=full, repeat_plain=repeated))

    def test_overlap_observer_wraps_ordinary_launch_and_captures_graph_evidence(self):
        import check_inkling_draft_overlap as overlap
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'benchmark-source.sha256').write_text('fixture')
            args = SimpleNamespace(out=root, runtime=root, port=18889, slots=2, full_context=True,
                                   plain_kv_gib=12, native_kv_gib=12, banks=root, check_overlap=True)
            devices = [{'uuid': 'test', 'memory.free': 85000, 'memory.total': 90000}]
            cmd = ['serve', '--compilation-config', '{"cudagraph_mode":"FULL"}']
            with (patch.object(check.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps({'command':cmd}))),
                  patch.object(check.subprocess, 'Popen') as launch, patch.object(check, 'wait_server'),
                  patch.object(check, 'stop_server'), patch.object(check, 'full_admission', return_value={}),
                  patch.object(check, 'snapshot', return_value={'test':{'head_sizes':[1]}}),
                  patch.object(check, 'capture', return_value={}), patch.object(overlap, 'capture') as measured,
                  patch.object(check, 'graph_proof', return_value={'target': 'replayed'}) as proof):
                check.run_control(args, 'plain', root/'target', root/'draft', {}, devices, label='plain-repeat')
            argv = launch.call_args.args[0]
            self.assertEqual(Path(argv[1]).name, 'verify_batch_schedule.py')
            self.assertEqual(argv[2:6], ['serve', '--directory', str(root/'plain-repeat/schedule'), '--'])
            self.assertEqual(Path(argv[7]).name, 'serve.py')
            self.assertEqual(measured.call_args.args[0].mode, 'plain')
            self.assertEqual(proof.call_count, 5)
            self.assertEqual(json.loads((root/'plain-repeat/graph-proof-overlap.json').read_bytes()),
                             {'target':'replayed'})

    def test_refuses_empty_head_inventory_before_real_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'benchmark-source.sha256').write_text('fixture')
            args = SimpleNamespace(out=root, runtime=root, port=18889, slots=1)
            devices = [{'uuid': 'test', 'memory.free': 16000, 'memory.total': 16384}]
            cmd = ['serve', '--compilation-config', '{"cudagraph_mode":"FULL"}']
            dry = SimpleNamespace(stdout=json.dumps({'command': cmd}))
            with (patch.object(check.subprocess, 'run', return_value=dry),
                  patch.object(check.subprocess, 'Popen'), patch.object(check, 'wait_server'),
                  patch.object(check, 'stop_server') as stopped,
                  patch.object(check, 'snapshot', return_value={'test': {'head_sizes': []}}),
                  patch.object(check, 'capture') as capture):
                with self.assertRaisesRegex(ValueError, 'before the request'):
                    check.run_control(args, 'plain', root/'target', root/'draft', {}, devices)
                capture.assert_not_called()
                stopped.assert_called_once()

    def test_checkpoint_receipt_binds_exact_publisher_revision_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'repo': 'fixture/inkling', 'revision': 'abc', 'files': {'original.gguf': {'bytes': 3,
                    'sha256': hashlib.sha256(b'abc').hexdigest()}}}
            directory = root / spec['repo'] / spec['revision']
            directory.mkdir(parents=True)
            file = directory / 'original.gguf'
            file.write_bytes(b'abc')
            receipt = directory / 'download-inkling.json'
            receipt.write_text(json.dumps({'source': 'huggingface.co', **spec}))
            self.assertEqual(check.checkpoint_file(root, 'inkling', spec), file)
            file.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                check.checkpoint_file(root, 'inkling', spec)
            file.write_bytes(b'ab')
            with self.assertRaisesRegex(ValueError, 'size'):
                check.checkpoint_file(root, 'inkling', spec)
            receipt.write_text(json.dumps({'source': 'huggingface.co', **spec, 'revision': 'other'}))
            with self.assertRaisesRegex(ValueError, 'receipt'):
                check.checkpoint_file(root, 'inkling', spec)

    def test_full_admission_checks_exact_concurrency_and_context(self):
        template = 'GPU KV cache size: {} tokens, Maximum concurrency for {} tokens per request: {}x'
        result = check.full_admission(template.format('74,356', '73,728', '1.01'), 1)
        self.assertEqual(result['tokens'], 74356)
        for row in [('73,696', '73,728', '1.00'), ('5,414', '512', '10.57'), ('90,000', '73,728', '1.01')]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                check.full_admission(template.format(*row), 1)
        with self.assertRaises(ValueError):
            check.full_admission(template.format('74,356', '73,728', '1.01'), 2)

    def test_full_native_probe_uses_full_context_and_retains_forced_spec_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'benchmark-source.sha256').write_text('fixture')
            args = SimpleNamespace(out=root, runtime=root, port=18889, slots=1, full_context=True,
                                   plain_kv_gib=3.0625, native_kv_gib=3.75)
            devices = [{'uuid': 'test', 'memory.free': 16000, 'memory.total': 16384}]
            argv = ['serve', '--compilation-config', '{"cudagraph_mode":"FULL"}',
                    '--speculative-config', '{"method":"dflash","num_speculative_tokens":5}']
            dry = SimpleNamespace(stdout=json.dumps({'command':argv}))
            def ready(proc, url, log):
                log.write_text('GPU KV cache size: 75,266 tokens, Maximum concurrency for 73,728 tokens per request: 1.02x')
            with (patch.object(check.subprocess, 'run', return_value=dry),
                  patch.object(check.subprocess, 'Popen') as server, patch.object(check, 'wait_server', side_effect=ready),
                  patch.object(check, 'stop_server'), patch.object(check, 'snapshot',
                      return_value={'test': {'head_sizes':[1,6], 'draft_captures':6}}),
                  patch.object(check, 'capture', return_value={}) as capture,
                  patch.object(check, 'verify_interval'), patch.object(check, 'verify_head_interval'),
                  patch.object(check, 'verify_draft_interval'), patch.object(check, 'write_json')):
                check.run_control(args, 'forced', root/'target', root/'draft', {}, devices)
            cmd = server.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--max-model-len')+1], '73728')
            self.assertEqual(cmd[cmd.index('--max-num-batched-tokens')+1], '2048')
            self.assertEqual(cmd[cmd.index('--kv-cache-memory-bytes')+1], str(3840 << 20))
            extra = json.loads(cmd[cmd.index('--extra-vllm-args')+1])
            self.assertIn('--no-enable-prefix-caching', extra)
            spec = json.loads(extra[extra.index('--speculative-config')+1])
            self.assertEqual(spec['synthetic_acceptance_rates'], [0.]*5)
            self.assertEqual([c.kwargs['prompt_length'] for c in capture.call_args_list], [128,511,2048,8192])

    def test_plain_self_repeat_reports_first_difference_per_domain(self):
        plain = {128: {'prompt_hash':'same', 'client_concurrency':1,
                       'output_ids':{'prose':list(range(48)), 'code':list(range(48))},
                       'counts':{'draft_tokens':0, 'accepted_tokens':0}}}
        repeated = copy.deepcopy(plain)
        self.assertEqual(check.compare_plain_repeats(plain,repeated), {'tokens_checked':96,'first_differences':[]})
        repeated[128]['output_ids']['prose'][25:] = [43]*23
        agreement = check.compare_plain_repeats(plain,repeated)
        self.assertEqual(agreement['first_differences'], [
            {'prefix':128,'domain':'prose','first_index':25,'expected':25,'actual':43}])
        repeated[128]['prompt_hash'] = 'different'
        with self.assertRaisesRegex(ValueError,'input'):
            check.compare_plain_repeats(plain,repeated)
