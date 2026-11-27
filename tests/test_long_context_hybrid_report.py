import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks/long_context'))
import report
from benchmark import full_graph_runtime
from historical_timings import previously_accepted
from test_long_context_revisions import raw_plain


def hybrid_raw(stages=1, count=1):
    key = 'inkling-loop-iq4'
    raw = raw_plain(key, report.MANIFEST['models'][key]['revision'])
    env = raw['environment']
    source, lock = full_graph_runtime(key, 'plain', output_head='direct')
    devices = [dict(env['devices'][0], uuid='private-'+str(i)) for i in range(count)]
    uuids = [r['uuid'] for r in devices]
    heads = uuids[-(count//stages):]
    env.update(benchmark_output_head='direct', direct_head_devices=heads, cuda_graph_head_devices=[],
               cuda_graph_drafter_devices=[], measured_gguf_sha256=raw['sha256'], kai_humming_deterministic=True,
               kai_source=source, engine_revision=lock, gpu_names=['NVIDIA GeForce RTX 5090']*count, devices=devices)
    if stages != 1:
        env['server_argv'].extend(['--pipeline-parallel-size', str(stages)])
    raw['target_head_direct'] = {};raw['cuda_graphs'] = {}
    for phase, labels in report.GRAPH_WAVES.items():
        waves = {label: {uid: 1+i for i, uid in enumerate(uuids)} for label in labels}
        total = sum(sum(o.values()) for o in waves.values())
        raw['cuda_graphs'][phase]={'scope':'timed-wave','eager_steps':0,'unclassified_steps':0,
                                  'graph_replays':total,'per_wave_by_gpu':waves}
        terminal = {label:{uid:waves[label][uid] for uid in heads} for label in labels}
        raw['target_head_direct'][phase]={'scope':'timed-wave','execution':'direct',
                                          'completed_calls':sum(sum(o.values()) for o in terminal.values()),
                                          'per_wave_by_gpu':terminal}
    return raw


class HybridReportTests(unittest.TestCase):
    def test_terminal_ranks_weights_source_and_every_wave(self):
        raw=hybrid_raw(stages=2,count=4)
        compact=report.get_record(raw)
        self.assertTrue(report.timing_accepted(compact));self.assertFalse(report.graphs_verified(compact))
        self.assertNotIn('private-',str(compact));self.assertEqual(compact['target_head_direct']['device_ranks'],[2,3])
        for mutate in (
                lambda d:d['target_head_direct']['prefill']['per_wave_by_gpu'].pop('prefill-2'),
                lambda d:d['target_head_direct']['decode']['per_wave_by_gpu']['full'].update({'private-2':8}),
                lambda d:d['target_head_direct']['decode'].update(scope='startup'),
                lambda d:d['target_head_direct']['decode'].update(completed_calls=100),
                lambda d:d['cuda_graphs']['decode'].update(eager_steps=1),
                lambda d:d['environment'].update(direct_head_devices=['private-0','private-1']),
                lambda d:d['environment'].update(measured_gguf_sha256=['bad']),
                lambda d:d['environment'].update(kai_source='unregistered'),
                lambda d:d['environment']['server_argv'].append('--enforce_eager=true'),
                lambda d:d.update(target_head_cuda_graphs={}),
                lambda d:d.update(drafter_cuda_graphs={})):
            changed=copy.deepcopy(raw);mutate(changed)
            with self.assertRaises(ValueError): report.get_record(changed)

    def test_historical_eager_kept_hybrid_and_strict_compared_without_duplicate_cells(self):
        graph_dir=ROOT/'benchmarks/long_context'
        archive=[json.loads(line) for line in (graph_dir/'results.jsonl').read_text().splitlines()]
        eager=next(r for r in archive if previously_accepted(r) and report.historical_graph_mode(r)=='eager')
        row=report.get_record(hybrid_raw());ref={**row,'engine':'llama.cpp','engine_revision':'fixture','full_seconds':row['full_seconds']/2}
        del ref['target_head_direct'],ref['measured_gguf_sha256'];self.assertTrue(report.graphs_verified(ref))
        self.assertFalse(report.graphs_verified(eager));self.assertTrue(report.timing_accepted(eager))
        self.assertEqual(report.counts([row,ref,eager],[])['filled'],3)
        for view in report.tables([row,ref,eager]):
            self.assertIn('kaiLLM Graphed backbone, eager head · llama.cpp Forced graphs',view)
            self.assertIn('\\color{red}',view);self.assertIn('| Eager |',view)
            data=[line for line in view.splitlines() if '| kaiLLM Graphed backbone' in line or line.endswith('| Full graphs |')]
            self.assertEqual(len(data),1)
        other={**ref,'hardware_id':'different-physical-card'}
        for view in report.tables([row,other]):
            self.assertIn('| Graphed backbone, eager head |',view)
            self.assertNotIn('\\color{red}',view)

    def test_collector_cli_preserves_direct_evidence_and_counts(self):
        with TemporaryDirectory() as d:
            root=Path(d);raw_path=root/'input';raw_path.mkdir();raw=hybrid_raw()
            (raw_path/'fixture-prose.json').write_text(json.dumps(raw))
            argv=['report',str(root/'results.jsonl'),str(root/'speed.md'),str(raw_path)]
            with patch.object(sys,'argv',argv):report.main()
            with patch.object(sys,'argv',argv):report.main()
            compact=json.loads((root/'results.jsonl').read_bytes());self.assertTrue(report.direct_head_verified(compact))
            self.assertNotIn('private-',str(compact))
            for fn in ('speed.md','speed-by-model.md'):
                self.assertIn('Graphed backbone, eager head',(root/fn).read_text())
