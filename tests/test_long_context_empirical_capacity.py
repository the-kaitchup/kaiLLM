import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/long_context'))

import benchmark
import empirical_capacity as empirical
import kai_default
import report
import worker
from test_long_context_revisions import raw_plain


def trace():
    key='inkling-udq2'
    raw=raw_plain(key,benchmark.MANIFEST['models'][key]['revision'])
    raw['batch']=2;raw['prompt_hashes']=['first','second']
    for w in (*raw['prefill'],raw['full']):
        w['requests']*=2
    env=raw['environment']
    cmd=env['server_argv']
    cmd[cmd.index('--max-num-seqs')+1]='2'
    cmd[cmd.index('--kv-cache-memory-bytes')+1]=str(5120*2**20)
    cmd+=['--served-model-name','bench','--spec-tokens','0']
    actual=cmd[:-2]+['--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--no-enable-prefix-caching']
    pin=benchmark.MANIFEST['kaillm']['default_dispatch_runtime']
    lock=empirical.source_lock()
    env.update(kai_policy='default',kai_humming_deterministic=True,kai_source=pin['revision'],engine_revision=pin['source'],
               measured_gguf_sha256=raw['sha256'],effective_command=actual,max_concurrency=1.67,empirical_capacity_harness=lock)
    counts={k:0 for k in empirical.FIELDS};waves={};dispatch={}
    for name in empirical.WAVES:
        full=name=='full'
        updates=empirical.expected(2,65536 if full else 1)
        after={k:counts[k]+updates[k] for k in counts}
        waves[name]={'before':counts,'after':after};counts=after
        # Three mixed forwards already appear once in decode and once in prefill.
        masks=({'full':0,'direct':9,'direct_head':9,'other':0},
               {'full':65535 if full else 0,'direct':3 if full else 0,'direct_head':65538 if full else 0,'other':0})
        dispatch[name]={'example':dict(zip(('prefill','decode'),masks))}
    raw['kai_default_dispatch']=dispatch
    raw['kai_empirical_capacity']={'protocol':empirical.CONTRACT,'harness_sha256':lock,'waves':waves,
                                   'overlap':empirical.overlap(dispatch['full'])}
    return raw


def metrics_text(values):
    common='engine="0",model_name="bench"'
    lines=[]
    for name,key in empirical.METRICS.items():
        extra=(',source="local_compute"' if key=='local_compute' else ',finished_reason="length"' if key=='length_finishes' else '')
        lines.append(name+'{'+common+extra+'} '+str(values[key]))
    lines.append('vllm:request_success_total{engine="0",model_name="bench",finished_reason="abort"} 0')
    return '\n'.join(lines)


class EmpiricalCapacityTests(unittest.TestCase):
    def test_real_dispatch_arithmetic_and_nominal_capacity_survive_the_report(self):
        raw=trace()
        row=report.get_record(raw)
        assert report.timing_accepted(row)
        self.assertEqual(row['kai_kv'],{'bytes_per_device':5120*2**20,'full_context_sequences':1.67})
        self.assertEqual(row['kai_empirical_capacity']['overlap'],
                         {'decode_dispatches':65538,'decode_token_advances':131070,
                          'min_paired_decode_dispatches':65532,'max_other_decode_dispatches':6})
        self.assertIn('1.67 nominal slots',report.execution_label(row))
        self.assertNotIn('example',json.dumps(row['kai_empirical_capacity']))

    def test_fail_closed_on_partial_or_unmatched_evidence(self):
        expected=trace()
        for mutate in (
                lambda r:r['environment'].update(kai_source='unregistered'),
                lambda r:r['environment'].update(max_concurrency=0.98),
                lambda r:r['environment'].pop('empirical_capacity_harness'),
                lambda r:r.pop('kai_empirical_capacity'),
                lambda r:r['kai_empirical_capacity']['waves']['full']['after'].update(preemptions=1),
                lambda r:r['kai_empirical_capacity']['waves']['full']['after'].update(finished_preemptions=1),
                lambda r:r['kai_empirical_capacity']['waves']['full']['after'].pop('local_compute'),
                lambda r:r['kai_empirical_capacity']['waves']['full']['after'].update(cached_tokens=True),
                lambda r:r['kai_empirical_capacity']['waves']['prefill-2']['before'].update(finished_requests=0),
                lambda r:r['kai_empirical_capacity']['overlap'].update(decode_dispatches=True),
                lambda r:r['kai_default_dispatch']['full']['example']['decode'].update(full=131067,direct_head=131070),
                lambda r:r['kai_default_dispatch']['full']['example']['decode'].update(other=1),
                lambda r:r['kai_default_dispatch']['full']['example']['decode'].update(direct_head=0),
                lambda r:r['environment']['effective_command'].extend(['--speculative-config','{}']),
                lambda r:r['full']['requests'].pop()):
            raw=copy.deepcopy(expected);mutate(raw)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError,KeyError)):report.get_record(raw)
        row=report.get_record(expected)
        for key in ('kai_empirical_capacity','kai_kv','kai_default_dispatch'):
            changed=copy.deepcopy(row);changed.pop(key)
            self.assertFalse(report.timing_accepted(changed),key)
        full=copy.deepcopy(expected)
        full.pop('kai_empirical_capacity');full['environment'].pop('empirical_capacity_harness')
        with self.assertRaisesRegex(ValueError,'KV allocation'):report.get_record(full)
        full['environment']['max_concurrency']=2.1
        self.assertTrue(report.timing_accepted(report.get_record(full)))

    def test_prometheus_label_identity_integer_types_and_eventual_settling(self):
        before=empirical.expected(1,32);wanted=empirical.expected(2,65536)
        end={k:before[k]+wanted[k] for k in before}
        encoded=metrics_text(end)
        self.assertEqual(empirical.parse(encoded),end)
        for bad in (encoded+'\n'+encoded.splitlines()[0],encoded.replace('engine="0"','engine="1"'),
                    encoded.replace('model_name="bench"','model_name="other"'),encoded.replace(' 24576',' NaN'),
                    encoded.replace('vllm:prompt_tokens_total{','missing_name{')):
            with self.assertRaises(ValueError):empirical.parse(bad)
        delayed=dict(end,finished_requests=before['finished_requests'])
        replies=iter([OSError('transient metrics timeout'),delayed,end])
        def read(_):
            value=next(replies)
            if isinstance(value,Exception):raise value
            return value
        with patch.object(empirical.time,'sleep'):
            self.assertEqual(empirical.settled('unused',before,2,65536,read=read),end)
        with self.assertRaisesRegex(ValueError,'Extra'):
            empirical.settled('unused',before,2,65536,read=lambda _:dict(end,generated_tokens=end['generated_tokens']+1))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            empirical.settled('unused',before,2,65536,timeout=0,read=lambda _:delayed)
        with self.assertRaisesRegex(ValueError,'reset'):
            empirical.settled('unused',before,2,65536,read=lambda _:{k:0 for k in end})

    def test_benchmark_entry_point_keeps_the_common_waves_and_omits_failed_timings(self):
        src=trace(); env=src['environment']; initial={k:0 for k in empirical.FIELDS}
        for serial in (False, True):
            worker_counts=[0]*8
            metrics=dict(initial)
            calls=[]
            def finish(batch,output):
                changes=empirical.expected(batch,output)
                for key in metrics:metrics[key]+=changes[key]
                pp=4*batch
                dc=(0 if output==1 else (batch if serial else 1)*(output-1))
                for offset,n in ((1,pp),(3,pp),(4,dc),(7,dc)):worker_counts[offset]+=n
            def one(engine,url,rows,count,timeout,*,sampling):
                self.assertEqual(sampling,benchmark.sampling_for('inkling'))
                calls.append(('warmup',1,count));finish(1,count)
            def wave(engine,url,rows,count,timeout,*,sampling):
                self.assertEqual(sampling,benchmark.sampling_for('inkling'))
                calls.append(('timed',len(rows),count));finish(len(rows),count)
                return copy.deepcopy(src['full'] if count==65536 else src['prefill'][0])
            def settled(url,before,batch,count):
                assert empirical.delta(before,metrics)==empirical.expected(batch,count)
                return dict(metrics)
            with tempfile.TemporaryDirectory() as d:
                folder=Path(d)
                prompts=folder/'bank'; prompts.write_text(json.dumps({'domains':{'prose':{'rows':[[i] for i in range(33)],'hashes':['first','second']}}}))
                environment=folder/'env.json';environment.write_text(json.dumps(env))
                args=SimpleNamespace(prompts=prompts,environment=environment,checkpoint=src['checkpoint'],
                                     domain='prose',batch=2,engine='kaiLLM',url='unused',timeout=1,
                                     mode='plain',output=folder/'result.json',graph_dir=folder)
                with (patch.object(benchmark,'verify_prompt_bank'),patch.object(benchmark,'completion',side_effect=one),
                      patch.object(benchmark,'wave',side_effect=wave),patch.object(empirical,'snapshot',side_effect=lambda _:dict(metrics)),
                      patch.object(empirical,'settled',side_effect=settled),
                      patch.object(kai_default,'read',side_effect=lambda *_:{'example':(177,tuple(worker_counts))})):
                    if serial:
                        with self.assertRaisesRegex(ValueError,'overlap'):benchmark.run(args)
                        self.assertFalse(args.output.exists())
                    else:
                        benchmark.run(args)
                        proof=report.get_record(json.loads(args.output.read_bytes()))
                        self.assertTrue(report.timing_accepted(proof))
                self.assertEqual(calls,[('warmup',1,32)]+[('timed',2,1)]*3+[('timed',2,65536)])

    def test_existing_boolean_flag_and_matching_explicit_pool_required(self):
        raw=trace();env=raw['environment'];model=raw['checkpoint']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'results'
            args=SimpleNamespace(engine='kaiLLM', checkpoint=model,mode='plain',batch=2,kai_policy='default',llama_policy='strict',
                                 devices=[0],empirical_b2_capacity=True,kv_gib=5,out=out,weights=root,kai_commit=env['kai_source'])
            with patch.object(worker,'gpu_inventory',return_value=env['devices']),patch('devices.visible_cuda_uuids',return_value=['example']),patch.object(worker,'checkpoint_files',return_value=[root/'fixture']):
                out.mkdir()
                for dom in ('prose','code'):
                    f=out/f'{model}-kaiLLM-plain-b2-empirical-{5120*2**20}-kai-default-{dom}.json'
                    current=copy.deepcopy(raw);current['domain']=dom
                    current['sampling']=benchmark.sampling_for(benchmark.MANIFEST['models'][model]['family'])
                    f.write_text(json.dumps(current))
                # No model or GPU is touched by a verified idempotent retry.
                self.assertIsNone(worker.main(args))
                truncated=copy.deepcopy(current)
                truncated['full']['requests'][0]['output_tokens']=65535
                f.write_text(json.dumps(truncated))
                with self.assertRaisesRegex(ValueError,'request'):worker.main(args)
                current['environment']['empirical_capacity_harness']='wrong'
                f.write_text(json.dumps(current))
                with self.assertRaisesRegex(ValueError,'harness changed'):worker.main(args)
                args.kv_gib=None
                with self.assertRaisesRegex(ValueError,'explicit'):worker.main(args)
                args.batch=1
                with self.assertRaisesRegex(ValueError,'B2'):worker.main(args)


if __name__=='__main__':unittest.main()
