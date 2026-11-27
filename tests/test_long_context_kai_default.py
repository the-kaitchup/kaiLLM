import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'/'long_context'))
import kai_default
from benchmark import MANIFEST, inkling_fused_q8
from coverage_matrix import graphs_verified, timing_accepted
from report import execution_label


class KaiDefaultTest(unittest.TestCase):
    def test_only_the_measured_mixed_policy_is_accepted(self):
        pin=MANIFEST['kaillm']['default_dispatch_runtime']
        uuid='GPU-example'
        zero=(0,)*8
        def read_wave(increments, phases):
            return kai_default.interval({uuid:(999,zero)},{uuid:(999,increments)},phases)
        raw={w:read_wave((0,4,0,1,0,0,0,0),('prefill',)) for w in kai_default.WAVES[:-1]}
        raw['full']=read_wave((0,4,0,1,65535,0,0,65535),kai_default.PHASES)
        proof=kai_default.compact(raw,[uuid])
        row=dict(engine='kaiLLM',mode='plain',checkpoint='inkling-udq2',gpu=['RTX 5080'],
                 kai_source=pin['revision'],engine_revision=pin['source'],sha256=['file-digest'],
                 measured_gguf_sha256=['file-digest'],kai_humming_deterministic=True,kai_default_dispatch=proof)
        self.assertTrue(timing_accepted(row))
        self.assertFalse(graphs_verified(row))
        self.assertIn('prefill eager, decode graphs, eager head', execution_label(row))
        self.assertFalse(timing_accepted(dict(row,kai_source='other')))
        self.assertFalse(timing_accepted(dict(row,cuda_graphs={})))
        corrupt=copy.deepcopy(row)
        corrupt['kai_default_dispatch']['full'][0]['decode']['other']=1
        self.assertFalse(timing_accepted(corrupt))
        with self.assertRaisesRegex(ValueError,'dispatch'):
            read_wave((0,4,0,1,0,5,0,5),kai_default.PHASES)

    def test_optimized_source_and_previous_receipts_keep_their_own_model_and_mode(self):
        manifest = copy.deepcopy(MANIFEST)
        old = manifest['kaillm']['default_dispatch_runtime']
        new = {'revision': 'candidate-rev', 'source': 'candidate-sha', 'models': ['inkling-loop-iq4']}
        manifest['kaillm']['default_dispatch_runtimes'] = [new]
        self.assertEqual(kai_default.runtime(manifest, 'inkling-loop-iq4', 'plain', new['revision']),
                         (new['revision'], new['source']))
        self.assertEqual(kai_default.runtime(manifest, old['models'][0], 'plain', old['revision']),
                         (old['revision'], old['source']))
        self.assertIsNone(kai_default.runtime(manifest, 'inkling-udq2', 'plain', new['revision']))
        self.assertIsNone(kai_default.runtime(manifest, 'inkling-loop-iq4', 'dspark5', new['revision']))

    def test_q8_grouping_is_specific_to_the_measured_checkpoint_source_gpu_and_batch(self):
        name = 'NVIDIA GeForce RTX 3090'
        candidate = dict(revision='grouped', source='sha', models=['inkling-loop-iq4'], inkling_fused_q8=[name])
        args = ['inkling-loop-iq4', 'plain', 1, [name], ('grouped', 'sha')]
        with patch.dict(MANIFEST['kaillm'], default_dispatch_runtimes=[candidate]):
            self.assertTrue(inkling_fused_q8(*args))
            for index, other in enumerate(('inkling-udq2', 'dspark5', 2, [name, name], ('older', 'sha'))):
                changed = args.copy()
                changed[index] = other
                self.assertFalse(inkling_fused_q8(*changed))
            self.assertFalse(inkling_fused_q8(*args[:3], ['NVIDIA GeForce RTX 5090'], args[4]))

    def test_sampler_reads_the_actual_device_and_process(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'GPU-fixture.998.json'
            p.write_text(json.dumps(dict(schema=1,uuid='GPU-fixture',pid=998,config_mode='FULL_DECODE_ONLY',
                                         enforce_eager=False,fields=list(kai_default.MODES))))
            p.with_suffix('.dispatch-counts').write_bytes(struct.pack('<8Q',1,2,3,4,5,6,7,8))
            observed=kai_default.read(Path(d),['GPU-fixture'])
            self.assertEqual(observed['GPU-fixture'],(998,(1,2,3,4,5,6,7,8)))
            with self.assertRaisesRegex(ValueError,'device'):
                kai_default.read(Path(d),['GPU-other'])

class LfmNormalPolicy(unittest.TestCase):
    def test_default_mode_is_consistent_across_launch_and_physical_server(self):
        command=['vllm','--compilation-config',json.dumps({'cudagraph_mode':'FULL_AND_PIECEWISE'}),
                 '--additional-config',json.dumps({'cudagraph_specialize_prefill':True})]
        self.assertEqual(kai_default.launch(command,'lfm'),'FULL_AND_PIECEWISE')
        for altered in (command[:-2], command[:-1]+['{"cudagraph_specialize_prefill":false}'],
                        [*command,'--speculative-config','{}'], [*command,'--enforce-eager']):
            with self.assertRaises(ValueError):kai_default.launch(altered,'lfm')
        with self.assertRaises(ValueError):kai_default.launch(command,'inkling')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'GPU-fixture.998.json'
            p.write_text(json.dumps(dict(schema=1,uuid='GPU-fixture',pid=998,config_mode='FULL_AND_PIECEWISE',
                                         enforce_eager=False,fields=list(kai_default.MODES))))
            p.with_suffix('.dispatch-counts').write_bytes(struct.pack('<8Q',0,0,0,0,0,0,0,0))
            before=kai_default.read(Path(d),['GPU-fixture'],'lfm')
            p.with_suffix('.dispatch-counts').write_bytes(struct.pack('<8Q',2,1,0,2,6,0,0,6))
            after=kai_default.read(Path(d),['GPU-fixture'],'lfm')
            proof=kai_default.interval(before,after,('prefill','decode'))
            self.assertEqual(proof['GPU-fixture']['prefill']['direct'],1)
            # The v1 runtime places PIECEWISE, unbatched unknowns and unclassified work in "other".
            p.with_suffix('.dispatch-counts').write_bytes(struct.pack('<8Q',2,1,1,2,6,0,0,6))
            with self.assertRaises(ValueError):kai_default.interval(before,kai_default.read(Path(d),['GPU-fixture'],'lfm'),('prefill','decode'))
            with self.assertRaises(ValueError):kai_default.read(Path(d),['GPU-fixture'],'inkling')

    def test_real_piecewise_and_unrecognized_forwards_have_distinct_counters(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'GPU-fixture.998.json'
            p.write_text(json.dumps(dict(schema=2, uuid='GPU-fixture', pid=998, config_mode='FULL_AND_PIECEWISE',
                                         enforce_eager=False, fields=list(kai_default.PIECEWISE_MODES))))
            counts = p.with_suffix('.dispatch-counts')
            counts.write_bytes(struct.pack('<10Q', *([0]*10)))
            before = kai_default.read(d, ['GPU-fixture'], 'lfm')
            counts.write_bytes(struct.pack('<10Q', 0,0,0,1,3, 3,0,0,3,0))
            actual = kai_default.interval(before, kai_default.read(d, ['GPU-fixture'], 'lfm'), ('prefill','decode'))
            self.assertEqual(actual['GPU-fixture']['prefill']['piecewise'], 3)
            counts.write_bytes(struct.pack('<10Q', 0,0,1,1,3, 3,0,0,3,0))
            with self.assertRaises(ValueError):
                kai_default.interval(before, kai_default.read(d, ['GPU-fixture'], 'lfm'), ('prefill','decode'))

    def test_lfm_row_requires_source_specific_pin_and_original_gemm_policy(self):
        uid='GPU-fixture'
        proof={};one=dict(full=2,direct=0,other=0,direct_head=2);zero={key:0 for key in one}
        for label in kai_default.WAVES:
            proof[label]=[{'prefill':one,'decode':one if label=='full' else zero}]
        model='lfm-qad'
        manifest={'models':{model:{'family':'lfm'}},'kaillm':{'default_dispatch_runtimes':[{'models':[model],'revision':'r','source':'s'}]}}
        row={'checkpoint':model,'engine':'kaiLLM','mode':'plain','gpu':[uid],'batch':1,'kai_kv':dict(bytes_per_device=1,full_context_sequences=1),
             'kai_source':'r','engine_revision':'s','kai_default_config_mode':'FULL_AND_PIECEWISE','kai_humming_deterministic':False,
             'sha256':['weight'],'measured_gguf_sha256':['weight'],'kai_default_dispatch':proof}
        self.assertTrue(kai_default.verified(row,manifest))
        for update in ({'kai_humming_deterministic':True},{'kai_default_config_mode':'FULL_DECODE_ONLY'},{'kai_source':'unmeasured'},{'mode':'ngram3'}):
            self.assertFalse(kai_default.verified(dict(row,**update),manifest))
