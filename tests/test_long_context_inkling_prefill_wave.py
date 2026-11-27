import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/long_context'))
from verify_inkling_prefill_wave import verify


def trace(tiled, dense_counts=(96,96,48,48,16,48), scope='execute_context_1(2048)_generation_0(0)'):
    steps = 16
    names = {'iq4_nl_moe_kernel'+('_m32' if tiled else ''):88*steps,'q8_0_moe_kernel':8*steps}
    for i,(shape,count) in enumerate(zip(((3072,2304),(1024,2304),(2304,3072),(512,2304),(256,2304),(256,2304)),dense_counts)):
        names[f'_Z7hummingI10MmaOpClass5ShapeILj0ELj{shape[0]}ELj{shape[1]}EE{i}'] = count*steps
    children=[{'frame':{'name':n},'metrics':{'count':count,'time (ns)':1,'device_id':'0'}} for n,count in names.items()]
    return json.dumps([{'CUDA':{'0':{'arch':'120'}}},{'frame':{'name':scope},'children':children}]).encode()


class PrefillWave(unittest.TestCase):
    def test_ordinary_and_tiled_full_concurrent_waves(self):
        for tiled in (True,False):
            result = verify(trace(tiled),4,tiled)
            self.assertEqual(result['prefill_steps'],16)
            self.assertEqual(sum(result['expert_kernels'].values()),1536)
            self.assertEqual(sum(result['attention_kernels'].values()),5632)

    def test_wrong_tile_or_counts_with_same_total(self):
        with self.assertRaisesRegex(ValueError,'expert launches'):
            verify(trace(False),4,True)
        with self.assertRaisesRegex(ValueError,'dense attention'):
            verify(trace(True,(96,96,48,48,17,47)),4,True)

    def test_wrong_batch_generation_or_chunk_shape(self):
        for scope in ('execute_context_5(2048)_generation_0(0)','execute_context_1(512)_generation_0(0)',
                      'execute_context_1(2047)_generation_1(1)'):
            with self.assertRaisesRegex(ValueError,'uniquely attributed'):
                verify(trace(True,scope=scope),4,True)
