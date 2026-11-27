import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'benchmarks/long_context'))
from verify_inkling_iq4_selection import verify


class Selection(unittest.TestCase):
    def fixture(self, names, *, phase='prefill', scope=None, arch='86'):
        scope = scope or ('execute_context_1(2048)_generation_0(0)' if phase=='prefill'
                          else 'execute_context_0(0)_generation_1(1)')
        inner = [{'frame':{'name':name}, 'metrics':{'count':count,'device_id':'0','time (ns)':2000}}
                 for name,count in names.items()]
        trace=json.dumps([{'frame':{'name':scope},'children':inner}, {'CUDA':{'0':{'arch':arch}}}]).encode()
        return {'phase':phase,'measurement_iterations':1,'trace_sha256':hashlib.sha256(trace).hexdigest(),
                'gpu_uuid':'unit-test-device'},trace

    def test_both_prefill_policies_and_single_qualified_decode_projection(self):
        cases = [
            ('3090-prefill-and-gate-up','prefill','86',{'iq4_nl_moe_kernel_m32':44,'iq4_nl_moe_kernel_m16':44}),
            ('5090-prefill','prefill','120',{'iq4_nl_moe_kernel_m32':88}),
            ('3090-gate-up','decode','86',{'routed_iq4_b1_kernel':44,'iq4_nl_moe_kernel':44}),
        ]
        for policy,phase,arch,names in cases:
            counts=names|{'q8_0_moe_kernel':8}
            profile,trace=self.fixture(counts,phase=phase,arch=arch)
            self.assertEqual(verify(profile,trace,policy)['expert_kernel_calls'],counts)
            with self.assertRaises(ValueError):
                verify(profile,trace,'baseline')

    def test_packed_iq4_and_q8_physical_runtime(self):
        for phase, counts in [
            ('prefill', {'iq4_nl_moe_kernel_m32_n64_k64':44, 'iq4_nl_moe_kernel_m16_n64_k64':44,
                         'q8_0_moe_kernel_m32_n64':8}),
            ('decode', {'routed_iq4_b1_kernel':44, 'iq4_nl_moe_kernel_m4_n32':44, 'q8_0_moe_kernel':8}),
        ]:
            profile, trace = self.fixture(counts, phase=phase)
            self.assertEqual(verify(profile, trace, '3090-packed')['expert_kernel_calls'], counts)
            bad, raw = self.fixture(counts | {'q8_0_moe_kernel_m16_n64': 1}, phase=phase)
            with self.assertRaises(ValueError):
                verify(bad, raw, '3090-packed')

    def test_final_single_token_q8_and_iq4_selections(self):
        calls = {'routed_iq4_b1_kernel':44,'iq4_nl_moe_kernel_m4_n16':44,
                 'q8_0_moe_kernel_m4_n16':4,'q8_0_moe_kernel_m4_n64':4}
        profile, trace = self.fixture(calls, phase='decode')
        self.assertEqual(verify(profile,trace,'3090-routed')['expert_kernel_calls'],calls)
        with self.assertRaises(ValueError):
            verify(profile,trace,'3090-packed')
        metadata, missing = self.fixture(calls|{'q8_0_moe_kernel_m4_n64':3},phase='decode')
        with self.assertRaises(ValueError):
            verify(metadata,missing,'3090-routed')

    def test_partial_count_wrong_scope_or_modified_trace(self):
        counts={'iq4_nl_moe_kernel_m32':44,'iq4_nl_moe_kernel_m16':44,'q8_0_moe_kernel':8}
        profile,trace=self.fixture(counts)
        with self.assertRaises(ValueError):
            verify(profile,trace+b' ','3090-prefill-and-gate-up')
        for inputs in [self.fixture(counts|{'iq4_nl_moe_kernel_m16':43}),self.fixture(counts,scope='untimed_warmup')]:
            with self.assertRaises(ValueError):
                verify(*inputs,'3090-prefill-and-gate-up')


if __name__=='__main__':
    unittest.main()
