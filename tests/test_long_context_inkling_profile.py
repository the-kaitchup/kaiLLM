from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/long_context'))
from profile_inkling_runtime import kernel_summary

class ProtonTests(unittest.TestCase):
    def test_physical_exclusive_kernel_times_aggregated_under_real_worker(self):
        def kernel(name,t,c):
            return {'frame':{'name':name},'metrics':{'time (ns)':t,'device_id':'0','count':c},'children':[]}
        children=[kernel('_rel_attn_split_kernel',18,2),kernel('humming',11,2),kernel('conv',7,2),kernel('moe',10,1)]
        phase={'frame':{'name':'execute_context_0(0)_generation_1(1)'},'metrics':{'time (ns)':999},'children':children}
        root={'frame':{'name':'ROOT'},'metrics':{},'children':[phase,kernel('moe',4,1)]}
        out=kernel_summary([root,{'CUDA':{'0':{'arch':'86'}}}])
        self.assertEqual((out['device_kernel_ns'],out['device_kernel_count']),(50,8))
        self.assertEqual(out['kernels'][1],{'name':'moe','calls':2,'total_ns':14,'share':.28})
        self.assertEqual(out['scope_kernel_ns']['outside worker model annotation'],4)
        with self.assertRaisesRegex(ValueError,'actual Inkling'):
            kernel_summary([dict(root,children=[kernel('cudaGraphLaunch',50,1)])])
