"""Exercise the tensor plans without CUDA or model weights."""
import importlib.util
from pathlib import Path
import unittest
import os
import sys

import torch


path = Path(__file__).resolve().parents[1]/'runtime-source/vllm_gguf_plugin/qwen35_graph_gdn.py'
spec = importlib.util.spec_from_file_location('qwen35_graph_gdn', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RaggedGDN(unittest.TestCase):
    def test_neutral_chunk_padding_preserves_live_recurrences(self):
        for lengths, capacity in [([63,1,65,0],144),([0,5,0,0],16),([2048],2048),([2,0,0,7],32)]:
            offsets = torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            source, valid, unpack, padded, mapping, per_seq = module.chunk_plan(offsets, capacity, 64)
            self.assertEqual(int(valid.sum()), sum(lengths))
            self.assertTrue(torch.equal(source[unpack[:sum(lengths)]],torch.arange(sum(lengths))))
            self.assertTrue(valid[unpack[:sum(lengths)]].all())
            for chunk, (owner, local) in enumerate(mapping.tolist()):
                self.assertEqual(chunk, per_seq[owner] + local)
            torch.manual_seed(88)
            x = torch.randn(capacity,2,dtype=torch.float64)
            factor = (torch.rand(capacity,2,dtype=torch.float64)*.3+.7)
            states = torch.randn(len(lengths),2,dtype=torch.float64)
            actual = torch.zeros(len(source),2,dtype=torch.float64)
            expected = torch.zeros(capacity,2,dtype=torch.float64)
            for row, length in enumerate(lengths):
                a = states[row]; b = a.clone()
                for pos in range(offsets[row], offsets[row+1]):
                    a = a * factor[pos] + x[pos]
                    expected[pos] = a
                for i in range(padded[row],padded[row+1]):
                    if valid[i]: b = b * factor[source[i]] + x[source[i]]
                    actual[i] = b
                torch.testing.assert_close(a,b)
            torch.testing.assert_close(actual[unpack[:sum(lengths)]], expected[:sum(lengths)])


class SpecGraphGDN(unittest.TestCase):
    def test_verifier_compaction_with_zero_empty_full_and_mixed_rows(self):
        from types import SimpleNamespace as NS
        for lengths, verify in [([0,0],[False,False]),([2,5,0],[True,True,False]),
                                ([91,1,6,0],[False,True,True,False]),([1],[True])]:
            offsets = torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
            md = NS(offsets=offsets, verify=torch.tensor(verify))
            capacity = max(1,sum(lengths)+3)
            cu, source, restore, mask = module.verification_plan(md,capacity,5)
            expected = [t for row, enabled in enumerate(verify) if enabled for t in range(offsets[row],offsets[row+1])]
            self.assertEqual(source[:int(cu[-1])].tolist(),expected)
            self.assertEqual(mask.nonzero().flatten().tolist(),expected)
            self.assertEqual(source[restore[mask]].tolist(),expected)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1','requires CPU Triton interpreter and compatible vLLM')
    def test_packed_recurrent_kernel_keeps_each_candidate_and_reads_accepted_slot(self):
        from types import SimpleNamespace as NS
        sys.path.insert(0,str(path.parents[1]))
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata
        try:
            from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
        except ImportError:
            self.skipTest('compatible vLLM kernels are required')
        rng = torch.Generator().manual_seed(195)
        steps, width, nslots = 5,4,24
        layer=NS(num_spec=steps, A_log=torch.tensor([-.6]),dt_bias=torch.tensor([.32]),
                 rearrange_mixed_qkv=lambda x: tuple(z.reshape(1,len(x),1,width) for z in x.split(width,dim=-1)))
        table=torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]],dtype=torch.int32)
        for verify_only in (False,True):
            lengths = [6,3,1]
            offsets=torch.tensor([0,6,9,10],dtype=torch.int32)
            md=ConvGraphMetadata(offsets,torch.tensor([13,60,27]),table,None,torch.tensor([3,1,4]),torch.tensor([True,False,True]))
            T=sum(lengths)
            x=torch.randn(T,width*3,generator=rng)
            a,b=torch.randn(T,1,generator=rng),torch.randn(T,1,generator=rng)
            cache=torch.randn(nslots,1,width,width,generator=rng)
            expected_cache=cache.clone()
            expected=torch.zeros(T,1,width)
            for r, length in enumerate(lengths):
                if verify_only and not md.verify[r]: continue
                col=int(md.accepted[r])-1 if md.verify[r] else 0
                h=cache[table[r,col],0].clone()
                for i in range(length):
                    t=int(offsets[r])+i
                    q,k,v=x[t].split(width)
                    q=q*torch.rsqrt((q*q).sum()+1e-6)/width**.5
                    k=k*torch.rsqrt((k*k).sum()+1e-6)
                    h=h*torch.exp(-torch.exp(layer.A_log)*torch.nn.functional.softplus(a[t]+layer.dt_bias))
                    h=h+(torch.sigmoid(b[t])*(v-h@k))[:,None]*k[None]
                    expected[t,0]=h@q
                    expected_cache[table[r,i if md.verify[r] else 0],0]=h
            result=module.run_recurrent(layer,x,a,b,cache,md,md.slots(),verify_only=verify_only)
            if verify_only:
                out,mask=result
                torch.testing.assert_close(out[mask],expected[mask],rtol=1.e-5,atol=1.e-5)
            else:
                torch.testing.assert_close(result,expected,rtol=1.e-5,atol=1.e-5)
            torch.testing.assert_close(cache,expected_cache,rtol=1.e-5,atol=1.e-5)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1','requires CPU Triton interpreter')
    def test_accepted_conv_history_and_recurrent_write_guards(self):
        sys.path.insert(0,str(path.parents[1]))
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata
        from vllm_gguf_plugin.lfm3_graph_conv import silu_spec_conv
        rng = torch.Generator().manual_seed(1138)
        channels, taps, steps = 7, 4, 5
        slots = torch.tensor([[2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16,17,18,19],[0,0,0,0,0,0]],dtype=torch.int32)
        weights = torch.randn(channels,taps,generator=rng)
        conv = torch.randn(20,channels,taps-1+steps,generator=rng)
        for lens, prior, accepts, ver in [([6,9,0,0],[72,0,0,0],[3,1,1,1],[True,False,False,False]),
                                         ([3,2,1,0],[74,9,12,0],[1,1,4,1],[True,False,True,False])]:
            offsets = torch.tensor([0,*torch.tensor(lens).cumsum(0).tolist()],dtype=torch.int32)
            md = ConvGraphMetadata(offsets,torch.tensor(lens)+torch.tensor(prior),slots,None,torch.tensor(accepts),torch.tensor(ver))
            tokens=sum(lens)+2
            x = torch.randn(tokens,channels,generator=rng)
            before=conv.clone()
            expected=torch.zeros_like(x)
            stored=before.clone()
            for r,n in enumerate(lens):
                if n==0: continue
                slot=slots[r,0].item()
                shift=accepts[r]-1 if ver[r] else 0
                old=before[slot,:,shift:shift+taps-1].T if prior[r] else x.new_zeros(taps-1,channels)
                span=torch.cat((old,x[offsets[r]:offsets[r+1]]))
                for j in range(n):
                    expected[offsets[r]+j]=torch.nn.functional.silu((span[j:j+taps].T*weights).sum(-1))
                data=span[1:] if ver[r] else span[-(taps-1):]
                stored[slot,:,:len(data)]=data.T
            actual=silu_spec_conv(x,weights,None,conv,md,steps)
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(conv,stored)
            state=torch.randn(20,1,2,3,generator=rng)
            target=state.clone()
            finals=torch.randn(len(lens)+1,1,2,3,generator=rng)
            for r,n in enumerate(lens):
                if n and not ver[r]: target[slots[r,0]]=finals[r]
            module.state_update(state,finals,slots[:,0],md,False)
            torch.testing.assert_close(state,target)


if __name__ == '__main__':
    unittest.main()
