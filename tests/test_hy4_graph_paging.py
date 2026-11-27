"""Use the CPU Triton interpreter to check the actual paged Hy4 graph kernels."""
import os
from types import SimpleNamespace
import unittest

import torch


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'Requires the CPU Triton interpreter')
class Hy4PagedGraphTest(unittest.TestCase):
    def test_paged_group_cache_and_selection_across_ragged_prompt_and_decode(self):
        from vllm_gguf_plugin.hy4 import attention as ref
        from vllm_gguf_plugin.hy4.paged_index import write_rows, complete_groups, select_paged

        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(671)
        def rand(*shape):
            return torch.randn(shape, generator=rng)
        dim, heads, rot, ratio, chosen, page, spacing = 16, 4, 8, 4, 3, 4, 6
        width = ratio * chosen
        cache = torch.full((32*spacing, dim), 99.)
        table = torch.randperm(32, generator=rng, dtype=torch.int32)[:24].reshape(3,8)
        weights = ref.IndexerWeights(None, rand(heads, 12), rand(dim), rand(dim,dim), rand(dim,dim),
                                     rand(ratio,dim), rand(dim), heads, dim, rot, ratio, chosen, 10000., 1.e-5)
        key_bank = [rand(32,dim) for _ in range(3)]
        lengths = [0,0,0]
        for step in ((4,7,0),(4,8,5),(7,1,3),(1,8,7),(3,1,1)):
            starts = torch.tensor([0,*step],dtype=torch.int32).cumsum(0).int()
            req = torch.repeat_interleave(torch.arange(3,dtype=torch.int32),torch.tensor(step))
            p = torch.cat([torch.arange(lengths[r], lengths[r]+step[r]) for r in range(3)])
            keys = torch.cat([key_bank[r][lengths[r]:lengths[r]+step[r]] for r in range(3)])
            slots = table[req.long(),p//page].long()*page+p%page
            # Two padded rows have no live slot and must not modify any cache data.
            slots = torch.cat((slots,torch.tensor([-1,-1])))
            req = torch.cat((req,torch.tensor([0,0],dtype=torch.int32)))
            pos = torch.cat((p,torch.tensor([0,0])))
            before = cache.clone()
            write_rows(cache,torch.cat((keys,rand(2,dim))),slots,page,spacing)
            lengths = [n+k for n,k in zip(lengths,step)]
            # The final padded request owns neither a token nor a valid page.
            md = SimpleNamespace(query_start_loc=torch.cat((starts, starts[-1:])),
                                 seq_lens=torch.tensor([*lengths, 0], dtype=torch.int32),
                                 slot_mapping=slots, req_id_per_token=req,
                                 block_table=torch.cat((table, torch.full_like(table[:1], 1000000))))
            complete_groups(cache,spacing,md,len(pos),weights,page)
            for r,n in enumerate(lengths):
                for g in range(n//ratio):
                    end=(g+1)*ratio-1
                    row=int(table[r,end//page])*spacing+end%page
                    expected=ref.pool_groups(key_bank[r][g*ratio:(g+1)*ratio],weights)[0]
                    torch.testing.assert_close(cache[row],expected)
            guards=torch.arange(len(cache))%spacing>=page
            torch.testing.assert_close(cache[guards],before[guards])
            h=rand(len(pos),12)
            for q in (rand(len(pos),heads*dim),torch.zeros(len(pos),heads*dim)):
                got=select_paged(h,q,pos,cache,spacing,md,weights,page,32,width)
                for r,n in enumerate(lengths):
                    if not step[r]:
                        continue
                    chunk=slice(int(starts[r]),int(starts[r+1]))
                    if n <= width:
                        expected=ref.dense_indices(pos[chunk],width)
                    else:
                        expected=ref.select_groups(h[chunk],None,pos[chunk],ref.pool_groups(key_bank[r][:n],weights),
                                                   weights,q=q[chunk])
                    torch.testing.assert_close(got[chunk],expected,rtol=0,atol=0)

    def test_full_context_selects_the_same_2048_indices(self):
        from vllm_gguf_plugin.hy4.attention import IndexerWeights, select_groups
        from vllm_gguf_plugin.hy4.paged_index import select_paged

        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(7129)
        n, ratio, heads, dim, rotary, candidates, page = 73728, 4, 16, 128, 64, 2048, 16
        ng = n // ratio
        keys = torch.randn(ng, dim, generator=rng).to(torch.bfloat16)
        hidden = torch.randn(1, 128, generator=rng)
        q = torch.randn(1, heads*dim, generator=rng).to(torch.bfloat16)
        head_weights = torch.randn(heads, 128, generator=rng)
        weights = IndexerWeights(None, head_weights, None, None, None, None, None, heads, dim,
                                 rotary, ratio, candidates//ratio, 10000., 1.e-6)
        pos = torch.tensor([n-1])
        table = torch.randperm(n//page, generator=rng,dtype=torch.int32)[None]
        ends = (torch.arange(ng)+1)*ratio-1
        slots = table[0,ends//page].long()*page + ends%page
        cache = torch.zeros(n,dim,dtype=torch.bfloat16)
        cache[slots] = keys
        md = SimpleNamespace(req_id_per_token=torch.tensor([0],dtype=torch.int32), seq_lens=torch.tensor([n],dtype=torch.int32),
                             block_table=table, slot_mapping=slots[-1:])
        actual = select_paged(hidden,q,pos,cache,page,md,weights,page,n,candidates)
        expected = select_groups(hidden,None,pos,keys,weights,q=q)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


class Hy4EngramGraphTest(unittest.TestCase):
    def test_graph_gpu_gather_requires_registered_rows_before_copy(self):
        from unittest.mock import patch
        from vllm_gguf_plugin.hy4.engram import HostEngramTable

        table = HostEngramTable.__new__(HostEngramTable)
        table._graph_rows = None
        with patch.dict(os.environ, {"KAILLM_HY4_FULL_GRAPHS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "registered host Engram rows"):
                table.gather(SimpleNamespace(is_cuda=True, device=torch.device("cuda")), 8)

    def test_gather_from_stage_local_host_view_at_global_row_ids(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from vllm_gguf_plugin.hy4.engram import HostEngramTable

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)/'source'
            values = np.arange(80,dtype=np.float16).reshape(10,8)
            f.write_bytes(values.tobytes())
            table = HostEngramTable(str(f),0,values.nbytes,'F16',10,8)
            ids = torch.tensor([6,4,7,4])
            expected = table.gather(ids,8,dtype=torch.float32)
            # A CPU alias exercises the same row indexing and GGUF decode path.
            table._graph_rows = table.packed[3:8]
            table._first_graph_row = 3
            table.packed = None
            torch.testing.assert_close(table.gather(ids,8,dtype=torch.float32),expected,rtol=0,atol=0)

    def test_registers_only_stage_pages_without_copying_table(self):
        import gc
        import mmap
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import Mock, patch
        from vllm_gguf_plugin.hy4.engram import HostEngramTable

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)/'GGUF'
            f.write_bytes(b'\x00' * (6*mmap.PAGESIZE))
            table = HostEngramTable(str(f),0,f.stat().st_size,'F16',6*mmap.PAGESIZE//16,8)
            start, stop = mmap.PAGESIZE//16+3, 3*mmap.PAGESIZE//16-7
            driver = Mock()
            driver.cudaHostRegister.return_value.value = 0
            view = Mock(side_effect=lambda t:t)
            with (patch.dict(sys.modules,{'vllm.utils.torch_utils':SimpleNamespace(get_accelerator_view_from_cpu_tensor=view)}),
                  patch.object(torch.cuda,'cudart',return_value=driver),patch.object(torch.Tensor,'is_pinned',return_value=True)):
                table.prepare_graph_rows(start,stop)
            pointer, length, flags = driver.cudaHostRegister.call_args.args
            self.assertEqual(pointer,table.packed.data_ptr()+mmap.PAGESIZE)
            self.assertEqual(length,2*mmap.PAGESIZE)
            self.assertEqual(flags,10)
            self.assertEqual(view.call_args.args[0].data_ptr(),table.packed[start].data_ptr())
            del table
            gc.collect()
            driver.cudaHostUnregister.assert_called_once_with(pointer)
