"""Set TRITON_INTERPRET=1 to exercise the actual paged-attention kernel on CPU."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'Requires the CPU Triton interpreter')
class PrefillBiasTests(unittest.TestCase):
    def test_partitioned_attention_matches_dense_with_partial_head_group(self):
        import torch

        source = Path(__file__).resolve().parents[1] / 'runtime-source/vllm_gguf_plugin/inkling/triton_attention.py'
        spec = importlib.util.spec_from_file_location('inkling_bias_cpu', source)
        attention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(attention)
        generator = torch.Generator().manual_seed(214)
        def random(*shape):
            return torch.randn(*shape, generator=generator).to(torch.float16) * .4

        query_starts = torch.tensor([0, 3, 8], dtype=torch.int32)
        lengths = torch.tensor([5, 10], dtype=torch.int32)
        pages = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        metadata = SimpleNamespace(query_start_loc=query_starts, query_start_loc_cpu=query_starts,
                                   num_reqs=2, seq_lens=lengths, block_table=pages)
        tokens, kv_heads, group, dim, extent, relative_dim = 8, 5, 3, 32, 4, 4
        heads = kv_heads * group
        q, rel = random(tokens, heads, dim), random(tokens, heads, relative_dim)
        sink, bias = random(tokens, heads), random(extent, relative_dim)
        cache = random(4, kv_heads, 8, 2*dim)
        full_bias = attention.bias_table(rel, bias, sink)
        bytes_per_kv_head = tokens * group * (extent+1) * 4
        for window in (0, 4):
            def run(heads_per_group):
                with patch.object(attention, 'PREFILL_BIAS_BYTES', heads_per_group * bytes_per_kv_head):
                    return attention.rel_attention_triton(q, rel, sink, bias, None, cache, metadata,
                                                          dim**-.5, window, extent)
            full = run(kv_heads)
            for size in (1, 2, 3, 4):
                self.assertTrue(torch.equal(run(size), full))
            for req in range(2):
                first, stop = query_starts[req:req+2].tolist()
                context = lengths[req].item() - (stop-first)
                for i in range(first, stop):
                    position = context + i-first
                    keys = torch.arange(max(0, position-window+1) if window else 0, position+1)
                    for head in range(heads):
                        kv = cache[pages[req, keys//8], head//group, keys%8].float()
                        logits = (kv[:, :dim] @ q[i, head].float()) * dim**-.5
                        logits += full_bias[i, head, (position-keys).clamp_max(extent)]
                        probabilities = torch.cat([torch.zeros(1), logits]).softmax(0)[1:]
                        dense = probabilities @ kv[:, dim:]
                        torch.testing.assert_close(full[i, head], dense, atol=.001, rtol=.002)
