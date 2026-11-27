"""Compare the mixed-length graph kernel to LFM3's gated depthwise convolution."""
import os
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "Requires CPU Triton interpreter")
class Lfm3GraphConvTest(unittest.TestCase):
    def test_gated_convolution_and_cached_history(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))
        import torch
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata
        from vllm_gguf_plugin.lfm3_graph_conv import gated_conv

        rng = torch.Generator().manual_seed(595)
        for dtype in (torch.float16, torch.float32):
            def random(*shape):
                return torch.randn(shape, generator=rng).to(dtype)
            for tokens, lengths, ctx in [(8, [2, 3, 0], [0, 4, 0]), (1, [1], [8]),
                                          (1, [1], [0]), (2, [0, 0], [0, 0])]:
                offsets = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
                lens = torch.tensor(lengths) + torch.tensor(ctx)
                slots = torch.tensor([[2], [1], [0]], dtype=torch.int32)[:len(lengths)]
                md = ConvGraphMetadata(offsets, lens, slots, None)
                for taps in (2, 4):
                    channels = 15
                    gate_batch = random(tokens, 3 * channels)
                    b, c, x = gate_batch.chunk(3, dim=-1)
                    weights = random(channels, taps)
                    for bias in (None, random(channels)):
                        state = random(4, taps - 1, channels).transpose(1, 2)
                        state[0].fill_(34)
                        expected_state = state.clone()
                        expected = torch.zeros_like(x)
                        for i, n in enumerate(lengths):
                            if not n:
                                continue
                            start = int(offsets[i])
                            slot = int(slots[i, 0])
                            if not ctx[i]:
                                state[slot].fill_(float("nan"))
                            prior = state[slot].t() if ctx[i] else torch.zeros_like(state[slot].t())
                            bx = (x * b)[start:start+n]
                            buffer = torch.cat((prior, bx), dim=0)
                            windows = buffer.unfold(0, taps, 1)
                            sums = (windows.float() * weights.float()).sum(-1)
                            if bias is not None:
                                sums += bias.float()
                            expected[start:start+n] = sums.to(dtype) * c[start:start+n]
                            expected_state[slot] = buffer[-(taps-1):].t()
                        actual = gated_conv(b, c, x, weights, bias, state, md)
                        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
                        torch.testing.assert_close(state, expected_state, equal_nan=True)
