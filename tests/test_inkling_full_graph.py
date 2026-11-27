"""Check mixed-shape graph operators without requiring a GPU."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "Requires CPU Triton interpreter")
class InklingGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_tile_table_tracks_changed_request_boundaries(self):
        from vllm_gguf_plugin.inkling.triton_attention import BLOCK_M, tiles_for

        torch = self.torch
        offsets = torch.zeros(5, dtype=torch.int32)
        md = SimpleNamespace(query_start_loc=offsets)
        group = 3
        width = BLOCK_M // group
        tokens = width * 3
        for lengths in ([1, width, width + 1, 0], [tokens, 0, 0, 0], [0, 0, 0, 0], [0, 1, width + 1, 2]):
            offsets[1:] = torch.tensor(lengths).cumsum(0)
            result = tiles_for(md, group, tokens)
            expected = [[i, j] for i, length in enumerate(lengths) for j in range(0, length, width)]
            self.assertEqual(result[result[:, 0] >= 0].tolist(), expected)
            self.assertEqual(result.shape, (6, 2))

    def test_graph_conv_matches_request_reference_with_padding_and_resume(self):
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata, conv_mixed
        from vllm_gguf_plugin.inkling.ref import convolve
        from vllm_gguf_plugin.inkling.triton_conv import conv_decode

        torch = self.torch
        rng = torch.Generator().manual_seed(221)
        def random(*shape):
            return torch.randn(shape, generator=rng)
        channels, taps, heads, offset = 12, 4, 3, 2
        weights = random(channels, taps)
        slots = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 5, 5], dtype=torch.int32)
        seq_lens = torch.tensor([2, 5, 0], dtype=torch.int32)
        for block_size in (None, 4):
            for dynamic in (True, False):
                md = ConvGraphMetadata(offsets, seq_lens, slots, block_size)
                states = random(5, channels + 3, taps - 1).to(torch.float16)
                states[0].fill_(37)
                states[1, offset:offset + channels] = float("nan")
                state = SimpleNamespace(n_conv=taps, states=lambda: states)
                for lengths, contexts in (([2, 3, 0], [0, 2, 0]), ([3, 1, 0], [2, 3, 0]), ([0, 0, 0], [0, 0, 0])):
                    offsets[1:] = torch.tensor(lengths).cumsum(0)
                    seq_lens.copy_(torch.tensor(lengths) + torch.tensor(contexts))
                    x = random(7, channels)
                    dyn = random(7, heads * taps) if dynamic else None
                    expected = torch.zeros_like(x)
                    next_states = states.clone()
                    for i, n in enumerate(lengths):
                        if n == 0:
                            continue
                        slot = int(md.slots()[i])
                        start = int(offsets[i])
                        h = states[slot, offset:offset + channels].t().float()
                        h = h if contexts[i] else torch.zeros_like(h)
                        y, new = convolve(x[start:start+n], weights, h, None if dyn is None else dyn[start:start+n], heads)
                        expected[start:start+n] = y
                        next_states[slot, offset:offset + channels] = new.t()
                    actual = conv_mixed(state, x, weights, offset, md, dyn, heads, x.dtype)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(states, next_states, equal_nan=True)
        for fresh in (False, True):
            states = random(3, channels + 3, taps - 1).to(torch.float16)
            x = random(1, channels)
            dyn = random(1, heads*taps)
            prior = states[1, offset:offset + channels].t().float()
            if fresh:
                states[1, offset:offset + channels] = float("nan")
                prior.zero_()
            expected, new = convolve(x, weights, prior, dyn, heads)
            actual = conv_decode(states, x, weights, offset, slots[:1, 0], dyn, heads,
                                 seq_lens=torch.tensor([1 if fresh else 7]), offsets=torch.tensor([0, 1]))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(states[1, offset:offset+channels], new.t().to(states.dtype))
