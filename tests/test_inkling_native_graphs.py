"""Check ragged replay input staging and target rollback without using a GPU."""
import os
from types import SimpleNamespace
import unittest

import torch


class DraftContextGraphTest(unittest.TestCase):
    def test_partial_spill_replay_updates_only_live_mapped_slots(self):
        from vllm_gguf_plugin.draft_context_graph import DraftContextGraph

        n, chunk, dims = 13, 4, 3
        features = torch.arange(n * dims).reshape(n, dims).float()
        positions = torch.arange(n)
        slots = torch.stack((torch.arange(n), torch.arange(n).flip(0)))
        p = SimpleNamespace(max_num_tokens=n, hidden_states=features, context_positions=positions,
                            _context_slot_mappings=slots)
        projection = DraftContextGraph(p, chunk_size=chunk)
        calls = []
        class Replay:
            def replay(self):
                calls.append(tuple(x.clone() for x in (projection.features, projection.positions, projection.slots)))
        projection.graph = Replay()
        evidence = []
        projection.replay(4, 13, SimpleNamespace(draft_replayed=lambda: evidence.append(len(calls))))
        self.assertEqual(evidence, [1, 2, 3])
        for i, (captured, pos, mapped) in enumerate(calls):
            lo, hi = 4 + chunk*i, min(4 + chunk*(i+1), 13)
            valid = hi-lo
            torch.testing.assert_close(captured[:valid], features[lo:hi])
            torch.testing.assert_close(pos[:valid], positions[lo:hi])
            torch.testing.assert_close(mapped[:, :valid], slots[:, lo:hi])
            self.assertTrue((mapped[:, valid:] == -1).all())
        projection.graph = None
        with self.assertRaisesRegex(RuntimeError, 'No captured'):
            projection.replay(0, 1)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'Requires CPU Triton interpreter')
class GraphVerifyConvTest(unittest.TestCase):
    def test_mixed_verification_matches_serial_after_all_rejection_counts(self):
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata, conv_spec_mixed
        from vllm_gguf_plugin.inkling.ref import convolve
        from vllm_gguf_plugin.inkling.triton_conv import conv_decode

        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(1034)
        def random(*shape):
            return torch.randn(shape, generator=rng)
        taps, heads, channels, offset = 4, 2, 16, 2
        weight = random(channels, taps)
        x = random(17, channels)
        slots = torch.tensor([[1], [0], [3], [2]], dtype=torch.int32)
        starts = torch.tensor([0, 6, 6, 13, 15], dtype=torch.int32)
        seq = torch.tensor([35, 0, 7, 42], dtype=torch.int32)
        verify = torch.tensor([True, True, False, True])
        for dynamic in (None, random(17, heads * taps)):
            for a in range(1, 7):
                states = random(4, channels+offset, taps-1+5).to(torch.float16)
                before = states.clone()
                accepted = torch.tensor([a, 1, 1, 7-a], dtype=torch.int32)
                md = ConvGraphMetadata(starts, seq, slots, None, accepted, verify)
                actual = conv_spec_mixed(SimpleNamespace(n_conv=taps, states=lambda: states), x, weight, offset,
                                         md, dynamic, heads, x.dtype)
                for i in (0, 3):
                    slot = int(slots[i, 0])
                    advance = int(accepted[i])-1
                    serial = before[slot:slot+1, :, advance:advance+taps-1].clone().contiguous()
                    for pos in range(int(starts[i]), int(starts[i+1])):
                        row = slice(pos, pos+1)
                        y = conv_decode(serial, x[row], weight, offset, torch.zeros(1, dtype=torch.int32),
                                        None if dynamic is None else dynamic[row], heads)
                        torch.testing.assert_close(actual[row], y, rtol=0, atol=0)
                        k = pos-int(starts[i])
                        torch.testing.assert_close(states[slot, offset:, k:k+taps-1], serial[0, offset:], rtol=0, atol=0)
                prompt = slice(6, 13)
                expected, history = convolve(x[prompt], weight, torch.zeros(taps-1, channels),
                                             None if dynamic is None else dynamic[prompt], heads)
                torch.testing.assert_close(actual[prompt], expected)
                torch.testing.assert_close(states[3, offset:, :taps-1], history.t().to(states.dtype))
                torch.testing.assert_close(states[0], before[0], rtol=0, atol=0)
                torch.testing.assert_close(states[:, :offset], before[:, :offset], rtol=0, atol=0)
                self.assertTrue((actual[15:] == 0).all())
