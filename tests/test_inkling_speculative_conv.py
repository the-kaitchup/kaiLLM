# SPDX-License-Identifier: Apache-2.0
"""Check the target's recurrent state after rejection at every draft position."""
from types import SimpleNamespace
import unittest

import torch

from vllm_gguf_plugin.inkling.conv import conv_step_batched
from vllm_gguf_plugin.inkling.ref import convolve


def metadata(decode_slots=(), decode_lengths=(), accepted=(), prefill_slots=(), prefill_lengths=()):
    def boundaries(lengths):
        return torch.tensor([0, *lengths], dtype=torch.long).cumsum(0)

    return SimpleNamespace(num_decodes=len(decode_slots), num_prefills=len(prefill_slots),
                           num_decode_tokens=sum(decode_lengths), num_prefill_tokens=sum(prefill_lengths),
                           query_start_loc_d=boundaries(decode_lengths), query_start_loc_p=boundaries(prefill_lengths),
                           state_indices_tensor_d=torch.tensor(decode_slots), state_indices_tensor_p=torch.tensor(prefill_slots),
                           has_initial_states_p=None, num_accepted_tokens=torch.tensor(accepted))


def check_rejection(accepted, dynamic):
    gen = torch.Generator().manual_seed(41)
    channels, kernel, heads, drafts = 8, 4, 2, 5
    weight = torch.randn(channels, kernel, generator=gen)
    store = torch.full((4, channels + 2, kernel - 1 + drafts), 99.0)
    state = SimpleNamespace(n_conv=kernel, states=lambda: store)

    def chunk(length):
        x = torch.randn(length, channels, generator=gen)
        taps = torch.randn(length, heads * kernel, generator=gen) if dynamic else None
        return x, taps

    def check(x, taps, md, spans):
        actual = conv_step_batched(state, x, weight, 1, md, taps, heads)
        for start, end, past in spans:
            expected, _ = convolve(x[start:end], weight, past, None if taps is None else taps[start:end], heads)
            torch.testing.assert_close(actual[start:end], expected)

    zero = torch.zeros(kernel - 1, channels)
    prompt, taps = chunk(12)
    check(prompt, taps, metadata(prefill_slots=[2, 1], prefill_lengths=[7, 5]), [(0, 7, zero), (7, 12, zero)])
    first, taps = chunk(10)
    check(first, taps, metadata([1, 2], [6, 4], [1, 1]), [(0, 6, prompt[9:12]), (6, 10, prompt[4:7])])
    baseline_a = torch.cat([prompt[7:12], first[:accepted]])[-(kernel-1):]
    baseline_b = torch.cat([prompt[:7], first[6:8]])[-(kernel-1):]
    second, taps = chunk(11)
    check(second, taps, metadata([2, 1], [3, 6], [2, accepted], [3], [2]),
          [(0, 3, baseline_b), (3, 9, baseline_a), (9, 11, zero)])
    assert (store[:, (0, -1)] == 99).all()


class InklingSpeculativeConvTest(unittest.TestCase):
    def test_rejection_matches_sequential_target(self):
        for dynamic in (False, True):
            for accepted in range(1, 7):
                with self.subTest(dynamic=dynamic, accepted=accepted):
                    check_rejection(accepted, dynamic)


if __name__ == '__main__':
    unittest.main()
