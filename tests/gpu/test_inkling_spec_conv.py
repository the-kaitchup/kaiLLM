# SPDX-License-Identifier: Apache-2.0
"""Compare speculative convolution against successive single-token target kernel calls."""
import argparse
from types import SimpleNamespace

import torch

from vllm_gguf_plugin.inkling.triton_conv import conv_decode, conv_verify


def check(device, state_dtype, dynamic, channel_first):
    taps, drafts, channels, offset = 4, 5, 32, 7
    width = taps - 1 + drafts
    torch.manual_seed(219)
    base = torch.randn(4, channels+offset, width, device=device, dtype=state_dtype)
    if not channel_first:
        base = base.transpose(1, 2).contiguous().transpose(1, 2)
    generated = base.clone(memory_format=torch.preserve_format)
    starts = torch.tensor([0, 6, 8], device=device, dtype=torch.int32)
    slots = torch.tensor([3, 1], device=device, dtype=torch.int32)
    w = torch.randn(channels, taps, device=device)
    x = torch.randn(8, channels, device=device, dtype=torch.float32)
    dyn = torch.randn(8, 4*taps, device=device) if dynamic else None
    for n in range(1, drafts+2):
        generated.copy_(base)
        accepted = torch.tensor([n, drafts+2-n], device=device, dtype=torch.int32)
        md = SimpleNamespace(num_decodes=2, num_decode_tokens=8, state_indices_tensor_d=slots[:, None],
                             query_start_loc_d=starts, num_accepted_tokens=accepted)
        actual = conv_verify(generated, x, w, offset, md, dyn, 4)
        for start, length, slot, advance in ((0, 6, 3, n-1), (6, 2, 1, drafts+1-n)):
            serial = base[slot:slot+1, :, advance:advance+taps-1].contiguous().clone()
            ref_slot = torch.zeros(1, device=device, dtype=torch.int32)
            for k in range(length):
                row = slice(start+k, start+k+1)
                expected = conv_decode(serial, x[row], w, offset, ref_slot, dyn[row] if dynamic else None, 4)
                torch.testing.assert_close(actual[row], expected, rtol=0, atol=0)
                # Each accepted prefix must recover exactly the serial committed history.
                torch.testing.assert_close(generated[slot, offset:, k:k+taps-1], serial[0, offset:], rtol=0, atol=0)
        torch.testing.assert_close(generated[:, :offset], base[:, :offset], rtol=0, atol=0)
        torch.testing.assert_close(generated[[0, 2]], base[[0, 2]], rtol=0, atol=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--device',default='cuda');device=cli.parse_args().device
    for dtype in (torch.float32, torch.bfloat16):
        for dyn in (False, True):
            for layout in (False, True):
                check(device,dtype,dyn,layout)
    print('All speculative convolution outputs and rollback states equal serial target results.')
