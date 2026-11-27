# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from torch.nn import functional as F

from vllm_gguf_plugin.inkling.dspark import ContextMix, InklingMarkovHead
from vllm_gguf_plugin.inkling.dspark_attention import local_block_attention


class InklingDSparkHeadTest(unittest.TestCase):
    def test_block_attention_has_one_window_per_anchor_and_isolates_requests(self):
        torch.manual_seed(61)
        heads, group, dim, width, window = 2, 3, 4, 6, 5
        cache = torch.randn(8, heads, 4, 2 * dim)
        table = torch.tensor([[3, 6, 2, 0], [1, 4, 7, 5]])
        anchors = [0, 9]
        cache[table[0, 0]].fill_(float('nan'))
        pos = torch.cat([torch.arange(a, a + width) for a in anchors])
        q = torch.randn(len(pos), heads * group, dim)
        k, v = (torch.randn(len(pos), heads, dim) for _ in range(2))
        out = local_block_attention(q, k, v, pos, table, cache, window, width)
        for batch, anchor in enumerate(anchors):
            block = slice(batch * width, (batch + 1) * width)
            past = torch.arange(max(0, anchor - window), anchor)
            keys, values = cache.transpose(1, 2)[table[batch, past // 4], past % 4].chunk(2, -1)
            keys = torch.cat((keys, k[block])).repeat_interleave(group, dim=1)
            values = torch.cat((values, v[block])).repeat_interleave(group, dim=1)
            scores = torch.einsum('qhd,khd->hqk', q[block], keys) * dim ** -.5
            ref = torch.einsum('hqk,khd->qhd', scores.softmax(-1), values)
            torch.testing.assert_close(out[block], ref.flatten(1))

    def test_folded_head_matches_native_projection_over_five_draws(self):
        torch.manual_seed(51)
        h, rank, vocab, batch = 12, 5, 29, 3
        embedding = torch.randn(vocab, h)
        project = lambda x: F.linear(x, embedding)
        native = InklingMarkovHead(h, rank, vocab, folded=False, softcap=30.0)
        folded = InklingMarkovHead(h, rank, vocab, folded=True, softcap=30.0)
        with torch.no_grad():
            folded.token.load_state_dict(native.token.state_dict())
            folded.joint.load_state_dict(native.joint.state_dict())
            folded.output.weight.copy_(embedding @ native.output.weight)
            folded.output.bias.copy_(embedding @ native.output.bias)
        taps = torch.randn(batch, 4, h)
        mix = ContextMix(4, h, 1e-6)
        with torch.no_grad():
            mix.mix.copy_(torch.randn(4).softmax(0))
            mix.norm.copy_(torch.randn(h))
        expected = sum(mix.mix[i] * taps[:, i] for i in range(4))
        expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6) * mix.norm
        ctx = mix(taps)
        torch.testing.assert_close(ctx, expected)
        prev = torch.arange(batch)
        state = torch.zeros(batch, rank)
        for _ in range(5):
            hidden = torch.randn(batch, h)
            gate, cand, out = native.joint(torch.cat((state, native.token(prev), hidden, ctx), -1)).chunk(3, -1)
            reference_state = gate.sigmoid() * state + (1 - gate.sigmoid()) * cand.tanh()
            reference_logits = 30 * (project(hidden + native.output(out.tanh())) / 30).tanh()
            for head in (folded, native):
                actual_state, delta = head.step(prev, state, hidden, ctx)
                torch.testing.assert_close(actual_state, reference_state)
                torch.testing.assert_close(head.logits(hidden, delta, project), reference_logits)
            state, prev = reference_state, reference_logits.argmax(-1)


if __name__ == '__main__':
    unittest.main()
