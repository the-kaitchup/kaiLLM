# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 block-drafter feature fusion and recurrent token head."""

import torch
from torch import nn
from torch.nn import functional as F


class ContextMix(nn.Module):
    """A separate trained mixture of target features for each drafter layer and the Markov head."""

    def __init__(self, taps, hidden_size, eps):
        super().__init__()
        self.mix = nn.Parameter(torch.empty(taps))
        self.norm = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps

    def forward(self, features):
        # The converted GGUF stores the softmax of the learned mixture weights.
        x = torch.einsum('...kh,k->...h', features, self.mix.to(features.dtype))
        return F.rms_norm(x.float(), (x.shape[-1],), self.norm.float(), self.eps).to(features.dtype)


class InklingMarkovHead(nn.Module):
    def __init__(self, hidden_size, rank, vocab_size, *, folded, softcap):
        super().__init__()
        self.token = nn.Embedding(vocab_size, rank)
        self.joint = nn.Linear(2 * (hidden_size + rank), 3 * rank)
        self.output = nn.Linear(rank, vocab_size if folded else hidden_size)
        self.folded = folded
        self.softcap = softcap

    def step(self, previous_id, state, draft_hidden, anchor_context):
        """Advance the recurrent head using the token actually drawn at the preceding position."""
        token = self.token(previous_id)
        joined = torch.cat((state, token, draft_hidden, anchor_context), dim=-1)
        gate, candidate, readout = self.joint(joined.to(self.joint.weight.dtype)).float().chunk(3, dim=-1)
        candidate = candidate.tanh()
        state = candidate + gate.sigmoid() * (state.float() - candidate)
        adjustment = self.output(readout.tanh().to(self.output.weight.dtype))
        return state, adjustment

    def logits(self, hidden, adjustment, project):
        """The folded GGUF adds a vocabulary bias. Native weights add a hidden-space correction."""
        if self.folded:
            logits = project(hidden)[..., :adjustment.shape[-1]].float() + adjustment.float()
        else:
            logits = project((hidden.float() + adjustment.float()).to(hidden.dtype)).float()
        if self.softcap:
            logits = self.softcap * (logits / self.softcap).tanh()
        return logits
