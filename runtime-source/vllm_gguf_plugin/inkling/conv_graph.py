# SPDX-License-Identifier: Apache-2.0
"""Short-convolution metadata for full, mixed-shape Inkling CUDA graphs."""

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass
class ConvGraphMetadata:
    offsets: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    block_size: int | None
    accepted: torch.Tensor | None = None
    verify: torch.Tensor | None = None

    def slots(self):
        if self.block_size is None:
            return self.block_table[:, 0]
        column = ((self.seq_lens - 1).clamp_min(0) // self.block_size).long()
        return self.block_table.gather(1, column[:, None]).flatten()


@lru_cache(maxsize=2)
def conv_graph_backend(name="INKLING_SHORT_CONV"):
    from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
    from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend, ShortConvAttentionMetadataBuilder

    # The hybrid runner delivers accepted counts to recognised Mamba builder types.
    # Address-stable buffers are refreshed on every step, including padded requests.
    class Builder(ShortConvAttentionMetadataBuilder):
        _cudagraph_support = AttentionCGSupport.ALWAYS
        supports_update_block_table = False
        needs_causal_conv1d_metadata = False
        reorder_batch_threshold = None

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            AttentionMetadataBuilder.__init__(self, kv_cache_spec, layer_names, vllm_config, device)
            self.spec_tokens = vllm_config.num_speculative_tokens
            if self.spec_tokens:
                requests = vllm_config.scheduler_config.max_num_seqs
                self.accepted = torch.ones(requests, dtype=torch.int32, device=device)
                self.verify = torch.zeros(requests, dtype=torch.bool, device=device)
            mode = vllm_config.cache_config.mamba_cache_mode
            if mode not in ("none", "align"):
                raise ValueError(f"Unsupported Inkling convolution cache mode {mode}")
            self.block_size = kv_cache_spec.block_size if mode == "align" else None

        def _metadata(self, m):
            n = m.num_reqs
            return ConvGraphMetadata(m.query_start_loc, m.seq_lens, m.block_table_tensor, self.block_size,
                                     self.accepted[:n] if self.spec_tokens else None,
                                     self.verify[:n] if self.spec_tokens else None)

        def build_for_cudagraph_capture(self, common_attn_metadata):
            if self.spec_tokens:
                self.accepted.fill_(1)
                self.verify.zero_()
            return self._metadata(common_attn_metadata)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False, *,
                  num_accepted_tokens=None, num_decode_draft_tokens_cpu=None):
            m = common_attn_metadata
            if self.spec_tokens:
                if num_accepted_tokens is None or num_decode_draft_tokens_cpu is None:
                    raise ValueError("Inkling speculative convolution requires live accepted counts and request kinds")
                n = m.num_reqs
                self.accepted[:n].copy_(num_accepted_tokens[:n])
                self.verify[:n].copy_(num_decode_draft_tokens_cpu[:n] >= 0, non_blocking=True)
            return self._metadata(m)

    class Backend(ShortConvAttentionBackend):
        @staticmethod
        def get_name():
            return name

        @staticmethod
        def get_builder_cls():
            return Builder

    return Backend


def forward_plan_cache():
    """One cache per forward. The runner reuses metadata across warmup and capture."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return {}
    context = get_forward_context()
    cache = getattr(context, "_inkling_plans", None)
    if cache is None:
        context._inkling_plans = cache = {}
    return cache


def gather_plan(md, tokens, taps):
    """Return fixed-shape gathers for real tokens and a mask for padded tokens."""
    key = ("convolution", id(md), tokens, taps)
    cache = forward_plan_cache() if md.offsets.is_cuda else {}
    if key in cache:
        return cache[key]
    starts, ends = md.offsets[:-1], md.offsets[1:]
    lens = ends - starts
    requests = starts.shape[0]
    history = taps - 1
    indices = torch.arange(tokens, device=starts.device, dtype=starts.dtype)
    owners = torch.searchsorted(ends.contiguous(), indices, right=True).clamp_max(requests - 1)
    local = indices - starts[owners]
    prior = torch.arange(taps, device=starts.device) - history
    slot_rows = torch.arange(requests, device=starts.device)[:, None] * history
    window_rows = torch.where(local[:, None] + prior >= 0, requests * history + indices[:, None] + prior,
                              owners[:, None] * history + local[:, None] + prior + history)
    end_local = lens[:, None] + prior[:-1]
    history_rows = torch.where(end_local >= 0, requests * history + starts[:, None] + end_local,
                               slot_rows + history + end_local)
    slots = md.slots()
    valid = indices < ends[-1]
    starts_with_history = (md.seq_lens > lens) & (slots > 0)
    plan = slots, starts_with_history, lens, valid, window_rows, history_rows
    cache[key] = plan
    return plan


def conv_mixed(state, x, weight, offset, md, dynamic, heads, out_dtype, *, prefill_only=False):
    """Execute convolution using only device data for request boundaries and state."""
    import math
    from .triton_conv import store_conv_history

    tokens, channels = x.shape
    taps = state.n_conv
    history = taps - 1
    slots, init, lengths, valid, win, new = gather_plan(md, tokens, taps)
    states = state.states()
    prior = states[slots.long(), offset:offset + channels, :history].transpose(1, 2).float()
    prior = torch.where(init[:, None, None], prior, 0.)
    values = x.float()
    buffer = torch.cat((prior.reshape(-1, channels), values), dim=0)
    window = buffer[win]
    output = (window * weight.float().t()[None]).sum(1)
    if dynamic is not None:
        lagged = window.flip(1).reshape(tokens, taps, heads, channels // heads)
        dynamic_taps = dynamic.float().reshape(tokens, heads, taps).transpose(1, 2)
        output = output + (lagged * dynamic_taps[..., None]).sum(1).reshape(tokens, channels)
    scale = 1. / math.sqrt((2 if dynamic is not None else 1) * taps + 1)
    result = torch.where(valid[:, None], (values + output) * scale, 0.).to(out_dtype)
    if history:
        if prefill_only:
            lengths = torch.where(md.verify, 0, lengths)
        store_conv_history(states, offset, slots, lengths, buffer[new])
    return result


def conv_spec_mixed(state, x, weight, offset, md, dynamic, heads, out_dtype):
    """Prefill commits its entire chunk. Verification retains each candidate state."""
    from .triton_conv import conv_verify_graph

    out = conv_mixed(state, x, weight, offset, md, dynamic, heads, out_dtype, prefill_only=True)
    conv_verify_graph(state.states(), x, weight, offset, md, out, dynamic, heads)
    return out
