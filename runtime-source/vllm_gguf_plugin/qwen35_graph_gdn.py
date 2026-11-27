# SPDX-License-Identifier: Apache-2.0
"""Graph-safe Qwen3.5 Gated DeltaNet for packed, dynamically sized text requests."""
import os

import torch
import triton
import triton.language as tl


def chunk_plan(offsets, tokens, chunk_size):
    """Align each request to complete FLA chunks with neutral padding and one dummy sequence."""
    lengths = offsets.diff()
    requests = len(lengths)
    capacity = (tokens + chunk_size - 1) // chunk_size + requests - 1
    chunks = (lengths + chunk_size - 1) // chunk_size
    boundaries = torch.cat((offsets.new_zeros(1), chunks.cumsum(0, dtype=torch.int32),
                            offsets.new_full((1,), capacity)))
    packed_offsets = boundaries * chunk_size
    chunk_ids = torch.arange(capacity, device=offsets.device, dtype=torch.int32)
    owners = torch.searchsorted(boundaries[1:].contiguous(), chunk_ids, right=True)
    chunk_map = torch.stack((owners.to(torch.int32), chunk_ids - boundaries[owners]), dim=1)
    positions = torch.arange(capacity * chunk_size, device=offsets.device)
    owner = owners[positions // chunk_size]
    local = positions - packed_offsets[owner]
    row = owner.clamp_max(requests - 1)
    valid = (owner < requests) & (local < lengths[row])
    source = (offsets[row] + local).clamp(0, tokens - 1)
    real = torch.arange(tokens, device=offsets.device)
    real_owner = torch.searchsorted(offsets[1:].contiguous(), real.to(offsets.dtype), right=True).clamp_max(requests-1)
    unpack = (packed_offsets[real_owner] + real - offsets[real_owner]).clamp(0, len(positions)-1)
    return source, valid, unpack, packed_offsets, chunk_map, boundaries


@triton.jit
def _state_kernel(CACHE, FINAL, SLOTS, OFF, SEQ, stride_state, stride_slot,
                  ELEMENTS: tl.constexpr, BLOCK: tl.constexpr, RESET: tl.constexpr, VERIFY=None, EXCLUDE_VERIFY: tl.constexpr=False):
    row = tl.program_id(0)
    start, end = tl.load(OFF + row), tl.load(OFF + row + 1)
    slot = tl.load(SLOTS + row * stride_slot).to(tl.int64)
    if end <= start or slot <= 0:
        return
    if EXCLUDE_VERIFY and tl.load(VERIFY + row):
        return
    if RESET and tl.load(SEQ + row) > end - start:
        return
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if RESET:
        new = tl.full((BLOCK,), 0, tl.float32)
    else:
        new = tl.load(FINAL + row * ELEMENTS + i, mask=i < ELEMENTS, other=0.)
    tl.store(CACHE + slot * stride_state + i, new, mask=i < ELEMENTS)


def state_update(cache, final, slots, md, reset):
    elements = cache[0].numel()
    if not cache[0].is_contiguous():
        raise ValueError('Qwen GDN expects contiguous per-slot recurrent state')
    _state_kernel[(slots.shape[0], triton.cdiv(elements, 512))](
        cache, final if final is not None else cache, slots, md.offsets, md.seq_lens, cache.stride(0), slots.stride(0),
        ELEMENTS=elements, BLOCK=512, RESET=reset, VERIFY=md.verify, EXCLUDE_VERIFY=md.verify is not None)


def ragged_chunk(layer, mixed, a, b, cache, md, slots):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from .inkling.conv_graph import forward_plan_cache

    tokens = mixed.shape[0]
    key = ('qwen-gdn', md.offsets.data_ptr(), tokens, len(slots))
    plans = forward_plan_cache()
    if key not in plans:
        plans[key] = chunk_plan(md.offsets, tokens, FLA_CHUNK_SIZE)
    source, valid, unpack, cu, indices, chunk_offsets = plans[key]
    q, k, v, decay, beta = fused_post_conv_prep(
        conv_output=mixed, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads//layer.tp_size, head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim, apply_l2norm=True, output_g_exp=False)

    def pack(value):
        mask = valid.reshape((-1,) + (1,) * (value.ndim - 1))
        return torch.where(mask, value[source], 0).unsqueeze(0)

    # The dummy sequence consumes remaining capture capacity. Setting its
    # K/V and log-decay to zero leaves all live recurrent states unchanged.
    history = (md.seq_lens > md.offsets.diff()) & (slots > 0)
    state = torch.where(history[:, None, None, None], cache[slots.clamp_min(0).long()], 0)
    state = torch.cat((state, torch.zeros_like(state[:1])), 0)
    out, final = chunk_gated_delta_rule(pack(q), pack(k), pack(v), pack(decay), pack(beta), initial_state=state,
                                      output_final_state=True, cu_seqlens=cu, chunk_indices=indices,
                                      chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=False)
    state_update(cache, final.contiguous(), slots, md, False)
    return out[0, unpack]


def verification_plan(md, tokens, steps):
    """Compact live verifier positions into a bounded, fixed-capacity graph buffer."""
    lengths = md.offsets.diff()
    verify_lengths = torch.where(md.verify, lengths, 0)
    starts = torch.cat((md.offsets.new_zeros(1), verify_lengths.cumsum(0, dtype=torch.int32)))
    capacity = len(lengths) * (steps + 1)
    compact = torch.arange(capacity, device=md.offsets.device)
    owner = torch.searchsorted(starts[1:].contiguous(), compact.to(starts.dtype), right=True).clamp_max(len(lengths)-1)
    source = (md.offsets[owner] + compact - starts[owner]).clamp(0,tokens-1)
    positions = torch.arange(tokens, device=md.offsets.device)
    req = torch.searchsorted(md.offsets[1:].contiguous(), positions.to(starts.dtype), right=True).clamp_max(len(lengths)-1)
    restore = (starts[req] + positions - md.offsets[req]).clamp(0,capacity-1)
    mask = md.verify[req] & (positions < md.offsets[-1])
    return starts, source, restore, mask


def run_recurrent(layer, projected, a, b, cache, md, slots, *, verify_only):
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    q, k, v = layer.rearrange_mixed_qkv(projected)
    columns = projected.shape[0] if md.verify is None else max(projected.shape[0], layer.num_spec+1)
    if md.verify is None:
        indices = slots[:, None].expand(-1, columns).contiguous()
        accepted = None
    else:
        candidate = torch.arange(columns, device=slots.device).clamp_max(layer.num_spec)
        indices = torch.where(md.verify[:,None], md.block_table[:,candidate], slots[:,None])
        accepted = torch.where(md.verify, md.accepted, 1)
    offsets = md.offsets
    if verify_only:
        offsets, source, restore, mask = verification_plan(md, projected.shape[0], layer.num_spec)
        q, k, v = (z[:,source] for z in (q,k,v))
        a, b = a[source], b[source]
    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, q=q, k=k, v=v, initial_state=cache,
        inplace_final_state=True, cu_seqlens=offsets, ssm_state_indices=indices,
        num_accepted_tokens=accepted, use_qk_l2norm_in_kernel=True)
    if verify_only:
        return out[0,restore], mask
    return out[0]


def graph_core(layer, mixed, a, b, md):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_recurrent_gated_delta_rule_packed_decode, is_conv_state_dim_first)
    from .lfm3_graph_conv import silu_conv, silu_spec_conv
    conv, cache = layer.kv_cache
    conv = conv if is_conv_state_dim_first() else conv.transpose(-1,-2)
    weight = layer.conv1d.weight[:,0,:]
    if md.verify is None:
        projected = silu_conv(mixed, weight, layer.conv1d.bias, conv, md)
    else:
        if md.block_table.shape[1] < layer.num_spec + 1:
            raise ValueError('Qwen graph verification needs a distinct recurrent-state slot for every candidate')
        projected = silu_spec_conv(mixed, weight, layer.conv1d.bias, conv, md, layer.num_spec)
    tokens = mixed.shape[0]
    slots = md.slots()
    if tokens == 1 and md.verify is None:
        state_update(cache, None, slots, md, True)
        out = mixed.new_empty((tokens, 1, layer.num_v_heads//layer.tp_size, layer.head_v_dim))
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=projected, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-.5, initial_state=cache, out=out, ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True)
        return out[:,0]
    if tokens <= max(16, len(slots)*2):
        state_update(cache, None, slots, md, True)
        return run_recurrent(layer, projected, a, b, cache, md, slots, verify_only=False)
    plain = ragged_chunk(layer, projected, a, b, cache, md, slots)
    if md.verify is not None:
        verified, mask = run_recurrent(layer, projected, a, b, cache, md, slots, verify_only=True)
        plain = torch.where(mask[:,None,None], verified, plain)
    return plain


def install():
    if os.environ.get('KAILLM_QWEN35_FULL_GRAPHS') != '1':
        return
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from .inkling.conv_graph import ConvGraphMetadata, conv_graph_backend
    cls = QwenGatedDeltaNetAttention
    if getattr(cls, '_kai_plain_graph_gdn', False):
        return
    original_backend, original_forward = cls.get_attn_backend, cls.forward_cuda

    def enabled(layer):
        return (layer.model_config.hf_text_config.model_type == 'qwen3_5_text'
                and not layer.gqa_interleaved_layout)

    def backend(self):
        if enabled(self):
            if self.cache_config.mamba_cache_mode != 'none' or self.activation not in ('silu', 'swish'):
                raise ValueError('Qwen full GDN graphs require linear SiLU layers and full-sequence state caching')
            return conv_graph_backend('QWEN35_GRAPH_GDN')
        return original_backend(self)

    def forward(self, hidden_states):
        if not enabled(self):
            return original_forward(self, hidden_states)
        raw = get_forward_context().attn_metadata
        metadata = raw.get(self.prefix) if isinstance(raw, dict) else None
        if not raw:  # initial memory/kernel profiling
            return original_forward(self, hidden_states)
        if not isinstance(metadata, ConvGraphMetadata):
            raise RuntimeError('Qwen GDN is missing its full graph metadata')
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        value_dim = self.value_dim//self.tp_size
        qkv, z = qkvz.split([(self.key_dim*2+self.value_dim)//self.tp_size, value_dim], dim=-1)
        b, a = self.split_ba(ba)
        output = graph_core(self, qkv, a.contiguous(), b.contiguous(), metadata)
        return self._output_projection(output, z.reshape(z.shape[0],-1,self.head_v_dim))

    cls.get_attn_backend, cls.forward_cuda = backend, forward
    cls._kai_plain_graph_gdn = True
