# SPDX-License-Identifier: Apache-2.0
"""LFM3 gated depthwise convolution for mixed-length full CUDA graphs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gated_conv_kernel(BX, GATE, W, BIAS, ST, SLOTS, SEQ, QSL, OUT,
                       stride_gate, stride_w, stride_tap, stride_slot, stride_state, stride_c, stride_s,
                       C: tl.constexpr, K: tl.constexpr, REQS: tl.constexpr, R: tl.constexpr,
                       BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr, SINGLE: tl.constexpr, SILU: tl.constexpr = False,
                       ACCEPTED=None, VERIFY=None, SPEC: tl.constexpr = False):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if REQS == 1:
        req = 0
    else:
        r = tl.arange(0, R)
        ends = tl.load(QSL + r + 1, mask=r < REQS, other=0)
        req = tl.minimum(tl.sum(((r < REQS) & (t >= ends)).to(tl.int32), 0), REQS - 1)
    start = tl.load(QSL + req)
    end = tl.load(QSL + req + 1)
    slot = tl.load(SLOTS + req * stride_slot).to(tl.int64)
    cm = c < C
    if t >= end or slot <= 0:
        tl.store(OUT + t * C + c, 0., mask=cm)
        return
    has_history = tl.load(SEQ + req) > end - start
    state = ST + slot * stride_state + c * stride_c
    if SPEC:
        shift = tl.where(tl.load(VERIFY + req), tl.load(ACCEPTED + req) - 1, 0)
        state += shift * stride_s
    acc = tl.full((BLOCK,), 0., tl.float32)
    if HAS_BIAS:
        acc = tl.load(BIAS + c, mask=cm, other=0.).to(tl.float32)
    for k in tl.static_range(K):
        src = t - (K - 1) + k
        v = tl.load(BX + src * C + c, mask=cm & (src >= start), other=0.).to(tl.float32)
        h = tl.load(state + (t - start + k) * stride_s,
                    mask=cm & (src < start) & has_history, other=0.).to(tl.float32)
        tap = tl.where(src < start, h, v)
        acc += tap * tl.load(W + c * stride_w + k * stride_tap, mask=cm, other=0.).to(tl.float32)
        # Read each channel's old history in ascending order, before its overwrite.
        if SINGLE and k > 0:
            tl.store(state + (k - 1) * stride_s, tap, mask=cm)
    if SILU:
        y = acc * tl.sigmoid(acc)
    else:
        gate = tl.load(GATE + t * stride_gate + c, mask=cm, other=0.)
        y = acc.to(BX.dtype.element_ty).to(tl.float32) * gate.to(tl.float32)
    tl.store(OUT + t * C + c, y, mask=cm)


@triton.jit
def _commit_kernel(BX, ST, SLOTS, SEQ, QSL,
                   stride_slot, stride_state, stride_c, stride_s,
                   C: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    start = tl.load(QSL + req)
    end = tl.load(QSL + req + 1)
    slot = tl.load(SLOTS + req * stride_slot).to(tl.int64)
    if end <= start or slot <= 0:
        return
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = c < C
    state = ST + slot * stride_state + c * stride_c
    has_history = tl.load(SEQ + req) > end - start
    for k in tl.static_range(K - 1):
        src = end - (K - 1) + k
        x = tl.load(BX + src * C + c, mask=cm & (src >= start), other=0.)
        # Any old column used here is strictly newer than k, so later loop
        # iterations cannot observe the store from an earlier iteration.
        prior = tl.load(state + (end - start + k) * stride_s,
                        mask=cm & (src < start) & has_history, other=0.)
        tl.store(state + k * stride_s, tl.where(src < start, prior, x), mask=cm)


def gated_conv(b, c, x, weight, bias, state, metadata):
    bx = (b * x).contiguous()
    tokens, channels = bx.shape
    taps = weight.shape[1]
    if metadata.accepted is not None:
        steps = state.shape[-1] - (taps - 1)
        if steps < 1 or c.stride(1) != 1:
            raise ValueError("LFM verification requires candidate history slots and contiguous gate channels")
        return _spec_conv(bx, c, weight, bias, state, metadata, steps)
    if taps - 1 != state.shape[-1] or c.stride(1) != 1:
        raise ValueError("LFM graph convolution needs one history slot per prior tap and contiguous gate channels")
    slots = metadata.slots()
    requests = slots.shape[0]
    out = torch.empty_like(bx)
    block = 128
    _gated_conv_kernel[(tokens, triton.cdiv(channels, block))](
        bx, c, weight, bias if bias is not None else weight, state, slots, metadata.seq_lens, metadata.offsets, out,
        c.stride(0), *weight.stride(), slots.stride(0), *state.stride(), C=channels, K=taps, REQS=requests,
        R=triton.next_power_of_2(requests), BLOCK=block, HAS_BIAS=bias is not None, SINGLE=tokens == 1)
    if tokens > 1 and taps > 1:
        _commit_kernel[(requests, triton.cdiv(channels, block))](
            bx, state, slots, metadata.seq_lens, metadata.offsets, slots.stride(0), *state.stride(),
            C=channels, K=taps, BLOCK=block)
    return out


def silu_conv(x, weight, bias, state, metadata):
    """Causal depthwise SiLU used on Qwen's packed Q/K/V input."""
    x = x.contiguous()
    tokens, channels = x.shape
    taps = weight.shape[1]
    if taps - 1 != state.shape[-1]:
        raise ValueError('Qwen plain convolution needs one cache column per prior tap')
    slots = metadata.slots()
    requests = slots.shape[0]
    out = torch.empty_like(x)
    block = 128
    _gated_conv_kernel[(tokens, triton.cdiv(channels, block))](
        x, x, weight, bias if bias is not None else weight, state, slots, metadata.seq_lens, metadata.offsets, out,
        x.stride(0), *weight.stride(), slots.stride(0), *state.stride(), C=channels, K=taps, REQS=requests,
        R=triton.next_power_of_2(requests), BLOCK=block, HAS_BIAS=bias is not None, SINGLE=tokens == 1, SILU=True)
    if tokens > 1 and taps > 1:
        _commit_kernel[(requests, triton.cdiv(channels, block))](
            x, state, slots, metadata.seq_lens, metadata.offsets, slots.stride(0), *state.stride(),
            C=channels, K=taps, BLOCK=block)
    return out


@triton.jit
def _spec_commit_kernel(X, ST, SLOTS, SEQ, QSL, ACCEPTED, VERIFY,
                        stride_slot, stride_state, stride_c, stride_s,
                        C: tl.constexpr, K: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin, end = tl.load(QSL + row), tl.load(QSL + row + 1)
    length = end - begin
    slot = tl.load(SLOTS + row * stride_slot).to(tl.int64)
    if length <= 0 or slot <= 0:
        return
    verify = tl.load(VERIFY + row)
    shift = tl.where(verify, tl.load(ACCEPTED + row) - 1, 0)
    history = tl.load(SEQ + row) > length
    state = ST + slot * stride_state + c * stride_c
    # A verifier writes the rolled history plus each speculative input. The next
    # visit selects its accepted prefix with the GPU-resident accepted count.
    for col in tl.static_range(K - 1 + STEPS):
        relative = tl.where(verify, col - (K - 2), length - (K - 1) + col)
        live = col < tl.where(verify, K - 2 + length, K - 1)
        values = tl.load(X + (begin + relative) * C + c,
                         mask=(c < C) & live & (relative >= 0) & (relative < length), other=0.)
        prior = tl.load(state + (shift + relative + K - 1) * stride_s,
                        mask=(c < C) & live & history & (relative < 0), other=0.)
        tl.store(state + col * stride_s, tl.where(relative < 0, prior, values), mask=(c < C) & live)


def silu_spec_conv(x, weight, bias, state, metadata, steps):
    return _spec_conv(x.contiguous(), None, weight, bias, state, metadata, steps)


def _spec_conv(x, gate, weight, bias, state, metadata, steps):
    tokens, channels = x.shape
    taps = weight.shape[1]
    if state.shape[-1] < taps - 1 + steps:
        raise ValueError('Verification needs room for all candidate convolution states')
    slots = metadata.slots()
    requests = len(slots)
    out = torch.empty_like(x)
    block = 128
    _gated_conv_kernel[(tokens, triton.cdiv(channels, block))](
        x, x if gate is None else gate, weight, bias if bias is not None else weight, state, slots, metadata.seq_lens, metadata.offsets, out,
        (x if gate is None else gate).stride(0), *weight.stride(), slots.stride(0), *state.stride(), C=channels, K=taps, REQS=requests,
        R=triton.next_power_of_2(requests), BLOCK=block, HAS_BIAS=bias is not None, SINGLE=False, SILU=gate is None,
        ACCEPTED=metadata.accepted, VERIFY=metadata.verify, SPEC=True)
    _spec_commit_kernel[(requests, triton.cdiv(channels, block))](
        x, state, slots, metadata.seq_lens, metadata.offsets, metadata.accepted, metadata.verify,
        slots.stride(0), *state.stride(), C=channels, K=taps, STEPS=steps, BLOCK=block)
    return out
