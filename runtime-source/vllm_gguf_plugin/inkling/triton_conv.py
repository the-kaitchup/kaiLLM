# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 short convolution of a decode-only batch (one token per request) in one Triton kernel: reads the token
and its state's n_conv - 1 history columns, writes `(x + conv(x) [+ dynamic taps]) / sqrt((2 if dynamic else 1)
n_conv + 1)` (ref.convolve) and shifts the history in place.  Rows whose state index is the null block (CUDA-graph
padding) compute garbage into it, as vLLM's own conv kernels do.  No host synchronisation (CUDA-graph safe).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_decode_kernel(X, W, DYN, ST, SIDX, SEQ, QSL, OUT, C, offset, scale,
                        stride_x, stride_dyn, stride_sb, stride_sc, stride_sk, stride_sidx, stride_out,
                        NC: tl.constexpr, HAS_DYN: tl.constexpr, LIVE_METADATA: tl.constexpr, CH: tl.constexpr, BLOCK_C: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    sidx = tl.load(SIDX + t * stride_sidx).to(tl.int64)
    live = cm
    init = cm
    if LIVE_METADATA:
        length = tl.load(QSL + t + 1) - tl.load(QSL + t)
        live = cm & (length > 0) & (sidx > 0)
        init = live & (tl.load(SEQ + t) > length)
    x = tl.load(X + t * stride_x + c, mask=cm, other=0.0).to(tl.float32)
    st = ST + sidx * stride_sb + (offset + c) * stride_sc
    out = x * tl.load(W + c * NC + (NC - 1), mask=cm, other=0.0)            # tap n_conv - 1: the current token
    if HAS_DYN:
        head = c // CH
        out = tl.fma(x, tl.load(DYN + t * stride_dyn + head * NC, mask=cm, other=0.0).to(tl.float32), out)
    prev = x
    for j in tl.static_range(NC - 1):
        # history column NC - 2 - j holds x[t - 1 - j]
        h = tl.load(st + (NC - 2 - j) * stride_sk, mask=init, other=0.0).to(tl.float32)
        out = tl.fma(h, tl.load(W + c * NC + (NC - 2 - j), mask=cm, other=0.0), out)
        if HAS_DYN:
            out = tl.fma(h, tl.load(DYN + t * stride_dyn + head * NC + j + 1, mask=cm, other=0.0).to(tl.float32), out)
        # shift: column NC - 2 - j takes the newer value (the token for the last column)
        tl.store(st + (NC - 2 - j) * stride_sk, prev.to(ST.dtype.element_ty), mask=live)
        prev = h
    tl.store(OUT + t * stride_out + c, ((x + out) * scale).to(OUT.dtype.element_ty), mask=cm)


def conv_decode(states: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offset: int, sidx: torch.Tensor,
                dynamic: torch.Tensor | None = None, n_heads: int = 1, out_dtype=None, *, seq_lens=None, offsets=None) -> torch.Tensor:
    """states [slots, channels, n_conv - 1] (oldest first, any strides), x [T, C] (T rows = T requests; bf16 / f16 /
    f32, widened in the kernel), weight [C, n_conv] float32, sidx [T] state index of each row, dynamic
    [T, n_heads * n_conv] (any float dtype) -> [T, C] out_dtype (default x.dtype)."""
    T, C = x.shape
    n_conv = weight.shape[-1]
    out = torch.empty_like(x, dtype=out_dtype or x.dtype)
    w = weight if (weight.dtype == torch.float32 and weight.is_contiguous()) else weight.float().contiguous()
    dyn = dynamic
    if dyn is not None and dyn.stride(-1) != 1:
        dyn = dyn.contiguous()
    xin = x if x.stride(-1) == 1 else x.contiguous()
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * n_conv + 1)
    BLOCK_C = 256
    _conv_decode_kernel[(T, triton.cdiv(C, BLOCK_C))](
        xin, w, dyn if dyn is not None else w, states, sidx, seq_lens if seq_lens is not None else sidx,
        offsets if offsets is not None else sidx, out, C, int(offset), scale,
        xin.stride(0), dyn.stride(0) if dyn is not None else 0, states.stride(0), states.stride(1), states.stride(2),
        sidx.stride(0), out.stride(0),
        NC=n_conv, HAS_DYN=dynamic is not None, LIVE_METADATA=seq_lens is not None, CH=max(C // max(n_heads, 1), 1), BLOCK_C=BLOCK_C, num_warps=4)
    return out


@triton.jit
def _store_history_kernel(ST, SLOTS, LENS, NEW, offset, stride_sb, stride_sc, stride_sk,
                          stride_slot, C: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    slot = tl.load(SLOTS + req * stride_slot).to(tl.int64)
    if slot <= 0 or tl.load(LENS + req) <= 0:
        return
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for k in tl.static_range(H):
        value = tl.load(NEW + (req * H + k) * C + c, mask=c < C, other=0)
        tl.store(ST + slot * stride_sb + (offset + c) * stride_sc + k * stride_sk, value, mask=c < C)


def store_conv_history(states, offset, slots, lengths, history):
    """Commit only live requests; padded rows may alias the null state."""
    requests, taps, channels = history.shape
    _store_history_kernel[(requests, triton.cdiv(channels, 128))](
        states, slots, lengths, history.contiguous(), offset, *states.stride(), slots.stride(0),
        C=channels, H=taps, BLOCK=128)


@triton.jit
def _conv_verify_kernel(X, W, DYN, ST, SIDX, STARTS, ACCEPTED, VERIFY, SEQ, OUT, C, offset, scale,
                        stride_x, stride_dyn, stride_sb, stride_sc, stride_sk, stride_sidx, stride_out,
                        NC: tl.constexpr, WIDTH: tl.constexpr, HAS_DYN: tl.constexpr, CH: tl.constexpr,
                        HAS_STARTS: tl.constexpr, HAS_ACCEPTED: tl.constexpr, LIVE: tl.constexpr, BLOCK_C: tl.constexpr):
    request = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c < C
    slot = tl.load(SIDX + request * stride_sidx).to(tl.int64)
    state = ST + slot * stride_sb + (offset + c) * stride_sc
    if HAS_STARTS:
        start = tl.load(STARTS + request)
        length = tl.load(STARTS + request + 1) - start
    else:
        start = request
        length = 1
    if LIVE:
        if slot <= 0 or length <= 0 or not tl.load(VERIFY + request):
            return
        initial = tl.load(SEQ + request) > length
    else:
        initial = True
    history = 0
    if HAS_ACCEPTED:
        history = tl.minimum(WIDTH - NC + 1, tl.maximum(0, tl.load(ACCEPTED + request) - 1))
    for pos in range(length):
        t = start + pos
        x = tl.load(X + t * stride_x + c, mask=mask, other=0.0).to(tl.float32)
        out = x * tl.load(W + c * NC + NC - 1, mask=mask, other=0.0)
        if HAS_DYN:
            out = tl.fma(x, tl.load(DYN + t * stride_dyn + (c // CH) * NC, mask=mask, other=0.0).to(tl.float32), out)
        # Match the single-token target's FMA order and state-dtype rounding.
        for lag in tl.static_range(1, NC):
            if pos >= lag:
                prior = tl.load(X + (t - lag) * stride_x + c, mask=mask, other=0.0).to(ST.dtype.element_ty)
            else:
                prior = tl.load(state + (history + NC - 1 + pos - lag) * stride_sk, mask=mask & initial, other=0.0)
            h = prior.to(tl.float32)
            out = tl.fma(h, tl.load(W + c * NC + NC - 1 - lag, mask=mask, other=0.0), out)
            if HAS_DYN:
                out = tl.fma(h, tl.load(DYN + t * stride_dyn + (c // CH) * NC + lag, mask=mask, other=0.0).to(tl.float32), out)
        tl.store(OUT + t * stride_out + c, (x + out) * scale, mask=mask)
    # The anchor always commits. Preserve every following candidate so the next
    # scheduler step can start after the accepted prefix. Ascending columns keep
    # the previous history readable while it shifts left in place.
    for col in tl.static_range(WIDTH):
        if col < NC - 2:
            value = tl.load(state + (history + col + 1) * stride_sk, mask=mask & initial, other=0.0)
        else:
            position = col - (NC - 2)
            value = tl.load(X + (start + position) * stride_x + c, mask=mask & (position < length), other=0.0)
        tl.store(state + col * stride_sk, value, mask=mask)


def conv_verify(states, x, weight, offset, metadata, dynamic=None, n_heads=1, out_dtype=None):
    channels, taps = weight.shape
    requests = int(metadata.num_decodes)
    indices = metadata.state_indices_tensor_d.reshape(requests, -1)[:, 0]
    starts = metadata.query_start_loc_d
    accepted = metadata.num_accepted_tokens
    xin = x if x.stride(-1) == 1 else x.contiguous()
    if dynamic is not None and dynamic.stride(-1) != 1:
        dynamic = dynamic.contiguous()
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    if int(metadata.num_decode_tokens) < x.shape[0]:
        out[int(metadata.num_decode_tokens):].zero_()
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * taps + 1)
    _conv_verify_kernel[(requests, triton.cdiv(channels, 256))](
        xin, weight.float().contiguous(), dynamic if dynamic is not None else xin, states, indices,
        starts if starts is not None else indices, accepted if accepted is not None else indices, indices, indices,
        out, channels, int(offset), scale, xin.stride(0), dynamic.stride(0) if dynamic is not None else 0,
        states.stride(0), states.stride(1), states.stride(2), indices.stride(0), out.stride(0),
        NC=taps, WIDTH=states.shape[-1], HAS_DYN=dynamic is not None, CH=channels // n_heads,
        HAS_STARTS=starts is not None, HAS_ACCEPTED=accepted is not None, LIVE=False, BLOCK_C=256, num_warps=4)
    return out


def conv_verify_graph(states, x, weight, offset, md, out, dynamic, heads):
    """Overwrite only verify rows, retaining the accepted prefix of the previous step."""
    channels, taps = weight.shape
    slots = md.slots()
    x = x if x.stride(-1) == 1 else x.contiguous()
    if dynamic is not None and dynamic.stride(-1) != 1:
        dynamic = dynamic.contiguous()
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * taps + 1)
    _conv_verify_kernel[(slots.shape[0], triton.cdiv(channels, 256))](
        x, weight.float().contiguous(), dynamic if dynamic is not None else x, states, slots,
        md.offsets, md.accepted, md.verify, md.seq_lens, out, channels, offset, scale,
        x.stride(0), dynamic.stride(0) if dynamic is not None else 0,
        *states.stride(), slots.stride(0), out.stride(0), NC=taps, WIDTH=states.shape[-1],
        HAS_DYN=dynamic is not None, CH=channels // heads, HAS_STARTS=True, HAS_ACCEPTED=True,
        LIVE=True, BLOCK_C=256, num_warps=4)
