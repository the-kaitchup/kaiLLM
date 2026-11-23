# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 loop controller of a few tokens (decode batches) in two Triton kernels, ref.control:

    g   = [rms(cur); rms(message)] . down^T                  down [r, 2D]: the two RMS scales applied after the dots
    out = cur - tanh(g . forget^T) cur + tanh(g . gain^T) message      gain / forget [D, r]

float32 throughout (as the reference); one program per (token, CTRL_BLOCK_R rank rows), then per (token,
CTRL_BLOCK_N channels).  Grids come from tensor shapes, no host synchronisation (CUDA-graph safe).  The torch path
(GEMMs) takes larger token counts (glue.CONTROL_MAX_T).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

CTRL_BLOCK_R = 4
CTRL_BLOCK_D = 256
CTRL_BLOCK_N = 64


@triton.jit
def _ctrl_down_kernel(CUR, MSG, DOWN, G, D, stride_c, stride_m, stride_down, stride_g, eps,
                      BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    acc_c = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc_m = tl.zeros([BLOCK_R], dtype=tl.float32)
    ss_c = tl.zeros([BLOCK_D], dtype=tl.float32)
    ss_m = tl.zeros([BLOCK_D], dtype=tl.float32)
    w_rows = DOWN + r[:, None] * stride_down
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        c = tl.load(CUR + t * stride_c + d, mask=dm, other=0.0)
        m = tl.load(MSG + t * stride_m + d, mask=dm, other=0.0)
        ss_c += c * c
        ss_m += m * m
        wc = tl.load(w_rows + d[None, :], mask=dm[None, :], other=0.0)
        wm = tl.load(w_rows + D + d[None, :], mask=dm[None, :], other=0.0)
        acc_c += tl.sum(wc * c[None, :], axis=1)
        acc_m += tl.sum(wm * m[None, :], axis=1)
    sc = libdevice.rsqrt(tl.sum(ss_c, axis=0) / D + eps)
    sm = libdevice.rsqrt(tl.sum(ss_m, axis=0) / D + eps)
    tl.store(G + t * stride_g + r, acc_c * sc + acc_m * sm)


@triton.jit
def _ctrl_out_kernel(CUR, MSG, G, GAIN, FORGET, OUT, D, stride_c, stride_m, stride_g, stride_gain, stride_forget,
                     stride_o, RANK: tl.constexpr, BLOCK_N: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n < D
    rr = tl.arange(0, RANK)
    g = tl.load(G + t * stride_g + rr)
    wa = tl.load(GAIN + n[:, None] * stride_gain + rr[None, :], mask=nm[:, None], other=0.0)
    wf = tl.load(FORGET + n[:, None] * stride_forget + rr[None, :], mask=nm[:, None], other=0.0)
    ga = libdevice.tanh(tl.sum(wa * g[None, :], axis=1))
    fo = libdevice.tanh(tl.sum(wf * g[None, :], axis=1))
    c = tl.load(CUR + t * stride_c + n, mask=nm, other=0.0)
    m = tl.load(MSG + t * stride_m + n, mask=nm, other=0.0)
    tl.store(OUT + t * stride_o + n, c - fo * c + ga * m, mask=nm)


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def control_fused(cur: torch.Tensor, message: torch.Tensor, down: torch.Tensor, gain: torch.Tensor,
                  forget: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """cur / message [T, D] float32 (row strides free), down [r, 2D], gain / forget [D, r] float32 -> [T, D]."""
    T, D = cur.shape
    rank = down.shape[0]
    out = torch.empty(T, D, dtype=torch.float32, device=cur.device)
    if T == 0:
        return out
    if rank % CTRL_BLOCK_R or rank & (rank - 1) or down.shape[1] != 2 * D:
        raise ValueError(f"controller shapes: down {tuple(down.shape)}, D {D}")
    cur, message = _rows(cur), _rows(message)
    down, gain, forget = _rows(down), _rows(gain), _rows(forget)
    g = torch.empty(T, rank, dtype=torch.float32, device=cur.device)
    _ctrl_down_kernel[(T, rank // CTRL_BLOCK_R)](
        cur, message, down, g, D, cur.stride(0), message.stride(0), down.stride(0), g.stride(0), float(eps),
        BLOCK_R=CTRL_BLOCK_R, BLOCK_D=CTRL_BLOCK_D, num_warps=4)
    _ctrl_out_kernel[(T, triton.cdiv(D, CTRL_BLOCK_N))](
        cur, message, g, gain, forget, out, D, cur.stride(0), message.stride(0), g.stride(0), gain.stride(0),
        forget.stride(0), out.stride(0), RANK=rank, BLOCK_N=CTRL_BLOCK_N, num_warps=4)
    return out


# ---- depth mix (ref.mix): softmax over the RMS-normalized states (blocks + cur) with learned queries ---------------
# K1, one program per (token, state): sum of squares and the query dots of that state's row -> the state's RMS scale
# and its scaled scores; K2, one program per (token, channel block): softmax over the states for each query, then the
# weighted sum of the normalized states.  The states stay separate tensors (pointer arguments, unrolled over S).
MIX_MAX_STATES = 8
MIX_BLOCK_D = 256


@triton.jit
def _state_ptr(s: tl.constexpr, P0, P1, P2, P3, P4, P5, P6, P7):
    if s == 0:
        return P0
    elif s == 1:
        return P1
    elif s == 2:
        return P2
    elif s == 3:
        return P3
    elif s == 4:
        return P4
    elif s == 5:
        return P5
    elif s == 6:
        return P6
    else:
        return P7


@triton.jit
def _mix_scores_kernel(P0, P1, P2, P3, P4, P5, P6, P7, Q, SC, RS, D, stride_p, stride_q, inv_sqrt_d, eps,
                       S: tl.constexpr, NQ: tl.constexpr, NQP: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s_id = tl.program_id(1)
    qi = tl.arange(0, NQP)
    qm = qi < NQ
    ss = tl.zeros([BLOCK_D], dtype=tl.float32)
    dots = tl.zeros([NQP, BLOCK_D], dtype=tl.float32)
    for s in tl.static_range(S):
        if s_id == s:
            base = _state_ptr(s, P0, P1, P2, P3, P4, P5, P6, P7) + t * stride_p
            for d0 in range(0, D, BLOCK_D):
                d = d0 + tl.arange(0, BLOCK_D)
                dm = d < D
                x = tl.load(base + d, mask=dm, other=0.0)
                ss += x * x
                qv = tl.load(Q + qi[:, None] * stride_q + d[None, :], mask=qm[:, None] & dm[None, :], other=0.0)
                dots += qv * x[None, :]
    rs = libdevice.rsqrt(tl.sum(ss, axis=0) / D + eps)
    tl.store(RS + t * S + s_id, rs)
    tl.store(SC + (t * NQ + qi) * S + s_id, tl.sum(dots, axis=1) * rs * inv_sqrt_d, mask=qm)


@triton.jit
def _mix_out_kernel(P0, P1, P2, P3, P4, P5, P6, P7, SC, RS, OUT, D, stride_p, stride_ot, stride_oq,
                    S: tl.constexpr, SP: tl.constexpr, NQ: tl.constexpr, NQP: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < D
    qi = tl.arange(0, NQP)
    qm = qi < NQ
    si = tl.arange(0, SP)
    sm = si < S
    sc = tl.load(SC + (t * NQ + qi[:, None]) * S + si[None, :], mask=qm[:, None] & sm[None, :], other=-float("inf"))
    mx = tl.max(sc, axis=1)
    mx = tl.where(qm, mx, 0.0)
    e = tl.exp(sc - mx[:, None])
    e = tl.where(qm[:, None] & sm[None, :], e, 0.0)
    den = tl.sum(e, axis=1)
    den = tl.where(qm, den, 1.0)
    rs = tl.load(RS + t * S + si, mask=sm, other=0.0)
    wts = e / den[:, None] * rs[None, :]                                    # [NQP, SP]: p(s) x rms scale of s
    acc = tl.zeros([NQP, BLOCK_D], dtype=tl.float32)
    for s in tl.static_range(S):
        x = tl.load(_state_ptr(s, P0, P1, P2, P3, P4, P5, P6, P7) + t * stride_p + d, mask=dm, other=0.0)
        w_s = tl.sum(tl.where(si[None, :] == s, wts, 0.0), axis=1)            # [NQP]
        acc += w_s[:, None] * x[None, :]
    tl.store(OUT + t * stride_ot + qi[:, None] * stride_oq + d[None, :], acc, mask=qm[:, None] & dm[None, :])


def mix_fused(blocks: list, cur: torch.Tensor, query: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ref.mix of float32 states (blocks + [cur], each [T, D] with unit last stride and the same row stride) and
    query [nq, D] -> [T, nq, D] float32."""
    states = list(blocks) + [cur]
    S = len(states)
    T, D = cur.shape
    nq = query.shape[0]
    out = torch.empty(T, nq, D, dtype=torch.float32, device=cur.device)
    if T == 0:
        return out
    stride = cur.stride(0)
    if S > MIX_MAX_STATES or any(x.stride(-1) != 1 or x.stride(0) != stride or x.shape != cur.shape
                                 or x.dtype != torch.float32 for x in states):
        raise ValueError("mix_fused: states must be float32 [T, D] rows with the same strides")
    query = _rows(query)
    ptrs = states + [cur] * (MIX_MAX_STATES - S)
    sc = torch.empty(T, nq, S, dtype=torch.float32, device=cur.device)
    rs = torch.empty(T, S, dtype=torch.float32, device=cur.device)
    nqp = triton.next_power_of_2(nq)
    _mix_scores_kernel[(T, S)](*ptrs, query, sc, rs, D, stride, query.stride(0), 1.0 / D ** 0.5, float(eps),
                               S=S, NQ=nq, NQP=nqp, BLOCK_D=MIX_BLOCK_D, num_warps=4)
    _mix_out_kernel[(T, triton.cdiv(D, MIX_BLOCK_D))](*ptrs, sc, rs, out, D, stride, out.stride(0), out.stride(1),
                                                     S=S, SP=triton.next_power_of_2(S), NQ=nq, NQP=nqp,
                                                     BLOCK_D=MIX_BLOCK_D, num_warps=4)
    return out
