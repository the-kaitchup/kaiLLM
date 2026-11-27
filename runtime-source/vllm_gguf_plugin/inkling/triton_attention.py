# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 paged attention in Triton: relative-position bias, per-query sink, sliding window, GQA.

logit(i, j) = scale q_i . k_j + P[i, idx(d)] with d = pos_i - pos_j, idx = d if d < E else E, and an extra zero logit
(the sink column), exactly ref.attention / llama.cpp.  P [T, H, E + 1] = R_i . bias^T (zero past the extent) - t_i is
built once per layer visit (one small GEMM), so the kernel gathers one value per (row, key).

One program per (tile, KV head): a tile is up to BLOCK_M // G query tokens of one request x the G query heads of the KV
head (rows token-major); decode-only batches use 16-row tiles (one token per request) taken from the metadata
builder's persistent buffer, so decode steps replay as CUDA graphs.  The sink enters the online softmax as its
initial state (m = 0, l = 1).  Keys are read from vLLM's logical [blocks, kv_heads, block_size, K | V] page through
its strides; only keys inside the causal window are touched, so sliding-window caches whose old blocks were released
never read them.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from . import chima_verifier, spark_verify_router


@triton.jit
def _rel_attn_kernel(Q, CACHE, BT, SEQ, QSL, TILES, P, OUT,
                     stride_qt, stride_qh, stride_cb, stride_ch, stride_cn, stride_bt, stride_ot, stride_oh,
                     scale, window, E, first_kv_head,
                     BIAS_HEADS: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    local_kvh = tl.program_id(1)
    kvh = first_kv_head + local_kvh
    req = tl.load(TILES + 2 * tile)
    if req < 0:
        return
    q0 = tl.load(TILES + 2 * tile + 1)
    q_start = tl.load(QSL + req)
    q_len = tl.load(QSL + req + 1) - q_start
    seq_len = tl.load(SEQ + req)
    ctx = seq_len - q_len
    TQ: tl.constexpr = BLOCK_M // G
    rows = tl.arange(0, BLOCK_M)
    tq = rows // G
    hg = rows - tq * G
    tok = q0 + tq
    row_ok = (tq < TQ) & (tok < q_len)
    head = kvh * G + hg
    tglob = (q_start + tok).to(tl.int64)
    qpos = ctx + tok
    offs_d = tl.arange(0, D)
    q = tl.load(Q + tglob[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
                mask=row_ok[:, None], other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)            # the sink logit (0)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)        # exp(0 - 0)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    last_tok = tl.minimum(q0 + TQ, q_len) - 1
    k_end = ctx + last_tok + 1                             # exclusive
    k_begin = 0
    if window > 0:
        k_begin = tl.maximum(0, ctx + q0 - window + 1)
    p_row = P + (tglob * BIAS_HEADS + local_kvh * G + hg) * (E + 1)
    bt_row = BT + req.to(tl.int64) * stride_bt
    offs_n = tl.arange(0, BLOCK_N)
    for k0 in range(k_begin, k_end, BLOCK_N):
        kpos = k0 + offs_n
        k_ok = kpos < k_end
        blk = tl.load(bt_row + kpos // PAGE, mask=k_ok, other=0).to(tl.int64)
        base = CACHE + blk * stride_cb + kvh * stride_ch + (kpos % PAGE) * stride_cn
        k = tl.load(base[:, None] + offs_d[None, :], mask=k_ok[:, None], other=0.0)
        v = tl.load(base[:, None] + D + offs_d[None, :], mask=k_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k.to(q.dtype)), out_dtype=tl.float32) * scale       # FP8 K widened exactly
        d = qpos[:, None] - kpos[None, :]
        vis = (d >= 0) & k_ok[None, :] & row_ok[:, None]
        if window > 0:
            vis = vis & (d < window)
        idx = tl.where(d < E, d, E)
        pb = tl.load(p_row[:, None] + idx, mask=vis, other=0.0)
        s = tl.where(vis, s + pb, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype), out_dtype=tl.float32)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + tglob[:, None] * stride_ot + head[:, None] * stride_oh + offs_d[None, :], out,
             mask=row_ok[:, None])


@triton.jit
def _rel_attn_split_kernel(Q, CACHE, BT, SEQ, QSL, P, ML, ACC,
                           stride_qt, stride_qh, stride_cb, stride_ch, stride_cn, stride_bt,
                           scale, window, E,
                           H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, S: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WIDTH: tl.constexpr = 1):
    """Decode (one query token per request, 0 for padding): keys of request pid0 / KV head pid1 split in S chunks;
    chunk pid2 writes its unnormalised softmax state (m, l, acc) for the G heads; the sink enters in the reduce."""
    req = tl.program_id(0) // WIDTH
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    q_start = tl.load(QSL + req)
    q_len = tl.load(QSL + req + 1) - q_start
    seq_len = tl.load(SEQ + req)
    ctx = seq_len - q_len
    slot_row = req
    if WIDTH > 1:
        pos = tl.program_id(0) % WIDTH
        if pos >= q_len:
            return
        q_start += pos
        ctx += pos
        seq_len = ctx + 1
        slot_row = q_start
    rows = tl.arange(0, BLOCK_M)
    row_ok = (rows < G) & (q_len > 0)
    head = kvh * G + rows
    tglob = q_start.to(tl.int64)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + tglob * stride_qt + head[:, None] * stride_qh + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    k_end = seq_len
    k_begin = 0
    if window > 0:
        k_begin = tl.maximum(0, ctx - window + 1)
    n = tl.maximum(k_end - k_begin, 0)
    chunk = tl.cdiv(tl.cdiv(n, S), BLOCK_N) * BLOCK_N
    s_begin = k_begin + split * chunk
    s_end = tl.minimum(k_end, s_begin + chunk)
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    p_row = P + (tglob * H + head) * (E + 1)
    bt_row = BT + req.to(tl.int64) * stride_bt
    offs_n = tl.arange(0, BLOCK_N)
    for k0 in range(s_begin, s_end, BLOCK_N):
        kpos = k0 + offs_n
        k_ok = kpos < s_end
        blk = tl.load(bt_row + kpos // PAGE, mask=k_ok, other=0).to(tl.int64)
        base = CACHE + blk * stride_cb + kvh * stride_ch + (kpos % PAGE) * stride_cn
        k = tl.load(base[:, None] + offs_d[None, :], mask=k_ok[:, None], other=0.0)
        v = tl.load(base[:, None] + D + offs_d[None, :], mask=k_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k.to(q.dtype)), out_dtype=tl.float32) * scale
        d = ctx - kpos
        vis = (d >= 0) & k_ok
        if window > 0:
            vis = vis & (d < window)
        vis2 = vis[None, :] & row_ok[:, None]
        idx = tl.where(d < E, d, E)
        pb = tl.load(p_row[:, None] + idx[None, :], mask=vis2, other=0.0)
        s = tl.where(vis2, s + pb, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype), out_dtype=tl.float32)
        m_i = m_new
    slot = (slot_row.to(tl.int64) * H + head) * S + split
    tl.store(ML + 2 * slot, m_i, mask=rows < G)
    tl.store(ML + 2 * slot + 1, l_i, mask=rows < G)
    tl.store(ACC + slot[:, None] * D + offs_d[None, :], acc, mask=(rows < G)[:, None])


@triton.jit
def _rel_attn_split_reduce(ML, ACC, QSL, OUT, stride_ot, stride_oh,
                           H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr = 1):
    """out = (sum_s acc_s e^(m_s - M)) / (e^(0 - M) + sum_s l_s e^(m_s - M)), M = max(0, m_s): the sink is a zero
    logit with a zero value."""
    req = tl.program_id(0) // WIDTH
    head = tl.program_id(1)
    q_start = tl.load(QSL + req)
    q_len = tl.load(QSL + req + 1) - q_start
    slot_row = req
    if WIDTH > 1:
        pos = tl.program_id(0) % WIDTH
        if pos >= q_len:
            return
        q_start += pos
        slot_row = q_start
    if q_len > 0:
        offs_s = tl.arange(0, S)
        offs_d = tl.arange(0, D)
        slot = (slot_row.to(tl.int64) * H + head) * S + offs_s
        m = tl.load(ML + 2 * slot)
        l = tl.load(ML + 2 * slot + 1)
        big = tl.maximum(tl.max(m, axis=0), 0.0)
        w = tl.exp(m - big)
        acc = tl.load(ACC + slot[:, None] * D + offs_d[None, :])
        num = tl.sum(acc * w[:, None], axis=0)
        den = tl.exp(0.0 - big) + tl.sum(l * w, axis=0)
        tl.store(OUT + q_start.to(tl.int64) * stride_ot + head * stride_oh + offs_d, num / den)


BLOCK_M = 64
BLOCK_N = 64
# Prefill can require a 192 MiB relative-bias matrix. Splitting by KV head
# retains each query's complete attention window and bounds temporary memory.
PREFILL_BIAS_BYTES = 64 << 20
# decode: keys split in SPLITS chunks per (request, KV head) (flash-decoding), a fixed count so the grid does not depend
# on the context (CUDA graphs); KAILLM_INKLING_SPLITS=0 keeps the one-program-per-(request, KV head) kernel
SPLITS = int(os.environ.get("KAILLM_INKLING_SPLITS", "16"))
SPARK_VERIFY_ROW_BIAS = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_BIAS") == "1"
FP8_QUERIES = os.environ.get("KAILLM_INKLING_FP8_QUERIES") == "1"


def tiles_for(md, G: int, tokens: int):
    """Build a fixed-size tile table from the runner's live device query offsets."""
    fixed = getattr(md, "tiles", None)
    if fixed is not None:
        return fixed
    cache = {}
    if md.query_start_loc.is_cuda:
        from .conv_graph import forward_plan_cache
        cache = forward_plan_cache()
    key = ("attention", id(md), G, tokens)
    if key in cache:
        return cache[key]
    width = BLOCK_M // G
    offsets = md.query_start_loc
    requests = offsets.shape[0] - 1
    counts = (offsets[1:] - offsets[:-1] + width - 1) // width
    ends = torch.cumsum(counts, dim=0, dtype=torch.int32)
    # sum(ceil(length / width)) <= ceil(total / width) + requests - 1.
    size = triton.cdiv(tokens, width) + requests - 1
    indices = torch.arange(size, device=offsets.device, dtype=torch.int32)
    owner = torch.searchsorted(ends, indices, right=True, out_int32=True)
    safe_owner = owner.clamp_max(requests - 1).long()
    first = (indices - (ends - counts)[safe_owner]) * width
    table = torch.stack((torch.where(owner < requests, owner, -1), first), dim=1)
    cache[key] = table
    return table


def bias_table(rel, bias, sink):
    """P [T, H, E + 1] float32 = rel . [bias; 0]^T - sink (one GEMM).  [bias; 0] is cached on the weight tensor."""
    T, H, R = rel.shape
    table = getattr(bias, "_inkling_table", None)
    if table is None or table.device != bias.device:
        table = torch.cat([bias.float(), bias.new_zeros(1, bias.shape[1]).float()], dim=0)
        try:
            bias._inkling_table = table
        except AttributeError:
            pass
    P = (rel.reshape(T * H, R).float() @ table.t()).view(T, H, -1)
    return P.sub_(sink[..., None]).contiguous()          # sink in any float dtype: widened inside the sub


def verification_bias_table(rel, bias, sink):
    """Preserve the one-token FP32 relative-bias GEMM shape for each verifier row."""
    return torch.cat([bias_table(rel[i:i+1], bias, sink[i:i+1]) for i in range(rel.shape[0])], dim=0)


def rel_attention_triton(q, rel, sink, bias, positions, cache, md, scale: float, window, extent: int):
    """q [T, H, D], rel [T, H, n_rel], sink [T, H], bias [extent, n_rel]; cache [B, Hkv, bs, 2D] -> [T, H, D] f32."""
    T, H, D = q.shape
    Hkv = cache.shape[1]
    G = H // Hkv
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.float8_e4m3fn)
    # queries, scores and probabilities in BF16 (an FP8 cache is widened exactly in the kernel);
    # KAILLM_INKLING_FP8_QUERIES=1 rounds them to E4M3 instead
    qdt = torch.bfloat16 if cache.dtype == torch.float8_e4m3fn and not FP8_QUERIES else cache.dtype
    q = q.contiguous()
    if q.dtype != qdt:
        q = q.to(qdt)
    out = torch.empty(T, H, D, dtype=torch.float32, device=q.device)
    bt = md.block_table
    decode = getattr(md, "tiles", None) is not None
    width = getattr(md, 'verify_width', 1)
    serial_attention = decode or width > 1 and SPLITS > 1
    block_m = max(16, triton.next_power_of_2(G)) if serial_attention else BLOCK_M
    if serial_attention and SPLITS > 1 and not getattr(md, "one_pass_decode", False):
        if width > 1 and 1 < T <= 6 and md.num_reqs == 1 and (
            chima_verifier.active() or (SPARK_VERIFY_ROW_BIAS and width == T and spark_verify_router.active())
        ):
            P = verification_bias_table(rel, bias, sink)
        else:
            P = bias_table(rel, bias, sink)
        # One verifier query sees exactly the K/V prefix, partitions and BF16
        # probability rounding that a standalone target decode would see.
        requests = md.num_reqs if width > 1 else md.tiles.shape[0]
        rows = T if width > 1 else requests
        S = triton.next_power_of_2(SPLITS)
        ml = torch.empty(rows, H, S, 2, dtype=torch.float32, device=q.device)
        acc = torch.empty(rows, H, S, D, dtype=torch.float32, device=q.device)
        _rel_attn_split_kernel[(requests * width, Hkv, S)](
            q, cache, bt, md.seq_lens, md.query_start_loc, P, ml, acc,
            q.stride(0), q.stride(1), cache.stride(0), cache.stride(1), cache.stride(2), bt.stride(0),
            float(scale), int(window or 0), int(extent),
            H=H, G=G, D=D, PAGE=cache.shape[2], S=S, BLOCK_M=block_m, BLOCK_N=BLOCK_N, WIDTH=width,
            num_warps=4, num_stages=2)
        _rel_attn_split_reduce[(requests * width, H)](ml, acc, md.query_start_loc, out, out.stride(0), out.stride(1),
                                                      H=H, D=D, S=S, WIDTH=width, num_warps=4)
        return out
    tiles = tiles_for(md, G, T)
    bytes_per_kv_head = max(1, T * G * (extent + 1) * 4)
    group = Hkv if decode else min(Hkv, max(1, PREFILL_BIAS_BYTES // bytes_per_kv_head))
    for first in range(0, Hkv, group):
        count = min(group, Hkv - first)
        query_heads = slice(first * G, (first + count) * G)
        P = bias_table(rel[:, query_heads], bias, sink[:, query_heads])
        _rel_attn_kernel[(tiles.shape[0], count)](
            q, cache, bt, md.seq_lens, md.query_start_loc, tiles, P, out,
            q.stride(0), q.stride(1), cache.stride(0), cache.stride(1), cache.stride(2), bt.stride(0),
            out.stride(0), out.stride(1), float(scale), int(window or 0), int(extent), first,
            BIAS_HEADS=count * G, G=G, D=D, PAGE=cache.shape[2], BLOCK_M=block_m, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2)
        del P  # The next GEMM can reuse this group's workspace on the same CUDA stream.
    return out
