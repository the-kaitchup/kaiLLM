# SPDX-License-Identifier: Apache-2.0
"""Incremental paged group pooling and sparse selection for Hy4 target CUDA graphs."""
from functools import lru_cache

import triton
import triton.language as tl


@triton.jit
def _write_rows(CACHE, VALUES, SLOTS, CACHE_STRIDE, VALUE_STRIDE, SLOT_STRIDE, PAGE, STRIDE_ROWS,
                WIDTH: tl.constexpr, TILE: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(SLOTS + t * SLOT_STRIDE).to(tl.int64)
    if slot < 0:
        return
    row = (slot // PAGE) * STRIDE_ROWS + slot % PAGE
    k = tl.arange(0, TILE)
    value = tl.load(VALUES + t * VALUE_STRIDE + k, mask=k < WIDTH, other=0.)
    tl.store(CACHE + row * CACHE_STRIDE + k, value, mask=k < WIDTH)


def write_rows(cache, values, slots, page, stride_rows):
    if cache.stride(-1) != 1 or values.stride(-1) != 1 or values.shape[1] != cache.shape[1]:
        raise ValueError("Hy4 paged cache requires contiguous row vectors")
    if slots.numel():
        _write_rows[(slots.numel(),)](cache, values, slots, cache.stride(0), values.stride(0),
                                      slots.stride(0), page, stride_rows, WIDTH=cache.shape[1],
                                      TILE=triton.next_power_of_2(cache.shape[1]))


@triton.jit
def _scores(Q, HEAD_W, KEY, BT, REQ, POS, LENS, SLOT, COS, SIN, LOGITS, BEGIN,
            STRIDE_Q, STRIDE_W, STRIDE_K, STRIDE_BT, STRIDE_L, STRIDE_ANG,
            PAGE: tl.constexpr, STRIDE_ROWS, RATIO: tl.constexpr, CANDIDATES: tl.constexpr,
            HEADS: tl.constexpr, DIM: tl.constexpr, ROT: tl.constexpr, GROUPS: tl.constexpr,
            BH: tl.constexpr, BG: tl.constexpr, BD: tl.constexpr):
    token = BEGIN + tl.program_id(0)
    first = tl.program_id(1) * BG
    req = tl.load(REQ + token)
    own = tl.load(POS + token) // RATIO
    if first >= own or tl.load(LENS + req) <= CANDIDATES or tl.load(SLOT + token) < 0:
        return
    g = first + tl.arange(0, BG)
    valid = (g < own) & (g < GROUPS)
    endpos = (g + 1) * RATIO - 1
    block = tl.load(BT + req * STRIDE_BT + endpos // PAGE, mask=valid, other=0).to(tl.int64)
    row = (block * STRIDE_ROWS + endpos % PAGE) * STRIDE_K
    d = tl.arange(0, BD)
    h = tl.arange(0, BH)
    q = tl.load(Q + token * STRIDE_Q + h[:, None] * DIM + d[None, :], mask=(h[:, None] < HEADS) & (d[None, :] < DIM), other=0.)
    key = tl.load(KEY + row[None, :] + d[:, None], mask=(d[:, None] < DIM) & valid[None, :], other=0.).to(tl.float32)
    if ROT:
        delta = d - (DIM - ROT)
        rotated = (delta >= 0) & (d < DIM)
        pair = d ^ 1
        companion = tl.load(KEY + row[None, :] + pair[:, None], mask=rotated[:, None] & valid[None, :], other=0.).to(tl.float32)
        angle = g[None, :] * STRIDE_ANG + delta[:, None] // 2
        cos = tl.load(COS + angle, mask=rotated[:, None] & valid[None, :], other=1.)
        sin = tl.load(SIN + angle, mask=rotated[:, None] & valid[None, :], other=0.)
        key = tl.where(rotated[:, None], key * cos + tl.where((d % 2)[:, None] == 0, -companion, companion) * sin, key)
    scores = tl.maximum(tl.dot(q, key, input_precision='ieee'), 0.)
    w = tl.load(HEAD_W + token * STRIDE_W + h, mask=h < HEADS, other=0.)
    value = tl.sum(w[:, None] * scores, axis=0)
    tl.store(LOGITS + tl.program_id(0) * STRIDE_L + g, value, mask=valid)


def complete_groups(flat, stride, md, tokens, weights, page):
    """Pool newly completed groups once. Their last raw token slot now holds the group key."""
    import torch
    from .attention import pool_groups

    ratio = weights.ratio
    lengths = md.query_start_loc[1:] - md.query_start_loc[:-1]
    context = (md.seq_lens - lengths).clamp_min(0)
    first_group = context // ratio
    counts = (md.seq_lens // ratio - first_group).clamp_min(0)
    ends = torch.cumsum(counts, 0)
    grid = torch.arange(triton.cdiv(tokens, ratio) + lengths.numel(), device=flat.device)
    owner = torch.searchsorted(ends, grid, right=True).clamp_max(lengths.numel() - 1)
    live = grid < ends[-1]
    group = first_group[owner] + grid - (ends - counts)[owner]
    pos = torch.where(live[:, None], group[:, None] * ratio + torch.arange(ratio, device=flat.device), 0)
    local_row = md.query_start_loc[owner] + pos[:, -1] - context[owner]
    # A padded request's block-table row need not contain valid page IDs.
    slots = torch.where(live, md.slot_mapping[local_row.clamp(0, tokens - 1)], -1)
    block = md.block_table[owner[:, None], pos // page].long()
    rows = torch.where(slots[:, None] >= 0, block * stride + pos % page, 0)
    packed = flat[rows]
    value = pool_groups(packed.reshape(-1, weights.head_dim), weights)
    write_rows(flat, value, slots, page, stride)


@lru_cache(maxsize=8)
def group_rotary(groups, ratio, rope_dim, theta, device):
    import torch
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / max(1, rope_dim)))
    phase = (torch.arange(groups, device=device) * ratio).float()[:, None] * inv
    return phase.cos(), phase.sin()


def select_paged(hidden, q, positions, cache, stride, md, w, page, max_context, width):
    """Select the same compressed groups as the reference for ragged, paged requests."""
    import torch
    from .attention import dense_indices, rope_interleaved, topk_groups

    tokens = positions.numel()
    groups = max(1, max_context // w.ratio)
    count = min(w.group_topk - 1, groups)
    cols = torch.arange(w.group_topk, device=q.device)
    offs = torch.arange(w.ratio, device=q.device)
    head = w.head_dim - w.rope_dim
    q = q.reshape(tokens, w.num_heads, w.head_dim).float()
    if w.rope_dim:
        q = torch.cat((q[..., :head], rope_interleaved(q[..., head:], positions, w.theta)), dim=-1)
    q = q.contiguous()
    mixing = ((hidden.float() @ w.head_weight_projection.float().t()) *
              (w.num_heads**-.5 * w.head_dim**-.5)).contiguous()
    cos, sin = group_rotary(groups, w.ratio, w.rope_dim, w.theta, q.device)
    index = torch.empty(tokens, width, dtype=torch.int32, device=q.device)
    for start in range(0, tokens, 32):
        stop = min(start+32, tokens)
        logits = torch.full((stop-start, groups), -float('inf'), dtype=torch.float32, device=q.device)
        _scores[(stop-start, triton.cdiv(groups, 32))](
            q, mixing, cache, md.block_table, md.req_id_per_token, positions, md.seq_lens, md.slot_mapping,
            cos, sin, logits, start, q.stride(0), mixing.stride(0), cache.stride(0), md.block_table.stride(0),
            logits.stride(0), cos.stride(0), page, stride, w.ratio, width, w.num_heads, w.head_dim,
            w.rope_dim, groups, BH=max(16,triton.next_power_of_2(w.num_heads)), BG=32,
            BD=max(16,triton.next_power_of_2(w.head_dim)), enable_fp_fusion=False)
        pos = positions[start:stop]
        lens = md.seq_lens[md.req_id_per_token[start:stop].long()]
        own = pos // w.ratio
        choices = torch.full((stop-start, w.group_topk), -1, dtype=torch.long, device=q.device)
        if count:
            chosen = topk_groups(logits, count)
            choices[:, :count] = torch.where(logits.gather(1, chosen) > -float('inf'), chosen, -1)
        where_own = (lens // w.ratio).clamp(0, w.group_topk-1)
        choices = torch.where(cols[None] == where_own[:, None], own[:, None], choices)
        picked = choices[..., None] * w.ratio + offs
        sparse = torch.where((choices[..., None] >= 0) & (picked <= pos[:, None, None]), picked, -1)
        dense = dense_indices(pos, width)
        use_sparse = lens > width
        index[start:stop] = torch.where(use_sparse[:, None], sparse.reshape(stop-start, width), dense)
    return index
