"""BF16 projections over the eight selected experts for each Inkling token."""
import torch
import triton
import triton.language as tl

from ..triton.gemm.iq_quant.iq_tables import get_iq_table_tensors
from ..triton.gemm.utils import load_f16_from_u8


@triton.jit
def routed_iq4_b1_kernel(x, weights, ids, out, table, ROWS: tl.constexpr,
                         WIDTH: tl.constexpr, TILE_N: tl.constexpr, TWO_TOKENS: tl.constexpr = False):
    route = tl.program_id(1)
    expert = tl.load(ids + route).to(tl.int64)
    if TWO_TOKENS:
        x += (route // 8) * WIDTH
    rows = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    valid_rows = rows < ROWS
    row_bytes: tl.constexpr = WIDTH // 32 * 18
    base = weights + (expert * ROWS + rows[:, None]) * row_bytes
    blocks = tl.arange(0, 4)
    nibbles = tl.arange(0, 16)
    accum = tl.full((16, TILE_N), 0, tl.float32)
    # Keep the frozen projection's low/high interleave and 128-column MMA order.
    for begin in range(0, WIDTH // 32, 4):
        k = begin + blocks
        wptr = base + 18 * k[None, :]
        half = x.dtype.element_ty
        scale = load_f16_from_u8(wptr, valid_rows[:, None]).to(half)
        packed = tl.load(wptr[:, :, None] + 2 + nibbles[None, None, :],
                         valid_rows[:, None, None], other=0)
        w0 = tl.load(table + (packed & 15).to(tl.int32)).to(half) * scale[:, :, None]
        w1 = tl.load(table + (packed >> 4).to(tl.int32)).to(half) * scale[:, :, None]
        xptr = x + k[:, None] * 32 + nibbles[None, :]
        x0 = tl.load(xptr)
        x1 = tl.load(xptr + 16)
        wtile = tl.reshape(tl.join(w0, w1), (TILE_N, 128))
        xrow = tl.reshape(tl.join(x0, x1), (128,))
        xmat = tl.broadcast_to(xrow[None, :], (16, 128))
        accum = tl.dot(xmat, tl.trans(wtile), acc=accum)
    lanes = tl.arange(0, 16)
    tl.store(out + route * ROWS + rows[None, :] + lanes[:, None] * 0,
             accum, (lanes[:, None] == 0) & valid_rows[None, :])


def direct_gate_up(x, weights, ids, tile):
    rows = weights.shape[1]
    tokens = x.shape[0]
    if tokens not in (1, 2, 4) or ids.shape != (tokens, 8):
        raise ValueError("Expected one, two or four independent top-eight Inkling token rows")
    out = torch.empty((tokens * 8, rows), dtype=x.dtype, device=x.device)
    table = get_iq_table_tensors(x.device)['kvalues_iq4nl']
    routed_iq4_b1_kernel[(triton.cdiv(rows, tile), tokens * 8)](
        x, weights, ids, out, table, rows, x.shape[-1], tile, tokens > 1,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


def down_route(device, tokens=1):
    """One valid route and three masked lanes for each selected token/expert pair."""
    if tokens not in (1, 2, 4):
        raise ValueError('Expected one, two or four top-eight routes')
    rows = tokens * 8
    order = torch.full((rows, 4), rows, dtype=torch.int32, device=device)
    order[:, 0] = torch.arange(rows, dtype=torch.int32, device=device)
    return order.flatten(), torch.tensor([rows * 4], dtype=torch.int32, device=device)


def grouped_down(x, weights, ids, tile, route):
    from ..triton.fused_moe.iq_quant.iq4_nl import iq4_nl_moe_kernel
    from ..triton.fused_moe.utils import run_triton_fused_moe_kernel
    from ..triton.gemm.utils import GGML_TYPE_IQ4_NL

    order, padded = route
    # N16 uses the two-warp register decoder qualified on the RTX 3090.
    # Retain the explicit N32 table specialization for independent controls.
    register = tile == 16
    table = None if register else get_iq_table_tensors(weights.device)['kvalues_iq4nl']
    tuning = {'num_warps':2, 'num_stages':2} if register else {}
    return run_triton_fused_moe_kernel(
        iq4_nl_moe_kernel, weights, x, order, ids.flatten().int(), padded, weights.shape[1],
        1, x.shape[0], GGML_TYPE_IQ4_NL, extra_args=(table,), block_m=4, block_n=tile, **tuning,
    )
