"""Routed Inkling IQ1/IQ2 projection candidates for timing against the standard BF16 path."""
import torch
import triton
import triton.language as tl

from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import get_iq_table_tensors
from vllm_gguf_plugin.triton.gemm.utils import load_f16_from_u8, load_u16_from_u8, load_u32_from_u8

BLOCKS = {16: (66, 'iq2xxs_grid'), 17: (74, 'iq2xs_grid'), 19: (50, 'iq1s_grid'), 22: (82, 'iq2s_grid')}


@triton.jit
def unpack_block(p, grid, sign_table, elements, TYPE: tl.constexpr, DTYPE: tl.constexpr):
    # p is [N,1]; each N row starts one packed 256-element GGUF superblock.
    component = elements % 8
    group = elements // 8
    ib = group // 4
    il = group % 4
    d = load_f16_from_u8(p, True)
    if TYPE == 19:
        qh = load_u16_from_u8(p + 34 + 2 * ib[None, :], True)
        codes = tl.load(p + 2 + group[None, :]).to(tl.int32)
        idx = codes | (((qh >> (3 * il[None, :])) & 7).to(tl.int32) << 8)
        delta = tl.where((qh & 0x8000) != 0, -9, -7).to(tl.int16)
        scale = (d.to(DTYPE) * (2 * ((qh >> 12) & 7).to(DTYPE) + 1.0) * 0.125).to(DTYPE)
        lookup = tl.load(grid + idx * 8 + component[None, :]).to(tl.int16)
        w = ((lookup * 8 + delta).to(DTYPE) * scale).to(DTYPE)
    else:
        if TYPE == 16:
            codeptr = p + 2 + 8 * ib[None, :]
            idx = tl.load(codeptr + il[None, :]).to(tl.int32)
            aux = load_u32_from_u8(codeptr + 4, True)
            signs = tl.load(sign_table + ((aux >> (7 * il[None, :])) & 127))
            ds = (d * ((aux >> 28).to(tl.float32) + 0.5) * 0.25).to(DTYPE)
        else:
            if TYPE == 17:
                code = load_u16_from_u8(p + 2 + 2 * group[None, :], True).to(tl.int32)
                idx = code & 511
                signs = tl.load(sign_table + (code >> 9))
                sc = tl.load(p + 66 + ib[None, :])
            else:
                idx = tl.load(p + 2 + group[None, :]).to(tl.int32)
                high = tl.load(p + 66 + ib[None, :]).to(tl.int32)
                idx = idx | ((high << (8 - 2 * il[None, :])) & 0x300)
                signs = tl.load(p + 34 + group[None, :])
                sc = tl.load(p + 74 + ib[None, :])
            scale = (((sc >> (4 * (il[None, :] // 2))) & 15).to(DTYPE) + 0.5) * 0.25
            ds = (d.to(DTYPE) * scale).to(DTYPE)
        magnitude = tl.load(grid + idx * 8 + component[None, :]).to(DTYPE)
        sign = tl.where(((signs >> component[None, :]) & 1) != 0, -1, 1).to(DTYPE)
        w = (magnitude * sign * ds).to(DTYPE)
    return w


@triton.jit
def dot8_bf16(x, weights, ROWS: tl.constexpr, CHUNK: tl.constexpr):
    # The standard SM120 IQ kernel reduces with warp shuffles 4, 2, 1.
    # Its first add fuses the low lane's product and rounds the high product first.
    groups: tl.constexpr = CHUNK // 8
    xlo, xhi = tl.split(tl.permute(tl.reshape(x, (ROWS, groups, 2, 4)), (0, 1, 3, 2)))
    wlo, whi = tl.split(tl.permute(tl.reshape(weights, (ROWS, groups, 2, 4)), (0, 1, 3, 2)))
    high = xhi * whi
    pairs = tl.inline_asm_elementwise("fma.rn.bf16 $0, $1, $2, $3;", constraints="=h,h,h,h",
                                      args=[xlo, wlo, high], dtype=tl.bfloat16, is_pure=True, pack=1)
    lo2, hi2 = tl.split(tl.permute(tl.reshape(pairs, (ROWS, groups, 2, 2)), (0, 1, 3, 2)))
    x0, x1 = tl.split(lo2 + hi2)
    return x0 + x1


@triton.jit
def decode_gate(x, w, ids, out, grid, signs, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                BLOCK_BYTES: tl.constexpr, TYPE: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr):
    expert_slot = tl.program_id(1)
    expert = tl.load(ids + expert_slot).to(tl.int64)
    rows = tl.program_id(0) * N + tl.arange(0, N)
    base = w + (expert * ROWS + rows[:, None]) * (WIDTH // 256 * BLOCK_BYTES)
    acc = tl.full((N,), 0, tl.float32)
    lane = tl.arange(0, CHUNK)
    for superblock in range(WIDTH // 256):
        for start in tl.static_range(0, 256, CHUNK):
            k = start + lane
            weight = unpack_block(base + superblock * BLOCK_BYTES, grid, signs, k, TYPE, x.dtype.element_ty)
            activation = tl.load(x + superblock * 256 + k)
            # Preserve the original eight-lane product/sum and sequential FP32 accumulation.
            sums = dot8_bf16(tl.broadcast_to(activation[None, :], (N, CHUNK)), weight, N, CHUNK)
            for group in tl.static_range(CHUNK // 8):
                col = tl.gather(sums, tl.full((N, 1), group, tl.int32), 1)
                acc += tl.reshape(col, (N,))
    tl.store(out + expert_slot * ROWS + rows, acc)


@triton.jit
def prefill_gate(x, w, order, experts, out, grid, signs, COUNT: tl.constexpr,
                 ROWS: tl.constexpr, WIDTH: tl.constexpr, BYTES: tl.constexpr, TYPE: tl.constexpr,
                 M: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr, ORDERED: tl.constexpr=False):
    block = tl.program_id(0)
    expert = tl.load(experts + block).to(tl.int64)
    if expert < 0:
        return
    routed = tl.load(order + block * M + tl.arange(0, M))
    valid = (routed >= 0) & (routed < COUNT)
    rows = tl.program_id(1) * N + tl.arange(0, N)
    base = w + (expert * ROWS + rows[:, None]) * (WIDTH // 256 * BYTES)
    acc = tl.full((M, N), 0, tl.float32)
    lane = tl.arange(0, CHUNK)
    for superblock in range(WIDTH // 256):
        for start in tl.static_range(0, 256, CHUNK):
            k = start + lane
            weight = unpack_block(base + superblock * BYTES, grid, signs, k, TYPE, x.dtype.element_ty)
            activation = tl.load(x + (routed // 8)[:, None] * WIDTH + superblock * 256 + k[None, :],
                                 valid[:, None], other=0)
            if ORDERED:
                ax = tl.reshape(tl.broadcast_to(activation[:, None, :], (M, N, CHUNK)), (M*N, CHUNK))
                wx = tl.reshape(tl.broadcast_to(weight[None, :, :], (M, N, CHUNK)), (M*N, CHUNK))
                sums = tl.reshape(dot8_bf16(ax, wx, M*N, CHUNK), (M, N, CHUNK//8))
                for group in tl.static_range(CHUNK // 8):
                    part = tl.gather(sums, tl.full((M, N, 1), group, tl.int32), 2)
                    acc += tl.reshape(part, (M, N))
            else:
                acc = tl.dot(activation, tl.trans(weight), acc=acc)
    tl.store(out + routed[:, None] * ROWS + rows[None, :], acc, valid[:, None])


def project(x, weights, ids, ggml_type, tile_n, chunk, warps, tile_m=16, ordered=False):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    tables = get_iq_table_tensors(x.device)
    block_bytes, key = BLOCKS[ggml_type]
    rows = weights.shape[1]
    tokens, width = x.shape
    assert ids.shape == (tokens,8) and rows % tile_n == 0 and width % 256 == 0
    output = x.new_empty((tokens * 8, rows))
    if tokens == 1:
        decode_gate[(rows // tile_n, 8)](x, weights, ids, output, tables[key], tables['ksigns_iq2xs'],
            rows, width, block_bytes, ggml_type, tile_n, chunk, num_warps=warps, num_stages=1)
    else:
        order, experts, _ = moe_align_block_size(ids.int(), tile_m, weights.shape[0])
        need = experts.numel() * tile_m - order.numel()
        if need:
            order = torch.nn.functional.pad(order, (0, need), value=tokens*8)
        prefill_gate[(experts.numel(), rows // tile_n)](x, weights, order, experts, output,
            tables[key], tables['ksigns_iq2xs'], tokens*8, rows, width, block_bytes, ggml_type,
            tile_m, tile_n, chunk, ORDERED=ordered, num_warps=warps, num_stages=2)
    return output
