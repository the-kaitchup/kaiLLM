# SPDX-License-Identifier: Apache-2.0
"""BF16 routed gate/up projection for the standard Inkling IQ1/IQ2/IQ3 expert layouts."""
import os
from pathlib import Path

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
    elif TYPE == 18:
        control = load_u32_from_u8(p + 66 + 4 * ib[None, :], True)
        codes = tl.load(p + 2 + (elements // 4)[None, :]).to(tl.int32)
        signs = tl.load(sign_table + ((control >> (7 * il[None, :])) & 127))
        scale = (d.to(DTYPE) * ((control >> 28).to(DTYPE) + 0.5) * 0.5).to(DTYPE)
        magnitude = tl.load(grid + codes * 4 + (elements % 4)[None, :]).to(DTYPE)
        sign = tl.where(((signs >> component[None, :]) & 1) != 0, -1, 1).to(DTYPE)
        w = (magnitude * sign * scale).to(DTYPE)
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
def dot4_bf16(x, weights, ROWS: tl.constexpr, CHUNK: tl.constexpr):
    # IQ3_XXS accumulates each four-element result into FP32 in sequence. The
    # original SM120 kernel shuffles lanes by 2, then 1, using a BF16 FMA first.
    groups: tl.constexpr = CHUNK // 4
    lo, hi = tl.split(tl.permute(tl.reshape(x, (ROWS, groups, 2, 2)), (0, 1, 3, 2)))
    wl, wh = tl.split(tl.permute(tl.reshape(weights, (ROWS, groups, 2, 2)), (0, 1, 3, 2)))
    pairs = tl.inline_asm_elementwise("fma.rn.bf16 $0, $1, $2, $3;", constraints="=h,h,h,h",
                                      args=[lo, wl, hi * wh], dtype=tl.bfloat16, is_pure=True, pack=1)
    even, odd = tl.split(pairs)
    return even + odd


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
            if TYPE == 18:
                sums = dot4_bf16(tl.broadcast_to(activation[None, :], (N, CHUNK)), weight, N, CHUNK)
            else:
                sums = dot8_bf16(tl.broadcast_to(activation[None, :], (N, CHUNK)), weight, N, CHUNK)
            for group in tl.static_range(CHUNK // (4 if TYPE == 18 else 8)):
                col = tl.gather(sums, tl.full((N, 1), group, tl.int32), 1)
                acc += tl.reshape(col, (N,))
    tl.store(out + expert_slot * ROWS + rows, acc)


@triton.jit
def decode_gate_rows(x, w, ids, out, table, signs, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                     BYTES: tl.constexpr, TYPE: tl.constexpr, TILE: tl.constexpr, CHUNK: tl.constexpr):
    token = tl.program_id(2)
    decode_gate(x + token * WIDTH, w, ids + token * 8, out + token * 8 * ROWS, table, signs,
                ROWS, WIDTH, BYTES, TYPE, TILE, CHUNK)


@triton.jit
def prefill_gate(x, w, order, experts, padded_count, out, grid, signs, COUNT: tl.constexpr,
                 ROWS: tl.constexpr, WIDTH: tl.constexpr, BYTES: tl.constexpr, TYPE: tl.constexpr,
                 M: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr):
    block = tl.program_id(0)
    if block * M >= tl.load(padded_count):
        return
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
            ax = tl.reshape(tl.broadcast_to(activation[:, None, :], (M, N, CHUNK)), (M*N, CHUNK))
            wx = tl.reshape(tl.broadcast_to(weight[None, :, :], (M, N, CHUNK)), (M*N, CHUNK))
            if TYPE == 18:
                sums = tl.reshape(dot4_bf16(ax, wx, M*N, CHUNK), (M, N, CHUNK//4))
            else:
                sums = tl.reshape(dot8_bf16(ax, wx, M*N, CHUNK), (M, N, CHUNK//8))
            for group in tl.static_range(CHUNK // (4 if TYPE == 18 else 8)):
                part = tl.gather(sums, tl.full((M, N, 1), group, tl.int32), 2)
                acc += tl.reshape(part, (M, N))
    tl.store(out + routed[:, None] * ROWS + rows[None, :], acc, valid[:, None])


def run_projection(x, weights, ids, ggml_type, tile, *, verifier_rows=False):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
    tile_m, tile_n, chunk, warps = tile
    tables = get_iq_table_tensors(x.device)
    block_bytes, key = (98, 'iq3xxs_grid') if ggml_type == 18 else BLOCKS[ggml_type]
    rows = weights.shape[1]
    tokens, width = x.shape
    assert ids.shape == (tokens,8) and rows % tile_n == 0 and width % 256 == 0
    output = x.new_empty((tokens * 8, rows))
    if tokens == 1 and not verifier_rows:
        decode_gate[(rows // tile_n, 8)](x, weights, ids, output, tables[key], tables['ksigns_iq2xs'],
            rows, width, block_bytes, ggml_type, tile_n, chunk, num_warps=warps, num_stages=1)
    elif tile_m == 1:
        decode_gate_rows[(rows // tile_n, 8, tokens)](x, weights, ids, output, tables[key], tables['ksigns_iq2xs'],
            rows, width, block_bytes, ggml_type, tile_n, chunk, num_warps=warps, num_stages=1)
    else:
        order, experts, padded_count = moe_align_block_size(ids.int(), tile_m, weights.shape[0], pad_sorted_ids=True)
        prefill_gate[(experts.numel(), rows // tile_n)](x, weights, order, experts, padded_count, output,
            tables[key], tables['ksigns_iq2xs'], tokens*8, rows, width, block_bytes, ggml_type,
            tile_m, tile_n, chunk, num_warps=warps, num_stages=2)
    return output


# Both cards selected the same tiles in the original-weight comparison.
QUALIFIED_GPUS = frozenset(("NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 5080"))

def portable_tiles(config, device, ggml_type, row_bytes):
    """Select the measured packed geometry for ordinary BF16 expert routes."""
    phase = os.getenv("KAILLM_INKLING_PORTABLE_PACKED_GATE")
    if phase not in ("decode", "prefill", "both") or config is None:
        return None
    block_bytes = 98 if ggml_type == 18 else BLOCKS.get(ggml_type, (0,))[0]
    if (not block_bytes or row_bytes != 9 * block_bytes or device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (12, 0)
            or config.speculative_config is not None
            or config.parallel_config.world_size_across_dp != 1
            or config.scheduler_config.max_num_seqs != 1
            or config.model_config.dtype != torch.bfloat16
            or config.model_config.hf_config.model_type != "inkling2_loop"):
        return None
    tiles = {}
    if phase in ("decode", "both"):
        # The 70-SM device qualified a wider tile for these two packed formats.
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        columns = 32 if sms == 70 and ggml_type in (19, 22) else 16
        tiles[1] = (1, columns, 64, 4)
    if phase in ("prefill", "both"):
        tiles[2048] = (8, 64, 16, 4)
    return tiles


def measured_tiles(gpu, ggml_type, config, visible_devices, row_bytes):
    if gpu == "NVIDIA GB10" and ggml_type in BLOCKS and row_bytes == 9 * BLOCKS[ggml_type][0]:
        if (config is None or visible_devices != 1 or config.parallel_config.world_size_across_dp != 1
                or config.speculative_config is not None or config.scheduler_config.max_num_seqs != 1
                or config.scheduler_config.max_num_batched_tokens != 2048):
            return None
        model = config.model_config
        weights = getattr(model, "model_weights", None) or getattr(model, "model", None)
        if (not isinstance(weights, (str, Path)) or Path(weights).name != "Inkling-2-Mini-dondon-Q2_K_XS.gguf"
                or getattr(model.hf_config, "model_type", None) != "inkling2_loop"
                or model.dtype != torch.bfloat16 or model.max_model_len != 73728
                or torch.cuda.get_device_capability() != (12, 1)):
            return None
        # IQ1_S uses 64 input elements per tile. IQ2_XXS, IQ2_XS and IQ2_S use 128.
        return {1: (1, 16, 64 if ggml_type == 19 else 128, 4), 2048: (8, 64, 16, 4)}
    if gpu == "NVIDIA GeForce RTX 5090" and ggml_type in (16, 17, 18, 19, 22):
        from .dondon_5090 import released_tier
        size = 98 if ggml_type == 18 else BLOCKS[ggml_type][0]
        tier = released_tier(config, gpu, visible_devices) if row_bytes == 9 * size else None
        if tier == "S" or (tier == "M" and ggml_type != 19):
            if ggml_type == 18:
                return {1: (1, 16, 64, 4), 2048: (8, 32, 16, 4)}
            from .udq2_5090 import GATE_TILES
            return GATE_TILES[ggml_type]
    if ggml_type in BLOCKS and row_bytes == 9 * BLOCKS[ggml_type][0]:
        if gpu == "NVIDIA GeForce RTX 3090" and ggml_type in (16, 17, 19, 22) and config is not None and visible_devices == 1:
            model = getattr(config, "model_config", None)
            weights = getattr(model, "model_weights", None) or getattr(model, "model", None)
            if (config.speculative_config is None and config.scheduler_config.max_num_seqs == 1
                    and config.parallel_config.world_size_across_dp == 1
                    and getattr(getattr(model, "hf_config", None), "model_type", None) == "inkling2_loop"
                    and isinstance(weights, (str, Path)) and Path(weights).name == "Inkling-2-Mini-dondon-Q2_K_XS.gguf"):
                return {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)}
        from . import pro_udq2, udq2_5090
        if pro_udq2.supported(config, gpu, visible_devices):
            return pro_udq2.GATE_TILES.get(ggml_type)
        if gpu == "NVIDIA GeForce RTX 5090" and ggml_type in udq2_5090.DONDON_XS_B2:
            from .dondon_5090 import is_dondon_xs_b4
            if is_dondon_xs_b4(config, gpu, visible_devices):
                return {4: udq2_5090.DONDON_XS_B2[ggml_type], 2048: udq2_5090.GATE_TILES[ggml_type][2048]}
        if (ggml_type in udq2_5090.DONDON_XS_B2
                and udq2_5090.supports_dondon_xs_b2(config, gpu, visible_devices)):
            return {**udq2_5090.GATE_TILES[ggml_type], 2: udq2_5090.DONDON_XS_B2[ggml_type]}
        if udq2_5090.supported(config, gpu, visible_devices, include_dondon_xs=True):
            return udq2_5090.GATE_TILES.get(ggml_type)
    if gpu == "NVIDIA GeForce RTX 5080" and ggml_type in BLOCKS and row_bytes == 9 * BLOCKS[ggml_type][0]:
        from .projection_streams import supports_5080_b2
        if supports_5080_b2(config, gpu, visible_devices):
            prefill = (8, 64, 16, 4) if ggml_type in (16, 22) else (8, 32, 32, 4)
            return {1: (1, 16, 64, 4), 2: (1, 16, 64, 4), 2048: prefill}
    if (gpu not in QUALIFIED_GPUS or ggml_type not in BLOCKS or config is None or visible_devices != 1
            or config.parallel_config.world_size_across_dp != 1
            or config.scheduler_config.max_num_seqs != 1 or row_bytes != 9 * BLOCKS[ggml_type][0]):
        return None
    # Keys are the real number of input tokens. Values are M, N, K and warps per block.
    prefill = (8, 64, 16, 4) if ggml_type in (16, 22) else (8, 32, 32, 4)
    spec = config.speculative_config
    if spec is not None:
        from ..dspine import INKLING_ARCHS
        model = getattr(config, "model_config", None)
        source = getattr(model, "model_weights", None) or getattr(model, "model", None)
        draft = getattr(spec, "draft_model_config", None)
        if (gpu != "NVIDIA GeForce RTX 5070 Ti" or getattr(spec, "method", None) != "dflash"
                or getattr(spec, "num_speculative_tokens", None) != 5
                or not set(INKLING_ARCHS).intersection(getattr(draft, "architectures", ()) or ())
                or getattr(getattr(model, "hf_config", None), "model_type", None) != "inkling2_loop"
                or not isinstance(source, (str, Path)) or Path(source).name != "Inkling-2-Mini-UD-Q2_K_XL.gguf"):
            return None
        return {**dict.fromkeys(range(1, 7), (1, 16, 64, 4)), 2048: prefill}
    decode_n = 32 if gpu == "NVIDIA GeForce RTX 5080" and ggml_type in (16, 19, 22) else 16
    return {1: (1, decode_n, 64, 4), 2048: prefill}
