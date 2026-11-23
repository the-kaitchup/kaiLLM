# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache with speculative verification (KAILLM_TQ_SPEC=1; with --kv-cache-dtype turboquant_*).

vLLM's TurboQuant backend runs a multi-token query (a verification of 1 + k tokens) through its prefill path, which
for short continuations loops over requests on the host with Python-side lengths: slow, and baked into a FULL CUDA
graph at capture time (an RTX 3060 DSpine 7 run accepted 1.03 tokens per step).  This routes every multi-token query
batch (verifications and, with KAILLM_TQ_PREFILL_V3=1, the default, prefill chunks too: vLLM's continuation path
dequantizes a max_model_len workspace per chunk and layer) and every decode batch to the SoA TurboQuant unified
attention kernel in its split-KV form, launched directly with a 64-row query block (16 for decode), Q rotated outside
the kernel, and a fixed power-of-2 number of KV segments, so the launch is the same at capture and at replay.  The
backend writes the SoA layout that kernel reads (_soa_store = True; vLLM enables it only with AMD's FlyDSL decode).
Also: KAILLM_TQ_SLOT_ALIGN (page size with BF16 skip layers) and the SoA FP8 key store fix for Ampere.
"""
from __future__ import annotations

import os
import sys

import torch

MAX_VERIFY = 16


def _segments(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(idx).multi_processor_count
    n = int(os.getenv("KAILLM_TQ_SEGMENTS", "0")) or max(4, sms // 4)
    return triton_next_pow2(n)          # vLLM's reduce_segments needs a power of 2


def install() -> bool:
    if os.getenv("KAILLM_TQ_SPEC", "0") != "1":
        return False
    try:
        from vllm.v1.attention.backends import turboquant_attn as tq
        from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
        from vllm.v1.attention.ops.turboquant_soa import triton_turboquant_unified_attention as v3
    except ImportError:
        return False
    Impl = tq.TurboQuantAttentionImpl
    if getattr(Impl, "_kai_tq_spec", False):
        return True
    _slot_alignment()
    _fp8_store_fix()
    original_init, original_prefill = Impl.__init__, Impl._prefill_attention
    reported = []

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._soa_store = True

    def prefill(self, query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT=None, layer=None):
        # every multi-token query batch (verifications and prefill chunks; the chunk's own K/V are already in the
        # cache): vLLM's continuation path dequantizes a max_model_len workspace per chunk and layer (a 3k-token
        # prompt took 6.5 minutes on an RTX 3060 with a 115k context limit)
        if attn_metadata.max_query_len > 1 and os.getenv("KAILLM_TQ_PREFILL_V3", "1") == "1" or \
                1 < attn_metadata.max_query_len <= MAX_VERIFY:
            if not reported:
                reported.append(1)
                print(f"kaiLLM: TurboQuant verification batches ({attn_metadata.max_query_len} tokens) on the "
                      f"split-KV SoA kernel", file=sys.stderr, flush=True)
            return tq_verify(self.tq_config, self.scale, getattr(self, "sinks", None), query, kv_cache,
                             attn_metadata, Pi, centroids, PiT)
        return original_prefill(self, query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer=layer)

    original_decode = Impl._decode_attention

    def decode(self, query, kv_cache, attn_metadata, Pi, centroids, PiT=None, layer=None):
        # one-token decode batches: vLLM's SoA decode wrapper fuses the Q rotation (256 KB of shared memory for
        # 4-bit keys, over the sm_86 / sm_120 limit); same split-KV launch with 16-row query blocks
        if self._soa_store and not self._use_flydsl:
            return tq_verify(self.tq_config, self.scale, getattr(self, "sinks", None), query, kv_cache, attn_metadata,
                             Pi, centroids, PiT, block_m=16)
        return original_decode(self, query, kv_cache, attn_metadata, Pi, centroids, PiT, layer)

    Impl.__init__, Impl._prefill_attention, Impl._decode_attention = init, prefill, decode
    Impl._kai_tq_spec = True
    return True


def tq_verify(tqc, scale, sinks, query, kv_cache, md, Pi, centroids, PiT, block_m=None):
    """Split-KV SoA TurboQuant attention for a batch of 2-16-token queries (md: query_start_loc, seq_lens,
    block_table)."""
    from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
    from vllm.v1.attention.ops.turboquant_soa import triton_turboquant_unified_attention as v3

    N, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    qpk = Hq // Hk
    key_fp8 = tqc.key_fp8
    mse_bits = tqc.key_mse_bits                 # 0 for FP8 keys, as the backend's own launches
    cfg = v3._get_layout(D, mse_bits, tqc.effective_value_quant_bits)
    vqb = tqc.effective_value_quant_bits
    if key_fp8:
        q_in, pit, pit_s0, pit_s1, fuse = query.contiguous(), centroids, 0, 0, 0
    else:
        # rotate Q here (one small GEMM): the fused prologue stages the 256 x 256 FP32 rotation in shared memory
        # (256 KB, over the 99 KB per-block limit of sm_86 / sm_120)
        pit = (PiT if PiT is not None else Pi.T).float()
        q_in = (query.float().reshape(-1, D) @ pit).to(query.dtype).reshape(query.shape).contiguous()
        pit, pit_s0, pit_s1, fuse = centroids, 0, 0, 0
    use_pair_lut = (not key_fp8) and mse_bits == 4
    pair_lut = v3._get_pair_lut(centroids) if use_pair_lut else centroids
    block_m = block_m or int(os.getenv("KAILLM_TQ_BLOCK_M", "64"))
    block_m = max(block_m, triton_next_pow2(qpk))
    block_q = block_m // qpk
    tile = int(os.getenv("KAILLM_TQ_TILE", "16"))
    num_seqs = int(md.query_start_loc.shape[0] - 1)
    total_q_blocks = N // block_q + num_seqs
    S = _segments(query.device)
    hp = cfg["BLOCK_D"]
    segm_o = torch.empty((N, Hq, S, hp), dtype=torch.float32, device=query.device)
    segm_m = torch.empty((N, Hq, S), dtype=torch.float32, device=query.device)
    segm_e = torch.empty((N, Hq, S), dtype=torch.float32, device=query.device)
    out = torch.empty_like(query)
    key_data_bytes = D if key_fp8 else cfg["mse_bytes"]
    data_bytes = key_data_bytes + cfg["val_data_bytes"]
    sinks_f32 = sinks.float().contiguous() if sinks is not None else centroids
    v3.kernel_tq_unified_attention_3d[(total_q_blocks, Hk, S)](
        segm_output_ptr=segm_o, segm_max_ptr=segm_m, segm_expsum_ptr=segm_e, query_ptr=q_in,
        KV_cache_ptr=kv_cache, KV_cache_u16_ptr=kv_cache.view(torch.uint16), Centroids_ptr=centroids,
        Pair_lut_ptr=pair_lut, PiT_ptr=pit, block_tables_ptr=md.block_table, seq_lens_ptr=md.seq_lens,
        query_start_len_ptr=md.query_start_loc, sinks_ptr=sinks_f32, scale=scale, num_query_heads=Hq,
        num_queries_per_kv=qpk, block_table_stride=md.block_table.stride(0), query_stride_0=q_in.stride(0),
        query_stride_1=q_in.stride(1), stride_cache_block=kv_cache.stride(0), pit_stride_0=pit_s0,
        pit_stride_1=pit_s1, BLOCK_SIZE=block_size, TILE_SIZE=tile, HEAD_SIZE=D, HEAD_SIZE_PADDED=hp,
        BLOCK_Q=block_q, BLOCK_M=block_m, num_seqs=num_seqs, NUM_SEGMENTS_PER_SEQ=S, MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"], VQB=vqb, VAL_DATA_BYTES=cfg["val_data_bytes"],
        N_CENTROIDS=int(centroids.numel()), KEY_FP8=1 if key_fp8 else 0, USE_PAIR_LUT=1 if use_pair_lut else 0,
        NUM_KV_HEADS=Hk, KEY_DATA_BYTES=key_data_bytes, META_REGION_OFFSET=block_size * Hk * data_bytes,
        NUM_SOA_FIELDS=2 if key_fp8 else 3, SOA_K_NORM=0, SOA_V_SCALE=0 if key_fp8 else 1,
        SOA_V_ZERO=1 if key_fp8 else 2, NORM_CORRECTION=1 if tqc.norm_correction else 0,
        FP8_E4B15=v3._use_fp8_e4b15(query.device.index or 0), FUSE_Q_ROT=fuse,
        USE_SINKS=1 if sinks is not None else 0, num_warps=int(os.getenv("KAILLM_TQ_WARPS", "4")),
        num_stages=int(os.getenv("KAILLM_TQ_STAGES", "1")))
    reduce_segments[(N, Hq)](
        output_ptr=out, segm_output_ptr=segm_o, segm_max_ptr=segm_m, segm_expsum_ptr=segm_e,
        seq_lens_ptr=md.seq_lens, num_seqs=num_seqs, num_query_heads=Hq, out_scale_inv=1.0,
        output_stride_0=out.stride(0), output_stride_1=out.stride(1), block_table_stride=md.block_table.stride(0),
        TILE_SIZE=tile, HEAD_SIZE=D, HEAD_SIZE_PADDED=hp, query_start_len_ptr=md.query_start_loc,
        BLOCK_Q=block_q, NUM_SEGMENTS_PER_SEQ=S, USE_FP8=False)
    return out


def _slot_alignment() -> None:
    """KAILLM_TQ_SLOT_ALIGN=N: round the packed slot (bytes per token and KV head) up to a multiple of N.

    With BF16 skip layers (the DSpine drafter) sharing the block pool, vLLM sizes the page from
    lcm(TurboQuant page, BF16 page) with 16-token blocks: Qwen3.8 K4V4 (262-byte slots) -> an 8.58 MiB page, the GDN
    state (3.06 MiB) padded by 156 %.  272-byte slots (16-aligned) allow a 3.19 MiB page.  The SoA kernels address a
    block through its stride and lay data / metadata out from byte counts, so the pad is unused space."""
    align = int(os.getenv("KAILLM_TQ_SLOT_ALIGN", "0"))
    if align <= 1:
        return
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig

    TurboQuantConfig.slot_size_aligned = property(lambda self: -(-self.slot_size // align) * align)


def _fp8_store_fix() -> None:
    """The SoA store casts BF16 keys straight to FP8 (Triton rejects bf16 -> fp8e4b15 on Ampere); the AoS store goes
    through float32.  Recompile the SoA kernel with the float32 step."""
    import inspect

    from vllm.v1.attention.ops.turboquant_soa import triton_turboquant_store as st

    fn = getattr(st, "_tq_fused_store_fp8", None)
    if fn is None or getattr(st, "_kai_fp8_fix", False):
        return
    src = inspect.getsource(getattr(fn, "fn", fn))
    old = "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)\n"
    if src.count(old) != 1:
        return
    src = src.replace(old, "k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)\n")
    exec(compile(src, st.__file__, "exec"), st.__dict__)
    st._kai_fp8_fix = True


def triton_next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p
