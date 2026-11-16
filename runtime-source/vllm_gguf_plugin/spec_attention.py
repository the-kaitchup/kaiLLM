# SPDX-License-Identifier: Apache-2.0
"""Split-KV attention for speculative verification steps (opt-in: KAILLM_SPEC_ATTENTION_3D=1).

vLLM's Triton unified attention runs a verification step (1 < query length) through its split-KV ("3D") kernel only
up to MAX_3D_QUERY_LEN = 8 query tokens, with query blocks of BLOCK_M = 16 rows of (query token x query head of the
KV group).  With 6 query heads per KV head (Qwen3.5 / 3.8) a block holds 2 query tokens, and a 16-token DSpine
verification (above the limit) runs the 2-D kernel, 8 programs per KV head walking the whole context each (~4.5 ms
per layer at 100k tokens).  This raises the limit to 16 (the backend sizes its split-K scratch from it) and picks the
query block, KV tile, warps and stages per verification size (_spec_config).  The kernel is unchanged.
Tuning: KAILLM_SPEC_ATTENTION_BLOCK_M / _TILE / _WARPS / _STAGES force one configuration for every verification.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import textwrap

logger = logging.getLogger(__name__)

SPEC_MAX_QUERY_LEN = 16
_SM120 = False
_ENV_FORCE = None
_FORCE = None          # (BLOCK_M, KV tile, warps, stages) for every verification batch: tuning only


def _env_force():
    names = ("BLOCK_M", "TILE", "WARPS", "STAGES")
    vals = [os.getenv(f"KAILLM_SPEC_ATTENTION_{n}") for n in names]
    return None if vals[0] is None else tuple(int(v or d) for v, d in zip(vals, ("64", "64", "8", "0")))


def _spec_config(max_seqlen_q: int, num_queries_per_kv: int):
    """(BLOCK_M, KV tile, warps, stages; 0 = Triton's default) for a verification batch, None = vLLM's choice.

    Tuned on sm_120 (RTX PRO 6000) for Qwen3.8 attention (6 query heads per KV head, head 256) at 100k context, per
    layer, rows = query tokens x query heads per KV head: <= 24 rows (4-token MTP verification) vLLM's 16-row blocks
    are the fastest (342 us); <= 48 rows, 32-row blocks over 64-token KV tiles (378 us vs 405 us); up to 96 rows
    (16-token DSpine verification), 64-row blocks, 64-token tiles, 8 warps, 1 stage (518 us; 16-row blocks 693 us,
    64-row blocks over decode's 16-token tiles 856 us: the 64 x 256 FP32 accumulator is rescaled once per tile).
    Other GPUs take 32-token tiles and 1 stage (the same shared-memory budget, untuned)."""
    force = _FORCE or _ENV_FORCE
    if force is not None:
        return force
    if num_queries_per_kv > 16:
        return None
    rows = max_seqlen_q * num_queries_per_kv
    if rows <= 24:
        return None
    tile = 64 if _SM120 else 32
    if rows <= 48:
        return (32, tile, 4, 0 if _SM120 else 1)
    return (64, tile, 8, 1)


_BLOCK_M = (
    "    BLOCK_M = (\n"
    "        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)\n"
    "    )\n"
)
_SPEC_BLOCK_M = (
    "    _kai_spec = (_kai_spec_config(max_seqlen_q, num_queries_per_kv)\n"
    "                 if 1 < max_seqlen_q <= MAX_3D_QUERY_LEN else None)\n"
    "    if _kai_spec is not None:\n"
    "        BLOCK_M = _kai_spec[0]   # kaiLLM: verification query block sized to the request\n"
)
_WARPS = "    launch_num_stages: int | None = None\n"
_SPEC_WARPS = (
    "    if _kai_spec is not None:\n"
    "        launch_num_warps = _kai_spec[2] or None\n"
    "        launch_num_stages = _kai_spec[3] or None\n"
)
_TILE = "    if tuned_large_head:\n        TILE_SIZE_PREFILL = 128\n"
_SPEC_TILE = (
    "    if _kai_spec is not None:\n"
    "        TILE_SIZE_DECODE = _kai_spec[1]   # kaiLLM: wider KV tiles for many-row verification blocks\n"
)


def install() -> bool:
    if os.getenv("KAILLM_SPEC_ATTENTION_3D", "0") != "1":
        return False
    from vllm.v1.attention.ops import triton_unified_attention as ops

    if getattr(ops, "_kai_spec_attention", False):
        return True
    src = textwrap.dedent(inspect.getsource(ops.unified_attention))
    if (src.count(_BLOCK_M) != 1 or src.count(_WARPS) != 1 or src.count(_TILE) != 1
            or "def unified_attention(" not in src):
        logger.warning("kaiLLM spec attention: unexpected vLLM unified_attention source, not patched")
        return False
    src = (src.replace(_BLOCK_M, _BLOCK_M + _SPEC_BLOCK_M).replace(_WARPS, _WARPS + _SPEC_WARPS)
           .replace(_TILE, _TILE + _SPEC_TILE))
    global _SM120, _ENV_FORCE
    from vllm.platforms import current_platform     # NVML: no CUDA context in the API / engine parent processes

    _SM120 = current_platform.is_cuda() and current_platform.is_device_capability((12, 0))
    _ENV_FORCE = _env_force()
    ops.MAX_3D_QUERY_LEN = SPEC_MAX_QUERY_LEN
    ops._kai_spec_config = _spec_config
    exec(compile(src, ops.__file__, "exec"), ops.__dict__)      # redefines ops.unified_attention, module globals
    ops._kai_spec_attention = True
    backend = sys.modules.get("vllm.v1.attention.backends.triton_attn")
    if backend is not None:                                      # imported before the plugin: patch its bindings
        backend.unified_attention = ops.unified_attention
        backend.MAX_3D_QUERY_LEN = SPEC_MAX_QUERY_LEN
    from . import verify_attn

    if verify_attn.enabled():                                    # CUDA split-KV kernel for batch-1 verifications
        verify_attn.wrap(ops)
    return True
