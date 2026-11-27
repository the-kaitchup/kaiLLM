# SPDX-License-Identifier: Apache-2.0
"""Optional IQ gate/up tiles for the published Inkling Dondon M GGUF on PRO 6000."""
import os

import torch

from .dondon_5090 import released_gguf_tier


_PREFILL_TILES = {18: (16, 128), 22: (16, 64)}
_ROW_BYTES = {18: 9 * 98, 22: 9 * 82}
_VERIFY_WIDTHS = {18: (1, 4, 6), 22: (1,)}
_WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def iq_gate_tuning(config, gpu, visible, quant_type, row_bytes, device):
    prefill = os.getenv("KAILLM_INKLING_DONDON_IQ_PREFILL") == "1"
    verify = os.getenv("KAILLM_INKLING_DONDON_IQ_VERIFY") == "1"
    if (not (prefill or verify) or config is None or gpu != _WORKSTATION or visible != 1
            or quant_type not in _ROW_BYTES or row_bytes != _ROW_BYTES[quant_type]):
        return None, {}
    scheduler, model, spec = config.scheduler_config, config.model_config, config.speculative_config
    if (config.parallel_config.world_size_across_dp != 1
            or (scheduler.max_num_seqs, scheduler.max_num_batched_tokens) != (1, 512)
            or model.max_model_len != 73728 or model.dtype != torch.bfloat16
            or getattr(model.hf_config, "model_type", None) != "inkling2_loop"
            or getattr(spec, "method", None) != "dflash" or config.num_speculative_tokens != 5):
        return None, {}
    weights = getattr(model, "model_weights", None) or model.model
    if released_gguf_tier(weights) != "M":
        return None, {}
    routes = {}
    if verify:
        for width in _VERIFY_WIDTHS[quant_type]:
            count = width * 8
            routes[width] = (
                torch.arange(count, dtype=torch.int32, device=device),
                torch.full((1,), count, dtype=torch.int32, device=device),
            )
    return _PREFILL_TILES[quant_type] if prefill else None, routes


def iq_gate_projection(x, data, ids, quant_type, rows, prefill_tile, direct_routes):
    if quant_type == 18:
        from ..triton.fused_moe.iq_quant.iq3_xxs import ggml_moe_iq3_xxs_triton as project
    else:
        from ..triton.fused_moe.iq_quant.iq2_s import ggml_moe_iq2_s_triton as project

    tokens, top_k = ids.shape
    if direct_routes is not None:
        order, padded = direct_routes
        return project(x, data, order, ids.flatten(), padded, rows, top_k, tokens, block_m=1, block_n=128)

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    block_m, block_n = prefill_tile
    aligned = moe_align_block_size(ids, block_m, data.shape[0])
    return project(x, data, *aligned, rows, top_k, tokens, block_m=block_m, block_n=block_n)
