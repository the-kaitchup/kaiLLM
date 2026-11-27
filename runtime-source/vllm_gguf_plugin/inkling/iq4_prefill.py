"""Inkling IQ4_NL prefill with the installed Triton arithmetic and larger expert tiles."""
import os

import torch

from ..triton.fused_moe.iq_quant.iq4_nl import iq4_nl_moe_kernel
from ..triton.fused_moe.utils import run_triton_fused_moe_kernel
from ..triton.gemm.iq_quant.iq_tables import get_iq_table_tensors
from ..triton.gemm.utils import GGML_TYPE_IQ4_NL


def supports_down_prefill_tile(config, device):
    """Allow physical qualification of the 32-row IQ4_NL down tile on BF16 CUDA."""
    return (os.getenv('KAILLM_INKLING_IQ4_DOWN_PREFILL') == '1'
            and config is not None
            and config.speculative_config is None
            and config.parallel_config.world_size_across_dp == 1
            and config.model_config.hf_config.model_type == 'inkling2_loop'
            and config.model_config.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(device) >= (8, 0))


def qualified_projection_tiles(config, gpu_name, visible_devices, native_available, input_width, ggml_type=20):
    """Return the measured prefill and decode tiles for this GPU and serving batch size."""
    if (config is None or visible_devices != 1 or native_available or ggml_type not in (8, 20) or input_width not in (1152, 2304)
            or config.speculative_config is not None
            or config.parallel_config.world_size_across_dp != 1):
        return None, None
    slots = config.scheduler_config.max_num_seqs
    if slots != 1:
        from .projection_streams import is_iq4_nl_donor
        if slots not in (2, 4) or gpu_name != 'NVIDIA GeForce RTX 5090' or not is_iq4_nl_donor(getattr(config, 'model_config', None)):
            return None, None
    if gpu_name.endswith('GeForce RTX 5090'):
        if ggml_type == 20 and input_width == 1152:
            from .dondon_5090 import released_tier
            if released_tier(config, gpu_name, visible_devices) is not None:
                # S uses ordinary down projections. ExpertStage selects the measured M prefill separately.
                return None, None
        if ggml_type == 8:
            gate = 32 if slots == 1 else 64
            return None, gate if input_width == 2304 else 16
        gate = 32 if slots == 4 else 64
        return 32, gate if input_width == 2304 else 16
    if gpu_name.endswith('GeForce RTX 3090'):
        return (32, 64) if input_width == 2304 else (16, 16)
    return None, None


def pro_projection_tiles(config, gpu_name, visible_devices, width):
    from .projection_streams import PRO_6000, supported
    if gpu_name != PRO_6000 or width not in (1152, 2304) or not supported(config, gpu_name, visible_devices):
        return None
    return {1: (8, 64, 4) if width == 2304 else (16, 64, 2), 2048: (32, 64, 2)}


def tiled_prefill(x, weight, ids, shared, rows, block_m, sm86_packed=False, *, pro_geometry=None):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    routes = ids.int().contiguous()
    if not shared:
        routes = routes.reshape(-1, 1)
    order, experts, padded = moe_align_block_size(routes, block_m, weight.shape[0])
    # The 2,048-token SM86 screen qualifies two packed layouts with the same BF16 output.
    narrow = sm86_packed and ids.shape[0] == 2048
    table = None if narrow and not shared else get_iq_table_tensors(weight.device)['kvalues_iq4nl']
    tuning = dict(block_n=64, block_k_blocks=2, num_warps=4, num_stages=2) if narrow else {}
    if pro_geometry is not None:
        tuning = dict(block_n=pro_geometry[0], block_k_blocks=pro_geometry[1], ORIGINAL_TABLE=True)
    return run_triton_fused_moe_kernel(
        iq4_nl_moe_kernel, weight, x, order, experts, padded, rows, routes.shape[1], x.shape[0],
        GGML_TYPE_IQ4_NL, extra_args=(table,), block_m=block_m, **tuning,
    )
