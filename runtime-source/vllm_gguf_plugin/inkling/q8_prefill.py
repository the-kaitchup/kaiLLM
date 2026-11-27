"""Qualified RTX 3090 Q8_0 routed expert prefill shapes."""
from ..triton.fused_moe.standard_quant.q8_0 import q8_0_moe_kernel
from ..triton.fused_moe.utils import run_triton_fused_moe_kernel
from ..triton.gemm.utils import GGML_TYPE_Q8_0


def tiled_q8_prefill(x, weight, ids, shared, rows):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    # A 512-token shared projection favors M16. Both 2,048-token
    # projections and the nonshared 512-token projection favor M32.
    block_m = 16 if shared and ids.shape[0] == 512 else 32
    routes = (ids if shared else ids.reshape(-1, 1)).int().contiguous()
    order, experts, padded = moe_align_block_size(routes, block_m, weight.shape[0])
    return run_triton_fused_moe_kernel(
        q8_0_moe_kernel, weight, x, order, experts, padded, rows, routes.shape[1],
        x.shape[0], GGML_TYPE_Q8_0, block_m=block_m, block_n=64,
    )
