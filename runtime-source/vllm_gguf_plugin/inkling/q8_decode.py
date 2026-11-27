"""Single-token BF16 Q8_0 expert projections with one active input row per expert tile."""
from ..triton.fused_moe.standard_quant.q8_0 import q8_0_moe_kernel
from ..triton.fused_moe.utils import run_triton_fused_moe_kernel
from ..triton.gemm.utils import GGML_TYPE_Q8_0


def grouped_q8_decode(x, weights, ids, shared, tile_n, fixed_route):
    order, padded = fixed_route
    return run_triton_fused_moe_kernel(
        q8_0_moe_kernel, weights, x, order, ids.flatten().int(), padded, weights.shape[1],
        ids.shape[1] if shared else 1, x.shape[0], GGML_TYPE_Q8_0, block_m=4, block_n=tile_n,
    )
