# SPDX-License-Identifier: Apache-2.0
"""One-kernel Gemma-style RMSNorm (optional residual add) with an F32 (1 + w) multiplier.

Replicates vLLM's native rms_norm / fused_add_rms_norm arithmetic for 16-bit
activations and an F32 weight (F32 sum with the residual, F32 variance,
x * rsqrt(var + eps) * (1 + w), one rounding to the activation dtype), which
otherwise runs as ~9 eager PyTorch kernels per norm when compilation is off.
KAILLM_FUSED_GEMMA_NORM=0 restores the original forward.
"""

import os
import types

import torch
import triton
import triton.language as tl


def enabled():
    return os.getenv("KAILLM_FUSED_GEMMA_NORM", "1") == "1"


def f32_residual():
    # Keep the residual stream in F32 (as GGML does) instead of the activation dtype.
    return os.getenv("KAILLM_F32_RESIDUAL", "1") == "1"


@triton.jit
def _gemma_norm(X, R, W, Y, ROUT, stride_x, stride_r, stride_y, stride_ro, N, eps,
                HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(ROUT + row * stride_ro + cols, x.to(ROUT.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride_y + cols, (y * w).to(Y.dtype.element_ty), mask=mask)


def gemma_rms_norm(x, residual, weight32, eps):
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)
    if residual is not None:
        r2 = residual.reshape(-1, n)
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
        rout = torch.empty(y.shape, dtype=torch.float32 if f32_residual() else residual.dtype, device=x.device)
    else:
        r2, rout = x2, y
    if x2.shape[0]:
        block = triton.next_power_of_2(n)
        _gemma_norm[(x2.shape[0],)](x2, r2, weight32, y, rout, x2.stride(0), r2.stride(0), y.stride(0),
                                    rout.stride(0), n, eps, HAS_RES=residual is not None, BLOCK=block,
                                    num_warps=8 if block >= 4096 else 4)
    y = y.reshape(x.shape)
    return y if residual is None else (y, rout.reshape(x.shape))


def _forward(self, x, residual=None):
    if (
        not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.shape[-1] != self.weight.shape[0]
        or (residual is not None and (residual.dtype not in (x.dtype, torch.float32) or residual.shape != x.shape))
    ):
        if self.__dict__.get("_kai_np_consumers"):
            from . import native_fused

            native_fused.STASH.pop(id(self), None)
        return self._kai_original_forward(x, residual)
    weight32 = self.__dict__.get("_kai_weight32")
    if weight32 is None or weight32.device != x.device:
        weight32 = (self.weight.detach().float() + 1.0).contiguous()
        self.__dict__["_kai_weight32"] = weight32
    if self.__dict__.get("_kai_np_consumers") and f32_residual() and os.getenv("KAILLM_FUSED_NORM_PREPARE", "1") == "1":
        from . import native_fused

        fused = native_fused.norm_prepare(id(self), x, residual, weight32, self.variance_epsilon)
        if fused is not None:
            y, rout = fused
            return y if residual is None else (y, rout)
    if self.__dict__.get("_kai_np_consumers"):
        from . import native_fused

        native_fused.STASH.pop(id(self), None)
    return gemma_rms_norm(x, residual, weight32, self.variance_epsilon)


def bind(module):
    """Rebind one GemmaRMSNorm instance (CustomOp binds its forward at construction)."""
    if not enabled() or getattr(module, "_kai_original_forward", None) is not None:
        return False
    module._kai_original_forward = module._forward_method
    module._forward_method = types.MethodType(_forward, module)
    return True
