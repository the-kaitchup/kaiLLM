# SPDX-License-Identifier: Apache-2.0
"""Packed YAQA execution, retaining the checkpoint's original bitstreams.

QTIP runs in its Hadamard basis. This avoids reconstructing a dense matrix,
but does not reproduce the final per-weight BF16 rounding of a dense export.
The canonical reconstruction below is provided for numerical validation.
"""

import math
import os

import torch


def hadamard_reference(x, factor, transpose=False):
    """Apply the checkpoint's orthogonal transform in the input precision."""
    n = x.shape[-1]
    k = factor.shape[0]
    p = n // k
    if n % k or p < 1 or p & (p - 1):
        raise ValueError("Invalid YAQA Hadamard factorization")
    value = x.reshape(-1, k, p)
    width = 1
    while width < p:
        grouped = value.reshape(-1, k, p // (2 * width), 2, width)
        a, b = grouped[..., 0, :], grouped[..., 1, :]
        value = torch.stack((a + b, a - b), dim=-2).reshape(-1, k, p)
        width *= 2
    if k > 1:
        h = factor.T if transpose else factor
        value = h.to(device=x.device, dtype=x.dtype) @ value
    return (value * (n**-0.5)).reshape(x.shape)


def decode_qtip_reference(components, shape, bits, permutation=True):
    """Decode circular L16/V2 QTIP windows, including crossing K3/K5 windows."""
    m, n = shape
    packed = components["trellis"].to(torch.int64) & 65535
    if tuple(packed.shape) != (m * n // 256, 16 * bits):
        raise ValueError("YAQA trellis size differs from its matrix shape")
    start = torch.arange(128, device=packed.device) * (2 * bits)
    word, shift = start // 16, start % 16
    states = (
        ((packed[:, word] << 16) | packed[:, (word + 1) % (16 * bits)]) >> (16 - shift)
    ) & 65535
    hashed = states * (states + 1)
    indices = (hashed >> 6) & 511
    values = components["tlut"].to(packed.device)[indices].float()
    values[..., 0] *= 1 - ((hashed >> 15) & 1) * 2
    values = values.reshape(-1, 256)
    if permutation:
        permute = (
            torch.arange(256, device=packed.device)
            .reshape(2, 8, 2, 4, 2)
            .permute(1, 3, 2, 0, 4)
            .flatten()
        )
        values = values[:, torch.argsort(permute)]
    return values.reshape(m // 16, n // 16, 16, 16).transpose(1, 2).reshape(m, n)


def reconstruct_yaqa_reference(descriptor, components, dtype=torch.bfloat16):
    """Reconstruct canonical dense weights for tests and model evaluation."""
    shape = tuple(descriptor["shape"])
    codec = descriptor["codec"]
    if codec == "qtip":
        value = decode_qtip_reference(
            components,
            shape,
            descriptor["bits"],
            descriptor.get("tensor_core_permutation", True),
        ).double()
        value *= components["Wscale"].double().reshape(())
        value = hadamard_reference(value, components["hadamard_input"])
        value *= components["SU"].double()
        value = hadamard_reference(value.T, components["hadamard_output"])
        value = (value * components["SV"].double()).T.contiguous().float()
        if "post_output_scale" in components:
            value *= components["post_output_scale"].float()[:, None]
        if "post_input_scale" in components:
            value *= components["post_input_scale"].float()[None, :]
    elif codec == "ternary":
        count = math.prod(shape)
        packed = components["packed"].view(torch.uint8).long().flatten()
        powers = torch.tensor([1, 3, 9, 27, 81], device=packed.device)
        codes = ((packed[:, None] // powers) % 3).flatten()[:count] - 1
        scales = components.get("post_output_scale", components["row_scales"])
        value = (codes.reshape(shape).double() * scales.double()[:, None]).float()
    elif codec == "scalar-int4":
        packed = components["packed"].view(torch.uint8).flatten()
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten()
        codes = codes[: math.prod(shape)].reshape(shape).float() - 8
        value = codes * components["scales"].float()[:, None]
    else:
        raise ValueError(f"Unsupported YAQA codec: {codec}")
    return value.to(dtype)


class PackedYAQALinear(torch.nn.Module):
    """Batched YAQA linear; only activations and small transforms are expanded."""

    def __init__(
        self,
        descriptor,
        components,
        dtype=torch.bfloat16,
        device=None,
        *,
        prefill_cache_plan=None,
        prefill_cache_key=None,
        index40_plan=None,
        index40_cache_key=None,
    ):
        super().__init__()
        self.shape = tuple(descriptor["shape"])
        self.codec = descriptor["codec"]
        self.bits = descriptor["bits"]
        self.permutation = descriptor.get("tensor_core_permutation", True)
        self.dtype = dtype
        self.prefill_cache_plan = prefill_cache_plan if self.codec == "qtip" else None
        self.prefill_cache_key = prefill_cache_key
        self.prefill_cache_identity = None
        opaque = os.getenv("VLLM_GGUF_YAQA_OPAQUE", "0")
        if opaque not in ("0", "1"):
            raise ValueError("VLLM_GGUF_YAQA_OPAQUE must be0or1")
        self.opaque_dispatch = opaque == "1"
        self.index40_plan = index40_plan
        self.index40_cache_key = index40_cache_key
        self.index40_identity = None
        self.cuda_index40_key = None
        from .yaqa_prefill import prefill_mode

        self.prefill_backend = prefill_mode()
        self.transient_prefill_eligible = False
        from .yaqa_power import power_hadamard_mode

        self.hadamard_backend = power_hadamard_mode()
        from .yaqa_sm120_b1_policy import fused_post_enabled

        self.fuse_k1_post = (
            fused_post_enabled() and os.getenv("VLLM_GGUF_YAQA_CUDA_ARCH") == "sm120"
        )
        self.cuda_power_key = None
        self.gemv_backend = os.getenv("VLLM_GGUF_YAQA_GEMV", "triton")
        if self.gemv_backend not in ("triton", "cuda-k2", "cuda-k12", "cuda-k1234"):
            raise ValueError(
                "VLLM_GGUF_YAQA_GEMV must be triton, cuda-k2, cuda-k12 or cuda-k1234"
            )
        self.cuda_k1_key = None
        self.cuda_k2_key = None
        self.cuda_k34_key = None
        self.cuda_latent_key = None
        self.cuda_mma_key = None
        self.cuda_residual_mma_key = None
        self.cuda_ternary_key = None
        self.scalar_backend = os.getenv("VLLM_GGUF_YAQA_SCALAR", "triton")
        if self.scalar_backend not in ("triton", "cuda"):
            raise ValueError("VLLM_GGUF_YAQA_SCALAR must be triton or cuda")
        validation = os.getenv("VLLM_GGUF_YAQA_VALIDATE_B1", "0")
        if validation not in ("0", "1"):
            raise ValueError("VLLM_GGUF_YAQA_VALIDATE_B1 must be 0 or 1")
        self.validate_b1 = validation == "1"
        self.gemm_precision = os.getenv("VLLM_GGUF_YAQA_GEMM", "tf32x3")
        if self.gemm_precision not in ("tf32x3", "native-fp16", "native-fp16-mma"):
            raise ValueError(
                "VLLM_GGUF_YAQA_GEMM must be tf32x3, native-fp16 or native-fp16-mma"
            )
        for name, value in components.items():
            self.register_buffer(name, value.detach().to(device=device).contiguous())
        device = next(self.buffers()).device
        self._native_device_capability = (
            tuple(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        )
        from .yaqa_factor_storage import prepare_layer as prepare_factor_storage

        prepare_factor_storage(self, components)
        from .yaqa_sm120_b1_policy import k1_post_power_resource

        self.k1_post_power_num_warps = k1_post_power_resource(
            self.fuse_k1_post and self.codec == "qtip",
            self._native_device_capability,
            self.shape[0],
            getattr(getattr(self, "hadamard_output", None), "shape", ()),
            self.dtype,
            getattr(getattr(self, "Wscale", None), "dtype", None),
        )
        if self.opaque_dispatch:
            from .yaqa_opaque import register_layer

            self._opaque_key = register_layer(self, descriptor)
        # The existing prefill kernel always loads this factor as F32.
        factor_cpu = components.get("hadamard_input")
        if (
            self.codec == "qtip"
            and self._native_device_capability == (8, 6)
            and self.shape[1] == 7680
            and factor_cpu is not None
            and factor_cpu.device.type == "cpu"
            and factor_cpu.dtype == torch.float64
            and factor_cpu.shape == (60, 60)
            and bool(torch.isfinite(factor_cpu).all())
            and torch.equal(factor_cpu, factor_cpu.float().double())
        ):
            self.register_buffer(
                "hadamard_input_prefill",
                factor_cpu.float().to(device=device).contiguous(),
            )
        if self.codec == "qtip":
            if (
                self.hadamard_backend == "cuda"
                and self.shape[1] == 2048
                and self.hadamard_input.shape[0] == 1
                and self.trellis.device.type == "cuda"
            ):
                from .yaqa_power import load_cuda_power

                self.cuda_power_key = load_cuda_power(self.trellis.device)
            if (
                self.gemv_backend in ("cuda-k2", "cuda-k12", "cuda-k1234")
                and self.bits == 2
                and self.permutation
                and self.tlut.dtype == torch.bfloat16
                and self.trellis.device.type == "cuda"
            ):
                from .yaqa_cuda import load_cuda_k2

                self.cuda_k2_key = load_cuda_k2(self.trellis.device)
            if (
                self.gemv_backend in ("cuda-k12", "cuda-k1234")
                and self.bits == 1
                and self.permutation
                and self.tlut.dtype == torch.bfloat16
                and self.trellis.device.type == "cuda"
            ):
                from .yaqa_cuda_k1 import load_cuda_k1

                self.cuda_k1_key = load_cuda_k1(self.trellis.device)
            if (
                self.gemv_backend == "cuda-k1234"
                and self.bits in (3, 4)
                and self.permutation
                and self.tlut.dtype == torch.bfloat16
                and self.trellis.device.type == "cuda"
            ):
                from .yaqa_cuda_k34 import load_cuda_k34

                self.cuda_k34_key = load_cuda_k34(self.trellis.device, self.bits)
            self.native_fp16_gemm = (
                self.gemm_precision in ("native-fp16", "native-fp16-mma")
                and self.tlut.dtype == torch.bfloat16
                and torch.equal(
                    components["tlut"], components["tlut"].half().bfloat16()
                )
            )
            self.transient_prefill_eligible = (
                self.prefill_backend in ("transient", "cuda-transient")
                and self.native_fp16_gemm
                and bool(torch.isfinite(components["tlut"]).all().item())
            )
            if (
                self.gemm_precision == "native-fp16-mma"
                and self.native_fp16_gemm
                and self.permutation
                and 1 <= self.bits <= 4
                and min(self.shape) >= 32
                and self.trellis.device.type == "cuda"
                and torch.isfinite(components["tlut"]).all().item()
            ):
                from .yaqa_mma import load_register_mma

                self.cuda_mma_key = load_register_mma(self.trellis.device)
                if self.cuda_mma_key is not None:
                    signed = self.tlut.half().repeat(2, 1)
                    signed[512:, 0].neg_()
                    self.register_buffer("mma_lut", signed.contiguous())
                    if os.getenv("VLLM_GGUF_YAQA_REGISTER_PRECISION", "none") != "none":
                        from .yaqa_residual_mma import load as load_residual_mma

                        self.cuda_residual_mma_key = load_residual_mma(
                            self.trellis.device, self.bits, self.cuda_mma_key
                        )
            if self.prefill_backend == "cuda-transient":
                from .yaqa_latent_cuda import decoder_warps, load_decoder

                if decoder_warps(self) is not None:
                    self.cuda_latent_key = load_decoder(self.trellis.device)
            for axis, sign, post in (
                ("input", "SU", "post_input_scale"),
                ("output", "SV", "post_output_scale"),
            ):
                scale = getattr(self, sign).float()
                if hasattr(self, post):
                    scale = scale * getattr(self, post).float()
                self.register_buffer(axis + "_scale", scale)
        else:
            self.native_fp16_gemm = False
            if (
                self.scalar_backend == "cuda"
                and self.codec == "ternary"
                and self.dtype == torch.bfloat16
                and self.shape[1] in (2048, 7680)
                and self.packed.device.type == "cuda"
            ):
                from .yaqa_scalar import load_cuda_ternary

                self.cuda_ternary_key = load_cuda_ternary(self.packed.device)

        from .yaqa_prefill_cache import prepare_layer

        prepare_layer(
            self, descriptor, components, self.prefill_cache_plan, prefill_cache_key
        )
        from .yaqa_index40 import prepare_layer as prepare_index40

        prepare_index40(self, descriptor, components, index40_plan, index40_cache_key)

    def forward(self, x, bias=None):
        if x.shape[-1] != self.shape[1]:
            raise ValueError("YAQA input width differs from the packed matrix")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.shape[0]))
        from .yaqa_prefill_cache import apply as apply_prefill_cache
        from .yaqa_prefill_cache import selected

        if self.opaque_dispatch and x.device.type == "cuda":
            from .yaqa_opaque import opaque_linear

            out = opaque_linear(
                x, list(self.buffers()), self._opaque_key, self.shape[0]
            )
        elif selected(x, self):
            out = apply_prefill_cache(x, self)
        elif x.device.type != "cuda":
            components = dict(self.named_buffers())
            if self.codec == "qtip":
                value = hadamard_reference(
                    x.float() * self.input_scale, self.hadamard_input, transpose=True
                )
                weight = decode_qtip_reference(
                    components, self.shape, self.bits, self.permutation
                )
                value = value @ weight.T
                value = hadamard_reference(value, self.hadamard_output)
                value = value * self.output_scale * self.Wscale.float()
                out = value.to(x.dtype)
            else:
                weight = reconstruct_yaqa_reference(
                    {"codec": self.codec, "shape": self.shape}, components, self.dtype
                )
                out = (x.float() @ weight.float().T).to(x.dtype)
        else:
            from .yaqa_kernels import packed_yaqa_linear

            out = packed_yaqa_linear(x, self)
        return out if bias is None else out + bias
