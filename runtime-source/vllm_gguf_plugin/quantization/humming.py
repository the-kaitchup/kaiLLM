# SPDX-License-Identifier: Apache-2.0
"""One-time GGUF repacking followed by Humming fused low-bit dense GEMM.

Humming loads packed integer weights and performs conversion inside its Tensor
Core kernel. Forward execution never materializes a dense floating-point weight.
Activations remain FP16/BF16; no activation quantization is introduced here.
"""

import importlib.metadata
import json

import numpy as np
import torch
from packaging.version import Version

from .gguf_repack import FORMATS, repack_gguf

MIN_HUMMING_VERSION = "0.1.18"
SUPPORTED_QUANT_TYPES = frozenset(FORMATS)


def _cast_metadata(values: np.ndarray, dtype: torch.dtype, name: str) -> torch.Tensor:
    """Reject metadata range loss before creating device tensors."""
    if np.any(np.abs(values) > torch.finfo(dtype).max):
        raise ValueError(f"GGUF {name} metadata overflows {dtype}")
    converted = torch.from_numpy(values).to(dtype=dtype)
    if name == "weight_scale" and np.any(
        (values != 0) & (converted.float().numpy() == 0)
    ):
        raise ValueError(
            f"GGUF weight_scale metadata underflows {dtype}; "
            "use a dtype with sufficient scale range, such as bfloat16"
        )
    return converted


def humming_available() -> bool:
    """Check the optional dependency without importing CUDA code on CPU."""
    try:
        version = importlib.metadata.version("humming-kernels")
    except importlib.metadata.PackageNotFoundError:
        return False
    return Version(version) >= Version(MIN_HUMMING_VERSION)


class GGUFHummingLinear(torch.nn.Module):
    """Prepared Humming GEMM state, registered for device and memory accounting.

    Construct through :func:`prepare_gguf_humming`. Every matrix has independent
    Stream-K locks, including matrices in the same merged vLLM layer.
    """

    def __init__(self, layer_config, tensors, *, shape, dtype, compute_config):
        super().__init__()
        from vllm.utils.humming import create_humming_locks, humming_forward

        self.layer_config = layer_config
        self.shape = shape
        self.dtype = dtype
        self.compute_config = compute_config
        self._humming_forward = humming_forward
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)
        self.register_buffer(
            "locks", create_humming_locks(tensors["weight"].device), persistent=False
        )

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.shape[0]), dtype=self.dtype)
        output = self._humming_forward(
            self.layer_config,
            inputs=x.reshape(-1, x.shape[-1]),
            weight=self.weight,
            weight_scale=self.weight_scale,
            zero_point=getattr(self, "zero_point", None),
            locks=self.locks,
            compute_config=self.compute_config,
        )
        # Humming's output epilogue requires N to be a multiple of eight.
        # Preparation adds zero rows when needed; expose only the GGUF rows.
        output = output[..., : self.shape[0]].reshape(*x.shape[:-1], self.shape[0])
        if bias is not None:
            output = output + bias
        return output


def prepare_gguf_humming(
    data: torch.Tensor,
    quant_type: int,
    dtype: torch.dtype,
    *,
    chunk_rows: int = 128,
) -> GGUFHummingLinear:
    """Decode metadata once, repack integer bits, and prepare Humming GEMM.

    ``data`` must contain only actual row bytes, with any storage padding
    removed. Tensor-parallel slicing and Qwen projection permutations must have
    happened before this function. Shape padding required by Humming is handled
    by its transform routine. Inference consumes only the transformed tensors.

    Errors are deliberate: an unsupported or numerically invalid conversion
    must not silently fall back to a different quantization/activation scheme.
    """
    if int(quant_type) not in SUPPORTED_QUANT_TYPES:
        raise ValueError(f"Unsupported GGUF Humming type: {quant_type}")
    if data.dtype != torch.uint8 or data.ndim != 2:
        raise ValueError("Humming GGUF input must be a two-dimensional uint8 tensor")
    if not data.is_cuda or torch.version.hip is not None:
        raise ValueError("GGUF Humming acceleration requires an NVIDIA CUDA device")
    if torch.cuda.get_device_capability(data.device) < (7, 5):
        raise ValueError("GGUF Humming acceleration requires NVIDIA SM75 or newer")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("GGUF Humming requires FP16 or BF16 activations")
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability(data.device) < (
        8,
        0,
    ):
        raise ValueError("BF16 Humming activation requires NVIDIA SM80 or newer")
    if not humming_available():
        raise ImportError(
            f"GGUF Humming requires humming-kernels>={MIN_HUMMING_VERSION}"
        )

    from vllm import envs
    from vllm.utils.humming import (
        HummingInputSchema,
        HummingWeightSchema,
        InputQuantizationMode,
        prepare_and_transform_humming_tensors,
    )

    packed = repack_gguf(
        data.detach().cpu().numpy(), int(quant_type), chunk_rows=chunk_rows
    )
    # Check on CPU once. A nonzero scale that rounds to zero would silently
    # remove a group's weights, especially IQ1 or very small MXFP4 scales.
    scale = _cast_metadata(packed.scales, dtype, "weight_scale")
    zero_point = (
        _cast_metadata(packed.zero_points, dtype, "zero_point")
        if packed.zero_points is not None
        else None
    )
    tensors = {
        "weight": torch.from_numpy(packed.weight).to(device=data.device),
        "weight_scale": scale.to(device=data.device),
    }
    if zero_point is not None:
        tensors["zero_point"] = zero_point.to(device=data.device)

    # Humming can remove layout padding only in eight-element output vectors.
    # Align the logical row count first so arbitrary GGUF row slices also work.
    shape_n = (packed.shape[0] + 7) // 8 * 8
    if shape_n != packed.shape[0]:
        tensors = {
            name: torch.nn.functional.pad(tensor, (0, 0, 0, shape_n - packed.shape[0]))
            for name, tensor in tensors.items()
        }

    weight_schema = HummingWeightSchema(
        b_dtype=f"uint{packed.bits}",
        weight_scale_group_size=packed.group_size,
        has_zero_point=packed.zero_points is not None,
        is_fp_zero_point=packed.zero_points is not None,
    )
    # Explicitly disable activation quantization, including Humming defaults.
    input_schema = HummingInputSchema(input_quant_mode=InputQuantizationMode.Disabled)
    config, tensors = prepare_and_transform_humming_tensors(
        tensors=tensors,
        shape_n=shape_n,
        shape_k=packed.shape[1],
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=False,
        torch_dtype=dtype,
        device=data.device,
    )
    return GGUFHummingLinear(
        config,
        tensors,
        shape=packed.shape,
        dtype=dtype,
        compute_config=json.dumps(
            {
                "gemm_type": "dense",
                "use_f16_accum": False,
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
            }
        ),
    )
