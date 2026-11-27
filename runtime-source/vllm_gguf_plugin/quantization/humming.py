# SPDX-License-Identifier: Apache-2.0
"""One-time GGUF repacking followed by Humming fused low-bit dense GEMM.

Humming loads packed integer weights and performs conversion inside its Tensor
Core kernel. Forward execution never materializes a dense floating-point weight.
Activations remain FP16/BF16; no activation quantization is introduced here.
"""

import importlib.metadata
import json
import os

import numpy as np
import torch
from packaging.version import Version

from .gguf_repack import FORMATS, repack_gguf

MIN_HUMMING_VERSION = "0.1.18"
SUPPORTED_QUANT_TYPES = frozenset(FORMATS)

# Weight repacking does not change with the two K tile sizes. These are the
# original GGUF shapes and quantizations checked against the dense BF16 kernel.
_K64_PROJECTIONS = {
    (384, 2304): (8, 12, 13, 14),
    (1024, 2304): (8, 12, 13, 14),
    (2304, 3072): (8, 13, 14),
    (3072, 2304): (8, 12, 13, 14, 18),
}


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

    dondon_m_decode_tuning = None
    two_row_tuning = None
    four_row_tuning = None

    def __init__(self, layer_config, tensors, *, shape, dtype, compute_config, decode_tuning=None,
                 two_row_tuning=None, four_row_tuning=None):
        super().__init__()
        from vllm.utils.humming import create_humming_locks, humming_forward

        self.layer_config = layer_config
        self.shape = shape
        self.dtype = dtype
        self.compute_config = compute_config
        self.decode_tuning = decode_tuning
        self.two_row_tuning = two_row_tuning
        self.four_row_tuning = four_row_tuning
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
        tuning = self.decode_tuning if x.numel() == self.shape[1] else None
        if (getattr(self, "dondon_m_decode_tuning", None) and x.numel() == self.shape[1]
                and x.dtype == self.dtype and x.is_contiguous()):
            tuning = self.dondon_m_decode_tuning
        if self.two_row_tuning and x.numel() == self.shape[1] * 2:
            tuning = self.two_row_tuning
        elif self.four_row_tuning and x.numel() == self.shape[1] * 4:
            tuning = self.four_row_tuning
        output = self._humming_forward(
            self.layer_config,
            inputs=x.reshape(-1, x.shape[-1]),
            weight=self.weight,
            weight_scale=self.weight_scale,
            zero_point=getattr(self, "zero_point", None),
            locks=self.locks,
            compute_config=self.compute_config,
            tuning_config=tuning,
        )
        # Humming's output epilogue requires N to be a multiple of eight.
        # Preparation adds zero rows when needed; expose only the GGUF rows.
        output = output[..., : self.shape[0]].reshape(*x.shape[:-1], self.shape[0])
        if bias is not None:
            output = output + bias
        return output


def _inkling_q8_decode_tuning(config, policy, weight_shape, ggml_type, dtype, device, tokens=1):
    """Select the deterministic K64 schedule for the qualified Inkling projection and row count."""
    if dtype != torch.bfloat16 or not policy["use_batch_invariant"] or torch.cuda.device_count() != 1:
        return None
    name = torch.cuda.get_device_name(device)
    is_3090 = name == "NVIDIA GeForce RTX 3090"
    from ..inkling.projection_streams import PRO_6000
    measured_iq4 = name in ("NVIDIA GeForce RTX 5090", PRO_6000)
    if not (is_3090 or measured_iq4) and name not in ("NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 5080"):
        return None
    shape = tuple(weight_shape)
    from vllm.config import get_current_vllm_config_or_none
    serving = get_current_vllm_config_or_none()
    if (serving is None or serving.speculative_config is not None
            or serving.parallel_config.world_size_across_dp != 1 or serving.scheduler_config.max_num_seqs != tokens
            or getattr(serving.model_config.hf_config, "model_type", None) != "inkling2_loop"):
        return None
    from ..inkling import pro_udq2, udq2_5090
    mixed_sms = None
    if pro_udq2.supported(serving, name, 1):
        mixed_sms = pro_udq2.ATTENTION_SMS.get((ggml_type, *shape))
        if mixed_sms is None:
            return None
    elif udq2_5090.supported(serving, name, 1, include_dondon_xs=True):
        mixed_sms = udq2_5090.ATTENTION_SMS.get((ggml_type, *shape))
        if mixed_sms is None:
            return None
    is_5080_b2 = tokens == 2 and name == "NVIDIA GeForce RTX 5080"
    if is_5080_b2:
        from ..inkling.projection_streams import supports_5080_b2
        if not supports_5080_b2(serving, name, 1):
            return None
        specs = {(384, 2304): (8, (8, 12, 13, 14)), (1024, 2304): (16, (8, 12, 13, 14)),
                 (2304, 3072): (36, (8, 13, 14)), (3072, 2304): (48, (8, 12, 13, 14, 18))}
        sms, qtypes = specs.get(shape, (None, ()))
        if ggml_type not in qtypes:
            return None
    elif tokens in (2, 4):
        # Both 5090 batches use the qualified input and convolution projections. The output retains stock Humming.
        if name != "NVIDIA GeForce RTX 5090" or ggml_type != 8 or shape not in ((24,2304), (32,2304), (384,2304), (1024,2304), (3072,2304)):
            return None
    elif tokens != 1:
        return None
    large_q8 = ggml_type == 8 and shape in ((1024, 2304), (2304, 3072), (3072, 2304))
    small_q8 = (is_3090 or measured_iq4) and ggml_type == 8 and shape in ((24, 2304), (384, 2304))
    if tokens == 1 and mixed_sms is None and not (large_q8 or small_q8):
        # Padded Humming output rows differ from the GGUF's 384 relative-feature rows.
        measured_types = {(384, 2304): (8, 12, 13, 14), (1024, 2304): (12, 13, 14),
                          (2304, 3072): (13, 14), (3072, 2304): (12, 13, 14, 18)}
        if name != "NVIDIA GeForce RTX 5080" or ggml_type not in measured_types.get(tuple(weight_shape), ()):
            return None
    if measured_iq4 and mixed_sms is None:
        from ..inkling.projection_streams import is_iq4_nl_donor
        if not is_iq4_nl_donor(serving.model_config):
            return None
    from vllm.utils.humming import get_heuristics_config
    baseline = get_heuristics_config(config, shape_m=tokens, **policy, device=device.index)
    if is_5080_b2 and (baseline.get('num_sms') != sms or not baseline.get('use_pdl')
                        or baseline.get('output_chunk_rows') != 0 or baseline.get('raster_group_m') != 1):
        return None
    if measured_iq4:
        # Humming aligns output rows to 256, including the narrow relative-bias and sink matrices.
        measured_sms = {(24, 2304): 4, (32, 2304): 4, (384, 2304): 8, (1024, 2304): 16, (2304, 3072): 36, (3072, 2304): 48}
        expected_sms = mixed_sms if mixed_sms is not None else measured_sms[shape]
        if baseline['num_sms'] != expected_sms:
            return None
    original_warp_n = 64 if is_3090 else 32
    if (tuple(baseline["block_shape"]) != (16, 64, 32) or tuple(baseline["warp_shape"]) != (16, original_warp_n, 32)
            or baseline["use_stream_k"] or baseline["use_f16_accum"] or baseline["num_stages"] != 3
            or baseline["num_ctas_per_sm"] != 1 or baseline["mma_type"] != "mma"):
        return None
    return json.dumps(dict(baseline, block_shape=[16, 64, 64], warp_shape=[16, 32, 64]))


def _portable_k64_decode_tuning(config, policy, shape, ggml_type, dtype, device):
    """Select the K64 tile when the existing Humming schedule preserves the reduction."""
    if (os.getenv("KAILLM_HUMMING_PORTABLE_K64") != "1" or dtype != torch.bfloat16
            or device.type != "cuda" or not policy["use_batch_invariant"] or policy["use_f16_accum"]
            or ggml_type not in _K64_PROJECTIONS.get(tuple(shape), ())
            or torch.cuda.get_device_capability(device) != (12, 0)):
        return None
    from vllm.config import get_current_vllm_config_or_none
    from vllm.utils.humming import get_heuristics_config

    serving = get_current_vllm_config_or_none()
    if (serving is None or serving.speculative_config is not None
            or serving.parallel_config.world_size_across_dp != 1 or serving.scheduler_config.max_num_seqs != 1):
        return None
    baseline = get_heuristics_config(config, shape_m=1, **policy, device=device.index)
    if not all(isinstance(baseline.get(key), (list, tuple)) for key in ("block_shape", "warp_shape")):
        return None
    expected = {
        "block_shape": [16, 64, 32], "warp_shape": [16, 32, 32], "use_stream_k": False,
        "use_f16_accum": False, "num_sms": config.shape_n // 64, "num_stages": 3,
        "num_ctas_per_sm": 1, "output_chunk_rows": 0, "use_pdl": True,
        "mma_type": "mma", "raster_group_m": 1,
    }
    normalized = dict(baseline, block_shape=list(baseline["block_shape"]), warp_shape=list(baseline["warp_shape"]))
    if normalized == expected:
        return json.dumps(dict(baseline, block_shape=[16, 64, 64], warp_shape=[16, 32, 64]))
    return None


def prepare_gguf_humming(
    data: torch.Tensor,
    quant_type: int,
    dtype: torch.dtype,
    *,
    chunk_rows: int = 128,
    prefix: str = "",
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
    policy = {
        "gemm_type": "dense",
        "use_f16_accum": False,
        # Keep each K reduction in one warp when batch-invariant GEMMs are requested.
        "use_batch_invariant": envs.VLLM_BATCH_INVARIANT
        or os.environ.get("KAILLM_HUMMING_DETERMINISTIC", "0") == "1",
    }
    layer = GGUFHummingLinear(
        config, tensors, shape=packed.shape, dtype=dtype, compute_config=json.dumps(policy),
        decode_tuning=_inkling_q8_decode_tuning(config, policy, packed.shape, quant_type, dtype, data.device),
        two_row_tuning=_inkling_q8_decode_tuning(config, policy, packed.shape, quant_type, dtype, data.device, 2),
        four_row_tuning=_inkling_q8_decode_tuning(config, policy, packed.shape, quant_type, dtype, data.device, 4),
    )
    if prefix and data.is_cuda and dtype == torch.bfloat16 and layer.decode_tuning is None:
        from ..inkling.dondon_5090 import attention_decode_tuning
        layer.dondon_m_decode_tuning = attention_decode_tuning(layer, int(quant_type), prefix, data.device)
    if layer.decode_tuning is None and layer.dondon_m_decode_tuning is None:
        layer.decode_tuning = _portable_k64_decode_tuning(config, policy, packed.shape, quant_type, dtype, data.device)
    return layer
