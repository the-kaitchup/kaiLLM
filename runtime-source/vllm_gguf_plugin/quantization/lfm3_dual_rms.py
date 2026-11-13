# SPDX-License-Identifier: Apache-2.0
"""Default-off intralayer dual RMS, retaining the original decoder fallback.

Only model preparation reads the environment, validates providers and queries
the device. The runtime path uses tensor metadata and the explicit graph phase.
No core/model source, parameters, caches or cross-layer interfaces are changed.
"""

import hashlib
import inspect
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType

ENV = "VLLM_GGUF_LFM3_DUAL_RMS"
ORIGINAL_NORM_SHA = "8e2ca90679121c4dbcd4542695070835da6e9ada1a0132ae701c2071a6568745"
KERNEL_SHA = "d290fb975df9baa25d1c11a2ed91e39306e5c747cdffe36526674a98d1480b9e"
MODEL_SHA = "e379d909654f57862bfd5b796d1ab831915333d4ee644bc4db432e1e35877d95"
MAX_ROWS = ((1 << 31) - 1) // 2048


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PreparedDualRMS:
    original_forward: object
    device_index: int
    compute_capability: tuple
    original_model_provider: str
    original_model_sha256: str


def explicit_decode(context):
    return (
        getattr(getattr(context, "batch_descriptor", None), "is_prefill", None) is False
    )


def _context():
    from vllm.forward_context import get_forward_context

    try:
        return get_forward_context()
    except AssertionError:
        return None


def _inputs_eligible(x, residual, weight1, weight2, device_index):
    return (
        x.device.type == "cuda"
        and x.device.index == device_index
        and x.ndim == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.shape[1] == 2048
        and str(x.dtype) in ("torch.bfloat16", "torch.float32")
        and x.is_contiguous()
        and residual is not None
        and residual.device == x.device
        and residual.dtype == x.dtype
        and residual.shape == x.shape
        and all(
            weight.shape == (2048,)
            and str(weight.dtype) == "torch.float32"
            and weight.device == x.device
            for weight in (weight1, weight2)
        )
    )


def selected(layer, hidden_states, residual, context=None):
    state = getattr(layer, "_gguf_dual_rms", None)
    if state is None or state.compute_capability != (12, 0):
        return False
    if context is None:
        context = _context()
    # The first original operator_norm creates residual from hidden_states when
    # residual is None. No residual values are changed during this admission.
    check_residual = hidden_states if residual is None else residual
    return (
        explicit_decode(context)
        and not layer.post_operator_norm._forward_hooks
        and not layer.post_operator_norm._forward_pre_hooks
        and not layer.ffn_norm._forward_hooks
        and not layer.ffn_norm._forward_pre_hooks
        and _inputs_eligible(
            hidden_states,
            check_residual,
            layer.post_operator_norm.weight,
            layer.ffn_norm.weight,
            state.device_index,
        )
    )


@lru_cache(maxsize=1)
def _registered_op():
    import torch

    from .lfm3_dual_rms_kernel import dual_rms
    from .lfm3_norm import fused_lfm3_rms_norm

    @torch.library.custom_op("gguf_yaqa::lfm3_dual_rms", mutates_args=())
    def op(
        x: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        residual: torch.Tensor,
        epsilon1: float,
        epsilon2: float,
        activation_precast1: bool,
        activation_precast2: bool,
        device_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if explicit_decode(_context()) and _inputs_eligible(
            x, residual, weight1, weight2, device_index
        ):
            return dual_rms(
                x,
                weight1,
                weight2,
                residual,
                epsilon1,
                epsilon2,
                activation_precast1=activation_precast1,
                activation_precast2=activation_precast2,
            )
        intermediate = fused_lfm3_rms_norm(
            x, weight1, epsilon1, activation_precast=activation_precast1
        )
        return fused_lfm3_rms_norm(
            intermediate,
            weight2,
            epsilon2,
            activation_precast=activation_precast2,
            residual=residual,
        )

    @op.register_fake
    def fake(
        x,
        weight1,
        weight2,
        residual,
        epsilon1,
        epsilon2,
        activation_precast1,
        activation_precast2,
        device_index,
    ):
        return x.new_empty(x.shape), x.new_empty(x.shape)

    return op


def pair(layer, hidden_states, residual):
    first, second = layer.post_operator_norm, layer.ffn_norm
    state = layer._gguf_dual_rms
    # These original modules remain authoritative if the operator changes the
    # output/residual metadata after entry (for example mixed residual dtype).
    if not _inputs_eligible(
        hidden_states, residual, first.weight, second.weight, state.device_index
    ):
        return second(first(hidden_states), residual)
    return torch.ops.gguf_yaqa.lfm3_dual_rms(
        hidden_states,
        first.weight,
        second.weight,
        residual,
        first.variance_epsilon,
        second.variance_epsilon,
        first.gguf_activation_precast,
        second.gguf_activation_precast,
        state.device_index,
    )


def _decoder_forward(self, positions, hidden_states, residual):
    if not selected(self, hidden_states, residual):
        return self._gguf_dual_rms.original_forward(positions, hidden_states, residual)
    if residual is None:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
    else:
        hidden_states, residual = self.operator_norm(hidden_states, residual)
    if self.is_attention_layer:
        hidden_states = self.self_attn(positions, hidden_states)
    else:
        output = torch.empty_like(hidden_states)
        self.short_conv(hidden_states, output)
        hidden_states = output
    hidden_states, residual = pair(self, hidden_states, residual)
    hidden_states = self.feed_forward(hidden_states)
    if self.use_sandwich_norm:
        hidden_states = self.post_ffn_norm(hidden_states)
    return hidden_states, residual


@lru_cache(maxsize=16)
def _device_capability(device):
    import torch

    return tuple(torch.cuda.get_device_capability(device))


def prepare_model(model):
    if mode() == "none":
        return 0
    import torch as torch_module
    from vllm.model_executor.models.lfm3 import Lfm3DecoderLayer

    from ..weights_adapter.lfm3 import _fused_norm_forward

    global torch
    torch = torch_module
    root = Path(__file__).resolve().parent
    if (
        sha(root / "lfm3_norm.py") != ORIGINAL_NORM_SHA
        or sha(root / "lfm3_dual_rms_kernel.py") != KERNEL_SHA
    ):
        raise ValueError("Qualified original/fused RMS providers differ")
    provider = Path(inspect.getfile(Lfm3DecoderLayer)).absolute()
    if sha(provider) != MODEL_SHA:
        raise ValueError("Qualified original intralayer decoder provider differs")
    prepared = 0
    for layer in model.modules():
        if type(layer) is not Lfm3DecoderLayer or not layer.use_sandwich_norm:
            continue
        if hasattr(layer, "_gguf_dual_rms"):
            continue
        if getattr(layer.forward, "__func__", None) is not Lfm3DecoderLayer.forward:
            continue
        first, second = layer.post_operator_norm, layer.ffn_norm
        if (
            not all(
                getattr(norm._forward_method, "__func__", None) is _fused_norm_forward
                and norm.weight.device.type == "cuda"
                and str(norm.weight.dtype) == "torch.float32"
                and norm.weight.shape == (2048,)
                and not norm._forward_hooks
                and not norm._forward_pre_hooks
                for norm in (first, second)
            )
            or first.weight.device != second.weight.device
        ):
            continue
        capability = _device_capability(first.weight.device)
        if capability != (12, 0):
            continue
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare dual RMS before capture")
        _registered_op()
        layer._gguf_dual_rms = PreparedDualRMS(
            layer.forward,
            first.weight.device.index,
            capability,
            str(provider),
            MODEL_SHA,
        )
        layer.forward = MethodType(_decoder_forward, layer)
        prepared += 1
    return prepared


def runtime_marker(target=None):
    value = mode()
    root = Path(__file__).resolve().parent
    paths = (
        Path(__file__),
        root / "lfm3_dual_rms_kernel.py",
        root / "lfm3_norm.py",
        root.parent / "weights_adapter/lfm3.py",
    )
    return {
        "mode": value,
        "active": value == "auto" and tuple(target or ()) == (12, 0),
        "compute_capability": [12, 0],
        "sources": {str(path.absolute()): sha(path) for path in paths},
        "original_model_provider_sha256": MODEL_SHA,
        "scope": "Only actual intralayer sandwich post_operator_norm -> ffn_norm",
        "decode_only": True,
        "physical_rows": "Any positive M within original row-major signed-index bound",
        "width": 2048,
        "dtypes": ["bfloat16", "float32"],
        "new_GPU_weight_or_state_bytes": 0,
        "original_forward_fallback": (
            "Prefill/unknown/unsupported/SM86 retain saved callable"
        ),
        "cross_layer_or_terminal_fusion": False,
    }
