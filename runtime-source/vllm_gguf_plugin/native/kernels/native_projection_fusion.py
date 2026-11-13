"""Exact native gating/Q8 fusion at dense FFN and GDN output boundaries.

Scheduling-dependent choices stay inside opaque operators so compiled graphs
cannot freeze a prefill layout as T1 decode. Unsupported layers retain their
original forward methods, including separately selected body MMA kernels.
"""

from functools import wraps
from types import MethodType

import torch

from . import current_ggml_direct as direct
from . import current_ggml_norm as norm


def _sm86(tensor):
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) == (8, 6)


def _ssm_prequant_shape(qtype, n, k, tokens):
    return (
        k == 2048
        and 1 <= tokens <= 8
        and (
            tokens == 1
            or (qtype, n) in ((13, 1024), (14, 1024), (8, 1024), (8, 2048))
        )
    )


@torch.library.custom_op("gguf_current::swiglu_projection", mutates_args=())
def fused_swiglu_projection(
    gate_up: torch.Tensor, packed: torch.Tensor, qtype: int, k: int
) -> torch.Tensor:
    if gate_up.ndim != 2 or gate_up.shape[1] != 2 * k:
        raise ValueError("Expected native FFN gate/up [tokens,2K]")
    if 1 <= gate_up.shape[0] <= 8 and _sm86(gate_up):
        q8 = norm.fused_swiglu_q8(gate_up)
        return direct.current_ggml_prequant(q8, packed, qtype, k)
    activated = norm.fused_swiglu(gate_up)
    return direct.current_ggml_direct(activated, packed, qtype, k)


@fused_swiglu_projection.register_fake
def _fake_swiglu_projection(gate_up, packed, qtype, k):
    return gate_up.new_empty((gate_up.shape[0], packed.shape[0]))


@torch.library.custom_op("gguf_current::gated_rms_projection", mutates_args=())
def fused_gated_rms_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    packed: torch.Tensor,
    qtype: int,
    k: int,
    epsilon: float,
) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1:] != (16, 128) or k != 2048:
        raise ValueError("Expected native GDN core [tokens,16,128] and K2048")
    tokens = x.shape[0]
    groups = direct._runtime_ssm_groups(tokens)
    if (
        groups == [(0, 1, tokens)]
        and _ssm_prequant_shape(qtype, packed.shape[0], k, tokens)
        and _sm86(x)
    ):
        q8 = norm.fused_gated_rms_q8(x.contiguous(), weight, gate, epsilon)
        return direct.current_ggml_prequant(q8, packed, qtype, k, channels=True)
    activated = norm.fused_gated_rms(x.contiguous(), weight, gate, epsilon)
    return direct.current_ggml_ssm(activated.reshape(tokens, k), packed, qtype, k)


@fused_gated_rms_projection.register_fake
def _fake_gated_rms_projection(x, weight, gate, packed, qtype, k, epsilon):
    return x.new_empty((x.shape[0], packed.shape[0]))


def _projection_operation(layer, expected_type):
    """Resolve after weight loading; never bypass TP, bias, or another backend."""
    if getattr(layer, "tp_size", None) != 1 or getattr(layer, "bias", None) is not None:
        return None
    parts = getattr(layer, "gguf_humming", None)
    if parts is None or len(parts) != 1:
        return None
    operation = getattr(parts[0], "operation", None)
    if (
        type(operation) is not expected_type
        or getattr(parts[0], "dtype", None) != torch.float32
    ):
        return None
    return operation


def _packed(operation):
    return operation.packed_storage[: operation.shape[0] * operation.row_bytes].view(
        operation.shape[0], operation.row_bytes
    )


def _install_mlp(module):
    if getattr(module, "_gguf_native_projection_fusion", False):
        return
    original = module.forward

    @wraps(original)
    def forward(self, x):
        operation = _projection_operation(
            self.down_proj, direct.CurrentGGMLDirectLinear
        )
        if (
            operation is None
            or self.expert_gate is not None
            or type(self.act_fn).__name__ != "SiluAndMul"
            or x.ndim != 2
            or x.dtype != torch.float32
        ):
            return original(x)
        gate_up, _ = self.gate_up_proj(x)
        return fused_swiglu_projection(
            gate_up, _packed(operation), operation.qtype, operation.k
        )

    module.forward = MethodType(forward, module)
    module._gguf_native_projection_fusion = True


def _install_gdn(module):
    if getattr(module, "_gguf_native_projection_fusion", False):
        return
    original = module._output_projection

    @wraps(original)
    def output_projection(self, x, gate):
        operation = _projection_operation(self.out_proj, direct.CurrentGGMLSSMLinear)
        if (
            operation is None
            or x.ndim != 3
            or x.shape[1:] != (16, 128)
            or x.dtype != torch.float32
            or gate.shape != x.shape
            or operation.k != 2048
            or self.norm.group_size is not None
            or not self.norm.norm_before_gate
            or self.norm.weight.dtype != torch.float32
            or gate.stride(-1) != 1
            or gate.stride(-2) < 128
            or gate.stride(0) < 16 * gate.stride(1)
        ):
            return original(x, gate)
        return fused_gated_rms_projection(
            x,
            self.norm.weight,
            gate,
            _packed(operation),
            operation.qtype,
            operation.k,
            self.norm.eps,
        )

    module._output_projection = MethodType(output_projection, module)
    module._gguf_native_projection_fusion = True


def install_native_projection_fusions():
    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    original = Qwen35GGUFAdapter.prepare_model
    if getattr(original, "_native_projection_fusions", False):
        return

    @wraps(original)
    def prepare(adapter, model):
        original(adapter, model)
        if not direct.has_prequant() or not norm.has_gate_fusion():
            return
        for module in model.modules():
            if type(module).__name__ == "Qwen2MoeMLP":
                _install_mlp(module)
            elif type(module).__name__ == "QwenGatedDeltaNetAttention":
                _install_gdn(module)

    prepare._native_projection_fusions = True
    Qwen35GGUFAdapter.prepare_model = prepare
