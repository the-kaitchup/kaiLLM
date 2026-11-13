"""Diagnostic graph-compatible F32 normalization and unary GGML operators.

Uses the exact same local library and private C++ ABI as current_ggml_direct.
Initialize the library before capturing CUDA graphs. This is not a packaged
serving dependency.
"""

import argparse
import ctypes
import json
import os
import shlex
import subprocess
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths


def library_path():
    return Path(
        os.environ.get("GGUF_CURRENT_NORM", str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_norm.so"))
    )


def build():
    _, directory, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entry = next(
        entry
        for entry in json.loads((directory / "compile_commands.json").read_text())
        if entry["file"].endswith("/ggml-cuda.cu")
    )
    command = shlex.split(entry["command"])
    library = library_path()
    obj = library.with_suffix(".o")
    command[command.index(entry["file"])] = str(Path(__file__).with_suffix(".cu"))
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, check=True, cwd=entry["directory"])
    subprocess.run(
        [
            "g++",
            "-shared",
            str(obj),
            f"-L{directory / 'bin'}",
            f"-L{cuda_library}",
            f"-Wl,-rpath,{directory / 'bin'}",
            f"-Wl,-rpath,{cuda_library}",
            "-lggml-cuda",
            "-lggml-base",
            "-l:libcudart.so.13",
            "-o",
            str(library),
        ],
        check=True,
    )


@lru_cache(maxsize=1)
def _library():
    library = ctypes.CDLL(str(library_path()))
    library.current_norm_error.restype = ctypes.c_char_p
    library.current_norm_prepare.argtypes = [ctypes.c_int]
    library.current_norm_prepare.restype = ctypes.c_int
    library.current_norm_run.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int64] * 2
        + [ctypes.c_float] * 2
        + [ctypes.c_int] * 2
        + [ctypes.c_size_t]
    )
    library.current_norm_run.restype = ctypes.c_int
    library.current_conv_run.argtypes = (
        [ctypes.c_void_p] * 3
        + [ctypes.c_int64] * 2
        + [ctypes.c_int] * 3
        + [ctypes.c_size_t]
    )
    library.current_conv_run.restype = ctypes.c_int
    if hasattr(library, "current_conv_indexed_run"):
        library.current_conv_indexed_run.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 4
            + [ctypes.c_int64] * 4
            + [ctypes.c_int] * 2
            + [ctypes.c_size_t]
        )
        library.current_conv_indexed_run.restype = ctypes.c_int
    if hasattr(library, "native_swiglu_run"):
        library.native_swiglu_run.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 3 + [ctypes.c_size_t]
        )
        library.native_swiglu_run.restype = ctypes.c_int
        library.native_swiglu_q8_run.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 4 + [ctypes.c_size_t]
        )
        library.native_swiglu_q8_run.restype = ctypes.c_int
        library.native_gated_rms_run.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int64] * 4
            + [ctypes.c_float, ctypes.c_size_t]
        )
        library.native_gated_rms_run.restype = ctypes.c_int
        library.native_gated_rms_q8_run.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int64] * 3
            + [ctypes.c_float, ctypes.c_size_t]
        )
        library.native_gated_rms_q8_run.restype = ctypes.c_int
        library.native_q8_reference_run.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 3 + [ctypes.c_size_t]
        )
        library.native_q8_reference_run.restype = ctypes.c_int
    return library


def has_gate_fusion():
    """Old frozen reference helpers remain valid without the fusion exports."""
    return hasattr(_library(), "native_swiglu_run")


_GEMMA_CACHE = "_gguf_native_rms_weight"


def _gemma_rms_weight(module):
    if getattr(module, "_gguf_native_rms_cache_ready", False):
        return getattr(module, _GEMMA_CACHE)
    return module.weight + 1.0


@torch.no_grad()
def _refresh_gemma_rms_cache(module):
    """Run the same F32 addition after loading, retaining graph-owned storage."""
    module._gguf_native_rms_cache_ready = False
    weight = module.weight
    if weight.dtype != torch.float32 or weight.is_meta:
        return
    value = weight.detach() + 1.0
    previous = getattr(module, _GEMMA_CACHE, None)
    if (
        previous is not None
        and previous.shape == value.shape
        and previous.dtype == value.dtype
        and previous.device == value.device
    ):
        previous.copy_(value)
    else:
        module.register_buffer(_GEMMA_CACHE, value, persistent=False)
    module._gguf_native_rms_cache_ready = True


def _install_gemma_rms_cache(model):
    """Cache only after a completed model load, including subsequent reloads.

    vLLM's layerwise reload restores original nonpersistent buffers before
    model.load_weights returns. Updating those buffers in place preserves the
    pointers held by captured CUDA graphs. A failed load raises immediately;
    callers must not resume inference on a partially loaded model.
    """
    from functools import wraps
    from types import MethodType

    if getattr(model, "_gguf_native_rms_load_wrapped", False):
        return
    original = model.load_weights

    @wraps(original)
    def load_weights(self, *args, **kwargs):
        norms = [
            module
            for module in self.modules()
            if type(module).__name__ == "GemmaRMSNorm"
        ]
        previous = [
            (module.weight, getattr(module, _GEMMA_CACHE, None)) for module in norms
        ]
        for module in norms:
            module._gguf_native_rms_cache_ready = False
        try:
            result = original(*args, **kwargs)
        except BaseException:
            # Keep already captured buffers consistent with any partial
            # in-place copies, while leaving eager caches invalid. This does
            # not make a failed or partially loaded model safe to execute.
            with torch.no_grad():
                for module, (weight, cached) in zip(norms, previous):
                    # Layerwise reload may have hidden these on meta at
                    # entry, then restored their original storage before
                    # a later layer failed.
                    restored = getattr(module, _GEMMA_CACHE, None)
                    if restored is not None and not module.weight.is_meta:
                        weight, cached = module.weight, restored
                    if (
                        cached is not None
                        and not weight.is_meta
                        and cached.shape == weight.shape
                        and cached.dtype == weight.dtype
                        and cached.device == weight.device
                    ):
                        # The original failed load remains the error;
                        # its invalid model must not resume inference.
                        with suppress(RuntimeError):
                            cached.copy_(weight.detach() + 1.0)
            raise
        for module in norms:
            _refresh_gemma_rms_cache(module)
        return result

    model.load_weights = MethodType(load_weights, model)
    model._gguf_native_rms_load_wrapped = True


def prepare(device=None):
    library = _library()
    index = torch.cuda.current_device() if device is None else device
    if library.current_norm_prepare(index):
        raise RuntimeError(library.current_norm_error().decode())


@torch.library.custom_op("gguf_current::norm_unary", mutates_args=())
def current_norm_unary(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    scale: float,
    operation: int,
) -> torch.Tensor:
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("Expected contiguous CUDA FP32 input")
    if weight is not None and (
        weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
        or weight.numel() != x.shape[-1]
    ):
        raise ValueError("Expected contiguous FP32 weight with matching width/device")
    if operation not in range(5) or epsilon < 0 or (weight is not None and scale != 1):
        raise ValueError("Invalid diagnostic normalization operation")
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output
    library = _library()
    with torch.cuda.device(x.device):
        result = library.current_norm_run(
            x.data_ptr(),
            weight.data_ptr() if weight is not None else None,
            output.data_ptr(),
            x.shape[-1],
            x.numel() // x.shape[-1],
            epsilon,
            scale,
            operation,
            x.device.index,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if result:
        raise RuntimeError(library.current_norm_error().decode())
    return output


@current_norm_unary.register_fake
def _fake(x, weight, epsilon, scale, operation):
    return torch.empty_like(x)


@torch.library.custom_op("gguf_current::fused_swiglu", mutates_args=())
def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.stride(1) != 1
        or x.shape[1] % 2
        or x.shape[1] == 0
    ):
        raise ValueError("Expected row-strided CUDA F32 [M,2K]")
    rows, width = x.shape[0], x.shape[1] // 2
    output = x.new_empty((rows, width))
    if not rows:
        return output
    with torch.cuda.device(x.device):
        status = _library().native_swiglu_run(
            x.data_ptr(),
            output.data_ptr(),
            width,
            rows,
            x.stride(0),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native SwiGLU launch failed: {status}")
    return output


@fused_swiglu.register_fake
def _fake_swiglu(x):
    return x.new_empty((x.shape[0], x.shape[1] // 2))


@torch.library.custom_op("gguf_current::fused_swiglu_q8", mutates_args=())
def fused_swiglu_q8(x: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.stride(1) != 1
        or x.shape[1] % 2
        or x.shape[1] == 0
        or not 1 <= x.shape[0] <= 8
    ):
        raise ValueError("Expected row-strided CUDA F32 [M,2K], M1..8")
    rows, width = x.shape[0], x.shape[1] // 2
    padded = (width + 511) // 512 * 512
    output = torch.empty((rows, padded // 32 * 36), device=x.device, dtype=torch.uint8)
    with torch.cuda.device(x.device):
        status = _library().native_swiglu_q8_run(
            x.data_ptr(),
            output.data_ptr(),
            width,
            rows,
            x.stride(0),
            padded,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native SwiGLU Q8 launch failed: {status}")
    return output


@fused_swiglu_q8.register_fake
def _fake_swiglu_q8(x):
    padded = (x.shape[1] // 2 + 511) // 512 * 512
    return torch.empty(
        (x.shape[0], padded // 32 * 36), device=x.device, dtype=torch.uint8
    )


def gate_q8_reference(x):
    if (
        x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.ndim != 2
    ):
        raise ValueError("Expected contiguous CUDA F32 [M,K]")
    rows, width = x.shape
    padded = (width + 511) // 512 * 512
    output = torch.empty((rows, padded // 32 * 36), device=x.device, dtype=torch.uint8)
    with torch.cuda.device(x.device):
        status = _library().native_q8_reference_run(
            x.data_ptr(),
            output.data_ptr(),
            width,
            rows,
            padded,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native Q8 reference launch failed: {status}")
    return output


@torch.library.custom_op("gguf_current::fused_gated_rms", mutates_args=())
def fused_gated_rms(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if (
        x.ndim not in (2, 3)
        or x.shape[-1] != 128
        or gate.shape != x.shape
        or weight.shape != (128,)
        or epsilon < 0
        or any(
            t.dtype != torch.float32 or not t.is_cuda or t.device != x.device
            for t in (x, weight, gate)
        )
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or gate.stride(-1) != 1
        or gate.stride(-2) < 128
        or (gate.ndim == 3 and gate.stride(0) < gate.shape[1] * gate.stride(1))
    ):
        raise ValueError(
            "Expected contiguous CUDA F32 [M,H,128] or [M,128], "
            "weight[128], row-strided gate"
        )
    output = torch.empty_like(x)
    if not x.numel():
        return output
    with torch.cuda.device(x.device):
        status = _library().native_gated_rms_run(
            x.data_ptr(),
            weight.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            x.numel() // 128,
            x.shape[-2],
            gate.stride(0) if gate.ndim == 3 else 0,
            gate.stride(-2),
            epsilon,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native gated RMS launch failed: {status}")
    return output


@fused_gated_rms.register_fake
def _fake_gated_rms(x, weight, gate, epsilon):
    return torch.empty_like(x)


@torch.library.custom_op("gguf_current::fused_gated_rms_q8", mutates_args=())
def fused_gated_rms_q8(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if (
        x.ndim != 3
        or x.shape[1:] != (16, 128)
        or not 1 <= x.shape[0] <= 8
        or gate.shape != x.shape
        or weight.shape != (128,)
        or epsilon < 0
        or any(
            t.dtype != torch.float32 or not t.is_cuda or t.device != x.device
            for t in (x, weight, gate)
        )
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or gate.stride(-1) != 1
        or gate.stride(-2) < 128
        or gate.stride(0) < 16 * gate.stride(1)
    ):
        raise ValueError("Expected CUDAF32 [M1..8,16,128], weight[128], gate")
    output = torch.empty((x.shape[0], 64 * 36), device=x.device, dtype=torch.uint8)
    with torch.cuda.device(x.device):
        status = _library().native_gated_rms_q8_run(
            x.data_ptr(),
            weight.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            x.shape[0],
            gate.stride(0),
            gate.stride(1),
            epsilon,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native gated RMS Q8 launch failed: {status}")
    return output


@fused_gated_rms_q8.register_fake
def _fake_gated_rms_q8(x, weight, gate, epsilon):
    return torch.empty((x.shape[0], 64 * 36), device=x.device, dtype=torch.uint8)


def rms_norm(x, weight=None, epsilon=1e-6, scale=1.0):
    return current_norm_unary(x.contiguous(), weight, epsilon, scale, 0)


def silu(x):
    return current_norm_unary(x.contiguous(), None, 0.0, 1.0, 1)


def sigmoid(x):
    return current_norm_unary(x.contiguous(), None, 0.0, 1.0, 2)


def softplus(x):
    return current_norm_unary(x.contiguous(), None, 0.0, 1.0, 3)


@torch.library.custom_op("gguf_current::conv", mutates_args=())
def current_conv(
    history: torch.Tensor, weight: torch.Tensor, use_silu: bool
) -> torch.Tensor:
    if (
        history.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != history.shape[0]
        or weight.shape[1] != 4
        or history.shape[1] < 4
        or history.dtype != torch.float32
        or weight.dtype != torch.float32
        or not history.is_cuda
        or weight.device != history.device
        or not history.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Diagnostic convolution requires contiguous F32 [C,T+3], [C,4]"
        )
    channels, tokens = history.shape[0], history.shape[1] - 3
    result = history.new_empty(tokens, channels)
    if tokens > 32 and tokens % 32:
        # Native SSM convolution loads a whole 32-token shared-memory tile
        # even when only its prefix contributes to valid output. GGML's
        # allocation padding protects the final channel; Torch storage has
        # no such contract. Preserve row strides and add an explicit guard.
        extra = (-tokens) % 32
        storage = history.new_empty(history.numel() + extra)
        padded_history = storage[: history.numel()].view(history.shape)
        padded_history.copy_(history)
        storage[history.numel() :].zero_()
        history = padded_history
    library = _library()
    with torch.cuda.device(history.device):
        status = library.current_conv_run(
            history.data_ptr(),
            weight.data_ptr(),
            result.data_ptr(),
            channels,
            tokens,
            4,
            int(use_silu),
            history.device.index,
            torch.cuda.current_stream(history.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_norm_error().decode())
    return result


@current_conv.register_fake
def _fake_conv(history, weight, use_silu):
    return history.new_empty(history.shape[1] - 3, history.shape[0])


@torch.library.custom_op("gguf_current::conv_indexed", mutates_args=("state",))
def current_conv_indexed(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    starts: torch.Tensor | None,
    has_state: torch.Tensor | None,
    use_silu: bool,
) -> torch.Tensor:
    """Ragged prefill or B-token decode; slot zero is the reserved null block."""
    decode = starts is None
    if (
        x.ndim != 2
        or x.stride(1) != 1
        or weight.shape != (x.shape[1], 4)
        or not weight.is_contiguous()
        or state.ndim != 3
        or state.shape[1:] != (x.shape[1], 3)
        or slots.ndim != 1
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or any(t.dtype != torch.float32 for t in (x, weight, state))
        or any(not t.is_cuda or t.device != x.device for t in (x, weight, state, slots))
        or (decode and slots.numel() != x.shape[0])
    ):
        raise ValueError("Expected F32 [T,C], [C,4], [slots,C,3], and I64 slots")
    if not decode and (
        starts.shape != (slots.numel() + 1,)
        or starts.dtype != torch.int32
        or not starts.is_contiguous()
        or starts.device != x.device
        or has_state is None
        or has_state.shape != slots.shape
        or has_state.dtype != torch.bool
        or not has_state.is_contiguous()
        or has_state.device != x.device
    ):
        raise ValueError(
            "Ragged convolution requires I32 starts and bool initial state"
        )
    result = x.new_empty(x.shape)
    if not x.numel():
        return result
    library = _library()
    if not hasattr(library, "current_conv_indexed_run"):
        raise RuntimeError("Rebuild the native norm helper for indexed convolution")
    with torch.cuda.device(x.device):
        status = library.current_conv_indexed_run(
            x.data_ptr(),
            weight.data_ptr(),
            state.data_ptr(),
            result.data_ptr(),
            slots.data_ptr(),
            0 if decode else starts.data_ptr(),
            0 if decode else has_state.data_ptr(),
            x.shape[1],
            x.shape[0],
            slots.numel(),
            state.shape[0],
            x.stride(0),
            *state.stride(),
            int(decode),
            int(use_silu),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Indexed native convolution CUDA error {status}")
    return result


@current_conv_indexed.register_fake
def _fake_conv_indexed(x, weight, state, slots, starts, has_state, use_silu):
    return x.new_empty(x.shape)


def install_native_conv():
    """Use request-indexed native-order convolution for prefill and decode."""
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

    if getattr(gdn, "_current_native_conv", False):
        return

    def prefill(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=None,
        has_initial_state=None,
        activation="silu",
        **kwargs,
    ):
        if bias is not None or activation not in ("silu", "swish"):
            raise ValueError("Native convolution expects bias-free Qwen SiLU")
        count = query_start_loc.numel() - 1
        result = current_conv_indexed(
            x.T,
            weight.contiguous(),
            conv_states,
            cache_indices[:count].to(torch.int64).contiguous(),
            query_start_loc.to(torch.int32).contiguous(),
            has_initial_state[:count].contiguous(),
            True,
        )
        return result.T

    def decode(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        conv_state_indices=None,
        num_accepted_tokens=None,
        out=None,
        **kwargs,
    ):
        if x.ndim != 2 or bias is not None or num_accepted_tokens is not None:
            raise ValueError(
                "Native convolution expects non-speculative Qwen token rows"
            )
        result = current_conv_indexed(
            x,
            weight.contiguous(),
            conv_state,
            conv_state_indices[: x.shape[0]].to(torch.int64).contiguous(),
            None,
            None,
            activation in (True, "silu", "swish"),
        )
        # The enclosing GDN custom op does not declare mixed_qkv mutable.
        # Return fresh storage unless the caller explicitly supplies an output.
        if out is None:
            return result
        out.copy_(result)
        return out

    gdn.causal_conv1d_fn = prefill
    gdn.causal_conv1d_update = decode
    gdn._current_native_conv = True


def install_native_norms():
    """Use reference norm/unary arithmetic in the diagnostic model only."""
    from types import MethodType

    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    original = Qwen35GGUFAdapter.prepare_model
    if getattr(original, "_current_native_norm", False):
        return
    gate_fusion_available = False

    def gemma(module, x, residual=None):
        combined = x if residual is None else x + residual
        output = rms_norm(
            combined, _gemma_rms_weight(module), epsilon=module.variance_epsilon
        )
        return output if residual is None else (output, combined)

    def gated(module, x, z=None):
        if module.group_size is not None or not module.norm_before_gate:
            raise ValueError("Diagnostic expects Qwen3.5 norm-before-gate")
        if (
            gate_fusion_available
            and z is not None
            and x.ndim in (2, 3)
            and x.shape[-1] == 128
            and x.is_contiguous()
            and z.stride(-1) == 1
            and z.stride(-2) >= 128
            and (z.ndim == 2 or z.stride(0) >= z.shape[1] * z.stride(1))
        ):
            return fused_gated_rms(x, module.weight, z, module.eps)
        output = rms_norm(x, module.weight, epsilon=module.eps)
        return output if z is None else output * silu(z)

    def swiglu(module, x):
        if gate_fusion_available and x.ndim == 2 and x.stride(1) == 1:
            return fused_swiglu(x)
        gate, up = x.chunk(2, dim=-1)
        return silu(gate) * up

    def prepare_model(adapter, model):
        nonlocal gate_fusion_available
        original(adapter, model)
        prepare()
        gate_fusion_available = has_gate_fusion()
        if gate_fusion_available:
            _install_gemma_rms_cache(model)
        for module in model.modules():
            name = type(module).__name__
            if name == "GemmaRMSNorm":
                module._forward_method = MethodType(gemma, module)
            elif name == "RMSNormGated":
                module._forward_method = MethodType(gated, module)
            elif name == "SiluAndMul":
                module._forward_method = MethodType(swiglu, module)
            if hasattr(module, "use_fused_qk_norm_rope_gate"):
                module.use_fused_qk_norm_rope_gate = False

    prepare_model._current_native_norm = True
    Qwen35GGUFAdapter.prepare_model = prepare_model
    install_native_conv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    build()
