# SPDX-License-Identifier: Apache-2.0
"""F32 GDN diagnostic preserving the local native CUDA arithmetic order.

Build without GPU execution with ``python -m kernels.native_order_gdn --build``.
Then compare identical captured inputs with ``--check DIRECTORY --output FILE``.
The kernel supports torch.compile and CUDA graph replay after eager warmup.
The optional process-local vLLM adapter supports ragged prefills and indexed
batched decode, with no MTP and F32 activations. This helper deliberately
excludes packaged serving support.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
from functools import lru_cache, wraps
from pathlib import Path

import torch

from .current_ggml_cuda import _cuda_library_dir, _paths


def library_path():
    return Path(
        os.environ.get(
            "GGUF_NATIVE_ORDER_GDN", str(Path(__file__).resolve().parents[1] / "libraries/sm86/libnative_order_gdn.so")
        )
    )


def build():
    _, native_build, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entries = json.loads((native_build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/gated_delta_net.cu"))
    command = shlex.split(entry["command"])
    source = Path(__file__).with_suffix(".cu")
    library = library_path()
    library.parent.mkdir(parents=True, exist_ok=True)
    obj = library.with_suffix(".o")
    linked = library.with_suffix(".building.so")
    command[command.index(entry["file"])] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, check=True, cwd=entry["directory"])
    subprocess.run(
        [
            "g++",
            "-shared",
            str(obj),
            f"-L{cuda_library}",
            f"-Wl,-rpath,{cuda_library}",
            "-l:libcudart.so.13",
            "-o",
            str(linked),
        ],
        check=True,
    )
    linked.replace(library)
    library.with_suffix(".build.json").write_text(
        json.dumps(
            {
                "command": command,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return library


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(library_path()))
    lib.native_order_gdn_run.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int] * 2 + [ctypes.c_size_t]
    )
    lib.native_order_gdn_run.restype = ctypes.c_int
    lib.native_order_gdn_batched_run.argtypes = (
        [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_int64, ctypes.c_size_t]
    )
    lib.native_order_gdn_batched_run.restype = ctypes.c_int
    return lib


@torch.library.custom_op("gguf_current::gdn_native_order", mutates_args=())
def native_order_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output [1,T,H,128] and final state [1,H,128,128]."""
    if (
        q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
        or k.shape != q.shape
        or v.shape != q.shape
    ):
        raise ValueError(
            "Native GDN diagnostic requires one sequence and equal 128-wide Q/K/V"
        )
    _, tokens, heads, _ = q.shape
    if tokens <= 0 or initial_state.shape != (1, heads, 128, 128):
        raise ValueError("Invalid GDN initial state or empty token sequence")
    if g.shape != (1, tokens, heads) or beta.shape != g.shape:
        raise ValueError("GDN g/beta must be scalar per token/head")
    tensors = (q, k, v, g, beta, initial_state)
    if any(
        t.dtype != torch.float32 or t.device != q.device or not t.is_cuda
        for t in tensors
    ):
        raise ValueError("Native GDN diagnostic requires F32 CUDA inputs on one device")
    tensors = tuple(t.contiguous() for t in tensors)
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    state = torch.empty_like(initial_state, memory_format=torch.contiguous_format)
    with torch.cuda.device(q.device):
        status = _library().native_order_gdn_run(
            *(t.data_ptr() for t in tensors),
            output.data_ptr(),
            state.data_ptr(),
            heads,
            tokens,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Native GDN CUDA launch error {status}")
    return output, state


@native_order_gdn.register_fake
def _fake_native_order_gdn(q, k, v, g, beta, initial_state):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(initial_state, memory_format=torch.contiguous_format),
    )


def _validate_batched_inputs(q, k, v, g, beta, state, cu_seqlens):
    if (
        q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
        or k.shape != q.shape
        or v.shape != q.shape
        or g.shape != q.shape[:-1]
        or beta.shape != g.shape
    ):
        raise ValueError("Expected flat F32 Q/K/V [1,T,H,128] and g/beta [1,T,H]")
    if (
        state.ndim != 4
        or state.shape[1:] != (q.shape[2], 128, 128)
        or state.shape[0] <= 0
    ):
        raise ValueError("Expected F32 state [slots,H,128,128]")
    if any(
        x.dtype != torch.float32 or not x.is_cuda or x.device != q.device
        for x in (q, k, v, g, beta, state)
    ):
        raise ValueError("GDN inputs and state must be F32 on the same CUDA device")
    if (
        cu_seqlens.ndim != 1
        or not 2 <= cu_seqlens.numel() <= 65536
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.device != q.device
        or q.shape[1] >= 2**31
        or state.shape[0] >= 2**31
    ):
        raise ValueError("Expected GPU cu_seqlens [requests+1] within I32 bounds")


def _launch_batched(q, k, v, g, beta, state, cu_seqlens, indices, indexed):
    tensors = tuple(x.contiguous() for x in (q, k, v, g, beta))
    starts = cu_seqlens
    if starts.dtype == torch.int64:
        # Preserve invalid-range guards when accepting wider external metadata;
        # a direct I64->I32 cast could wrap an invalid offset into a valid one.
        starts = starts.clamp(min=-1, max=q.shape[1] + 1)
    starts = starts.to(torch.int32).contiguous()
    # Zero padding deterministically: inactive requests do not touch output or
    # cache in the native kernel. Real token rows are subsequently overwritten.
    output = torch.zeros_like(q, memory_format=torch.contiguous_format)
    final_state = (
        state
        if indexed
        else torch.empty_like(state, memory_format=torch.contiguous_format)
    )
    with torch.cuda.device(q.device):
        status = _library().native_order_gdn_batched_run(
            *(x.data_ptr() for x in tensors),
            state.data_ptr(),
            output.data_ptr(),
            final_state.data_ptr(),
            starts.data_ptr(),
            None if indices is None else indices.data_ptr(),
            q.shape[2],
            q.shape[1],
            starts.numel() - 1,
            state.shape[0],
            int(indexed),
            state.stride(0),
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(f"Batched native GDN CUDA launch error {status}")
    return output, final_state


@torch.library.custom_op("gguf_current::gdn_native_varlen", mutates_args=())
def native_order_gdn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged prefill with one dense initial/final state per request.

    The caller supplies monotonic cu_seqlens spanning the valid tokens. Empty
    request ranges preserve their state. Inputs retain the native B1 layout.
    """
    _validate_batched_inputs(q, k, v, g, beta, initial_state, cu_seqlens)
    if initial_state.shape[0] != cu_seqlens.numel() - 1:
        raise ValueError("Prefill needs one initial state per request")
    return _launch_batched(
        q, k, v, g, beta, initial_state.contiguous(), cu_seqlens, None, False
    )


@native_order_gdn_varlen.register_fake
def _fake_native_order_gdn_varlen(q, k, v, g, beta, initial_state, cu_seqlens):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(initial_state, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op(
    "gguf_current::gdn_native_decode_indexed", mutates_args=("state_cache",)
)
def native_order_gdn_decode_indexed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode into request-indexed cache slots without state gather/scatter.

    Active request indices must be unique. Slot zero is vLLM's reserved null
    slot; slots <=0 or >=cache size and empty ranges are inactive. Exactly one
    token per active request is supported, with zero-filled graph padding.
    """
    _validate_batched_inputs(q, k, v, g, beta, state_cache, cu_seqlens)
    if (
        state_cache.stride()[1:] != (128 * 128, 128, 1)
        or state_cache.stride(0) < q.shape[2] * 128 * 128
    ):
        raise ValueError(
            "Indexed GDN requires contiguous head/row/column interiors; "
            "padding between cache pages is supported"
        )
    if (
        state_indices.ndim != 1
        or state_indices.numel() != cu_seqlens.numel() - 1
        or state_indices.dtype not in (torch.int32, torch.int64)
        or state_indices.device != q.device
    ):
        raise ValueError("Expected one GPU integer state index per decode request")
    indices = state_indices
    if indices.dtype == torch.int64:
        indices = torch.where(
            (indices > 0) & (indices < state_cache.shape[0]), indices, 0
        )
    indices = indices.to(torch.int32).contiguous()
    output, _ = _launch_batched(
        q, k, v, g, beta, state_cache, cu_seqlens, indices, True
    )
    return output


@native_order_gdn_decode_indexed.register_fake
def _fake_native_order_gdn_decode_indexed(
    q, k, v, g, beta, state_cache, cu_seqlens, state_indices
):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def native_post_conv_prep_reference(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
):
    """Use native norm/unary arithmetic and the original negative GGUF A."""
    from .current_ggml_norm import current_norm_unary, rms_norm, sigmoid, softplus

    if conv_output.dtype != torch.float32:
        raise ValueError("Native diagnostic preparation requires F32 activations")
    count, _ = conv_output.shape
    heads_v = A_log.numel()
    q, k, v = conv_output.split(
        [num_k_heads * head_k_dim] * 2 + [heads_v * head_v_dim], dim=-1
    )
    q = q.reshape(count, num_k_heads, head_k_dim)
    k = k.reshape(count, num_k_heads, head_k_dim)
    v = v.reshape(count, heads_v, head_v_dim).contiguous()
    if apply_l2norm:
        options = dict(epsilon=1e-6 / head_k_dim, scale=head_k_dim**-0.5)
        q, k = rms_norm(q, **options), rms_norm(k, **options)
    else:
        q, k = q.contiguous(), k.contiguous()
    original_a = getattr(A_log, "_gguf_native_A", None)
    if original_a is None:
        raise RuntimeError("Native preparation requires original GGUF ssm_a")
    g = softplus(a.float() + dt_bias.float()) * original_a
    if output_g_exp:
        g = current_norm_unary(g.contiguous(), None, 0.0, 1.0, 4)
    return q, k, v, g, sigmoid(b.float())


def native_post_conv_prep(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
):
    implementation = native_post_conv_prep_reference
    if os.environ.get("GGUF_PARITY_FUSED_PREP") == "1":
        from .native_postconv_fused import native_post_conv_prep as implementation
    return implementation(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm,
        output_g_exp,
    )


def _packed_qkv_views(mixed_qkv):
    """Split native row-packed Qwen Q/K/V without changing their storage."""
    if (
        mixed_qkv is None
        or mixed_qkv.ndim != 2
        or mixed_qkv.dtype != torch.float32
        or mixed_qkv.shape[1] != 3 * 16 * 128
        or not mixed_qkv.is_contiguous()
    ):
        return None
    count = mixed_qkv.shape[0]
    return tuple(
        part.view(1, count, 16, 128)
        for part in mixed_qkv.split(16 * 128, dim=-1)
    )


def _reuse_packed_qkv(q, k, v):
    """Recover only proven adjacent row views; other layouts use the old cat.

    Native decode runs inside the opaque GDN op. These storage checks happen
    when its CUDA graph is recorded, not during steady graph replay.
    """
    if torch.compiler.is_compiling():
        return None
    width = 16 * 128
    row_width = 3 * width
    if (
        q.ndim != 4
        or q.shape[0] != 1
        or q.shape[2:] != (16, 128)
        or q.shape[1] < 1
        or q.device.type == "meta"
    ):
        return None
    count = q.shape[1]
    for tensor in (q, k, v):
        if (
            tensor.shape != q.shape
            or tensor.dtype != torch.float32
            or tensor.device != q.device
            or tensor.stride()[-2:] != (128, 1)
            or (count > 1 and tensor.stride(1) != row_width)
        ):
            return None
    storage = q.untyped_storage()
    offset = q.storage_offset()
    for index, tensor in enumerate((q, k, v)):
        if (
            tensor.untyped_storage().data_ptr() != storage.data_ptr()
            or tensor.storage_offset() != offset + index * width
        ):
            return None
    if (offset + count * row_width) * q.element_size() > storage.nbytes():
        return None
    return q.as_strided((count, row_width), (row_width, 1), offset)


def install_native_order_gdn():
    """Patch only diagnostic F32 recurrent execution, retaining current prep."""
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    if getattr(gdn, "_gguf_native_order_gdn", False):
        return
    original_chunk = gdn.fla_chunk_gated_delta_rule
    original_decode = gdn.fused_sigmoid_gating_delta_rule_update
    original_prepare = Qwen35GGUFAdapter.prepare_model
    original_rearrange = gdn.QwenGatedDeltaNetAttention.rearrange_mixed_qkv
    native_norm = os.environ.get("GGUF_PARITY_NATIVE_NORM") == "1"

    @wraps(original_rearrange)
    def rearrange(self, mixed_qkv):
        if (
            getattr(self, "_gguf_native_A", None) is not None
            and self.tp_size == 1
            and self.key_dim == self.value_dim == 16 * 128
            and self.head_k_dim == self.head_v_dim == 128
        ):
            views = _packed_qkv_views(mixed_qkv)
            if views is not None:
                return views
        return original_rearrange(self, mixed_qkv)

    def chunk(**kwargs):
        if kwargs["q"].dtype != torch.float32:
            return original_chunk(**kwargs)
        if kwargs.get("use_qk_l2norm_in_kernel", False):
            raise ValueError("Diagnostic prefill expects already normalized Q/K")
        inputs = tuple(kwargs[n] for n in ("q", "k", "v", "g", "beta", "initial_state"))
        starts = kwargs.get("cu_seqlens")
        if starts is None:
            output, state = native_order_gdn(*inputs)
        else:
            output, state = native_order_gdn_varlen(*inputs, starts)
        if kwargs.get("core_attn_out") is not None:
            kwargs["core_attn_out"].copy_(output)
        return output, state

    def decode(**kwargs):
        if kwargs["q"].dtype != torch.float32:
            return original_decode(**kwargs)
        q, k, v = (kwargs[n] for n in ("q", "k", "v"))
        if q.shape[0] != 1 or kwargs.get("num_accepted_tokens") is not None:
            raise ValueError("Diagnostic decode requires flat tokens and no MTP")
        packed = _reuse_packed_qkv(q, k, v)
        if packed is None:
            packed = torch.cat(
                [q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1
            ).squeeze(0)
        q, k, v, g, beta = gdn.fused_post_conv_prep(
            conv_output=packed,
            a=kwargs["a"],
            b=kwargs["b"],
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
            num_k_heads=q.shape[2],
            head_k_dim=128,
            head_v_dim=128,
            apply_l2norm=kwargs.get("use_qk_l2norm_in_kernel", False),
            output_g_exp=False,
        )
        if not kwargs.get("inplace_final_state", False):
            raise ValueError("Diagnostic decode requires cache-state update")
        starts = kwargs["cu_seqlens"]
        indices = kwargs["ssm_state_indices"].reshape(-1)[: starts.numel() - 1]
        output = native_order_gdn_decode_indexed(
            *(t.unsqueeze(0) for t in (q, k, v, g, beta)),
            kwargs["initial_state"],
            starts,
            indices,
        )
        return output, kwargs["initial_state"]

    @wraps(original_prepare)
    def prepare(self, model):
        original_prepare(self, model)
        originals = {}
        if native_norm:
            import gguf

            from .current_ggml_norm import prepare as prepare_norm

            prepare_norm()
            # Read only the tiny original decay vectors, avoiding the lossy
            # log/exp roundtrip expected by upstream HF parameter conventions.
            for path in self.load_spec.weights_source:
                reader = gguf.GGUFReader(path)
                for tensor in reader.tensors:
                    if tensor.name.endswith(".ssm_a"):
                        index = int(tensor.name.split(".")[1])
                        originals[index] = torch.from_numpy(tensor.data.copy())
        for name, module in model.named_modules():
            if name.endswith(".linear_attn"):
                module.enable_packed_recurrent_decode = False
                if native_norm:
                    index = int(name.split(".")[-2])
                    module.register_buffer(
                        "_gguf_native_A",
                        originals[index].to(module.A_log.device, torch.float32),
                        persistent=False,
                    )
                    module.A_log._gguf_native_A = module._gguf_native_A

    gdn.fla_chunk_gated_delta_rule = chunk
    gdn.fused_sigmoid_gating_delta_rule_update = decode
    gdn.QwenGatedDeltaNetAttention.rearrange_mixed_qkv = rearrange
    if native_norm:
        gdn.fused_post_conv_prep = native_post_conv_prep
    Qwen35GGUFAdapter.prepare_model = prepare
    gdn._gguf_native_order_gdn = True


def check(directory, output, layers, compile_and_graph=False):
    import numpy as np

    manifest_path = directory / "manifest.jsonl"
    manifest = {
        e["name"]: e for e in map(json.loads, manifest_path.read_text().splitlines())
    }

    def load(name):
        return torch.from_numpy(
            np.load(directory / manifest[name]["file"]).copy()
        ).cuda()

    results = []
    for layer in layers:
        q, k, v = (load(f"{n}_conv_predelta-{layer}") for n in ("q", "k", "v"))
        tokens, heads = q.shape[1:3]
        g = load(f"gate-{layer}").reshape(1, tokens, heads)
        beta = load(f"beta_sigmoid-{layer}").reshape(1, tokens, heads)
        initial = load(f"state_predelta-{layer}")
        actual, state = native_order_gdn(q, k, v, g, beta, initial)
        expected = load(f"attn_output-{layer}")
        expected_state = load(f"new_state-{layer}")
        row = {"layer": layer}
        if compile_and_graph:
            inputs = (q, k, v, g, beta, initial)
            compiled = torch.compile(native_order_gdn, fullgraph=True)
            compiled_values = compiled(*inputs)
            row["compiled_exact"] = all(
                torch.equal(a, b)
                for a, b in zip(compiled_values, (expected, expected_state))
            )
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graph_values = native_order_gdn(*inputs)
            for _ in range(3):
                graph.replay()
            row["graph_exact"] = all(
                torch.equal(a, b)
                for a, b in zip(graph_values, (expected, expected_state))
            )
            assert row["compiled_exact"] and row["graph_exact"]
        for label, a, b in (
            ("output", actual, expected),
            ("state", state, expected_state),
        ):
            delta = a - b
            row[label] = {
                "exact": torch.equal(a, b),
                "unequal_elements": int(torch.count_nonzero(delta)),
                "max_absolute": float(delta.abs().max()),
                "relative_l2": float(
                    delta.double().norm() / b.double().norm().clamp_min(1e-30)
                ),
            }
        results.append(row)
    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library_path().read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row[label]["exact"] for row in results for label in ("output", "state")):
        raise AssertionError("Native-order GDN differs from native captured arithmetic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--compile-and-graph", action="store_true")
    args = parser.parse_args()
    if args.build:
        print(build())
    if args.check:
        if args.output is None:
            parser.error("--check requires --output")
        check(args.check, args.output, args.layers, args.compile_and_graph)


if __name__ == "__main__":
    main()
