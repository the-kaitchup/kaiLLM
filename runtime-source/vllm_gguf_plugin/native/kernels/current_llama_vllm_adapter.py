"""Diagnostic vLLM projection adapter for the current local GGML CUDA library.

By default this uses synchronized GGML graph calls to isolate arithmetic.
GGUF_PARITY_CURRENT_DIRECT=1 selects the graph-compatible direct kernel bridge.
GGUF_PARITY_SHARED_Q8=1 additionally shares compatible activation quantization
within merged projections; original GGUF matrix boundaries are retained.
Both depend on the local reference library and are experimental adapters.
"""

import os
from functools import wraps


def _is_ssm_output(layer):
    return getattr(layer, "prefix", "").endswith(".linear_attn.out_proj")


def _install_ssm_decode_guard():
    """Keep all prefill projections outside padded T1/channel-B graphs."""
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    from .native_gdn_capture import _restrict_replay

    if getattr(CudaGraphManager, "_gguf_ssm_decode_guard", False):
        return
    original = CudaGraphManager.dispatch

    @wraps(original)
    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
        num_ubatches=1,
    ):
        result = original(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len,
            num_ubatches,
        )
        return _restrict_replay(
            result,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            # MRV2 calls an existing-context one-token prefill extension
            # decode-eligible, but still passes max_query_len=None because
            # has_prefill is true. Padding a B3/T1 remainder to a B4 decode
            # graph changes the native projection geometry and reductions.
            # Pure decode and dummy decode capture explicitly pass 1, but
            # must also retain the actual row count: even F32 N16 projection
            # arithmetic changes when B3 is padded to a captured B4 graph.
            uniform_token_count=(
                uniform_token_count
                if max_query_len == 1 and result.num_tokens == num_tokens
                else None
            ),
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )

    CudaGraphManager.dispatch = dispatch
    CudaGraphManager._gguf_ssm_decode_guard = True


def _q6_mma_head_supported(parts):
    """The optimized head belongs only to the measured original Q6 matrix."""
    return (
        len(parts) == 1
        and int(parts[0][1]) == 14
        and tuple(parts[0][0].shape) == (248320, 840)
    )


def _original_matrix_parts(layer):
    """Restore GGUF matrix boundaries after vLLM's logical shard expansion."""
    weight = layer.qweight
    default_type = layer.qweight_type.weight_type
    if not getattr(weight, "shard_offset_map", None):
        return [(weight, default_type)]
    groups = [(shard,) for shard in weight.shard_id]
    if getattr(layer, "prefix", "").endswith(".linear_attn.in_proj_qkvz"):
        if list(weight.shard_id) != [0, 1, 2, 3]:
            raise ValueError("Qwen GDN native adapter expects Q/K/V/Z logical shards")
        # GGUF attn_qkv is one matrix, expanded by the loader into Q/K/V
        # logical shards. Z is a separate original matrix, even at equal type.
        groups = [(0, 1, 2), (3,)]
    parts = []
    for shards in groups:
        spans = [weight.shard_offset_map[shard] for shard in shards]
        types = [
            layer.qweight_type.shard_weight_type.get(shard, default_type)
            for shard in shards
        ]
        if any(kind != types[0] for kind in types):
            raise ValueError("An original GGUF matrix cannot mix quantization types")
        if any(width != spans[0][2] for _, _, width in spans) or any(
            left[1] != right[0] for left, right in zip(spans, spans[1:])
        ):
            raise ValueError("Original GGUF matrix shards must have contiguous rows")
        start, width = spans[0][0], spans[0][2]
        end = spans[-1][1]
        parts.append((weight[start:end, :width].contiguous(), types[0]))
    return parts


def install_current_llama_adapter():
    import gguf
    import torch

    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    if getattr(GGUFLinearMethod, "_parity_current_llama", False):
        return

    use_direct = os.environ.get("GGUF_PARITY_CURRENT_DIRECT") == "1"
    use_q6_mma_head = os.environ.get("GGUF_PARITY_Q6_MMA_HEAD") == "1"
    use_shared_q8 = os.environ.get("GGUF_PARITY_SHARED_Q8") == "1"
    use_body_mma = os.environ.get("GGUF_PARITY_BODY_MMA") == "1"
    if use_q6_mma_head and not use_direct:
        raise ValueError("Q6 MMA head diagnostic requires GGUF_PARITY_CURRENT_DIRECT=1")
    if use_q6_mma_head:
        from .q6_mmvq_mma import Q6MMVQMMA
    if use_shared_q8:
        if not use_direct:
            raise ValueError("Shared Q8 requires GGUF_PARITY_CURRENT_DIRECT=1")
        from .current_ggml_direct import CurrentGGMLSharedLinear
    if use_body_mma:
        if not use_direct or not use_shared_q8:
            raise ValueError("Body MMA requires current-direct and shared-Q8 opt-ins")
        from .current_ggml_body_mma import (
            CurrentGGMLBodyMMALinear,
            body_mma_shape_policy,
            install_body_decode_guard,
        )

        install_body_decode_guard()

    if use_direct:
        from .current_ggml_direct import (
            CurrentGGMLDirectLinear as CurrentGGMLLinear,
        )
        from .current_ggml_direct import (
            CurrentGGMLSSMLinear,
        )

        _install_ssm_decode_guard()
    else:
        from .current_ggml_cuda import CurrentGGMLLinear

    class CurrentLinear(torch.nn.Module):
        def __init__(self, data, quant_type, dtype, *, q6_mma=False, ssm=False):
            super().__init__()
            block, size = gguf.GGML_QUANT_SIZES[int(quant_type)]
            self.shape = (data.shape[0], data.shape[1] // size * block)
            self.dtype = dtype
            self.ssm = ssm
            # The direct operator owns Torch storage; the graph bridge owns a
            # native allocation. Neither retains a duplicate packed matrix.
            packed = data.detach() if use_direct else data.detach().cpu()
            if q6_mma:
                self.operation = Q6MMVQMMA(packed.contiguous(), self.shape[1])
            else:
                implementation = (
                    CurrentGGMLSSMLinear if ssm and use_direct else CurrentGGMLLinear
                )
                self.operation = implementation(
                    packed.contiguous(), int(quant_type), self.shape[1]
                )

        def forward(self, x, bias=None):
            if x.numel() == 0:
                return x.new_empty((*x.shape[:-1], self.shape[0]))
            flat = x.reshape(-1, x.shape[-1]).float().contiguous()
            if self.ssm and not use_direct:
                from .current_ggml_direct import _runtime_ssm_groups

                y = flat.new_zeros((flat.shape[0], self.shape[0]))
                for begin, t, b in _runtime_ssm_groups(flat.shape[0]):
                    end = begin + t * b
                    y[begin:end] = self.operation.forward_channels(
                        flat[begin:end].view(b, t, self.shape[1])
                    ).view(t * b, self.shape[0])
            else:
                y = self.operation(flat)
            y = y.to(self.dtype).reshape(*x.shape[:-1], self.shape[0])
            return y if bias is None else y + bias

    def prepare(self, layer):
        # MMQ stream-K reduction depends on each original matrix's row count.
        parts = _original_matrix_parts(layer)
        ssm = _is_ssm_output(layer)
        if ssm and (len(parts) != 1 or self.params_dtype != torch.float32):
            raise ValueError("Native SSM output requires one FP32-activation matrix")
        if ssm:
            layer._gguf_parity_ssm_channels = True
        q6_mma = (
            use_q6_mma_head
            and getattr(layer, "_gguf_tied_lm_head", False)
            and _q6_mma_head_supported(parts)
        )
        if q6_mma:
            if self.params_dtype != torch.float32:
                raise ValueError(
                    "Q6 MMA head diagnostic requires the original tied "
                    "248320x1024 Q6_K matrix and FP32 model activations"
                )
            layer._gguf_parity_q6_mma_head = True
        body_mma = (
            use_body_mma
            and not ssm
            and ".layers." in f".{getattr(layer, 'prefix', '')}"
            and not getattr(layer, "_gguf_tied_lm_head", False)
            and any(
                body_mma_shape_policy(
                    int(qtype),
                    data.shape[0],
                    data.shape[1]
                    // gguf.GGML_QUANT_SIZES[int(qtype)][1]
                    * gguf.GGML_QUANT_SIZES[int(qtype)][0],
                    8,
                )
                for data, qtype in parts
            )
        )
        if body_mma or (use_shared_q8 and len(parts) > 1):
            if self.params_dtype != torch.float32:
                raise ValueError(
                    "Shared native Q8 diagnostic requires FP32 activations"
                )
            block, size = gguf.GGML_QUANT_SIZES[int(parts[0][1])]
            width = parts[0][0].shape[1] // size * block
            implementation = (
                CurrentGGMLBodyMMALinear if body_mma else CurrentGGMLSharedLinear
            )
            kwargs = {"decode_only": True} if body_mma else {}
            layer.gguf_humming = torch.nn.ModuleList(
                [implementation(parts, width, **kwargs)]
            )
            if len(parts) > 1:
                layer._gguf_parity_shared_q8 = True
            if body_mma:
                layer._gguf_parity_body_mma = True
        else:
            layer.gguf_humming = torch.nn.ModuleList(
                CurrentLinear(
                    data, quant_type, self.params_dtype, q6_mma=q6_mma, ssm=ssm
                )
                for data, quant_type in parts
            )
        if not getattr(layer, "_gguf_keep_embedding_weight", False):
            del layer.qweight

    GGUFLinearMethod._prepare_humming_weights = prepare
    GGUFLinearMethod._parity_current_llama = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # The production chunked GDN entry point rejects F32. For this batch-one
    # diagnostic, use the existing recurrent kernel over the complete prefill.
    # Its state history is temporary; return only the last state to vLLM.
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    original_chunk = gdn.fla_chunk_gated_delta_rule

    def chunk(**kwargs):
        if kwargs["q"].dtype != torch.float32:
            return original_chunk(**kwargs)
        assert kwargs["initial_state"].shape[0] == 1, "Diagnostic requires batch one"
        assert kwargs["q"].shape[0] == 1
        assert kwargs.get("cu_seqlens") is None or len(kwargs["cu_seqlens"]) == 2
        output, history = fused_recurrent_gated_delta_rule(
            q=kwargs["q"],
            k=kwargs["k"],
            v=kwargs["v"],
            g=kwargs["g"],
            beta=kwargs["beta"],
            initial_state=kwargs["initial_state"],
            inplace_final_state=False,
            cu_seqlens=kwargs.get("cu_seqlens"),
            use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", False),
        )
        if kwargs.get("core_attn_out") is not None:
            kwargs["core_attn_out"].copy_(output)
        return output, history[-1:].clone()

    gdn.fla_chunk_gated_delta_rule = chunk

    if os.environ.get("GGUF_PARITY_TRITON_IEEE") == "1":
        from triton.backends.nvidia.compiler import CUDABackend

        original_options = CUDABackend.parse_options

        def options(self, values):
            values = dict(values)
            values["default_dot_input_precision"] = "ieee"
            return original_options(self, values)

        CUDABackend.parse_options = options

    if use_direct:
        from .current_float_vllm_adapter import install_current_float_adapter
        from .native_projection_fusion import install_native_projection_fusions

        install_current_float_adapter()
        install_native_projection_fusions()
