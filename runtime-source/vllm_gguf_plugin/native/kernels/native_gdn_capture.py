# SPDX-License-Identifier: Apache-2.0
"""Optional batched FP32 native-GDN capture inside vLLM's PIECEWISE graphs.

Call install_native_gdn_capture() after installing the native GDN/norm helpers
and before constructing LLM, in both parent and spawned worker. Set:
    additional_config['gguf_parity_capture_gdn'] = True
    compilation_config = native_gdn_capture_compilation_config()

Only the plain GDN core is removed from split points. Standard attention and
KV-cache writes remain dynamic. The GDN builder stages changing cache-slot
indices in its existing persistent decode buffers. Fresh one-token prompts
cannot replay the decode graph, since their state-initialization branch differs.
This changes launch scheduling only; it does not replace arithmetic kernels.

Scope: one CUDA device, FP32, no speculation/LoRA/DBO, MRV2, one decode token
per active request, and the existing native diagnostic GDN/norm helpers.
Default execution is unchanged unless this installer and config opt-in are used.
"""

import os
from dataclasses import replace
from functools import wraps

_MARKER = "gguf_parity_capture_gdn"
_GDN_OP = "vllm::qwen_gdn_attention_core"
_KV_OPS = ("vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update")


def native_gdn_capture_compilation_config(capture_sizes=(1,)):
    """Return the ordinary PIECEWISE config with just plain GDN captured."""
    from vllm.config.compilation import CompilationConfig

    sizes = sorted(set(capture_sizes))
    if not sizes or any(
        type(size) is not int or not 1 <= size <= 65535 for size in sizes
    ):
        raise ValueError("Native GDN graph capture sizes must be positive integers")
    operations = list(CompilationConfig._attention_ops)
    if operations.count(_GDN_OP) != 1:
        raise RuntimeError("Unexpected upstream GDN splitting-op configuration")
    operations.remove(_GDN_OP)
    operations.extend(op for op in _KV_OPS if op not in operations)
    return {
        "mode": 3,
        "cudagraph_mode": "PIECEWISE",
        "cudagraph_capture_sizes": sizes,
        "splitting_ops": operations,
        "use_inductor_graph_partition": False,
    }


def _enabled(config):
    return (
        isinstance(config.additional_config, dict)
        and config.additional_config.get(_MARKER, False) is True
    )


def _validate(config, device):
    import torch
    from vllm import envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    cc = config.compilation_config
    parallel = config.parallel_config
    if (
        device.type != "cuda"
        or config.model_config.dtype != torch.float32
        or not 1 <= config.scheduler_config.max_num_seqs <= 65535
        or config.speculative_config is not None
        or config.lora_config is not None
        or not config.use_v2_model_runner
        or any(
            getattr(parallel, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or getattr(parallel, "enable_dbo", False)
    ):
        raise ValueError(
            "Native GDN capture requires CUDA FP32 MRV2, "
            "TP/PP/DP/DCP=1, no MTP/LoRA/DBO"
        )
    if (
        cc.mode != CompilationMode.VLLM_COMPILE
        or cc.cudagraph_mode != CUDAGraphMode.PIECEWISE
        or cc.cudagraph_capture_sizes is None
        or not cc.cudagraph_capture_sizes
        or any(
            not 1 <= size <= config.scheduler_config.max_num_seqs
            for size in cc.cudagraph_capture_sizes
        )
        or cc.use_inductor_graph_partition
        or cc.splitting_ops is None
        or _GDN_OP in cc.splitting_ops
        or "vllm::unified_attention_with_output" not in cc.splitting_ops
        or any(op not in cc.splitting_ops for op in _KV_OPS)
        or envs.VLLM_USE_BREAKABLE_CUDAGRAPH
    ):
        raise ValueError("Use native_gdn_capture_compilation_config() with BREAKABLE=0")


def _restrict_replay(
    descriptor,
    *,
    num_reqs,
    num_tokens,
    uniform_token_count,
    num_active_loras,
    num_ubatches,
):
    from vllm.config.compilation import CUDAGraphMode

    if descriptor.cg_mode != CUDAGraphMode.PIECEWISE:
        return descriptor
    if num_reqs < 1 or num_active_loras or num_ubatches != 1:
        raise ValueError("Native GDN graph replay requires non-LoRA requests, no DBO")
    if num_tokens == num_reqs and uniform_token_count == 1:
        return descriptor
    # MRV2 supplies uniform_token_count=None for a fresh one-token prefill.
    # PIECEWISE's default wildcard would otherwise match its decode graph.
    return replace(
        descriptor,
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform_token_count=uniform_token_count,
        max_query_len=None,
    )


def _clear_decode_padding(builder, metadata, *, query_starts):
    """Clear persistent tails beyond the actual request count.

    PIECEWISE attention metadata is unpadded. A captured larger GDN graph still
    reads its original full view of these same persistent buffers; clearing
    only metadata's shorter view would leave stale, live cache-slot indices.
    """
    if metadata.num_prefills or metadata.num_spec_decodes:
        return
    count = metadata.num_decodes
    builder.non_spec_state_indices_tensor[count:].zero_()  # NULL_BLOCK_ID == 0
    if query_starts:
        starts = builder.non_spec_query_start_loc
        starts[count + 1 :].copy_(
            starts[count : count + 1].expand_as(starts[count + 1 :])
        )


def install_native_gdn_capture():
    """Install process-local hooks; no vLLM or default adapter files are changed."""
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if not getattr(gdn, "_gguf_native_order_gdn", False) or not getattr(
        gdn, "_current_native_conv", False
    ):
        raise RuntimeError("Install native GDN and native norms/convolution first")
    if getattr(GDNAttentionMetadataBuilder, "_gguf_native_capture_hook", False):
        return
    original_init = GDNAttentionMetadataBuilder.__init__
    original_build = GDNAttentionMetadataBuilder.build
    original_update = GDNAttentionMetadataBuilder.update_block_table
    original_dispatch = CudaGraphManager.dispatch

    @wraps(original_init)
    def initialize(self, kv_cache_spec, layer_names, vllm_config, device):
        enabled = _enabled(vllm_config)
        if enabled:
            _validate(vllm_config, device)
        original_init(self, kv_cache_spec, layer_names, vllm_config, device)
        if enabled:
            if type(self) is not GDNAttentionMetadataBuilder or self.num_spec != 0:
                raise ValueError(
                    "Native GDN capture only supports the non-speculative GDN builder"
                )
            # Only controls _stage_decode / _stage_spec_decode in this builder;
            # it does not switch attention or the graph manager into FULL mode.
            # build() and update_block_table() now both refresh persistent IDs.
            self.use_full_cuda_graph = True
            self._gguf_native_gdn_capture = True

    @wraps(original_build)
    def build(self, *args, **kwargs):
        result = original_build(self, *args, **kwargs)
        if getattr(self, "_gguf_native_gdn_capture", False):
            _clear_decode_padding(self, result, query_starts=True)
        return result

    @wraps(original_update)
    def update_block_table(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        if getattr(self, "_gguf_native_gdn_capture", False):
            # Query starts remain shared with the source group's metadata.
            _clear_decode_padding(self, result, query_starts=False)
        return result

    @wraps(original_dispatch)
    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
        num_ubatches=1,
    ):
        result = original_dispatch(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len,
            num_ubatches,
        )
        if _enabled(self.vllm_config):
            result = _restrict_replay(
                result,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_token_count,
                num_active_loras=num_active_loras,
                num_ubatches=num_ubatches,
            )
        return result

    GDNAttentionMetadataBuilder.__init__ = initialize
    GDNAttentionMetadataBuilder.build = build
    GDNAttentionMetadataBuilder.update_block_table = update_block_table
    GDNAttentionMetadataBuilder._gguf_native_capture_hook = True
    CudaGraphManager.dispatch = dispatch

    from .native_model_decode_graph import install_native_model_decode_graph

    install_native_model_decode_graph()
