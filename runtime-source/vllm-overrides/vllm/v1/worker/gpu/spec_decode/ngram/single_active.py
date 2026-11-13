# SPDX-License-Identifier: Apache-2.0
"""Default-off K3 singleton capability; ordinary public batches stay public.

The legacy B1 runtime remains untouched. This adapter binds a fixed scalar
predicate to ABI2; persistent request-state counts are staged on serving stream.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .conditional import ConditionalNgramRuntime

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ADAPTERS = {
    (8, 6): (
        "23b47047f1536d1eadf040fff3dbda2e822082888a0b7d4628552ed777a5fd04",
        "d51e894d27e8aa8b6328ef7ac8c987c08b8413999a5c49842dcdf0d6093b6f4d",
    ),
    (12, 0): (
        "393b04653df66321daf37746f00db2b4175195ef13081927f4f1836919e8e7a0",
        "66329fff2778dcdd60f9b4590117d4c672b0b0704fe24b17a571959116a62d79",
    ),
}


def adapter_identity(record):
    if not isinstance(record, dict) or record.get("ABI") != 2:
        raise ValueError(
            "A source-bound normal-catalog ABI2 adapter record is required"
        )
    target = tuple(record.get("compute_capability", ()))
    if target not in ADAPTERS:
        raise ValueError("Single-active adapter requires qualified SM86/SM120")
    metadata = record.get("metadata")
    if not isinstance(metadata, list) or len(metadata) != 1:
        raise ValueError("Require the actual one-file raw ABI2 build identity")
    binary, meta = Path(record["path"]).absolute(), Path(metadata[0]["path"]).absolute()
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    raw = meta.read_bytes()
    meta_sha = hashlib.sha256(raw).hexdigest()
    if (
        (actual, meta_sha) != ADAPTERS[target]
        or record.get("sha256") != actual
        or metadata[0].get("sha256") != meta_sha
    ):
        raise ValueError(
            "ABI2 binary/raw-metadata/target differs from qualified catalog"
        )
    build = json.loads(raw)
    if (
        build.get("library_sha256") != actual
        or "abi2" not in build.get("contract", "").lower()
        or build.get("source_sha256") != record.get("source_sha256")
    ):
        raise ValueError("ABI2 source/interface identity differs")
    return {
        "path": str(binary),
        "sha256": actual,
        "metadata": str(meta),
        "metadata_sha256": meta_sha,
        "compute_capability": target,
        "ABI": 2,
    }


def singleton_descriptor(desc, query):
    return (
        getattr(getattr(desc, "cg_mode", None), "name", None) == "FULL"
        and desc.num_reqs == 1
        and desc.num_tokens == query
        and desc.uniform_token_count == query
        and getattr(desc, "num_active_loras", 0) == 0
        and getattr(desc, "num_ubatches", 1) == 1
    )


def singleton_step_eligible(batch, desc, *, has_prefill, greedy, dummy=False):
    return (
        not dummy
        and not has_prefill
        and greedy is True
        and batch.num_reqs == 1
        and batch.num_tokens == 4
        and batch.num_draft_tokens == 3
        and tuple(int(v) for v in batch.num_scheduled_tokens) == (4,)
        and len(batch.idx_mapping_np) == 1
        and singleton_descriptor(desc, 4)
    )


def _load_library(path):
    lib = ctypes.CDLL(path)
    lib.yaqa_ngram_conditional_abi_version.restype = ctypes.c_int
    if lib.yaqa_ngram_conditional_abi_version() != 2:
        raise ValueError("Single-active adapter must expose ABI2")
    lib.yaqa_ngram_conditional_create.argtypes = [ctypes.c_void_p] * 6 + [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.yaqa_ngram_conditional_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.yaqa_ngram_conditional_destroy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    for name in ("create", "launch", "destroy"):
        getattr(lib, "yaqa_ngram_conditional_" + name).restype = ctypes.c_int
    return lib


def validate_single_active_profile(config: VllmConfig) -> None:
    import torch

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    spec = config.speculative_config
    assert spec is not None and spec.ngram_gpu_single_active
    adapter_identity(spec.ngram_gpu_single_active_adapter)
    piecewise_prefill = (
        config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    expected_graph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE
        if piecewise_prefill
        else CUDAGraphMode.FULL_DECODE_ONLY
    )
    if piecewise_prefill:
        from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled

        sizes = set(config.compilation_config.cudagraph_capture_sizes or [])
        prefill_sizes = {size for size in sizes if size > 4}
        if (
            not is_breakable_cudagraph_enabled()
            or not prefill_sizes
            or max(sizes) > config.scheduler_config.max_num_batched_tokens
            or config.compilation_config.max_cudagraph_capture_size < max(sizes)
        ):
            raise ValueError(
                "Conditional piecewise prefill requires stock breakable graphs "
                "and prefill capture sizes covered by the token/capture budgets"
            )
    scheduler = config.scheduler_config
    model = config.model_config
    parallel = config.parallel_config
    backend = config.attention_config.backend
    if (
        not config.use_v2_model_runner
        or scheduler.async_scheduling is not True
        or scheduler.max_num_seqs < 1
        or scheduler.scheduler_cls is not None
        or scheduler.max_num_batched_tokens < 4
        or (
            scheduler.max_num_scheduled_tokens is not None
            and scheduler.max_num_scheduled_tokens < 4
        )
        or 0 < scheduler.long_prefill_token_threshold < 4
        or config.cache_config.enable_prefix_caching
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.enable_dbo
        or model is None
        or model.hf_text_config.model_type != "lfm3"
        or model.dtype not in (torch.float32, torch.bfloat16)
        or model.is_multimodal_model
        or model.runner_type != "generate"
        or getattr(backend, "name", backend) != "TRITON_ATTN"
        or config.compilation_config.mode != CompilationMode.NONE
        or config.compilation_config.cudagraph_mode != expected_graph_mode
        or config.lora_config is not None
        or config.aux_output_config.enabled
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
    ):
        raise ValueError(
            "ngram_gpu_single_active requires F32 or BF16 LFM3/TRITON_ATTN, Model Runner V2, "
            "async positive serving capacity, single rank, no prefix cache/LoRA/aux/transfer, "
            "mode0 FULL_DECODE_ONLY or explicit stock piecewise prefill, "
            "and scheduling capacity for four queries."
        )
    if 4 not in (config.compilation_config.cudagraph_capture_sizes or []):
        raise ValueError("ngram_gpu_single_active requires public capture size4")


class SingleActiveConditionalNgramRuntime(ConditionalNgramRuntime):
    single_active = True

    def __init__(self, config, buffers, state_counts):
        import torch

        validate_single_active_profile(config)
        self.adapter = adapter_identity(
            config.speculative_config.ngram_gpu_single_active_adapter
        )
        actual = tuple(torch.cuda.get_device_capability(buffers.device))
        if actual != self.adapter["compute_capability"]:
            raise ValueError(
                "Actual CUDA tensor device differs from ABI2 adapter target"
            )
        if (
            state_counts.dtype != torch.int32
            or not state_counts.is_cuda
            or state_counts.device != buffers.device
            or not state_counts.is_contiguous()
            or tuple(state_counts.shape) != (buffers.max_num_reqs,)
        ):
            raise ValueError("Persistent request-state count capacity/dtype differs")
        self.config = config
        self.state_counts = state_counts
        prefix = copy.copy(buffers)
        prefix.max_num_reqs = 1
        prefix.query_start_loc = buffers.query_start_loc[:2]
        prefix.seq_lens = buffers.seq_lens[:1]
        prefix.dcp_local_seq_lens = buffers.dcp_local_seq_lens[:1]
        self.buffers = prefix
        self.m1_buffers = copy.copy(prefix)
        self.m1_buffers.query_start_loc = torch.tensor(
            [0, 1], dtype=torch.int32, device=buffers.device
        )
        self.m1_buffers.seq_lens = torch.ones(
            1, dtype=torch.int32, device=buffers.device
        )
        self.match_counts = torch.zeros(1, dtype=torch.int32, device=buffers.device)[:1]
        self.branch_counts = torch.zeros(2, dtype=torch.int64, device=buffers.device)
        self.children = ()
        self.handle = ctypes.c_void_p()
        self.message = ctypes.create_string_buffer(2048)
        self.streams = set()
        self.active_singleton = False
        # Load/pin ABI and ctypes widths BEFORE any graph capture.
        self.lib = _load_library(self.adapter["path"])

    def attach(self, m1, m4):
        import torch

        self.close()
        self.children = (m1, m4)
        with torch.cuda.device(self.buffers.device):
            self._check(
                self.lib.yaqa_ngram_conditional_create(
                    m1.raw_cuda_graph(),
                    m4.raw_cuda_graph(),
                    self.match_counts.data_ptr(),
                    self.buffers.seq_lens.data_ptr(),
                    self.m1_buffers.seq_lens.data_ptr(),
                    self.branch_counts.data_ptr(),
                    ctypes.byref(self.handle),
                    self.message,
                    len(self.message),
                )
            )

    def begin_step(self, batch, desc, *, has_prefill, greedy, dummy=False):
        batch.ngram_single_active = False
        self.active_singleton = False
        if not self.handle.value or not singleton_step_eligible(
            batch, desc, has_prefill=has_prefill, greedy=greedy, dummy=dummy
        ):
            return False
        index = int(batch.idx_mapping_np[0])
        if not 0 <= index < self.state_counts.numel():
            raise ValueError(
                "Actual active request-state index is outside count capacity"
            )
        # CPU-known index creates a GPU view; copy is ordered before replay on
        # the same serving stream. ABI2 predicate pointer never changes.
        self.match_counts.copy_(self.state_counts[index : index + 1])
        self.active_singleton = True
        batch.ngram_single_active = True
        return True

    def invalid_drafts(self, local_positions):
        return (
            super().invalid_drafts(local_positions) if self.active_singleton else None
        )

    def close(self):
        super().close()
        self.active_singleton = False
