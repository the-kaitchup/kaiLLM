# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-request GPU match-only conditional body graphs."""

from __future__ import annotations

import copy
import ctypes
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.input_batch import InputBuffers


def library_identity(library: str | None) -> tuple[str, str]:
    if library is None or not Path(library).is_file():
        raise ValueError("ngram_gpu_match_only requires an existing ABI2 CUDA library")
    path = Path(library).resolve()
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def validate_profile(config: VllmConfig) -> None:
    import torch

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    spec = config.speculative_config
    assert spec is not None and spec.ngram_gpu_match_only
    library_identity(spec.ngram_gpu_match_only_library)
    piecewise_prefill = spec.ngram_gpu_match_only_piecewise_prefill
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
        or scheduler.max_num_seqs != 1
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
            "ngram_gpu_match_only requires F32 or BF16 LFM3/TRITON_ATTN, Model Runner V2, "
            "async max_num_seqs=1, single rank, no prefix cache/LoRA/aux/transfer, "
            "mode0 FULL_DECODE_ONLY or explicit stock piecewise prefill, "
            "and scheduling capacity for four queries."
        )
    if not {1, 4}.issubset(config.compilation_config.cudagraph_capture_sizes or []):
        raise ValueError("ngram_gpu_match_only requires capture sizes1 and4")


class ConditionalNgramRuntime:
    def __init__(
        self, config: VllmConfig, buffers: InputBuffers, match_counts: torch.Tensor
    ):
        import torch

        validate_profile(config)
        if (
            config.speculative_config.ngram_gpu_match_only_piecewise_prefill
            and buffers.device.type == "cuda"
            and torch.cuda.get_device_capability(buffers.device)
            not in ((8, 6), (12, 0))
        ):
            raise ValueError("Conditional piecewise prefill is qualified on SM86/SM120")
        assert buffers.max_num_reqs == 1
        assert match_counts.shape == (1,) and match_counts.dtype == torch.int32
        self.config = config
        self.match_counts = match_counts
        self.buffers = buffers
        self.m1_buffers = copy.copy(buffers)
        self.m1_buffers.query_start_loc = torch.tensor(
            [0, 1], dtype=torch.int32, device=buffers.device
        )
        self.m1_buffers.seq_lens = torch.ones(
            1, dtype=torch.int32, device=buffers.device
        )
        self.branch_counts = torch.zeros(2, dtype=torch.int64, device=buffers.device)
        self.children = ()
        self.handle = ctypes.c_void_p()
        self.message = ctypes.create_string_buffer(2048)
        self.streams = set()
        self.lib = None

    def _check(self, status):
        if status:
            raise RuntimeError(self.message.value.decode() + f" (CUDA error{status})")

    def attach(self, m1, m4):
        import torch

        self.close()
        path, _ = library_identity(
            self.config.speculative_config.ngram_gpu_match_only_library
        )
        lib = ctypes.CDLL(path)
        lib.yaqa_ngram_conditional_abi_version.restype = ctypes.c_int
        if lib.yaqa_ngram_conditional_abi_version() != 2:
            raise RuntimeError("GPU match-only CUDA adapter must use ABI2")
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
        for name in ["create", "launch", "destroy"]:
            getattr(lib, "yaqa_ngram_conditional_" + name).restype = ctypes.c_int
        assert self.match_counts.is_cuda
        self.lib = lib
        self.children = (m1, m4)
        with torch.cuda.device(self.buffers.device):
            self._check(
                lib.yaqa_ngram_conditional_create(
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

    def replay(self):
        import torch

        if not self.handle.value:
            raise RuntimeError("GPU match-only graph is not captured")
        stream = torch.cuda.current_stream(self.buffers.device)
        self.streams.add(stream)
        self._check(
            self.lib.yaqa_ngram_conditional_launch(
                self.handle, stream.cuda_stream, self.message, len(self.message)
            )
        )

    def invalid_drafts(self, local_positions):
        from vllm.v1.worker.gpu.spec_decode.ngram.conditional_kernels import (
            invalid_drafts,
        )

        return invalid_drafts(self.match_counts, local_positions)

    def close(self):
        if not self.handle.value:
            self.children = ()
            return
        for stream in self.streams:
            stream.synchronize()
        self._check(
            self.lib.yaqa_ngram_conditional_destroy(
                self.handle, self.message, len(self.message)
            )
        )
        self.handle = ctypes.c_void_p()
        self.children = ()
        self.streams.clear()
