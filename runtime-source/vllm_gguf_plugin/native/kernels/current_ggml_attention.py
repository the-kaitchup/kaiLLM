# SPDX-License-Identifier: Apache-2.0
"""Ragged-batch diagnostic using local llama.cpp CUDA FlashAttention kernels.

Build: PYTHONPATH=benchmarks python -m kernels.current_ggml_attention --build
The private GGML ABI must match the locally linked library. No production
backend is changed. Torch owns memory and the stream; attention itself does no
copies/synchronizations. The paged-cache adapter gathers/casts K/V on device.
By default each request keeps the B1 native launch topology; the C++ dispatcher
issues those launches within one Torch op. An opt-in 4-D batch backend handles
uniform decode, and an independent opt-in workspace updates KV incrementally.
"""

import argparse
import ctypes
import json
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from .current_ggml_cuda import _cuda_library_dir, _paths


def _path():
    return Path(
        os.environ.get(
            "GGUF_CURRENT_ATTN", str(Path(__file__).resolve().parents[1] / "libraries/sm86/libcurrent_attention.so")
        )
    )


def build():
    _, directory, cuda, _ = _paths()
    cuda_library = _cuda_library_dir(cuda)
    entries = json.loads((directory / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/ggml-cuda.cu"))
    command = shlex.split(entry["command"])
    library = _path()
    obj = library.with_suffix(".o")
    command[command.index(entry["file"])] = str(Path(__file__).with_suffix(".cu"))
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=entry["directory"], check=True)
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
    return library


@lru_cache(maxsize=1)
def _library():
    library = ctypes.CDLL(str(_path()))
    library.current_attention_error.restype = ctypes.c_char_p
    library.current_attention_run.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_size_t]
        + [ctypes.c_int] * 6
        + [ctypes.c_float, ctypes.c_size_t]
    )
    library.current_attention_run.restype = ctypes.c_int
    if hasattr(library, "current_attention_many_run"):
        library.current_attention_many_run.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_size_t]
            + [ctypes.c_int] * 4
            + [ctypes.POINTER(ctypes.c_int32)] * 2
            + [ctypes.c_float, ctypes.c_size_t]
        )
        library.current_attention_many_run.restype = ctypes.c_int
    if hasattr(library, "current_attention_many_strided_run"):
        library.current_attention_many_strided_run.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_size_t]
            + [ctypes.c_int] * 5
            + [ctypes.POINTER(ctypes.c_int32)] * 2
            + [ctypes.c_float, ctypes.c_size_t]
        )
        library.current_attention_many_strided_run.restype = ctypes.c_int
    return library


@torch.library.custom_op("gguf_current::attention", mutates_args=())
def native_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """F32 Q/out [T,8,256]; half K/V [padded_K,2,256], mask [T,padded_K]."""
    if (
        q.ndim != 3
        or q.shape[1:] != (8, 256)
        or q.dtype != torch.float32
        or k.ndim != 3
        or k.shape[1:] != (2, 256)
        or k.shape[0] % 256
        or v.shape != k.shape
        or mask.shape != (q.shape[0], k.shape[0])
        or any(t.dtype != torch.float16 for t in (k, v, mask))
        or any(
            not t.is_cuda or not t.is_contiguous() or t.device != q.device
            for t in (q, k, v, mask)
        )
    ):
        raise ValueError("Expected contiguous B1 Qwen3.5 F32 Q and padded F16 KV/mask")
    output = torch.empty_like(q)
    # Conservative for this B1/head256 diagnostic; GGML arena checks every
    # allocation and fails explicitly if a future device/shape needs more.
    scratch = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    library = _library()
    with torch.cuda.device(q.device):
        status = library.current_attention_run(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            mask.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            scratch.numel(),
            q.device.index,
            q.shape[0],
            k.shape[0],
            q.shape[1],
            k.shape[1],
            q.shape[2],
            scale,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_attention_error().decode())
    return output


@native_attention.register_fake
def _fake(q, k, v, mask, scale):
    return torch.empty_like(q)


@torch.library.custom_op("gguf_current::attention_many", mutates_args=())
def native_attention_many(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    query_counts: list[int],
    kv_padded: list[int],
    scale: float,
) -> torch.Tensor:
    """Padded storage with independent native dimensions and B1 reductions.

    Q/out [B,Qmax,8,256], K/V [B,Kmax,2,256], mask [B,Qmax,Kmax].
    Host lists describe launch shapes only. GPU masks and indices remain live.
    """
    if (
        q.ndim != 4
        or q.shape[2:] != (8, 256)
        or q.dtype != torch.float32
        or k.ndim != 4
        or k.shape[0] != q.shape[0]
        or k.shape[2:] != (2, 256)
        or k.shape[1] % 256
        or v.shape != k.shape
        or mask.ndim != 3
        or mask.shape[:2] != q.shape[:2]
        or mask.shape[2] % 256
        or len(query_counts) != q.shape[0]
        or len(kv_padded) != q.shape[0]
        or any(t.dtype != torch.float16 for t in (k, v, mask))
        or any(
            not t.is_cuda or not t.is_contiguous() or t.device != q.device
            for t in (q, k, v, mask)
        )
        or any(n <= 0 or n > q.shape[1] for n in query_counts)
        or any(
            n <= 0 or n > min(k.shape[1], mask.shape[2]) or n % 256 for n in kv_padded
        )
        or any(nq > nk for nq, nk in zip(query_counts, kv_padded))
    ):
        raise ValueError("Invalid native attention ragged batch storage/shapes")
    # Unused query slots are deliberately initialized, never exposed by scatter.
    output = torch.zeros_like(q)
    scratch = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    counts = (ctypes.c_int32 * len(query_counts))(*query_counts)
    lengths = (ctypes.c_int32 * len(kv_padded))(*kv_padded)
    library = _library()
    if not hasattr(library, "current_attention_many_run"):
        raise RuntimeError("Rebuild the native attention helper for batch support")
    shape_args = [q.device.index, q.shape[0], q.shape[1], k.shape[1]]
    operation = library.current_attention_many_run
    if mask.shape[2] != k.shape[1]:
        if not hasattr(library, "current_attention_many_strided_run"):
            raise RuntimeError("Rebuild native attention for persistent KV strides")
        operation = library.current_attention_many_strided_run
        shape_args.append(mask.shape[2])
    with torch.cuda.device(q.device):
        status = operation(
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            mask.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            scratch.numel(),
            *shape_args,
            counts,
            lengths,
            scale,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    if status:
        raise RuntimeError(library.current_attention_error().decode())
    return output


@native_attention_many.register_fake
def _fake_many(q, k, v, mask, query_counts, kv_padded, scale):
    return torch.empty_like(q)


@triton.jit
def _gather_cache(
    cache,
    blocks,
    keys,
    values,
    seq_lens,
    STRIDE_B: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PADDED: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr = 0,
):
    request = tl.program_id(1)
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < PADDED * HEADS * DIM
    dim = offsets % DIM
    head = offsets // DIM % HEADS
    token = offsets // (DIM * HEADS)
    length = tl.load(seq_lens + request)
    valid_token = token < length
    block = tl.load(
        blocks + request * BLOCK_TABLE_STRIDE + token // BLOCK_SIZE,
        mask=valid & valid_token,
        other=0,
    ).to(tl.int64)
    address = (
        block * STRIDE_B
        + head * STRIDE_H
        + token % BLOCK_SIZE * STRIDE_T
        + dim * STRIDE_D
    )
    k = tl.load(cache + address, mask=valid & valid_token, other=0).to(tl.float16)
    v = tl.load(cache + address + DIM * STRIDE_D, mask=valid & valid_token, other=0).to(
        tl.float16
    )
    destination = request * PADDED * HEADS * DIM + offsets
    tl.store(keys + destination, k, valid)
    tl.store(values + destination, v, valid)


@triton.jit
def _causal_mask(
    mask, seq_lens, QUERIES: tl.constexpr, PADDED: tl.constexpr, TILE: tl.constexpr
):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    q = offsets // PADDED
    k = offsets % PADDED
    length = tl.load(seq_lens)
    visible = (k < length) & (k <= length - QUERIES + q)
    tl.store(mask + offsets, tl.where(visible, 0.0, float("-inf")), q < QUERIES)


@triton.jit
def _pack_queries(
    q,
    starts,
    seq_lens,
    packed,
    TOTAL: tl.constexpr,
    QUERIES: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(1)
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    local_q = offsets // 2048
    start = tl.load(starts + request)
    count = tl.load(starts + request + 1) - start
    active = tl.load(seq_lens + request) > 0
    valid = (local_q < count) & (start + local_q < TOTAL) & active
    value = tl.load(q + start * 2048 + offsets, mask=valid, other=0)
    tl.store(
        packed + request * QUERIES * 2048 + offsets, value, offsets < QUERIES * 2048
    )


@triton.jit
def _batched_causal_mask(
    mask,
    starts,
    seq_lens,
    QUERIES: tl.constexpr,
    PADDED: tl.constexpr,
    TILE: tl.constexpr,
    STORAGE_QUERIES: tl.constexpr = 0,
):
    request = tl.program_id(1)
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    local_q, key = offsets // PADDED, offsets % PADDED
    count = tl.load(starts + request + 1) - tl.load(starts + request)
    length = tl.load(seq_lens + request)
    active = (count > 0) & (length > 0)
    visible = (
        (local_q < count) & (key < length) & (key <= length - count + local_q) & active
    )
    # Inactive request slots still use a valid zero-valued attention row.
    # All -inf would produce NaNs; scatter suppresses these dummy results.
    visible |= (~active) & (local_q == 0) & (key == 0)
    tl.store(
        mask
        + request * (STORAGE_QUERIES if STORAGE_QUERIES else QUERIES) * PADDED
        + offsets,
        tl.where(visible, 0.0, float("-inf")),
        local_q < QUERIES,
    )


@triton.jit
def _scatter_queries(
    packed,
    starts,
    seq_lens,
    output,
    TOTAL: tl.constexpr,
    QUERIES: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(1)
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    local_q = offsets // 2048
    start = tl.load(starts + request)
    count = tl.load(starts + request + 1) - start
    valid = (local_q < count) & (local_q < QUERIES) & (start + local_q < TOTAL)
    active = tl.load(seq_lens + request) > 0
    value = tl.load(
        packed + request * QUERIES * 2048 + offsets, mask=valid & active, other=0
    )
    tl.store(output + start * 2048 + offsets, value, valid)


def _launch_shapes(query_starts, seq_lengths):
    """CPU-only launch plan; exact non-speculative vLLM metadata is required."""
    if len(query_starts) != len(seq_lengths) + 1 or query_starts[0] != 0:
        raise ValueError("Expected cumulative query starts and per-request lengths")
    counts = [b - a for a, b in zip(query_starts, query_starts[1:])]
    if any(n < 0 or length < n for n, length in zip(counts, seq_lengths)):
        raise ValueError("Invalid ragged query/context lengths")
    # Decode has one fixed slot per request even for inactive rows, permitting
    # live GPU query starts/activity to change on graph replay.
    return [max(1, n) for n in counts], [
        max(256, triton.cdiv(length, 256) * 256) for length in seq_lengths
    ]


def _uniform_prefill_batch(query_starts, seq_lengths):
    """Only actual equal-length B>1 prefill groups; never B1 or padded slots."""
    if (
        len(seq_lengths) <= 1
        or len(query_starts) != len(seq_lengths) + 1
        or query_starts[0] != 0
    ):
        return False
    counts = [stop - start for start, stop in zip(query_starts, query_starts[1:])]
    return counts[0] > 1 and all(
        count == counts[0] and length >= count
        for count, length in zip(counts, seq_lengths)
    )


def paged_native_attention(
    q,
    kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    scale,
    query_start_loc=None,
    query_starts_cpu=None,
    seq_lens_cpu=None,
    workspace=None,
):
    """Gather the existing vLLM cache; preserve its request and cache semantics."""
    requests = len(seq_lens)
    if block_table.shape[0] != requests or block_table.stride(1) != 1:
        raise ValueError("Expected one contiguous block-table row per request")
    if kv_cache.ndim != 4 or kv_cache.shape[1] != 2 or kv_cache.shape[3] != 512:
        raise ValueError("Expected vLLM Triton cache [blocks,2,block_size,512]")
    if kv_cache.dtype not in (torch.float32, torch.float16):
        raise ValueError("Native attention diagnostic requires unquantized cache")
    if (
        q.ndim != 3
        or q.shape[1:] != (8, 256)
        or q.dtype != torch.float32
        or not q.is_cuda
        or kv_cache.device != q.device
        or block_table.device != q.device
        or seq_lens.device != q.device
        or seq_lens.dtype != torch.int32
        or block_table.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("Expected CUDA F32 queries and I32 cache metadata")
    if requests > 1 or query_start_loc is not None:
        if query_start_loc is None or query_starts_cpu is None or seq_lens_cpu is None:
            raise ValueError("Ragged attention requires CPU and GPU request metadata")
        if query_start_loc.shape != (requests + 1,) or len(seq_lens_cpu) != requests:
            raise ValueError("Ragged attention metadata shape mismatch")
        if (
            query_start_loc.dtype != torch.int32
            or query_start_loc.device != q.device
            or not query_start_loc.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA I32 query starts")
        counts, padding = _launch_shapes(query_starts_cpu, seq_lens_cpu)
        if query_starts_cpu[-1] > q.shape[0]:
            raise ValueError("Query starts exceed the input token storage")
        padded, max_queries = max(padding), max(counts)
        native_batch = os.environ.get("GGUF_PARITY_NATIVE_BATCH_FA") == "1" and all(
            stop - start == 1 and length > 1
            for start, stop, length in zip(
                query_starts_cpu, query_starts_cpu[1:], seq_lens_cpu
            )
        )
        native_prefill = os.environ.get(
            "GGUF_PARITY_NATIVE_PREFILL_BATCH_FA"
        ) == "1" and _uniform_prefill_batch(query_starts_cpu, seq_lens_cpu)
        if workspace is None:
            keys = torch.empty(
                (requests, padded, 2, 256), dtype=torch.float16, device=q.device
            )
            values = torch.empty_like(keys)
            _gather_cache[(triton.cdiv(padded * 512, 256), requests)](
                kv_cache,
                block_table,
                keys,
                values,
                seq_lens,
                *kv_cache.stride(),
                kv_cache.shape[2],
                2,
                256,
                padded,
                256,
                block_table.stride(0),
            )
        else:
            if padded > workspace.context:
                raise ValueError("Native KV workspace context capacity exceeded")
            keys, values = workspace.update(
                kv_cache,
                block_table,
                query_start_loc,
                seq_lens,
                padded if native_batch or native_prefill else 0,
            )
        packed_q = torch.empty(
            (requests, max_queries, 8, 256), dtype=torch.float32, device=q.device
        )
        mask_queries = (
            triton.cdiv(max_queries, 64) * 64
            if native_batch or native_prefill
            else max_queries
        )
        mask = torch.empty(
            (requests, mask_queries, padded), dtype=torch.float16, device=q.device
        )
        _pack_queries[(triton.cdiv(max_queries * 2048, 256), requests)](
            q.contiguous(),
            query_start_loc,
            seq_lens,
            packed_q,
            q.shape[0],
            max_queries,
            256,
        )
        _batched_causal_mask[(triton.cdiv(mask_queries * padded, 256), requests)](
            mask,
            query_start_loc,
            seq_lens,
            mask_queries,
            padded,
            256,
        )
        if native_prefill:
            from .current_ggml_attention_batch import native_batch_prefill_attention

            result = native_batch_prefill_attention(
                packed_q, keys, values, mask, float(scale)
            )
        elif native_batch:
            from .current_ggml_attention_batch import native_batch_attention

            result = native_batch_attention(packed_q, keys, values, mask, float(scale))
        else:
            result = native_attention_many(
                packed_q, keys, values, mask, counts, padding, float(scale)
            )
        output = torch.zeros_like(q)
        _scatter_queries[(triton.cdiv(max_queries * 2048, 256), requests)](
            result,
            query_start_loc,
            seq_lens,
            output,
            q.shape[0],
            max_queries,
            256,
        )
        return output
    padded = triton.cdiv(max_seq_len, 256) * 256
    keys = torch.empty((padded, 2, 256), dtype=torch.float16, device=q.device)
    values = torch.empty_like(keys)
    mask = torch.empty((q.shape[0], padded), dtype=torch.float16, device=q.device)
    _gather_cache[(triton.cdiv(keys.numel(), 256),)](
        kv_cache,
        block_table,
        keys,
        values,
        seq_lens,
        *kv_cache.stride(),
        kv_cache.shape[2],
        2,
        256,
        padded,
        256,
    )
    _causal_mask[(triton.cdiv(mask.numel(), 256),)](
        mask,
        seq_lens,
        q.shape[0],
        padded,
        256,
    )
    return native_attention(q.contiguous(), keys, values, mask, float(scale))


def install_native_attention():
    """Opt-in ragged adapter; retains vLLM's normal KV cache writer."""
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionImpl,
        TritonAttentionMetadataBuilder,
    )

    from .current_ggml_norm import sigmoid

    original = TritonAttentionImpl.forward
    if getattr(original, "_current_native_attention", False):
        return
    original_build = TritonAttentionMetadataBuilder.build

    def build_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        config = self.vllm_config
        if config.speculative_config is not None:
            raise ValueError(
                "Native attention CPU launch planning excludes speculative decoding"
            )
        if (
            config.parallel_config.decode_context_parallel_size != 1
            or config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError("Native attention requires unsplit request contexts")
        metadata = original_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        lengths = common_attn_metadata.seq_lens_cpu_upper_bound
        if lengths is None:
            raise ValueError(
                "Native attention needs CPU context lengths; no GPU sync fallback"
            )
        metadata.native_query_starts = common_attn_metadata.query_start_loc_cpu.tolist()
        metadata.native_seq_lens = lengths.tolist()
        if os.environ.get("GGUF_PARITY_PERSISTENT_KV") == "1":
            if config.cache_config.enable_prefix_caching:
                raise ValueError(
                    "Native persistent KV requires prefix caching disabled"
                )
            metadata.native_workspace_shape = (
                config.scheduler_config.max_num_seqs,
                config.model_config.max_model_len,
            )
        if os.environ.get("GGUF_PARITY_ATTN_DECODE_GRAPHS") == "1" and (
            os.environ.get("GGUF_PARITY_PERSISTENT_KV") != "1"
            or os.environ.get("GGUF_PARITY_NATIVE_BATCH_FA") != "1"
        ):
            raise ValueError(
                "Attention decode graphs require persistent KV and batch FA"
            )
        return metadata

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            return output.fill_(0)
        if (
            output_scale is not None
            or output_block_scale is not None
            or self.sliding_window != (-1, -1)
            or self.alibi_slopes is not None
            or self.sinks is not None
            or self.logits_soft_cap
            or not attn_metadata.causal
        ):
            raise ValueError(
                "Native attention diagnostic expects plain causal Qwen3.5 attention"
            )
        count = attn_metadata.num_actual_tokens
        workspace = None
        if os.environ.get("GGUF_PARITY_PERSISTENT_KV") == "1":
            from .current_ggml_attention_workspace import PersistentNativeKV

            if not hasattr(self, "_gguf_native_kv_workspace"):
                self._gguf_native_kv_workspace = PersistentNativeKV(
                    *attn_metadata.native_workspace_shape,
                    query.device,
                    kv_cache.shape[0],
                )
            workspace = self._gguf_native_kv_workspace
        if os.environ.get("GGUF_PARITY_ATTN_DECODE_GRAPHS") == "1":
            from .native_attention_decode_graph import NativeDecodeGraphCache

            if (
                not hasattr(self, "_gguf_native_decode_graph_cache")
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Allocate once during profiling even though its fresh dummy
                # rows cannot be captured. Include storage in the memory estimate.
                self._gguf_native_decode_graph_cache = NativeDecodeGraphCache(workspace)
            graph_cache = getattr(self, "_gguf_native_decode_graph_cache", None)
            if graph_cache is not None and graph_cache.run(
                query[:count],
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                output[:count],
                attn_metadata.native_query_starts,
                attn_metadata.native_seq_lens,
                float(self.scale),
            ):
                return output
        ragged = (
            len(attn_metadata.native_seq_lens) != 1
            or not attn_metadata.native_seq_lens[0]
            or workspace is not None
        )
        result = paged_native_attention(
            query[:count],
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.max_seq_len,
            self.scale,
            attn_metadata.query_start_loc if ragged else None,
            attn_metadata.native_query_starts if ragged else None,
            attn_metadata.native_seq_lens if ragged else None,
            workspace,
        )
        output[:count].copy_(result)
        return output

    def qwen_forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        result = self.attn(q, k, v)
        if gate is not None:
            result = result * sigmoid(gate)
        result, _ = self.o_proj(result)
        return result

    forward._current_native_attention = True
    TritonAttentionMetadataBuilder.build = build_metadata
    TritonAttentionImpl.forward = forward
    Qwen3NextAttention.forward = qwen_forward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build())
