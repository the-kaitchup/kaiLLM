"""Bounded, opt-in graphs around the eager native attention breakpoint.

Only uniform decode with existing prefixes is eligible. Captures preserve the
actual request count and native common KV bucket. All graph entries in a layer
share buffers and execute serially on one worker stream.
"""

from collections import OrderedDict

import torch
import triton

from .current_ggml_attention import (
    _batched_causal_mask,
    _pack_queries,
    _scatter_queries,
)
from .current_ggml_attention_batch import native_batch_attention_out


def _signature(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype)


def _can_elide_query_copies(q, output, protected=()):
    """Whether uniform single-query packing/scattering are safe identity views.

    The caller has already established authoritative CPU starts [0, ..., B].
    Compare whole storage ranges conservatively: even disjoint slices of the
    same allocation retain the old copy path. No device reads or synchronization
    are needed, and the check runs only when recording a new graph.
    """
    if (
        q.ndim != 3
        or not 1 <= q.shape[0] <= 8
        or q.shape[1:] != (8, 256)
        or output.shape != q.shape
        or q.dtype != torch.float32
        or output.dtype != torch.float32
        or q.device != output.device
        or q.device.type == "meta"
        or not q.is_contiguous()
        or not output.is_contiguous()
        or q.data_ptr() % 16
        or output.data_ptr() % 16
    ):
        return False

    def storage_range(tensor):
        storage = tensor.untyped_storage()
        begin = storage.data_ptr()
        return begin, begin + storage.nbytes()

    def overlaps(left, right):
        return left[0] < right[1] and right[0] < left[1]

    q_range, output_range = storage_range(q), storage_range(output)
    if overlaps(q_range, output_range):
        return False
    for tensor in protected:
        if tensor.device == q.device:
            extent = storage_range(tensor)
            if overlaps(q_range, extent) or overlaps(output_range, extent):
                return False
    return True


class NativeDecodeGraphCache:
    def __init__(self, workspace, *, max_entries=32, elide_identity_copies=True):
        if max_entries < 1:
            raise ValueError("Native attention graph cache must have positive capacity")
        self.workspace = workspace
        self.max_entries = max_entries
        self.elide_identity_copies = elide_identity_copies
        self.entries = OrderedDict()
        device = workspace.keys.device
        self.q_buffer = torch.empty(
            workspace.requests * 2048, device=device, dtype=torch.float32
        )
        self.result_buffer = torch.empty_like(self.q_buffer)
        self.mask_buffer = torch.empty(
            workspace.requests * 64 * workspace.context,
            device=device,
            dtype=torch.float16,
        )
        self.scratch = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.uint8)
        self.capture_stream = torch.cuda.Stream(device=device)
        self.execution_stream = None
        self.mask_layout = None
        self.rebind_epoch = workspace.rebind_count
        self.counts = dict(
            hits=0,
            misses=0,
            captures=0,
            evictions=0,
            rebind_invalidations=0,
            outer_capture_fallbacks=0,
            ineligible_fallbacks=0,
            mask_layout_resets=0,
            identity_copy_elision_captures=0,
            copied_query_captures=0,
        )

    def audit(self):
        return dict(
            self.counts,
            entries=len(self.entries),
            max_entries=self.max_entries,
            elide_identity_copies=self.elide_identity_copies,
            storage_bytes=sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    self.q_buffer,
                    self.result_buffer,
                    self.mask_buffer,
                    self.scratch,
                )
            ),
        )

    def _run(
        self,
        q,
        cache,
        blocks,
        starts,
        lengths,
        output,
        requests,
        padded,
        scale,
        identity_views,
    ):
        keys, values = self.workspace.update(cache, blocks, starts, lengths, padded)
        if identity_views:
            packed = q.view(requests, 1, 8, 256)
            result = output.view_as(packed)
        else:
            packed = self.q_buffer[: requests * 2048].view(requests, 1, 8, 256)
            result = self.result_buffer[: requests * 2048].view_as(packed)
        # Flat storage allows each bucket's mask to be contiguous without
        # allocating a separate full-context mask for every captured key.
        mask = self.mask_buffer[: requests * 64 * padded].view(requests, 64, padded)
        if not identity_views:
            _pack_queries[(8, requests)](q, starts, lengths, packed, requests, 1, 256)
        _batched_causal_mask[(triton.cdiv(padded, 256), requests)](
            mask, starts, lengths, 1, padded, 256, 64
        )
        native_batch_attention_out(
            packed, keys, values, mask, result, self.scratch, scale
        )
        if not identity_views:
            _scatter_queries[(8, requests)](
                result, starts, lengths, output, requests, 1, 256
            )

    def run(
        self,
        q,
        cache,
        blocks,
        starts,
        lengths,
        output,
        query_starts_cpu,
        seq_lens_cpu,
        scale,
    ):
        """Write the caller's output and return True, or request eager fallback.

        No output aliases are returned. Existing vLLM downstream graphs keep
        their original output pointer, including any untouched padding rows.
        """
        if torch.cuda.is_current_stream_capturing():
            self.counts["outer_capture_fallbacks"] += 1
            return False
        # Must run on the host before graph lookup: replay bypasses update's
        # Python binding logic and cannot discover a replacement profiling KV.
        self.workspace.bind_cache(cache)
        if self.workspace.rebind_count != self.rebind_epoch:
            self.entries.clear()
            self.rebind_epoch = self.workspace.rebind_count
            self.counts["rebind_invalidations"] += 1
        requests = len(seq_lens_cpu)
        if (
            not 1 <= requests <= self.workspace.requests
            or query_starts_cpu != list(range(requests + 1))
            or any(length <= 1 for length in seq_lens_cpu)
            or q.shape != (requests, 8, 256)
            or output.shape != q.shape
            or q.dtype != torch.float32
            or output.dtype != torch.float32
            or not q.is_contiguous()
            or not output.is_contiguous()
        ):
            self.counts["ineligible_fallbacks"] += 1
            return False
        padded = triton.cdiv(max(seq_lens_cpu), 256) * 256
        if padded > self.workspace.context:
            raise ValueError("Attention graph KV bucket exceeds workspace capacity")
        if (
            cache.ndim != 4
            or cache.shape[1] != 2
            or cache.shape[3] != 512
            or cache.dtype not in (torch.float16, torch.float32)
            or blocks.ndim != 2
            or blocks.shape[0] != requests
            or blocks.dtype != torch.int32
            or blocks.stride(1) != 1
            or starts.shape != (requests + 1,)
            or lengths.shape != (requests,)
            or any(
                tensor.dtype != torch.int32 or not tensor.is_contiguous()
                for tensor in (starts, lengths)
            )
            or any(
                tensor.device != self.workspace.keys.device
                for tensor in (q, cache, blocks, starts, lengths, output)
            )
        ):
            raise ValueError("Invalid native attention graph input metadata")
        stream = torch.cuda.current_stream(q.device)
        if self.execution_stream is None:
            self.execution_stream = stream.cuda_stream
        elif stream.cuda_stream != self.execution_stream:
            raise RuntimeError("Native attention graphs require one execution stream")
        if self.mask_layout != padded:
            # Changing the flat view's KV width moves every request's row0.
            # Clear old visible entries before any old/new graph uses that
            # layout. Within a bucket only row0 changes; all guard rows stay
            # -inf. This also applies when revisiting a cached older bucket.
            self.mask_buffer.fill_(float("-inf"))
            self.mask_layout = padded
            self.counts["mask_layout_resets"] += 1
        key = (
            requests,
            padded,
            float(scale),
            self.rebind_epoch,
            *(_signature(t) for t in (q, cache, blocks, starts, lengths, output)),
        )
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            entry[0].replay()
            self.counts["hits"] += 1
            return True
        self.counts["misses"] += 1
        identity_views = self.elide_identity_copies and _can_elide_query_copies(
            q,
            output,
            (
                cache,
                blocks,
                starts,
                lengths,
                *(
                    t
                    for t in vars(self.workspace).values()
                    if isinstance(t, torch.Tensor)
                ),
                self.q_buffer,
                self.result_buffer,
                self.mask_buffer,
                self.scratch,
            ),
        )
        # This real call also warms all JIT kernels and native launch attributes.
        # Capture records the same operations without executing them, so this
        # step's workspace and accounting update occur exactly once.
        self._run(
            q,
            cache,
            blocks,
            starts,
            lengths,
            output,
            requests,
            padded,
            scale,
            identity_views,
        )
        graph = torch.cuda.CUDAGraph()
        self.capture_stream.wait_stream(stream)
        with torch.cuda.graph(graph, stream=self.capture_stream):
            self._run(
                q,
                cache,
                blocks,
                starts,
                lengths,
                output,
                requests,
                padded,
                scale,
                identity_views,
            )
        stream.wait_stream(self.capture_stream)
        if len(self.entries) == self.max_entries:
            self.entries.popitem(last=False)
            self.counts["evictions"] += 1
        # Keep small external buffers alive. The authoritative cache remains
        # engine-owned: its signature/epoch is checked before every replay.
        self.entries[key] = (graph, (q, blocks, starts, lengths, output))
        self.counts["captures"] += 1
        self.counts[
            "identity_copy_elision_captures"
            if identity_views
            else "copied_query_captures"
        ] += 1
        return True
