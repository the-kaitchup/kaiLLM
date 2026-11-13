# SPDX-License-Identifier: Apache-2.0
"""Bounded full-decode graphs around the existing native PIECEWISE model.

Native attention still uses its exact common 256-token KV bucket. GPU sequence
lengths, query starts, block tables, state slots and input tokens remain live.
Only uniform, unpadded text decode is eligible; prefill keeps the original path.
Recording bypasses inner graph wrappers without changing PIECEWISE dispatch or
arithmetic. A miss executes the model once, then records without replaying.
"""

from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
from types import MethodType
from weakref import WeakSet

_RECORDING = ContextVar("gguf_native_model_graph_recording", default=False)
_MODEL_STATE = "_gguf_native_decode_weight_state"
_MANAGER_STATE = "_gguf_native_model_decode_graphs"
_CACHES = WeakSet()


def model_decode_graph_audits(*, include_memory=False):
    """Read worker-local counters outside timed inference calls."""
    return [state.audit(include_memory=include_memory) for state in list(_CACHES)]


def decode_geometry(metadata, num_tokens):
    """Return exact (request count, native KV bucket), or refuse graph replay."""
    if (
        not isinstance(metadata, dict)
        or type(num_tokens) is not int
        or not 1 <= num_tokens <= 8
    ):
        return None
    seen, attention, gdn = set(), [], False
    for item in metadata.values():
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "native_seq_lens"):
            lengths = item.native_seq_lens
            if (
                len(lengths) != num_tokens
                or item.native_query_starts != list(range(num_tokens + 1))
                # Avoid capturing the short startup/warmup requests. Real short
                # prefixes use ordinary PIECEWISE until the ninth context token.
                or any(length <= 8 for length in lengths)
                or item.num_actual_tokens != num_tokens
            ):
                return None
            attention.append((max(lengths) + 255) // 256 * 256)
        elif hasattr(item, "num_decodes"):
            if (
                item.num_prefills
                or item.num_spec_decodes
                or item.num_decode_tokens != num_tokens
                or item.num_decodes != num_tokens
                or item.num_actual_tokens != num_tokens
                or item.non_spec_query_start_loc is None
                or item.non_spec_state_indices_tensor is None
            ):
                return None
            gdn = True
        else:
            return None
    if not attention or not gdn or len(set(attention)) != 1:
        return None
    return num_tokens, attention[0]


def tensor_bindings(value, torch, prefix=()):
    """Sign every device tensor, retaining its storage owner for graph lifetime.

    CPU values controlling launch geometry are checked separately. In
    particular, a newly allocated CPU length tensor must not create a new graph.
    """
    signatures, owners = [], []

    def collect(item, path):
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                signatures.append(
                    (
                        path,
                        item.data_ptr(),
                        item.shape,
                        item.stride(),
                        item.dtype,
                        item.device,
                    )
                )
                owners.append(item)
        elif isinstance(item, dict):
            for name, child in item.items():
                if isinstance(child, (torch.Tensor, dict, tuple, list)):
                    collect(child, (*path, name))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                if isinstance(child, (torch.Tensor, dict, tuple, list)):
                    collect(child, (*path, index))

    collect(value, prefix)
    return signatures, owners


def guard_model_weights(model):
    """Track public model weight loads without scanning all weights every token."""
    state = getattr(model, _MODEL_STATE, None)
    if state is not None:
        return state
    # Called during adapter preparation, before the initial model weight load.
    state = {"generation": 0, "ready": False}
    original = model.load_weights

    @wraps(original)
    def load_weights(self, *args, **kwargs):
        state["generation"] += 1
        state["ready"] = False
        result = original(*args, **kwargs)
        state["ready"] = True
        return result

    model.load_weights = MethodType(load_weights, model)
    setattr(model, _MODEL_STATE, state)
    return state


class NativeModelDecodeGraphs:
    def __init__(self, *, max_entries=32):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Native model graph capacity must be positive")
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.model = None
        self.weight_generation = None
        self.kv_binding = None
        self.execution_stream = None
        self.capture_stream = None
        self.counts = dict(
            hits=0,
            captures=0,
            fallbacks=0,
            evictions=0,
            model_invalidations=0,
            weight_invalidations=0,
            kv_invalidations=0,
        )
        _CACHES.add(self)

    def audit(self, *, include_memory=False):
        result = dict(
            self.counts,
            entries=len(self.entries),
            max_entries=self.max_entries,
            geometries=[list(key[1:3]) for key in self.entries],
        )
        if include_memory:
            import torch

            pools = {tuple(entry[0].pool()) for entry in self.entries.values()}
            segments = [
                item
                for item in torch.cuda.memory_snapshot(include_traces=False)
                if tuple(item.get("segment_pool_id", (0, 0))) in pools
            ]
            result["memory"] = dict(
                private_pools=len(pools),
                segments=len(segments),
                reserved_bytes=sum(item["total_size"] for item in segments),
                allocated_bytes=sum(item["allocated_size"] for item in segments),
                active_bytes=sum(item["active_size"] for item in segments),
                scope="PyTorch private pools; excludes CUDA graph executable metadata",
            )
        return result

    def invalidate(self, reason, torch):
        if self.entries:
            # An old hidden-state buffer can still be consumed by the preceding
            # step's head. Drain the worker stream before releasing graph pools.
            # This is only on a model/load/KV binding change, never a normal hit.
            torch.cuda.current_stream().synchronize()
            self.entries.clear()
            self.counts[reason + "_invalidations"] += 1

    def bindings(self, context, model, inputs, torch, geometry):
        signatures, owners = tensor_bindings(inputs, torch, ("inputs",))
        seen = set()
        for name, metadata in context.attn_metadata.items():
            if id(metadata) not in seen:
                seen.add(id(metadata))
                sigs, refs = tensor_bindings(vars(metadata), torch, ("meta", name))
                signatures.extend(sigs)
                owners.extend(refs)
        kv_signatures = []
        for name, layer in context.no_compile_layers.items():
            # Layer identity also detects replacements retaining the same cache.
            kv_signatures.append((name, id(layer)))
            sigs, refs = tensor_bindings(
                getattr(layer, "kv_cache", None), torch, ("kv", name)
            )
            kv_signatures.extend(sigs)
            owners.extend(refs)
        for name, value in (
            ("slots", context.slot_mapping),
            ("padding", context.is_padding),
        ):
            sigs, refs = tensor_bindings(value, torch, (name,))
            signatures.extend(sigs)
            owners.extend(refs)
        # KV bindings are checked independently: revisiting an old cache must
        # execute the ordinary bind/reset path, not hit a historical graph.
        return ((id(model), *geometry, tuple(signatures)), tuple(kv_signatures), owners)

    def run(self, original, manager, model, inputs):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        previous_weights = getattr(model, _MODEL_STATE, None)
        if previous_weights is not None and not previous_weights["ready"]:
            raise RuntimeError("Cannot use native graphs after a failed weight load")
        descriptor = context.batch_descriptor
        geometry = None
        if (
            descriptor is not None
            and not descriptor.has_lora
            and not descriptor.num_active_loras
            and not context.skip_compiled
            and context.ubatch_slices is None
            and context.dp_metadata is None
            and context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            and not torch.cuda.is_current_stream_capturing()
            and getattr(manager, "_graphs_captured", False)
            and not getattr(manager, "use_breakable_cg", False)
            and type(model).__name__ == "Qwen3_5ForCausalLM"
            and set(inputs).issubset(
                {"input_ids", "positions", "inputs_embeds", "intermediate_tensors"}
            )
            and isinstance(inputs.get("input_ids"), torch.Tensor)
            and isinstance(inputs.get("positions"), torch.Tensor)
            and inputs["input_ids"].is_cuda
            and inputs["positions"].is_cuda
            and inputs.get("inputs_embeds") is None
            and inputs.get("intermediate_tensors") is None
        ):
            geometry = decode_geometry(context.attn_metadata, descriptor.num_tokens)
        if geometry is None:
            self.counts["fallbacks"] += 1
            return original(manager, model, inputs)
        weight_state = getattr(model, _MODEL_STATE, None)
        if weight_state is None:
            # Never mutate a model after compilation to arm this optimization.
            self.counts["fallbacks"] += 1
            return original(manager, model, inputs)
        if not weight_state["ready"]:
            raise RuntimeError("Cannot replay native graphs after a failed weight load")
        if self.model is not model:
            self.invalidate("model", torch)
            self.model = model
            self.kv_binding = None
            self.weight_generation = None
        if self.weight_generation != weight_state["generation"]:
            self.invalidate("weight", torch)
            self.weight_generation = weight_state["generation"]
        stream = torch.cuda.current_stream()
        stream_key = (stream.device, stream.cuda_stream)
        if self.execution_stream is None:
            self.execution_stream = stream_key
        elif self.execution_stream != stream_key:
            raise RuntimeError("Native model graphs require one worker CUDA stream")
        key, binding, owners = self.bindings(context, model, inputs, torch, geometry)
        if self.kv_binding != binding:
            self.invalidate("kv", torch)
            self.kv_binding = binding
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            entry[0].replay()
            self.counts["hits"] += 1
            return entry[1]
        # This is the only execution on a miss. Recording never advances the
        # recurrent state and the captured graph is first replayed next token.
        warm_output = original(manager, model, inputs)
        graph = torch.cuda.CUDAGraph()
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=stream.device)
        self.capture_stream.wait_stream(stream)
        token = _RECORDING.set(True)
        try:
            with torch.cuda.graph(graph, stream=self.capture_stream):
                captured_output = original(manager, model, inputs)
        finally:
            _RECORDING.reset(token)
        stream.wait_stream(self.capture_stream)
        if not isinstance(captured_output, torch.Tensor):
            raise TypeError("Native model graphs require one hidden-state tensor")
        if len(self.entries) >= self.max_entries:
            # torch.cuda.graph entered with a device barrier, so prior entries
            # have no outstanding consumers when this old pool is released.
            self.entries.popitem(last=False)
            self.counts["evictions"] += 1
        self.entries[key] = (graph, captured_output, owners, model)
        self.counts["captures"] += 1
        return warm_output


def install_native_model_decode_graph():
    """Install hooks; only managers with the existing native-GDN marker opt in."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    from .native_gdn_capture import _enabled

    original = CudaGraphManager.run_pw_graph
    if getattr(original, "_gguf_native_model_decode_graph", False):
        return
    original_wrapper = CUDAGraphWrapper.__call__
    original_prepare = Qwen35GGUFAdapter.prepare_model

    @wraps(original_prepare)
    def prepare_model(adapter, model):
        result = original_prepare(adapter, model)
        if _enabled(model.vllm_config):
            guard_model_weights(model)
        return result

    @wraps(original_wrapper)
    def compiled_call(self, *args, **kwargs):
        if _RECORDING.get() and self.runtime_mode == CUDAGraphMode.PIECEWISE:
            return self.runnable(*args, **kwargs)
        return original_wrapper(self, *args, **kwargs)

    @wraps(original)
    def run(self, model, model_inputs):
        if not _enabled(self.vllm_config):
            return original(self, model, model_inputs)
        state = getattr(self, _MANAGER_STATE, None)
        if state is None:
            state = NativeModelDecodeGraphs()
            setattr(self, _MANAGER_STATE, state)
        return state.run(original, self, model, model_inputs)

    run._gguf_native_model_decode_graph = True
    CUDAGraphWrapper.__call__ = compiled_call
    CudaGraphManager.run_pw_graph = run
    Qwen35GGUFAdapter.prepare_model = prepare_model
