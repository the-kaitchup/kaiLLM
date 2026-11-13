# SPDX-License-Identifier: Apache-2.0
"""vLLM loading and merged-projection support for mixed YAQA/GGUF matrices."""

import os

import gguf
import numpy as np
import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from ..yaqa_gguf import load_yaqa_components, read_yaqa_profile
from .yaqa import PackedYAQALinear
from .yaqa_prefill_cache_out import apply as apply_merged_prefill
from .yaqa_prefill_cache_out import selected as select_merged_prefill


def configure_yaqa_loading(quant_config, load_spec):
    """Bind packed sources before vLLM constructs its projection modules."""
    mapping = load_spec.gguf_to_hf_name_map or {}
    sources = {}
    profiles = {}
    has_yaqa_matrices = False
    readers = {}
    tensor_maps = {}

    def add_source(name, source):
        hf_name = mapping[name]
        if hf_name in sources:
            raise ValueError(f"Duplicate GGUF weight source for {hf_name}: {name}")
        sources[hf_name] = source

    for path in load_spec.weights_source:
        reader = gguf.GGUFReader(path)
        readers[path] = reader
        tensor_maps[path] = {tensor.name: tensor for tensor in reader.tensors}
        profile = read_yaqa_profile(reader)
        if profile is not None:
            profiles[path] = profile
        descriptors = {} if profile is None else profile["tensors"]
        has_yaqa_matrices |= bool(descriptors)
        for tensor in reader.tensors:
            if tensor.name in mapping and len(tensor.shape) == 2:
                add_source(tensor.name, (path, tensor.name, None))
        for name, descriptor in descriptors.items():
            if name not in mapping:
                raise ValueError(f"YAQA logical matrix has no model mapping: {name}")
            add_source(name, (path, name, descriptor))
    from .yaqa_prefill_cache import create_plan

    # The entire selected profile must fit before any physical cache allocation.
    quant_config.yaqa_prefill_cache_plan = create_plan(sources)
    from .yaqa_index40 import create_plan as create_index40_plan

    quant_config.yaqa_index40_plan = create_index40_plan(sources)
    quant_config.yaqa_sources = sources
    quant_config.yaqa_readers = readers
    quant_config.yaqa_tensor_maps = tensor_maps
    quant_config.yaqa_profiles = profiles
    quant_config.has_yaqa_profile = bool(profiles)
    quant_config.yaqa_has_quantized_matrices = has_yaqa_matrices
    quant_config.compact_gguf = (
        bool(profiles) or os.getenv("VLLM_GGUF_BACKEND") == "compact"
    )
    # Compact GGUF uses canonical reconstruction independently of YAQA metadata.
    # Existing per-format, shape, GPU and memory admission still selects kernels.
    quant_config.canonical_gguf_rounding = quant_config.compact_gguf
    quant_config.yaqa_consumed_names = set()
    quant_config.yaqa_consumed_tensors = set()
    quant_config.gguf_row_permutations = dict(getattr(load_spec, "gguf_row_permutations", None) or {})
    quant_config.gguf_column_permutations = dict(
        getattr(load_spec, "gguf_column_permutations", None) or {}
    )


def get_yaqa_parts(quant_config, prefix):
    """Resolve individual and fused projections without changing shard order."""
    sources = getattr(quant_config, "yaqa_sources", {})
    direct = prefix + ".weight"
    compact = getattr(quant_config, "compact_gguf", False)
    if direct in sources and (compact or sources[direct][2] is not None):
        return [(direct, sources[direct])]
    parent, _, leaf = prefix.rpartition(".")
    names = getattr(quant_config, "packed_modules_mapping", {}).get(leaf, [])
    parts = [((parent + "." if parent else "") + name + ".weight") for name in names]
    if not parts or not any(
        name in sources and (compact or sources[name][2] is not None) for name in parts
    ):
        return None
    missing = [name for name in parts if name not in sources]
    if missing:
        raise ValueError(f"Missing shards of mixed YAQA projection {prefix}: {missing}")
    return [(name, sources[name]) for name in parts]


def _ordinary_tensor(tensor):
    value = np.array(tensor.data, copy=True)
    if int(tensor.tensor_type) == 30:
        return torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(value)


def _permute_stored(quant_config, tensor_name, data, qtype):
    """Apply adapter row/column permutations to stored GGUF rows.

    Returns (data, gather). Column permutations that move whole GGML blocks are
    applied to the stored bytes; otherwise the stored columns are kept and
    gather (x_stored = x.index_select(-1, gather)) permutes the activation.
    """
    rows = getattr(quant_config, "gguf_row_permutations", {}).get(tensor_name)
    if rows is not None:
        data = data[rows]
    columns = getattr(quant_config, "gguf_column_permutations", {}).get(tensor_name)
    if columns is None:
        return data, None
    if data.dtype != torch.uint8:
        return data[:, columns].contiguous(), None
    block, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
    if columns.numel() % block == 0:
        blocks = columns.reshape(-1, block)
        starts = blocks[:, :1]
        if bool((starts % block == 0).all()) and torch.equal(
            blocks - starts, torch.arange(block, dtype=columns.dtype, device=columns.device).expand_as(blocks)
        ):
            order = (starts[:, 0] // block).to(torch.long)
            stored = data.reshape(data.shape[0], -1, type_size)
            return stored[:, order].reshape(data.shape[0], -1).contiguous(), None
    gather = torch.empty_like(columns)
    gather[columns] = torch.arange(columns.numel(), dtype=columns.dtype, device=columns.device)
    return data, gather


def _transpose_view(gather):
    """(A, B, S) when gather == arange(N).view(A, B, S).transpose(0, 1).flatten(), else None."""
    n = gather.numel()
    values = gather.tolist()
    if not values or values[0] != 0:
        return None
    s = 1
    while s < n and values[s] == s:
        s += 1
    if s == n or n % s or values[s] % s:
        return None
    b = values[s] // s
    if b <= 1 or n % (b * s):
        return None
    a = n // (b * s)
    expected = torch.arange(n, dtype=gather.dtype, device=gather.device).view(a, b, s).transpose(0, 1)
    return (a, b, s) if torch.equal(gather, expected.reshape(-1)) else None


class YAQAMixedLinearMethod(LinearMethodBase):
    def __init__(self, quant_config, parts):
        self.quant_config = quant_config
        self.parts = parts
        if not hasattr(quant_config, "yaqa_marlin_physical_plan"):
            from .yaqa_marlin_physical_prepare import create_plan

            quant_config.yaqa_marlin_physical_plan = create_plan(quant_config)
        self.marlin_plan = quant_config.yaqa_marlin_physical_plan

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size, extra_weight_attrs
        if getattr(layer, "tp_size", 1) != 1:
            raise ValueError("Packed YAQA currently requires tensor_parallel_size=1")
        from .compact import PackedGGUFLinear

        device = torch.empty(0).device
        modules = []
        expected_rows = (
            [sum(output_partition_sizes)]
            if len(self.parts) == 1
            else output_partition_sizes
        )
        if len(expected_rows) != len(self.parts) and all(
            descriptor is None for _, (_, _, descriptor) in self.parts
        ):
            # One GGUF tensor may hold several consecutive vLLM output partitions
            # (e.g. Qwen3.5 attn_qkv = q|k|v of in_proj_qkvz = q|k|v|z): group them.
            grouped, index = [], 0
            for _, (path, tensor_name, _) in self.parts:
                tensor = self.quant_config.yaqa_tensor_maps[path][tensor_name]
                rows = int(tuple(reversed(tensor.shape))[0])
                total = 0
                while index < len(output_partition_sizes) and total < rows:
                    total += output_partition_sizes[index]
                    index += 1
                grouped.append(total)
            if index == len(output_partition_sizes):
                expected_rows = grouped
        if len(expected_rows) != len(self.parts):
            raise ValueError("Mixed YAQA shards differ from vLLM output partitions")
        gathers = []
        for (name, (path, tensor_name, descriptor)), expected in zip(
            self.parts, expected_rows
        ):
            reader = self.quant_config.yaqa_readers[path]
            if descriptor is not None and (
                tensor_name in getattr(self.quant_config, "gguf_row_permutations", {})
                or tensor_name in getattr(self.quant_config, "gguf_column_permutations", {})
            ):
                raise ValueError(f"Permuted YAQA matrices are not supported: {tensor_name}")
            if descriptor is not None:
                module = PackedYAQALinear(
                    descriptor,
                    load_yaqa_components(reader, descriptor),
                    params_dtype,
                    device=device,
                    prefill_cache_plan=getattr(
                        self.quant_config, "yaqa_prefill_cache_plan", None
                    ),
                    prefill_cache_key=name,
                    index40_plan=getattr(self.quant_config, "yaqa_index40_plan", None),
                    index40_cache_key=name,
                )
                from .yaqa_power_input_resource import prepare_body_layer

                prepare_body_layer(module)
                from .yaqa_hin_high_low import prepare_body_layer as prepare_hin_hilo

                prepare_hin_hilo(module)
                from .yaqa_raw_factor_post import prepare_body_layer as prepare_raw_post

                prepare_raw_post(module)
                rows, cols = module.shape
            else:
                tensor = self.quant_config.yaqa_tensor_maps[path][tensor_name]
                rows, cols = map(int, reversed(tensor.shape))
                stored, gather = _permute_stored(
                    self.quant_config, tensor_name, _ordinary_tensor(tensor), int(tensor.tensor_type)
                )
                if gather is not None:
                    gathers.append(gather)
                module = PackedGGUFLinear(
                    stored.to(device),
                    int(tensor.tensor_type),
                    params_dtype,
                    role="body",
                    canonical_rounding=getattr(
                        self.quant_config, "canonical_gguf_rounding", False
                    ),
                )
            if cols != input_size_per_partition:
                raise ValueError(f"Mixed YAQA input shape mismatch for {name}")
            if rows != expected:
                raise ValueError(f"Mixed YAQA output shape mismatch for {name}")
            modules.append(module)
            self.quant_config.yaqa_consumed_names.add(name)
            self.quant_config.yaqa_consumed_tensors.add(tensor_name)
        if sum(module.shape[0] for module in modules) != sum(output_partition_sizes):
            raise ValueError("Mixed YAQA output shape differs from vLLM partitions")
        layer.yaqa_parts = torch.nn.ModuleList(modules)
        if gathers:
            if len(gathers) != len(modules) or any(not torch.equal(g, gathers[0]) for g in gathers):
                raise ValueError("Merged GGUF projections need one shared input permutation")
            # Stored columns are not block-aligned with the permutation: gather inputs.
            layer._kai_input_gather = gathers[0].to(device)
            layer._kai_input_view = _transpose_view(gathers[0])
        plan = getattr(self.quant_config, "yaqa_prefill_cache_plan", None)
        if plan is not None and plan.qkv_merge:
            from .yaqa_qkv_cache import merge_cached_qkv

            if merge_cached_qkv(layer):
                plan.qkv_joined.append(self.parts[0][0])
                from vllm.logger import init_logger

                init_logger("vllm.gguf.yaqa_linear").info(
                    "Physical QKV cache join enabled for %s; active payload unchanged",
                    self.parts[0][0],
                )

        from .yaqa_marlin_physical_prepare import prepare as prepare_marlin

        layer._yaqa_marlin_physical = prepare_marlin(layer, self.marlin_plan)

    def process_weights_after_loading(self, layer):
        plan = getattr(self.quant_config, "yaqa_prefill_cache_plan", None)
        if plan is not None:
            # All create_weights constructors have run at this existing model
            # lifecycle boundary. Never begin inference with a partial cache.
            plan.ensure_complete()
        indexed = getattr(self.quant_config, "yaqa_index40_plan", None)
        if indexed is not None:
            indexed.ensure_complete()
        from .yaqa_grouped import prepare

        if not hasattr(layer, "_yaqa_grouped_state"):
            layer._yaqa_grouped_state = prepare(layer)
        from .yaqa_grouped_hin import prepare as prepare_grouped_hin

        if not hasattr(layer, "_yaqa_grouped_hin"):
            layer._yaqa_grouped_hin = prepare_grouped_hin(layer)

        if self.marlin_plan is not None:
            self.marlin_plan.ensure_complete()
        from .yaqa_marlin_physical import bind as bind_marlin

        bind_marlin(layer, self._apply_original)

    def _fused(self, layer):
        fused = getattr(layer, "_kai_fused", None)
        if fused is None:
            from .compact import PackedGGUFLinear
            from .native_fused import prepare as prepare_fused

            parts = list(layer.yaqa_parts)
            fused = (
                prepare_fused(parts)
                if all(type(part) is PackedGGUFLinear for part in parts)
                else None
            ) or False
            layer._kai_fused = fused
            producer = layer.__dict__.get("_kai_act_producer")
            if fused and producer is not None:
                from .native_fused import CONSUMER_OF

                CONSUMER_OF[fused.weights[0].data_ptr()] = producer
        return fused

    def _ext_rows(self, layer, x, width):
        from .native_fused import MAX_ROWS, ext_available

        if getattr(layer, "_yaqa_marlin_physical", None) is not None or not x.is_cuda:
            return None
        fused = self._fused(layer)
        rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
        if not fused or x.shape[-1] != width(fused) or not 0 < rows <= MAX_ROWS or rows not in fused._supported:
            return None
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not ext_available():
            return None
        return fused

    def apply_swiglu(self, layer, gate_up, act_fn):
        """down_proj(silu(gate) * up): the activation is prepared straight from gate_up for <= 8 rows."""
        if getattr(layer, "_kai_input_gather", None) is None and os.getenv("KAILLM_FUSED_SWIGLU", "1") == "1":
            fused = self._ext_rows(layer, gate_up, lambda f: 2 * f.k)
            if fused is not None:
                return fused.call_swiglu(gate_up)
        return self.apply(layer, act_fn(gate_up))

    def apply(self, layer, x, bias=None):
        gather = getattr(layer, "_kai_input_gather", None)
        if gather is not None and bias is None and getattr(layer, "_kai_input_view", None) is not None:
            fused = self._ext_rows(layer, x, lambda f: f.k)
            if fused is not None:
                return fused.call_view(x, layer._kai_input_view)
        if gather is not None:
            view = layer._kai_input_view
            if view is not None:
                # One strided copy: x[..., (a, b, s)] -> x[..., (b, a, s)].
                x = x.unflatten(-1, view).transpose(-3, -2).flatten(-3)
            else:
                x = x.index_select(-1, gather)
        state = getattr(layer, "_yaqa_marlin_physical", None)
        if state is not None:
            from .yaqa_marlin_physical import apply_prepared

            return apply_prepared(x, state, bias)
        return self._apply_original(layer, x, bias)

    def _apply_original(self, layer, x, bias=None):
        grouped = getattr(layer, "_yaqa_grouped_state", None)
        if grouped is not None:
            from .yaqa_grouped import opaque_grouped

            output = opaque_grouped(x, grouped.buffers(), grouped.key, grouped.total)
            return output if bias is None else output + bias
        hin = getattr(layer, "_yaqa_grouped_hin", None)
        if hin is not None and x.device.type == "cuda":
            from .yaqa_grouped_hin import opaque_grouped_hin

            output = opaque_grouped_hin(x, hin.buffers(), hin.key, hin.total)
            return output if bias is None else output + bias
        joined = getattr(layer, "_yaqa_qkv_cache", None)
        if (
            joined is not None
            and x.device.type == "cuda"
            and x.dtype in (torch.float32, torch.bfloat16)
            and x.numel() // x.shape[-1] in (1, 4)
        ):
            from .yaqa_qkv_cache import fused_cached_qkv

            output = fused_cached_qkv(x, joined)
            return output if bias is None else output + bias
        fused = self._fused(layer)
        if fused and fused.supports(x):
            output = fused(x)
            return output if bias is None else output + bias
        if select_merged_prefill(x, layer.yaqa_parts):
            output = apply_merged_prefill(x, layer.yaqa_parts)
        else:
            outputs = [part(x) for part in layer.yaqa_parts]
            if len(outputs) == 1:
                output = outputs[0]
            else:
                output = torch.ops.vllm._gguf_concat_projections(outputs)
        return output if bias is None else output + bias
