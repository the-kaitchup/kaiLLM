# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)

# Use vLLM's configured logger hierarchy so the selected execution backend is
# visible in normal server logs, including spawned engine processes.
logger = init_logger("vllm.gguf.linear")


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


def _gguf_concat_projections(parts: list[torch.Tensor]) -> torch.Tensor:
    """Keep merged projection layout stable across compiled graph boundaries.

    Inductor can eliminate a cat followed by split and return a contiguous
    individual projection where a later graph expects a strided merged view.
    Qwen3.5 passes that view across its GDN attention custom-op boundary. Keep
    this small concatenation opaque so real and fake output strides agree.
    """
    return torch.cat(parts, dim=-1)


def _gguf_concat_projections_fake(parts: list[torch.Tensor]) -> torch.Tensor:
    return parts[0].new_empty(
        (*parts[0].shape[:-1], sum(part.shape[-1] for part in parts))
    )


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
    direct_register_custom_op(
        op_name="_gguf_concat_projections",
        op_func=_gguf_concat_projections,
        fake_impl=_gguf_concat_projections_fake,
    )
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    operator_role = "body"

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "tp_rank": layer.tp_rank,
                "tp_size": layer.tp_size,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "tp_rank": layer.tp_rank,
                "tp_size": layer.tp_size,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        self._create_padded_weight_param(layer)
        if getattr(self.quant_config, "compact_gguf", False):
            self._prepare_compact_weights(layer)
            return
        self._prepare_humming_weights(layer)
        accelerated = hasattr(layer, "gguf_humming") or hasattr(
            layer, "gguf_packed_embedding"
        )
        if not accelerated:
            shard_types = [
                layer.qweight_type.shard_weight_type.get(idx, qweight_type)
                for idx in layer.qweight.shard_id
            ] or [qweight_type]
            for shard_type in shard_types:
                if shard_type not in (UNQUANTIZED_TYPES | DEQUANT_TYPES):
                    name = WeightType(shard_type).name
                    raise ValueError(
                        f"GGUF type {name} requires the Humming CUDA backend "
                        f"for layer {type(layer).__name__}."
                    )

    def _prepare_compact_weights(self, layer: torch.nn.Module) -> None:
        from .compact import PackedGGUFLinear

        weight = layer.qweight
        fallback = layer.qweight_type.weight_type
        if weight.shard_id and hasattr(weight, "shard_offset_map"):
            parts = []
            for idx in weight.shard_id:
                start, end, width = weight.shard_offset_map[idx]
                data = weight[start:end, :width].contiguous()
                wtype = layer.qweight_type.shard_weight_type.get(idx, fallback)
                parts.append(
                    PackedGGUFLinear(
                        data,
                        wtype,
                        self.params_dtype,
                        role=self.operator_role,
                        canonical_rounding=getattr(
                            self.quant_config, "canonical_gguf_rounding", False
                        ),
                    )
                )
        else:
            parts = [
                PackedGGUFLinear(
                    weight,
                    fallback,
                    self.params_dtype,
                    role=self.operator_role,
                    canonical_rounding=getattr(
                        self.quant_config, "canonical_gguf_rounding", False
                    ),
                )
            ]
        layer.gguf_compact = torch.nn.ModuleList(parts)
        del layer.qweight
        logger.info_once("GGUF weights use compact wire-format CUDA kernels.")

    def _prepare_humming_weights(self, layer: torch.nn.Module) -> None:
        if not current_platform.is_cuda() or self.params_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return
        from .humming import (
            SUPPORTED_QUANT_TYPES,
            humming_available,
            prepare_gguf_humming,
        )

        if not humming_available() or not current_platform.has_device_capability(75):
            return

        qweight = layer.qweight
        fallback_type = layer.qweight_type.weight_type
        shard_ids = qweight.shard_id
        shard_types = [
            layer.qweight_type.shard_weight_type.get(idx, fallback_type)
            for idx in shard_ids
        ] or [fallback_type]
        if not all(wtype in SUPPORTED_QUANT_TYPES for wtype in shard_types):
            return

        if len(set(shard_types)) == 1:
            parts = [(qweight, shard_types[0])]
        else:
            parts = []
            for idx, wtype in zip(shard_ids, shard_types):
                start, end, width = qweight.shard_offset_map[idx]
                part = qweight[start:end, :width].contiguous()
                if parts and parts[-1][1] == wtype:
                    parts[-1] = (torch.cat((parts[-1][0], part)), wtype)
                else:
                    parts.append((part, wtype))
        layer.gguf_humming = torch.nn.ModuleList(
            prepare_gguf_humming(data, wtype, self.params_dtype)
            for data, wtype in parts
        )
        logger.info_once(
            "GGUF linear weights use Humming WNA16 kernels with FP32 accumulation."
        )
        if not getattr(layer, "_gguf_keep_embedding_weight", False):
            del layer.qweight

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            padded_param.tp_rank = qweight.tp_rank
            padded_param.tp_size = qweight.tp_size
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(
                    0, dtype=qweight.dtype, device=qweight.device
                )
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(layer, "gguf_compact"):
            fused = getattr(layer, "_kai_fused", None)
            if fused is None:
                from .native_fused import prepare as prepare_fused

                fused = prepare_fused(list(layer.gguf_compact)) or False
                layer._kai_fused = fused
            if fused and fused.supports(x):
                out = fused(x)
                return out if bias is None else out + bias
            outputs = [part(x) for part in layer.gguf_compact]
            out = (
                outputs[0]
                if len(outputs) == 1
                else torch.ops.vllm._gguf_concat_projections(outputs)
            )
            return out if bias is None else out + bias
        if hasattr(layer, "gguf_humming"):
            outputs = [part(x) for part in layer.gguf_humming]
            out = (
                outputs[0]
                if len(outputs) == 1
                else torch.ops.vllm._gguf_concat_projections(outputs)
            )
            return out if bias is None else out + bias

        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        shard_id = layer.qweight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, qweight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    fused_mul_mat_gguf_op(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf_op(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out
