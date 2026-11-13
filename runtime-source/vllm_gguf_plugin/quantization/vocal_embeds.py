# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .linear import GGUFLinearMethod
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_embedding_weight_loader,
    _gguf_embedding_weight_type_loader,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
)
from .utils import DEQUANT_TYPES, UNQUANTIZED_TYPES


def _apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    if qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        if qweight_type in (WeightType.Q4_K, WeightType.Q5_K, WeightType.Q6_K):
            from ..triton.dequantize.interface import ggml_dequantize_triton

            # Native GGML reconstructs these rows in F32. The legacy decoder
            # rounds intermediate scales/products to F16, even for F32 output.
            dequant = ggml_dequantize_triton(
                quant, qweight_type, hidden_size, x_flat.shape[0], dtype
            )
        else:
            dequant = ops.ggml_dequantize(
                quant, qweight_type, hidden_size, x_flat.shape[0], dtype
            )
        return dequant.view(*x.shape, hidden_size)
    qweight_type = WeightType(qweight_type)
    raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    del qweight, qweight_type
    return torch.empty(*x.shape, hidden_size, dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding
except AttributeError as error:
    raise error


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF."""

    operator_role = "vocabulary"

    def __init__(self, quant_config):
        super().__init__(quant_config)
        from .native_head import native_head_mode

        self.native_head_mode = native_head_mode()

    def _prepare_compact_weights(self, layer: torch.nn.Module) -> None:
        super()._prepare_compact_weights(layer)
        is_output = isinstance(layer, ParallelLMHead) or getattr(
            layer, "_gguf_tied_lm_head", False
        )
        if is_output and self.native_head_mode == "auto-native":
            from .auto_head import prepare_output_head

            layer.gguf_auto_head_routes = [
                prepare_output_head(part) for part in layer.gguf_compact
            ]
            return
        if is_output and getattr(self.quant_config, "canonical_gguf_rounding", False):
            for part in layer.gguf_compact:
                prepare_q3 = getattr(part, "prepare_native_bf16_q3_head", None)
                if prepare_q3 is not None:
                    prepare_q3()
        if is_output:
            from .native_bf16 import mode
            from .native_bf16_sm120 import mode as sm120_mode

            for part in layer.gguf_compact:
                capability = (
                    torch.cuda.get_device_capability(part.weight.device)
                    if part.weight.is_cuda
                    else None
                )
                if capability == (8, 6) and mode() == "body-q6":
                    part.prepare_native_bf16_q6_head()
                elif capability == (12, 0) and sm120_mode() == "body-q6":
                    part.prepare_native_sm120_q6_head()
        if self.native_head_mode in ("native", "native-mma") and is_output:
            if not getattr(self.quant_config, "canonical_gguf_rounding", False):
                raise ValueError("Native head mode requires a canonical YAQA/RCO model")
            for part in layer.gguf_compact:
                if part._native_sm120_q6_head is None:
                    part.prepare_native_head(mode=self.native_head_mode)
        elif self.native_head_mode == "native-q6" and is_output:
            if not getattr(self.quant_config, "canonical_gguf_rounding", False):
                raise ValueError("Native Q6 head mode requires a canonical RCO model")
            for part in layer.gguf_compact:
                part.prepare_native_q6_head()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(layer, "gguf_compact") and getattr(
            self.quant_config, "canonical_gguf_rounding", False
        ):
            # GGML's output projection exposes the FP32 accumulator directly
            # to its scorer. The same vocabulary tensor still embeds to BF16.
            outputs = [
                part(x, output_dtype=torch.float32) for part in layer.gguf_compact
            ]
            output = (
                outputs[0]
                if len(outputs) == 1
                else torch.ops.vllm._gguf_concat_projections(outputs)
            )
            return output if bias is None else output + bias
        return super().apply(layer, x, bias)

    def _prepare_humming_weights(self, layer: torch.nn.Module) -> None:
        extra_format = (
            layer.qweight_type.weight_type == WeightType.Q8_1
            or layer.qweight_type.weight_type not in DEQUANT_TYPES | UNQUANTIZED_TYPES
        )
        if extra_format and not isinstance(layer, ParallelLMHead):
            from .packed_embedding import prepare_packed_embedding

            layer.gguf_packed_embedding = prepare_packed_embedding(
                layer.qweight, layer.qweight_type.weight_type, self.params_dtype
            )
        is_output = isinstance(layer, ParallelLMHead) or getattr(
            layer, "_gguf_tied_lm_head", False
        )
        if is_output:
            layer._gguf_keep_embedding_weight = (
                not isinstance(layer, ParallelLMHead) and not extra_format
            )
            super()._prepare_humming_weights(layer)
            if extra_format and not hasattr(layer, "gguf_humming"):
                raise ValueError("This GGUF output projection requires Humming on CUDA")
        elif extra_format:
            del layer.qweight

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
        extra_weight_attrs.pop("weight_loader", None)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": partial(_gguf_embedding_weight_loader, layer),
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

        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": _gguf_embedding_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "tp_rank": layer.tp_rank,
                "tp_size": layer.tp_size,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(
            layer,
            "qweight",
            fallback_weight_loader=partial(_gguf_embedding_weight_loader, layer),
        )

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(
            layer,
            "qweight_type",
            fallback_weight_loader=_gguf_embedding_weight_type_loader,
        )

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if hasattr(layer, "gguf_compact"):
            return layer.gguf_compact[0].embedding(x)
        if hasattr(layer, "gguf_packed_embedding"):
            return layer.gguf_packed_embedding(x)

        from . import apply_gguf_embedding as apply_gguf_embedding_op

        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]
        return apply_gguf_embedding_op(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: VocabParallelEmbedding):
        del layer
        embed_tokens._gguf_tied_lm_head = True
        return embed_tokens
