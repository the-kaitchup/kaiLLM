# SPDX-License-Identifier: Apache-2.0
"""LFM3 names and precision-preserving GGUF parameter loading."""

from types import MethodType

import gguf
import torch

from ..lfm3_metadata import lfm3_config_dict
from ..qwen35_metadata import read_gguf_metadata, resolve_gguf_shards
from ..weight_utils import SCALAR_GGUF_TYPES, gguf_quant_weights_iterator_multi
from ..yaqa_gguf import read_yaqa_profile
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec


def lfm3_name_map(num_layers):
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.embedding_norm.weight",
        "output.weight": "lm_head.weight",
    }
    suffixes = {
        "attn_norm.weight": "operator_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "post_attention_norm.weight": "post_operator_norm.weight",
        "post_ffw_norm.weight": "post_ffn_norm.weight",
        "ffn_gate.weight": "feed_forward.w1.weight",
        "ffn_up.weight": "feed_forward.w3.weight",
        "ffn_down.weight": "feed_forward.w2.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_gate.weight": "self_attn.gate_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "shortconv.conv.weight": "conv.conv.weight",
        "shortconv.in_proj.weight": "conv.in_proj.weight",
        "shortconv.out_proj.weight": "conv.out_proj.weight",
    }
    for layer in range(num_layers):
        for source, target in suffixes.items():
            mapping[f"blk.{layer}.{source}"] = f"model.layers.{layer}.{target}"
    return mapping


def _f32_norm_forward(self, x, residual=None):
    if residual is not None:
        x = x + residual
        residual = x
    value = x.float()
    value = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + self.variance_epsilon
    )
    if self.gguf_activation_precast:
        value = value.to(x.dtype).float()
    value = (value * self.weight.float()).to(x.dtype)
    return value if residual is None else (value, residual)


def _fused_norm_forward(self, x, residual=None):
    if not x.is_cuda or (residual is not None and residual.dtype != x.dtype):
        return _f32_norm_forward(self, x, residual)
    from ..quantization.lfm3_norm import fused_lfm3_rms_norm

    return fused_lfm3_rms_norm(
        x,
        self.weight,
        self.variance_epsilon,
        activation_precast=self.gguf_activation_precast,
        residual=residual,
    )


class Lfm3GGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config):
        return config.get_text_config().model_type == "lfm3"

    def prepare_loading(self, model_path, model_config):
        paths = resolve_gguf_shards(model_path)
        readers = [gguf.GGUFReader(str(p)) for p in paths]
        tensors, logical, components = {}, {}, set()
        has_profile = False
        for reader in readers:
            profile = read_yaqa_profile(reader)
            has_profile |= profile is not None
            descriptors = {} if profile is None else profile["tensors"]
            if logical.keys() & descriptors.keys():
                raise ValueError("Duplicate logical YAQA weights across GGUF shards")
            logical.update(descriptors)
            for descriptor in descriptors.values():
                components.update(descriptor["components"].values())
            for tensor in reader.tensors:
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
                tensors[tensor.name] = tensor
        names = set(tensors) | set(logical)
        factors = tensors.get("rope_freqs.weight")
        config_dict = lfm3_config_dict(
            read_gguf_metadata(readers[0]),
            names,
            factors.data.reshape(-1).tolist() if factors is not None else None,
            yaqa=has_profile,
        )
        config = model_config.hf_config.get_text_config()
        for key in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "layer_types",
            "conv_L_cache",
            "sliding_window",
            "use_attention_gate",
            "use_sandwich_norm",
            "conv_bias",
            "rope_parameters",
        ):
            if getattr(config, key) != config_dict[key]:
                raise ValueError(f"HF config {key} disagrees with LFM3 GGUF metadata")
        config.tie_word_embeddings = config_dict["tie_word_embeddings"]
        config.gguf_lfm3_norm_dtype = "float32"
        config.gguf_yaqa_profile = config_dict["gguf_yaqa_profile"]
        config.gguf_lfm3_norm_impl = config_dict["gguf_lfm3_norm_impl"]
        mapping = lfm3_name_map(config.num_hidden_layers)
        expected = {"token_embd.weight", "output_norm.weight"}
        for name, target in mapping.items():
            if not name.startswith("blk."):
                continue
            layer = int(name.split(".")[1])
            is_conv = config.layer_types[layer] == "conv"
            if ".conv." in target and not is_conv:
                continue
            if ".self_attn." in target and is_conv:
                continue
            if "post_" in target and not config.use_sandwich_norm:
                continue
            if (
                target.endswith("self_attn.gate_proj.weight")
                and not config.use_attention_gate
            ):
                continue
            expected.add(name)
        if not config.tie_word_embeddings:
            expected.add("output.weight")
        if missing := sorted(expected - names):
            raise ValueError(f"Missing LFM3 GGUF tensors: {missing}")
        if unexpected := sorted(
            names - mapping.keys() - components - {"rope_freqs.weight"}
        ):
            raise ValueError(f"Unmapped LFM3 GGUF tensors: {unexpected}")
        mapping = {k: v for k, v in mapping.items() if k in names}
        unquantized = [
            mapping[name].removesuffix(".weight")
            for name, tensor in tensors.items()
            if name in mapping and tensor.tensor_type.name in SCALAR_GGUF_TYPES
        ]
        self.load_spec = GGUFLoadSpec([str(p) for p in paths], unquantized, mapping)
        return self.load_spec

    def prepare_model(self, model):
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.model_loader.reload import (
            record_metadata_for_reloading,
        )

        self._parameter_dtypes = {}
        for name, module in model.named_modules():
            if isinstance(module, RMSNorm):
                module.weight.data = module.weight.data.float()
                module.gguf_activation_precast = (
                    self.config.gguf_lfm3_norm_impl == "activation_precast_f32_gain_v1"
                )
                module._forward_method = MethodType(_fused_norm_forward, module)
                with module.weight.device:
                    record_metadata_for_reloading(module)
            elif name.endswith(".short_conv.conv"):
                # The donor's depthwise convolution kernels also retain F32.
                module.weight.data = module.weight.data.float()
                with module.weight.device:
                    record_metadata_for_reloading(module)
        self._parameter_dtypes = {n: p.dtype for n, p in model.named_parameters()}
        from ..quantization.lfm3_dual_rms import prepare_model as prepare_dual_rms

        prepare_dual_rms(model)

    def prepare_weights(self, model_config):
        del model_config
        for name, weight in gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source, self.load_spec.gguf_to_hf_name_map
        ):
            if name.endswith(".conv.conv.weight"):
                weight = weight.unsqueeze(1)
            destination = name.replace(".conv.", ".short_conv.", 1)
            dtype = self._parameter_dtypes.get(destination)
            if dtype is not None and weight.is_floating_point():
                weight = weight.to(dtype)
            yield name, weight
