# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense GGUF tensor mappings and reversible export transforms."""

from __future__ import annotations

from collections.abc import Iterable

import gguf
import torch

from ..qwen35_metadata import (
    QWEN35_GDN_NORM_DTYPE,
    QWEN35_GDN_NORM_IMPL,
    configure_qwen35_gguf_norm,
    qwen35_config_dict,
    read_gguf_metadata,
    resolve_gguf_shards,
)
from ..weight_utils import gguf_quant_weights_iterator_multi
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec


def qwen35_name_map(num_layers: int) -> dict[str, str]:
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_a": "linear_attn.A_log",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
    }
    for layer in range(num_layers):
        for original, target in suffixes.items():
            mapping[f"blk.{layer}.{original}"] = f"model.layers.{layer}.{target}"
    return mapping


MTP_NORMS = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def qwen35_mtp_name_map(num_layers: int) -> dict[str, str]:
    """GGUF nextn/MTP layer (blk.<num_layers>) -> Qwen3_5MTP module names."""
    mtp = f"blk.{num_layers}."
    # The drafter's unquantized embed_tokens is replaced by the target's (vLLM
    # shares MTP embeddings and LM heads after loading), so it is not loaded.
    mapping = {
        "output.weight": "lm_head.weight",
        mtp + "nextn.eh_proj.weight": "model.fc.weight",
        mtp + "nextn.enorm.weight": "model.pre_fc_norm_embedding.weight",
        mtp + "nextn.hnorm.weight": "model.pre_fc_norm_hidden.weight",
        mtp + "nextn.shared_head_norm.weight": "model.norm.weight",
    }
    for original, target in qwen35_name_map(1).items():
        if original.startswith("blk.0.") and ".linear_attn." not in target:
            mapping[mtp + original.removeprefix("blk.0.")] = target
    return mapping


def qwen35_v_head_permutations(text_config) -> dict[str, torch.Tensor] | None:
    """Indices that undo llama.cpp's grouped->tiled linear-attention V-head order.

    When num_v_heads != num_k_heads, convert_hf_to_gguf stores the V heads of
    in_proj_qkv, in_proj_z, in_proj_a/b, A_log, dt_bias, the conv1d V channels
    and the out_proj input columns in tiled order (stored head j*K + k is group k,
    member j) so that ggml broadcasting pairs V head t with K head t % K. vLLM's
    GDN kernels pair V head g with K head g // (HV / K) (grouped order), so the
    loaded data must be stored[index] with these indices.
    """
    k = int(text_config.linear_num_key_heads)
    hv = int(text_config.linear_num_value_heads)
    if hv == k:
        return None
    if hv % k:
        raise ValueError("Qwen3.5 linear_num_value_heads must be a multiple of the key heads")
    per = hv // k
    grouped = torch.arange(hv, dtype=torch.long, device="cpu")
    heads = (grouped % per) * k + grouped // per
    dv = int(text_config.linear_value_head_dim)
    v = (heads[:, None] * dv + torch.arange(dv, dtype=torch.long, device="cpu")[None, :]).reshape(-1)
    qk = 2 * k * int(text_config.linear_key_head_dim)
    return {"heads": heads, "v": v, "qkv": torch.cat([torch.arange(qk, dtype=torch.long, device="cpu"), qk + v])}


def is_qwen35_mtp(config) -> bool:
    return "Qwen3_5MTP" in (getattr(config, "architectures", None) or [])


class Qwen35GGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config) -> bool:
        return config.get_text_config().model_type in ("qwen3_5_text", "qwen3_5_mtp")

    def prepare_model(self, model: torch.nn.Module) -> None:
        from vllm.model_executor.model_loader.reload import (
            record_metadata_for_reloading,
        )

        text_config = self.config.get_text_config()
        if not getattr(self, "is_mtp", False) and (
            getattr(text_config, "gguf_gdn_norm_dtype", None) != QWEN35_GDN_NORM_DTYPE
            or getattr(text_config, "gguf_gdn_norm_impl", None) != QWEN35_GDN_NORM_IMPL
        ):
            raise ValueError("GGUF GDN norm policy disagrees with its config markers")

        # GGUF stores these weights in F32, and gated RMS norm already computes
        # with F32 weights independently of its input/output dtype. Allocate F32
        # storage before loading so copying does not discard the original bits.
        self._parameter_dtypes = {}
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        from ..quantization import kai_gemma_norm

        for module_name, module in model.named_modules():
            if (
                isinstance(module, GemmaRMSNorm)
                and module_name.endswith(
                    ("input_layernorm", "post_attention_layernorm", "model.norm",
                     "pre_fc_norm_embedding", "pre_fc_norm_hidden")
                )
                and kai_gemma_norm.bind(module)
            ):
                # GGUF stores the exact F32 multiplier (1 + w): keep w in F32.
                module.weight.data = module.weight.data.to(dtype=torch.float32)
                with module.weight.device:
                    record_metadata_for_reloading(module)
        import types as _types

        def _mlp_forward(mlp, x):
            gate_up, _ = mlp.gate_up_proj(x)
            method = getattr(mlp.down_proj, "quant_method", None)
            if hasattr(method, "apply_swiglu"):
                return method.apply_swiglu(mlp.down_proj, gate_up, mlp.act_fn)
            out, _ = mlp.down_proj(mlp.act_fn(gate_up))
            return out

        for module_name, module in model.named_modules():
            if (
                module_name.endswith(".mlp")
                and all(hasattr(module, a) for a in ("gate_up_proj", "down_proj", "act_fn"))
                and getattr(module, "expert_gate", None) is None
            ):
                module.forward = _types.MethodType(_mlp_forward, module)
            if module_name.endswith(("input_layernorm", "post_attention_layernorm")) and getattr(
                module, "_kai_original_forward", None
            ) is not None:
                parent = model.get_submodule(module_name.rsplit(".", 1)[0])
                if module_name.endswith("input_layernorm"):
                    owners = ((getattr(parent, "linear_attn", None), ("in_proj_qkvz", "in_proj_ba")),
                              (getattr(parent, "self_attn", None), ("qkv_proj",)))
                else:
                    owners = ((getattr(parent, "mlp", None), ("gate_up_proj",)),)
                consumers = [getattr(owner, name, None) for owner, names in owners if owner is not None
                             for name in names]
                consumers = [c for c in consumers if c is not None]
                if consumers:
                    # The norm writes its output's Q8_1 activation; these projections reuse it.
                    module.__dict__["_kai_np_consumers"] = True
                    for consumer in consumers:
                        consumer.__dict__["_kai_act_producer"] = id(module)
        for name, parameter in model.named_parameters():
            if name.endswith(".linear_attn.norm.weight"):
                parameter.data = parameter.data.to(dtype=torch.float32)
                norm = model.get_submodule(name.rsplit(".", 1)[0])
                if parameter.device.type == "cuda":
                    # Use vLLM's existing FLA CUDA norm explicitly for these
                    # GGUF layers. CPU dispatch and other norms are unchanged.
                    norm._forward_method = norm.forward_cuda
                # initialize_model captured the original model dtype. Refresh
                # only this norm, without recapturing processed quantized layers.
                with parameter.device:
                    record_metadata_for_reloading(norm)
            self._parameter_dtypes[name] = parameter.data.dtype

    def prepare_loading(self, model_path, model_config) -> GGUFLoadSpec:
        # Direct loader callers may bypass EngineArgs and the GGUF config
        # parser. This still runs before initialize_model creates AOT wrappers.
        self.is_mtp = is_qwen35_mtp(model_config.hf_config)
        configure_qwen35_gguf_norm(model_config.hf_config)
        paths = resolve_gguf_shards(model_path)
        readers = [gguf.GGUFReader(str(path)) for path in paths]
        tensors = {}
        for reader in readers:
            for tensor in reader.tensors:
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
                tensors[tensor.name] = tensor
        metadata = read_gguf_metadata(readers[0])
        config_dict = qwen35_config_dict(metadata, set(tensors))
        text_config = model_config.hf_config.get_text_config()
        for key in ("num_hidden_layers", "hidden_size", "vocab_size", "layer_types"):
            if getattr(text_config, key) != config_dict[key]:
                raise ValueError(f"HF config {key} disagrees with GGUF metadata")
        text_config.tie_word_embeddings = config_dict["tie_word_embeddings"]
        if self.is_mtp:
            mapping = qwen35_mtp_name_map(text_config.num_hidden_layers)
            missing = sorted(
                name for name in mapping
                if name not in tensors
                and not (name == "output.weight" and text_config.tie_word_embeddings)
            )
            if missing:
                raise ValueError(f"Missing Qwen3.5 MTP GGUF tensors: {missing}")
            mapping = {name: target for name, target in mapping.items() if name in tensors}
            unquantized = [
                mapping[name].removesuffix(".weight")
                for name, tensor in tensors.items()
                if name in mapping
                and name.endswith(".weight")
                and tensor.tensor_type.name in ("F32", "F16", "BF16")
            ]
            self.load_spec = GGUFLoadSpec(
                weights_source=[str(path) for path in paths],
                unquantized_modules=unquantized,
                gguf_to_hf_name_map=mapping,
            )
            return self.load_spec
        mapping = qwen35_name_map(text_config.num_hidden_layers)
        expected = {
            "token_embd.weight",
            "output_norm.weight",
        }
        for name in mapping:
            if not name.startswith("blk."):
                continue
            layer = int(name.split(".")[1])
            is_linear = text_config.layer_types[layer] == "linear_attention"
            target = mapping[name]
            if (".linear_attn." in target and not is_linear) or (
                ".self_attn." in target and is_linear
            ):
                continue
            expected.add(name)
        if not text_config.tie_word_embeddings:
            expected.add("output.weight")
        missing = sorted(expected - tensors.keys())
        if missing:
            raise ValueError(f"Missing Qwen3.5 GGUF tensors: {missing}")
        main_names = {
            name
            for name in tensors
            if not name.startswith("blk.")
            or int(name.split(".")[1]) < text_config.num_hidden_layers
        }
        unexpected = sorted(main_names - mapping.keys())
        if unexpected:
            raise ValueError(f"Unmapped Qwen3.5 GGUF tensors: {unexpected}")
        unquantized = [
            mapping[name].removesuffix(".weight")
            for name, tensor in tensors.items()
            if name in mapping
            and name.endswith(".weight")
            and tensor.tensor_type.name in ("F32", "F16", "BF16")
        ]
        rows, columns = {}, {}
        self._row_permutations, self._column_permutations = {}, {}
        perms = qwen35_v_head_permutations(text_config)
        if perms is not None:
            if any(key.startswith("prism.hadamard") for key in readers[0].fields):
                raise ValueError("Hadamard-rotated Qwen3.5 GGUFs are not supported")
            for layer, kind in enumerate(text_config.layer_types):
                if kind != "linear_attention":
                    continue
                blk = f"blk.{layer}."
                rows[blk + "attn_qkv.weight"] = perms["qkv"]
                rows[blk + "attn_gate.weight"] = perms["v"]
                rows[blk + "ssm_alpha.weight"] = perms["heads"]
                rows[blk + "ssm_beta.weight"] = perms["heads"]
                rows[blk + "ssm_conv1d.weight"] = perms["qkv"]
                rows[blk + "ssm_dt.bias"] = perms["heads"]
                rows[blk + "ssm_a"] = perms["heads"]
                columns[blk + "ssm_out.weight"] = perms["v"]
            # Tensors that reach prepare_weights (F32 and non-compact matrices).
            self._row_permutations = {mapping[name]: index for name, index in rows.items()}
            self._column_permutations = {mapping[name]: index for name, index in columns.items()}
        self.load_spec = GGUFLoadSpec(
            weights_source=[str(path) for path in paths],
            unquantized_modules=unquantized,
            gguf_to_hf_name_map=mapping,
            gguf_row_permutations=rows or None,
            gguf_column_permutations=columns or None,
        )
        return self.load_spec

    def prepare_weights(self, model_config) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        for name, weight in gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source, self.load_spec.gguf_to_hf_name_map
        ):
            is_type = name.endswith(".qweight_type")
            base = name.removesuffix(".qweight") + ".weight" if name.endswith(".qweight") else name
            row_index = None if is_type else getattr(self, "_row_permutations", {}).get(base)
            column_index = None if is_type else getattr(self, "_column_permutations", {}).get(base)
            if name.endswith(".qweight") and (row_index is not None or column_index is not None):
                # Compact GGUF layers apply these permutations at construction.
                raise ValueError(f"V-head reordered Qwen3.5 GGUF matrices need the compact backend: {name}")
            if row_index is not None:
                weight = weight[row_index]
            if column_index is not None:
                weight = weight[:, column_index]
            if name.endswith(".A_log"):
                if not torch.all(torch.isfinite(weight) & (weight < 0)):
                    raise ValueError(f"Expected finite negative GGUF SSM A: {name}")
                weight = torch.log(-weight.float())
            elif name.endswith(".conv1d.weight"):
                weight = weight.unsqueeze(1)
            elif name == "model.norm.weight" or name.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".self_attn.q_norm.weight",
                    ".self_attn.k_norm.weight",
                )
                + MTP_NORMS
            ):
                weight = weight.float() - 1.0
            # Match the copy into known floating destinations, after all F32
            # export transforms. Fused shards without an exact parameter name
            # retain their source dtype for their specialized weight loader.
            target_dtype = getattr(self, "_parameter_dtypes", {}).get(name)
            if (
                weight.is_floating_point()
                and target_dtype is not None
                and target_dtype.is_floating_point
            ):
                weight = weight.to(dtype=target_dtype)
            if getattr(self, "is_mtp", False) and name.startswith("model.") and "embed_tokens" not in name:
                # Qwen3_5MTP.load_weights accepts mtp.* names and maps them to model.*.
                name = "mtp." + name.removeprefix("model.")
            yield name, weight
