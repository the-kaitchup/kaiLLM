# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash (`hy_v4_flash`) GGUF names -> HYV4FlashForCausalLM parameters.

The mapping inverts the converter from llama.cpp PR 32517. The model memory-maps the Engram
table from the GGUF file through config.hy4_engram_table.
"""

from __future__ import annotations

import re

import gguf

from ..qwen35_metadata import read_gguf_metadata, resolve_gguf_shards
from ..weight_utils import SCALAR_GGUF_TYPES, gguf_quant_weights_iterator_multi
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

TABLE = "per_layer_token_embd.weight"
GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
    "output_hc_fn.weight": "model.hc_head_fn",
    "output_hc_base.weight": "model.hc_head_base",
    "output_hc_scale.weight": "model.hc_head_scale",
}
LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "post_attention_norm.weight": "self_attn.post_proj_norm.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_k_b.weight": "self_attn.k_b.weight",
    "attn_v_b.weight": "self_attn.v_b.weight",
    "attn_gate.weight": "self_attn.gate_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_sinks.weight": "self_attn.sinks",
    "hc_attn_fn.weight": "attn_hc.fn", "hc_attn_base.weight": "attn_hc.base", "hc_attn_scale.weight": "attn_hc.scale",
    "hc_ffn_fn.weight": "ffn_hc.fn", "hc_ffn_base.weight": "ffn_hc.base", "hc_ffn_scale.weight": "ffn_hc.scale",
    "indexer.attn_k.weight": "self_attn.indexer.token_key_projection.weight",
    "indexer.k_norm.weight": "self_attn.indexer.token_key_norm",
    "indexer.attn_q_b.weight": "self_attn.indexer.query_projection.weight",
    "indexer.proj.weight": "self_attn.indexer.head_weight_projection",
    "indexer_compressor_ape.weight": "self_attn.indexer.intra_group_ape",
    "indexer_compressor_gate.weight": "self_attn.indexer.key_gate_projection",
    "indexer_compressor_norm.weight": "self_attn.indexer.key_norm",
    "indexer_compressor_value.weight": "self_attn.indexer.key_value_projection",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.router", "exp_probs_b.bias": "mlp.router_bias",
    "ffn_gate_exps.weight": "mlp.experts.gate.weight", "ffn_up_exps.weight": "mlp.experts.up.weight",
    "ffn_down_exps.weight": "mlp.experts.down.weight",
    "ffn_gate_shexp.weight": "mlp.shared.gate_proj.weight", "ffn_up_shexp.weight": "mlp.shared.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared.down_proj.weight",
    "ffn_routed_down.weight": "mlp.routed_down.weight", "ffn_routed_up.weight": "mlp.routed_up.weight",
    "engram_kv.weight": "engram.kv.weight", "engram_kv_b.weight": "engram.kv_bias",
    "engram_gate.weight": "engram.gate", "engram_lookup_norm.weight": "engram.lookup_norm",
    "engram_hc_scale.weight": "engram.hc_scale", "engram_pre_gamma.weight": "engram.pre_gamma",
}
# linear modules (vLLM quant methods); tensors stored F32/F16/BF16 load them unquantized
LINEAR_SUFFIXES = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "gate_proj", "o_proj", "up_proj", "down_proj",
                   "token_key_projection", "query_projection", "routed_down", "routed_up", "kv")
# targets that take quantized payloads as-is (GGUF linears, embeddings, packed experts, k_b / v_b)
_QUANT_OK = ("model.embed_tokens.weight", "lm_head.weight")


def takes_quantized(target: str) -> bool:
    if target in _QUANT_OK or ".mlp.experts." in target or target.endswith((".k_b.weight", ".v_b.weight")):
        return True
    mod = target.rsplit(".", 1)[0]
    return target.endswith(".weight") and mod.rsplit(".", 1)[-1] in LINEAR_SUFFIXES


def iter_dequantized(paths, names: dict[str, str]):
    """Small tensors community GGUFs quantize but the model keeps as plain float parameters
    (hyper-connection fn, indexer.proj, indexer compressor gate / value: Q8_0, BF16, NVFP4 seen),
    dequantized on the CPU with gguf-py."""
    import numpy as np
    import torch
    from gguf.quants import dequantize
    for path in paths:
        for t in gguf.GGUFReader(str(path)).tensors:
            if t.name in names:
                arr = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
                yield names[t.name], torch.from_numpy(np.ascontiguousarray(arr))


def hy4_name_map(config, names: set[str]) -> dict[str, str]:
    mapping = {}
    dense = config.first_k_dense_replace
    for g in names:
        if g == TABLE:
            continue
        if g in GLOBAL:
            mapping[g] = GLOBAL[g]
            continue
        m = re.match(r"blk\.(\d+)\.(.+)", g)
        if not m:
            raise ValueError(f"Unmapped Hy4 GGUF tensor: {g}")
        layer, tail = int(m.group(1)), m.group(2)
        if layer >= config.num_hidden_layers:
            continue  # draft (MTP / DFly) blocks are not part of the trunk
        if tail == "post_ffw_norm.weight":
            target = "mlp.combine_norm.weight" if layer >= dense else "mlp.post_norm.weight"
        elif tail in LAYER:
            target = LAYER[tail]
        else:
            raise ValueError(f"Unmapped Hy4 GGUF tensor: {g}")
        mapping[g] = f"model.layers.{layer}.{target}"
    return mapping


class Hy4GGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config):
        return config.get_text_config().model_type == "hy_v4_flash"

    def prepare_loading(self, model_path, model_config):
        paths = resolve_gguf_shards(model_path)
        readers = [gguf.GGUFReader(str(p)) for p in paths]
        tensors, where = {}, {}
        for path, r in zip(paths, readers):
            for t in r.tensors:
                tensors[t.name] = t
                where[t.name] = path
        md = read_gguf_metadata(readers[0])
        config = model_config.hf_config.get_text_config()
        mapping = hy4_name_map(config, set(tensors))
        unquantized = []
        for g, target in mapping.items():
            mod = target.rsplit(".", 1)[0]
            if (target.endswith(".weight") and mod.rsplit(".", 1)[-1] in LINEAR_SUFFIXES
                    and tensors[g].tensor_type.name in SCALAR_GGUF_TYPES):
                unquantized.append(mod)
        if tensors.get("output.weight") is not None and tensors["output.weight"].tensor_type.name in SCALAR_GGUF_TYPES:
            unquantized.append("lm_head")
        if tensors["token_embd.weight"].tensor_type.name in SCALAR_GGUF_TYPES:
            unquantized.append("model.embed_tokens")
        if TABLE in tensors:
            t = tensors[TABLE]
            dims = [int(x) for x in t.shape]                   # [dim, rows]
            config.hy4_engram_table = {"path": str(where[TABLE]), "offset": int(t.data_offset),
                                       "nbytes": int(t.n_bytes), "ggml_type": t.tensor_type.name,
                                       "rows": dims[1], "dim": dims[0]}
            config.hy4_engram_token_map = [int(x) for x in md.get("hy_v4_flash.engram.token_map", [])]
        self.dequant_names = {g: t for g, t in mapping.items()
                              if not takes_quantized(t) and tensors[g].tensor_type.name not in SCALAR_GGUF_TYPES}
        direct = {g: t for g, t in mapping.items() if g not in self.dequant_names}
        self.load_spec = GGUFLoadSpec([str(p) for p in paths], unquantized, direct)

    def prepare_weights(self, model_config):
        yield from iter_dequantized(self.load_spec.weights_source, self.dequant_names)
        yield from gguf_quant_weights_iterator_multi(self.load_spec.weights_source,
                                                     self.load_spec.gguf_to_hf_name_map)
