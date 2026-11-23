# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (`inkling2_loop`, llama.cpp master layout) GGUF names -> Inkling2LoopForCausalLM parameters."""

from __future__ import annotations

import re

import gguf

from ..qwen35_metadata import resolve_gguf_shards
from ..weight_utils import SCALAR_GGUF_TYPES, gguf_quant_weights_iterator_multi
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "token_embd_norm.weight": "model.embed_norm",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm",
    "output_res_query.weight": "model.output_res_query",
}
LINEARS = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_output": "o_proj",
           "attn_gate": "gate_proj", "attn_rel": "rel_proj", "attn_sink_proj": "sink_proj",
           "attn_k_conv_proj": "k_conv_proj", "attn_v_conv_proj": "v_conv_proj"}
PARAMS = {"attn_q_clip.weight": "attn.q_clip", "attn_rel_clip.weight": "attn.rel_clip",
          "attn_rel_bias.weight": "attn.rel_bias", "attn_k_conv.weight": "attn.k_conv",
          "attn_v_conv.weight": "attn.v_conv", "attn_post_conv.weight": "attn.post_conv",
          "ffn_post_conv.weight": "ffn_post_conv", "attn_norm.weight": "attn_norm",
          "attn_q_inp_norm.weight": "q_norm", "attn_k_inp_norm.weight": "k_norm", "attn_v_inp_norm.weight": "v_norm",
          "attn_res_query.weight": "attn_res_query", "ffn_res_query.weight": "ffn_res_query",
          "ffn_norm.weight": "ffn_norm", "ffn_gate_inp.weight": "router", "exp_probs_b.bias": "router_bias",
          "loop_msg_query.weight": "msg_query"}
EXPERTS = {"ffn_gate_up_exps.weight": "experts.gate_up.weight", "ffn_down_exps.weight": "experts.down.weight",
           "ffn_gate_up_exps.scale": "experts.gate_up_scale.weight",
           "ffn_down_exps.scale": "experts.down_scale.weight"}
CTRL = {"q": "q", "k": "k", "v": "v", "ffn": "ffn"}


def inkling_name_map(names: set[str]) -> dict[str, str]:
    out = {}
    for g in names:
        if g in GLOBAL:
            out[g] = GLOBAL[g]
            continue
        if g.startswith("itq.table."):
            continue
        m = re.match(r"blk\.(\d+)\.(.+)$", g)
        if not m:
            raise ValueError(f"Unmapped Inkling GGUF tensor: {g}")
        layer, tail = int(m.group(1)), m.group(2)
        base = f"model.layers.{layer}."
        lm = re.match(r"(attn_[a-z_]+)\.weight$", tail)
        cm = re.match(r"loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight$", tail)
        if lm and lm.group(1) in LINEARS:
            out[g] = base + f"attn.{LINEARS[lm.group(1)]}.weight"
        elif tail in PARAMS:
            out[g] = base + PARAMS[tail]
        elif tail in EXPERTS:
            out[g] = base + EXPERTS[tail]
        elif cm:
            out[g] = base + f"ctrl_{cm.group(1)}_{cm.group(2)}"
        elif ".itq." in tail:
            continue                       # ITQ components (yaqa.manifest): loaded by inkling/itq_moe.py
        else:
            raise ValueError(f"Unmapped Inkling GGUF tensor: {g}")
    return out


def _is_linear(target: str) -> bool:
    return ".attn." in target and target.endswith("_proj.weight") or target == "lm_head.weight"


def takes_quantized(target: str) -> bool:
    return (_is_linear(target) or target == "model.embed_tokens.weight" or ".experts.gate_up.weight" in target
            or ".experts.down.weight" in target)


class InklingGGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config):
        return config.get_text_config().model_type == "inkling2_loop"

    def prepare_loading(self, model_path, model_config):
        from ..yaqa_gguf import read_yaqa_manifest
        paths = resolve_gguf_shards(model_path)
        tensors, manifest = {}, {}
        for p in paths:
            reader = gguf.GGUFReader(str(p))
            for t in reader.tensors:
                tensors[t.name] = t
            manifest.update(read_yaqa_manifest(reader))
        components = {c for d in manifest.values() for c in d.get("components", {}).values()}
        mapping = inkling_name_map(set(tensors) - components)
        # dense itq linears (attention projections): kaiLLM's YAQA linear method loads their components; itq-moe
        # experts are built by inkling/loading.py from the manifest
        dense_itq = {name: d for name, d in manifest.items() if d.get("codec") == "itq"}
        per_pass = sorted(name for name, d in dense_itq.items() if "pass_output_scale" in d.get("components", {}))
        if per_pass:
            # the spec allows it in core layers; the model applies per-pass scales to experts only so far
            raise ValueError(f"dense itq pass_output_scale is not supported by the Inkling model yet: {per_pass[:4]}")
        mapping.update(inkling_name_map(set(dense_itq)))
        self.itq_experts = sorted(name for name, d in manifest.items() if d.get("codec") == "itq-moe")
        unquantized = []
        for g, target in mapping.items():
            if g not in tensors:          # logical dense-itq matrix
                continue
            scalar = tensors[g].tensor_type.name in SCALAR_GGUF_TYPES
            if scalar and _is_linear(target):
                unquantized.append(target.rsplit(".", 1)[0])
            if scalar and target == "model.embed_tokens.weight":
                unquantized.append("model.embed_tokens")
        # small float parameters that a file stores quantized: dequantized on the CPU at load
        self.dequant_names = {g: t for g, t in mapping.items()
                              if g in tensors and not takes_quantized(t)
                              and tensors[g].tensor_type.name not in SCALAR_GGUF_TYPES}
        direct = {g: t for g, t in mapping.items() if g not in self.dequant_names}
        self.load_spec = GGUFLoadSpec([str(p) for p in paths], unquantized, direct, yaqa_subset=True,
                                      compact_gguf=False)

    def prepare_weights(self, model_config):
        import numpy as np
        import torch
        from ..inkling.ggml_types import NEW_TYPES, dequantize as dq_new
        for path in self.load_spec.weights_source:
            for t in gguf.GGUFReader(str(path)).tensors:
                if t.name in self.dequant_names:
                    if int(t.tensor_type) in NEW_TYPES:
                        arr = dq_new(torch.from_numpy(np.array(t.data, copy=True)), int(t.tensor_type)).numpy()
                    else:
                        from gguf.quants import dequantize
                        arr = dequantize(t.data, t.tensor_type)
                    shape = tuple(reversed([int(x) for x in t.shape]))
                    yield self.dequant_names[t.name], torch.from_numpy(
                        np.ascontiguousarray(arr, dtype=np.float32).reshape(shape))
        yield from gguf_quant_weights_iterator_multi(self.load_spec.weights_source,
                                                     self.load_spec.gguf_to_hf_name_map)
