# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (`inkling2_loop`, text model) for kaiLLM GGUF serving.

Semantics: llama.cpp master src/models/inkling2{,-loop}.cpp, mirrored op for op by vllm_gguf_plugin/inkling/ref.py
(checked against llama.cpp on toy GGUFs). 24 physical layers; layers [loop_start, loop_end) run loop_count passes
with shared weights, every pass restarting from the prelude output and, after the first, blending messages from the
previous pass into the q / k / v / ffn inputs. Each layer visit owns a KV cache (`model.slots.<slot>.attn`) and a
conv state (`model.slots.<slot>.conv`): 48 slots, numbered like llama.cpp's cache slots.

The residual stream, mixes, norms, controllers, convs, router and attention softmax run in float32 (as llama.cpp);
linears and experts take the model dtype: the norms in front of them write it directly, and their outputs are
widened inside the next op (no separate cast kernels). Tensor / pipeline parallelism: not supported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.sequence import IntermediateTensors

from vllm_gguf_plugin.inkling import glue as G
from vllm_gguf_plugin.inkling import ref as R
from vllm_gguf_plugin.inkling.attention import attend, make_kv_cache
from vllm_gguf_plugin.inkling.conv import conv_step, make_conv_state
from vllm_gguf_plugin.inkling.moe import InklingExperts, route

CTRL = ("q", "k", "v", "ffn")


def _param(*shape):
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


class InklingLayer(nn.Module):
    """One physical decoder layer (weights shared by its visits)."""

    def __init__(self, config, layer: int, quant_config, prefix: str):
        super().__init__()
        c = config
        self.layer = layer
        self.swa = bool(c.sliding_window_pattern[layer])
        self.looped = c.loop_start <= layer < c.loop_end
        D, H, Hkv, hd, nrel, nconv = (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                                      c.relative_length, c.conv_L_cache)
        self.D, self.H, self.Hkv, self.hd, self.nrel = D, H, Hkv, hd, nrel
        kvd = hd * Hkv

        def lin(name, n_in, n_out):
            return ReplicatedLinear(n_in, n_out, bias=False, quant_config=quant_config, prefix=f"{prefix}.attn.{name}")

        self.attn = nn.Module()
        a = self.attn
        a.q_proj, a.k_proj, a.v_proj = lin("q_proj", D, H * hd), lin("k_proj", D, kvd), lin("v_proj", D, kvd)
        a.o_proj, a.gate_proj = lin("o_proj", H * hd, D), lin("gate_proj", D, H * hd)
        a.rel_proj, a.sink_proj = lin("rel_proj", D, nrel * H), lin("sink_proj", D, H)
        if not self.swa:
            a.k_conv_proj, a.v_conv_proj = lin("k_conv_proj", D, nconv * Hkv), lin("v_conv_proj", D, nconv * Hkv)
        a.q_clip, a.rel_clip = _param(H, hd), _param(H, nrel)
        self.extent = c.sliding_window if self.swa else c.relative_extent
        a.rel_bias = _param(self.extent, nrel)
        a.k_conv, a.v_conv, a.post_conv = _param(kvd, nconv), _param(kvd, nconv), _param(D, nconv)
        self.ffn_post_conv = _param(D, nconv)
        if self.swa:
            self.attn_norm = _param(D)
        else:
            self.q_norm, self.k_norm, self.v_norm = _param(D), _param(D), _param(D)
        self.attn_res_query = _param(1 if self.swa else 3, D) if layer > 0 else None
        self.ffn_res_query = _param(1, D)
        self.ffn_norm = _param(D)
        self.router = _param(c.num_experts, D)
        self.router_bias = _param(c.num_experts)
        self.experts = InklingExperts(c.num_experts, D, c.moe_intermediate_size, c.num_experts_per_tok, c.situ_beta,
                                      c.situ_linear_beta, c.expert_weights_scale)
        if self.looped:
            self.msg_query = _param(4, D)
            r = c.loop_gate_rank
            for cc in CTRL:
                setattr(self, f"ctrl_{cc}_down", _param(r, 2 * D))
                setattr(self, f"ctrl_{cc}_gain", _param(D, r))
                setattr(self, f"ctrl_{cc}_forget", _param(D, r))
        self.eps = c.rms_norm_eps
        self.qk_clip, self.rel_clip_thr = c.qk_clip, c.relative_clip
        self.window = c.sliding_window if self.swa else None
        self.block = c.attn_res_block_size
        self.kvd = kvd
        self.act_dtype = torch.get_default_dtype()       # the model dtype while vLLM builds the model

    def _lin(self, m, x):
        """Projection of a model-dtype input; the output stays in the model dtype (consumers widen it)."""
        y = m(x if x.dtype == self.act_dtype else x.to(self.act_dtype))
        return y[0] if isinstance(y, tuple) else y

    def forward(self, x, blocks: list, positions, tau, kv, conv, messages=None, pass_index: int = 0):
        a = self.attn
        T, D = x.shape
        inp_sa = x
        inputs = [x, x, x]
        if self.attn_res_query is not None:
            mixed = G.mix(blocks, x, self.attn_res_query)
            inputs = [mixed[:, 0 if self.swa else j] for j in range(3)]
        if messages is not None:
            inputs = [G.control(inputs[j], messages[:, j], getattr(self, f"ctrl_{CTRL[j]}_down"),
                                getattr(self, f"ctrl_{CTRL[j]}_gain"), getattr(self, f"ctrl_{CTRL[j]}_forget"))
                      for j in range(3)]
        dt, f32 = self.act_dtype, torch.float32
        if self.swa:
            if messages is not None:
                inputs = [G.rms(t, self.attn_norm, self.eps, dt) for t in inputs]
            else:
                inputs = [G.rms(inputs[0], self.attn_norm, self.eps, dt)] * 3
        else:
            inputs = [G.rms(inputs[0], self.q_norm, self.eps, dt), G.rms(inputs[1], self.k_norm, self.eps, dt),
                      G.rms(inputs[2], self.v_norm, self.eps, dt)]
        q = self._lin(a.q_proj, inputs[0])
        kp = self._lin(a.k_proj, inputs[1])
        vp = self._lin(a.v_proj, inputs[2])
        dk = None if self.swa else self._lin(a.k_conv_proj, inputs[1])
        dv = None if self.swa else self._lin(a.v_conv_proj, inputs[2])
        k = conv_step(conv, kp, a.k_conv, 0, dk, self.Hkv, f32)
        v = conv_step(conv, vp, a.v_conv, self.kvd, dv, self.Hkv, f32)
        rel = self._lin(a.rel_proj, inputs[0])
        q, k, v, rel = G.post_projection(q, k, v, rel, a.q_clip, a.rel_clip, tau, self.qk_clip, self.rel_clip_thr,
                                         self.eps, self.H, self.Hkv, self.hd, self.nrel, not self.swa)
        sink = self._lin(a.sink_proj, inputs[0])
        att = attend(kv, q, k, v, rel, sink, a.rel_bias, positions, 1.0 / math.sqrt(self.hd), self.window,
                     self.extent)
        att = G.gate_attn(att, self._lin(a.gate_proj, inputs[0]), dt)
        out = conv_step(conv, self._lin(a.o_proj, att), a.post_conv, 2 * self.kvd, None, 1, f32)
        if self.layer % self.block == 0:
            blocks.append(inp_sa)
            x = out
        else:
            x = inp_sa + out
        ffn_inp = x
        h = G.mix(blocks, x, self.ffn_res_query)[:, 0]
        if messages is not None:
            h = G.control(h, messages[:, 3], self.ctrl_ffn_down, self.ctrl_ffn_gain, self.ctrl_ffn_forget)
        h, h_dt = G.rms_pair(h, self.ffn_norm, self.eps, dt)
        ids, w = G.route(h, self.router, self.router_bias, self.experts.k, self.experts.w_scale)
        y = self.experts(h_dt, ids, w, pass_index, f32)
        y = conv_step(conv, y, self.ffn_post_conv, 2 * self.kvd + D, None, 1, f32)
        return ffn_inp + y


class InklingModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        self.config = c
        self.embed_tokens = VocabParallelEmbedding(c.vocab_size, c.hidden_size, quant_config=quant_config,
                                                   prefix=f"{prefix}.embed_tokens")
        self.embed_norm = _param(c.hidden_size)
        self.layers = nn.ModuleList([InklingLayer(c, i, quant_config, f"{prefix}.layers.{i}")
                                     for i in range(c.num_hidden_layers)])
        dtype = vllm_config.model_config.dtype
        kv_dtype = str(getattr(cache_config, "cache_dtype", "auto") or "auto")
        if kv_dtype.startswith("fp8"):
            # keys are clipped (L2 <= qk_clip) and values RMS-normalized: E4M3 without scales holds both
            dtype = torch.float8_e4m3fn
        elif kv_dtype in ("bfloat16", "float16", "float32"):
            dtype = getattr(torch, kv_dtype)
        self.slots = nn.ModuleList()
        for s in range(c.num_slots):
            slot = nn.Module()
            phys = s if s < c.num_hidden_layers else c.loop_start + (s - c.num_hidden_layers) % (c.loop_end - c.loop_start)
            window = c.sliding_window if c.sliding_window_pattern[phys] else None
            slot.attn = make_kv_cache(c.num_key_value_heads, c.head_dim, dtype, f"{prefix}.slots.{s}.attn",
                                      cache_config, window=window)
            slot.conv = make_conv_state(c.conv_channels, c.conv_L_cache, f"{prefix}.slots.{s}.conv",
                                        vllm_config.model_config, cache_config)
            self.slots.append(slot)
        self.norm = _param(c.hidden_size)
        self.output_res_query = _param(1, c.hidden_size)
        self.model_dtype = vllm_config.model_config.dtype

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        c = self.config
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        x = G.rms(x, self.embed_norm, c.rms_norm_eps)
        tau = R.log_tau(positions, c.log_scale_floor, c.log_scale_alpha).float()
        blocks: list = []

        def visit(layer, pass_index, x, blocks, messages=None):
            s = c.slot(layer, pass_index)
            return self.layers[layer](x, blocks, positions, tau, self.slots[s].attn, self.slots[s].conv, messages,
                                      pass_index)

        for il in range(c.loop_start):
            x = visit(il, 0, x, blocks)
        prelude, prelude_blocks = x, list(blocks)
        bank = bank_out = None
        for p in range(c.loop_count):
            if p > 0:
                bank, bank_out = blocks[len(prelude_blocks) + 1:], x
                x, blocks = prelude, list(prelude_blocks)
            for ip in range(c.loop_start, c.loop_end):
                msgs = G.mix(bank, bank_out, self.layers[ip].msg_query) if p > 0 else None
                x = visit(ip, p, x, blocks, msgs)
        for il in range(c.loop_end, c.num_hidden_layers):
            x = visit(il, 0, x, blocks)
        h = G.mix(blocks, x, self.output_res_query)[:, 0]
        return G.rms(h, self.norm, c.rms_norm_eps, self.model_dtype)


class Inkling2LoopForCausalLM(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        c = vllm_config.model_config.hf_text_config
        self.config = c
        self.model = InklingModel(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(c.vocab_size, c.hidden_size, quant_config=vllm_config.quant_config,
                                      prefix="lm_head")
        self.logits_processor = LogitsProcessor(c.vocab_size, soft_cap=c.final_logit_softcapping or None)
        from vllm_gguf_plugin.qwen35_metadata import resolve_gguf_shards
        self.gguf_paths = []
        mc = vllm_config.model_config
        for ref in (getattr(mc, "model_weights", None), mc.model):     # the plugin keeps the GGUF in model_weights
            try:
                if ref and str(ref).endswith(".gguf"):
                    self.gguf_paths = [str(p) for p in resolve_gguf_shards(ref)]
                    break
            except Exception:              # not a local GGUF (no ITQ experts to load)
                continue

    # hybrid cache planning: one short-conv state per layer visit
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
        return MambaStateDtypeCalculator.short_conv_state_dtype(vllm_config.model_config.dtype,
                                                                vllm_config.cache_config.mamba_cache_dtype)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
        c = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.short_conv_state_shape(tp_world_size=1, intermediate_size=c.conv_channels,
                                                               conv_kernel=c.conv_L_cache)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.config.vocab_size_unpadded < logits.shape[-1]:
            logits[..., self.config.vocab_size_unpadded:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm_gguf_plugin.inkling.loading import load_inkling_weights
        return load_inkling_weights(self, weights)
