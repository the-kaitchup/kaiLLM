# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash (`hy_v4_flash`) text model for kaiLLM GGUF serving.

Math: transformers PR 50903 (checked on CPU against kaiLLM's pure blocks,
vllm_gguf_plugin/hy4/{attention,blocks}.py). Structure follows Tencent's vLLM fork: every attention
layer is sparse MLA over the paged latent cache with per-token index lists (sliding windows, or the
compressed indexer's picks), learnable sinks, gated output and a sandwich norm; 4-stream
hyper-connections; latent MoE (GGUF experts in a 1536-wide latent, clamped SwiGLU); Engram n-gram
memory read from a host-resident table. Tensor parallelism and pipeline parallelism: not yet.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

from vllm_gguf_plugin.hy4 import attention as A
from vllm_gguf_plugin.hy4 import blocks as B
from vllm_gguf_plugin.hy4.moe import Hy4GGUFExperts
from vllm_gguf_plugin.hy4.engram import HostEngramTable


def _prefix_layer(prefix: str) -> int:
    return int(prefix.split("layers.")[1].split(".")[0])


class _PerTokenCache(nn.Module, AttentionLayerBase):
    """A paged per-token vector cache (indexer token keys, Engram token ids), managed by vLLM's
    KV cache manager next to the MLA latent cache."""

    def __init__(self, width: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.width, self.dtype, self.prefix, self.cache_config = width, dtype, prefix, cache_config
        ctx = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in ctx:
            raise ValueError(f"Duplicate layer name: {prefix}")
        ctx[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)           # [blocks, block_size, width]

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        return MLAAttentionSpec(block_size=self.cache_config.block_size, num_kv_heads=1, max_tp_shards=1,
                                head_size=self.width, dtype=self.dtype, cache_role=SparseCacheRole.INDEXER)

    def forward(self): ...

    @property
    def uses_shuffled_layout(self) -> bool:
        return False

    def get_attn_backend(self):
        return A.hy4_sparse_mla_backend()

    # helpers -----------------------------------------------------------------
    def metadata(self):
        md = get_forward_context().attn_metadata
        return None if md is None else (md[self.prefix] if isinstance(md, dict) else md)

    def _rows(self):
        """Flat [row, width] view and physical rows per block (blocks may be strided apart)."""
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
        return flat_kv_row_view(self.kv_cache, self.kv_cache.shape[1])

    def write(self, values: torch.Tensor, md) -> None:
        flat, stride = self._rows()
        bs = self.kv_cache.shape[1]
        n = min(values.shape[0], md.slot_mapping.shape[0])
        slots = md.slot_mapping[:n].long()
        ok = slots >= 0
        slots = slots[ok]
        flat[(slots // bs) * stride + slots % bs] = values[:n][ok].to(flat.dtype)

    def gather(self, req: torch.Tensor, positions: torch.Tensor, md) -> torch.Tensor:
        """Cached vectors at `positions` (>= 0) of requests `req` (batch-local ids), same shape + [width]."""
        flat, stride = self._rows()
        bs = self.kv_cache.shape[1]
        pos = positions.long()
        rows = md.block_table[req.long(), pos // bs].long() * stride + pos % bs
        return flat[rows]

    def read_positions(self, req: int, positions: torch.Tensor, md) -> torch.Tensor:
        return self.gather(torch.full_like(positions, req), positions, md)


class Hy4CompressedIndexer(nn.Module):
    """Indexed layers: all previous tokens below candidate_topk, else the compressed indexer."""

    def __init__(self, config, layer_idx: int, topk_buf: torch.Tensor, prefix: str, cache_config, quant_config):
        super().__init__()
        self.topk_indices_buffer = topk_buf
        self.topk_tokens = topk_buf.shape[1]
        self.candidate_topk = int(config.compressed_indexer_candidate_topk)
        self.heads = int(config.compressed_indexer_num_attention_heads)
        self.head_dim = int(config.compressed_indexer_head_dim)
        self.ratio = int(config.compressed_indexer_compress_ratio)
        self.group_topk = int(config.compressed_indexer_group_topk)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.theta = config.rope_theta_for(layer_idx)
        self.eps = config.rms_norm_eps
        H = config.hidden_size
        self.query_projection = ReplicatedLinear(config.q_lora_rank, self.heads * self.head_dim, bias=False,
                                                 quant_config=quant_config, prefix=f"{prefix}.query_projection")
        self.token_key_projection = ReplicatedLinear(H, self.head_dim, bias=False, quant_config=quant_config,
                                                     prefix=f"{prefix}.token_key_projection")
        f32 = torch.float32
        self.head_weight_projection = nn.Parameter(torch.empty(self.heads, H, dtype=f32), requires_grad=False)
        self.token_key_norm = nn.Parameter(torch.empty(self.head_dim, dtype=f32), requires_grad=False)
        self.key_value_projection = nn.Parameter(torch.empty(self.head_dim, self.head_dim, dtype=f32), requires_grad=False)
        self.key_gate_projection = nn.Parameter(torch.empty(self.head_dim, self.head_dim, dtype=f32), requires_grad=False)
        self.intra_group_ape = nn.Parameter(torch.empty(self.ratio, self.head_dim, dtype=f32), requires_grad=False)
        self.key_norm = nn.Parameter(torch.empty(self.head_dim, dtype=f32), requires_grad=False)
        # raw token keys in the model dtype (BF16 serving, like Tencent's runtime; F32 in exactness tests)
        self.key_cache = _PerTokenCache(self.head_dim, torch.get_default_dtype(), f"{prefix}.key_cache", cache_config)

    def _weights(self):
        return A.IndexerWeights(None,
                                self.head_weight_projection, self.token_key_norm, self.key_value_projection,
                                self.key_gate_projection, self.intra_group_ape, self.key_norm, self.heads,
                                self.head_dim, self.rope_dim, self.ratio, self.group_topk, self.theta, self.eps)

    def forward(self, hidden_states, q_c, positions, rope=None):
        T = hidden_states.shape[0]
        out = self.topk_indices_buffer[:T]
        md = self.key_cache.metadata()
        keys = A.rms(self.token_key_projection(hidden_states)[0], self.token_key_norm, self.eps)
        if md is None:  # profiling run
            out.copy_(A.dense_indices(positions[:T], self.topk_tokens))
            return out
        self.key_cache.write(keys, md)
        out.copy_(A.dense_indices(positions[:T], self.topk_tokens))
        seq_lens = md.seq_lens.tolist()
        if max(seq_lens) <= self.candidate_topk:
            return out
        starts = md.query_start_loc.tolist()
        w = self._weights()
        q_all = self.query_projection(q_c)[0]            # GGUF linear (any backend), [T, H_i * D_i]
        for r, L in enumerate(seq_lens):
            if L <= self.candidate_topk:
                continue
            s, e = starts[r], starts[r + 1]
            hist = self.key_cache.read_positions(r, torch.arange(L, device=keys.device), md)
            groups = A.pool_groups(hist, w)
            out[s:e] = A.select_groups(hidden_states[s:e], q_c[s:e], positions[s:e], groups, w, q=q_all[s:e])
        return out



class Hy4Attention(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config, cache_config, topk_buf, prefix: str):
        super().__init__()
        self.sliding = config.is_sliding(layer_idx)
        H = config.heads_for(layer_idx)
        self.num_heads, self.layer_idx = H, layer_idx
        self.nope, self.rope = config.qk_nope_head_dim, config.qk_rope_head_dim
        self.v_dim, self.kv_lora, self.q_lora = config.v_head_dim, config.kv_lora_rank, config.q_lora_rank
        self.theta = config.rope_theta_for(layer_idx)
        hid, eps = config.hidden_size, config.rms_norm_eps
        lin = lambda i, o, name, q=quant_config: ReplicatedLinear(i, o, bias=False, quant_config=q,
                                                                  prefix=f"{prefix}.{name}")
        self.q_a_proj = lin(hid, self.q_lora, "q_a_proj")
        self.q_a_layernorm = RMSNorm(self.q_lora, eps=eps)
        self.q_b_proj = lin(self.q_lora, H * (self.nope + self.rope), "q_b_proj")
        self.kv_a_proj_with_mqa = lin(hid, self.kv_lora + self.rope, "kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora, eps=eps)
        self.kv_b_proj = lin(self.kv_lora, H * (self.nope + self.v_dim), "kv_b_proj", None)  # dense, from k_b / v_b
        self.gate_proj = lin(hid, H * self.v_dim, "gate_proj")
        self.o_proj = lin(H * self.v_dim, hid, "o_proj")
        self.post_proj_norm = RMSNorm(hid, eps=eps)
        self.sinks = nn.Parameter(torch.zeros(H, dtype=torch.float32), requires_grad=False)
        if self.sliding:
            self.indexer = A.Hy4WindowIndexer(config.sliding_window, topk_buf)
        else:
            self.indexer = Hy4CompressedIndexer(config, layer_idx, topk_buf, f"{prefix}.indexer", cache_config,
                                                quant_config)
        self.mla_attn = MLAAttention(
            num_heads=H, scale=(self.nope + self.rope) ** -0.5, qk_nope_head_dim=self.nope,
            qk_rope_head_dim=self.rope, v_head_dim=self.v_dim, q_lora_rank=self.q_lora, kv_lora_rank=self.kv_lora,
            kv_b_proj=self.kv_b_proj, cache_config=cache_config, quant_config=None, prefix=f"{prefix}.attn",
            attn_backend=A.hy4_sparse_mla_backend(), use_sparse=True, indexer=self.indexer,
            topk_indices_buffer=topk_buf, hy4_sinks=self.sinks)

    def forward(self, positions: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        T = h.shape[0]
        q_c = self.q_a_layernorm(self.q_a_proj(h)[0])
        kv = self.kv_a_proj_with_mqa(h)[0]
        kv_c, k_pe = kv.split([self.kv_lora, self.rope], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c.contiguous())
        q = self.q_b_proj(q_c)[0].view(T, self.num_heads, self.nope + self.rope)
        q[..., self.nope:] = A.rope_interleaved(q[..., self.nope:], positions.view(T, 1).expand(T, self.num_heads),
                                                self.theta)
        k_pe = A.rope_interleaved(k_pe, positions, self.theta).unsqueeze(1)
        self.indexer(h, q_c, positions)
        o = self.mla_attn(q, kv_c, k_pe, output_shape=(T, self.num_heads * self.v_dim))
        o = o * torch.sigmoid(self.gate_proj(h)[0])
        return self.post_proj_norm(self.o_proj(o)[0])


class Hy4MLP(nn.Module):
    def __init__(self, hidden, inter, quant_config, prefix, post_norm_eps=None):
        super().__init__()
        lin = lambda i, o, n: ReplicatedLinear(i, o, bias=False, quant_config=quant_config, prefix=f"{prefix}.{n}")
        self.gate_proj, self.up_proj, self.down_proj = lin(hidden, inter, "gate_proj"), lin(hidden, inter, "up_proj"), \
            lin(inter, hidden, "down_proj")
        self.post_norm = RMSNorm(hidden, eps=post_norm_eps) if post_norm_eps is not None else None

    def forward(self, x):
        y = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]
        return self.post_norm(y) if self.post_norm is not None else y


class Hy4MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        H, lat = config.hidden_size, config.routed_expert_hidden_size or config.hidden_size
        E, I = config.n_routed_experts, config.moe_intermediate_size
        self.top_k, self.norm, self.scale = config.num_experts_per_tok, config.norm_topk_prob, config.routed_scaling_factor
        self.router = nn.Parameter(torch.empty(E, H, dtype=torch.float32), requires_grad=False)
        self.router_bias = nn.Parameter(torch.zeros(E, dtype=torch.float32), requires_grad=False)
        lin = lambda i, o, n: ReplicatedLinear(i, o, bias=False, quant_config=quant_config, prefix=f"{prefix}.{n}")
        self.routed_down = lin(H, lat, "routed_down")
        self.routed_up = lin(lat, H, "routed_up")
        self.experts = Hy4GGUFExperts(E, I, lat, config.swiglu_limit)
        self.shared = Hy4MLP(H, I * config.n_shared_experts, quant_config, f"{prefix}.shared")
        self.combine_norm = RMSNorm(H, eps=config.rms_norm_eps)

    def forward(self, h):
        w, idx = B.route(h, self.router, self.router_bias, self.top_k, self.norm, self.scale)
        x = self.routed_down(h)[0]
        y = self.routed_up(self.experts(x, w.to(x.dtype), idx.to(torch.int32)))[0]
        return self.combine_norm(y + self.shared(h))


class Hy4Engram(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config, table: HostEngramTable, base_row: int, prefix):
        super().__init__()
        hc, H = config.hc_mult, config.hidden_size
        self.n, self.heads = config.engram_max_ngram_size, config.engram_n_head_per_ngram
        self.pad = config.engram_pad_id
        lid = str(layer_idx + 1)
        sizes = [s for order in config.engram_layer_head_vocab_sizes[lid] for s in order]
        d = config.engram_n_embed_per_ngram // self.heads
        self.table, self.dim = table, d
        self.register_buffer("vocab_sizes", torch.tensor(sizes), persistent=False)
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]) + base_row
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.long), persistent=False)
        k = config.engram_layer_ids.index(layer_idx + 1)
        mult = config.engram_multipliers[k * self.n:(k + 1) * self.n]
        self.register_buffer("multipliers", torch.tensor(mult, dtype=torch.long), persistent=False)
        engram_size = (self.n - 1) * config.engram_n_embed_per_ngram
        self.kv = ReplicatedLinear(engram_size, (hc + 1) * H, bias=False, quant_config=quant_config, prefix=f"{prefix}.kv")
        f32 = torch.float32
        self.kv_bias = nn.Parameter(torch.empty(hc + 1, H, dtype=f32), requires_grad=False)
        self.gate = nn.Parameter(torch.empty(hc, H, dtype=f32), requires_grad=False)
        self.lookup_norm = nn.Parameter(torch.empty(len(sizes), d, dtype=f32), requires_grad=False)
        self.hc_scale = nn.Parameter(torch.empty(hc, H, dtype=f32), requires_grad=False)
        self.pre_gamma = nn.Parameter(torch.empty(H, dtype=f32), requires_grad=False)
        tm = getattr(config, "hy4_engram_token_map", None)
        self.register_buffer("token_map", torch.tensor(tm if tm else [], dtype=torch.int32), persistent=False)

    def forward(self, streams, ngram):
        hid = B.hash_ngrams(ngram, self.token_map, self.multipliers, self.vocab_sizes, self.heads, self.pad)
        rows = self.table.gather((hid + self.offsets).reshape(-1), self.dim).to(streams.device)
        rows = rows.view(hid.shape[0], hid.shape[1], self.dim)
        kv = lambda e: self.kv(e.to(streams.dtype))[0]   # GGUF linear (any backend)
        return B.engram_residual(streams, rows, self.lookup_norm, kv, self.kv_bias, self.gate,
                                 self.hc_scale, self.pre_gamma, ngram[:, 0] != self.pad)


class Hy4HC(nn.Module):
    def __init__(self, hc: int, hidden: int):
        super().__init__()
        f32 = torch.float32
        self.fn = nn.Parameter(torch.empty(2 * hc, hc * hidden, dtype=f32), requires_grad=False)
        self.base = nn.Parameter(torch.empty(2 * hc, dtype=f32), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(2, dtype=f32), requires_grad=False)


class Hy4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, quant_config, cache_config, topk_buf, table, engram_base, prefix):
        super().__init__()
        self.cfg, self.idx = config, layer_idx
        H, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = Hy4Attention(config, layer_idx, quant_config, cache_config, topk_buf, f"{prefix}.self_attn")
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Hy4MoE(config, quant_config, f"{prefix}.mlp")
        else:
            self.mlp = Hy4MLP(H, config.intermediate_size, quant_config, f"{prefix}.mlp", post_norm_eps=eps)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(H, eps=eps), RMSNorm(H, eps=eps)
        self.attn_hc, self.ffn_hc = Hy4HC(config.hc_mult, H), Hy4HC(config.hc_mult, H)
        self.engram = None
        if config.engram_layer_ids and layer_idx + 1 in config.engram_layer_ids:
            self.engram = Hy4Engram(config, layer_idx, quant_config, table, engram_base, f"{prefix}.engram")

    def forward(self, positions, streams, ngram):
        c = self.cfg
        if self.engram is not None and ngram is not None:
            streams = streams + self.engram(streams, ngram)
        post, h = B.hc_pre(streams, self.attn_hc.fn, self.attn_hc.base, self.attn_hc.scale, c.rms_norm_eps,
                           c.hc_eps, c.hc_magnitude)
        h = self.self_attn(positions, self.input_layernorm(h))
        streams = B.hc_post(post, h, streams)
        post, h = B.hc_pre(streams, self.ffn_hc.fn, self.ffn_hc.base, self.ffn_hc.scale, c.rms_norm_eps,
                           c.hc_eps, c.hc_magnitude)
        h = self.mlp(self.post_attention_layernorm(h))
        return B.hc_post(post, h, streams)


class HYV4FlashModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config, cache_config = vllm_config.quant_config, vllm_config.cache_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config,
                                                   prefix=f"{prefix}.embed_tokens")
        width = int(config.compressed_indexer_candidate_topk)
        self.topk_buf = torch.full((vllm_config.scheduler_config.max_num_batched_tokens, width), -1,
                                   dtype=torch.int32)
        self.table = HostEngramTable.from_config(config)
        bases, base = {}, 0
        for lid in sorted(config.engram_layer_ids or []):
            bases[lid - 1] = base
            base += sum(sum(o) for o in config.engram_layer_head_vocab_sizes[str(lid)])
        self.layers = nn.ModuleList([
            Hy4DecoderLayer(config, i, quant_config, cache_config, self.topk_buf, self.table, bases.get(i, 0),
                            f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)])
        # named under layer 0: vLLM orders KV caches by the one integer in each layer name
        self.token_cache = (_PerTokenCache(1, torch.float32, f"{prefix}.layers.0.engram_token_cache", cache_config)
                            if config.engram_layer_ids else None)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        f32 = torch.float32
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, config.hc_mult * config.hidden_size, dtype=f32),
                                       requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=f32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=f32), requires_grad=False)

    def _ngrams(self, input_ids, positions):
        """[T, n] tokens (current first) for the Engram layers, using the paged token-id cache for
        the history of every request (chunked prefill, prefix caching, decode)."""
        n, pad = self.config.engram_max_ngram_size, self.config.engram_pad_id
        tc = self.token_cache
        md = tc.metadata()
        T = input_ids.shape[0]
        cols = [input_ids]
        if md is None:
            for k in range(1, n):
                prev = torch.full_like(input_ids, pad)
                prev[k:] = input_ids[:-k]
                cols.append(prev)
            return torch.stack(cols, dim=-1)
        tc.write(input_ids.float().unsqueeze(-1), md)
        A_ = min(T, md.req_id_per_token.shape[0])          # padded tail (if any) gets pad history
        req = md.req_id_per_token[:A_].long()
        for k in range(1, n):
            p = positions[:A_] - k
            ok = p >= 0
            prev = tc.gather(req, p.clamp_min(0), md)[..., 0].round().long()
            col = torch.full_like(input_ids, pad)
            col[:A_] = torch.where(ok, prev, torch.full_like(prev, pad))
            cols.append(col)
        return torch.stack(cols, dim=-1)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        c = self.config
        if self.topk_buf.device != positions.device:
            self.topk_buf.data = self.topk_buf.to(positions.device)
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        ngram = self._ngrams(input_ids, positions) if self.token_cache is not None else None
        streams = x.unsqueeze(1).expand(-1, c.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(positions, streams, ngram)
        h = B.hc_head(streams, self.hc_head_fn, self.hc_head_base, self.hc_head_scale, c.rms_norm_eps, c.hc_eps)
        return self.norm(h)


class HYV4FlashForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = HYV4FlashModel(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=vllm_config.quant_config,
                                      prefix="lm_head")
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm_gguf_plugin.hy4.loading import load_hy4_weights
        return load_hy4_weights(self, weights)
