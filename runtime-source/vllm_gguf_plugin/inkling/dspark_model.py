# SPDX-License-Identifier: Apache-2.0
"""Inkling's block drafter, with one projection of the target taps per layer."""

import torch
from torch import nn
from torch.nn import functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix

from .dspark import ContextMix, InklingMarkovHead
from .dspark_attention import local_query


class DraftLayer(DFlashQwen3DecoderLayer):
    def __init__(self, vllm_config, config, index, prefix, block_length):
        super().__init__(vllm_config, config=config, layer_idx=index, cache_config=vllm_config.cache_config,
                         quant_config=None, prefix=prefix)
        h = config.hidden_size
        self.context_mix = ContextMix(len(config.dflash_config['layer_ids']), h, config.rms_norm_eps)
        self.context_kv = nn.Linear(h, 2 * config.num_key_value_heads * config.head_dim, bias=False)
        self.context_k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.window = config.context_window if config.local_layers[index] else None
        self.block_length = block_length

    def inject(self, features, positions, slots):
        a = self.self_attn
        context = self.context_mix(features)
        k, v = self.context_kv(context).reshape(-1, 2, a.num_kv_heads, a.head_dim).unbind(1)
        k = self.context_k_norm(k.contiguous()).reshape(-1, a.kv_size)
        # The rotary layer rotates Q and K in place on some backends.
        k, _ = a.rotary_emb(positions, k, None)
        if slots is not None:
            cache = a.attn
            cache.impl.do_kv_cache_update(cache, k.reshape(-1, a.num_kv_heads, a.head_dim), v.contiguous(), cache.kv_cache, slots)

    def block_forward(self, x, positions):
        hidden = self.input_layernorm(x)
        if self.window is None:
            attended = self.self_attn(positions, hidden)
        else:
            attended = local_query(self.self_attn, hidden, positions, self.window, self.block_length)
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x))


class DraftBody(nn.Module):
    def __init__(self, vllm_config, config, prefix):
        super().__init__()
        self.embed_tokens = None  # shared with the loaded target
        self.embed_norm = nn.Parameter(torch.ones(config.hidden_size))
        self.block_length = vllm_config.num_speculative_tokens + 1
        self.masks = nn.Parameter(torch.empty(config.block_size - 1, config.hidden_size))
        self.eps = config.rms_norm_eps
        start = vllm_config.model_config.get_total_num_hidden_layers()
        self.layers = nn.ModuleList(DraftLayer(vllm_config, config, i, f'{prefix}.layers.{start+i}', self.block_length)
                                    for i in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=self.eps)

    def embed_input_ids(self, ids):
        # Mask slots may use a real vocabulary ID as the scheduler placeholder.
        # Their position in the block determines which trained embedding to use.
        slot = torch.arange(ids.numel(), device=ids.device) % self.block_length
        anchors = self.embed_tokens(ids[::self.block_length])
        anchors = F.rms_norm(anchors.float(), (anchors.shape[-1],), self.embed_norm.float(), self.eps).to(anchors.dtype)
        return torch.where((slot == 0)[:, None], anchors.repeat_interleave(self.block_length, 0)[:ids.numel()],
                           self.masks[(slot - 1).clamp_min(0)])


class InklingDSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError('The Inkling loop target and its native drafter require tensor parallel size one')
        self.config = c = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DraftBody(vllm_config, c, maybe_prefix(prefix, 'model'))
        self.lm_head = None  # shared after loading
        self.logits_processor = LogitsProcessor(c.vocab_size)
        self.markov_context = ContextMix(len(c.dflash_config['layer_ids']), c.hidden_size, c.rms_norm_eps)
        self.markov = InklingMarkovHead(c.hidden_size, c.markov_rank, c.unpadded_vocab_size if c.folded_markov else c.vocab_size,
                                      folded=c.folded_markov, softcap=c.final_logit_softcapping)
        self.draft_id_to_target_id = None

    def embed_input_ids(self, input_ids, *args, **kwargs):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
        h = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.layers:
            h = layer.block_forward(h, positions)
        return self.model.norm(h)

    def get_draft_kv_cache_layer_names(self):
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self):
        return [False] * len(self.model.layers)

    def combine_hidden_states(self, features):
        return features

    def precompute_and_store_context_kv(self, features, positions, context_slot_mapping=None):
        features = features.reshape(features.shape[0], -1, self.config.hidden_size)
        for i, layer in enumerate(self.model.layers):
            slots = context_slot_mapping[i] if isinstance(context_slot_mapping, (list, tuple)) else context_slot_mapping
            layer.inject(features, positions, slots)

    def compute_logits(self, h):
        return self.logits_processor(self.lm_head, h)

    def map_draft_to_target(self, ids):
        return ids

    def load_weights(self, weights):
        from .dspark_weights import load_parameters
        return load_parameters(self, dict(weights))
