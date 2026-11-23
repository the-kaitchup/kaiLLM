# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (`inkling2_loop`, text model) config for kaiLLM, built from llama.cpp master GGUF metadata
(vllm_gguf_plugin/inkling_metadata.py). Field names follow the GGUF keys; the official checkpoint's config.json
(`Inkling2LoopForConditionalGeneration`) is not read by this runtime."""

from transformers.configuration_utils import PretrainedConfig


class Inkling2LoopConfig(PretrainedConfig):
    model_type = "inkling2_loop"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=201024, vocab_size_unpadded=200058, hidden_size=2304, num_hidden_layers=24,
                 num_attention_heads=24, num_key_value_heads=8, head_dim=128, num_experts=128,
                 num_experts_per_tok=8, moe_intermediate_size=1152, intermediate_size=1152, sliding_window=1024,
                 sliding_window_pattern=None, rms_norm_eps=1e-6, max_position_embeddings=1048576,
                 relative_length=16, relative_extent=1024, relative_clip=8.0, qk_clip=22.627416610717773,
                 log_scale_floor=8192, log_scale_alpha=0.1, conv_L_cache=4, attn_res_block_size=6,
                 situ_beta=4.0, situ_linear_beta=25.0, expert_weights_scale=8.0, final_logit_softcapping=30.0,
                 loop_start=6, loop_end=18, loop_count=3, loop_gate_rank=64, tie_word_embeddings=False,
                 **kwargs):
        self.vocab_size = vocab_size
        self.vocab_size_unpadded = vocab_size_unpadded
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.sliding_window_pattern = list(sliding_window_pattern or [i % 6 != 5 for i in range(num_hidden_layers)])
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.relative_length = relative_length
        self.relative_extent = relative_extent
        self.relative_clip = relative_clip
        self.qk_clip = qk_clip
        self.log_scale_floor = log_scale_floor
        self.log_scale_alpha = log_scale_alpha
        self.conv_L_cache = conv_L_cache
        self.conv_bias = False
        self.attn_res_block_size = attn_res_block_size
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.expert_weights_scale = expert_weights_scale
        self.final_logit_softcapping = final_logit_softcapping
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_count = loop_count
        self.loop_gate_rank = loop_gate_rank
        # every layer has attention AND short convs; a non-attention spelling makes vLLM plan it as a hybrid
        # (ModelConfig.is_hybrid: attention KV caches + conv states)
        self.layer_types = ["hybrid"] * num_hidden_layers
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_slots(self) -> int:
        return self.num_hidden_layers + (self.loop_count - 1) * (self.loop_end - self.loop_start)

    def slot(self, layer: int, pass_index: int) -> int:
        if pass_index == 0 or not (self.loop_start <= layer < self.loop_end):
            return layer
        return self.num_hidden_layers + (pass_index - 1) * (self.loop_end - self.loop_start) + (layer - self.loop_start)

    @property
    def conv_channels(self) -> int:
        return 2 * self.head_dim * self.num_key_value_heads + 2 * self.hidden_size
