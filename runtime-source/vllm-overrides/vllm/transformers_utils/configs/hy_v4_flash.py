# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash text config for transformers releases without `hy_v4_flash` (kaiLLM backport).

Field names follow Tencent's released config.json (and kaiLLM's GGUF metadata conversion,
vllm_gguf_plugin/hy4_metadata.py). Layer types are normalized to `indexed_attention` (the
name transformers validates, used by the transformers PR) / `sliding_attention`; the vision tower is not part of this text-only config.
"""

from typing import Any

from transformers.configuration_utils import PreTrainedConfig

SPARSE = "indexed_attention"
_SPARSE = {"sparse", "sparse_attention", "indexed_attention", "deepseek_sparse_attention"}

DEFAULTS: dict[str, Any] = dict(
    vocab_size=120832, hidden_size=3072, intermediate_size=13824, moe_intermediate_size=1536,
    routed_expert_hidden_size=1536, num_hidden_layers=42, num_attention_heads=32, num_key_value_heads=32,
    swa_num_attention_heads=64, q_lora_rank=1024, kv_lora_rank=512, qk_nope_head_dim=192, qk_rope_head_dim=64,
    qk_head_dim=256, v_head_dim=256, swa_qk_nope_head_dim=192, swa_qk_rope_head_dim=64, swa_qk_head_dim=256,
    swa_v_head_dim=256, head_dim=64, n_routed_experts=512, num_experts_per_tok=16, n_shared_experts=1,
    routed_scaling_factor=3.998, norm_topk_prob=True, moe_router_use_sigmoid=True, n_group=1, topk_group=1,
    first_k_dense_replace=1, mlp_layer_types=None, layer_types=None, swa_pattern=None, indexer_types=None,
    sliding_window=512, compressed_indexer_num_attention_heads=16, compressed_indexer_head_dim=128,
    compressed_indexer_candidate_topk=2048, compressed_indexer_compress_ratio=4, compressed_indexer_group_topk=512,
    compressed_indexer_select_tail=True, rope_parameters=None, rms_norm_eps=1e-5,
    max_position_embeddings=1048576, swiglu_limit=10.0, hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
    enable_ihc=True, gated_mla=True, learnable_sink=True, sandwich_norm=True, hidden_act="silu",
    engram_layer_ids=None, engram_max_ngram_size=3, engram_n_head_per_ngram=8, engram_n_embed_per_ngram=768,
    engram_layer_head_vocab_sizes=None, engram_pad_id=120002, engram_multipliers=None, engram_seed=0,
    engram_fp8_embedding=False, num_nextn_predict_layers=0, attention_bias=False, initializer_range=0.006,
)


class HYV4FlashConfig(PreTrainedConfig):
    model_type = "hy_v4_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, tie_word_embeddings: bool = False, bos_token_id=120000, eos_token_id=120025,
                 pad_token_id=120002, **kwargs):
        text = kwargs.pop("text_config", None) or {}
        values = {**DEFAULTS, **text, **{k: v for k, v in kwargs.items() if k in DEFAULTS}}
        rest = {k: v for k, v in kwargs.items() if k not in DEFAULTS}
        n = values["num_hidden_layers"]
        if values["layer_types"] is None:
            pattern = values["swa_pattern"] or [0] * n
            values["layer_types"] = ["sliding_attention" if p else SPARSE for p in pattern]
        values["layer_types"] = [SPARSE if t in _SPARSE else t for t in values["layer_types"]]
        if values["mlp_layer_types"] is None:
            k = values["first_k_dense_replace"]
            values["mlp_layer_types"] = ["dense"] * k + ["sparse"] * (n - k)
        if values["rope_parameters"]:
            values["rope_parameters"] = {(SPARSE if k in _SPARSE else k): dict(v)
                                         for k, v in values["rope_parameters"].items()}
        for k, v in values.items():
            setattr(self, k, v)
        super().__init__(tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, pad_token_id=pad_token_id, **rest)

    def rope_theta_for(self, layer_idx: int) -> float:
        kind = self.layer_types[layer_idx]
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", 10_000.0))

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    def heads_for(self, layer_idx: int) -> int:
        return self.swa_num_attention_heads if self.is_sliding(layer_idx) else self.num_attention_heads


__all__ = ["HYV4FlashConfig"]
