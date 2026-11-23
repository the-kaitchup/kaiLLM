"""hy_v4_flash GGUF metadata -> Tencent config fields and an identical tokenizer.

    python tests/test_hy4_metadata.py <toy.gguf> <toy_tencent_dir> <tencent tokenizer.json> [texts.jsonl ...]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime-source"))

from vllm_gguf_plugin.hy4_metadata import prepare_hy4_metadata  # noqa: E402

FIELDS = ["hidden_size", "intermediate_size", "moe_intermediate_size", "routed_expert_hidden_size",
          "num_hidden_layers", "num_attention_heads", "swa_num_attention_heads", "q_lora_rank", "kv_lora_rank",
          "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "n_routed_experts", "num_experts_per_tok",
          "n_shared_experts", "routed_scaling_factor", "norm_topk_prob", "sliding_window",
          "compressed_indexer_num_attention_heads", "compressed_indexer_head_dim",
          "compressed_indexer_candidate_topk", "compressed_indexer_compress_ratio", "compressed_indexer_group_topk",
          "hc_mult", "hc_magnitude", "engram_layer_ids", "engram_max_ngram_size", "engram_n_head_per_ngram",
          "engram_n_embed_per_ngram", "engram_layer_head_vocab_sizes", "engram_pad_id", "swiglu_limit",
          "rms_norm_eps", "vocab_size"]


def main():
    gguf_path, tencent_dir, tok_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    texts = [json.loads(line)["text"] for f in sys.argv[4:] for line in open(f)]
    with tempfile.TemporaryDirectory() as td:
        out = prepare_hy4_metadata(gguf_path, cache_dir=td)
        cfg = json.loads((out / "config.json").read_text())
        ref = json.loads((tencent_dir / "config.json").read_text())
        ref = ref.get("text_config", ref)
        bad = [(k, cfg.get(k), ref.get(k)) for k in FIELDS if k in ref and cfg.get(k) != ref.get(k)
               and not (isinstance(ref.get(k), float) and abs(cfg.get(k) - ref.get(k)) < 1e-6 * abs(ref.get(k)))]
        lt = ["indexed_attention" if x != "sliding_attention" else x for x in ref["layer_types"]]
        if cfg["layer_types"] != lt:
            bad.append(("layer_types", cfg["layer_types"][:6], lt[:6]))
        for name, rp in ref["rope_parameters"].items():
            key = "sliding_attention" if name == "sliding_attention" else "indexed_attention"
            if abs(cfg["rope_parameters"][key]["rope_theta"] - rp["rope_theta"]) > 1e-3:
                bad.append((f"rope {name}", cfg["rope_parameters"][key], rp))
        print(f"config: {len(FIELDS)} fields + layer types + rope checked, mismatches: {bad}")
        from tokenizers import Tokenizer
        ours, theirs = Tokenizer.from_file(str(out / "tokenizer.json")), Tokenizer.from_file(str(tok_path))
        tc = json.loads((out / "tokenizer_config.json").read_text())
        sample = texts + [tc.get("chat_template", "")[:4000], "Hello world! 12345 日本語のテキスト ひらがな カタカナ 中文测试 हिन्दी"]
        diff = sum(ours.encode(t, add_special_tokens=False).ids != theirs.encode(t, add_special_tokens=False).ids
                   for t in sample)
        n_tok = sum(len(theirs.encode(t, add_special_tokens=False).ids) for t in sample)
        dec = sum(ours.decode(ours.encode(t, add_special_tokens=False).ids) != t for t in sample[:20])
        print(f"tokenizer: {len(sample)} texts ({n_tok} tokens), id mismatches {diff}, decode mismatches {dec}/20; "
              f"vocab {ours.get_vocab_size()} vs {theirs.get_vocab_size()}; files {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
