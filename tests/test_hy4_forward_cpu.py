"""Whole Hy4 forward from kaiLLM's pure blocks vs transformers PR 50903 (toy, float32, CPU).

Composes hy4/blocks.py (hyper-connections, Engram, router, latent MoE, MLP) and hy4/attention.py
(sparse MLA via index lists, indexers) in the decoder order of the vLLM model and compares logits.

    python tests/test_hy4_forward_cpu.py "$TOY_MODEL"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HY4 = ROOT / "runtime-source/vllm_gguf_plugin/hy4"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HY4 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A, B = _load("attention"), _load("blocks")
sys.path.insert(0, str(ROOT / "tests"))
from test_hy4_attention_cpu import ours as attention_ours  # noqa: E402


def forward(lm, head, cfg, ids):
    T = ids.shape[0]
    pos = torch.arange(T)
    eps, hc = cfg.rms_norm_eps, cfg.hc_mult
    x = lm.embed_tokens.weight[ids]
    streams = x.unsqueeze(1).expand(T, hc, -1).contiguous()
    pad = cfg.engram_pad_id
    for L, layer in enumerate(lm.layers):
        el = layer.engram_layer
        if el is not None:
            ng = B.ngram_tokens(ids, None, cfg.engram_max_ngram_size, pad)
            hid = B.hash_ngrams(ng, el.lookup_table, el.hash_multipliers, el.head_vocab_sizes,
                                el.n_head_per_ngram, pad)
            table = el.multi_head_embedding.embedding.weight
            rows = table[hid + el.multi_head_embedding.offsets]
            if el.fp8_embedding:
                rows = A_fp8(rows)
            kv_w = torch.cat([k.weight for k in el.key_projs] + [el.value_proj.weight])
            kv_b = torch.stack([k.bias for k in el.key_projs] + [el.value_proj.bias])
            gate_w = torch.stack([nk.weight * nq.weight for nk, nq in zip(el.norm_key, el.norm_query)])
            norms = torch.stack([n.weight for n in el.lookup_per_head_norms])
            scales = torch.stack(list(el.unified_per_hc_scales))
            streams = streams + B.engram_residual(streams, rows, norms, kv_w, kv_b, gate_w, scales,
                                                  el.unified_pre_gamma, ids != pad)
        a = layer.attn_hc
        post, h = B.hc_pre(streams, a.fn, a.base, a.scale, eps, cfg.hc_eps, cfg.hc_magnitude)
        h = B.rms_norm(h, layer.input_layernorm.weight, eps)
        h = attention_ours(layer.self_attn, cfg, h, pos)
        streams = B.hc_post(post, h, streams)
        f = layer.ffn_hc
        post, h = B.hc_pre(streams, f.fn, f.base, f.scale, eps, cfg.hc_eps, cfg.hc_magnitude)
        h = B.rms_norm(h, layer.post_attention_layernorm.weight, eps)
        m = layer.mlp
        if cfg.mlp_layer_types[L] == "sparse":
            w, idx = B.route(h, m.gate.weight, m.gate.e_score_correction_bias, cfg.num_experts_per_tok,
                             cfg.norm_topk_prob, cfg.routed_scaling_factor)
            xl = h @ m.routed_expert_down_proj.weight.t()
            I = m.experts.gate_up_proj.shape[1] // 2
            r = B.dense_experts(xl, m.experts.gate_up_proj[:, :I], m.experts.gate_up_proj[:, I:],
                                m.experts.down_proj, w, idx, cfg.swiglu_limit)
            out = r @ m.routed_expert_up_proj.weight.t()
            se = m.shared_experts
            out = out + B.mlp(h, se.gate_proj.weight, se.up_proj.weight, se.down_proj.weight)
            h = B.rms_norm(out, m.combine_output_norm.weight, eps)
        else:
            h = B.rms_norm(B.mlp(h, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight),
                           m.post_fc2_norm.weight, eps)
        streams = B.hc_post(post, h, streams)
        hh = lm.hc_head
    h = B.hc_head(streams, hh.hc_fn, hh.hc_base, hh.hc_scale, eps, cfg.hc_eps)
    h = B.rms_norm(h, lm.norm.weight, eps)
    return h.float() @ head.weight.float().t()


def A_fp8(rows):
    info = torch.finfo(torch.float8_e4m3fn)
    s = rows.abs().amax(-1, keepdim=True).clamp(min=1e-12) / info.max
    return (rows / s).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(rows.dtype) * s


def main():
    toy = Path(sys.argv[1])
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M
    model = M.HYV4FlashForConditionalGeneration.from_pretrained(toy, dtype=torch.float32).eval()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hy4_tie_patch import patch_indexer_ties
    patch_indexer_ties(model)   # kaiLLM's rule for exact score ties on both sides
    lm, cfg = model.model.language_model, model.model.language_model.config
    torch.manual_seed(0)
    for T in (24, 80):
        ids = torch.randint(10, 2000, (T,))
        with torch.no_grad():
            ref = model(input_ids=ids[None]).logits[0]
            got = forward(lm, model.lm_head, cfg, ids)
        la, lb = ref.log_softmax(-1), got.log_softmax(-1)
        kl = float((la.exp() * (la - lb)).sum(-1).mean())
        print(f"T={T}: logits max |diff| {float((got - ref).abs().max()):.3g}, KL {kl:.3g}, "
              f"top-1 agree {float((got.argmax(-1) == ref.argmax(-1)).float().mean()):.3f}")


if __name__ == "__main__":
    main()
