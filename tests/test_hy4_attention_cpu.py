"""CPU check of kaiLLM's Hy4 attention math against transformers PR 50903 (toy weights, float32).

Rebuilds what the vLLM path computes (latent cache rows [kv_c_normed, k_pe], absorbed queries,
window / dense / compressed-indexer index lists, sinks, gate, o_proj, post norm) and compares the
attention output of an indexed and a sliding layer with transformers, below and above
candidate_topk. Use a Python environment with the referenced Transformers implementation.

    python tests/test_hy4_attention_cpu.py "$TOY_MODEL"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hy4attn", ROOT / "runtime-source/vllm_gguf_plugin/hy4/attention.py")
A = importlib.util.module_from_spec(spec)
sys.modules["hy4attn"] = A
spec.loader.exec_module(A)


def ours(attn, cfg, hidden, positions):
    """kaiLLM-path attention for one sequence (batch 1): hidden [T, H] -> [T, H]."""
    T = hidden.shape[0]
    sliding = attn.attention_type == "sliding_attention"
    nope, rope, vd, kvr = attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim, attn.kv_lora_rank
    H = attn.num_heads
    eps = cfg.rms_norm_eps
    theta = cfg.rope_parameters["sliding_attention" if sliding else "indexed_attention"]["rope_theta"]
    q_c = A.rms(hidden @ attn.q_a_proj.weight.t(), attn.q_a_layernorm.weight, eps)
    q = (q_c @ attn.q_b_proj.weight.t()).view(T, H, nope + rope)
    kv = hidden @ attn.kv_a_proj_with_mqa.weight.t()
    kv_c, k_pe = kv.split([kvr, rope], dim=-1)
    kv_c = A.rms(kv_c, attn.kv_a_layernorm.weight, eps)
    q_rot = A.rope_interleaved(q[..., nope:], positions.view(T, 1).expand(T, H), theta)
    k_rot = A.rope_interleaved(k_pe, positions, theta)
    wkv = attn.kv_b_proj.weight.view(H, nope + vd, kvr)
    w_uk, w_uv = wkv[:, :nope], wkv[:, nope:]                          # [H, nope, kvr], [H, vd, kvr]
    q_abs = torch.cat([torch.einsum("thn,hnl->thl", q[..., :nope], w_uk), q_rot], dim=-1)
    rows_cache = torch.cat([kv_c, k_rot], dim=-1)                      # [T, kvr + rope]
    width = cfg.compressed_indexer_candidate_topk if not sliding else cfg.compressed_indexer_candidate_topk
    if sliding:
        idx = A.window_indices(positions, cfg.sliding_window, width)
    elif T <= cfg.compressed_indexer_candidate_topk:
        idx = A.dense_indices(positions, width)
    else:
        ci = attn.compressed_indexer
        w = A.IndexerWeights(ci.query_projection.weight, ci.head_weight_projection.weight, ci.token_key_norm.weight,
                             ci.key_value_projection.weight, ci.key_gate_projection.weight, ci.intra_group_ape,
                             ci.key_norm.weight, ci.num_heads, ci.head_dim, ci.rope_dim, ci.compress_ratio,
                             ci.group_topk, theta, eps)
        token_keys = A.rms(hidden @ ci.token_key_projection.weight.t(), ci.token_key_norm.weight, eps)
        idx = A.select_groups(hidden, q_c, positions, A.pool_groups(token_keys, w), w)
    lat = A.reference_sparse_mla(q_abs, rows_cache, idx, attn.scaling, kvr, attn.sinks)   # [T, H, kvr]
    o = torch.einsum("thl,hvl->thv", lat, w_uv)
    gate = (hidden @ attn.gate_proj.weight.t()).view(T, H, vd)
    o = (o * torch.sigmoid(gate)).reshape(T, H * vd)
    return A.rms(o @ attn.o_proj.weight.t(), attn.post_proj_norm.weight, eps)


def main():
    toy = Path(sys.argv[1])
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M
    model = M.HYV4FlashForConditionalGeneration.from_pretrained(toy, dtype=torch.float32).eval()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hy4_tie_patch import patch_indexer_ties
    patch_indexer_ties(model)   # kaiLLM's rule for exact score ties on both sides
    lm = model.model.language_model
    cfg = lm.config
    caps = {}
    for L, layer in enumerate(lm.layers):
        layer.self_attn.register_forward_hook(
            lambda m, args, kw, out, L=L: caps.__setitem__(L, (kw["hidden_states"].detach(), out[0].detach())),
            with_kwargs=True)
    torch.manual_seed(0)
    for T in (24, 80):
        caps.clear()
        ids = torch.randint(10, 1000, (1, T))
        with torch.no_grad():
            model(input_ids=ids)
        for L in (1, 2, 5):
            hidden, ref = caps[L]
            with torch.no_grad():
                got = ours(lm.layers[L].self_attn, cfg, hidden[0], torch.arange(T))
            err = float((got - ref[0]).norm() / ref[0].norm())
            kind = lm.layers[L].attention_type
            print(f"T={T:3d} layer {L} ({kind:17s}): relative error {err:.2e}")


if __name__ == "__main__":
    main()
