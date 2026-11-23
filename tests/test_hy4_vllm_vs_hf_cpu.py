"""Exactness of the whole kaiLLM vLLM path (with the compressed indexer selecting) vs transformers PR 50903.

Step 1 runs tests/test_hy4_vllm_cpu.py in float32 on the toy's F32 GGUF with the
Engram FP8 round trip baked into the table (transformers applies it at read; toy6-f32-rt.gguf) and saves
every scenario's logits; step 2 recomputes them with
transformers on the toy checkpoint and compares. Both sides compute in float32, so the only
differences are summation order (no BF16 / FP8 rounding to flip near-tie group picks). Exact score
ties (frequent on the toy: 4 indexer heads, ReLU zeros) use kaiLLM's rule on both sides.

    KAILLM_HY4_ROOT="$PWD" PYTHONPATH="$PWD/tests/gpu/site" CUDA_VISIBLE_DEVICES= \
      python tests/test_hy4_vllm_cpu.py "$TOY_GGUF" \
      --dump "$DUMP_LOGITS" --tokens "$TEST_TOKENS" \
      --save /tmp/hy4-vllm-cpu.npz
    python tests/test_hy4_vllm_vs_hf_cpu.py "$TOY_MODEL" /tmp/hy4-vllm-cpu.npz
"""

import sys
from pathlib import Path

import numpy as np
import torch


def main():
    toy, saved = Path(sys.argv[1]), np.load(sys.argv[2])
    # float32 on both sides: ~5e-7 KL per position is summation-order noise (llama.cpp sits at the same
    # distance); a rare near-tie group pick at that noise level adds an isolated spike, so gate on the mean
    kl_max = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M
    model = M.HYV4FlashForConditionalGeneration.from_pretrained(toy, dtype=torch.float32).eval()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hy4_tie_patch import patch_indexer_ties
    patch_indexer_ties(model)   # kaiLLM's tie rule on the transformers side too
    refs = {}
    with torch.no_grad():
        for k in ("seq_a", "seq_b"):
            ids = torch.from_numpy(saved[k]).long()
            refs[k] = model(input_ids=ids[None]).logits[0].double()
    ok = True
    for name in ("single", "chunked", "decode", "batch_a", "batch_b", "single_b"):
        got = torch.from_numpy(saved[name]).double()
        ref = refs["seq_b" if name.endswith("_b") else "seq_a"]
        lp, lq = ref.log_softmax(-1), got.log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        top1 = (got.argmax(-1) == ref.argmax(-1)).double().mean().item()
        worst = int(kl.argmax())
        bad = (kl > 1e-8).nonzero().flatten().tolist()
        if bad:
            print(f"   first positions with KL > 1e-8: {bad[:12]}; KL there: {[f'{kl[i].item():.1e}' for i in bad[:12]]}")
        print(f"{name:9s} tokens {got.shape[0]:4d}  max|dlogit| {(got - ref).abs().max().item():.2e}  "
              f"mean KL {kl.mean().item():.2e}  max KL {kl.max().item():.2e} (pos {worst})  top-1 {top1:.4f}")
        ok &= kl.mean().item() <= kl_max and top1 >= 0.99
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
