"""GPU unit test: Hy4 latent-MoE experts on kaiLLM's GGML MoE kernels vs a dequantized reference.

    source tests/gpu/hy4_env.sh <python>
    $PY tests/gpu/test_hy4_moe_kernel.py <model-00001-of-0000N.gguf> [layer ...]

Real expert tensors of the given layers (default 1 20 41), random latent inputs, 16 distinct experts
per token: the MMQ path (> 64 tokens), the MMVQ path (64-token chunks) and float32 matmuls on
ggml_dequantize'd experts. Pass: relative max error of both paths <= 3e-2.
"""
import sys

import numpy as np
import torch


def main():
    import gguf
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.hy4.blocks import clamped_swiglu
    from vllm_gguf_plugin.hy4.moe import hy4_gguf_moe

    path = sys.argv[1]
    layers = [int(x) for x in sys.argv[2:]] or [1, 20, 41]
    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    torch.manual_seed(0)
    worst = 0.0
    for L in layers:
        g, u, d = (t[f"blk.{L}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))
        D_in, I, E = (int(v) for v in g.shape)           # ggml dims: [in, out, experts]
        D_out = int(d.shape[1])
        w13 = torch.cat([torch.from_numpy(np.asarray(g.data)).view(E, I, -1),
                         torch.from_numpy(np.asarray(u.data)).view(E, I, -1)], dim=1).contiguous().cuda()
        w2 = torch.from_numpy(np.asarray(d.data)).view(E, D_out, -1).contiguous().cuda()
        t13, t2 = int(g.tensor_type), int(d.tensor_type)
        T, K = 192, 16
        x = (torch.randn(T, D_in, device="cuda") * 0.5).bfloat16()
        ids = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)]).to(torch.int32)
        wts = torch.rand(T, K, device="cuda").bfloat16()

        y_mmq = hy4_gguf_moe(x, w13, w2, wts, ids, t13, t2, 10.0).float()
        y_mmvq = torch.cat([hy4_gguf_moe(x[s:s + 64], w13, w2, wts[s:s + 64], ids[s:s + 64], t13, t2, 10.0)
                            for s in range(0, T, 64)]).float()
        ref = torch.zeros(T, D_out, device="cuda")
        for e in ids.unique().tolist():
            tok, slot = torch.where(ids == e)
            a = ops.ggml_dequantize(w13[e].contiguous(), t13, 2 * I, D_in, torch.float32)
            b = ops.ggml_dequantize(w2[e].contiguous(), t2, D_out, I, torch.float32)
            h = x[tok].float() @ a.t()
            y = clamped_swiglu(h[:, :I], h[:, I:], 10.0) @ b.t()
            ref.index_add_(0, tok, y * wts[tok, slot, None].float())
        scale = ref.abs().max().clamp_min(1e-6)
        e1 = ((y_mmq - ref).abs().max() / scale).item()
        e2 = ((y_mmvq - ref).abs().max() / scale).item()
        e3 = ((y_mmq - y_mmvq).abs().max() / scale).item()
        worst = max(worst, e1, e2)
        print(f"layer {L}: E={E} I={I} D_in={D_in} D_out={D_out} types {gguf.GGMLQuantizationType(t13).name}/"
              f"{gguf.GGMLQuantizationType(t2).name}  MMQ vs ref {e1:.2e}  MMVQ vs ref {e2:.2e}  MMQ vs MMVQ {e3:.2e}")
    ok = worst <= 3e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
