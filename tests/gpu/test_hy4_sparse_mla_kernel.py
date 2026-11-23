"""GPU unit test: vLLM's Triton sparse-MLA kernel + kaiLLM's sink correction vs the PyTorch reference.

    KAILLM_HY4_ROOT="$PWD" PYTHONPATH="$PWD/tests/gpu/site" python tests/gpu/test_hy4_sparse_mla_kernel.py

Real Hy4 dims (latent 512 + rope 64, d_v 512), 32 and 64 heads, window (512) and indexer (2048)
index lists with -1 padding, learnable sinks. Pass: relative max error <= 2e-2 (BF16 inputs) and
the sink-free output matches too (checks that the kernel's LSE is natural-log).
"""
import sys

import torch

from vllm_gguf_plugin.hy4.attention import reference_sparse_mla, triton_sparse_mla


def case(T, H, W, R, sinks_on, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    D, dv, rope = 576, 512, 64
    q = (torch.randn(T, H, D, device="cuda", generator=g) * 0.5).bfloat16()
    kv = (torch.randn(R, D, device="cuda", generator=g) * 0.5).bfloat16()
    n_valid = torch.randint(1, W + 1, (T,), device="cuda", generator=g)
    rows = torch.randint(0, R, (T, W), device="cuda", generator=g, dtype=torch.int32)
    rows = torch.where(torch.arange(W, device="cuda")[None] < n_valid[:, None], rows, -1)
    sinks = (torch.randn(H, device="cuda", generator=g) * 2).float() if sinks_on else None
    scale = 256 ** -0.5
    ref = reference_sparse_mla(q, kv, rows, scale, dv, sinks).float()
    got = triton_sparse_mla(q, kv, rows, scale, dv, rope, sinks).float()
    err = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    return err


def main():
    worst = 0.0
    for T, H, W in ((7, 32, 2048), (37, 64, 512), (64, 32, 512), (5, 64, 2048)):
        for sinks_on in (False, True):
            e = case(T, H, W, 6000, sinks_on, seed=T * 131 + H + W)
            worst = max(worst, e)
            print(f"T={T:3d} H={H} W={W:4d} sinks={sinks_on!s:5s}  relative max error {e:.2e}")
    ok = worst <= 2e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
