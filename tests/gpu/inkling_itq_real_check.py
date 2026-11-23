# SPDX-License-Identifier: Apache-2.0
"""ITQ expert kernels on a real itq-moe GGUF: for a few of its tensors, the kernel outputs (prefill path: BF16 rotated
rows; decode path: int8 per 32) vs the exact product of the float64 reconstruction (yaqa_gguf.reconstruct_itq_moe,
the oracle) with the same BF16 activations, next to what a BF16-decoded twin computes (the reconstruction rounded to
BF16, exact accumulation): mean / max relative error of each.  Tells whether the ITQ path loses more precision than
a decoded twin would.  GPU only, < 2 GB.

    $PY tests/gpu/inkling_itq_real_check.py FILE.gguf [--tensors 4] [--experts 8] [--tokens 16]
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensors", type=int, default=4, help="gate_up-type and down tensors each, spread over layers")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=16)
    a = ap.parse_args()
    import gguf
    from vllm_gguf_plugin import yaqa_gguf as yg
    from vllm_gguf_plugin.inkling.itq_moe import ITQExpertStage
    reader = gguf.GGUFReader(a.gguf)
    man = {n: d for n, d in yg.read_yaqa_manifest(reader).items() if d.get("codec") == "itq-moe"}
    names = sorted(man, key=lambda n: int(re.search(r"blk\.(\d+)", n).group(1)))
    ups = [n for n in names if re.search(r"_(gate_up|gate|up)_exps", n)]
    downs = [n for n in names if "_down_exps" in n]
    pick = [ups[int(i)] for i in np.linspace(0, len(ups) - 1, min(a.tensors, len(ups)))] if ups else []
    pick += [downs[int(i)] for i in np.linspace(0, len(downs) - 1, min(a.tensors, len(downs)))] if downs else []
    print(f"{len(man)} itq-moe tensors ({len(ups)} gate/up, {len(downs)} down); checking {len(pick)}", flush=True)
    gen = torch.Generator().manual_seed(0)
    rows_out = []
    for name in pick:
        desc = man[name]
        comps = yg.load_yaqa_components(reader, desc)
        stage = ITQExpertStage(desc, {k: v.clone() for k, v in comps.items()}, "cuda")
        E, m, n = desc["shape"]
        down = "_down_exps" in name
        shared = not down
        experts = torch.randperm(E, generator=gen)[: a.experts]
        T, k = a.tokens, 1
        ids = experts[torch.randint(0, len(experts), (T, 1), generator=gen)]
        # realistic scales: gate_up reads an RMS-normalised row (x ffn_norm), down a SITU output
        x = torch.randn(T, n, generator=gen) * (0.3 if down else 1.0)
        xb = x.to(torch.bfloat16)
        p = 1 if desc.get("components", {}).get("pass_output_scale") else None
        out = {}
        with torch.inference_mode():
            for path in ("prefill", "decode"):
                f = stage._run_prefill if path == "prefill" else stage._run_decode
                out[path] = f(xb.cuda(), ids.cuda(), shared, p).cpu().double()
        flat = ids.reshape(-1)
        exact = torch.empty(T, m, dtype=torch.float64)
        twin = torch.empty(T, m, dtype=torch.float64)
        dense = {}
        for r in range(T):
            e = int(flat[r])
            if e not in dense:
                dense[e] = yg.reconstruct_itq_moe(desc, comps, experts=[e], pass_index=p)[0].double()
            exact[r] = dense[e] @ xb[r].double()
            twin[r] = dense[e].to(torch.bfloat16).double() @ xb[r].double()
        ref = exact.abs().mean()
        row = {"tensor": name, "shape": [E, m, n]}
        for label, y in (("prefill kernel", out["prefill"]), ("decode kernel", out["decode"]), ("bf16 twin", twin)):
            err = (y - exact).abs()
            row[label] = (float(err.mean() / ref), float(err.max() / exact.abs().max()))
        rows_out.append(row)
        print(f"{name:34s} " + "  ".join(f"{lab}: mean {row[lab][0]:.2e} max {row[lab][1]:.2e}"
                                         for lab in ("prefill kernel", "decode kernel", "bf16 twin")), flush=True)
    for lab in ("prefill kernel", "decode kernel", "bf16 twin"):
        v = [r[lab][0] for r in rows_out]
        print(f"mean relative error, {lab}: {np.mean(v):.2e} (range {min(v):.2e} - {max(v):.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
