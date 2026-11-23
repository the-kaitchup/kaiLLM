# SPDX-License-Identifier: Apache-2.0
"""ITQ routed experts (itq-moe v1) on the GPU vs the FP32 oracle (yaqa_gguf.reconstruct_itq_moe), synthetic tensors at
Inkling's shapes: gate_up [E, 2304, 2304] (block256 in / out), down [E, 2304, 1152] (block128 in), experts mixing V4 / V2
formats and bit rates, per-expert wscale, post_output_scale, pass_output_scale, per-expert post_input_scale (down).

The kernels see the input after their own activation rounding, so the reference applies the oracle weights to that
input: decode (int8 per 32 after the rotation) x' = pis^-1 su H(Q8(H(su pis x))); prefill (BF16 rotated rows)
x' = pis^-1 su H(bf16(H(su pis x))).  Pass: max |kernel - reference| / max |reference| < 1e-2 (the kernels round
their output to BF16 before the scales) and mean relative error < 3e-3.  The single-launch decode GEMV must equal
the per-format launches bit for bit.

    $PY tests/gpu/test_inkling_itq_moe.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def gaussian_table(V, bits, gen):
    t = np.clip(np.round(gen.standard_normal((1 << bits, V)) * 32), -127, 127).astype(np.int8)
    return torch.from_numpy(t)


def make_tensor(E, m, n, formats, gen, pis_per_expert=False, passes=3):
    from vllm_gguf_plugin import yaqa_gguf as yg
    desc = {"codec": "itq-moe", "version": 1, "shape": [E, m, n], "L": 16, "hash": "mul",
            "layout": "tile16-rowmajor-steps-lsb", "V": [f[0] for f in formats], "bps": [f[1] for f in formats],
            "rotation_in": "fwht" if n & (n - 1) == 0 else f"block{n & -n}",
            "rotation_out": "fwht" if m & (m - 1) == 0 else f"block{m & -m}", "components": {}}
    total = yg.itq_moe_offsets(desc)[-1]
    comps = {"stream": torch.from_numpy(gen.integers(-128, 128, size=total, dtype=np.int8)),
             "su": torch.from_numpy((gen.integers(0, 2, size=n) * 2 - 1).astype(np.int8)),
             "sv": torch.from_numpy((gen.integers(0, 2, size=m) * 2 - 1).astype(np.int8)),
             "wscale": torch.from_numpy((gen.uniform(0.5, 2.0, size=E) / (32 * np.sqrt(n))).astype(np.float32)),
             "post_output_scale": torch.from_numpy(gen.uniform(0.8, 1.25, size=(E, m)).astype(np.float32)),
             "pass_output_scale": torch.from_numpy(gen.uniform(0.9, 1.1, size=(passes, E, m)).astype(np.float32)),
             "post_input_scale": torch.from_numpy(gen.uniform(0.8, 1.25, size=(E, n) if pis_per_expert else (n,))
                                                  .astype(np.float32))}
    for V in sorted({f[0] for f in formats}):
        comps[f"table_v{V}"] = gaussian_table(V, yg.ITQ_TABLE_BITS[V], gen)
        desc["components"][f"table_v{V}"] = f"itq.table.v{V}"
    for k in ("stream", "su", "sv", "wscale", "post_output_scale", "pass_output_scale", "post_input_scale"):
        desc["components"][k] = k
    yg.validate_yaqa_descriptor(desc)
    return desc, yg.check_itq_moe_components(desc, comps)


def hadamard(x, block):
    from vllm_gguf_plugin.yaqa_gguf import _block_hadamard
    return _block_hadamard(x, "fwht" if block == x.shape[-1] else f"block{block}")


def effective_input(x, su, pis, block, mode):
    """The input the kernels effectively see, back in the original basis (float64)."""
    v = hadamard(x.double() * pis * su.double(), block)
    if mode == "int8":
        g = v.reshape(v.shape[0], -1, 32)
        amax = g.abs().amax(-1, keepdim=True)
        scale = torch.where(amax > 0, amax.float().double() / 127.0, torch.ones_like(amax))
        q = torch.round(g / scale).clamp(-127, 127)
        v = (q * scale).reshape(v.shape)
    else:
        v = v.to(torch.bfloat16).double()
    return hadamard(v, block) * su.double() / pis


def main():
    from vllm_gguf_plugin import yaqa_gguf as yg
    from vllm_gguf_plugin.inkling.itq_moe import DECODE_ROWS, ITQExpertStage
    gen = np.random.default_rng(1)
    torch.manual_seed(1)
    E, k = 12, 4
    fm = [(4, 5), (2, 4), (4, 8), (2, 8), (4, 4), (2, 3), (4, 12), (2, 6), (4, 6), (2, 5), (4, 9), (2, 7)]
    ok = True
    for name, (m, n), pe in (("gate_up", (2304, 2304), False), ("down", (2304, 1152), True)):
        desc, comps = make_tensor(E, m, n, fm, gen, pis_per_expert=pe)
        stage = ITQExpertStage(desc, {kk: v.clone() for kk, v in comps.items()}, "cuda")
        shared = not pe
        dense = {}
        for T in (1, 3, 40, 160):
            ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
            rows = T if shared else T * k
            x = torch.randn(rows, n) * 0.7
            xb = x.to(torch.bfloat16)
            for p in (0, 2):
                mode = "int8" if ids.numel() <= DECODE_ROWS else "bf16"
                y = stage.run(xb.cuda(), ids.cuda(), shared, pass_index=p).cpu().double()
                flat = ids.reshape(-1)
                ref = torch.empty(T * k, m, dtype=torch.float64)
                for r in range(T * k):
                    e = int(flat[r])
                    if (e, p) not in dense:
                        dense[(e, p)] = yg.reconstruct_itq_moe(desc, comps, experts=[e], pass_index=p)[0].double()
                    pis = comps["post_input_scale"][e].double() if pe else comps["post_input_scale"].double()
                    xr = xb[r // k if shared else r].float()[None]
                    xe = effective_input(xr, comps["su"], pis, stage.block_in, mode)[0]
                    ref[r] = dense[(e, p)] @ xe
                err = (y - ref).abs()
                rel_max = float(err.max() / ref.abs().max())
                rel_mean = float(err.mean() / ref.abs().mean())
                good = rel_max < 1e-2 and rel_mean < 3e-3
                ok &= good
                print(f"{name:8s} [{m}x{n}] T={T:3d} pass {p} {mode:4s} rel max {rel_max:.2e} mean {rel_mean:.2e} "
                      f"{'ok' if good else 'FAIL'}", flush=True)
        # the prefill path at a decode-sized batch (and vice versa) agree too
        import vllm_gguf_plugin.inkling.itq_moe as M
        ids = torch.stack([torch.randperm(E)[:k] for _ in range(8)]).cuda()
        x = (torch.randn(8 if shared else 32, n) * 0.7).to(torch.bfloat16).cuda()
        a = stage._run_decode(x, ids, shared, 1)
        b = stage._run_prefill(x, ids, shared, 1)
        rel = float((a - b).abs().max() / b.abs().max())
        good = rel < 2e-2
        ok &= good
        print(f"{name:8s} decode vs prefill path, 8 tokens: rel max {rel:.2e} {'ok' if good else 'FAIL'}")
        # the single-launch GEMV computes exactly what the per-format launches do
        one = M.ONE_LAUNCH
        M.ONE_LAUNCH = True
        a1 = stage._run_decode(x, ids, shared, 1)
        M.ONE_LAUNCH = False
        a0 = stage._run_decode(x, ids, shared, 1)
        M.ONE_LAUNCH = one
        same = torch.equal(a0, a1)
        ok &= same
        print(f"{name:8s} one launch vs one per format ({len(stage.present)} formats): "
              f"{'identical' if same else 'DIFFERENT'}")
        del M
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
