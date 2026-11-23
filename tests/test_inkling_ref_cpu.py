# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini PyTorch reference vs llama.cpp (CPU, F32 weights and caches) on a toy `inkling2_loop` GGUF.

    tests/inkling_toy.py --header bf16-head.bin --out toy.gguf --scales
    HY4_KV_F32=1 dump_logits-cpu toy.gguf tok.i32 llama.f32 8
    python tests/test_inkling_ref_cpu.py toy.gguf tok.i32 llama.f32

The reference runs in float64; llama.cpp (float32) must be as close to it as the reference's own float32 run is
(the toy is deliberately sharp, so float32 rounding grows through the 3 passes): rel(llama, ref64) <= 3 x
rel(ref32, ref64) and top-1 100 %; and one prefill = chunked prefill = token-by-token decode in float64 (rel < 1e-9).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source"), str(ROOT / "runtime-source/gguf-reader")]

import gguf  # noqa: E402
from vllm_gguf_plugin.inkling import ggml_types as G  # noqa: E402
from vllm_gguf_plugin.inkling import ref  # noqa: E402


def gguf_kv(reader) -> dict:
    out = {}
    for f in reader.fields.values():
        if f.name.startswith("GGUF.") or f.name.startswith("tokenizer."):
            continue
        out[f.name] = f.contents()
    return out


def load_dense(path: Path):
    """(hparams, {name: float32 tensor in numpy order}) of a GGUF (F32 / F16 / BF16 or any type gguf-py dequantizes)."""
    r = gguf.GGUFReader(str(path))
    hp = ref.hparams_from_gguf_kv(gguf_kv(r))
    w = {}
    from vllm_gguf_plugin import yaqa_gguf as yg
    manifest = yg.read_yaqa_manifest(r)
    comps_names = {c for d in manifest.values() for c in d.get("components", {}).values()}
    split = {}
    for name, desc in manifest.items():       # ITQ experts: the FP32 oracle (pass-independent part)
        if desc.get("codec") == "itq-moe":
            dense = yg.reconstruct_itq_moe(desc, yg.load_yaqa_components(r, desc))
            if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")):
                split[name] = dense
            else:
                w[name] = dense
    for name in [n for n in split if n.endswith("ffn_gate_exps.weight")]:
        up = name.replace("ffn_gate_exps", "ffn_up_exps")
        w[name.replace("ffn_gate_exps", "ffn_gate_up_exps")] = torch.cat([split[name], split[up]], dim=1)
    for t in r.tensors:
        if t.name in comps_names or t.name.startswith("itq.table."):
            continue
        shape = tuple(reversed([int(x) for x in t.shape]))
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            a = np.asarray(t.data, dtype=np.float32).reshape(shape)
        elif int(t.tensor_type) in G.NEW_TYPES:
            a = G.dequantize(torch.from_numpy(np.array(t.data, copy=True)).reshape(-1), int(t.tensor_type)).numpy()
            a = a.reshape(shape)
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).reshape(shape)
        w[t.name] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return hp, w


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    return {"max_abs": float((a - b).abs().max()), "rel": float((a - b).abs().max() / a.abs().max()),
            "top1": float((a.argmax(-1) == b.argmax(-1)).double().mean()), "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("tokens", type=Path)
    ap.add_argument("llama", type=Path, nargs="?")
    a = ap.parse_args()
    torch.set_num_threads(8)
    hp, w = load_dense(a.gguf)
    tok = torch.from_numpy(np.fromfile(a.tokens, dtype="<i4").astype(np.int64))
    with torch.no_grad():
        ref.set_dtype(torch.float32)
        r32 = ref.InklingReference(hp, w).forward(tok)
        ref.set_dtype(torch.float64)
        m = ref.InklingReference(hp, w)
        full = m.forward(tok)
        c32 = compare(full, r32)
        print("reference float32 vs float64:", {k: f"{v:.3g}" for k, v in c32.items()})
        ok = True
        if a.llama is not None:
            ll = torch.from_numpy(np.fromfile(a.llama, dtype="<f4").reshape(len(tok), hp.vocab))
            c = compare(full, ll)
            print("llama.cpp vs reference float64:", {k: f"{v:.3g}" for k, v in c.items()})
            ok &= c["rel"] <= 3 * max(c32["rel"], 1e-6) and c["top1"] == 1.0
        m.reset()
        cuts = [0, 7, 8, 30, 31, len(tok) - 9, len(tok)]
        chunks = [m.forward(tok[s:e], start=s) for s, e in zip(cuts[:-1], cuts[1:])]
        c = compare(full, torch.cat(chunks))
        print("float64 prefill vs chunked + single-token steps:", {k: f"{v:.3g}" for k, v in c.items()})
        ok &= c["rel"] < 1e-9 and c["top1"] == 1.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
