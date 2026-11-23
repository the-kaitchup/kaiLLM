# SPDX-License-Identifier: Apache-2.0
"""Toy Inkling 2 Mini GGUFs (llama.cpp master `inkling2_loop` layout) for exact CPU parity tests.

Tiny dims, real tokenizer (copied from the BF16 GGUF's metadata header), weights scaled so that every clip,
softcap, window and relative extent is exercised by an ~80-token sequence:

    python tests/inkling_toy.py --header bf16-head.bin --out toy.gguf [--scales] [--seed N]

`--header`: the first ~12 MB of ggml-org's Inkling-2-Mini-BF16.gguf (the metadata section; tensor data not needed).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source/gguf-reader")]

import gguf  # noqa: E402

ARCH = "inkling2_loop"

TOY = dict(block_count=8, embedding_length=256, head=4, head_kv=2, head_dim=64, experts=8, used=3, ff=128,
           window=16, extent=24, n_rel=16, qk_clip=4.0, rel_clip=1.5, log_floor=8, log_alpha=0.1, conv=4,
           res_block=2, start=2, end=6, passes=3, rank=8)


def read_kv(path: Path) -> list:
    """(key, GGUFValueType, value, sub_type) of every metadata field of a (possibly truncated) GGUF file."""
    b = path.read_bytes()
    magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", b, 0)
    assert magic == b"GGUF" and ver == 3, (magic, ver)
    off = 24
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rstr(o):
        n = struct.unpack_from("<Q", b, o)[0]
        return b[o + 8:o + 8 + n].decode("utf-8", errors="surrogateescape"), o + 8 + n

    def rval(t, o):
        if t == 8:
            return rstr(o)
        if t == 9:
            st, n = struct.unpack_from("<IQ", b, o)
            o += 12
            out = []
            if st == 8:
                for _ in range(n):
                    s, o = rstr(o)
                    out.append(s)
                return (out, st), o
            size = struct.calcsize(fmt[st])
            arr = np.frombuffer(b, dtype=np.dtype(fmt[st][1:]).newbyteorder("<"), count=n, offset=o)
            return (arr.tolist(), st), o + size * n
        return struct.unpack_from(fmt[t], b, o)[0], o + struct.calcsize(fmt[t])

    out = []
    for _ in range(n_kv):
        k, off = rstr(off)
        t = struct.unpack_from("<I", b, off)[0]
        off += 4
        v, off = rval(t, off)
        if t == 9:
            out.append((k, t, v[0], v[1]))
        else:
            out.append((k, t, v, None))
    return out


GENTLE = dict(up=4.0, gate=1.5, head=3.0, sink=2.0, bias=1.0)     # --gentle: float noise does not blow up
SHARP = dict(up=20.0, gate=3.0, head=15.0, sink=8.0, bias=5.0)     # default: every clip / softcap triggers


def build(header: Path, out: Path, seed: int = 0, scales: bool = False, toy: dict = TOY, gentle: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sc = GENTLE if gentle else SHARP
    t = toy
    D, H, Hkv, hd, E, F = t["embedding_length"], t["head"], t["head_kv"], t["head_dim"], t["experts"], t["ff"]
    L, nrel, nconv = t["block_count"], t["n_rel"], t["conv"]
    swa = [i % 2 == 0 for i in range(L)]
    kv = read_kv(header)
    vocab_padded = next(v for k, _, v, _ in kv if k == f"{ARCH}.vocab_size")
    vocab = next(v for k, _, v, _ in kv if k == f"{ARCH}.vocab_size_unpadded")
    w = gguf.GGUFWriter(str(out), ARCH)
    VT = gguf.GGUFValueType
    for k, vt, v, st in kv:
        if k.startswith("tokenizer.") or k in ("general.type",):
            if vt == VT.ARRAY:
                w.add_key_value(k, v, VT(vt), sub_type=VT(st))
            else:
                w.add_key_value(k, v, VT(vt))
    w.add_name("inkling2-loop-toy")
    u32, f32 = VT.UINT32, VT.FLOAT32
    P = ARCH + "."
    for k, v in (("block_count", L), ("context_length", 4096), ("embedding_length", D), ("feed_forward_length", F),
                 ("expert_count", E), ("expert_used_count", t["used"]), ("attention.key_length", hd),
                 ("attention.value_length", hd), ("vocab_size", vocab_padded), ("vocab_size_unpadded", vocab),
                 ("expert_feed_forward_length", F), ("attention.key_length_swa", hd),
                 ("attention.value_length_swa", hd), ("rope.dimension_count", 0),
                 ("attention.sliding_window", t["window"]), ("shortconv.l_cache", nconv),
                 ("attn_res.block_size", t["res_block"]), ("attention.relative_length", nrel),
                 ("attention.relative_extent", t["extent"]), ("attention.log_scale_floor", t["log_floor"]),
                 ("loop.start", t["start"]), ("loop.end", t["end"]), ("loop.count", t["passes"]),
                 ("loop.gate_rank", t["rank"])):
        w.add_key_value(P + k, v, u32)
    for k, v in (("attention.layer_norm_rms_epsilon", 1e-6), ("expert_weights_scale", 8.0),
                 ("activation.situ_beta", 4.0), ("activation.situ_linear_beta", 25.0),
                 ("final_logit_softcapping", 30.0), ("attention.qk_clip", t["qk_clip"]),
                 ("attention.relative_clip", t["rel_clip"]), ("attention.log_scale_alpha", t["log_alpha"])):
        w.add_key_value(P + k, v, f32)
    w.add_key_value(P + "attention.head_count", [H] * L, VT.ARRAY, sub_type=VT.INT32)
    w.add_key_value(P + "attention.head_count_kv", [Hkv] * L, VT.ARRAY, sub_type=VT.INT32)
    w.add_key_value(P + "attention.sliding_window_pattern", swa, VT.ARRAY, sub_type=VT.BOOL)
    w.add_key_value(P + "loop.varying", False, VT.BOOL)

    def n(*shape, s=1.0):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def ones(*shape, s=0.1):
        return (1.0 + rng.standard_normal(shape) * s).astype(np.float32)

    T = {}
    T["token_embd.weight"] = n(vocab_padded, D)
    T["token_embd_norm.weight"] = ones(D)
    T["output.weight"] = n(vocab_padded, D, s=sc["head"] / np.sqrt(D))
    T["output_norm.weight"] = ones(D)
    T["output_res_query.weight"] = n(D)
    kvd = hd * Hkv
    for i in range(L):
        p = f"blk.{i}."
        T[p + "attn_q.weight"] = n(H * hd, D, s=0.6 / np.sqrt(D) * np.sqrt(D / hd))
        T[p + "attn_k.weight"] = n(kvd, D, s=0.6 / np.sqrt(D) * np.sqrt(D / hd) * 1.6)
        T[p + "attn_v.weight"] = n(kvd, D, s=1.0 / np.sqrt(D))
        T[p + "attn_output.weight"] = n(D, H * hd, s=1.0 / np.sqrt(H * hd))
        T[p + "attn_gate.weight"] = n(H * hd, D, s=1.0 / np.sqrt(D))
        T[p + "attn_q_clip.weight"] = ones(H, hd, s=0.2)
        T[p + "attn_sink_proj.weight"] = n(H, D, s=sc["sink"] / np.sqrt(D))
        T[p + "attn_rel.weight"] = n(nrel * H, D, s=0.5 / np.sqrt(D) * np.sqrt(D / nrel))
        ext = t["window"] if swa[i] else t["extent"]
        T[p + "attn_rel_bias.weight"] = n(ext, nrel, s=sc["bias"])
        T[p + "attn_rel_clip.weight"] = ones(H, nrel, s=0.2)
        T[p + "attn_k_conv.weight"] = n(kvd, nconv, s=0.5)
        T[p + "attn_v_conv.weight"] = n(kvd, nconv, s=0.5)
        T[p + "attn_post_conv.weight"] = n(D, nconv, s=0.5)
        T[p + "ffn_post_conv.weight"] = n(D, nconv, s=0.5)
        if swa[i]:
            T[p + "attn_norm.weight"] = ones(D)
        else:
            for c in "qkv":
                T[p + f"attn_{c}_inp_norm.weight"] = ones(D)
            T[p + "attn_k_conv_proj.weight"] = n(nconv * Hkv, D, s=0.5 / np.sqrt(D))
            T[p + "attn_v_conv_proj.weight"] = n(nconv * Hkv, D, s=0.5 / np.sqrt(D))
        if i > 0:
            T[p + "attn_res_query.weight"] = n(1 if swa[i] else 3, D)
        T[p + "ffn_res_query.weight"] = n(D)
        T[p + "ffn_norm.weight"] = ones(D)
        T[p + "ffn_gate_inp.weight"] = n(E, D, s=1.0 / np.sqrt(D))
        T[p + "exp_probs_b.bias"] = n(E, s=0.1)
        gu = np.concatenate([n(E, F, D, s=sc["gate"] / np.sqrt(D)), n(E, F, D, s=sc["up"] / np.sqrt(D))], axis=1)
        T[p + "ffn_gate_up_exps.weight"] = gu
        T[p + "ffn_down_exps.weight"] = n(E, D, F, s=0.1 / np.sqrt(F))
        if scales:
            T[p + "ffn_gate_up_exps.scale"] = rng.uniform(0.5, 1.5, size=E).astype(np.float32)
            T[p + "ffn_down_exps.scale"] = rng.uniform(0.5, 1.5, size=E).astype(np.float32)
        if t["start"] <= i < t["end"]:
            T[p + "loop_msg_query.weight"] = n(4, D)
            for c in ("q", "k", "v", "ffn"):
                T[p + f"loop_{c}_ctrl_down.weight"] = n(t["rank"], 2 * D, s=1.0 / np.sqrt(2 * D) * np.sqrt(2 * D / 16))
                T[p + f"loop_{c}_ctrl_gain.weight"] = n(D, t["rank"], s=1.0 / np.sqrt(t["rank"]))
                T[p + f"loop_{c}_ctrl_forget.weight"] = n(D, t["rank"], s=1.0 / np.sqrt(t["rank"]))
    for name, value in T.items():
        w.add_tensor(name, value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", action="store_true", help="per-expert .scale tensors (tests the reshare patch)")
    ap.add_argument("--gentle", action="store_true", help="smaller weights: GPU numerics tests (no chaos)")
    a = ap.parse_args()
    T = build(a.header, a.out, a.seed, a.scales, gentle=a.gentle)
    print(f"wrote {a.out}: {len(T)} tensors")


if __name__ == "__main__":
    main()
