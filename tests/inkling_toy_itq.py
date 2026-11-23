# SPDX-License-Identifier: Apache-2.0
"""Mixed-format toy Inkling GGUF with ITQ routed experts (itq-moe v1) beside stock GGML expert tensors, per tensor:

    layer 0  gate_up itq-moe (V2 b5 / V4 b8 alternating)        down: the source file's type (e.g. NVFP4 + .scale)
    layer 2  gate + up split itq-moe (V2 b4 / V4 b6)            down itq-moe (V2 b8)        (a looped core layer)
    layer 5  gate_up: the source file's type (e.g. Q4_K)        down itq-moe (V4 b12 / V2 b3)

Random trellis streams; wscale and post_output_scale give each ITQ expert the source tensor's per-row RMS, so the
toy keeps its activation ranges.  The stock tensors and metadata are copied from SRC unchanged.

    python tests/inkling_toy_itq.py SRC.gguf OUT.gguf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source"), str(ROOT / "runtime-source/gguf-reader"), str(ROOT / "tests")]

import gguf  # noqa: E402
import torch  # noqa: E402
from vllm_gguf_plugin import yaqa_gguf as yg  # noqa: E402

PLAN = {(0, "gate_up"): [(2, 5), (4, 8)], (2, "gate"): [(2, 4), (4, 6)], (2, "up"): [(4, 6), (2, 4)],
        (2, "down"): [(2, 8)], (5, "down"): [(4, 12), (2, 3)]}


def kv_items(reader):
    out = []
    for f in reader.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            sub = f.types[1]
            val = ([bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data]
                   if sub == gguf.GGUFValueType.STRING else [f.parts[i].tolist()[0] for i in f.data])
            out.append((f.name, val, vt, sub))
        elif vt == gguf.GGUFValueType.STRING:
            out.append((f.name, bytes(f.parts[f.data[0]]).decode("utf-8", errors="surrogateescape"), vt, None))
        else:
            out.append((f.name, f.parts[f.data[0]].tolist()[0], vt, None))
    return out


def dequant(t):
    from test_inkling_ref_cpu import load_dense  # noqa: F401  (same dequantization as the reference)
    from vllm_gguf_plugin.inkling import ggml_types as G
    shape = tuple(reversed([int(x) for x in t.shape]))
    if int(t.tensor_type) in G.NEW_TYPES:
        a = G.dequantize(torch.from_numpy(np.array(t.data, copy=True)).reshape(-1), int(t.tensor_type)).numpy()
    elif t.tensor_type == gguf.GGMLQuantizationType.F32:
        a = np.asarray(t.data, dtype=np.float32)
    else:
        a = gguf.quants.dequantize(t.data, t.tensor_type)
    return a.reshape(shape).astype(np.float32)


def itq_tensor(name, rows_rms, E, m, n, formats, gen, su=None):
    """components + descriptor of a random itq-moe tensor whose rows have RMS rows_rms [E, m]."""
    fm = [formats[e % len(formats)] for e in range(E)]
    desc = {"codec": "itq-moe", "version": 1, "shape": [E, m, n], "L": 16, "hash": "mul",
            "layout": "tile16-rowmajor-steps-lsb", "V": [f[0] for f in fm], "bps": [f[1] for f in fm],
            "rotation_in": "fwht" if n & (n - 1) == 0 else f"block{n & -n}",
            "rotation_out": "fwht" if m & (m - 1) == 0 else f"block{m & -m}", "components": {}}
    total = yg.itq_moe_offsets(desc)[-1]
    comps = {"stream": gen.integers(-128, 128, size=total, dtype=np.int8),
             "su": su if su is not None else (gen.integers(0, 2, size=n) * 2 - 1).astype(np.int8),
             "sv": (gen.integers(0, 2, size=m) * 2 - 1).astype(np.int8),
             "wscale": np.ones(E, dtype=np.float32)}
    tables = {V: np.clip(np.round(np.random.default_rng(V).standard_normal((1 << yg.ITQ_TABLE_BITS[V], V)) * 32),
                         -127, 127).astype(np.int8) for V in (2, 4)}
    tcomp = {k: torch.from_numpy(v) for k, v in comps.items()}
    for V in sorted({f[0] for f in fm}):
        tcomp[f"table_v{V}"] = torch.from_numpy(tables[V])
    desc_c = dict(desc, components={k: k for k in tcomp})
    dense = yg.reconstruct_itq_moe(desc_c, yg.check_itq_moe_components(desc_c, dict(tcomp)))   # unit wscale
    rms = dense.pow(2).mean(-1).sqrt().numpy()                                                   # [E, m]
    comps["post_output_scale"] = (rows_rms / np.maximum(rms, 1e-12)).astype(np.float32)
    base = name.removesuffix(".weight") + ".itq."
    out = {}
    for k, v in comps.items():
        out[base + k] = v
        desc["components"][k] = base + k
    for V in sorted({f[0] for f in fm}):
        desc["components"][f"table_v{V}"] = f"itq.table.v{V}t{yg.ITQ_TABLE_BITS[V]}"
    return desc, out, tables


def build(src: Path, dst: Path, seed: int = 3) -> dict:
    gen = np.random.default_rng(seed)
    r = gguf.GGUFReader(str(src))
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    tensors = {t.name: t for t in r.tensors}
    E = int(tensors["blk.0.ffn_gate_inp.weight"].shape[1])
    manifest, new, tables = {"version": 1, "tensors": {}}, {}, {}
    drop = set()
    shared_su = {}
    for (L, part), formats in PLAN.items():
        fused = tensors[f"blk.{L}.ffn_gate_up_exps.weight"]
        D = int(fused.shape[0])
        F = int(fused.shape[1]) // 2
        if part in ("gate_up", "gate", "up"):
            src_w = dequant(fused)                                                      # [E, 2F, D]
            sel = {"gate_up": slice(0, 2 * F), "gate": slice(0, F), "up": slice(F, 2 * F)}[part]
            w = src_w[:, sel]
            drop |= {f"blk.{L}.ffn_gate_up_exps.weight", f"blk.{L}.ffn_gate_up_exps.scale"}
        else:
            w = dequant(tensors[f"blk.{L}.ffn_down_exps.weight"])                       # [E, D, F]
            drop |= {f"blk.{L}.ffn_down_exps.weight", f"blk.{L}.ffn_down_exps.scale"}
        name = f"blk.{L}.ffn_{part}_exps.weight"
        rows_rms = np.sqrt((w.astype(np.float64) ** 2).mean(-1))
        su = None
        if part in ("gate", "up"):        # split gate / up share their input signs (spec)
            su = shared_su.setdefault(L, (gen.integers(0, 2, size=w.shape[2]) * 2 - 1).astype(np.int8))
        desc, comps, tabs = itq_tensor(name, rows_rms, E, w.shape[1], w.shape[2], formats, gen, su=su)
        manifest["tensors"][name] = desc
        new.update(comps)
        tables.update(tabs)
    w = gguf.GGUFWriter(str(dst), arch)
    for k, v, vt, st in kv_items(r):
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(k, v, vt, sub_type=st)
        else:
            w.add_key_value(k, v, vt)
    w.add_string(yg.MANIFEST_KEY, json.dumps(manifest))
    for t in r.tensors:
        if t.name not in drop:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    for name, v in new.items():
        w.add_tensor(name, v)
    for V in sorted(tables):
        w.add_tensor(f"itq.table.v{V}t{yg.ITQ_TABLE_BITS[V]}", tables[V])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return manifest


if __name__ == "__main__":
    m = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"wrote {sys.argv[2]}: {len(m['tensors'])} itq-moe tensors: {sorted(m['tensors'])}")
