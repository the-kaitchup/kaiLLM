# SPDX-License-Identifier: Apache-2.0
"""A synthetic routed-expert ITQ GGUF with mixed per-expert formats goes through
the manifest reader, the component loader and the FP32 oracle; each expert must equal the dense itq oracle on its
own byte range, and malformed descriptors must be refused.

    PYTHONPATH=runtime-source:runtime-source/gguf-reader python tests/test_itq_moe_format.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source"), str(ROOT / "runtime-source/gguf-reader")]

import gguf  # noqa: E402
from vllm_gguf_plugin import yaqa_gguf as yg  # noqa: E402


def table(V, seed):
    g = np.random.default_rng(seed)
    return np.clip(np.round(g.normal(0, 40, size=(1 << yg.ITQ_TABLE_BITS[V], V))), -127, 127).astype(np.int8)


def build(path: Path, E=6, m=64, n=96, formats=((4, 5), (2, 4), (2, 8), (4, 12), (2, 3), (4, 4)), passes=3):
    """One fused-style expert tensor + one down-style tensor (per-expert input scales)."""
    rng = np.random.default_rng(7)
    w = gguf.GGUFWriter(str(path), "inkling2_loop")
    w.add_string("general.name", "itq-moe synthetic")
    tensors, manifest = {}, {"version": 1, "tensors": {}}
    tensors["itq.table.v4t12"] = table(4, 1)
    tensors["itq.table.v2t10"] = table(2, 2)
    for name, (mm, nn), per_expert_pis in (("blk.6.ffn_gate_up_exps", (m, n), False),
                                            ("blk.6.ffn_down_exps", (n, m // 2 if (m // 2) % 32 == 0 else 32), True)):
        fm = [formats[e % len(formats)] for e in range(E)]
        desc = {"codec": "itq-moe", "version": 1, "shape": [E, mm, nn], "L": 16, "hash": "mul",
                "layout": "tile16-rowmajor-steps-lsb", "V": [V for V, _ in fm], "bps": [b for _, b in fm],
                "rotation_in": yg_auto(nn), "rotation_out": yg_auto(mm), "components": {}}
        total = yg.itq_moe_offsets(desc)[-1]
        comps = {"stream": rng.integers(-128, 128, size=total, dtype=np.int8),
                 "su": (rng.integers(0, 2, size=nn) * 2 - 1).astype(np.int8),
                 "sv": (rng.integers(0, 2, size=mm) * 2 - 1).astype(np.int8),
                 "wscale": rng.uniform(0.5, 2.0, size=E).astype(np.float32),
                 "post_output_scale": rng.uniform(0.8, 1.2, size=(E, mm)).astype(np.float32),
                 "post_input_scale": rng.uniform(0.8, 1.2, size=(E, nn) if per_expert_pis else (nn,)).astype(np.float32),
                 "pass_output_scale": rng.uniform(0.9, 1.1, size=(passes, E, mm)).astype(np.float32)}
        for field, value in comps.items():
            tensors[f"{name}.itq.{field}"] = value
            desc["components"][field] = f"{name}.itq.{field}"
        for V in sorted(set(desc["V"])):
            desc["components"][f"table_v{V}"] = f"itq.table.v{V}t{yg.ITQ_TABLE_BITS[V]}"
        manifest["tensors"][name + ".weight"] = desc
    w.add_string(yg.MANIFEST_KEY, json.dumps(manifest))
    for name, value in tensors.items():
        w.add_tensor(name, value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return manifest


def yg_auto(width):
    size = width & -width
    return "fwht" if size == width else f"block{size}"


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "moe.gguf"
        manifest = build(path)
        reader = gguf.GGUFReader(str(path))
        tensors = yg.read_yaqa_manifest(reader)
        assert set(tensors) == set(manifest["tensors"]), tensors.keys()
        for name, desc in tensors.items():
            comps = yg.load_yaqa_components(reader, desc)
            E, m, n = desc["shape"]
            dense = yg.reconstruct_itq_moe(desc, comps)
            assert dense.shape == (E, m, n)
            offsets = yg.itq_moe_offsets(desc)
            for e in range(E):
                V, bps = desc["V"][e], desc["bps"][e]
                # independent per-expert dense descriptor from the raw bytes
                raw = comps["stream"][offsets[e]:offsets[e + 1]]
                assert raw.numel() == m * n * bps // V // 8
                sub = {"codec": "itq", "version": 1, "shape": [m, n], "bps": bps, "V": V, "L": 16,
                       "table_bits": yg.ITQ_TABLE_BITS[V], "hash": "mul", "layout": "tile16-rowmajor-steps-lsb",
                       "rotation_in": desc["rotation_in"], "rotation_out": desc["rotation_out"],
                       "components": {f: f for f in ("stream", "su", "sv", "wscale", "table")}}
                yg.validate_yaqa_descriptor(sub)
                pis = comps["post_input_scale"]
                ref = yg.reconstruct_itq(sub, {"stream": raw.reshape(m * n // 256, -1), "su": comps["su"],
                                               "sv": comps["sv"], "wscale": comps["wscale"][e:e + 1],
                                               "table": comps[f"table_v{V}"],
                                               "post_output_scale": comps["post_output_scale"][e],
                                               "post_input_scale": pis[e] if pis.dim() == 2 else pis})
                assert torch.equal(dense[e], ref), (name, e)
                p1 = yg.reconstruct_itq_moe(desc, comps, experts=[e], pass_index=1)[0]
                assert torch.allclose(p1, ref * comps["pass_output_scale"][1, e][:, None]), (name, e)
            print(f"{name}: {E} experts {sorted(set(zip(desc['V'], desc['bps'])))} [{m}, {n}] "
                  f"{offsets[-1]} stream bytes OK")
        # refusals
        good = next(iter(tensors.values()))
        for change, why in ((lambda x: x.update(bps=[13] * x["shape"][0]), "bps out of range"),
                            (lambda x: x.update(V=3), "V"),
                            (lambda x: x.update(shape=x["shape"][1:]), "2-D shape"),
                            (lambda x: x.update(rotation_in="block48"), "block not a power of two"),
                            (lambda x: x["components"].pop("table_v2"), "missing table"),
                            (lambda x: x.update(bps=x["bps"][:-1]), "short bps list")):
            bad = copy.deepcopy(good)
            change(bad)
            try:
                yg.validate_yaqa_descriptor(bad)
            except ValueError:
                continue
            raise AssertionError("accepted a bad descriptor: " + why)
        print("refusals OK")


if __name__ == "__main__":
    main()
