# SPDX-License-Identifier: Apache-2.0
"""New GGML block types (MXFP4, NVFP4, Q1_0, Q2_0, F8_E4M3): kaiLLM's torch dequantization vs ggml's own C
`to_float` (libggml-base from a llama.cpp master build, through ctypes) on random bytes, bit-exact (NaN patterns
excluded: random F8 bytes include NaN encodings, compared as NaN == NaN).

    python tests/test_inkling_ggml_types.py "$GGML_BASE_LIB" [--cuda]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source")]

from vllm_gguf_plugin.inkling import ggml_types as G  # noqa: E402


class TypeTraits(ctypes.Structure):
    # ggml.h struct ggml_type_traits (master 29b52d3): type_name, blck_size, blck_size_interleave, type_size,
    # is_quantized, to_float, from_float_ref
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


def ggml_to_float(lib, t: int, raw: np.ndarray, n: int) -> np.ndarray:
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    tr = lib.ggml_get_type_traits(t).contents
    assert tr.blck_size == G.BLOCK[t][0] and tr.type_size == G.BLOCK[t][1], (t, tr.blck_size, tr.type_size)
    fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tr.to_float)
    out = np.empty(n, dtype=np.float32)
    fn(raw.ctypes.data, out.ctypes.data, n)
    return out


def main():
    lib = ctypes.CDLL(sys.argv[1])
    dev = "cuda" if "--cuda" in sys.argv else "cpu"
    rng = np.random.default_rng(0)
    ok = True
    for t, name in ((G.MXFP4, "MXFP4"), (G.NVFP4, "NVFP4"), (G.Q1_0, "Q1_0"), (G.Q2_0, "Q2_0"), (G.F8_E4M3, "F8_E4M3")):
        qk, size = G.BLOCK[t]
        nb = 4096
        raw = rng.integers(0, 256, size=nb * size, dtype=np.uint8)
        if t in (G.Q1_0, G.Q2_0):            # finite f16 scales (random bytes would give inf / NaN d)
            r = raw.reshape(nb, size)
            d = (rng.standard_normal(nb) * 0.05).astype(np.float16).view(np.uint8).reshape(nb, 2)
            r[:, :2] = d
        ref = ggml_to_float(lib, t, raw, nb * qk)
        got = G.dequantize(torch.from_numpy(raw.copy()).to(dev).view(nb, size), t).reshape(-1).cpu().numpy()
        same = (ref.view(np.uint32) == got.view(np.uint32)) | (np.isnan(ref) & np.isnan(got))
        print(f"{name:8s} {nb * qk:8d} values: {'bit-exact' if same.all() else f'{(~same).sum()} differ'}"
              f"  (first diff {np.nonzero(~same)[0][:3].tolist()})" if not same.all() else
              f"{name:8s} {nb * qk:8d} values: bit-exact")
        ok &= bool(same.all())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
