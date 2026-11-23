# SPDX-License-Identifier: Apache-2.0
"""kaiLLM's GGUF linear and MoE kernels on every quantized tensor of a GGUF (e.g. the Inkling toys) vs dequantize +
float matmul: finds kernels that are wrong at a given type / shape.  GPU only.

    $PY tests/gpu/test_inkling_gguf_ops_shapes.py FILE.gguf [T ...] [--only REGEX]

Also checks the CUDA dequantization (ops.ggml_dequantize, embeddings) of every selected tensor.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import torch


def main():
    import gguf
    from vllm_gguf_plugin.inkling import ggml_types as G
    from vllm_gguf_plugin.inkling.moe import ExpertStage
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
    from vllm_gguf_plugin import ops
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = re.compile(args[i + 1])
        args = args[:i] + args[i + 2:]
    path = args[0]
    Ts = [int(x) for x in args[1:]] or [1, 7, 80]
    print(f"GGUF CUDA extension: {ops._CUDA_ENABLED}", flush=True)
    r = gguf.GGUFReader(path)
    torch.manual_seed(0)
    bad = 0
    for t in r.tensors:
        tt = int(t.tensor_type)
        if tt in (0, 1, 30, 24) or ".itq." in t.name or (only is not None and not only.search(t.name)):
            continue
        shape = tuple(reversed([int(x) for x in t.shape]))
        raw = torch.from_numpy(np.array(t.data, copy=True))
        if tt in G.NEW_TYPES:
            dense = G.dequantize(raw.reshape(-1), tt).reshape(shape)
        else:
            dense = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float32))
        dense = dense.cuda()
        if tt not in G.NEW_TYPES:
            rows2 = int(np.prod(shape[:-1]))
            q2 = raw.reshape(rows2, -1).cuda()
            dq = ops.ggml_dequantize(q2, tt, rows2, shape[-1], torch.float32).reshape(shape)
            drel = float((dq - dense).abs().max() / dense.abs().max().clamp_min(1e-30))
            flag = drel > 1e-3
            bad += flag
            print(f"{t.name:36s} {t.tensor_type.name:6s} {shape} dequant rel {drel:.2e} {'BAD' if flag else ''}",
                  flush=True)
            del dq, q2
        if len(shape) == 2 and not t.name.startswith("token_embd"):
            q = raw.reshape(shape[0], -1).cuda()
            for T in Ts:
                x = (torch.randn(T, shape[1]) * 0.5).cuda().to(torch.bfloat16)
                try:
                    y = _fused_mul_mat_gguf(x, q, tt).float()
                except Exception as e:     # noqa: BLE001
                    print(f"{t.name:36s} {t.tensor_type.name:6s} {shape} T={T}: ERROR {e}")
                    bad += 1
                    continue
                ref = x.float() @ dense.t()
                rel = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                flag = rel > 5e-2
                bad += flag
                if flag or T == Ts[0]:
                    print(f"{t.name:36s} {t.tensor_type.name:6s} {shape} T={T}: rel {rel:.2e} {'BAD' if flag else ''}")
        elif len(shape) == 3:
            E, rows, cols = shape
            st = ExpertStage(tt, raw.reshape(E, rows, -1).cuda(), rows, cols, None)
            for T in Ts:
                k = min(3, E)
                ids = torch.stack([torch.randperm(E)[:k] for _ in range(T)]).cuda()
                x = (torch.randn(T, cols) * 0.5).cuda().to(torch.bfloat16)
                y = st.run(x, ids, True).float()
                ref = torch.einsum("tk,tjnk->tjn", x.float(), dense[ids]).reshape(T * k, rows)
                rel = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                flag = rel > 5e-2
                bad += flag
                if flag or T == Ts[0]:
                    print(f"{t.name:36s} {t.tensor_type.name:6s} {shape} T={T}: MoE rel {rel:.2e} "
                          f"{'BAD' if flag else ''}")
    print("PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
