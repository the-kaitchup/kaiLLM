# SPDX-License-Identifier: Apache-2.0
"""GPU memory checkpoints in the server log (KAILLM_MEMLOG=1): after model loading, KV-cache allocation and CUDA
graph capture -- torch allocated / reserved and the driver's free / total, to find VRAM that is neither weights, KV
cache nor graphs (cached free blocks, workspaces)."""
from __future__ import annotations

import os
import sys


def _report(what: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    gib = 2 ** 30
    print(f"kaiLLM memlog [{what}]: allocated {torch.cuda.memory_allocated() / gib:.3f} GiB, reserved "
          f"{torch.cuda.memory_reserved() / gib:.3f} GiB, driver free {free / gib:.3f} / {total / gib:.3f} GiB",
          file=sys.stderr, flush=True)


def install() -> bool:
    if os.getenv("KAILLM_MEMLOG", "0") != "1":
        return False
    hooked = False
    for module_name in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
        except ImportError:
            continue
        cls = getattr(module, "GPUModelRunner", None)
        if cls is None or getattr(cls, "_kai_memlog", False):
            continue
        for name in ("load_model", "initialize_kv_cache", "capture_model"):
            original = getattr(cls, name, None)
            if original is None:
                continue

            def wrapped(self, *args, __original=original, __name=name, **kwargs):
                result = __original(self, *args, **kwargs)
                _report(f"after {__name}")
                return result

            setattr(cls, name, wrapped)
        cls._kai_memlog = True
        hooked = True
    if hooked:
        try:
            from .host_tables import host_bytes

            import atexit

            atexit.register(lambda: print(f"kaiLLM memlog: pinned host tables {host_bytes() / 2**20:.1f} MiB",
                                          file=sys.stderr, flush=True))
        except ImportError:
            pass
    return hooked
