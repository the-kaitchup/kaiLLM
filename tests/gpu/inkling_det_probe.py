# SPDX-License-Identifier: Apache-2.0
"""Run-to-run determinism probe of the Inkling decode path: one model load, the same request run twice from scratch
(prefill of a corpus window's first tokens, then greedy decode steps, eager), every glue / projection / expert /
attention / conv call's output recorded, and the first call whose output differs bitwise between the two runs
reported (with its kind, layer, phase and the size of the difference).  Repeated --runs times to see which ops differ
on any pair.

    $PY tests/gpu/inkling_det_probe.py MODEL.gguf --corpus DIR [--prompt 64] [--steps 3] [--runs 3] --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.15)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.platforms import current_platform
    from inkling_runner import Runner
    import vllm.model_executor.models.inkling2_loop as M
    from vllm_gguf_plugin.inkling import glue as G
    from vllm_gguf_plugin.inkling import itq_moe, moe

    ea = EngineArgs(model=str(a.gguf), dtype="bfloat16", enforce_eager=True, max_model_len=4096,
                    max_num_batched_tokens=4096, max_num_seqs=4, enable_prefix_caching=False,
                    gpu_memory_utilization=a.gpu_mem)
    vc = ea.create_engine_config()
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}")
    torch.cuda.set_device(0)

    rec: list = []
    phase = {"p": "prefill"}

    def grab(x):
        if isinstance(x, torch.Tensor):
            return [x.detach().float().cpu().clone()] if x.is_floating_point() else [x.detach().cpu().clone()]
        if isinstance(x, (tuple, list)):
            return [t for v in x for t in grab(v)]
        return []

    def wrap(obj, name, label):
        orig = getattr(obj, name)

        def f(*args, **kw):
            out = orig(*args, **kw)
            rec.append((label, phase["p"], grab(out)))
            return out
        setattr(obj, name, f)

    for obj, name, label in ((G, "mix", "mix"), (G, "control", "control"), (G, "rms", "rms"),
                             (G, "post_projection", "post_projection"), (G, "route", "route"),
                             (G, "gate_up_epilogue", "gate_up_epilogue"), (G, "combine", "combine"),
                             (M, "attend", "attention"), (M, "conv_step", "conv"), (M.InklingLayer, "_lin", "linear"),
                             (moe.ExpertStage, "run", "experts"), (itq_moe.ITQExpertStage, "run", "itq experts")):
        if hasattr(obj, name):
            wrap(obj, name, label)
    for obj, name, label in ((G, "rms_pair", "rms_pair"), (G, "gate_attn", "gate_attn")):
        if hasattr(obj, name):
            wrap(obj, name, label)

    toks = np.fromfile(a.corpus / "tokens.bin", dtype="<i4")
    w = json.loads((a.corpus / "windows.json").read_text())[0]
    prompt = toks[w["start"]: w["start"] + a.prompt].tolist()
    runs = []
    t0 = time.time()
    with set_current_vllm_config(vc), torch.inference_mode():
        ensure_model_parallel_initialized(1, 1)
        model = get_model_loader(vc.load_config).load_model(vllm_config=vc, model_config=vc.model_config)
        current_platform.update_block_size_for_backend(vc)
        run = Runner(vc, model, num_blocks=a.blocks, device="cuda")
        print(f"loaded in {time.time() - t0:.0f} s", flush=True)
        for r in range(a.runs + 1):                  # run 0: warm-up (compiles), not compared
            rec.clear()
            phase["p"] = "prefill"
            logits = run.step([(r, prompt, 0, True)])
            toks_out = [int(logits[-1].argmax())]
            for s in range(a.steps):
                phase["p"] = f"decode{s}"
                logits = run.step([(r, [toks_out[-1]], len(prompt) + s, True)])
                toks_out.append(int(logits[-1].argmax()))
            run.release(r)
            torch.cuda.synchronize()
            if r:
                runs.append((list(rec), toks_out))
            print(f"run {r}: {len(rec)} calls, tokens {toks_out}", flush=True)
    report = {"pairs": []}
    base_rec, base_toks = runs[0]
    for i, (other, other_toks) in enumerate(runs[1:], 1):
        first, ndiff, kinds = None, 0, {}
        for j, ((la, pa, ta), (lb, pb, tb)) in enumerate(zip(base_rec, other)):
            diff = 0.0
            for x, y in zip(ta, tb):
                if x.shape != y.shape:
                    diff = float("inf")
                    break
                if not torch.equal(x, y):
                    diff = max(diff, float((x.double() - y.double()).abs().max()))
            if diff > 0:
                ndiff += 1
                kinds[f"{la}/{pa}"] = kinds.get(f"{la}/{pa}", 0) + 1
                if first is None:
                    first = {"call": j, "kind": la, "phase": pa, "max_abs_diff": diff,
                             "earlier_calls_identical": j}
        pair = {"runs": [1, i + 1], "calls": len(base_rec), "differing_calls": ndiff, "first": first,
                "differing_by_kind": kinds, "tokens": [base_toks, other_toks]}
        report["pairs"].append(pair)
        print(json.dumps(pair), flush=True)
    a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
