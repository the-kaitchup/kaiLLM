"""Hy4 through vLLM's real layers on CPU: kaiLLM's GGUF loader, MLAAttention, the KV-cache planner and
allocator, our metadata builder, driven step by step like the scheduler would (prefill, chunked
prefill, decode, two requests batched).

    KAILLM_HY4_ROOT="$PWD" PYTHONPATH="$PWD/tests/gpu/site" CUDA_VISIBLE_DEVICES= \
    python tests/test_hy4_vllm_cpu.py "$TOY_GGUF" \
        --dump "$DUMP_LOGITS" --tokens "$TEST_TOKENS" \
        [--dtype bfloat16] [--save logits.npz]

Gate: every multi-step scenario reproduces the one-pass prefill (self-consistency). Also printed:
agreement with llama.cpp (CPU) on the same GGUF. On the toy that agreement is exact (KL ~1e-6) until
the compressed indexer starts selecting, then near-tie group picks flip with llama.cpp's BF16 pooled
keys (its indexer query and head weights match ours to 1e-6; only pooled keys differ, by BF16
rounding), so exactness with selection active is checked against transformers instead
(--save, then tests/test_hy4_vllm_vs_hf_cpu.py).

Scenarios (one request unless noted):
  single   : one 160-token prefill
  chunked  : prefill 48 + 64 + 48 tokens (chunked prefill over the paged caches)
  decode   : prefill 120, then 40 one-token steps
  batched  : two requests of different lengths in the same steps (prefill, then decode)
CPU stand-ins: the latent cache write (vLLM's concat_and_cache_mla is CUDA-only) and direct calls
instead of vLLM's attention custom ops (registered for CUDA only).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def native_logits(dump, gguf, ids):
    with tempfile.TemporaryDirectory() as td:
        t, o = Path(td) / "t.i32", Path(td) / "o.f32"
        np.asarray(ids, dtype=np.int32).tofile(t)
        subprocess.run([str(dump), str(gguf), str(t), str(o), "8"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return torch.from_numpy(np.fromfile(o, dtype=np.float32).reshape(len(ids), -1))


def compare(name, got, ref):
    lp, lq = torch.log_softmax(ref.double(), -1), torch.log_softmax(got.double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (got.argmax(-1) == ref.argmax(-1)).double().mean().item()
    top = got.topk(10, dim=-1).indices
    ear = lp.exp().gather(-1, top).sum(-1) / lp.exp().topk(10, dim=-1).values.sum(-1)
    if os.environ.get("HY4_PER_POS"):
        print("  per-position KL:", " ".join(f"{i}:{v:.1e}" for i, v in enumerate(kl.tolist()) if i % 8 == 0 or v > 1e-3))
    print(f"{name:8s} tokens {got.shape[0]:4d}  max|dlogit| {(got - ref).abs().max().item():.2e}  "
          f"mean KL {kl.mean().item():.2e}  top-1 {top1:.4f}  EAR@10 {ear.mean().item():.4f}")
    return kl.mean().item(), top1


class Runner:
    """Steps a vLLM model by hand: block tables, slot mappings, metadata, forward context."""

    def __init__(self, vc, model, block_size, num_blocks):
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
        from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
        from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
        from vllm.v1.worker.utils import allocate_kv_cache, customize_attention_spec
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm_gguf_plugin.hy4.attention import hy4_sparse_mla_backend

        self.vc, self.model, self.bs = vc, model, block_size
        layers = {n: m for n, m in vc.compilation_config.static_forward_context.items()
                  if isinstance(m, AttentionLayerBase)}
        specs = {n: customize_attention_spec(m.get_attn_backend(), m.get_kv_cache_spec(vc)) for n, m in layers.items()}
        groups = get_kv_cache_groups(vc, dict(specs))
        assert len(groups) == 1, [type(g.kv_cache_spec).__name__ for g in groups]
        be = hy4_sparse_mla_backend()
        layout = resolve_kv_cache_layout(vc, [[x.name for x in get_supported_kv_cache_layouts([be])]],
                                         specs.values())
        page = sum(s.page_size_bytes for s in specs.values())
        kc = get_kv_cache_config_from_groups(vc, groups, page * num_blocks)
        caches = allocate_kv_cache(kc, torch.device("cpu"), layout)
        from vllm.v1.worker.utils import bind_kv_cache
        bind_kv_cache(caches, vc.compilation_config.static_forward_context, [], kv_cache_groups=kc.kv_cache_groups)
        for n, m in layers.items():
            if hasattr(m, "use_direct_call"):
                m.use_direct_call = True        # vLLM's MLA custom ops are registered for CUDA only
        self.layer_names = list(layers)
        self.builder = be.get_builder_cls()(groups[0].kv_cache_spec, self.layer_names, vc, torch.device("cpu"))
        self.free = list(range(1, kc.num_blocks))[::-1]        # block 0 stays the null block
        self.tables: dict[int, list[int]] = {}

        def cpu_cache_update(impl, kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
            flat, stride = flat_kv_row_view(kv_cache, kv_cache.shape[1])
            s = slot_mapping.long()
            ok = s >= 0
            rows = (s // kv_cache.shape[1]) * stride + s % kv_cache.shape[1]
            val = torch.cat([kv_c, k_pe.reshape(k_pe.shape[0], -1)], dim=-1)
            flat[rows[ok]] = val[: s.shape[0]][ok].to(flat.dtype)

        be.get_impl_cls().do_kv_cache_update = cpu_cache_update

    def step(self, batch):
        """batch: list of (req, tokens, start position). Returns logits for every scheduled token."""
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        ids, pos, slots, qsl, seq = [], [], [], [0], []
        for req, toks, start in batch:
            table = self.tables.setdefault(req, [])
            end = start + len(toks)
            while len(table) * self.bs < end:
                table.append(self.free.pop())
            p = torch.arange(start, end)
            ids.append(torch.tensor(toks))
            pos.append(p)
            slots.append(torch.tensor(table)[p // self.bs] * self.bs + p % self.bs)
            qsl.append(qsl[-1] + len(toks))
            seq.append(end)
        width = max(len(self.tables[r]) for r, _, _ in batch)
        bt = torch.zeros(len(batch), width, dtype=torch.int32)
        for i, (r, _, _) in enumerate(batch):
            bt[i, : len(self.tables[r])] = torch.tensor(self.tables[r], dtype=torch.int32)
        T = qsl[-1]
        q = torch.tensor(qsl, dtype=torch.int32)
        cm = CommonAttentionMetadata(
            query_start_loc=q, query_start_loc_cpu=q, seq_lens=torch.tensor(seq, dtype=torch.int32),
            num_reqs=len(batch), num_actual_tokens=T, max_query_len=max(len(t) for _, t, _ in batch),
            max_seq_len=max(seq), block_table_tensor=bt, slot_mapping=torch.cat(slots).long())
        md = self.builder.build(0, cm)
        input_ids, positions = torch.cat(ids), torch.cat(pos)
        sm = {n: cm.slot_mapping for n in self.layer_names}
        with set_forward_context({n: md for n in self.layer_names}, self.vc, num_tokens=T, slot_mapping=sm):
            h = self.model(input_ids, positions)
        return self.model.compute_logits(h).float()

    def release(self, req):
        self.free.extend(self.tables.pop(req, []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--kl-max", type=float, default=1e-4)
    ap.add_argument("--save", type=Path, default=None,
                    help="write every scenario's logits + tokens (.npz) for tests/test_hy4_vllm_vs_hf_cpu.py")
    ap.add_argument("--self-kl-max", type=float, default=None,
                    help="default 1e-6 in float32, 1e-4 in bfloat16 (BF16 matmuls round differently per batch shape)")
    a = ap.parse_args()
    if a.self_kl_max is None:
        a.self_kl_max = 1e-6 if a.dtype == "float32" else 1e-4
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.model_executor.model_loader import get_model_loader

    bs = 16
    ea = EngineArgs(model=str(a.gguf), dtype=a.dtype, enforce_eager=True, max_model_len=1024,
                    max_num_batched_tokens=1024, max_num_seqs=4, block_size=bs)
    vc = ea.create_engine_config()
    vc.device_config.device = torch.device("cpu")
    vc.device_config.device_type = "cpu"
    vc.compilation_config.custom_ops = ["none"]
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}")
    toks = np.fromfile(a.tokens, dtype=np.int32)
    seq_a, seq_b = toks[:160].tolist(), toks[1000:1100].tolist()
    ref_a, ref_b = native_logits(a.dump, a.gguf, seq_a), native_logits(a.dump, a.gguf, seq_b)
    worst, outs = [], {}
    with set_current_vllm_config(vc), torch.inference_mode():
        ensure_model_parallel_initialized(1, 1)
        model = get_model_loader(vc.load_config).load_model(vllm_config=vc, model_config=vc.model_config)
        run = Runner(vc, model, bs, num_blocks=64)

        outs["single"] = run.step([(0, seq_a, 0)]); run.release(0)

        out = [run.step([(1, seq_a[s:e], s)]) for s, e in ((0, 48), (48, 112), (112, 160))]
        outs["chunked"] = torch.cat(out); run.release(1)

        out = [run.step([(2, seq_a[:120], 0)])]
        out += [run.step([(2, seq_a[i:i + 1], i)]) for i in range(120, 160)]
        outs["decode"] = torch.cat(out); run.release(2)

        la, lb = run.step([(3, seq_a[:130], 0), (4, seq_b[:70], 0)]).split([130, 70])
        outa, outb = [la], [lb]
        for i in range(30):
            ya, yb = run.step([(3, seq_a[130 + i:131 + i], 130 + i), (4, seq_b[70 + i:71 + i], 70 + i)]).split([1, 1])
            outa.append(ya); outb.append(yb)
        outs["batch-a"], outs["batch-b"] = torch.cat(outa), torch.cat(outb)
        outs["single-b"] = run.step([(5, seq_b, 0)]); run.release(5)

    print("vs llama.cpp (CPU) on the same GGUF:")
    for k in ("single", "chunked", "decode", "batch-a"):
        worst.append(compare(k, outs[k], ref_a))
    worst.append(compare("batch-b", outs["batch-b"], ref_b))
    print("self-consistency (multi-step vs one prefill pass):")
    selfc = [compare(k, outs[k], outs["single"]) for k in ("chunked", "decode", "batch-a")]
    selfc.append(compare("batch-b", outs["batch-b"], outs["single-b"]))
    if a.save:
        np.savez(a.save, seq_a=np.asarray(seq_a), seq_b=np.asarray(seq_b),
                 **{k.replace("-", "_"): v.numpy() for k, v in outs.items()})
        print("saved", a.save)
    ok_self = all(kl <= a.self_kl_max and t1 == 1.0 for kl, t1 in selfc)
    ok_native = all(kl <= a.kl_max and t1 >= 0.97 for kl, t1 in worst)
    print(f"self-consistency: {'PASS' if ok_self else 'FAIL'}")
    print("llama.cpp: " + ("agrees" if ok_native else
          "differs once the compressed indexer selects (expected on the 4-head toy: llama.cpp's BF16 pooled "
          "keys flip near-tie group picks); check exactness with tests/test_hy4_vllm_vs_hf_cpu.py, or use a "
          "dense-indexer fixture to compare with llama.cpp"))
    sys.exit(0 if ok_self else 1)

if __name__ == "__main__":
    main()
