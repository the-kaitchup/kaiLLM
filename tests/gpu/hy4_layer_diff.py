"""Per-layer kaiLLM vs llama.cpp on the same Hy4 GGUF: which layer / block diverges first.

    source tests/gpu/hy4_env.sh <python>
    $PY tests/gpu/hy4_layer_diff.py <model.gguf> --dump-tensors <dump_tensors binary> --tokens tokens.bin \
        [--n 64] [--dtype float32]

llama.cpp (dump_tensors, CPU) writes attn_norm / attn_out / ffn_norm / ffn_out / engram_delta /
ffn_moe_logits per layer; kaiLLM runs the vLLM model in-process (real CUDA ops, paged caches, our
metadata builder) with forward hooks at the same points. Prints relative Frobenius error per layer.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

NAMES = ("attn_norm", "attn_out", "ffn_norm", "ffn_out", "engram_delta", "ffn_moe_logits")


def load_dump(d: Path, name: str):
    p = d / f"{name}.bin"
    if not p.is_file():
        return None
    raw = p.read_bytes()
    ne = np.frombuffer(raw[:32], dtype=np.int64)
    a = np.frombuffer(raw[32:], dtype=np.float32).reshape([int(x) for x in ne[::-1]])
    return torch.from_numpy(a.copy()).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--dump-tensors", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    ids = np.fromfile(a.tokens, dtype=np.int32)[: a.n]

    td = Path(tempfile.mkdtemp(prefix="hy4diff-"))
    ids.tofile(td / "t.i32")
    subprocess.run([str(a.dump_tensors), str(a.gguf), str(td / "t.i32"), str(td), ",".join(NAMES),
                    str(os.cpu_count() or 16)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("llama.cpp tensors:", len(list(td.glob("*.bin"))))

    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
    from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache, customize_attention_spec
    from vllm_gguf_plugin.hy4.attention import hy4_sparse_mla_backend

    n, bs, dev = int(a.n), 16, torch.device("cuda")
    ea = EngineArgs(model=str(a.gguf), dtype=a.dtype, enforce_eager=True, max_model_len=max(n + 16, 256),
                    max_num_batched_tokens=max(n + 16, 256), max_num_seqs=1, block_size=bs)
    vc = ea.create_engine_config()
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}")
    caps: dict[str, torch.Tensor] = {}
    with set_current_vllm_config(vc), torch.inference_mode():
        ensure_model_parallel_initialized(1, 1)
        model = get_model_loader(vc.load_config).load_model(vllm_config=vc, model_config=vc.model_config)
        layers = {k: m for k, m in vc.compilation_config.static_forward_context.items() if isinstance(m, AttentionLayerBase)}
        specs = {k: customize_attention_spec(m.get_attn_backend(), m.get_kv_cache_spec(vc)) for k, m in layers.items()}
        groups = get_kv_cache_groups(vc, dict(specs))
        be = hy4_sparse_mla_backend()
        layout = resolve_kv_cache_layout(vc, [[x.name for x in get_supported_kv_cache_layouts([be])]], specs.values())
        kc = get_kv_cache_config_from_groups(vc, groups, sum(s.page_size_bytes for s in specs.values()) * 64)
        caches = allocate_kv_cache(kc, dev, layout)
        bind_kv_cache(caches, vc.compilation_config.static_forward_context, [], kv_cache_groups=kc.kv_cache_groups)
        builder = be.get_builder_cls()(groups[0].kv_cache_spec, list(layers), vc, dev)

        def hook(name):
            def f(mod, inp, out):
                caps[name] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
            return f

        def pre_hook(name):
            def f(mod, inp):
                caps[name] = inp[0].detach().float().cpu()
            return f

        for i, layer in enumerate(model.model.layers):
            layer.input_layernorm.register_forward_hook(hook(f"attn_norm-{i}"))
            layer.self_attn.register_forward_hook(hook(f"attn_out-{i}"))
            layer.post_attention_layernorm.register_forward_hook(hook(f"ffn_norm-{i}"))
            layer.mlp.register_forward_hook(hook(f"ffn_out-{i}"))
            if layer.engram is not None:
                layer.engram.register_forward_hook(hook(f"engram_delta-{i}"))

        blocks = list(range(1, (n + bs - 1) // bs + 1))
        pos = torch.arange(n, device=dev)
        slots = (torch.tensor(blocks, device=dev)[pos // bs] * bs + pos % bs).long()
        q = torch.tensor([0, n], dtype=torch.int32)
        cm = CommonAttentionMetadata(
            query_start_loc=q.to(dev), query_start_loc_cpu=q, seq_lens=torch.tensor([n], dtype=torch.int32, device=dev),
            num_reqs=1, num_actual_tokens=n, max_query_len=n, max_seq_len=n,
            block_table_tensor=torch.tensor([blocks], dtype=torch.int32, device=dev), slot_mapping=slots)
        md = builder.build(0, cm)
        names = list(layers)
        with set_forward_context({k: md for k in names}, vc, num_tokens=n, slot_mapping={k: slots for k in names}):
            model(torch.from_numpy(ids.astype(np.int64)).to(dev), pos)
        for i, layer in enumerate(model.model.layers):
            if layer.idx >= model.config.first_k_dense_replace and f"ffn_norm-{i}" in caps:
                h = caps[f"ffn_norm-{i}"]
                caps[f"ffn_moe_logits-{i}"] = h @ layer.mlp.router.float().cpu().t()

    print(f"{'layer':>5} " + " ".join(f"{k:>14}" for k in NAMES))
    for i in range(len(model.model.layers)):
        row = []
        for k in NAMES:
            ref, got = load_dump(td, f"{k}-{i}"), caps.get(f"{k}-{i}")
            if ref is None or got is None:
                row.append(f"{'-':>14}")
                continue
            got = got.reshape(ref.shape) if got.numel() == ref.numel() else got
            if got.shape != ref.shape:
                row.append(f"{'shape':>14}")
                continue
            e = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            row.append(f"{e:14.2e}")
        print(f"{i:5d} " + " ".join(row))


if __name__ == "__main__":
    main()
