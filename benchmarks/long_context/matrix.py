"""Enumerate the requested run cells in batch order for an assigned set of checkpoints."""
import argparse

from benchmark import BATCHES, MANIFEST, inkling_draft_pair, inkling_native_placement


DGX_SPARK = "DGX Spark"


def modes(key, engine, batch, *, gpu=None):
    model = MANIFEST["models"][key]
    ordinary_only = batch in model.get("benchmark_plain_gpu_batches", {}).get(gpu, ())
    if gpu == DGX_SPARK and not ordinary_only and batch not in model.get("benchmark_additional_gpu_batches", {}).get(gpu, ()):
        return ()
    if gpu and model.get("benchmark_gpus") and gpu not in model["benchmark_gpus"]:
        if not ordinary_only and batch not in model.get("benchmark_additional_gpu_batches", {}).get(gpu, ()):
            return ()
    if engine == "llama.cpp" and model["kind"] == "chimachima":
        return ()
    if ordinary_only:
        return ["plain"]
    family = model["family"]
    selected = ["plain"]
    if family == "lfm" and batch <= 8:
        selected.append("ngram1" if batch == 8 else "ngram3")
    if model.get("native_mtp_layers", 0) > 0:
        selected.append("mtp5")
    if family in ("qwen", "daisy") and engine == "kaiLLM" and batch == 1:
        selected.extend(("dspine7", "dspine15"))
    if family == "inkling" and (engine == "llama.cpp" or model.get("dspark") or inkling_draft_pair(model, batch, gpu_names=[gpu] if gpu else ())):
        selected.append("dspark5")
        if engine == 'llama.cpp' and batch in model.get('llama_cpp_dspark3_gpu_batches', {}).get(gpu, ()):
            selected.append('dspark3')
        if engine == 'kaiLLM' and gpu:
            placement_gpu = gpu if gpu == DGX_SPARK or gpu.startswith(('RTX ', 'NVIDIA ')) else 'RTX ' + gpu
            if inkling_native_placement(model, batch, [placement_gpu], draft_tokens=3):
                selected.append('dspark3')
    return selected


def cells(checkpoints, batches=BATCHES, *, gpu=None):
    for batch in batches:
        for key in checkpoints:
            for engine in ("kaiLLM", "llama.cpp"):
                for mode in modes(key, engine, batch, gpu=gpu):
                    yield key, engine, mode, batch


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("checkpoints", nargs="*", metavar="CHECKPOINT")
    cli.add_argument("--batches", type=int, nargs="+", choices=BATCHES, default=BATCHES)
    args = cli.parse_args()
    unknown = set(args.checkpoints) - MANIFEST["models"].keys()
    if unknown:
        cli.error("Unknown checkpoints: " + ", ".join(sorted(unknown)))
    for cell in cells(args.checkpoints or MANIFEST["models"], args.batches):
        print(*cell)
