"""Enumerate the requested run cells in batch order for an assigned set of checkpoints."""
import argparse

from benchmark import BATCHES, MANIFEST


def modes(key, engine, batch):
    model = MANIFEST["models"][key]
    if engine == "llama.cpp" and model["kind"] == "chimachima":
        return ()
    family = model["family"]
    selected = ["plain"]
    if family == "lfm" and batch <= 8:
        selected.append("ngram1" if batch == 8 else "ngram3")
    if model["speculation"] in ("mtp", "auto-qwen"):
        selected.append("mtp5")
    if family in ("qwen", "daisy") and engine == "kaiLLM" and batch == 1:
        selected.extend(("dspine7", "dspine15"))
    if family == "inkling" and engine == "llama.cpp":
        selected.append("dspark5")
    return selected


def cells(checkpoints, batches=BATCHES):
    for batch in batches:
        for key in checkpoints:
            for engine in ("kaiLLM", "llama.cpp"):
                for mode in modes(key, engine, batch):
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
