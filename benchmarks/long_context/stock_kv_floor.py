"""Lower bound on a single GPU's upfront allocation under the pinned llama.cpp settings."""
import json
from pathlib import Path

from benchmark import MANIFEST, OUTPUT, PROMPT, llama_revision


CATALOG = json.loads(Path(__file__).with_name("stock_kv_floors.json").read_text())


def lfm_geometry(reader):
    def field(name):
        return reader.fields["lfm3." + name].contents()

    if reader.fields["general.architecture"].contents() != "lfm3":
        raise ValueError("Expected an LFM3 checkpoint")
    heads = field("attention.head_count_kv")
    swa = field("attention.sliding_window_pattern")
    count = field("block_count")
    if len(heads) != count or len(swa) != count:
        raise ValueError("Invalid per-layer attention geometry")
    layers = [(layer, n) for layer, (n, local) in enumerate(zip(heads, swa, strict=True)) if n and not local]
    if not layers:
        raise ValueError("The pinned architecture has no global attention cache")
    return {"global_layers": [i for i, _ in layers], "global_kv_heads": [n for _, n in layers],
            "key_head_width": field("attention.key_length"), "value_head_width": field("attention.value_length")}


def bytes_per_token(geometry):
    return 2 * sum(geometry["global_kv_heads"]) * (geometry["key_head_width"] + geometry["value_head_width"])


def minimum_bytes(checkpoint, batch):
    """Ignore all sliding caches, scratch space, and weights without verified CUDA placement."""
    item = CATALOG["checkpoints"].get(checkpoint)
    if not item:
        return None
    profile = CATALOG["profiles"][item["profile"]]
    model = MANIFEST["models"][checkpoint]
    if (llama_revision(model) != profile["upstream"]
            or [v["sha256"] for v in model["files"].values()] != item["file_sha256"]):
        return None
    cache = bytes_per_token(profile["geometry"]) * batch * (PROMPT + OUTPUT)
    return cache + item.get("minimum_resident_expert_bytes", 0)
