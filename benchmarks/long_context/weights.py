"""Resolve and fetch this campaign's GGUFs directly from pinned Hugging Face revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from copy import deepcopy
from pathlib import Path

from benchmark import MANIFEST, write_json


def checkpoints(manifest):
    targets = manifest["models"]
    shared = {name: manifest[name] for name in ("dspine", "dspark")}
    named = manifest.get("drafters", {})
    if set(named) & (targets.keys() | shared.keys()):
        raise ValueError("A drafter download key collides with another model or auxiliary")
    return {**targets, **shared, **named}


def resolve(output):
    from huggingface_hub import HfApi

    api = HfApi()
    # Query each repo once, and only retrieve weight metadata at its resolved commit.
    heads = {}
    frozen = json.loads(json.dumps(MANIFEST))
    for key, m in checkpoints(frozen).items():
        repo = m["repo"]
        if repo not in heads:
            heads[repo] = api.model_info(repo, revision="main", files_metadata=True)
        head = heads[repo]
        if key in frozen["models"] and head.sha != m["revision"]:
            frozen.setdefault("model_history", {}).setdefault(key, {})[m["revision"]] = deepcopy(m)
        if key in frozen.get("drafters", {}) and head.sha != m["revision"]:
            frozen.setdefault("drafter_history", {}).setdefault(key, {})[m["revision"]] = deepcopy(m)
        entries = {f.rfilename: f for f in head.siblings}
        for name in m["files"]:
            item = entries[name]
            if item.lfs and item.size:
                m["files"][name] = {"bytes": item.size, "sha256": item.lfs.sha256}
            elif name == "config.json":
                from huggingface_hub import hf_hub_url
                with urllib.request.urlopen(hf_hub_url(repo, name, revision=head.sha), timeout=30) as response:
                    data = response.read()
                m["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            else:
                raise ValueError(f"Missing LFS digest for {repo}/{name}")
        m["revision"] = head.sha
    write_json(output, frozen)


def download(key, metadata, dest):
    from huggingface_hub import hf_hub_download

    directory = dest / metadata["repo"] / metadata["revision"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in metadata["files"].items():
        file = Path(hf_hub_download(metadata["repo"], name, revision=metadata["revision"],
                                    local_dir=directory, cache_dir=dest / ".hub"))
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if file.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"GGUF integrity check failed for {name}")
    write_json(directory / ("download-" + key + ".json"),
               {"source": "huggingface.co", "repo": metadata["repo"],
                "revision": metadata["revision"], "files": metadata["files"]})
    return directory / next(iter(metadata["files"]))


def inventory(key, metadata, dest):
    """Read the stored byte counts of transformer-block and Engram tensors from all GGUF shards."""
    from gguf import GGUFReader

    base = dest / metadata["repo"] / metadata["revision"]
    proof = json.loads((base / f"download-{key}.json").read_text())
    if proof != {"source": "huggingface.co", **{k: metadata[k] for k in ("repo", "revision", "files")}}:
        raise ValueError("Expected a matching Hugging Face download receipt")
    blocks = 0
    block_bytes = engram_bytes = 0
    geometry = None
    for name, spec in metadata["files"].items():
        path = base / name
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Truncated GGUF shard: {name}")
        reader = GGUFReader(str(path))
        if metadata.get("family") == "lfm" and metadata.get("kind") != "chimachima":
            from stock_kv_floor import lfm_geometry
            geometry = lfm_geometry(reader)
        for tensor in reader.tensors:
            if tensor.name.startswith("blk."):
                block_bytes += int(tensor.n_bytes)
                blocks += 1
            elif tensor.name == "per_layer_token_embd.weight":
                engram_bytes += int(tensor.n_bytes)
    if not blocks:
        raise ValueError("No transformer block tensors found")
    record = {"checkpoint": key, "revision": metadata["revision"], "block_tensors": blocks,
            "transformer_block_bytes": block_bytes, "engram_bytes": engram_bytes}
    if geometry:
        from stock_kv_floor import bytes_per_token
        record.update(global_attention_geometry=geometry, minimum_bf16_target_kv_bytes_per_token=bytes_per_token(geometry))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("resolve")
    p.add_argument("output", type=Path)
    p = sub.add_parser("fetch")
    p.add_argument("checkpoint", choices=checkpoints(MANIFEST))
    p.add_argument("dest", type=Path)
    p = sub.add_parser("inventory")
    p.add_argument("checkpoint", choices=MANIFEST["models"])
    p.add_argument("dest", type=Path)
    args = parser.parse_args()
    if args.action == "resolve":
        resolve(args.output)
    elif args.action == "inventory":
        print(json.dumps(inventory(args.checkpoint, MANIFEST["models"][args.checkpoint], args.dest), indent=2))
    else:
        print(download(args.checkpoint, checkpoints(MANIFEST)[args.checkpoint], args.dest))
