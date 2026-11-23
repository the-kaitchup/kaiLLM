"""Resolve and fetch this campaign's GGUFs directly from pinned Hugging Face revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from benchmark import MANIFEST, write_json


def checkpoints(manifest):
    return {**manifest["models"], **{name: manifest[name] for name in ("dspine", "dspark")}}


def resolve(output):
    from huggingface_hub import HfApi

    api = HfApi()
    # Query each repo once, and only retrieve weight metadata at its resolved commit.
    heads = {}
    frozen = json.loads(json.dumps(MANIFEST))
    for m in checkpoints(frozen).values():
        repo = m["repo"]
        if repo not in heads:
            heads[repo] = api.model_info(repo, revision="main", files_metadata=True)
        head = heads[repo]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("resolve")
    p.add_argument("output", type=Path)
    p = sub.add_parser("fetch")
    p.add_argument("checkpoint", choices=checkpoints(MANIFEST))
    p.add_argument("dest", type=Path)
    args = parser.parse_args()
    if args.action == "resolve":
        resolve(args.output)
    else:
        print(download(args.checkpoint, checkpoints(MANIFEST)[args.checkpoint], args.dest))
