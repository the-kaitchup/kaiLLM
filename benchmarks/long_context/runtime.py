"""Combine the pinned installed binaries with the measured checkout in a separate directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sysconfig
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]


def link_or_copy(source, destination):
    try:
        return os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)


def assemble(dest, release=None, source=None):
    source = source or HERE / "runtime-source"
    release = release or Path(sysconfig.get_paths()["purelib"]) / "kaillm/_runtime"
    locked = json.loads((source / "source-lock.json").read_text())
    if not (release / "compatible-vllm/core").is_dir():
        raise FileNotFoundError("Run with the kaiLLM release's installed Python")
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256((source / "source-lock.json").read_bytes()).hexdigest()
    if dest.is_dir():
        if (dest / "benchmark-source.sha256").read_text().strip() == stamp:
            return
        raise ValueError("Choose a fresh runtime directory for this source revision")
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        root = Path(staging) / "runtime"
        shutil.copytree(release, root, symlinks=True, copy_function=link_or_copy,
                        ignore=shutil.ignore_patterns("__pycache__"))
        for relative, digest in locked["source_files"].items():
            file = source / relative
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {relative}")
            parts = Path(relative).parts
            target = (root / "compatible-vllm/core" / Path(*parts[1:])
                      if parts[0] == "vllm-overrides" else root / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            shutil.copy2(file, target)
        # The base runtime may carry a stamp. Never write through its hardlink.
        stamp_path = root / "benchmark-source.sha256"
        stamp_path.unlink(missing_ok=True)
        stamp_path.write_text(stamp + "\n")
        os.rename(root, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--release-runtime", type=Path, help="Runtime extracted from the pinned release wheel")
    p.add_argument("--source-tree", type=Path, help="Extracted runtime-source directory at the measured commit")
    args = p.parse_args()
    assemble(args.output, args.release_runtime, args.source_tree)
