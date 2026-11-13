"""Resolve hash-verified CUDA assets at their installed location.

This module does not select quantization or execute tensor operations. Asset
admission binds the original source, build contract, target architecture, native
library bytes and distributed metadata bytes. The returned library path is its
installed location; resolving an asset never edits its metadata.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Native asset path leaves its bundle")
    return path


@lru_cache(maxsize=8)
def catalog(manifest):
    path = Path(manifest).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != "kaillm-frozen-native-assets-v1":
        raise ValueError("Unsupported native asset manifest")
    return path.parent, value["assets"]


def resolve(manifest, source, contract, stem, architecture):
    """Return a qualified prebuilt identity, or None for an unbundled source."""
    root, entries = catalog(str(Path(manifest).resolve()))
    source_sha = digest(source)
    selected = [entry for entry in entries
                if entry["source_sha256"] == source_sha
                and entry["contract"] == contract
                and entry["stem"] == stem
                and entry["architecture"] == architecture]
    if not selected:
        return None
    if len(selected) != 1:
        raise ValueError("Ambiguous frozen native asset")
    entry = selected[0]
    library = contained(root, entry["library"])
    metadata = contained(root, entry["metadata"])
    if metadata.parent != library.parent or metadata.name != "identity.json":
        raise ValueError("Native library and identity must remain siblings")
    if digest(library) != entry["library_sha256"]:
        raise ValueError("Frozen native library bytes differ")
    if digest(metadata) != entry["metadata_sha256"]:
        raise ValueError("Frozen native metadata bytes differ")
    identity = json.loads(metadata.read_text())
    for key in ("source_sha256", "contract", "architecture", "library_sha256"):
        if identity.get(key) != entry[key]:
            raise ValueError("Native asset identity differs: " + key)
    # Leave the distributed identity.json byte-identical.
    return {**identity, "library": str(library)}
