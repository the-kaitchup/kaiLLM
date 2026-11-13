"""Validate the prepared native runtime without loading CUDA libraries."""

import hashlib
import json
import os
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

PROFILE = "qwen35-dense-0.8b-v1"
MODEL_SHA256 = "2da9c6ba82ffe7d2a01bdfebd4357662582e48a7f36e8708661207a9edd022f4"
# Registry membership describes supported input bytes, not validation status.
# Schema 2 manifests explicitly declare which entries have passed validation.
CHECKPOINTS = {
    MODEL_SHA256: {"profile": PROFILE, "architecture": "qwen35-dense-0.8b"},
    "d5171fb931c4864eceebc6978011f56d7ae9b6b035a519de243620f15476c03c": {
        "profile": "qwen35-dense-0.8b-iq2-m-v1",
        "architecture": "qwen35-dense-0.8b",
    },
    "5c0d172ef233f3d215a3fbe2989e1b8cbbb8d613c4e52b544438d7e8bb2e4dbc": {
        "profile": "qwen35-dense-0.8b-iq3-xxs-v1",
        "architecture": "qwen35-dense-0.8b",
    },
    "0af96165ea615bea39a04118d63f0b6d35908aea850ee4a51aa6151d851b8b35": {
        "profile": "qwen35-dense-2b-q4-k-xl-v1",
        "architecture": "qwen35-dense-2b",
    },
}
ARCHITECTURE_DIMENSIONS = {
    "qwen35-dense-0.8b": {"hidden_size": 1024, "intermediate_size": 3584},
    "qwen35-dense-2b": {"hidden_size": 2048, "intermediate_size": 6144},
}
LIBRARY_ENV = frozenset(
    {
        "GGUF_CURRENT_GGML_DIRECT",
        "GGUF_CURRENT_GGML_SHARED",
        "GGUF_CURRENT_GGML_BODY_MMA",
        "GGUF_Q6_MMA_LIBRARY",
        "GGUF_CURRENT_NORM",
        "GGUF_CURRENT_ATTN",
        "GGUF_CURRENT_ATTN_BATCH",
        "GGUF_CURRENT_ROPE",
        "GGUF_NATIVE_ORDER_GDN",
        "GGUF_NATIVE_POSTCONV_FUSED",
    }
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path():
    return (
        Path(
            os.environ.get(
                "VLLM_GGUF_NATIVE_MANIFEST",
                str(Path(__file__).with_name("bundle.json")),
            )
        )
        .expanduser()
        .resolve()
    )


def _artifact(root, item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Native bundle artifact needs a path and SHA256")
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Native bundle artifacts must stay inside the bundle")
    if sha256_file(path) != item.get("sha256"):
        raise ValueError(f"Native bundle checksum mismatch: {path}")
    return str(path)


def _features(features, *, schema, digest):
    allowed = {"body_mma", "fused_prep", "attention_graphs", "affine_mma"}
    if schema == 2:
        allowed.update({"q6_mma_head", "prefill_batch_fa", "llama_batch_schedule"})
    if (
        not isinstance(features, dict)
        or set(features) - allowed
        or any(type(value) is not bool for value in features.values())
    ):
        raise ValueError("Invalid native bundle feature selection")
    if features.get("affine_mma") and not features.get("body_mma"):
        raise ValueError("Native affine MMA requires body MMA")
    # Only the original 1024-wide Q6 head has a validated optimized kernel.
    # Other checkpoint heads must retain ordinary native GGML dispatch.
    if features.get("q6_mma_head") and digest != MODEL_SHA256:
        raise ValueError("Native Q6 MMA head is supported only by the original profile")
    return {
        name: features.get(name, name == "q6_mma_head" and digest == MODEL_SHA256)
        for name in sorted(allowed)
    }


def _checkpoints(data):
    schema = data["schema_version"]
    if schema == 1:
        if data.get("profile") != PROFILE or data.get("model_sha256") != MODEL_SHA256:
            raise ValueError("Native bundle profile/checkpoint is not supported")
        entries = [{**data, "validated": True}]
    else:
        entries = data.get("checkpoints")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Native bundle needs explicit checkpoint profiles")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid native checkpoint profile")
        digest = entry.get("model_sha256")
        known = CHECKPOINTS.get(digest) if isinstance(digest, str) else None
        if known is None or entry.get("profile") != known["profile"]:
            raise ValueError("Native bundle profile/checkpoint is not supported")
        if digest in result:
            raise ValueError("Duplicate native checkpoint profile")
        if type(entry.get("validated")) is not bool:
            raise ValueError("Native checkpoint needs an explicit validation status")
        if entry.get("architecture", known["architecture"]) != known["architecture"]:
            raise ValueError("Native checkpoint has an incompatible architecture")
        result[digest] = {
            **known,
            "model_sha256": digest,
            "validated": entry["validated"],
            "features": _features(
                entry.get("features", {}), schema=schema, digest=digest
            ),
        }
    return result


def selected_profile(bundle, digest, *, allow_experimental=False):
    """Select declared, pinned bytes; registry membership alone never enables them."""
    if not isinstance(digest, str):
        raise ValueError("Native checkpoint selection needs a SHA256 string")
    profile = bundle["checkpoints"].get(digest)
    if profile is None:
        raise ValueError("checkpoint is not declared in the prepared native bundle")
    if not profile["validated"] and not allow_experimental:
        raise ValueError("checkpoint has not passed native validation")
    return profile


def selection_marker(bundle, profile):
    marker = {
        "revision": bundle["schema_version"],
        "profile": profile["profile"],
        "manifest": bundle["path"],
        "manifest_sha256": bundle["sha256"],
        "model_sha256": profile["model_sha256"],
        "features": profile["features"],
    }
    if bundle["schema_version"] == 2:
        marker["architecture"] = profile["architecture"]
        marker["experimental_checkpoint"] = not profile["validated"]
    return marker


@lru_cache(maxsize=4)
def load_manifest(path_string, modified_ns, size):
    """The stat key permits replacing an installed bundle in a fresh selection."""
    path = Path(path_string)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported native bundle manifest schema")
    if data.get("enabled") is not True:
        raise ValueError("Native bundle is disabled")
    checkpoints = _checkpoints(data)
    capabilities = data.get("compute_capabilities")
    validated = data.get(
        "validated_compute_capabilities", [[8, 6]] if capabilities == [[8, 6]] else []
    )
    allowed = [[8, 6], [9, 0], [12, 0]]
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(item not in allowed for item in capabilities)
        or not isinstance(validated, list)
        or any(item not in capabilities for item in validated)
    ):
        raise ValueError("Unsupported native GPU architecture declaration")
    if (
        any(item != [8, 6] for item in capabilities)
        and data.get("experimental_architectures") is not True
    ):
        raise ValueError("Other GPU builds require explicit experimental enablement")
    for package in ("torch", "vllm"):
        if data.get(f"{package}_version") != version(package):
            raise ValueError(
                f"Native bundle requires {package}={data.get(f'{package}_version')}"
            )
    entries = data.get("libraries", {})
    if not isinstance(entries, dict) or not LIBRARY_ENV.issubset(entries):
        raise ValueError("Native bundle is missing required CUDA helpers")
    if any(not key.startswith("GGUF_") for key in entries):
        raise ValueError("Native bundle contains an invalid library environment key")
    root = path.parent
    libraries = {key: _artifact(root, item) for key, item in entries.items()}
    preloads = [_artifact(root, item) for item in data.get("preload_libraries", [])]
    if len(preloads) < 3:
        raise ValueError("Native bundle requires its GGML base/CPU/CUDA libraries")
    # Optional full package/source inventory supplements each helper manifest.
    for item in data.get("artifacts", []):
        _artifact(root, item)
    result = {
        "schema_version": data["schema_version"],
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoints": checkpoints,
        "compute_capabilities": capabilities,
        "validated_compute_capabilities": validated,
        "libraries": libraries,
        "preloads": preloads,
    }
    if data["schema_version"] == 1:
        # Preserve the public schema-1 fields for existing local bundle tools.
        result.update(checkpoints[MODEL_SHA256])
    return result


def prepared_manifest(path=None):
    path = manifest_path() if path is None else Path(path).resolve()
    stat = path.stat()
    return load_manifest(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _checkpoint_digest(path, modified_ns, size, inode):
    return sha256_file(path)


def checkpoint_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _checkpoint_digest(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
