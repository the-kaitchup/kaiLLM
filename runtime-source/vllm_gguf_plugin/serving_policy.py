# SPDX-License-Identifier: Apache-2.0
"""Standard-library serving policy and installed native artifact resolution."""

import copy
import ctypes
import hashlib
import importlib.machinery
import json
from pathlib import Path

CATALOG_SCHEMA = "vllm-gguf-serving-catalog-v1"


def choose_speculative_tokens(capacity, requested=3, verifier_row_budget=16):
    if (
        type(capacity) is not int
        or capacity < 1
        or type(requested) is not int
        or requested < 0
        or type(verifier_row_budget) is not int
        or verifier_row_budget < 1
    ):
        raise ValueError("Positive capacity/budget and nonnegative drafts required")
    return max(0, min(requested, verifier_row_budget // capacity - 1))


MTP_TOKENS = 4
MTP_MAX_CAPACITY = 8


def choose_mtp_tokens(capacity):
    """Default MTP draft length for a GGUF with MTP layers.

    Qwen3.8 27B (UD-Q4_K_XL, UD-Q2_K_XL) on an RTX PRO 6000: four draft tokens
    were the fastest or within 1% of the fastest at batches 1, 2 and 8; at 16
    concurrent sequences MTP was slower than plain decoding at every length.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Positive capacity required")
    return MTP_TOKENS if capacity <= MTP_MAX_CAPACITY else 0


def mtp_config(tokens):
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Nonnegative integer draft count required")
    return {"method": "mtp", "num_speculative_tokens": tokens} if tokens else None


def capture_family(capacity, spec_tokens, scheduler_tokens, tile=64):
    if any(type(v) is not int or v < 1 for v in (capacity, scheduler_tokens, tile)):
        raise ValueError("Positive integer capacity, scheduler budget and tile needed")
    if type(spec_tokens) is not int or spec_tokens < 0:
        raise ValueError("Nonnegative integer draft count required")
    maximum = capacity * (spec_tokens + 1)
    if scheduler_tokens < maximum:
        raise ValueError("Scheduler token budget is smaller than verification rows")
    decode, count = {maximum}, 1
    while count <= maximum:
        decode.add(count)
        count *= 2
    return sorted(
        decode | set(range(2 * tile, scheduler_tokens + 1, tile)) | {scheduler_tokens}
    )


def discover_capability(logical_device=0):
    """Query the CUDA driver without importing Torch or creating a context."""
    if type(logical_device) is not int or logical_device < 0:
        raise ValueError("Nonnegative logical CUDA device required")
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceComputeCapability": [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int
        ],
    }
    for name, arguments in signatures.items():
        call = getattr(driver, name)
        call.argtypes, call.restype = arguments, ctypes.c_int

    def call(name, *arguments):
        status = getattr(driver, name)(*arguments)
        if status:
            raise RuntimeError(f"CUDA capability query {name} failed: {status}")

    before, after = ctypes.c_void_p(), ctypes.c_void_p()
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    call("cuInit", 0)
    call("cuCtxGetCurrent", ctypes.byref(before))
    call("cuDeviceGet", ctypes.byref(device), logical_device)
    call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), device)
    call("cuCtxGetCurrent", ctypes.byref(after))
    if before.value != after.value:
        raise RuntimeError("CUDA capability query changed the current context")
    return f"{major.value}.{minor.value}"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    result = (root / relative).resolve()
    if relative.is_absolute() or not result.is_relative_to(root):
        raise ValueError("Catalog paths must remain relative to the installed package")
    return result


def load_target(catalog_path, capability, package_root, inherited_environment):
    """Resolve a target and preserve environment keys outside the catalog."""
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError("Unsupported serving catalog schema")
    targets = catalog.get("targets", {})
    target = targets.get(capability)
    if not target or target.get("available", True) is not True:
        reason = (
            target.get("reason", "selected native assets are not installed")
            if target else "target is not installed"
        )
        raise ValueError(f"CUDA target {capability} is unavailable: {reason}")
    root = Path(package_root).resolve()
    assets = target.get("assets", {})
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Selected target needs a nonempty native asset inventory")
    verified = {}

    def verify(relative, expected):
        path = installed_path(root, relative)
        if verified.get(str(path)) != expected:
            if file_sha256(path) != expected:
                raise ValueError(
                    "Installed native asset differs from its catalog: " + str(path)
                )
            verified[str(path)] = expected
        return str(path)

    for relative, digest in assets.items():
        verify(relative, digest)
    controlled = set(catalog.get("controlled_environment_keys", []))
    for record in targets.values():
        controlled.update(record.get("environment", {}))
        controlled.update(record.get("environment_paths", {}))
    environment = {
        key: value for key, value in inherited_environment.items()
        if key not in controlled
    }
    values = target.get("environment", {})
    if not isinstance(values, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()
    ):
        raise ValueError("Catalog environment must contain string pairs")
    environment.update(values)
    for key, relative in target.get("environment_paths", {}).items():
        if relative not in assets:
            raise ValueError("Environment asset is missing from the catalog: " + key)
        environment[key] = verify(relative, assets[relative])
    library_dirs = []
    for relative in target.get("library_dirs", []):
        directory = installed_path(root, relative)
        if not directory.is_dir():
            raise ValueError("Native library directory is missing: " + str(directory))
        library_dirs.append(str(directory))
    if library_dirs:
        previous = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = ":".join(
            library_dirs + ([previous] if previous else [])
        )
    adapter = copy.deepcopy(target.get("conditional_adapter"))
    if adapter is not None:
        if adapter.get("ABI") != 2:
            raise ValueError("The conditional adapter must implement ABI2")
        expected_capability = list(map(int, capability.split(".")))
        if adapter.get("compute_capability") != expected_capability:
            raise ValueError("Conditional adapter targets a different GPU")
        adapter["path"] = verify(adapter["path"], adapter["sha256"])
        metadata = adapter.get("metadata", [])
        metadata = [metadata] if isinstance(metadata, dict) else metadata
        if not metadata:
            raise ValueError("Conditional adapter build metadata is required")
        for record in metadata:
            record["path"] = verify(record["path"], record["sha256"])
        adapter["metadata"] = metadata
    return {
        "environment": environment,
        "conditional_adapter": adapter,
        "core_contract": target.get("core_contract", {}),
        "catalog_scope": catalog.get("scope"),
        "target_scope": target.get("scope"),
        "verified_native_assets": len(verified),
    }


def verify_core_contract(contract, search_path):
    """Read installed source pins without executing the vLLM package."""
    if contract.get("package", "vllm") != "vllm" or not contract.get("files"):
        raise ValueError("A nonempty installed vLLM core contract is required")
    spec = importlib.machinery.PathFinder.find_spec("vllm", search_path)
    if spec is None or spec.origin is None:
        raise ValueError("Install the compatible vLLM core before serving")
    root = Path(spec.origin).resolve().parent
    for relative, digest in contract["files"].items():
        path = installed_path(root, relative)
        if file_sha256(path) != digest:
            raise ValueError(
                "Installed vLLM core differs from the serving contract: " + str(path)
            )
    return {
        "package_root": str(root),
        "verified_files": len(contract["files"]),
        "base_identity": contract.get("base_identity"),
        "scope": contract.get(
            "scope", "Listed source pins only; not a whole-base identity"
        ),
    }


def speculative_config(tokens, adapter, lookup_min=2, lookup_max=8):
    if not 1 <= lookup_min <= lookup_max <= 32:
        raise ValueError("Require 1 <= lookup-min <= lookup-max <= 32")
    if tokens == 0:
        return None
    result = {
        "method": "ngram_gpu", "num_speculative_tokens": tokens,
        "prompt_lookup_min": lookup_min, "prompt_lookup_max": lookup_max,
        "rejection_sample_method": "standard",
    }
    if tokens == 3:
        if adapter is None:
            raise ValueError("K3 serving requires the actual-singleton adapter")
        result.update(
            ngram_gpu_single_active=True, ngram_gpu_single_active_adapter=adapter
        )
    return result
