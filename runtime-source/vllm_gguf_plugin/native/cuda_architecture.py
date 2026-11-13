# SPDX-License-Identifier: Apache-2.0
"""Architecture identities for the explicitly enabled native YAQA backends.

This module does not allocate or initialize CUDA. Parent-process auto discovery
uses vLLM's platform capability query; workers validate the actual tensor device.
"""

import os

ENVIRONMENT = "VLLM_GGUF_YAQA_CUDA_ARCH"
SUPPORTED = ((8, 6), (12, 0))
_LIBRARY_ARCHITECTURES = {}


def parse_architecture(value):
    if isinstance(value, str):
        normalized = value.lower().removeprefix("sm_").removeprefix("sm")
        values = {"86": (8, 6), "8.6": (8, 6), "120": (12, 0), "12.0": (12, 0)}
        result = values.get(normalized)
    elif isinstance(value, (tuple, list)):
        result = tuple(value)
    else:
        result = None
    if result not in SUPPORTED:
        raise ValueError("Native YAQA requires an explicit SM86 or SM120 target")
    return result


def resolve_architecture(value=None):
    """Resolve a build target; explicit values support GPU-free compilation."""
    if value is not None:
        return parse_architecture(value)
    requested = os.getenv(ENVIRONMENT, "auto")
    if requested != "auto":
        return parse_architecture(requested)
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None:
        raise RuntimeError(
            "Cannot discover native YAQA CUDA target; set "
            f"{ENVIRONMENT}=sm86 or sm120 for GPU-free build/configuration"
        )
    return parse_architecture(tuple(capability))


def device_architecture(capability):
    """Return the supported worker target; reject a conflicting explicit build."""
    actual = tuple(capability)
    if actual not in SUPPORTED:
        return None
    requested = os.getenv(ENVIRONMENT, "auto")
    if requested != "auto" and parse_architecture(requested) != actual:
        raise ValueError("Native YAQA build target differs from tensor GPU capability")
    return actual


def architecture_identity(value=None):
    major, minor = resolve_architecture(value)
    number = f"{major}{minor}"
    return {
        "compute_capability": [major, minor],
        "target": "sm_" + number,
        "gencode": f"-gencode=arch=compute_{number},code=sm_{number}",
        "architecture_contract": "explicit-sm86-sm120-native-YAQA-v1",
    }


def bind_library_architecture(key, architecture):
    target = parse_architecture(architecture)
    previous = _LIBRARY_ARCHITECTURES.setdefault(key, target)
    if previous != target:
        raise ValueError(
            "Native YAQA library key is already bound to another GPU target"
        )


def validate_library_device(key, capability):
    target = _LIBRARY_ARCHITECTURES.get(key)
    if target is None:
        raise RuntimeError("Native YAQA library must load before graph capture")
    if tuple(capability) != target:
        raise ValueError(
            "Native YAQA library target differs from tensor GPU capability"
        )
