# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module

from .base import BaseGGUFWeightsAdapter


def get_weights_adapter(config) -> BaseGGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to the default."""
    from .default import GGUFWeightsAdapter
    from .gemma3 import Gemma3GGUFAdapter
    from .lfm3 import Lfm3GGUFAdapter
    from .qwen35 import Qwen35GGUFAdapter

    for cls in (Qwen35GGUFAdapter, Lfm3GGUFAdapter, Gemma3GGUFAdapter):
        if cls.matches(config):
            return cls(config)
    return GGUFWeightsAdapter(config)


def __getattr__(name):
    modules = {
        "GGUFWeightsAdapter": ".default",
        "Gemma3GGUFAdapter": ".gemma3",
        "Qwen35GGUFAdapter": ".qwen35",
        "DiffusionGGUFAdapter": ".diffusion",
        "Flux2KleinDiffusionGGUFAdapter": ".diffusion",
        "QwenImageDiffusionGGUFAdapter": ".diffusion",
        "ZImageDiffusionGGUFAdapter": ".diffusion",
        "get_diffusion_gguf_adapter": ".diffusion",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)


__all__ = [
    "BaseGGUFWeightsAdapter",
    "DiffusionGGUFAdapter",
    "Flux2KleinDiffusionGGUFAdapter",
    "GGUFWeightsAdapter",
    "Gemma3GGUFAdapter",
    "QwenImageDiffusionGGUFAdapter",
    "Qwen35GGUFAdapter",
    "ZImageDiffusionGGUFAdapter",
    "get_diffusion_gguf_adapter",
    "get_weights_adapter",
]
