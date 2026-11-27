"""DSpine speculative decoding (tencent/dspine drafters) on vLLM's DFlash path.

Serve with ``--speculative-config '{"method": "dflash", "model": DIR, "num_speculative_tokens": 15}'`` where DIR is a
DSpine checkpoint whose config.json names ``"architectures": ["DSpineDraftModel"]`` (the HF export says
DFlashDraftModel; the DSpine fields live in dflash_config).  num_speculative_tokens = block size - 1 (2..16 blocks).
"""
from __future__ import annotations

import sys

# EAGLEConfig (method dflash) renames a draft architecture to DFlash{arch}: register and route both names.
ARCHS = ("DSpineDraftModel", "DFlashDSpineDraftModel")
INKLING_ARCHS = ("InklingDSparkDraftModel", "DFlashInklingDSparkDraftModel")


def register_dspine() -> None:
    from vllm.model_executor.models import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for arch in ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, "vllm_gguf_plugin.dspine.model:DSpineQwen3ForCausalLM")
    for arch in INKLING_ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, "vllm_gguf_plugin.inkling.dspark_model:InklingDSparkForCausalLM")

    import vllm.v1.worker.gpu.spec_decode as spec_decode

    original = spec_decode.init_speculator
    if getattr(original, "_dspine", False):
        return

    def init_speculator(vllm_config, device, req_states):
        spec = vllm_config.speculative_config
        if spec is not None and spec.method == 'dflash' and set(INKLING_ARCHS).intersection(spec.draft_model_config.architectures or []):
            from ..inkling.dspark_speculator import InklingDSparkSpeculator
            return InklingDSparkSpeculator(vllm_config, device)
        if spec is not None and spec.method == "dflash" and set(ARCHS) & set(spec.draft_model_config.architectures or []):
            from .speculator import DSpineSpeculator

            return DSpineSpeculator(vllm_config, device)
        return original(vllm_config, device, req_states)

    init_speculator._dspine = True
    spec_decode.init_speculator = init_speculator
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")     # imported before the plugin: patch its binding
    if runner is not None:
        runner.init_speculator = init_speculator
