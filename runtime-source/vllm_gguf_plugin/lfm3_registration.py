# SPDX-License-Identifier: Apache-2.0
"""Register the upstream LFM3 backport with older vLLM installations."""


def register_lfm3():
    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry

    try:
        from vllm.transformers_utils.configs.lfm3 import Lfm3Config
    except ModuleNotFoundError as error:
        if error.name != "vllm.transformers_utils.configs.lfm3":
            raise
        # Existing GGUF architectures still work with vLLM predating LFM3.
        return False

    AutoConfig.register("lfm3", Lfm3Config, exist_ok=True)
    if "Lfm3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Lfm3ForCausalLM", "vllm.model_executor.models.lfm3:Lfm3ForCausalLM"
        )
    return True
