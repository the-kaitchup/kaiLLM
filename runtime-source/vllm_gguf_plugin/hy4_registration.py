# SPDX-License-Identifier: Apache-2.0
"""Register the Hy4 flash (`hy_v4_flash`) config and model with kaiLLM's vLLM."""

MODEL_MODULE = "vllm.model_executor.models.hy_v4_flash"


def register_hy4(register_model: bool = True):
    from transformers import AutoConfig

    try:
        from vllm.transformers_utils.configs.hy_v4_flash import HYV4FlashConfig
    except ModuleNotFoundError as error:
        if error.name != "vllm.transformers_utils.configs.hy_v4_flash":
            raise
        return False
    AutoConfig.register("hy_v4_flash", HYV4FlashConfig, exist_ok=True)
    if register_model:
        import importlib.util

        from vllm.model_executor.models import ModelRegistry

        if (importlib.util.find_spec(MODEL_MODULE) is not None
                and "HYV4FlashForCausalLM" not in ModelRegistry.get_supported_archs()):
            ModelRegistry.register_model("HYV4FlashForCausalLM", f"{MODEL_MODULE}:HYV4FlashForCausalLM")
    return True
