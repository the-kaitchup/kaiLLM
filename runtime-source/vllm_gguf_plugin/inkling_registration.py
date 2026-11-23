# SPDX-License-Identifier: Apache-2.0
"""Register the Inkling 2 Mini (`inkling2_loop`) config and model with kaiLLM's vLLM."""

MODEL_MODULE = "vllm.model_executor.models.inkling2_loop"


def register_inkling(register_model: bool = True):
    from transformers import AutoConfig

    try:
        from vllm.transformers_utils.configs.inkling2_loop import Inkling2LoopConfig
    except ModuleNotFoundError as error:
        if error.name != "vllm.transformers_utils.configs.inkling2_loop":
            raise
        return False
    AutoConfig.register("inkling2_loop", Inkling2LoopConfig, exist_ok=True)
    if register_model:
        import importlib.util

        from vllm.model_executor.models import ModelRegistry

        if (importlib.util.find_spec(MODEL_MODULE) is not None
                and "Inkling2LoopForCausalLM" not in ModelRegistry.get_supported_archs()):
            ModelRegistry.register_model("Inkling2LoopForCausalLM", f"{MODEL_MODULE}:Inkling2LoopForCausalLM")
    return True
