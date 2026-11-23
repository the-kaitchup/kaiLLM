# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .qwen35_metadata import (
    configure_qwen35_gguf_norm,
    prepare_qwen35_metadata,
)


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )
        if not is_gguf(original_model) and getattr(config, "dflash_config", None) is not None:
            # An HF draft checkpoint (DFlash / DSpine) next to a GGUF target inherits the target's config
            # format; it keeps its own architectures (the causal-LM mapping below would drop the draft head).
            return config_dict, config

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        if config.model_type == "lfm3":
            model_type = "Lfm3ForCausalLM"
        elif config.model_type == "hy_v4_flash":
            model_type = "HYV4FlashForCausalLM"
        elif config.model_type == "inkling2_loop":
            model_type = "Inkling2LoopForCausalLM"
        elif config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        else:
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config_dict["architectures"] = [model_type]
        config.update({"architectures": [model_type]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        if configure_qwen35_gguf_norm(config):
            text_dict = (
                config_dict.setdefault("text_config", {})
                if (config.get_text_config() is not config)
                else config_dict
            )
            text_dict["gguf_gdn_norm_dtype"] = (
                config.get_text_config().gguf_gdn_norm_dtype
            )
            text_dict["gguf_gdn_norm_impl"] = (
                config.get_text_config().gguf_gdn_norm_impl
            )

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            if metadata_path := prepare_qwen35_metadata(model):
                return metadata_path
            from .lfm3_metadata import prepare_lfm3_metadata

            if metadata_path := prepare_lfm3_metadata(model):
                return metadata_path
            from .hy4_metadata import prepare_hy4_metadata

            if metadata_path := prepare_hy4_metadata(model):
                return metadata_path
            from .inkling_metadata import prepare_inkling_metadata

            if metadata_path := prepare_inkling_metadata(model):
                return metadata_path
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
