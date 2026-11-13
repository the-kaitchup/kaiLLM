# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import ResolvedRevision, hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .quantization import GGUFConfig
from .weight_utils import download_gguf, resolve_local_gguf
from .weights_adapter import get_weights_adapter

logger = init_logger(__name__)


def _revision_for_weights_repo(
    model_config: ModelConfig, weights_repo_id: str
) -> str | None:
    """Return a revision that is valid for the weights repository.

    A ``ResolvedRevision`` is bound to the repository for which vLLM loaded
    the model config. If the GGUF weights live in another repository, reuse
    the user's original revision instead of that repository's resolved SHA.
    An explicitly requested SHA remains unchanged, while ``None`` lets the
    weights repository resolve its own default branch.
    """
    revision = model_config.revision
    if isinstance(revision, ResolvedRevision) and weights_repo_id != model_config.model:
        return revision.initial
    return revision


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=_revision_for_weights_repo(model_config, local_dir),
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _prepare_adapter(self, model_config: ModelConfig):
        local_model_path = self._prepare_weights(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        adapter.prepare_loading(local_model_path, model_config)
        return adapter

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from .native.runtime import validate_reload

        validate_reload(model_config)
        if any(hasattr(module, "yaqa_parts") for module in model.modules()):
            raise ValueError(
                "Reloading compact GGUF/YAQA weights requires a new engine instance."
            )
        adapter = self._prepare_adapter(model_config)
        adapter.prepare_model(model)
        model.load_weights(adapter.prepare_weights(model_config))

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        from .native.runtime import activate_for_model
        from .quantization import shared_body_prefill_cache, yaqa_prefill_cache

        if (
            yaqa_prefill_cache.mode() != "none"
            and yaqa_prefill_cache.rows_mode() == "generic-prefill"
        ) or shared_body_prefill_cache.settings()["mode"] != "none":
            # Distinct graph keys and dual-phase warmup are required whenever
            # operators specialize their prefill and decode implementations.
            vllm_config.additional_config["cudagraph_specialize_prefill"] = True
        native_selected = activate_for_model(vllm_config, model_config)
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        vllm_config.model_config.hf_config = model_config.hf_config
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )
        from .quantization.yaqa_linear import configure_yaqa_loading

        configure_yaqa_loading(vllm_config.quant_config, adapter.load_spec)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            adapter.prepare_model(model)
            consumed = vllm_config.quant_config.yaqa_consumed_tensors
            if consumed:
                mapping = adapter.load_spec.gguf_to_hf_name_map
                adapter.load_spec.gguf_to_hf_name_map = {
                    gguf_name: hf_name
                    for gguf_name, hf_name in mapping.items()
                    if gguf_name not in consumed
                }
            model.load_weights(
                adapter.prepare_weights(model_config),
            )
            process_weights_after_loading(model, model_config, target_device)
        from .quantization.shared_body_prefill_cache import prepare_model

        prepare_model(model, vllm_config=vllm_config)
        from .quantization.q6_precision import prepare_model as prepare_q6_precision

        prepare_q6_precision(model)
        from .quantization.ordinary_marlin_physical_prepare import (
            prepare_model as prepare_ordinary_marlin,
        )

        ordinary_plan = prepare_ordinary_marlin(model, vllm_config=vllm_config)
        if ordinary_plan is not None and ordinary_plan.ready:
            from .quantization.ordinary_marlin_physical import bind as bind_ordinary_marlin

            for layer in model.modules():
                state = getattr(layer, "_ordinary_marlin_physical", None)
                if state is not None:
                    bind_ordinary_marlin(layer, state)
        vllm_config.quant_config.yaqa_readers.clear()
        vllm_config.quant_config.yaqa_tensor_maps.clear()
        if native_selected:
            model.gguf_native_runtime = vllm_config.additional_config[
                "gguf_native_runtime"
            ]
        return model
