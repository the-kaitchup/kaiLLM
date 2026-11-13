# SPDX-License-Identifier: Apache-2.0

import os
from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gguf_utils import check_gguf_file, is_gguf, is_remote_gguf, split_remote_gguf
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .qwen35_metadata import (
    configure_qwen35_gguf_norm,
    prepare_qwen35_metadata,
)
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return repo_id
    if check_gguf_file(model):
        return str(Path(model).parent)
    return model


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            metadata_path = None
            metadata_family = "Qwen3.5"
            if check_gguf_file(gguf_model):
                from .native.policy import configure_engine_args

                configure_engine_args(self, gguf_model)
                metadata_path = prepare_qwen35_metadata(gguf_model)
                if metadata_path is None:
                    from .lfm3_metadata import prepare_lfm3_metadata

                    metadata_path = prepare_lfm3_metadata(gguf_model)
                    metadata_family = "LFM3"
            if metadata_path is not None:
                from .qwen35_defaults import configure_local_qwen35_defaults

                if metadata_family == "Qwen3.5":
                    configure_local_qwen35_defaults(self)
                else:
                    configure_local_qwen35_defaults(self, model_family="LFM3")
                if self.hf_config_path is None:
                    self.hf_config_path = metadata_path
                if self.tokenizer is None or (
                    isinstance(self.tokenizer, str)
                    and _is_gguf_reference(self.tokenizer)
                ):
                    self.tokenizer = metadata_path
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
        if os.getenv("VLLM_GGUF_YAQA_QKV_CACHE_MERGE", "0") == "1":
            compilation = self.compilation_config
            compile_mode = (
                compilation.get("mode")
                if isinstance(compilation, dict)
                else getattr(compilation, "mode", None)
            )
            if compile_mode != 0:
                raise ValueError(
                    "QKV cache merge currently requires explicit compilation mode0"
                )
        model_config = original_create_model_config(self, *args, **kwargs)
        from .quantization.native_bf16_sm120 import mode as native_sm120_mode

        if native_sm120_mode() != "none":
            compilation = self.compilation_config
            native_compile_mode = (
                compilation.get("mode")
                if isinstance(compilation, dict)
                else getattr(compilation, "mode", None)
            )
            from .quantization.native_bf16_sm120 import validate_profile

            validate_profile(
                native_compile_mode,
                str(model_config.dtype),
                getattr(self, "max_num_seqs", None),
            )
        if self.quantization == "gguf" or self.load_format == "gguf":
            # Explicit HF config paths and config_format="hf" also load GGUF
            # weights through our adapter. Stamp before engine/AOT cache hashes.
            configure_qwen35_gguf_norm(model_config.hf_config)
            model_config.hf_config.get_text_config().gguf_yaqa_gemm = os.getenv(
                "VLLM_GGUF_YAQA_GEMM", "tf32x3"
            )
            model_config.hf_config.get_text_config().gguf_yaqa_head_output = "float32"
            from .quantization.native_bf16_q3 import runtime_marker as q3_marker
            from .quantization.native_bf16_sm120 import runtime_marker as sm120_marker

            model_config.hf_config.get_text_config().gguf_native_sm120_b1 = (
                sm120_marker()
            )
            model_config.hf_config.get_text_config().gguf_native_bf16_q3_head = (
                q3_marker()
            )
            from .quantization.native_head import native_head_cache_marker

            model_config.hf_config.get_text_config().gguf_lfm3_head = (
                native_head_cache_marker()
            )
            from .quantization.native_body import native_body_cache_marker

            model_config.hf_config.get_text_config().gguf_native_body = (
                native_body_cache_marker()
            )
            from .quantization.native_bf16 import runtime_marker as bf16_native_marker

            model_config.hf_config.get_text_config().gguf_native_bf16_adapter = (
                bf16_native_marker()
            )
            from .quantization.q6_precision import runtime_marker as q6_precision_marker

            model_config.hf_config.get_text_config().gguf_q6_precision = (
                q6_precision_marker()
            )
            from .quantization.shared_body_prefill_cache import (
                runtime_marker as body_cache_marker,
            )

            model_config.hf_config.get_text_config().gguf_shared_body_prefill_cache = (
                body_cache_marker()
            )
            from .quantization.yaqa_prefill_cache import (
                runtime_marker as prefill_cache_marker,
            )

            model_config.hf_config.get_text_config().gguf_yaqa_prefill_cache = (
                prefill_cache_marker(
                    parallel=(
                        getattr(self, "tensor_parallel_size", 1),
                        getattr(self, "pipeline_parallel_size", 1),
                        getattr(self, "data_parallel_size", 1) or 1,
                    )
                )
            )
            from .quantization.yaqa_cuda import cuda_k2_runtime_marker

            model_config.hf_config.get_text_config().gguf_yaqa_gemv = (
                cuda_k2_runtime_marker()
            )
            from .quantization.yaqa_mma import register_mma_runtime_marker

            model_config.hf_config.get_text_config().gguf_yaqa_mma = (
                register_mma_runtime_marker()
            )
            import json

            from .quantization.yaqa_residual_mma import (
                runtime_marker as residual_marker,
            )

            original_mma_identity = json.loads(
                model_config.hf_config.get_text_config().gguf_yaqa_mma
            )
            original_mma_target = tuple(
                original_mma_identity.get("architecture", {}).get(
                    "compute_capability", ()
                )
            )
            model_config.hf_config.get_text_config().gguf_yaqa_register_precision = (
                residual_marker(
                    target=original_mma_target
                    if len(original_mma_target) == 2
                    else None
                )
            )
            from .quantization.yaqa_residual_epilogue import (
                runtime_marker as epilogue_marker,
            )

            model_config.hf_config.get_text_config().gguf_yaqa_register_epilogue = (
                epilogue_marker(target=original_mma_target)
            )
            from .quantization.yaqa_grouped import runtime_marker as grouped_marker

            model_config.hf_config.get_text_config().gguf_yaqa_grouped = (
                grouped_marker()
            )
            from .quantization.yaqa_grouped_hin import (
                runtime_marker as grouped_hin_marker,
            )

            model_config.hf_config.get_text_config().gguf_yaqa_grouped_hin = (
                grouped_hin_marker()
            )
            from .quantization.yaqa_grouped_power_input_resource import (
                runtime_marker as grouped_power_resource_marker,
            )

            grouped_text_config = model_config.hf_config.get_text_config()
            grouped_identity = grouped_text_config.gguf_yaqa_grouped_hin
            grouped_text_config.gguf_yaqa_grouped_power_input_resource = (
                grouped_power_resource_marker(
                    target=grouped_identity.get("target"),
                    legacy_identity=grouped_identity.get("library"),
                )
            )
            from .quantization.yaqa_index40 import runtime_marker as index40_marker

            model_config.hf_config.get_text_config().gguf_yaqa_index40 = index40_marker(
                parallel=(
                    getattr(self, "tensor_parallel_size", 1),
                    getattr(self, "pipeline_parallel_size", 1),
                    getattr(self, "data_parallel_size", 1) or 1,
                ),
                activation_dtype=str(model_config.dtype),
                compilation=getattr(self, "compilation_config", None),
            )
            from .quantization.yaqa_prefill import transient_prefill_runtime_marker

            model_config.hf_config.get_text_config().gguf_yaqa_prefill = (
                transient_prefill_runtime_marker()
            )
            from .quantization.yaqa_power import power_hadamard_runtime_marker

            model_config.hf_config.get_text_config().gguf_yaqa_hadamard = (
                power_hadamard_runtime_marker()
            )
            from .quantization.yaqa_power_input_resource import (
                runtime_marker as power_resource_marker,
            )

            power_identity = model_config.hf_config.get_text_config().gguf_yaqa_hadamard
            power_target = tuple(
                power_identity.get("architecture", {}).get("compute_capability", ())
            )
            model_config.hf_config.get_text_config().gguf_yaqa_power_input_resource = (
                power_resource_marker(
                    target=power_target, legacy_identity=power_identity
                )
            )
            from .quantization.yaqa_hin_high_low import (
                runtime_marker as hin_hilo_marker,
            )

            hin_hilo = hin_hilo_marker(target=power_target)
            model_config.hf_config.get_text_config().gguf_yaqa_hin_high_low = hin_hilo
            if hin_hilo["active"]:
                self.additional_config = dict(self.additional_config or {})
                self.additional_config["cudagraph_specialize_prefill"] = True
            from .quantization.yaqa_raw_factor_post import runtime_marker as raw_post_marker

            model_config.hf_config.get_text_config().gguf_yaqa_raw_factor_post = (
                raw_post_marker(target=power_target)
            )
            from .quantization.yaqa_factor_storage import (
                runtime_marker as factor_storage_marker,
            )

            model_config.hf_config.get_text_config().gguf_yaqa_factor_storage = (
                factor_storage_marker(target=power_target)
            )
            from .quantization.lfm3_dual_rms import runtime_marker as dual_rms_marker

            dual_rms = dual_rms_marker(target=power_target)
            model_config.hf_config.get_text_config().gguf_lfm3_dual_rms = dual_rms
            if dual_rms["active"]:
                self.additional_config = dict(self.additional_config or {})
                self.additional_config["cudagraph_specialize_prefill"] = True
            from .quantization.yaqa_scalar import scalar_runtime_marker

            model_config.hf_config.get_text_config().gguf_yaqa_scalar = (
                scalar_runtime_marker()
            )
        if isinstance(self.additional_config, dict) and self.additional_config.get(
            "gguf_native_runtime"
        ):
            from .native.runtime import validate_model_config

            validate_model_config(
                model_config,
                self.additional_config["gguf_native_runtime"]["model_sha256"],
            )
        return model_config

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_mtp_draft_config() -> None:
    """MTP drafters of GGUF targets read the prepared HF metadata and the same GGUF.

    vLLM builds the MTP draft ModelConfig from the target's model_weights (the raw
    .gguf path), which has no HF config or processor files. Point the draft at the
    target's prepared metadata directory and keep the GGUF as its weights source.
    """
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, "_gguf_mtp_patched", False):
        return
    original = SpeculativeConfig.__post_init__

    @wraps(original)
    def __post_init__(self):
        target = getattr(self, "target_model_config", None)
        weights = getattr(target, "model_weights", None) if target is not None else None
        gguf_target = (
            self.model is None
            and self.method == "mtp"
            and target is not None
            and _is_gguf_reference(weights)
            and not _is_gguf_reference(target.model)
        )
        if gguf_target:
            self.model = target.model
        original(self)
        if gguf_target and getattr(self, "draft_model_config", None) is not None:
            self.draft_model_config.model_weights = weights

    SpeculativeConfig.__post_init__ = __post_init__
    SpeculativeConfig._gguf_mtp_patched = True


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    from .lfm3_registration import register_lfm3

    register_lfm3()
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    from .native.completion_batch import install_completion_batch_hook
    from .native.runtime import install_scheduler_hook

    install_scheduler_hook()
    install_completion_batch_hook()
    _patch_engine_args()
    _patch_speculator_probe()
    _patch_mtp_draft_config()
    _patch_diffusers_loader()
