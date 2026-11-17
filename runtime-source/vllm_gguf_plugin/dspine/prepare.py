"""Prepare a DSpine drafter for ``--dspine``.

kaiLLM serves DSpine drafters as DSpineDraftModel with 4-bit linear layers.  A drafter in that format (e.g. the
dspine-b16-int4 folder of kaitchup/Qwen3.8-27B-chimachima-kaiPTQ) is used as it is.  A drafter in the original format
(e.g. tencent/Qwen3.8-27B-DSpine-b16, whose config names DFlashDraftModel; the DSpine fields live in its dflash_config)
is converted once into $KAILLM_CACHE_DIR/dspine (default ~/.cache/kaillm/dspine): its attention, MLP and feature (fc)
projections become 4-bit weights for vLLM's GPTQ-Marlin kernels, with symmetric round-to-nearest, groups of 128 input
columns, and per group the range among 8 candidates with the lowest squared error.  The code bridge, codebook, norms
and version scalars stay BF16.  The drafter only proposes tokens and the target model verifies all of them, so its
precision changes the acceptance rate, not the distribution of the outputs.  GPTQ v1 layout: qweight int32 [in/8, out],
qzeros int32 [in/g, out/8] (zero point 8 stored as 7), scales fp16 [in/g, out].
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

GROUP = 128
QUANTIZED = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
             "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
ARCHITECTURE = "DSpineDraftModel"
EXTRA_FILES = ("generation_config.json", "LICENSE")


def cache_root() -> Path:
    base = os.environ.get("KAILLM_CACHE_DIR")
    if not base:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "kaillm"
    return Path(base) / "dspine"


def is_prepared(directory: Path) -> bool:
    config = directory / "config.json"
    if not config.is_file() or not any(directory.glob("*.safetensors")):
        return False
    data = json.loads(config.read_text())
    return (ARCHITECTURE in (data.get("architectures") or [])
            and (data.get("quantization_config") or {}).get("quant_method") == "gptq")


def resolve(drafter: str, prepare: bool = True) -> Path:
    """Return the prepared checkpoint directory, converting the drafter once if needed.

    DRAFTER is a local directory, a Hugging Face repository, or REPOSITORY:FOLDER for a drafter stored in a folder
    of a repository (e.g. kaitchup/Qwen3.8-27B-chimachima-kaiPTQ:dspine-b16-int4).  Nothing is downloaded or
    converted when prepare is false (dry runs).
    """
    local = Path(drafter).expanduser()
    source = None
    if local.is_dir():
        source = local.resolve()
        name = f"{source.name}-{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
    else:
        repository, _, folder = drafter.strip("/").partition(":")
        if prepare:
            source = download(repository, folder.strip("/"))
        name = drafter.strip("/").replace("/", "--").replace(":", "--")
    if source is not None and is_prepared(source):
        return source
    target = cache_root() / f"{name}-int4"
    if prepare and not is_prepared(target):
        build(source, target)
    return target


def download(repository: str, folder: str = "") -> Path:
    from huggingface_hub import snapshot_download

    prefix = folder + "/" if folder else ""
    patterns = [prefix + name for name in ("config.json", "*.safetensors", *EXTRA_FILES)]
    path = Path(snapshot_download(repository, allow_patterns=patterns)) / folder
    if not (path / "config.json").is_file():
        raise ValueError(f"No drafter config.json in {repository}" + (f" (folder {folder})" if folder else ""))
    return path


def quantize(base: str, weight) -> dict:
    """weight [out, in] -> GPTQ v1 tensors base.qweight / base.scales / base.qzeros (4-bit, symmetric, zero 8)."""
    import torch

    out, columns = weight.shape
    if columns % GROUP or out % 8:
        raise ValueError(f"{base}: shape {tuple(weight.shape)} does not fit 4-bit groups of {GROUP}")
    groups = weight.float().reshape(out, columns // GROUP, GROUP)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_error = best_scale = None
    for fraction in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
        scale = amax * fraction / 7.5
        error = ((torch.clamp(torch.round(groups / scale), -8, 7) * scale - groups) ** 2).sum(-1, keepdim=True)
        if best_error is None:
            best_error, best_scale = error, scale
        else:
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, scale, best_scale)
    codes = (torch.clamp(torch.round(groups / best_scale), -8, 7) + 8).to(torch.int64).reshape(out, columns)
    rows = codes.t().contiguous().reshape(columns // 8, 8, out)   # 8 consecutive input columns per int32 word
    packed = torch.zeros(columns // 8, out, dtype=torch.int64)
    for k in range(8):
        packed |= rows[:, k, :] << (4 * k)                         # lowest nibble first
    packed = (packed - (packed >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
    return {
        base + ".qweight": packed,
        base + ".scales": best_scale.reshape(out, columns // GROUP).t().contiguous().to(torch.float16),
        base + ".qzeros": torch.full((columns // GROUP, out // 8), 0x77777777, dtype=torch.int32),
    }


def build(source: Path, target: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text())
    if "dflash_config" not in config:
        raise ValueError(f"{source} is not a DSpine drafter checkpoint (its config has no dflash_config)")
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"{source} has no safetensors weights")
    print(f"kaiLLM: converting the DSpine drafter {source} to 4-bit layers in {target}", flush=True)
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    tensors = {}
    with torch.inference_mode():
        for path in weights:
            with safe_open(str(path), "pt") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    if name == "fc.weight" or (name.startswith("layers.") and name.endswith(QUANTIZED)):
                        tensors.update(quantize(name[: -len(".weight")], tensor))
                    else:
                        tensors[name] = tensor.contiguous()
    save_file(tensors, str(partial / "model.safetensors"), metadata={"format": "pt"})
    # Substrings, so they match the drafter's layers whatever their global index (vLLM numbers them after the target's
    # layers); the merged names too: DFlash passes no fused-module mapping to the GPTQ check.
    config["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": GROUP, "desc_act": False, "sym": True,
        "lm_head": False, "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": [name[: -len(".weight")] for name in QUANTIZED]
        + ["self_attn.qkv_proj", "mlp.gate_up_proj", "fc"],
    }
    config["architectures_hf"] = config.get("architectures")
    config["architectures"] = [ARCHITECTURE]
    (partial / "config.json").write_text(json.dumps(config, indent=2))
    for name in EXTRA_FILES:
        if (source / name).is_file():
            shutil.copy(source / name, partial / name)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
