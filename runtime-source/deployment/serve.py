"""Launch the retained frozen runtime with its measured hardware configuration."""

import json
from importlib import metadata
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_gguf_plugin import serve, serving_defaults, serving_policy as policy


def prepare(args, inherited, executable=sys.executable):
    capability = args.compute_capability or policy.discover_capability()
    architecture = {"8.6": "sm86", "12.0": "sm120"}.get(capability)
    if architecture is None:
        raise ValueError("The retained native assets support CUDA 8.6 and 12.0")
    package = ROOT / "vllm_gguf_plugin"
    # The benchmark starts with the same closed GGUF environment.
    inherited = {k: v for k, v in inherited.items()
                 if not k.startswith("VLLM_GGUF_")}
    selected = policy.load_target(args.catalog, capability, package, inherited)
    core_root = args.core_root or ROOT / "compatible-vllm/core"
    core = policy.verify_core_contract(selected["core_contract"], [str(core_root)])
    # GGUF-family defaults (tokenizer, memory plans, MTP) fill only unset options.
    family = serving_defaults.inspect(args.model)
    if family["architecture"] == "qwen35" and architecture != "sm120":
        raise ValueError("Qwen3.5/Qwen3.8 GGUFs are qualified on compute capability 12.0 only in this release")
    if args.tokenizer is None and family["tokenizer"] is not None:
        args.tokenizer = family["tokenizer"]
    prepared = serve.prepare_command(args, selected, executable, family=family)
    command, environment = prepared["command"], prepared["environment"]
    if (architecture == "sm86" and prepared["num_speculative_tokens"] == 3
            and prepared["speculative_method"] == "ngram_gpu"):
        # Preserve the exact Ampere configuration used in the frozen normal runs.
        index = command.index("--speculative-config") + 1
        spec = json.loads(command[index])
        spec.pop("ngram_gpu_single_active")
        spec.pop("ngram_gpu_single_active_adapter")
        if args.max_num_seqs == 1:
            # The legacy adapter's admitted contract is a single declared slot.
            spec.update(ngram_gpu_match_only=True,
                        ngram_gpu_match_only_library=selected["conditional_adapter"]["path"],
                        ngram_gpu_match_only_piecewise_prefill=True)
        command[index] = json.dumps(spec)
    if architecture == "sm86":
        environment["VLLM_GGUF_NATIVE_Q6_MMA_DEPENDENCY_DIR"] = str(
            package / "native/libraries/sm86/ggml"
        )
    # Do not inherit a temporary experiment's sitecustomize or package roots.
    # Third-party dependencies remain on the selected interpreter's normal path.
    environment["PYTHONPATH"] = os.pathsep.join(map(str, (
        ROOT / "deployment", ROOT, core_root, ROOT / "gguf-reader", ROOT / "third-party"
    )))
    environment["PATH"] = str(Path(executable).absolute().parent) + os.pathsep + environment.get("PATH", "")
    try:
        ninja = metadata.distribution("ninja")
    except metadata.PackageNotFoundError:
        ninja = None
    if ninja is not None:
        for entry in ninja.files or ():
            tool = Path(ninja.locate_file(entry))
            if tool.name == "ninja" and tool.is_file() and os.access(tool, os.X_OK):
                environment["PATH"] = str(tool.resolve().parent) + os.pathsep + environment["PATH"]
                break
    cuda_home = args.cuda_home or environment.get("CUDA_HOME")
    if cuda_home is None:
        cuda_home = next((str(Path(p) / "nvidia/cu13") for p in sys.path
                          if (Path(p) / "nvidia/cu13/bin/nvcc").is_file()), None)
    if cuda_home:
        cuda_home = str(Path(cuda_home).absolute())
        environment["CUDA_HOME"] = cuda_home
        environment["PATH"] = cuda_home + "/bin:" + environment.get("PATH", "")
        environment["LD_LIBRARY_PATH"] += ":" + cuda_home + "/lib"
    return {**prepared, "core_contract": core,
            "compute_capability": capability,
            "verified_native_assets": selected["verified_native_assets"]}


def main(argv=None):
    parser = serve.parser()
    parser.set_defaults(catalog=Path(__file__).with_name("serving-catalog.json"))
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--cuda-home")
    args = parser.parse_args(argv)
    if args.compute_capability and not args.dry_run:
        parser.error("--compute-capability is only available with --dry-run")
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    try:
        prepared = prepare(args, os.environ)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.dry_run:
        # Omit unrelated inherited environment values from logs.
        prepared["environment"] = {
            k: v for k, v in prepared["environment"].items()
            if os.environ.get(k) != v
        }
        print(json.dumps({**prepared, "runtime_started": False}, indent=2))
    else:
        os.execve(prepared["command"][0], prepared["command"], prepared["environment"])


if __name__ == "__main__":
    main()
