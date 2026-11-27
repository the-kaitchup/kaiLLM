"""Compare a multi-GPU Hy4 target with the pinned llama.cpp on the same checkpoint and cards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from benchmark import MANIFEST, full_graph_runtime, llama_revision, write_json
import cuda_graphs
import llama_graph
from devices import visible_cuda_uuids
from verify_gguf_loader import capture, compare
from worker import admission_fraction, gpu_inventory, selected_devices, stop_server, wait_server


GRAPH_CHECK_PROTOCOL = 1
# Cross the donor's 2,048-candidate dense/sparse boundary on a partial four-token group.
PREFIX_SIZES = (32, 127, 511, 2063)
MAX_MODEL_LEN = 3072


def run_capture(engine, command, env, output, bank, port, graph_path, uuids):
    log = output / (engine + ".log")
    url = f"http://127.0.0.1:{port}"
    with log.open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            wait_server(proc, url, log)
            if engine == "kaiLLM":
                before = cuda_graphs.snapshot(graph_path, uuids)
                head_uuids = [uid for uid in uuids if before[uid]["head_sizes"]]
                if head_uuids != uuids[-1:]:
                    raise ValueError("Hy4 output head must be graph-captured on the final pipeline rank")
            else:
                before = llama_graph.snapshot(graph_path)
            result = capture(engine, url, bank, "hy4", PREFIX_SIZES)
            phases = ("prefill", "decode")
            if engine == "kaiLLM":
                after = cuda_graphs.snapshot(graph_path, uuids)
                launches = cuda_graphs.verify_interval(before, after, phases)
                result["head_graph_launches"] = cuda_graphs.verify_head_interval(before, after, phases, head_uuids)
            else:
                after = llama_graph.snapshot(graph_path)
                verified = llama_graph.verify_interval(before, after, phases, uuids)
                launches = {phase: {uid: row["graph_launches"] for uid, row in rows.items()}
                            for phase, rows in verified.items()}
                result["reference_graph_counts"] = llama_graph.delta(before, after)
            result["graph_launches"] = launches
            write_json(output / (engine + ".json"), result)
            return result
        finally:
            stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=[k for k, m in MANIFEST["models"].items() if m["family"] == "hy4"])
    for name in ("weights", "banks", "out", "runtime", "python", "llama"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--port", type=int, default=18884)
    args = parser.parse_args()
    devices = selected_devices(gpu_inventory(), args.devices, visible_cuda_uuids())
    if len(devices) not in (2, 4) or len({d["name"] for d in devices}) != 1:
        raise ValueError("Select two or four GPUs of the same type")
    model = MANIFEST["models"][args.checkpoint]
    folder = args.weights / model["repo"] / model["revision"]
    receipt = json.loads((folder / ("download-" + args.checkpoint + ".json")).read_text())
    if receipt != {"repo": model["repo"], "revision": model["revision"], "source": "huggingface.co", "files": model["files"]}:
        raise ValueError("Hy4 GGUF provenance differs from the pinned checkpoint")
    files = [folder / name for name in model["files"]]
    if any(p.stat().st_size != model["files"][p.name]["bytes"] for p in files):
        raise ValueError("Incomplete Hy4 GGUF shard")
    pin = full_graph_runtime(args.checkpoint, "plain")
    if pin is None or (args.runtime / "benchmark-source.sha256").read_text().strip() != pin[1]:
        raise ValueError("A registered full-graph Hy4 pipeline source is required before model launch")
    if not all((args.runtime / "vllm_gguf_plugin" / name).is_file()
               for name in ("benchmark_graphs.py", "benchmark_output_head.py")):
        raise ValueError("Hy4 runtime lacks the target and output-head graph observers")
    ref = llama_revision("hy4")
    args.out.mkdir(parents=True, exist_ok=True)
    uuids = [d["uuid"] for d in devices]
    identity = {"checkpoint": args.checkpoint, "hf_revision": model["revision"], "files": model["files"],
                "devices": uuids, "kai_revision": pin[0], "kai_source_lock": pin[1], "llama_revision": ref}
    bank = json.loads((args.banks / "hy4.json").read_text())
    common = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids), OMP_NUM_THREADS="4")
    cpp = [str(args.llama), "-m", str(files[0]), "--host", "127.0.0.1", "--port", str(args.port),
           "--ctx-size", str(MAX_MODEL_LEN), "--parallel", "1", "-ngl", "999", "--fit", "off", "--flash-attn", "on",
           "--split-mode", "layer", "--cache-type-k", "bf16", "--cache-type-v", "bf16", "--cache-ram", "0",
           "--no-context-shift", "-b", "2048", "-ub", "512", "--spec-type", "none",
           "--override-tensor", r"^per_layer_token_embd\.weight$=CPU", "--load-mode", "mmap", "--lazy-mode", "on"]
    kai = [str(args.python), str(args.runtime / "deployment/serve.py"), str(files[0]), "--served-model-name", "bench",
           "--max-num-seqs", "1", "--max-model-len", str(MAX_MODEL_LEN), "--max-num-batched-tokens", "512",
           "--gpu-memory-utilization", str(admission_fraction(devices)), "--kv-cache-memory-bytes", str(1 << 30),
           "--spec-tokens", "0", "--pipeline-parallel-size", str(len(devices)), "--port", str(args.port),
           "--tokenizer", str(files[0])]
    proof_root = Path(tempfile.mkdtemp(prefix="graphs-", dir=args.out))
    kai_proof = proof_root / "kai"
    kai_proof.mkdir()
    kai_env = dict(common, KAILLM_NATIVE_ASSETS=str(args.runtime / "runtime-native-cache/native-assets.json"))
    kai_env[cuda_graphs.ENV] = str(kai_proof)
    preview = subprocess.run([*kai, "--dry-run", "--compute-capability", devices[0]["compute_cap"]],
                             env=kai_env, capture_output=True, check=True, text=True)
    cuda_graphs.validate_kai_launch(json.loads(preview.stdout)["command"])
    cpp_proof = proof_root / "llama.bin"
    cpp_env = {**common, **llama_graph.prepare(args.llama, cpp_proof, ref, len(uuids), base_env=common)}
    version = subprocess.check_output([str(args.llama), "--version"], env=cpp_env, stderr=subprocess.STDOUT, text=True)
    if ref[:7] not in version:
        raise ValueError("Incorrect Hy4 reference engine")
    identity["prefix_sizes"] = list(PREFIX_SIZES)
    write_json(args.out / "setup.json", {**identity, "kai_argv": kai, "llama_argv": cpp,
                                         "graph_check_protocol": GRAPH_CHECK_PROTOCOL})
    target = run_capture("kaiLLM", kai, kai_env, args.out, bank, args.port, kai_proof, uuids)
    reference = run_capture("llama.cpp", cpp, cpp_env, args.out, bank, args.port, cpp_proof, uuids)
    comparison = compare(reference, target)
    phase_counts = {capture["engine"]: {uid: {phase: capture["graph_launches"][phase][uid]
                                             for phase in ("prefill", "decode")} for uid in uuids}
                    for capture in (target, reference)}
    proof = {**identity, "protocol": GRAPH_CHECK_PROTOCOL, "per_device_graph_launches": phase_counts,
             "target_head_graph_launches": target["head_graph_launches"], "eager_steps": 0, "unclassified_steps": 0}
    write_json(args.out / "comparison.json", {**identity, **comparison, "graph_check_protocol": GRAPH_CHECK_PROTOCOL})
    write_json(args.out / "graph-only-validation.json", proof)



if __name__ == "__main__":
    main()
