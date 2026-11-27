"""Check optimized Inkling plain serving against its same-device production baseline."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import kai_default
from benchmark import MANIFEST, completion, verify_prompt_bank, write_json
from cuda_graphs import ENV as STRICT
from devices import visible_cuda_uuids
from hybrid_execution import verify_files
from verify_gguf_loader import predict
from worker import admission_fraction, checkpoint_files, gpu_inventory, reported_capacity, selected_devices, stop_server, wait_server

MODEL = "inkling-loop-iq4"
BASELINE = "a8d3a360b1273410799e1a01b6c00354db7bad49"
BASELINE_LOCK = "9871df199af56a02f409d22fff7f01d20e1926e5fddd28dbb0911469e64e17b7"
SCORED = 32
GENERATED = 129
DOMAINS = ("prose", "code")


def checked_sources(runtime, plan):
    lock = (runtime / "benchmark-source.sha256").read_text().strip()
    if lock != plan["source_lock"]:
        raise ValueError("Installed Inkling source lock differs")
    for rel, expected in plan["files"].items():
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid runtime source path")
        target = runtime / ("compatible-vllm/core" if path.parts[0] == "vllm-overrides" else "") / (
            Path(*path.parts[1:]) if path.parts[0] == "vllm-overrides" else path)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified runtime file: {rel}")
    if plan["role"] == "baseline" and (plan["source_revision"], lock) != (BASELINE, BASELINE_LOCK):
        raise ValueError("This is not the ordinary production baseline")


def model_wave(observer, gpu, function, *, chunks, decoded):
    before = kai_default.read(observer, [gpu])
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    after = kai_default.read(observer, [gpu])
    counts = kai_default.interval(before, after, ("prefill","decode"))[gpu]
    pre, dec = counts["prefill"], counts["decode"]
    if (pre["direct"] != chunks or pre["full"] or pre["other"]
            or dec["full"] != decoded or dec["direct"] or dec["other"] or dec["direct_head"] != decoded):
        raise ValueError(f"Actual warm-cache or dispatch policy differs: {counts}")
    return result, counts, elapsed


def compare_domains(baseline, optimized):
    if baseline["match"] != optimized["match"]:
        raise ValueError("The scored hardware, model, environment or workload changed")
    outputs = {}
    for domain in DOMAINS:
        old, new = baseline["domains"][domain], optimized["domains"][domain]
        ref, candidate = old["fixed_prefix"], new["fixed_prefix"]
        if ref["prompt_sha256"] != candidate["prompt_sha256"]:
            raise ValueError("The score captures used different prompt bytes")
        exact_tokens, exact_scores, shared_max = 0, 0, 0.
        match_length, preceding_equal = 0, True
        first_difference = None
        for i in range(SCORED):
            a, b = ref["tokens"][i], candidate["tokens"][i]
            pa, pb = ref["top_logprobs"][i], candidate["top_logprobs"][i]
            if len(pa) != 8 or len(pb) != 8:
                raise ValueError("Incomplete top-eight comparison")
            if preceding_equal:
                exact_tokens += a == b
                exact_scores += pa == pb
                common = pa.keys() & pb.keys()
                shared_max = max([shared_max] + [abs(pa[t]-pb[t]) for t in common])
                if first_difference is None and (a != b or pa != pb):
                    first_difference = {"position":i,"baseline_token":a,"candidate_token":b,
                                        "baseline_top":pa,"candidate_top":pb}
                if a == b:
                    match_length += 1
                else:
                    preceding_equal = False
        same_timed_output = old["timing"]["output_sha256"] == new["timing"]["output_sha256"]
        outputs[domain] = {"greedy_shared_prefix":match_length,"exact_token_choices_on_shared_history":exact_tokens,
                           "exact_top8_steps_on_shared_history":exact_scores,"max_logprob_delta_on_shared_ids":shared_max,
                           "first_score_or_token_difference":first_difference,"same_129_token_output_hash":same_timed_output,
                           "passed":match_length==SCORED and exact_scores==SCORED and same_timed_output,
                           "baseline_rates":old["rates"],"candidate_rates":new["rates"]}
    return {"passed":all(d["passed"] for d in outputs.values()),"domains":outputs}


def run(args):
    plan=json.loads(args.plan.read_bytes())
    checked_sources(args.runtime,plan)
    dev,=selected_devices(gpu_inventory(),[0],visible_cuda_uuids())
    if dev["uuid"] != plan["gpu_uuid"]:
        raise ValueError("Wrong physical GPU")
    model=MANIFEST["models"][args.checkpoint]
    target,=checkpoint_files(args.weights, args.checkpoint, model)
    digest,=verify_files([target],model)
    bank=json.loads((args.banks/"inkling.json").read_bytes())
    for domain in DOMAINS:
        verify_prompt_bank(bank,"inkling",domain)
    args.out.mkdir(parents=True,exist_ok=False)
    observer=args.out/"dispatch"
    observer.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev["uuid"],KAILLM_HUMMING_DETERMINISTIC="1",
             KAILLM_NATIVE_ASSETS=str(args.runtime/"runtime-native-cache/native-assets.json"),
             OMP_NUM_THREADS=str(min(8,max(1,len(os.sched_getaffinity(0))//2))))
    for key in [k for k in env if k.startswith("KAILLM_INKLING_")]:
        del env[key]
    settings = plan.get("kernel_environment", {})
    if plan["role"] == "baseline" and settings:
        raise ValueError("The ordinary baseline cannot select optimized kernels")
    for key, value in settings.items():
        if not re.fullmatch(r"KAILLM_INKLING_[A-Z0-9_]+", key) or not re.fullmatch(r"[a-zA-Z0-9_,-]{1,64}", value):
            raise ValueError(f"Invalid Inkling kernel selector {key}")
        env[key] = value
    if args.checkpoint == MODEL and plan["role"] == "candidate" and plan.get("require_grouped_q8", False) and env.get("KAILLM_INKLING_FUSED_Q8") != "1":
        raise ValueError("The combined candidate must use fused ordinary projections")
    env.pop(STRICT,None)
    env[kai_default.ENV]=str(observer)
    extra=["--enable-prompt-tokens-details","--enable-per-request-metrics","--no-enable-prefix-caching"]
    cmd=[sys.executable,str(args.runtime/"deployment/serve.py"),str(target),"--served-model-name","bench",
         "--spec-tokens","0","--max-num-seqs","1","--max-model-len","73728","--max-num-batched-tokens","2048",
         "--kv-cache-memory-bytes",str(args.kv_bytes),"--gpu-memory-utilization",str(admission_fraction([dev])),
         "--port",str(args.port),"--extra-vllm-args",json.dumps(extra)]
    argv=json.loads(subprocess.check_output([*cmd,"--dry-run"],env=env))["command"]
    kai_default.launch(argv)
    if "--profiler-config" in argv:
        raise ValueError("Timing requires unprofiled serving")
    match={"checkpoint_revision":model["revision"],"checkpoint_sha256":digest,"gpu_uuid":dev["uuid"],
           "driver":dev["driver_version"],"power_limit":dev["power.limit"],"gpu_name":dev["name"],
           "context":73728,"bf16_kv_bytes":args.kv_bytes,"microbatch":2048,"batch":1,"prompt_tokens":8192,
           "scored":SCORED,"generated":GENERATED,"humming_deterministic":True,"seed":42,"temperature":0,
           "capture_policy":"FULL_DECODE_ONLY","prefix_cache":False,"omp_threads":int(env["OMP_NUM_THREADS"])}
    raw={"match":match,"role":plan["role"],"source_revision":plan["source_revision"],"source_lock":plan["source_lock"],
         "argv":argv,"domains":{}}
    log=args.out/"server.log"
    url=f"http://127.0.0.1:{args.port}"
    with log.open("w") as stream:
        proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        wait_server(proc,url,log)
        capacity=reported_capacity(log.read_text(errors="replace"))
        if capacity is None or capacity<1:
            raise ValueError("Full context cannot fit in the requested KV allocation")
        raw["reported_capacity"]=capacity
        warm=bank["domains"]["prose"]["rows"][32][:256]
        completion("kaiLLM",url,warm,8,120)
        observed=kai_default.read(observer,[dev["uuid"]])
        pid=observed[dev["uuid"]][0]
        record=dict(item.split("=",1) for item in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in item)
        expected=dict(settings, KAILLM_HUMMING_DETERMINISTIC="1", KAILLM_INKLING_FUSED_Q8=env.get("KAILLM_INKLING_FUSED_Q8"))
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError("The real engine worker did not inherit the requested projection policy")
        raw["observed_environment"]=expected
        write_json(args.out/"launch.json",raw)
        for domain in DOMAINS:
            prompt=bank["domains"][domain]["rows"][0]
            fixed,score_counts,_=model_wave(observer,dev["uuid"],
                lambda:predict("kaiLLM",url,prompt,output_tokens=SCORED,timeout=600),chunks=4,decoded=SCORED-1)
            if any(len(row)!=8 or not all(math.isfinite(score) for score in row.values()) for row in fixed["top_logprobs"]):
                raise ValueError("Missing finite top-eight scores")
            timing,time_counts,elapsed=model_wave(observer,dev["uuid"],
                lambda:completion("kaiLLM",url,prompt,GENERATED,timeout=600),chunks=4,decoded=GENERATED-1)
            if timing["cache_hits"]!=0 or timing["output_tokens"]!=GENERATED or timing["prompt_tokens"]!=8192:
                raise ValueError("Timed request token counts or prefix-cache usage differ")
            metrics=timing["server_timings"]
            ttft,decode=metrics["time_to_first_token_ms"],metrics["generation_time_ms"]
            if not (math.isfinite(ttft) and math.isfinite(decode) and ttft>0 and decode>0):
                raise ValueError("Missing server phase metrics")
            rates={"prefill_tokens_per_second":8192e3/ttft,"decode_tokens_per_second":(GENERATED-1)*1e3/decode}
            raw["domains"][domain]={"fixed_prefix":fixed,"score_dispatch":score_counts,"timing":timing,
                                    "timing_dispatch":time_counts,"client_seconds":elapsed,"rates":rates}
            write_json(args.out/(domain+".json"),raw["domains"][domain])
        write_json(args.out/"capture.json",raw)
        if args.baseline is not None:
            comparison=compare_domains(json.loads(args.baseline.read_bytes()),raw)
            write_json(args.out/"comparison.json",comparison)
            if not comparison["passed"]:
                raise ValueError("Full-model fixed-prefix or deterministic output parity failed; receipts preserved")
    finally:
        stop_server(proc)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("runtime","weights","banks","out","plan"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--checkpoint",choices=(MODEL,"inkling-udq2"),default=MODEL)
    parser.add_argument("--kv-bytes",type=int,default=4<<30)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--port",type=int,default=18886)
    run(parser.parse_args())
