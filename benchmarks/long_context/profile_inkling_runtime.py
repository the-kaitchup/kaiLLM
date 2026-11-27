"""Profile actual GPU kernels of plain Inkling under its standard production dispatch."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import kai_default
from benchmark import MANIFEST, completion, hash_tokens, verify_prompt_bank, write_json
from cuda_graphs import ENV as STRICT
from devices import visible_cuda_uuids
from hybrid_execution import verify_files
from worker import admission_fraction, checkpoint_files, gpu_inventory, reported_capacity, selected_devices, stop_server, wait_server


def profile_inputs(bank, domain, prefix):
    verify_prompt_bank(bank, 'inkling', domain)
    rows = bank['domains'][domain]['rows']
    if prefix == 8192:
        return rows[0]
    if prefix == 65536:
        # These eight windows are only for inspecting a real long KV cache.
        return [token for index in range(0,32,4) for token in rows[index]]
    raise ValueError('This comparison requires an 8192- or 65536-token KV prefix')


def profile_request(url, endpoint, payload=None):
    req = urllib.request.Request(url+endpoint, data=json.dumps(payload or {}).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        if response.status != 200:
            raise ValueError('Profiler endpoint refused the request')


def kernel_summary(database):
    """Sum exclusive real-device kernel durations. An API/capture symbol cannot replace actual kernels."""
    kernels, calls, scope_ns = Counter(), Counter(), Counter()
    roots = [item for item in database if isinstance(item, dict) and 'frame' in item]
    if not roots:
        raise ValueError('Missing Proton Hatchet roots')

    def walk(node, scope):
        name = node['frame']['name']
        if name.startswith('execute_context_'):
            scope = name
        metrics = node.get('metrics', {})
        duration = metrics.get('time (ns)', 0)
        # Proton GPU records carry physical device metadata. Parent scope totals
        # are synthesized by viewers and must not be added to their children.
        if 'device_id' in metrics and duration:
            if (not isinstance(duration, (int,float)) or not math.isfinite(duration) or duration<=0
                    or type(metrics.get('count')) is not int or metrics['count']<=0):
                raise ValueError('Invalid CUPTI kernel counter')
            kernels[name] += duration
            calls[name] += metrics['count']
            scope_ns[scope] += duration
        for child in node.get('children', []):
            walk(child,scope)

    for root in roots:
        walk(root,'outside worker model annotation')
    if len(kernels)<4 or not any('rel_attn' in name for name in kernels):
        raise ValueError('CUPTI trace did not expose actual Inkling attention and component kernels')
    total = sum(kernels.values())
    return {'device_kernel_ns':total,'device_kernel_count':sum(calls.values()),'scope_kernel_ns':dict(scope_ns),
            'kernels':[{'name':name,'calls':calls[name],'total_ns':ns,'share':ns/total} for name,ns in kernels.most_common()]}


def profile_wave(url, traces, observer, uuid, bank, domain, out, prefix, phase):
    prompt=profile_inputs(bank,domain,prefix)
    if len(prompt)!=prefix:
        raise ValueError('KV prefix length differs')
    chunks=prefix//2048
    generated=1 if phase=='prefill' else 129
    # vLLM calls WorkerProfiler.step before each model step. On B1 the
    # first sampled token comes from the last prefill; the next 128 are decode.
    delay=0 if phase=='prefill' else chunks+1
    for f in traces.glob('*.hatchet'):
        if f.stat().st_size==0:raise ValueError('Empty preceding profiler output')
    prior_files=set(traces.glob('*.hatchet'))
    first=kai_default.read(observer,[uuid])
    profile_request(url,'/start_profile',{'profile_prefix':out.stem,'delay_iterations':delay,'max_iterations':chunks if phase=='prefill' else 128})
    try:
        req=completion('kaiLLM',url,prompt,generated,timeout=2700)
    finally:
        profile_request(url,'/stop_profile')
    last=kai_default.read(observer,[uuid])
    required=('prefill',) if phase=='prefill' else ('prefill','decode')
    actual=kai_default.interval(first,last,required)[uuid]
    if (actual['prefill']['full']+actual['prefill']['direct']!=chunks
            or actual['decode']['full']!=generated-1 or actual['decode']['direct']!=0):
        raise ValueError('Physical model dispatch differs from the expected profile phase boundaries')
    files=set(traces.glob('*.hatchet'))-prior_files
    if len(files)!=1:
        raise ValueError(f'Expected a single new CUPTI worker trace, got {sorted(map(str,files))}')
    trace=files.pop()
    raw=trace.read_bytes()
    summary=kernel_summary(json.loads(raw))
    for scope,ns in summary['scope_kernel_ns'].items():
        if scope.startswith('execute_context_'):
            if phase=='decode' and not scope.startswith('execute_context_0(0)_generation_1(1)'):
                raise ValueError(f'Prefill contaminated the decode profile: {scope}')
            if phase=='prefill' and not scope.startswith('execute_context_1(2048)_generation_0(0)'):
                raise ValueError(f'Decode contaminated the prefill profile: {scope}')
    if not any(name.startswith('execute_context_') for name in summary['scope_kernel_ns']):
        raise ValueError('The profile did not attach physical kernels to this worker operation')
    write_json(out,{'phase':phase,'prefix_tokens':prefix,'measurement_iterations':chunks if phase=='prefill' else 128,
               'prompt_sha256':hash_tokens(prompt),'window_indices':[0] if prefix==8192 else list(range(0,32,4)),
               'response':req,'gpu_uuid':uuid,'dispatch':actual,'trace':trace.name,'trace_sha256':hashlib.sha256(raw).hexdigest(),**summary})


def main(args):
    devices=selected_devices(gpu_inventory(),[args.device],visible_cuda_uuids())
    dev,=devices
    if args.expect_uuid and dev['uuid']!=args.expect_uuid:
        raise ValueError('Selected GPU differs from the requested physical device')
    model=MANIFEST['models'][args.checkpoint]
    pin=MANIFEST['kaillm']['default_dispatch_runtime']
    lock=(args.runtime/'benchmark-source.sha256').read_text().strip()
    if lock!=pin['source'] or args.checkpoint not in pin['models']:
        raise ValueError('The installed source has no matching ordinary Inkling policy')
    target,=checkpoint_files(args.weights,args.checkpoint,model)
    digest,=verify_files([target],model)
    bank=json.loads((args.banks/'inkling.json').read_bytes())
    args.out.mkdir(parents=True,exist_ok=False)
    traces=args.out/'traces';observer=args.out/'dispatch'
    traces.mkdir();observer.mkdir()
    profile_config={'profiler':'proton','proton_profiler_dir':str(traces),'proton_context':'shadow','proton_backend':'cupti',
                    'proton_data':'tree','proton_output_format':'hatchet','proton_graph_attribution':True}
    extra=['--enable-prompt-tokens-details','--enable-per-request-metrics','--no-enable-prefix-caching','--profiler-config',json.dumps(profile_config)]
    cmd=[sys.executable,str(args.runtime/'deployment/serve.py'),str(target),'--served-model-name','bench','--spec-tokens','0',
         '--max-num-seqs','1','--max-model-len','73728','--max-num-batched-tokens','2048','--kv-cache-memory-bytes',str(args.kv_bytes),
         '--gpu-memory-utilization',str(admission_fraction(devices)),'--port',str(args.port),'--extra-vllm-args',json.dumps(extra)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev['uuid'],KAILLM_HUMMING_DETERMINISTIC='1',
             KAILLM_NATIVE_ASSETS=str(args.runtime/'runtime-native-cache/native-assets.json'),
             OMP_NUM_THREADS=str(min(8,max(1,len(os.sched_getaffinity(0))//2))))
    env.pop(STRICT,None);env[kai_default.ENV]=str(observer)
    actual=json.loads(subprocess.check_output([*cmd,'--dry-run'],env=env))['command']
    kai_default.launch(actual)
    write_json(args.out/'launch.json',{'source':pin['revision'],'source_lock':lock,'checkpoint_sha256':digest,'model_revision':model['revision'],'kv_bytes':args.kv_bytes,
               'device':dev,'profile_config':profile_config,'argv':actual,'deterministic_humming':True,'port':args.port})
    log=args.out/'server.log';url=f'http://127.0.0.1:{args.port}'
    with log.open('w') as stream:
        proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        wait_server(proc,url,log)
        capacity=reported_capacity(log.read_text(errors='replace'))
        if capacity is None or capacity<1:raise ValueError('The full context cannot fit')
        completion('kaiLLM',url,bank['domains'][args.domain]['rows'][32][:256],8,120)
        for phase,prefix in [('prefill',8192),('decode',8192),('decode',65536)]:
            profile_wave(url,traces,observer,dev['uuid'],bank,args.domain,args.out/f'{phase}-{prefix}.json',prefix,phase)
    finally:
        stop_server(proc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','weights','banks','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--checkpoint',choices=('inkling-loop-iq4','inkling-udq2'),required=True)
    parser.add_argument('--expect-uuid')
    parser.add_argument('--domain',choices=('prose','code'),default='prose')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--kv-bytes',type=int,default=4<<30,help='Use the same KV allocation as the full-length control for this GPU')
    parser.add_argument('--port',type=int,default=18884)
    main(parser.parse_args())
