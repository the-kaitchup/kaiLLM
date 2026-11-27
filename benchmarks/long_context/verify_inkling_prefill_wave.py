"""Verify the 2,048-token expert and dense attention kernels in an unscored Inkling prefill wave."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from benchmark import wave, write_json
from profile_inkling_runtime import profile_request


_SCOPE = re.compile(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)')
_HUMMING = re.compile(r'_Z7hummingI10MmaOpClass5ShapeILj0ELj(\d+)ELj(\d+)')
IQ4 = 'iq4_nl_moe_kernel'
Q8 = 'q8_0_moe_kernel'


def verify(trace, batch, tiled):
    raw = json.loads(trace)
    cuda = [item['CUDA'] for item in raw if isinstance(item,dict) and 'CUDA' in item]
    if len(cuda) != 1 or len(cuda[0]) != 1:
        raise ValueError('Expected a single hardware device in the CUPTI trace')
    device_id, properties = next(iter(cuda[0].items()))
    if str(properties['arch']) != '120' or batch not in (1,2,4):
        raise ValueError('Expected an SM120 physical prefill trace at a supported batch')
    experts, attention, scopes = Counter(), Counter(), Counter()
    def walk(node, scope=None):
        name = node['frame']['name']
        if name.startswith('execute_context_'):
            scope = name
        metrics = node.get('metrics',{})
        dense = _HUMMING.match(name)
        if (name.startswith((IQ4,Q8)) or (dense and int(dense[1]) < 10000)) and 'device_id' in metrics:
            shape = _SCOPE.fullmatch(scope or '')
            if (shape is None or not 1 <= int(shape[1]) <= batch or shape.groups()[1:] != ('2048','0','0')
                    or str(metrics['device_id']) != str(device_id)
                    or type(metrics.get('count')) is not int or metrics['count'] <= 0 or metrics.get('time (ns)',0) <= 0):
                raise ValueError(f'Kernel was not uniquely attributed to a 2,048-token prefill forward: {name}')
            count = metrics['count']
            if dense:
                # The vocabulary head may evaluate a different number of selected rows.
                attention[name] += count
            else:
                scopes[scope] += count
                experts[name] += count
        for child in node.get('children',[]):
            walk(child,scope)
    for tree in raw:
        if isinstance(tree,dict) and 'frame' in tree:
            walk(tree)
    steps = 4*batch
    expected = {IQ4+('_m32' if tiled else ''):88*steps, Q8:8*steps}
    if experts != expected:
        raise ValueError(f'Unexpected physical prefill expert launches: {dict(experts)} (expected {expected})')
    expected_dense = {(3072,2304):[96], (1024,2304):[96], (2304,3072):[48],
                      (512,2304):[48], (256,2304):[16,48]}
    observed_dense = {}
    for name, count in attention.items():
        dimensions = tuple(map(int,_HUMMING.match(name).groups()))
        observed_dense.setdefault(dimensions,[]).append(count)
    if ({key:sorted(counts) for key,counts in observed_dense.items()}
            != {key:[count*steps for count in counts] for key,counts in expected_dense.items()}):
        raise ValueError('The physical dense attention matrix shapes or per-forward launch counts differ')
    return {'architecture':'120', 'prefill_steps':steps,'tokens_per_step':2048,'expert_kernels':dict(experts),
            'attention_kernels':dict(attention),'expert_calls_by_scope':dict(scopes),
            'trace_sha256':hashlib.sha256(trace).hexdigest()}


def capture(url, prompts, traces, out, device, tiled, label):
    if not prompts or any(len(row)!=8192 for row in prompts):
        raise ValueError('Use the same full-length prefill inputs as the timed wave')
    traces, out = Path(traces), Path(out)
    prior = set(traces.glob('*.hatchet'))
    profile_request(url,'/start_profile',{'profile_prefix':label,'delay_iterations':0,'max_iterations':64})
    try:
        response = wave('kaiLLM',url,prompts,1,600)
    finally:
        profile_request(url,'/stop_profile')
    requests = response['requests']
    if len(requests)!=len(prompts) or any(r['prompt_tokens']!=8192 or r['output_tokens']!=1 or r['cache_hits'] for r in requests):
        raise ValueError('Unexpected real prompt length, output count, or prefill cache hit')
    files = set(traces.glob('*.hatchet')) - prior
    if len(files)!=1:
        raise ValueError('Expected one new worker profiler trace')
    trace = files.pop()
    result = dict(verify(trace.read_bytes(),len(prompts),tiled),gpu_uuid=device,trace_name=trace.name,
                  prompt_tokens=sum(r['prompt_tokens'] for r in requests),label=label)
    write_json(out,result)
    return result
