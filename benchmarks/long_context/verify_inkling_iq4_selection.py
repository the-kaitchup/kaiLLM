"""Check the physical expert kernels in a completed canonical Inkling IQ4 CUPTI profile."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


POLICIES = ('baseline', '3090-gate-up', '3090-prefill-and-gate-up', '3090-packed', '3090-routed', '5090-prefill')
ORIGINAL = 'iq4_nl_moe_kernel'
DIRECT = 'routed_iq4_b1_kernel'
Q8 = 'q8_0_moe_kernel'


def verify(profile, trace, policy):
    if policy not in POLICIES or hashlib.sha256(trace).hexdigest() != profile['trace_sha256']:
        raise ValueError('Unrecognized policy or mismatched raw trace digest')
    phase = profile['phase']
    iterations = profile['measurement_iterations']
    if phase not in ('prefill', 'decode') or type(iterations) is not int or iterations <= 0:
        raise ValueError('Invalid measured iteration count or phase')
    data = json.loads(trace)
    cuda = [d['CUDA'] for d in data if isinstance(d, dict) and 'CUDA' in d]
    if len(cuda) != 1 or len(cuda[0]) != 1:
        raise ValueError('Expected one physical CUDA device')
    device_id, physical = next(iter(cuda[0].items()))
    arch = physical['arch']
    if policy != 'baseline' and str(arch) != ('86' if policy.startswith('3090') else '120'):
        raise ValueError('Policy does not match the physical CUDA architecture')
    # The canonical weights have 20 physical IQ4 expert layers, 12 of
    # which execute three times. Four Q8 layers execute once.
    iq4_visits = 44 * iterations
    expected = Counter({Q8: 8 * iterations})
    if phase == 'decode' and policy == '3090-routed':
        expected = Counter({DIRECT: iq4_visits, ORIGINAL+'_m4_n16': iq4_visits,
                            Q8+'_m4_n16': 4 * iterations, Q8+'_m4_n64': 4 * iterations})
    elif phase == 'decode' and policy.startswith('3090'):
        down = ORIGINAL + '_m4_n32' if policy == '3090-packed' else ORIGINAL
        expected.update({DIRECT: iq4_visits, down: iq4_visits})
    elif phase == 'prefill' and policy in ('3090-packed', '3090-routed'):
        expected = Counter({ORIGINAL+'_m32_n64_k64': iq4_visits, ORIGINAL+'_m16_n64_k64': iq4_visits,
                            Q8+'_m32_n64': 8 * iterations})
    elif phase == 'prefill' and policy == '3090-prefill-and-gate-up':
        expected.update({ORIGINAL+'_m16': iq4_visits, ORIGINAL+'_m32': iq4_visits})
    elif phase == 'prefill' and policy == '5090-prefill':
        expected[ORIGINAL+'_m32'] = 2 * iq4_visits
    else:
        expected[ORIGINAL] = 2 * iq4_visits
    wanted_scope = ('execute_context_1(2048)_generation_0(0)' if phase == 'prefill'
                    else 'execute_context_0(0)_generation_1(1)')
    counts = Counter()
    def walk(node, scope):
        name = node['frame']['name']
        if name.startswith('execute_context_'):
            scope = name
        metrics = node.get('metrics', {})
        if ('device_id' in metrics and (name.startswith((ORIGINAL, Q8)) or name == DIRECT)):
            duration = metrics.get('time (ns)', 0)
            if (str(metrics['device_id']) != str(device_id) or not scope.startswith(wanted_scope)
                    or type(metrics.get('count')) is not int or metrics['count'] <= 0
                    or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f'Invalid physical {phase} expert observation for {name}')
            counts[name] += metrics['count']
        for child in node.get('children', []):
            walk(child, scope)
    for root in data:
        if 'frame' in root:
            walk(root, '')
    if counts != expected:
        raise ValueError(f'Unexpected expert launches: observed {dict(counts)}, expected {dict(expected)}')
    return {'policy': policy, 'phase': phase, 'iterations': iterations, 'trace_sha256': profile['trace_sha256'],
            'gpu_uuid': profile['gpu_uuid'], 'architecture': str(arch), 'expert_kernel_calls': dict(counts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--policy', required=True, choices=POLICIES)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.profile.read_bytes()), args.trace.read_bytes(), args.policy), indent=2))
