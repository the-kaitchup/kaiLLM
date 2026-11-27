"""Compare actual upstream FLA chunk arithmetic with independent packed request controls."""
from contextlib import ExitStack, contextmanager
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl


@triton.jit
def _interpreted_exp(x):
    return tl.exp(x)


@triton.jit
def _interpreted_exp2(x):
    return tl.exp2(x)


class _LoopIndex(int):
    def to(self, dtype):
        # Kernel indices are int64 on CUDA. Python integers already cover the test range.
        return self


@contextmanager
def _cpu_interpreter():
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.interpreter import InterpreterBuilder

    original_dot = InterpreterBuilder.create_dot

    def numeric_dot(builder, a, b, acc, precision, maximum):
        # The pinned interpreter stores BF16 as uint16 and sends it directly to
        # numpy.matmul. Decode those bit patterns before accumulating in FP32.
        if a.dtype == tl.bfloat16:
            a = builder.cast_impl(a, tl.float32)
        if b.dtype == tl.bfloat16:
            b = builder.cast_impl(b, tl.float32)
        return original_dot(builder, a, b, acc, precision, maximum)

    with ExitStack() as scope:
        # Select one legal kernel shape. Hardware autotuning would request CUDA.
        scope.enter_context(patch.object(Autotuner, '_bench', lambda *args, **kwargs: 0.))
        scope.enter_context(patch.object(InterpreterBuilder, 'create_dot', numeric_dot))
        for name in ('chunk_scaled_dot_kkt', 'chunk_delta_h', 'chunk_o', 'wy_fast'):
            mod = importlib.import_module('vllm.third_party.flash_linear_attention.ops.'+name)
            # Module aliases otherwise retain the unpatched language function.
            for attr, fn in (('exp', _interpreted_exp), ('exp2', _interpreted_exp2)):
                if hasattr(mod, attr):
                    scope.enter_context(patch.object(mod, attr, fn))
            if name == 'chunk_delta_h':
                scope.enter_context(patch.object(mod, 'range', lambda *a: map(_LoopIndex, range(*a)), create=True))
        yield


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'CPU Triton interpreter required')
class QwenGDNChunkArithmetic(unittest.TestCase):
    def test_actual_chunk_outputs_and_final_states(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime-source'))
        from vllm_gguf_plugin.qwen35_graph_gdn import chunk_plan
        from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

        lengths, capacity, width, heads = (63,1,65,0), 144, 16, 2
        offsets = torch.tensor([0,63,64,129,129], dtype=torch.int32)
        source, valid, unpack, padded, chunk_indices, chunk_offsets = chunk_plan(offsets, capacity, 64)

        def call(values, initial, offsets, indices, chunks):
            return chunk_gated_delta_rule(
                *values, scale=width**-.5, initial_state=initial, output_final_state=True, cu_seqlens=offsets,
                chunk_indices=indices, chunk_offsets=chunks, use_qk_l2norm_in_kernel=False)

        with _cpu_interpreter():
            for dtype, tolerance in ((torch.float16,.001), (torch.bfloat16,.01)):
                rng = torch.Generator().manual_seed(181)
                def random():
                    return torch.randn(capacity, heads, width, generator=rng)
                q, k = (torch.nn.functional.normalize(random(), dim=-1).to(dtype) for _ in range(2))
                v = random().to(dtype)
                decay = -.75*torch.rand(capacity,heads,generator=rng)
                beta = torch.sigmoid(torch.randn(capacity,heads,generator=rng))
                values = (q,k,v,decay,beta)
                initial = torch.randn(len(lengths),heads,width,width,generator=rng)*.04
                initial[1].zero_()
                def pack(t):
                    mask = valid.reshape((-1,) + (1,)*(t.ndim-1))
                    return torch.where(mask,t[source],0).unsqueeze(0)
                output, final = call([pack(t) for t in values], torch.cat((initial,initial[:1]*0)),
                                     padded, chunk_indices, chunk_offsets)

                live = int(offsets[-1])
                isolated_outputs, isolated_states = [], []
                oracle = torch.zeros(live,heads,width,dtype=torch.float64)
                oracle_states = initial.double().clone()
                for row, length in enumerate(lengths):
                    begin, end = int(offsets[row]), int(offsets[row+1])
                    if length:
                        numbers = torch.arange((length+63)//64, dtype=torch.int32)
                        standalone, state = call(
                            [t[begin:end].unsqueeze(0) for t in values], initial[row:row+1],
                            torch.tensor([0,length],dtype=torch.int32), torch.stack((torch.zeros_like(numbers),numbers),1),
                            torch.tensor([0,len(numbers)],dtype=torch.int32))
                        isolated_outputs.append(standalone[0])
                        isolated_states.append(state)
                    else:
                        isolated_states.append(initial[row:row+1])
                    h = oracle_states[row]
                    for t in range(begin,end):
                        key = k[t].double()
                        h = h*torch.exp(decay[t].double())[:,None,None]
                        residual = v[t].double()-(h@key[:,:,None]).squeeze(-1)
                        h = h+(beta[t].double()[:,None]*residual)[:,:,None]*key[:,None,:]
                        oracle[t] = (h@q[t].double()[:,:,None]).squeeze(-1)/width**.5
                    oracle_states[row] = h

                packed_output = output[0,unpack[:live]]
                torch.testing.assert_close(packed_output, torch.cat(isolated_outputs), rtol=0, atol=0)
                torch.testing.assert_close(final[:len(lengths)], torch.cat(isolated_states), rtol=0, atol=0)
                torch.testing.assert_close(packed_output.double(), oracle, rtol=0, atol=tolerance)
                torch.testing.assert_close(final[:len(lengths)].double(), oracle_states, rtol=0, atol=tolerance)


if __name__ == '__main__':
    unittest.main()
