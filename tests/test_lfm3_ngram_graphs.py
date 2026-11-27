"""Check LFM3 GPU proposal arithmetic, replay staging and speculative convolution on CPU."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "runtime-source"
GPU = ROOT / "vllm-overrides/vllm/v1/worker/gpu"


def standalone_module(name, file, imports):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**imports, name: module}):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "Requires CPU Triton interpreter")
class Lfm3NgramGraphTests(unittest.TestCase):
    def test_proposals_keep_per_state_hits_and_batch_order_in_every_tail(self):
        import torch
        import triton
        import triton.language as tl
        module = standalone_module("lfm_candidate_ngram", GPU / "spec_decode/ngram/speculator.py", {
            "vllm.config": SimpleNamespace(VllmConfig=object),
            "vllm.triton_utils": SimpleNamespace(HAS_TRITON=True, tl=tl, triton=triton),
            "vllm.v1.worker.gpu.input_batch": SimpleNamespace(InputBatch=object),
            "vllm.v1.worker.gpu.spec_decode.speculator": SimpleNamespace(BaseSpeculator=object),
        })

        def reference(sequence, k, sampled):
            for width in range(8, 1, -1):
                for start in reversed(range(len(sequence)-width)):
                    if sampled and sequence[start:start+width] == sequence[-width:]:
                        after = sequence[start+width:start+width+k]
                        return after + [sequence[-1]] * (k-len(after)), k
            return [sequence[-1]] * k, 0

        class Graph:
            def __enter__(self):
                recording.append(self)

            def __exit__(self, *_):
                recording.pop()

            def replay(self):
                self.launches += 1
                self.operation()

            launches = 0

        for capacity, k in ((1, 3), (2, 3), (4, 3), (8, 1)):
            sequences = [[3, 5, 8, 2, 1, 3, 5], [4, 7, 2, 8], [2, 2, 2, 2, 2], [6, 9],
                         [1], [3, 5, 2, 6, 5, 2, 4, 5, 2], [7, 3, 7, 3], [4, 5, 3, 2, 6]][:capacity]
            bank = torch.zeros((capacity,32),dtype=torch.int32)
            for i, seq in enumerate(sequences):
                bank[i,:len(seq)] = torch.tensor(seq)
            state = SimpleNamespace(all_token_ids=SimpleNamespace(gpu=bank),
                                    last_sampled_tokens=torch.tensor([[seq[-1]] for seq in sequences]),
                                    total_len=SimpleNamespace(gpu=torch.tensor(list(map(len,sequences)),dtype=torch.int32)))
            config = SimpleNamespace(speculative_config=SimpleNamespace(
                num_speculative_tokens=k, prompt_lookup_min=2, prompt_lookup_max=8,
                ngram_gpu_single_active=k==3, ngram_gpu_match_only=False),
                scheduler_config=SimpleNamespace(max_num_seqs=capacity), model_config=SimpleNamespace(max_model_len=32))
            proposer = module.NgramGPUSpeculator(config, torch.device("cpu"), state)
            observed = []
            observer = SimpleNamespace(draft_replayed=lambda: observed.append("replay"))
            with self.assertRaises(RuntimeError):
                proposer.install_benchmark_graphs(observer)
            # Simulate CUDA capture in CPU-only CI. Each replay runs the real two
            # interpreted Triton kernels against the inputs originally captured.
            from contextlib import nullcontext
            recording = []
            lookup = proposer._lookup

            def launch(*args):
                if recording:
                    recording[-1].operation = lambda: lookup(*args)
                else:
                    lookup(*args)

            with patch.dict(os.environ, KAILLM_BENCHMARK_GRAPH_DIR="test"), \
                    patch.object(torch.cuda,"device",return_value=nullcontext()), \
                    patch.object(torch.cuda,"CUDAGraph",Graph), patch.object(torch.cuda,"graph",lambda graph:graph), \
                    patch.object(proposer,"_lookup",side_effect=launch):
                proposer.capture()
            self.assertEqual(proposer.install_benchmark_graphs(observer),capacity)
            for rows in list(range(1,capacity+1)) + list(reversed(range(1,capacity+1))):
                order = list(reversed(range(capacity)))[:rows]
                sampled = [int((idx + rows) % 3 != 0) for idx in order]
                batch = SimpleNamespace(idx_mapping=torch.tensor(order),num_reqs=rows)
                old = proposer.single_active_counts.clone() if k==3 else None
                actual = proposer.propose(batch,None,None,None,None,torch.tensor(sampled),None,
                                          state.last_sampled_tokens,None,None,None)
                expected = [reference(sequences[idx],k,keep) for idx,keep in zip(order,sampled)]
                self.assertEqual(actual.tolist(),[tokens for tokens,_ in expected])
                if k==3:
                    for idx,(_,hits) in zip(order,expected):
                        old[idx]=hits
                    self.assertEqual(proposer.single_active_counts.tolist(),old.tolist())
            self.assertEqual(len(observed),2*capacity)

    def test_speculative_convolution_commits_exactly_the_accepted_prefix(self):
        import torch
        sys.path.insert(0,str(ROOT))
        from vllm_gguf_plugin.inkling.conv_graph import ConvGraphMetadata
        from vllm_gguf_plugin.lfm3_graph_conv import gated_conv
        channels,taps = 9,4
        rng=torch.Generator().manual_seed(4923)
        def rand(*shape):
            return torch.randn(shape,generator=rng)
        for steps in (1,3):
            table=torch.tensor([[2],[1],[0]],dtype=torch.int32)
            weight,bias=rand(channels,taps),rand(channels)
            cache=rand(3,taps-1+steps,channels).transpose(-1,-2)
            for lengths,context,accepted,verify in (
                    ([10,4,0],[0,0,0],[1,1,1],[False,False,False]),
                    ([1+steps,1,0],[10,4,0],[1,1,1],[True,True,False]),
                    ([1,1+steps,0],[11,5,0],[1+steps,1,1],[True,True,False]),
                    ([1+steps,2,0],[12,0,0],[1,1,1],[True,False,False])):
                offsets=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
                md=ConvGraphMetadata(offsets,torch.tensor(lengths)+torch.tensor(context),table,None,
                                     torch.tensor(accepted),torch.tensor(verify))
                b,g,x=rand(sum(lengths)+2,3*channels).chunk(3,dim=-1)
                values=b*x
                expected=torch.zeros_like(values)
                before=cache.clone()
                stored=before.clone()
                for i,n in enumerate(lengths):
                    if not n:continue
                    slot=int(table[i,0])
                    shift=accepted[i]-1 if verify[i] else 0
                    history=before[slot,:,shift:shift+taps-1].T if context[i] else x.new_zeros(taps-1,channels)
                    span=torch.cat((history,values[offsets[i]:offsets[i+1]]))
                    for j in range(n):
                        expected[offsets[i]+j]=g[offsets[i]+j]*((span[j:j+taps].T*weight).sum(-1)+bias)
                    retained=span[1:] if verify[i] else span[-(taps-1):]
                    stored[slot,:,:len(retained)]=retained.T
                actual=gated_conv(b,g,x,weight,bias,cache,md)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
                torch.testing.assert_close(cache,stored)

    def test_full_dispatch_reserves_singleton_verification_and_covers_mixed_prefill(self):
        import torch
        try:
            from vllm.config.compilation import CompilationMode, CUDAGraphMode
        except ImportError:
            self.skipTest("Requires the pinned vLLM core")
        from vllm_gguf_plugin.serving_policy import capture_family
        utils = standalone_module("lfm_graph_dispatch", GPU / "cudagraph_utils.py", {})
        single = standalone_module("vllm.v1.worker.gpu.spec_decode.ngram.single_active",
                                   GPU / "spec_decode/ngram/single_active.py", {})
        for batch, k in ((1, 3), (2, 3), (4, 3), (8, 1)):
            options = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL, mode=CompilationMode.NONE,
                                      cudagraph_capture_sizes=capture_family(batch,k,2048),max_cudagraph_capture_size=2048)
            spec = SimpleNamespace(ngram_gpu_match_only=False,ngram_gpu_single_active=k==3,
                                   ngram_gpu_single_active_adapter={},uses_dynamic_speculative_decoding=lambda:False)
            cfg = SimpleNamespace(compilation_config=options,speculative_config=spec,use_v2_model_runner=True,
                                  cache_config=SimpleNamespace(use_kda_recoverssm=False,enable_prefix_caching=False),
                                  model_config=SimpleNamespace(dtype=torch.bfloat16,hf_text_config=SimpleNamespace(
                                      model_type="lfm3"),runner_type="generate",is_multimodal_model=False),
                                  parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,
                                      data_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1,
                                      enable_dbo=False),attention_config=SimpleNamespace(backend="TRITON_ATTN"),
                                  lora_config=None,aux_output_config=SimpleNamespace(enabled=False),kv_transfer_config=None,
                                  ec_transfer_config=None,scheduler_config=SimpleNamespace(scheduler_cls=None,
                                      async_scheduling=True,max_num_seqs=batch,max_num_batched_tokens=2048,
                                      max_num_scheduled_tokens=None,long_prefill_token_threshold=0))
            manager = utils.CudaGraphManager.__new__(utils.CudaGraphManager)
            manager.vllm_config=cfg
            manager.compilation_config=options
            manager.cudagraph_mode=CUDAGraphMode.FULL
            manager.max_num_reqs=batch
            manager.decode_query_len=k+1
            manager.varlen_decode=False
            manager.ubatch_runner=None
            manager.lora_capture_cases=[0]
            manager._candidates={}
            manager._capture_descs={}
            manager._graphs_captured=True
            manager._init_candidates()
            for n in (1,4,15,100,511,2048):
                self.assertEqual(manager.dispatch(min(n,batch),n,None,0).cg_mode,CUDAGraphMode.FULL)
            if k==3:
                with patch.dict(os.environ, KAILLM_BENCHMARK_GRAPH_DIR="test"), patch.object(single,"adapter_identity"):
                    single.validate_single_active_profile(cfg)
                desc=manager.dispatch(1,4,4,0)
                self.assertTrue(single.singleton_descriptor(desc,4))
                fake_batch=SimpleNamespace(num_reqs=1,num_tokens=4,num_draft_tokens=3,
                                           num_scheduled_tokens=[4],idx_mapping_np=[0])
                self.assertTrue(single.singleton_step_eligible(fake_batch,desc,has_prefill=False,greedy=True))
                self.assertFalse(single.singleton_step_eligible(fake_batch,desc,has_prefill=True,greedy=True))
