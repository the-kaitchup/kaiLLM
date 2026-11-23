# Inkling 2 Mini kaiPTQ files: ITQ for routed experts (`itq-moe`, version 1)

kaiLLM loads version 1 routed-expert descriptors from the GGUF manifest.

## Container

- GGUF v3 in llama.cpp master's layout: `general.architecture = "inkling2_loop"`, every `inkling2_loop.*` key
  and the tokenizer exactly as in ggml-org's BF16 GGUF, tensor names as llama.cpp master (`blk.L.attn_q.weight`,
  `blk.L.ffn_gate_up_exps.weight`, `blk.L.loop_q_ctrl_down.weight`, ...). Files in the PR-32695 layout
  (`inkling2`, block_count 48) are not ITQ containers.
- Like the dense kaiPTQ files (Qwen3.8, LFM3): metadata key `yaqa.manifest` = JSON string
  `{"version": 1, "tensors": {LOGICAL: DESCRIPTOR, ...}, "provenance": {...}}`. LOGICAL is the llama.cpp name
  of the weight it replaces (ends in `.weight`); that name must NOT also exist as a GGUF tensor. Every
  component is an ordinary GGUF tensor of a scalar type (I8 = 24, F32 = 0), referenced by name.
- Any tensor not in the manifest stays an ordinary GGML tensor and keeps its llama.cpp meaning (including
  stock `<name>.scale` per-expert scales of GGML expert tensors).
- Each layer and projection can be chosen independently: `blk.L.ffn_gate_up_exps` and `blk.L.ffn_down_exps` of one layer may be one in
  `itq-moe` and the other in any GGML type (e.g. gate_up ITQ, down NVFP4 with its stock `.scale`), each dense
  projection may be dense `itq` or GGML, and layers may differ. GGML types kaiLLM runs here: F32, F16, BF16,
  Q8_0, Q4_0/Q4_1/Q5_0/Q5_1, Q2_K..Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS,
  MXFP4, NVFP4, Q1_0, Q2_0 (F8_E4M3 for non-matmul tables). The MoE path dispatches each projection by its
  own format (gate_up stage, SITU, down stage), so no pairing is required. The only coupling: split gate/up
  (below) needs both halves in `itq-moe`, because stock GGUF has no split expert tensors for this arch.
- Shared trellis tables (one copy per file, as in the dense files): `itq.table.v4t12` I8 GGUF dims [4, 4096]
  (numpy (4096, 4)) and `itq.table.v2t10` I8 dims [2, 1024] (numpy (1024, 2)); int8 values of the codec's
  Gaussian tables (`codec.int8_table(codec.gaussian_table(V, table_bits))`). kaiLLM detects a kaiPTQ file by a
  tensor named `itq.table.*`.

## Expert tensors: codec `itq-moe`

Logical names (Inkling: E = 128, n_embd = 2304, n_ff = 1152):

| logical | per-expert matrix W_e [m, n] | rotations (encoder `auto_rotation`) |
|---|---|---|
| `blk.L.ffn_gate_up_exps.weight` (fused) | [2 n_ff, n_embd] = [2304, 2304]; rows [0, 1152) gate, [1152, 2304) up (llama.cpp's order) | in `block256`, out `block256` |
| or split: `blk.L.ffn_gate_exps.weight` + `blk.L.ffn_up_exps.weight` | [n_ff, n_embd] = [1152, 2304] each | in `block256`, out `block128` |
| `blk.L.ffn_down_exps.weight` | [n_embd, n_ff] = [2304, 1152] | in `block128`, out `block256` |

Split gate/up (lets gate and up of one expert take different rates) needs both tensors in `itq-moe` with the
same E, n, `rotation_in` and byte-identical `su`. Per layer, either form; a layer may also keep stock GGML
experts. y_e = W_e x, as llama.cpp's `mul_mat_id` (W_e is GGUF slice e: numpy [E, m, n] = (expert, out, in)).

Descriptor:

```json
{
  "codec": "itq-moe", "version": 1,
  "shape": [128, 2304, 2304],
  "L": 16, "hash": "mul", "layout": "tile16-rowmajor-steps-lsb",
  "V":   [2, 4, 2, ...],
  "bps": [5, 9, 4, ...],
  "rotation_in": "block256", "rotation_out": "block256",
  "rate": [2.5, 2.25, 2.0, ...],
  "components": {
    "stream": "blk.7.ffn_gate_up_exps.itq.stream",
    "su": "blk.7.ffn_gate_up_exps.itq.su",
    "sv": "blk.7.ffn_gate_up_exps.itq.sv",
    "wscale": "blk.7.ffn_gate_up_exps.itq.wscale",
    "table_v2": "itq.table.v2t10",
    "table_v4": "itq.table.v4t12",
    "post_output_scale": "blk.7.ffn_gate_up_exps.itq.post_output_scale",
    "pass_output_scale": "blk.7.ffn_gate_up_exps.itq.pass_output_scale"
  }
}
```

- `V`, `bps`: one int per expert (or one int for all). Allowed formats, rate = bps / V bits per weight:
  V = 4: bps 4..12 (1.00, 1.25, ..., 3.00); V = 2: bps 3..8 (1.5, 2.0, 2.5, 3.0, 3.5, 4.0). table_bits is
  implied (V4 -> 12, V2 -> 10). `rate` is informational. Experts may mix V and bps freely, in any order.
- `stream` (I8, any shape, read flat): the experts' tile streams concatenated in expert order 0..E-1, no
  padding. Expert e has (m n / 256) tiles x (32 bps_e / V_e) bytes, so its byte offset is the sum of the
  earlier experts' sizes (always a multiple of 16 bytes for these shapes). Inside an expert the bytes are
  exactly the dense codec's: 16 x 16 tiles in row-block-major order (tile (i, j) = rows 16i.., columns 16j..,
  index i (n / 16) + j), 256 / V steps of bps bits per tile LSB-first, 16-bit circular windows, multiplicative
  hash `((state * 0x9E3779B1) mod 2^32) >> (32 - table_bits)`, V int8 values per step, row-major inside the
  tile (`reconstruct_itq` / `codec.py`).
- `su` I8 [n], `sv` I8 [m]: +1 / -1 signs, ONE vector each for all experts of the tensor (the 8 experts of a
  token share one input preparation; down shares too, for one post kernel). `rotation_in` / `rotation_out`
  likewise shared: `fwht` (width a power of two) or `blockK`, K a power of two dividing the width; kaiLLM runs
  K in 32..4096 on the input side and 16..4096 on the output side.
- `wscale` F32 [E]: per-expert global scale, > 0. Fold any per-expert scalar (the stock `.scale`) into it: an
  ITQ logical tensor never has a `<name>.scale` tensor.
- Optional, F32: `post_output_scale` [E, m] (numpy order; GGUF dims [m, E]); `post_input_scale` [n] (all
  experts; required form for gate_up / gate / up) or [E, n] (down only); `pass_output_scale` [P, E, m]
  (GGUF dims [m, E, P]), P = `inkling2_loop.loop.count` = 3, only in core layers
  [`loop.start`, `loop.end`) = 6..17: pass p (0-based) multiplies output row r of expert e by [p, e, r]
  on top of `post_output_scale` (per-pass correction of the shared codes).

Reconstruction (FP64 in the oracle), with T_e the decoded table values [m, n] and R = orthonormal Sylvester
block Hadamard of the named size:

    W_e = diag(pass_p,e) diag(pos_e) diag(sv) R_out (wscale_e T_e) R_in diag(su) diag(pis_e)

i.e. the dense `itq` codec per expert (`yaqa_gguf.itq_moe_expert(descriptor, components, e)` returns that dense
descriptor and its components). Run time: input rotated and int8-quantized per 32 values (as dense ITQ),
trellis GEMV / GEMM per selected expert, R_out + signs + scales in the epilogue.

Size of an expert tensor: sum_e m n bps_e / (8 V_e) + n + m + 4 E bytes (+ optional scales). Inkling per
layer: gate_up 84.93 r MB + down 42.47 r MB at r bits per weight -> all 24 layers 3.057 r GB (r = 2.0:
6.11 GB).

## Other Inkling tensors

- Dense `itq` (version 1, the Qwen / LFM3 codec, unchanged; now also V2 bps 7 / 8) may replace
  `attn_q` [3072, 2304], `attn_k` / `attn_v` [1024, 2304], `attn_gate` [3072, 2304], `attn_rel` [384, 2304]
  and `attn_output` [2304, 3072]. Optional `pass_output_scale` F32 [P, m] (GGUF dims [m, P]) in core layers,
  same meaning as above.
- Inputs: in passes 2 and 3 of the core layers, Q, K and V read different (gated) inputs even in the sliding
  layers, so only these groups read one activation in every pass: {`attn_q`, `attn_rel`, `attn_gate`,
  `attn_sink_proj`}, {`attn_k`, `attn_k_conv_proj`}, {`attn_v`, `attn_v_conv_proj`}. Matrices of one group may
  share `su` + `rotation_in` (+ `post_input_scale`); kaiLLM then prepares the input once.
- Stay ordinary GGML tensors: `attn_sink_proj` [24, 2304] and the conv projections [32, 2304] (too small;
  m % 16 for the sink), `ffn_gate_inp` (router, F32 in llama.cpp), `exp_probs_b`, all norms, queries, convs,
  relative-bias tables, clips, the loop controllers (F32 / BF16 / Q8_0 ...), `token_embd` and `output`
  (GGML types: kaiLLM's `itqvocab` codec needs dim % 1024 == 0, Inkling's is 2304).

## Checks an exporter can run

    export PYTHONPATH="$PWD/runtime-source:$PWD/runtime-source/gguf-reader"
    python -c "import gguf; from vllm_gguf_plugin import yaqa_gguf as y; r = gguf.GGUFReader('FILE.gguf'); \
        m = y.read_yaqa_manifest(r); [y.load_yaqa_components(r, d) for d in m.values()]; print(len(m), 'OK')"

`reconstruct_itq_moe(descriptor, components, experts=None, pass_index=None)` gives the FP32 [E, m, n] weights
(the decode-to-BF16 path for llama.cpp scoring; fused gate/up = concatenate the split rows).
