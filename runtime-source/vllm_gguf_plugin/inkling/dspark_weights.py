# SPDX-License-Identifier: Apache-2.0
"""Prepare Inkling's standalone GGUF block drafter for the vLLM model loader."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def configuration(reader, target):
    def field(name):
        item = reader.get_field('dflash.' + name)
        if item is None:
            raise ValueError('Missing Inkling DSpark metadata dflash.' + name)
        return item.contents()

    tensors = {t.name: t for t in reader.tensors}
    if reader.get_field('general.architecture').contents() != 'dflash' or 'joint_proj.weight' not in tensors:
        raise ValueError('Expected an Inkling DSpark GGUF with a recurrent Markov head')
    h = int(field('embedding_length'))
    padded = int(target['inkling2_loop.vocab_size'])
    vocab = int(target['inkling2_loop.vocab_size_unpadded'])
    taps = list(map(int, field('target_layers')))
    folded = 'markov_w2.weight' in tensors
    rank, rows = map(int, tensors['markov_w1.weight'].shape)
    if (h != target['inkling2_loop.embedding_length'] or rows != (vocab if folded else padded)
            or tuple(tensors['joint_proj.weight'].shape) != (2 * (h + rank), 3 * rank)):
        raise ValueError('Inkling drafter and target have incompatible feature or vocabulary dimensions')
    layers = int(field('block_count'))
    slots = list(map(bool, field('attention.sliding_window_pattern')))
    if len(slots) != layers or taps != sorted(set(taps)) or field('sample_from_anchor'):
        raise ValueError('Unsupported Inkling drafter attention or feature layout')
    extra_visits = (int(target['inkling2_loop.loop.count']) - 1) * (
        int(target['inkling2_loop.loop.end']) - int(target['inkling2_loop.loop.start']))
    visits = int(target['inkling2_loop.block_count']) + extra_visits
    if not taps or taps[0] < 0 or taps[-1] > visits:
        raise ValueError('Inkling drafter taps exceed the target logical visit range')
    window = int(field('attention.sliding_window')) - int(folded)
    # The folded GGUF includes the anchor in its encoded sliding-window length.
    return dict(model_type='qwen3', architectures=['InklingDSparkDraftModel'], hidden_size=h,
                vocab_size=padded, unpadded_vocab_size=vocab, intermediate_size=int(field('feed_forward_length')),
                num_attention_heads=int(field('attention.head_count')), num_key_value_heads=int(field('attention.head_count_kv')),
                head_dim=int(field('attention.key_length')), num_hidden_layers=layers, rms_norm_eps=float(field('attention.layer_norm_rms_epsilon')),
                rope_theta=float(field('rope.freq_base')), max_position_embeddings=int(field('context_length')),
                hidden_act='silu', torch_dtype='bfloat16', tie_word_embeddings=False, is_causal=False,
                sample_from_anchor=False,
                layer_types=['sliding_attention' if local else 'full_attention' for local in slots],
                use_sliding_window=True, sliding_window=window + int(field('block_size')), local_layers=slots,
                context_window=window, block_size=int(field('block_size')), markov_rank=rank,
                folded_markov=folded, final_logit_softcapping=float(field('final_logit_softcapping')),
                dflash_config={'layer_ids': taps, 'mask_token_id': 0, 'causal': False, 'use_aux_hidden_state': True})


def prepare(source, target_path, *, materialize=True):
    import gguf
    import torch
    from filelock import FileLock
    from safetensors.torch import save_file

    from ..inkling_metadata import legacy_loop_metadata
    from ..serving_defaults import read_header

    source = Path(source)
    target, _ = read_header(Path(target_path), tensor_names=False)
    if target.get('general.architecture') == 'inkling2' and 'inkling2.num_loops' in target:
        # The lightweight header reader skips arrays. Check the full fixed-loop
        # schedule before mapping logical visits and vocabulary dimensions.
        reader = gguf.GGUFReader(str(target_path))
        target = legacy_loop_metadata({k: v.contents() for k, v in reader.fields.items()
                                       if k == 'general.architecture' or k.startswith('inkling2.')})
    if target.get('general.architecture') != 'inkling2_loop':
        raise ValueError('Inkling DSpark requires a fixed-loop Inkling target')
    root = Path(os.environ.get('KAILLM_CACHE_DIR', Path.home() / '.cache/kaillm')) / 'inkling-dspark'
    fingerprint = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            fingerprint.update(block)
    reader = gguf.GGUFReader(str(source))
    config = configuration(reader, target)
    dest = root / (fingerprint.hexdigest() + '-' + str(config['vocab_size']) + '-v2')
    if not materialize:
        return dest, config
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest) + '.lock'):
        if (dest / 'model.safetensors').is_file() and (dest / 'config.json').is_file():
            return dest, config
        with TemporaryDirectory(dir=root) as tmp:
            pending = Path(tmp) / 'prepared'
            pending.mkdir()
            weights = {}
            for tensor in reader.tensors:
                dtype = tensor.tensor_type
                if dtype not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.Q8_0):
                    raise ValueError(f'Unsupported Inkling DSpark tensor type for {tensor.name}: {dtype}')
                data = gguf.quants.dequantize(tensor.data, dtype)
                shape = tuple(reversed([int(n) for n in tensor.shape]))
                # GGUF weights are matrix-row-major after reversing the GGML dimensions.
                value = torch.from_numpy(data.copy()).reshape(shape)
                weights[tensor.name] = value if dtype == gguf.GGMLQuantizationType.F32 else value.to(torch.bfloat16)
            save_file(weights, str(pending / 'model.safetensors'))
            (pending / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            pending.rename(dest)
    return dest, config


def load_parameters(model, weights):
    import torch

    c = model.config
    params = dict(model.named_parameters())
    loaded = set()

    def assign(name, data):
        param = params[name]
        if param.shape != data.shape:
            raise ValueError(f'Inkling DSpark {name}: expected {tuple(param.shape)}, got {tuple(data.shape)}')
        with torch.no_grad():
            param.copy_(data)
        loaded.add(name)

    assign('markov.token.weight', weights.pop('markov_w1.weight'))
    assign('model.norm.weight', weights.pop('output_norm.weight'))
    masks = weights.pop('mask_embd.weight').reshape(-1, c.hidden_size)
    assign('model.masks', masks.expand(c.block_size - 1, -1))
    mix = (torch.stack([weights.pop(f'blk.{i}.ctx_mix.weight') for i in range(c.num_hidden_layers)] +
                       [weights.pop('markov_ctx_mix.weight')]) if c.folded_markov else weights.pop('dspark_fuse.weight').float().softmax(-1))
    assign('markov_context.mix', mix[-1])
    assign('markov_context.norm', weights.pop('markov_ctx_norm.weight' if c.folded_markov else 'dspark_norm.weight'))
    if 'token_embd_norm.weight' in weights:
        assign('model.embed_norm', weights.pop('token_embd_norm.weight'))
    else:
        # This export borrows both the input embedding and its normalization from the target.
        loaded.add('model.embed_norm')
    for suffix in ('weight', 'bias'):
        assign('markov.joint.' + suffix, weights.pop('joint_proj.' + suffix))
        assign('markov.output.' + suffix, weights.pop(('markov_w2.' if c.folded_markov else 'dspark_hidden_out.') + suffix))
    for i in range(c.num_hidden_layers):
        prefix = f'model.layers.{i}.'
        source = f'blk.{i}.'

        def pack(*names):
            return torch.cat([weights.pop(source + name + '.weight') for name in names])

        assign(prefix + 'context_mix.mix', mix[i])
        assign(prefix + 'context_kv.weight', pack('attn_k_ctx', 'attn_v_ctx'))
        assign(prefix + 'self_attn.qkv_proj.weight', pack('attn_q', 'attn_k', 'attn_v'))
        assign(prefix + 'mlp.gate_up_proj.weight', pack('ffn_gate', 'ffn_up'))
        for src, dest in [('attn_output', 'self_attn.o_proj'), ('ffn_down', 'mlp.down_proj'),
                          ('attn_norm', 'input_layernorm'), ('ffn_norm', 'post_attention_layernorm'),
                          ('attn_q_norm', 'self_attn.q_norm'), ('attn_k_norm', 'self_attn.k_norm'),
                          ('attn_k_ctx_norm', 'context_k_norm')]:
            assign(prefix + dest + '.weight', weights.pop(source + src + '.weight'))
        assign(prefix + 'context_mix.norm', weights.pop(source + ('ctx_norm.weight' if c.folded_markov else 'dflare_norm.weight')))
    if weights or set(params) != loaded:
        raise ValueError(f'Incomplete Inkling DSpark load: extra {sorted(weights)}, missing {sorted(set(params)-loaded)}')
    return loaded
