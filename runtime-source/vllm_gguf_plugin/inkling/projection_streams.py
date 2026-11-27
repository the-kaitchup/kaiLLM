"""Launch independent projections with their original Humming kernels and output buffers."""
from functools import lru_cache
import os
from pathlib import Path

import torch


# Each group runs sequentially on one stream. Full-attention layers also have the two conv projections.
PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
SCHEDULE = (
    ('q_proj', 'rel_proj'),
    ('gate_proj', 'sink_proj'),
    ('k_proj', 'k_conv_proj'),
    ('v_proj', 'v_conv_proj'),
)
OWNERS = dict(q_proj=0, gate_proj=0, rel_proj=0, sink_proj=0,
              k_proj=1, k_conv_proj=1, v_proj=2, v_conv_proj=2)


def is_iq4_nl_donor(model):
    source = getattr(model, 'model_weights', None) or getattr(model, 'model', None)
    return (isinstance(source, (str, Path)) and Path(source).name == 'Inkling-2-Mini-IQ4_NL.gguf'
            and getattr(model.hf_config, 'model_type', None) == 'inkling2_loop')


def supports_5080_b2(serving, gpu_name, count):
    if (serving is None or count != 1 or gpu_name != 'NVIDIA GeForce RTX 5080'
            or serving.speculative_config is not None or serving.parallel_config.world_size_across_dp != 1
            or serving.scheduler_config.max_num_seqs != 2):
        return False
    model = getattr(serving, "model_config", None)
    if model is None:
        return False
    source = getattr(model, 'model_weights', None) or getattr(model, 'model', None)
    return (getattr(model, 'max_model_len', None) == 73728 and getattr(model, 'dtype', None) == torch.bfloat16
            and getattr(model.hf_config, 'model_type', None) == 'inkling2_loop'
            and isinstance(source, (str, Path)) and Path(source).name == 'Inkling-2-Mini-UD-Q2_K_XL.gguf')


def supports_gb10_b1(serving, gpu_name, count):
    if (os.getenv("KAILLM_INKLING_SPARK_PROJECTION_STREAMS") != "1" or gpu_name != 'NVIDIA GB10'
            or serving is None or count != 1 or serving.speculative_config is not None
            or serving.parallel_config.world_size_across_dp != 1 or serving.scheduler_config.max_num_seqs != 1
            or serving.scheduler_config.max_num_batched_tokens != 2048
            or str(serving.cache_config.cache_dtype) not in ("auto", "bfloat16")):
        return False
    model = serving.model_config
    source = getattr(model, 'model_weights', None) or getattr(model, 'model', None)
    if not (model.max_model_len == 73728 and model.dtype == torch.bfloat16
            and getattr(model.hf_config, 'model_type', None) == 'inkling2_loop'
            and isinstance(source, (str, Path)) and Path(source).name == 'Inkling-2-Mini-dondon-Q2_K_XS.gguf'
            and torch.cuda.get_device_capability() == (12, 1)
            and torch.cuda.get_device_properties(0).multi_processor_count == 48):
        return False
    from .spark_verify_router import SHA256, _hash

    try:
        stat = Path(source).stat()
        return _hash(str(source), stat.st_ino, stat.st_size, stat.st_mtime_ns) == SHA256
    except OSError:
        return False


def supported(serving, gpu_name, count):
    if (serving is None or count != 1 or serving.speculative_config is not None
            or serving.parallel_config.world_size_across_dp != 1 or not is_iq4_nl_donor(serving.model_config)):
        return False
    slots = serving.scheduler_config.max_num_seqs
    return ((gpu_name == 'NVIDIA GeForce RTX 5090' and slots in (1, 2, 4))
            or (gpu_name == PRO_6000 and slots == 1))



@lru_cache(maxsize=None)
def pool(index, count):
    from vllm.logger import init_logger

    init_logger('vllm.gguf.inkling').info('Using %d independent Inkling attention projection streams', count + 1)
    return tuple(torch.cuda.Stream(device=index) for _ in range(count))


def prepare(attn):
    from vllm.config import get_current_vllm_config_or_none

    serving = get_current_vllm_config_or_none()
    if serving is None or not torch.cuda.is_available():
        return None
    from . import pro_udq2
    from .dondon_5090 import is_dondon_m

    gpu, count = torch.cuda.get_device_name(), torch.cuda.device_count()
    if (supports_5080_b2(serving, gpu, count) or is_dondon_m(serving, gpu, count)
            or supports_gb10_b1(serving, gpu, count)):
        from vllm import envs
        if not (envs.VLLM_BATCH_INVARIANT or os.getenv('KAILLM_HUMMING_DETERMINISTIC') == '1'):
            return None
    elif not (supported(serving, gpu, count) or pro_udq2.PARALLEL and pro_udq2.supported(serving, gpu, count)):
        return None
    groups = tuple(tuple((name, getattr(attn, name), OWNERS[name]) for name in group if hasattr(attn, name))
                   for group in SCHEDULE)
    return serving.scheduler_config.max_num_seqs, (groups, pool(torch.cuda.current_device(), len(groups) - 1))


def project(plan, linear, inputs):
    groups, side_streams = plan
    main = torch.cuda.current_stream(device=inputs[0].device)
    for stream in side_streams:
        stream.wait_stream(main)
    outputs = {}
    for index, group in enumerate(groups):
        stream = side_streams[index - 1] if index else main
        with torch.cuda.stream(stream):
            for name, layer, source in group:
                output = linear(layer, inputs[source])
                if index:
                    output.record_stream(main)
                outputs[name] = output
    for stream in side_streams:
        main.wait_stream(stream)
    return outputs
