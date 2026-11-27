"""Balance contiguous Hy4 pipeline stages by their resident GGUF weights."""
from itertools import accumulate
from pathlib import Path
import re

from ..serving_defaults import _shards


def layer_counts(blocks, embed_bytes, output_bytes, stages):
    """Minimize the largest stage including embeddings and output projections."""
    if not 1 <= stages <= len(blocks):
        raise ValueError("Hy4 needs at least one transformer block per pipeline stage")
    cumulative = [0, *accumulate(blocks)]
    # (max stage bytes, cuts) for each prefix, evaluated one pipeline stage at a time.
    costs = {0: (0, ())}
    for rank in range(stages):
        next_costs = {}
        for end in range(rank + 1, len(blocks) - (stages - rank - 1) + 1):
            options = []
            for start, (previous_max, cuts) in costs.items():
                if start >= end:
                    continue
                size = cumulative[end] - cumulative[start]
                size += embed_bytes if rank == 0 else 0
                size += output_bytes if rank + 1 == stages else 0
                options.append((max(size, previous_max), (*cuts, end - start)))
            if options:
                next_costs[end] = min(options)
        costs = next_costs
    return list(costs[len(blocks)][1])


def from_gguf(model, stages):
    import gguf

    block_bytes = {}
    embed = head = 0
    layers = None
    for shard in _shards(Path(model)):
        reader = gguf.GGUFReader(str(shard))
        if layers is None:
            layers = int(reader.get_field("hy_v4_flash.block_count").contents())
        for tensor in reader.tensors:
            if tensor.name == "per_layer_token_embd.weight":
                continue  # The table stays memory-mapped on the host.
            size = int(tensor.n_bytes)
            block = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
            if block:
                layer, name = int(block[1]), block[2]
                if layer >= layers:
                    continue  # Optional draft blocks are outside the target backbone.
                if name in ("attn_k_b.weight", "attn_v_b.weight"):
                    size = int(tensor.n_elements) * 2  # loaded into the combined BF16 projection
                block_bytes[layer] = block_bytes.get(layer, 0) + size
            elif tensor.name == "token_embd.weight":
                embed += size
            else:
                head += size
    if sorted(block_bytes) != list(range(layers or 0)) or not embed or not head:
        raise ValueError("Incomplete Hy4 GGUF pipeline weight inventory")
    return layer_counts([block_bytes[i] for i in range(len(block_bytes))], embed, head, stages)
