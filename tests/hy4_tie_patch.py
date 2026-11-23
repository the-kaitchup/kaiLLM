"""Make transformers' Hy4 compressed indexer break exact score ties like kaiLLM (later group first).

torch.topk orders equal values arbitrarily (and differently for different tensor sizes); exact ties
are common on the toy (4 indexer heads, ReLU zeros), so parity tests apply kaiLLM's rule to both sides.
"""
import torch


def tie_topk(self, k, dim=-1, largest=True, sorted=True):
    assert largest and dim in (-1, self.ndim - 1)
    G = self.shape[-1]
    idx = G - 1 - torch.sort(self.flip(-1), dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.return_types.topk((self.gather(-1, idx), idx))


def patch_indexer_ties(model) -> None:
    indexers = [m for m in model.modules() if type(m).__name__.endswith("CompressedIndexer")]
    assert indexers, "no compressed indexer module found"
    cls = type(indexers[0])
    if getattr(cls, "_kaillm_tie_patch", False):
        return
    fwd, real_topk = cls.forward, torch.Tensor.topk

    def patched(self, *args, **kwargs):
        torch.Tensor.topk = tie_topk
        try:
            return fwd(self, *args, **kwargs)
        finally:
            torch.Tensor.topk = real_topk

    cls.forward, cls._kaillm_tie_patch = patched, True
