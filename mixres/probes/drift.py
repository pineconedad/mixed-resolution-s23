"""Per-patch cosine similarity to saved A3 tokens on each view's lo grid."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from mixres.rmfa.grid import resize_tokens


class DriftObserver:
    """Capture A3 with reference=None, then pass its saved_tokens to other arms.

    saved_tokens maps (engine layer, view) to {'grid': lo, 'tokens': CPU ndarray}.
    Only patch tokens enter the comparison. records contains per-patch cosine
    arrays and their mean, not one minus cosine. Use one observer per engine run.
    Reference capture records contain tokens instead of cosine values.
    """

    def __init__(self, reference=None, pool_op="area"):
        self.reference = reference
        self.pool_op = pool_op
        self.saved_tokens = {}
        self.records = []

    def frame(self, layer, block, x, pos, indices, grid):
        pass

    def global_input(self, layer, tokens, grids):
        pass

    @torch.no_grad()
    def tokens(self, layer, tokens, grids, lo, special):
        if not (len(tokens) == len(grids) == len(lo)) or special < 0:
            raise ValueError("Invalid per-view token metadata")
        for view, (hidden, source, target) in enumerate(zip(tokens, grids, lo)):
            source, target = tuple(source), tuple(target)
            if (len(source) != 2 or len(target) != 2 or min(*source, *target) <= 0
                    or any(t > s for s, t in zip(source, target))):
                raise ValueError("Drift requires pooling to a valid lo grid")
            if (hidden.ndim != 3 or hidden.shape[0] != 1
                    or hidden.shape[1] != special + math.prod(source)):
                raise ValueError("Expected one view with matching patch and special tokens")
            patches = resize_tokens(hidden.detach().float(), source, target,
                                    special, self.pool_op)[0, special:].cpu().clone()
            key = (layer, view)
            saved = {"grid": target, "tokens": patches.numpy()}
            record = {"layer": layer, "view": view, "grid": target}
            if self.reference is None:
                record["tokens"] = saved["tokens"]
            else:
                if key not in self.reference:
                    raise ValueError(f"Missing A3 reference tokens for {key}")
                reference = self.reference[key]
                baseline = torch.as_tensor(np.asarray(reference["tokens"]), dtype=patches.dtype)
                if tuple(reference["grid"]) != target or baseline.shape != patches.shape:
                    raise ValueError(f"A3 reference grid/token shape mismatch for {key}")
                cosine = F.cosine_similarity(patches, baseline, dim=-1)
                record.update(cosine=cosine.numpy(), mean_cosine=cosine.mean().item())
            self.saved_tokens[key] = saved
            self.records.append(record)
