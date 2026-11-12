"""Zero-cost corrections for mixed-resolution token sets (salvation track A).

Two independent, training-free hypotheses about why a mix breaks while a uniform
size does not:

* moment calibration: small views leave frame attention with different per-channel
  token statistics than sharp views, so global attention compares mismatched
  features. Match mean/std of each small-grid group to the sharp reference group.
* logit bias (proportional attention): a token on a coarser grid stands for
  several fine tokens, so its view (and, in frame attention, its patch set versus
  the register/camera tokens) receives too little softmax mass. Add
  log(fine area / coarse area) to the logits of those keys.

Both default to off, and both are exact no-ops when every view is on its sharp grid.
"""
import math
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

CALIBRATIONS = {"none", "measure", "moment", "moment_transient"}
KV_POOLS = {"none", "mix"}
LOGIT_BIASES = {"none", "global", "frame", "both"}


def _moments(patches):
    values = patches.float().reshape(-1, patches.shape[-1])
    return values.mean(0), values.std(0).clamp_min(1e-6)


def calibrate_groups(tokens, grids, hi_grids, special, mode):
    """Return (tokens, shift). The reference is all views already on their sharp grid.

    shift is the mean absolute per-channel mean offset of the non-reference patch
    tokens in units of the reference standard deviation, or None when undefined.
    """
    if mode not in CALIBRATIONS:
        raise ValueError(mode)
    reference = [i for i, (g, h) in enumerate(zip(grids, hi_grids)) if g == h]
    others = [i for i in range(len(tokens)) if i not in reference]
    if mode == "none" or not reference or not others:
        return tokens, None
    ref_mean, ref_std = _moments(torch.cat([tokens[i][:, special:] for i in reference], 1))
    groups = {}
    for i in others:
        groups.setdefault(grids[i], []).append(i)
    output = list(tokens)
    shifts = []
    for indices in groups.values():
        mean, std = _moments(torch.cat([tokens[i][:, special:] for i in indices], 1))
        shifts.append(((mean - ref_mean).abs() / ref_std).mean().item())
        if mode == "measure":
            continue
        for i in indices:
            x = tokens[i]
            patches = (x[:, special:].float() - mean) / std * ref_std + ref_mean
            output[i] = torch.cat([x[:, :special], patches.to(x.dtype)], 1)
    return output, sum(shifts) / len(shifts)


def key_bias(grid, hi_grid, special, device):
    """Per-key additive logit bias for one view, or None when it is all zero."""
    ratio = (hi_grid[0] * hi_grid[1]) / (grid[0] * grid[1])
    if ratio == 1:
        return None
    bias = torch.zeros(special + grid[0] * grid[1], device=device)
    bias[special:] = math.log(ratio)
    return bias


def asymmetric_block(block, x, xpos, kv, kvpos, bias=None):
    """Run one pre-norm transformer block with queries from `x` and keys/values from `kv`.

    Every token of `x` keeps its own update, but attends to the (smaller) `kv` set.
    Equals block(x, xpos) when kv is x. Supports the pinned Pi3 and VGGT blocks in eval mode.
    """
    inner = getattr(block, "block", block)
    attn = inner.attn
    if inner.training:
        raise RuntimeError("asymmetric_block supports inference only")
    dim, heads = x.shape[-1], attn.num_heads
    weight, offset = attn.qkv.weight, attn.qkv.bias

    def split(tokens, parts):
        return tokens.reshape(tokens.shape[0], tokens.shape[1], parts, heads, dim // heads).permute(2, 0, 3, 1, 4)

    q = split(F.linear(inner.norm1(x), weight[:dim], None if offset is None else offset[:dim]), 1)[0]
    k, v = split(F.linear(inner.norm1(kv), weight[dim:], None if offset is None else offset[dim:]), 2)
    q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
    if attn.rope is not None:
        q, k = attn.rope(q, xpos), attn.rope(k, kvpos)
    mask = None if bias is None else bias.to(device=q.device, dtype=q.dtype)[None, None, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.transpose(1, 2).reshape(x.shape)
    x = x + inner.ls1(attn.proj_drop(attn.proj(out)))
    return x + inner.ls2(inner.mlp(inner.norm2(x)))


@contextmanager
def attention_bias(block, bias):
    """Add `bias` (one value per key) to every attention inside `block`.

    Works for the pinned Pi3 and VGGT blocks, which call
    scaled_dot_product_attention without a mask. Fails loudly if the bias was
    never applied, so an unsupported model cannot silently run uncorrected.
    """
    if bias is None:
        yield
        return
    original = F.scaled_dot_product_attention
    applied = 0

    def biased(q, k, v, *args, **kwargs):
        nonlocal applied
        if args or kwargs.get("attn_mask") is not None or k.shape[-2] != bias.shape[-1]:
            raise RuntimeError("Unsupported attention call for logit bias")
        applied += 1
        kwargs["attn_mask"] = bias.to(device=q.device, dtype=q.dtype)[None, None, None]
        # The flash kernel pinned by Pi3 under bf16 rejects masks.
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            return original(q, k, v, **kwargs)

    patched = [(F, original)]
    for module in block.modules():
        source = sys.modules.get(type(module).__module__)
        if source is not None and getattr(source, "scaled_dot_product_attention", None) is original:
            if all(source is not target for target, _ in patched):
                patched.append((source, original))
    for target, _ in patched:
        target.scaled_dot_product_attention = biased
    try:
        yield
    finally:
        for target, function in patched:
            target.scaled_dot_product_attention = function
    if not applied:
        raise RuntimeError("Logit bias was requested but no attention call received it")
