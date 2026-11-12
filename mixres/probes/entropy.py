"""Exact pre-dropout frame entropy, averaged over queries and divided by log(N).

Special tokens participate as both queries and keys. DA3 prefix records use
scope='prefix' and native prefix indices 0..8; scope='frame' uses engine indices.
Those layer numbers overlap and must not be combined without the scope label.
"""

import math

import torch


def _kind(module):
    return type(module).__module__, type(module).__name__


@torch.no_grad()
def frame_attention_entropy(block, x, pos, query_chunk_size=128):
    """Return a CPU numpy array [views, heads], or None for DA3 IdentityBlock.

    Reconstruct the pinned backbones' unmasked attention distribution, not a
    sample or an estimate from attention outputs. Score storage is O(B*H*chunk*N).
    Unknown block/attention implementations are deliberately unsupported.
    """
    if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer")
    wrapper = _kind(block)
    if wrapper == ("mixres.adapters.da3", "IdentityBlock"):
        return None
    if wrapper in {("mixres.adapters.vggt", "PositionBlock"),
                   ("mixres.adapters.da3", "LocalBlock")}:
        if wrapper[0] == "mixres.adapters.da3" and not block.use_rope:
            pos = None
        block = block.block
    supported_blocks = {
        ("pi3.models.layers.block", "BlockRope"),
        ("vggt.layers.block", "Block"),
        ("depth_anything_3.model.dinov2.layers.block", "Block"),
    }
    supported_attention = {
        ("pi3.models.layers.attention", "AttentionRope"),
        ("pi3.models.layers.attention", "FlashAttentionRope"),
        ("pi3.models.layers.attention", "MemEffAttentionRope"),
        ("vggt.layers.attention", "Attention"),
        ("depth_anything_3.model.dinov2.layers.attention", "Attention"),
    }
    if _kind(block) not in supported_blocks or _kind(block.attn) not in supported_attention:
        raise NotImplementedError(f"Unsupported exact frame attention: {_kind(block)}")
    attn = block.attn
    if block.training or attn.training:
        raise NotImplementedError("Exact entropy observer requires evaluation mode")
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("Expected nonempty [views, tokens, channels] input")
    batch, count, channels = x.shape
    heads = attn.num_heads
    if channels % heads:
        raise NotImplementedError("Unsupported QKV head layout")
    qkv = attn.qkv(block.norm1(x)).reshape(batch, count, 3, heads, channels // heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    pi3 = _kind(attn)[0] == "pi3.models.layers.attention"
    if pi3:
        q, k = q.to(v.dtype), k.to(v.dtype)
    if attn.rope is not None:
        if pos is not None or pi3:
            q, k = attn.rope(q, pos), attn.rope(k, pos)
        elif _kind(attn)[0] == "vggt.layers.attention":
            raise NotImplementedError("VGGT RoPE requires frame positions")
    # SDPA/xFormers use the head-dimension scale, not the mutable scale field.
    fused = _kind(attn)[1] != "AttentionRope" if pi3 else attn.fused_attn
    scale = (channels // heads) ** -0.5 if fused else attn.scale
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        q, k = q.to(dtype), k.to(dtype)
        total = torch.zeros(batch, heads, device=x.device, dtype=dtype)
        if count > 1:
            for start in range(0, count, query_chunk_size):
                scores = (q[:, :, start:start + query_chunk_size] * scale) @ k.transpose(-2, -1)
                logp = scores.log_softmax(dim=-1)
                total += -(logp.exp() * logp).sum(dim=-1).sum(dim=-1)
            total /= count * math.log(count)
    # For N=1 the distribution has zero entropy; define its normalized value as 0.
    return total.cpu().numpy()


class EntropyObserver:
    def __init__(self, query_chunk_size=128):
        if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")
        self.query_chunk_size = query_chunk_size
        self.records = []
        self._seen_global = False

    def frame(self, layer, block, x, pos, indices, grid):
        if len(indices) != x.shape[0] or x.shape[1] < math.prod(grid):
            raise ValueError("Frame metadata does not match tokens")
        entropy = frame_attention_entropy(block, x, pos, self.query_chunk_size)
        if entropy is None:
            return
        scope = ("prefix" if _kind(block) == ("mixres.adapters.da3", "LocalBlock")
                 and not self._seen_global else "frame")
        for index, values in zip(indices, entropy):
            self.records.append({"layer": layer, "scope": scope, "view": index,
                                 "grid": tuple(grid), "token_count": x.shape[1],
                                 "entropy": values.copy()})

    def global_input(self, layer, tokens, grids):
        self._seen_global = True

    def tokens(self, layer, tokens, grids, lo, special):
        pass
