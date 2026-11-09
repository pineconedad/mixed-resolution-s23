import os
os.environ["XFORMERS_DISABLED"] = "1"

import pytest
import torch
from torch import nn


@pytest.fixture
def tiny_pi3():
    """Upstream Pi3.forward/decode and real blocks/heads, scaled down for CPU tests."""
    from pi3.models.pi3 import Pi3
    from pi3.models.dinov2.models.vision_transformer import DinoVisionTransformer
    from pi3.models.layers.attention import FlashAttentionRope
    from pi3.models.layers.block import BlockRope
    from pi3.models.layers.camera_head import CameraHead
    from pi3.models.layers.pos_embed import RoPE2D, PositionGetter
    from pi3.models.layers.transformer_head import LinearPts3d, TransformerDecoder

    torch.manual_seed(7)
    model = Pi3.__new__(Pi3)
    nn.Module.__init__(model)
    model.patch_size = 14
    model.pos_type = "rope100"
    model.rope = RoPE2D(freq=100)
    model.position_getter = PositionGetter()
    model.encoder = DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=32,
                                          depth=2, num_heads=4, num_register_tokens=4, block_chunks=0)
    model.decoder = nn.ModuleList([BlockRope(dim=32, num_heads=4, attn_class=FlashAttentionRope,
                                             rope=model.rope, qk_norm=True, init_values=0.01) for _ in range(4)])
    model.patch_start_idx = 5
    model.register_token = nn.Parameter(torch.randn(1, 1, 5, 32) * 1e-6)
    for name in ("point", "conf", "camera"):
        setattr(model, f"{name}_decoder", TransformerDecoder(in_dim=64, out_dim=32,
                 dec_embed_dim=32, depth=1, dec_num_heads=4, rope=model.rope))
    model.point_head = LinearPts3d(patch_size=14, dec_embed_dim=32, output_dim=3)
    model.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=32, output_dim=1)
    model.camera_head = CameraHead(dim=32)
    model.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    model.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return model.eval()
