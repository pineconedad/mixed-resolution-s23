import pytest
import torch
from torch import nn


@pytest.fixture
def tiny_vggt():
    from vggt.models.vggt import VGGT
    from vggt.models.aggregator import Aggregator
    from vggt.heads.camera_head import CameraHead
    from vggt.heads.dpt_head import DPTHead

    torch.manual_seed(11)
    model = VGGT.__new__(VGGT)
    nn.Module.__init__(model)
    model.aggregator = Aggregator(img_size=56, patch_size=14, embed_dim=32, depth=4,
                                 num_heads=4, patch_embed="conv", cached_layer_indices=(0, 1, 2, 3))
    model.camera_head = CameraHead(dim_in=64, trunk_depth=1, num_heads=4)
    options = dict(dim_in=64, patch_size=14, features=16, out_channels=[16, 32, 32, 32],
                   intermediate_layer_idx=[0, 1, 2, 3])
    model.point_head = DPTHead(**options, output_dim=4, activation="inv_log", conf_activation="expp1")
    model.depth_head = DPTHead(**options, output_dim=2, activation="exp", conf_activation="expp1")
    model.track_head = None
    return model.eval()
