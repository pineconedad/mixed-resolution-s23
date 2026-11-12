from types import SimpleNamespace

import pytest
import torch

from mixres.adapters.da3 import DA3Adapter, GlobalBlock, _depth_points, load_da3
from mixres.rmfa.engine import run_engine


class CaptureBlock(torch.nn.Module):
    def forward(self, x, pos=None):
        self.pos = pos
        return x


def test_da3_global_positions_are_nondifferentiating():
    captured = CaptureBlock()
    block = GlobalBlock(captured)
    x = torch.zeros(1, 4, 3)
    positions = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [1.5, 2.0], [2.0, 1.0]]])
    block(x, positions)
    assert captured.pos.tolist() == [[[0, 0], [1, 1], [1, 1], [1, 1]]]
    assert captured.pos.dtype == torch.long


def test_depth_points_respect_intrinsics_and_pose():
    depth = torch.ones(2, 2)
    intrinsics = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    c2w = torch.eye(4)
    c2w[0, 3] = 3
    local, world = _depth_points(depth, intrinsics, c2w)
    assert torch.equal(local[1, 1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(world[1, 1], torch.tensor([4.0, 1.0, 1.0]))


@pytest.fixture
def native_adapter():
    pytest.importorskip("depth_anything_3")
    from addict import Dict
    from depth_anything_3.model.cam_dec import CameraDec
    from depth_anything_3.model.da3 import DepthAnything3Net
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = DinoVisionTransformer(
                img_size=28, patch_size=14, embed_dim=16, depth=24, num_heads=4,
                alt_start=8, rope_start=8, qknorm_start=8, cat_token=True)

        def forward(self, x, **kwargs):
            return self.pretrained.get_intermediate_layers(x, [11, 15, 19, 23], **kwargs)

    class Head(torch.nn.Module):
        def forward(self, feats, height, width, patch_start_idx):
            assert not torch.is_autocast_enabled("cpu")
            assert all(patches.dtype == cameras.dtype == torch.float32
                       for patches, cameras in feats)
            assert all(cameras.ndim == 3 for _, cameras in feats)
            self.feats = feats
            batch, views = feats[0][0].shape[:2]
            depth = torch.nn.functional.interpolate(
                feats[-1][0].mean(-1).reshape(batch * views, 1, height // 14, width // 14),
                size=(height, width), mode="nearest").reshape(batch, views, height, width).exp()
            return Dict(depth=depth, depth_conf=torch.ones_like(depth))

    torch.manual_seed(7)
    model = DepthAnything3Net(Backbone(), Head(), CameraDec(32), torch.nn.Identity())
    return DA3Adapter(model)


@pytest.mark.parametrize("views", [1, 3])
def test_native_reference_and_engine_identity(native_adapter, views):
    images = list(torch.rand(views, 3, 28, 28).unbind())
    metadata = [{"hw": [28, 28], "hi_hw": [28, 28], "rho": 1} for _ in images]
    with torch.inference_mode():
        expected = native_adapter.reference(images)
        expected_feats = native_adapter.model.head.feats
        actual, _ = run_engine(native_adapter, images, metadata, "A1")
    for expected_pair, actual_pair in zip(expected_feats, native_adapter.model.head.feats):
        for expected_tensor, actual_tensor in zip(expected_pair, actual_pair):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)
    for reference, prediction in zip(expected, actual):
        for key in reference:
            torch.testing.assert_close(prediction[key], reference[key], rtol=1e-5, atol=1e-6)


def test_reference_normalizes_exactly_like_embedding():
    adapter = DA3Adapter.__new__(DA3Adapter)
    adapter.register_buffer = {
        "mean": torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        "std": torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
    }
    captured = {}

    def prepare(pixels):
        captured["embed"] = pixels
        return torch.zeros(1, pixels.shape[1], 5, 4)

    def forward(pixels, ref_view_strategy):
        assert ref_view_strategy == "first"
        captured["reference"] = pixels
        return pixels

    adapter.backbone = SimpleNamespace(prepare_tokens_with_masks=prepare)
    adapter.model = forward
    adapter._results_from_output = lambda output: output
    images = list(torch.rand(3, 3, 28, 28).unbind())
    adapter.embed(images)
    adapter.reference(images)
    expected = ((torch.stack(images) - adapter.register_buffer["mean"])
                / adapter.register_buffer["std"])[None]
    torch.testing.assert_close(captured["embed"], expected, rtol=0, atol=0)
    torch.testing.assert_close(captured["reference"], expected, rtol=0, atol=0)


def test_heads_disable_autocast_and_promote_cached_features(native_adapter):
    cache = {layer: [torch.rand(1, 5, 16).bfloat16() for _ in range(3)]
             for layer in native_adapter.head_layers}
    seen = []

    def check_camera(module, args):
        assert args[0].shape == (1, 3, 32)
        assert args[0].dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        seen.append(True)

    handle = native_adapter.model.cam_dec.register_forward_pre_hook(check_camera)
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            results = native_adapter.run_heads(cache, [(2, 2)] * 3)
    finally:
        handle.remove()
    assert seen == [True]
    assert all(value.dtype == torch.float32 for result in results for value in result.values())


def test_heads_preserve_group_order_and_native_sky_processing(native_adapter, monkeypatch):
    grids = [(2, 2), (2, 4), (2, 2)]
    cache = {layer: [torch.rand(1, height * width + 1, 16) for height, width in grids]
             for layer in native_adapter.head_layers}
    seen = []

    def process_sky(output):
        seen.append(output.depth.shape)
        output.depth = torch.full_like(output.depth, len(seen))
        return output

    monkeypatch.setattr(native_adapter.model, "_process_mono_sky_estimation", process_sky)
    with torch.inference_mode():
        results = native_adapter.run_heads(cache, grids)
    assert seen == [torch.Size([1, 2, 28, 28]), torch.Size([1, 1, 28, 56])]
    assert [result["depth"].shape for result in results] == [(28, 28), (28, 56), (28, 28)]
    assert [result["depth"][0, 0].item() for result in results] == [1, 2, 1]


@pytest.mark.parametrize("omission", ["none", "aliases", "unshared", "required", "unexpected"])
def test_checkpoint_missing_weights_policy(native_adapter, monkeypatch, tmp_path, omission):
    import depth_anything_3.cfg
    import safetensors.torch

    model = native_adapter.model
    norm = torch.nn.LayerNorm(2)
    model.head.scratch = torch.nn.Module()
    model.head.scratch.output_conv2_aux = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                            torch.nn.LayerNorm(2) if omission == "unshared" else norm)
        for _ in range(4)
    ])
    weights = {f"model.{key}": value for key, value in model.state_dict().items()}
    if omission in {"aliases", "unshared"}:
        for level in (1, 2, 3):
            for field in ("weight", "bias"):
                del weights[f"model.head.scratch.output_conv2_aux.{level}.2.{field}"]
    elif omission == "required":
        del weights["model.backbone.pretrained.cls_token"]
    elif omission == "unexpected":
        weights["model.unknown"] = torch.zeros(1)
    monkeypatch.setattr(depth_anything_3.cfg, "create_object", lambda config: model)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: weights)
    (tmp_path / "config.json").write_text('{"config": {}}')
    if omission in {"unshared", "required", "unexpected"}:
        with pytest.raises(RuntimeError, match="Unexpected DA3 state mismatch"):
            load_da3(tmp_path, "cpu")
    else:
        assert load_da3(tmp_path, "cpu").model is model
