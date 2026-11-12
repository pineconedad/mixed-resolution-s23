import pytest
import torch
import json
from pathlib import Path
import yaml

from mixres.adapters.vggt import VGGTAdapter
from mixres.rmfa.arms import ARMS
from mixres.rmfa.engine import run_engine
from mixres.runtime import validate_views


@pytest.mark.parametrize("arm", list(ARMS))
def test_uniform_identity_against_native_vggt(tiny_vggt, arm):
    images = [torch.rand(3, 56, 56) for _ in range(3)]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]} for _ in images]
    with torch.inference_mode():
        native = tiny_vggt(torch.stack(images)[None])
        result, _ = run_engine(VGGTAdapter(tiny_vggt), images, views, arm)
    for key, expected in (("pose_encoding", "pose_enc"), ("points", "world_points"), ("depth", "depth")):
        actual = torch.stack([r[key] for r in result])
        assert (actual - native[expected][0]).abs().max() < 1e-4


@pytest.mark.parametrize("arm", list(ARMS))
def test_mixed_grid_vggt_keeps_camera_joint_and_dense_shapes(tiny_vggt, arm):
    adapter = VGGTAdapter(tiny_vggt)
    sizes = [56, 28 if ARMS[arm].entry in {"small", "token"} else 56, 56]
    images = [torch.rand(3, size, size) for size in sizes]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]},
             {"hw": [56, 56], "hi_hw": [56, 56]}]
    observed = []
    handle = tiny_vggt.camera_head.register_forward_pre_hook(lambda _, args: observed.append(args[0][-1].shape))
    with torch.inference_mode():
        results, trace = run_engine(adapter, images, views, arm)
    handle.remove()
    assert observed == [torch.Size([1, 3, 1, 64])]
    assert results[0]["points"].shape == (56, 56, 3)
    side = 56 if arm in {"A3", "A8"} else 28
    assert results[1]["points"].shape == (side, side, 3)
    assert all(t["special_tokens"] == 15 for t in trace)
    assert all(t["patch_tokens"] == (48 if arm in {"A3", "A6", "A8"} else 36) for t in trace)


def test_first_view_identity_is_not_reset_for_each_resolution_group(tiny_vggt):
    adapter = VGGTAdapter(tiny_vggt)
    with torch.inference_mode():
        tokens = adapter.embed([torch.rand(3, 28, 28), torch.rand(3, 56, 56), torch.rand(3, 56, 56)])
    agg = tiny_vggt.aggregator
    assert torch.equal(tokens[0][0, :1], agg.camera_token[0, 0])
    assert torch.equal(tokens[1][0, :1], agg.camera_token[0, 1])
    assert torch.equal(tokens[2][0, :1], agg.camera_token[0, 1])


def test_common_reference_preserves_native_camera_points_confidence_and_depth(tiny_vggt):
    adapter = VGGTAdapter(tiny_vggt)
    images = [torch.rand(3, 56, 56) for _ in range(3)]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]} for _ in images]
    with torch.inference_mode():
        native = adapter.reference(images)
        result, _ = run_engine(adapter, images, views, "A4")
    for expected, actual in zip(native, result):
        for key in expected:
            assert (expected[key] - actual[key]).abs().max() < 1e-4
        assert actual["camera_poses"].shape == (4, 4)
        assert torch.allclose(actual["camera_poses"][3], torch.tensor([0., 0., 0., 1.]), atol=1e-6)


def test_vggt_validation_reuses_the_committed_development_views():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/vggt-validation.yaml").read_text())
    config["views"] = str(root / config["views"])
    validate_views(config, json.loads(Path(config["views"]).read_text()))


@pytest.mark.parametrize("options", [{"calibrate": "moment"}, {"logit_bias": "both"},
                                     {"calibrate": "moment", "logit_bias": "both", "head_grid": "hi"}])
@pytest.mark.parametrize("arm", ["A2", "A9"])
def test_salvation_corrections_run_through_vggt_blocks(tiny_vggt, arm, options):
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]},
             {"hw": [56, 56], "hi_hw": [56, 56]}]
    sizes = [56, 28 if ARMS[arm].entry in {"small", "token"} else 56, 56]
    torch.manual_seed(5)
    images = [torch.rand(3, size, size) for size in sizes]
    adapter = VGGTAdapter(tiny_vggt)
    with torch.inference_mode():
        plain, plain_trace = run_engine(adapter, images, views, arm)
        corrected, trace = run_engine(adapter, images, views, arm, **options)
    plain_pose = torch.stack([r["pose_encoding"] for r in plain])
    pose = torch.stack([r["pose_encoding"] for r in corrected])
    assert torch.isfinite(pose).all() and not torch.equal(pose, plain_pose)
    assert [t["total_tokens"] for t in trace] == [t["total_tokens"] for t in plain_trace]
