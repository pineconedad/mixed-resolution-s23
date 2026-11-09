import numpy as np
import pytest
import torch
from PIL import Image

from mixres.data.resize import load_view, resize_image, transform_intrinsics
from mixres.data.sampler import farthest_points, sample_scene
from mixres.rmfa.grid import budget_sizes, patch_grid, resize_tokens
from mixres.rmfa.positions import ContinuousRoPE, positions


@pytest.mark.parametrize("rho", [1.25, 2, 3])
@pytest.mark.parametrize("budget", [2048, 4096, 8192])
def test_t4_pixel_sizes_are_patch_multiples(rho, budget):
    hi, lo = budget_sizes(budget, 4, 8, rho, 4 / 3)
    assert all(side > 0 and side % 14 == 0 for side in (*hi, *lo))
    assert abs(hi[1] - hi[0] * 4 / 3) <= 14
    assert all(a >= b for a, b in zip(hi, lo))


@pytest.mark.parametrize("rho", [2, 3])
def test_t2_nearest_lift_area_pool_exact(rho):
    x = torch.randn(1, 12, 8)
    hi = (3 * rho, 4 * rho)
    lifted = resize_tokens(x, (3, 4), hi, 0, "nearest")
    assert torch.allclose(resize_tokens(lifted, hi, (3, 4), 0, "area"), x, atol=1e-6, rtol=0)


def test_t2_bilinear_smooth_field_has_bounded_error():
    y, x = torch.meshgrid(torch.linspace(0, 1, 24), torch.linspace(0, 1, 24), indexing="ij")
    field = (x + y).flatten().reshape(1, -1, 1)
    lo = resize_tokens(field, (24, 24), (12, 12), 0, "area")
    actual = resize_tokens(lo, (12, 12), (24, 24), 0, "bilinear")
    assert (actual - field).abs().max() < 0.05


def test_t5_special_tokens_never_pooled():
    x = torch.randn(1, 5 + 16, 8)
    result = resize_tokens(x, (4, 4), (2, 2), 5, "area")
    assert torch.equal(result[:, :5], x[:, :5])
    assert result.shape == (1, 9, 8)


def test_t6_position_scaling_keeps_special_origin_and_fractional_values():
    native = positions((3, 3), (4, 4), 5, "native", "cpu")
    aligned = positions((3, 3), (4, 4), 5, "pi_hr", "cpu")
    assert (aligned[:, :5] == 0).all()
    assert torch.allclose(aligned[:, 5:] - 1, (native[:, 5:] - 1).float() * (4 / 3))
    assert aligned.is_floating_point()


def test_fractional_rope_matches_integer_formula():
    from pi3.models.layers.pos_embed import RoPE2D
    original = RoPE2D(freq=100)
    module = ContinuousRoPE(original)
    tokens = torch.randn(1, 4, 9, 8)
    pos = positions((2, 2), (2, 2), 5, "native", "cpu")
    assert torch.equal(module(tokens, pos), original(tokens, pos))
    assert torch.allclose(module(tokens, pos.float()), original(tokens, pos), atol=1e-6)


def test_fps_and_seeded_manifest_are_reproducible():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40)
    options = dict(seed=42, scene_index=2, n_views=12, n_sharp=4, rho=2, budget=4096)
    a = sample_scene(list(map(str, range(40))), poses, **options)
    assert a == sample_scene(list(map(str, range(40))), poses, **options)
    assert len({v["frame"] for v in a}) == 12
    assert sum(v["sharp"] for v in a) == 4
    assert farthest_points([[0, 0, 0], [1, 0, 0], [3, 0, 0]], 2) == [0, 2]


def test_sampler_can_enforce_dev_frozen_disjointness():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40)
    ids = list(map(str, range(40)))
    first = sample_scene(ids, poses, seed=42, scene_index=0, n_views=12, n_sharp=4, rho=2, budget=4096)
    second = sample_scene(ids, poses, seed=1000, scene_index=0, n_views=12, n_sharp=4, rho=2, budget=4096,
                          excluded=[v["frame"] for v in first])
    assert not set(v["frame"] for v in first) & set(v["frame"] for v in second)


def test_upsampling_does_not_reload_high_resolution_information(tmp_path):
    path = tmp_path / "image.png"
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)).save(path)
    view = {"sharp": False, "hw": [28, 28], "hi_hw": [56, 56], "crop": [0, 0, 1, 1]}
    small = load_view(path, view, "small")
    pixel = load_view(path, view, "pixel")
    real = load_view(path, view, "real")
    assert torch.equal(pixel, resize_image(small, [56, 56], downsample=False))
    assert not torch.allclose(pixel, real)


def test_intrinsics_crop_and_resize():
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    transformed = transform_intrinsics(k, (100, 100), (25, 25), [0.25, 0.25, 0.75, 0.75])
    assert np.allclose(transformed, [[50, 0, 12.5], [0, 50, 12.5], [0, 0, 1]])


def test_invalid_grid_is_rejected():
    with pytest.raises(ValueError):
        patch_grid((518, 783))
