import pytest
import torch

from mixres.adapters.pi3 import Pi3Adapter
from mixres.rmfa.arms import ARMS
from mixres.rmfa.engine import run_engine
from mixres.rmfa.grid import resize_tokens


@pytest.mark.parametrize("arm", list(ARMS))
def test_t1_uniform_matches_original_forward(tiny_pi3, arm):
    images = [torch.rand(3, 56, 56) for _ in range(3)]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]} for _ in images]
    with torch.inference_mode():
        reference = tiny_pi3(torch.stack(images)[None])
        adapter = Pi3Adapter(tiny_pi3)
        result, trace = run_engine(adapter, images, views, arm)
    for key in ("camera_poses", "points", "local_points", "conf"):
        assert (reference[key][0] - torch.stack([r[key] for r in result])).abs().max() < 1e-4
    assert all(layer["patch_tokens"] == 48 for layer in trace)
    assert all(layer["special_tokens"] == 15 for layer in trace)


@pytest.mark.parametrize("arm", list(ARMS))
def test_t3_mixed_shapes_and_global_budget(tiny_pi3, arm):
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]}]
    image_size = 56 if ARMS[arm].entry in {"pixel", "real"} else 28
    images = [torch.rand(3, 56, 56), torch.rand(3, image_size, image_size)]
    with torch.inference_mode():
        result, trace = run_engine(Pi3Adapter(tiny_pi3), images, views, arm)
    expected = 32 if arm in {"A0", "A3", "A6", "A8", "A10"} else 20
    assert all(layer["patch_tokens"] == expected for layer in trace)
    assert all(layer["total_tokens"] == expected + 10 for layer in trace)
    assert all(torch.isfinite(r["camera_poses"]).all() for r in result)


def test_t7_first_frame_sharp_view_is_unchanged(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]}]
    sharp = torch.rand(3, 56, 56)
    small = torch.rand(3, 28, 28)
    observations = []

    class Observer:
        def frame(self, layer, block, x, pos, indices, grid):
            if layer == 0 and 0 in indices:
                observations.append(block(x, xpos=pos)[indices.index(0)].clone())
        def global_input(self, *args):
            pass
        def tokens(self, *args):
            pass

    with torch.inference_mode():
        run_engine(adapter, [sharp, small], views, "A2", observer=Observer())
        upsampled = torch.nn.functional.interpolate(small[None], size=(56, 56), mode="bicubic", align_corners=False)[0]
        run_engine(adapter, [sharp, upsampled], views, "A4", observer=Observer())
    assert torch.allclose(*observations, atol=1e-6, rtol=0)


def test_lift_updates_does_not_replace_high_frequency_state():
    # The lifted global update must preserve hi details lost by pooling.
    hi = torch.tensor([[[1.0], [-1.0], [-1.0], [1.0]]])
    lo = resize_tokens(hi, (2, 2), (1, 1), 0, "area")
    delta = torch.ones_like(lo)
    lifted = hi + resize_tokens(delta, (1, 1), (2, 2), 0, "bilinear")
    assert torch.equal(lifted, hi + 1)
    assert not torch.equal(lifted, resize_tokens(lo + delta, (1, 1), (2, 2), 0, "bilinear"))


def test_hi_heads_read_persistent_detail_not_lifted_pooled_tokens():
    class Block:
        def __init__(self, update):
            self.update = update

        def __call__(self, x, xpos):
            return x + self.update

    class Adapter:
        patch_size = 1
        special_count = 0
        frame_blocks = [Block(0)]
        global_blocks = [Block(1)]
        head_layers = (1,)

        def embed(self, images):
            return [torch.tensor([[[1.0], [-1.0], [-1.0], [1.0]]])]

        def run_heads(self, cache, grids):
            self.cached = cache[1][0].clone()
            return [{"points": torch.zeros(2, 2, 3), "local_points": torch.zeros(2, 2, 3),
                     "conf": torch.zeros(2, 2, 1)}]

    adapter = Adapter()
    run_engine(adapter, [torch.zeros(3, 2, 2)],
               [{"hw": [1, 1], "hi_hw": [2, 2]}], "A4", head_grid="hi")
    assert torch.equal(adapter.cached, torch.tensor([[[2.0], [0.0], [0.0], [2.0]]]))


def test_t8_repeat_is_bit_identical(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    images = [torch.rand(3, 56, 56), torch.rand(3, 56, 56)]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]}]
    with torch.inference_mode():
        first, _ = run_engine(adapter, images, views, "A4")
        second, _ = run_engine(adapter, images, views, "A4")
    for a, b in zip(first, second):
        for key in a:
            assert torch.equal(a[key], b[key])


def test_fresh_uniform_baseline_uses_native_uniform_grid(tiny_pi3):
    images = [torch.rand(3, 42, 56) for _ in range(2)]
    views = [{"hw": [42, 56], "hi_hw": [42, 56]} for _ in images]
    result, trace = run_engine(Pi3Adapter(tiny_pi3), images, views, "A0")
    assert len(result) == 2
    assert all(layer["patch_tokens"] == 24 for layer in trace)


@pytest.mark.parametrize("layers", ["all", "first_half", "first_quarter"])
@pytest.mark.parametrize("head_grid", ["lo", "hi"])
def test_layer_and_head_variants_keep_declared_output_resolution(tiny_pi3, layers, head_grid):
    views = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]}]
    images = [torch.rand(3, 56, 56), torch.rand(3, 56, 56)]
    with torch.inference_mode():
        result, trace = run_engine(Pi3Adapter(tiny_pi3), images, views, "A4", rmfa_layers=layers, head_grid=head_grid)
    assert result[0]["points"].shape == (56, 56, 3)
    assert result[1]["points"].shape == (28, 28, 3)
    assert all(layer["patch_tokens"] == 20 for layer in trace)
