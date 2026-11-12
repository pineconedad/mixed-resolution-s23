import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from mixres.adapters.pi3 import Pi3Adapter
from mixres.rmfa.arms import ARMS
from mixres.rmfa.calibration import attention_bias, calibrate_groups, key_bias
from mixres.rmfa.engine import run_engine

ROOT = Path(__file__).resolve().parents[1]
MIXED = [{"hw": [56, 56], "hi_hw": [56, 56]}, {"hw": [56, 56], "hi_hw": [56, 56]},
         {"hw": [28, 28], "hi_hw": [56, 56]}, {"hw": [28, 28], "hi_hw": [56, 56]}]


def mixed_images(arm):
    torch.manual_seed(3)
    small = 28 if ARMS[arm].entry in {"small", "token"} else 56
    return [torch.rand(3, 56, 56), torch.rand(3, 56, 56),
            torch.rand(3, small, small), torch.rand(3, small, small)]


def poses(result):
    return torch.stack([view["camera_poses"] for view in result])


@pytest.mark.parametrize("arm", ["A2", "A4", "A9"])
def test_corrections_are_exact_noops_on_uniform_views(tiny_pi3, arm):
    images = [torch.rand(3, 56, 56) for _ in range(3)]
    views = [{"hw": [56, 56], "hi_hw": [56, 56]} for _ in images]
    adapter = Pi3Adapter(tiny_pi3)
    with torch.inference_mode():
        plain, _ = run_engine(adapter, images, views, arm)
        corrected, trace = run_engine(adapter, images, views, arm, calibrate="moment",
                                      logit_bias="both", head_grid="hi")
    assert torch.equal(poses(plain), poses(corrected))
    assert all(layer["group_shift"] is None for layer in trace)


def test_measure_reports_shift_without_changing_the_output(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    images = mixed_images("A2")
    with torch.inference_mode():
        plain, _ = run_engine(adapter, images, MIXED, "A2")
        measured, trace = run_engine(adapter, images, MIXED, "A2", calibrate="measure")
    assert torch.equal(poses(plain), poses(measured))
    assert all(layer["group_shift"] > 0 for layer in trace)


@pytest.mark.parametrize("options", [{"calibrate": "moment"}, {"calibrate": "moment_transient"},
                                     {"logit_bias": "global"}, {"logit_bias": "frame"},
                                     {"calibrate": "moment", "logit_bias": "both"}])
def test_each_correction_changes_the_mix_and_keeps_its_token_budget(tiny_pi3, options):
    adapter = Pi3Adapter(tiny_pi3)
    images = mixed_images("A2")
    with torch.inference_mode():
        plain, plain_trace = run_engine(adapter, images, MIXED, "A2")
        corrected, trace = run_engine(adapter, images, MIXED, "A2", **options)
    assert torch.isfinite(poses(corrected)).all()
    assert not torch.equal(poses(plain), poses(corrected))
    assert [layer["total_tokens"] for layer in trace] == [layer["total_tokens"] for layer in plain_trace]
    # The sharp reference views must be numerically untouched going into the first global block.
    assert all(view["points"].shape[:2] == (size, size)
               for view, size in zip(corrected, (56, 56, 28, 28)))


def test_a9_uses_sharp_frames_and_pooled_global_tokens(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    images = mixed_images("A9")
    with torch.inference_mode():
        low_heads, trace = run_engine(adapter, images, MIXED, "A9")
        high_heads, _ = run_engine(adapter, images, MIXED, "A9", head_grid="hi", logit_bias="global")
        full, full_trace = run_engine(adapter, images, MIXED, "A8")
    assert all(layer["patch_tokens"] == 2 * 16 + 2 * 4 for layer in trace)
    assert all(layer["frame_total_tokens"] == full_layer["frame_total_tokens"]
               for layer, full_layer in zip(trace, full_trace))
    assert all(view["points"].shape[:2] == (28, 28) for view in low_heads[2:])
    assert torch.isfinite(poses(high_heads)).all()


def test_moment_calibration_matches_reference_statistics():
    torch.manual_seed(0)
    sharp = [torch.randn(1, 2 + 16, 8) for _ in range(2)]
    small = [torch.randn(1, 2 + 4, 8) * 3 + 5 for _ in range(2)]
    tokens, grids, hi = sharp + small, [(4, 4)] * 2 + [(2, 2)] * 2, [(4, 4)] * 4
    output, shift = calibrate_groups(tokens, grids, hi, 2, "moment")
    assert shift > 1
    assert output[0] is tokens[0] and output[1] is tokens[1]
    assert all(torch.equal(new[:, :2], old[:, :2]) for new, old in zip(output[2:], small))
    reference = torch.cat([x[:, 2:] for x in sharp], 1).reshape(-1, 8)
    matched = torch.cat([x[:, 2:] for x in output[2:]], 1).reshape(-1, 8)
    assert torch.allclose(matched.mean(0), reference.mean(0), atol=1e-5)
    assert torch.allclose(matched.std(0), reference.std(0), atol=1e-5)
    unchanged, measured = calibrate_groups(tokens, grids, hi, 2, "measure")
    assert all(a is b for a, b in zip(unchanged, tokens)) and measured == shift


class TinyAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


def test_logit_bias_equals_duplicating_the_coarse_keys():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)
    block = TinyAttention()
    bias = torch.tensor([0.0, math.log(4), math.log(4)])
    with attention_bias(block, bias):
        biased = block(q, k, v)
    repeat = [0, 1, 1, 1, 1, 2, 2, 2, 2]
    duplicated = F.scaled_dot_product_attention(q, k[:, :, repeat], v[:, :, repeat])
    assert torch.allclose(biased, duplicated, atol=1e-5)
    assert torch.equal(block(q, k, v), F.scaled_dot_product_attention(q, k, v))


def test_logit_bias_fails_loudly_when_it_cannot_be_applied():
    with pytest.raises(RuntimeError, match="no attention call"):
        with attention_bias(torch.nn.Identity(), torch.zeros(3)):
            pass
    with pytest.raises(RuntimeError, match="Unsupported attention"):
        with attention_bias(TinyAttention(), torch.zeros(4)):
            TinyAttention()(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
    assert key_bias((4, 4), (4, 4), 2, "cpu") is None
    assert key_bias((2, 2), (4, 4), 2, "cpu").tolist() == pytest.approx([0, 0] + [math.log(4)] * 4)


def load_runner():
    spec = importlib.util.spec_from_file_location("salvation_run", ROOT / "scripts/04_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["salvation-a-pi3", "salvation-b-pi3", "salvation-a-vggt", "salvation-b-vggt"])
def test_salvation_configs_parse_and_accept_only_engine_options(name):
    runner = load_runner()
    config = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
    variants = [runner.parse_variant(entry) for entry in config["arms"]]
    assert len({variant[0] for variant in variants}) == len(variants)
    assert {"A0", "A2", "A4", "A3", "A8"} <= {variant[0] for variant in variants}
    assert config["study"] == f"salvation_{name.split('-')[1]}" and config["seed"] == 42
    with pytest.raises(ValueError):
        runner.parse_variant({"name": "bad", "arm": "A2", "not_an_option": 1})
    with pytest.raises(ValueError):
        runner.parse_variant({"name": "A4", "arm": "A2"})


def test_a0_resizes_every_view_to_the_equal_budget_uniform_size():
    runner = load_runner()
    views = [{"hw": [322, 420], "hi_hw": [322, 420], "sharp": True, "rho": 1.0}] * 4 + \
            [{"hw": [168, 210], "hi_hw": [322, 420], "sharp": False, "rho": 2.0}] * 8
    config = {"budget_tokens": 4096}
    assert runner.scene_views(views, "A2", config) is views
    uniform = runner.scene_views(views, "A0", config)
    assert all(view["hw"] == view["hi_hw"] == [224, 294] and view["sharp"] for view in uniform)
    mixed_tokens = sum((v["hw"][0] // 14) * (v["hw"][1] // 14) for v in views)
    uniform_tokens = sum((v["hw"][0] // 14) * (v["hw"][1] // 14) for v in uniform)
    assert (mixed_tokens, uniform_tokens) == (4200, 4032)
