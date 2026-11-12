import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mixres.adapters.da3 import DA3Adapter
from mixres.probes.cost import transformer_flops
from mixres.rmfa.arms import ARMS
from mixres.rmfa.engine import run_engine


class Block:
    def __call__(self, x, xpos):
        return x


class Adapter:
    patch_size = 2
    special_count = 1
    embed_dim = 4
    mlp_ratio = 4
    head_layers = (1,)

    def __init__(self, scalar=True, pairs=2):
        self.scalar = scalar
        self.frame_blocks = [Block() for _ in range(pairs)]
        self.global_blocks = [Block() for _ in range(pairs)]

    def embed(self, images):
        return [torch.zeros(1, (x.shape[-2] // 2) * (x.shape[-1] // 2) + 1, 4)
                for x in images]

    def run_heads(self, cache, grids):
        self.original = []
        for h, w in grids:
            values = torch.arange(h * w * 4, dtype=torch.float32).reshape(h * 2, w * 2)
            scalar = values if self.scalar else values[..., None]
            self.original.append(dict(points=values[..., None].expand(-1, -1, 3).clone(),
                                      local_points=values[..., None].expand(-1, -1, 3).clone(),
                                      conf=scalar.clone(), depth=scalar.clone(), depth_conf=scalar.clone(),
                                      intrinsics=torch.tensor([[12., 1., 5.], [0., 18., 3.], [0., 0., 1.]]),
                                      camera_poses=torch.eye(4), pose_encoding=torch.arange(9.)))
        return [dict(prediction) for prediction in self.original]


VIEWS = [{"hw": [4, 4], "hi_hw": [8, 12]}]
IMAGES = [torch.zeros(3, 8, 12)]


@pytest.mark.parametrize("arm", ["A4", "A5", "A6", "A7"])
@pytest.mark.parametrize("scalar", [True, False])
def test_hi_outputs_resize_all_pixel_fields_and_intrinsics_only(arm, scalar):
    adapter = Adapter(scalar)
    predictions, _ = run_engine(adapter, IMAGES, VIEWS, arm, head_grid="hi")
    prediction, original = predictions[0], adapter.original[0]
    for key in ("points", "local_points", "conf", "depth", "depth_conf"):
        value = original[key]
        channels = value[None] if value.ndim == 2 else value.permute(2, 0, 1)
        expected = F.interpolate(channels[None], (4, 4), mode="bilinear", align_corners=False)[0]
        expected = expected[0] if value.ndim == 2 else expected.permute(1, 2, 0)
        torch.testing.assert_close(prediction[key], expected)
    expected_k = original["intrinsics"].clone()
    expected_k[0] *= 1 / 3
    expected_k[1] *= 1 / 2
    torch.testing.assert_close(prediction["intrinsics"], expected_k)
    for key in ("camera_poses", "pose_encoding"):
        assert prediction[key] is original[key]
    assert original["intrinsics"][0, 0] == 12


@pytest.mark.parametrize("arm", list(ARMS))
def test_default_lo_preserves_head_outputs_and_adapters_need_no_da3_properties(arm):
    adapter = Adapter()
    predictions, trace = run_engine(adapter, IMAGES, VIEWS, arm)
    assert all(layer["frame_block_count"] == 1 for layer in trace)
    for key, value in adapter.original[0].items():
        assert predictions[0][key] is value


def test_equal_hi_lo_does_not_touch_outputs():
    adapter = Adapter()
    predictions, _ = run_engine(adapter, IMAGES, [{"hw": [8, 12], "hi_hw": [8, 12]}],
                                "A4", head_grid="hi")
    assert all(predictions[0][key] is value for key, value in adapter.original[0].items())


def test_flops_counts_multiply_and_add_separately_with_legacy_trace_defaults():
    trace = [dict(frame_total_tokens=5, frame_attention_quadratic=13,
                  total_tokens=5, global_attention_quadratic=25)]
    result = transformer_flops(trace, 4, 3)
    assert result["frame"] == 2 * (10 * 5 * 16 + 2 * 13 * 4)
    assert result["global"] == 2 * (10 * 5 * 16 + 2 * 25 * 4)
    assert result["heads"] is None
    assert "excludes_encoder_and_heads" in result["scope"]


@pytest.mark.parametrize("arm,layers", [("A1", "all"), ("A4", "all"), ("A4", "first_half")])
def test_da3_prefix_and_identity_costs_survive_trace_json_roundtrip(arm, layers):
    adapter = Adapter(pairs=8)
    adapter.prefix_frame_blocks = DA3Adapter.prefix_frame_blocks
    adapter.identity_frame_pairs = DA3Adapter.identity_frame_pairs
    views = VIEWS + [{"hw": [8, 12], "hi_hw": [8, 12]}]
    _, trace = run_engine(adapter, IMAGES * 2, views, arm, rmfa_layers=layers)
    assert [row["frame_block_count"] for row in trace] == [9] + [1] * 7
    assert sum(row["frame_block_count"] for row in trace) == 16
    frame = global_ = 0
    for pair in range(8):
        grids = [(2, 2), (4, 6)] if arm == "A1" or (layers == "first_half" and pair >= 4) else [(4, 6)] * 2
        lengths = [h * w + 1 for h, w in grids]
        frame += (9 if pair == 0 else 1) * (24 * sum(lengths) * 16 + 4 * sum(n * n for n in lengths) * 4)
        # Both arms use the same low-grid global sequence, including special tokens.
        global_ += 24 * 30 * 16 + 4 * 30**2 * 4
    result = transformer_flops(json.loads(json.dumps(trace)), 4)
    assert result["frame"] == frame
    assert result["global"] == global_


def test_scene_record_saves_trace_for_offline_recomputation(tmp_path, monkeypatch):
    import mixres.study_runner as study

    monkeypatch.chdir(tmp_path)
    runner = study.StudyRunner.__new__(study.StudyRunner)
    runner.model, runner.precision = "fake", "fp32"
    runner.warmup, runner.timed_passes = 1, 3
    runner.base_meta = {}
    runner.adapter = Adapter()
    predictions, trace = run_engine(runner.adapter, IMAGES, VIEWS, "A4")
    manifest = dict(dataset="eth3d", seed=42, budgets=[4096], scenes=[dict(scene="one", views=[
        dict(image="fake", image_sha256="hash", c2w=torch.eye(4).tolist(),
             sharp=False, rho=2, sizes={"4096": VIEWS[0]})])])
    path = tmp_path / "runs/views/s23/eth3d/seed42.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(study, "file_hash", lambda path: "hash")
    monkeypatch.setattr(study, "seed_everything", lambda seed: None)
    monkeypatch.setattr(runner, "cached_view", lambda *args: SimpleNamespace(cuda=lambda: None))
    monkeypatch.setattr(study, "timed_inference", lambda *args: ((predictions, trace), {}))
    monkeypatch.setattr(study, "score_poses_da3", lambda *args: {})
    result = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    record = result["per_scene"][0]
    assert record["status"] == "completed", record.get("error")
    assert record["global_trace"] == trace
    assert transformer_flops(record["global_trace"], 4) == record["flops"]
    scene = next((tmp_path / "runs/dev").rglob("scenes/one.json"))
    assert json.loads(scene.read_text())["global_trace"] == trace
