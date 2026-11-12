import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mixres.adapters.da3 import IdentityBlock, LocalBlock
from mixres.adapters.pi3 import Pi3Adapter
from mixres.adapters.vggt import PositionBlock
from mixres.probes.drift import DriftObserver
from mixres.probes.entropy import EntropyObserver, frame_attention_entropy
from mixres.rmfa.engine import run_engine
from mixres.rmfa.positions import ContinuousRoPE, positions


@pytest.fixture
def da3_layers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]
                                   / "third_party/Depth-Anything-3/src"))
    block = pytest.importorskip("depth_anything_3.model.dinov2.layers.block",
                               reason="DA3 optional dependencies are required", exc_type=ImportError)
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    return block.Block, RotaryPositionEmbedding2D


@pytest.mark.parametrize("backbone", ["pi3", "vggt", "da3"])
@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_entropy_matches_dense_actual_attention(backbone, chunk, request):
    torch.manual_seed(21)
    if backbone == "pi3":
        from pi3.models.layers.attention import FlashAttentionRope
        from pi3.models.layers.block import BlockRope
        from pi3.models.layers.pos_embed import RoPE2D

        raw = BlockRope(dim=16, num_heads=2, qk_norm=True,
                        attn_class=FlashAttentionRope, rope=ContinuousRoPE(RoPE2D(freq=100))).eval()
        block = raw
    elif backbone == "vggt":
        from vggt.layers.block import Block
        from vggt.layers.rope import RotaryPositionEmbedding2D

        raw = Block(dim=16, num_heads=2, qk_norm=True,
                    rope=ContinuousRoPE(RotaryPositionEmbedding2D(), base=100)).eval()
        block = PositionBlock(raw)
    else:
        Block, RotaryPositionEmbedding2D = request.getfixturevalue("da3_layers")
        raw = Block(dim=16, num_heads=2, qk_norm=True,
                    rope=RotaryPositionEmbedding2D()).eval()
        block = LocalBlock(raw)
    x = torch.randn(2, 7, 16)
    kind = "native" if backbone == "da3" else "pi_hr"
    pos = positions((2, 3), (4, 6), 1, kind, x.device).expand(2, -1, -1)
    attn = raw.attn
    with torch.no_grad():
        q, k, _ = attn.qkv(raw.norm1(x)).reshape(2, 7, 3, 2, 8).permute(2, 0, 3, 1, 4)
        q, k = attn.rope(attn.q_norm(q), pos), attn.rope(attn.k_norm(k), pos)
        p = ((q * 8 ** -0.5) @ k.transpose(-2, -1)).softmax(-1)
        expected = -(p * p.log()).sum(-1).mean(-1) / math.log(7)
    observer = EntropyObserver(chunk)
    original = x.clone()
    observer.frame(4, block, x, pos, [3, 1], (2, 3))
    actual = np.stack([r["entropy"] for r in observer.records])
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-7)
    assert [r["view"] for r in observer.records] == [3, 1]
    assert all(r["token_count"] == 7 for r in observer.records)
    assert all(r["scope"] == ("prefix" if backbone == "da3" else "frame")
               for r in observer.records)
    assert torch.equal(x, original)


def test_entropy_uniform_singleton_and_unsupported(tiny_pi3):
    block = Pi3Adapter(tiny_pi3).frame_blocks[0]
    with torch.no_grad():
        block.attn.qkv.weight.zero_()
    x = torch.randn(1, 6, 32, requires_grad=True)
    pos = positions((1, 1), (1, 1), 5, "native", x.device)
    np.testing.assert_allclose(frame_attention_entropy(block, x, pos, 2), 1, atol=1e-6)
    np.testing.assert_array_equal(frame_attention_entropy(block, x[:, :1], pos[:, :1]), 0)
    with pytest.raises(NotImplementedError, match="Unsupported"):
        frame_attention_entropy(nn.Identity(), x, pos)
    with pytest.raises(ValueError, match="positive"):
        EntropyObserver(0)
    block.train()
    with pytest.raises(NotImplementedError, match="evaluation"):
        frame_attention_entropy(block, x, pos)


def test_da3_identity_skip():
    observer = EntropyObserver(2)
    x = torch.randn(1, 6, 32)
    pos = positions((1, 1), (1, 1), 5, "native", x.device)
    observer.frame(0, IdentityBlock(), x, pos, [0], (1, 1))
    assert not observer.records


def test_da3_prefix_labels(da3_layers):
    Block, _ = da3_layers
    observer = EntropyObserver(2)
    x = torch.randn(1, 6, 32)
    pos = positions((1, 1), (1, 1), 5, "native", x.device)
    # The wrapper disables RoPE for the local prefix even though positions exist.
    raw = Block(dim=32, num_heads=4).eval()
    local = LocalBlock(raw, use_rope=False)
    observer.frame(2, local, x, pos, [0], (1, 1))
    observer.global_input(1, [x], [(1, 1)])
    observer.frame(2, LocalBlock(raw), x, pos, [0], (1, 1))
    assert [r["scope"] for r in observer.records] == ["prefix", "frame"]


def test_drift_pools_patches_to_saved_a3_and_preserves_views():
    baseline = DriftObserver()
    first = torch.tensor([[[99., 99.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]]])
    second = torch.tensor([[[88., 88.], [1., 1.]]])
    baseline.tokens(1, [first, second], [(2, 2), (1, 1)], [(1, 2), (1, 1)], 1)
    np.testing.assert_array_equal(baseline.saved_tokens[1, 0]["tokens"], [[.5, .5], [.5, .5]])
    candidate = DriftObserver(baseline.saved_tokens)
    low = torch.tensor([[[-999., -999.], [.5, .5], [.5, .5]]])
    candidate.tokens(1, [low, -second], [(1, 2), (1, 1)], [(1, 2), (1, 1)], 1)
    np.testing.assert_allclose(candidate.records[0]["cosine"], 1, atol=1e-6)
    np.testing.assert_allclose(candidate.records[1]["cosine"], -1, atol=1e-6)
    assert [r["view"] for r in candidate.records] == [0, 1]
    first.zero_()
    np.testing.assert_array_equal(baseline.saved_tokens[1, 0]["tokens"], [[.5, .5], [.5, .5]])


def test_drift_rejects_missing_or_mismatched_reference():
    x = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="Missing A3"):
        DriftObserver({}).tokens(1, [x], [(1, 2)], [(1, 2)], 0)
    reference = {(1, 0): {"grid": (2, 1), "tokens": np.ones((2, 4))}}
    with pytest.raises(ValueError, match="mismatch"):
        DriftObserver(reference).tokens(1, [x], [(1, 2)], [(1, 2)], 0)
    with pytest.raises(ValueError, match="matching"):
        DriftObserver().tokens(1, [x], [(2, 2)], [(1, 1)], 0)


def test_observers_accept_engine_hooks_without_changing_predictions(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    images = [torch.rand(3, 28, 28), torch.rand(3, 28, 28)]
    views = [{"hw": [28, 28], "hi_hw": [28, 28]} for _ in images]
    baseline = DriftObserver()
    entropy = EntropyObserver(3)
    with torch.inference_mode():
        expected, _ = run_engine(adapter, images, views, "A3", observer=baseline)
        actual, _ = run_engine(adapter, images, views, "A3", observer=entropy)
        drift = DriftObserver(baseline.saved_tokens)
        run_engine(adapter, images, views, "A3", observer=drift)
    assert len(entropy.records) == 4
    assert len(drift.records) == 4
    for record in drift.records:
        np.testing.assert_allclose(record["cosine"], 1, atol=1e-6)
    for a, b in zip(expected, actual):
        for key in a:
            assert torch.equal(a[key], b[key])
