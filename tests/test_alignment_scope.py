import torch

from mixres.adapters.pi3 import Pi3Adapter
from mixres.rmfa.engine import run_engine
from mixres.rmfa.positions import positions


def test_full_scope_uses_nominal_coordinates_in_frame_global_and_heads(tiny_pi3):
    adapter = Pi3Adapter(tiny_pi3)
    views = [{"hw": [56, 56], "hi_hw": [56, 56], "rho": 1.0},
             {"hw": [28, 28], "hi_hw": [56, 56], "rho": 2.0}]
    images = [torch.rand(3, 56, 56), torch.rand(3, 28, 28)]
    frame_seen, global_seen, heads_seen = [], [], []

    class Observer:
        def frame(self, layer, block, x, pos, indices, grid):
            if layer == 0 and 1 in indices:
                frame_seen.append(pos[indices.index(1):indices.index(1) + 1].clone())
        def global_input(self, *args):
            pass
        def tokens(self, *args):
            pass

    original_heads = adapter.run_heads
    def heads(cache, grids, head_positions=None):
        heads_seen.append(head_positions[1].clone())
        return original_heads(cache, grids, head_positions=head_positions)
    adapter.run_heads = heads
    handle = adapter.global_blocks[0].register_forward_pre_hook(
        lambda _, args, kwargs: global_seen.append(kwargs["xpos"][:, 21:].clone()), with_kwargs=True)
    with torch.inference_mode():
        run_engine(adapter, images, views, "A2", observer=Observer(),
                   alignment_scope="all", position_scale="nominal")
    handle.remove()
    expected = positions((2, 2), (4, 4), 5, "pi_hr", "cpu", scale=2)
    assert torch.equal(frame_seen[0], expected)
    assert torch.equal(global_seen[0], expected)
    assert torch.equal(heads_seen[0], expected)
    assert (expected[:, :5] == 0).all()
