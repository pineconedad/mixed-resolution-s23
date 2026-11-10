import torch
from torch import nn


def positions(grid, hi_grid, special_count, kind, device):
    y, x = torch.meshgrid(torch.arange(grid[0], device=device),
                          torch.arange(grid[1], device=device), indexing="ij")
    patches = torch.stack([y.flatten(), x.flatten()], dim=-1)
    if kind == "pi_hr" and grid != hi_grid:
        patches = patches.float() * torch.tensor([hi_grid[0] / grid[0], hi_grid[1] / grid[1]], device=device)
    elif kind != "native" and kind != "pi_hr":
        raise ValueError(kind)
    patches = patches + 1
    special = torch.zeros(special_count, 2, dtype=patches.dtype, device=device)
    return torch.cat([special, patches])[None]


class ContinuousRoPE(nn.Module):
    """Retain upstream integer RoPE exactly; support fractional pi_hr coordinates."""

    def __init__(self, original, base=None):
        super().__init__()
        self.original = original
        self.base = original.base if base is None else base

    def forward(self, tokens, pos):
        if not pos.is_floating_point():
            return self.original(tokens, pos)
        half = tokens.shape[-1] // 2
        frequency = 1 / (self.base ** (torch.arange(0, half, 2, device=tokens.device).float() / half))
        result = []
        for axis, component in enumerate(tokens.chunk(2, dim=-1)):
            angles = pos[..., axis, None].float() * frequency
            angles = torch.cat([angles, angles], -1).to(tokens.dtype)[:, None]
            left, right = component.chunk(2, dim=-1)
            rotated = torch.cat([-right, left], -1)
            result.append(component * angles.cos() + rotated * angles.sin())
        return torch.cat(result, dim=-1)
