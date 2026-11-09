import math

import torch
import torch.nn.functional as F


def round_to_patch(value, patch=14):
    return max(patch, int(math.floor(value / patch + 0.5)) * patch)


def budget_sizes(budget, n_sharp, n_small, rho, aspect, patch=14):
    if min(budget, n_sharp, rho, aspect) <= 0 or n_small < 0:
        raise ValueError("Invalid budget or mix")
    tokens = budget / (n_sharp + n_small / rho**2)
    h = patch * math.sqrt(tokens / aspect)
    w = h * aspect
    hi = (round_to_patch(h, patch), round_to_patch(w, patch))
    lo = tuple(round_to_patch(side / rho, patch) for side in hi)
    return hi, lo


def patch_grid(hw, patch=14):
    if len(hw) != 2 or any(side <= 0 or side % patch for side in hw):
        raise ValueError(f"Invalid pixel size {hw}")
    return tuple(side // patch for side in hw)


def resize_tokens(tokens, source, target, special_count, mode):
    if source == target:
        return tokens
    if tokens.shape[1] != special_count + math.prod(source):
        raise ValueError("Token length does not match source grid")
    special, patches = tokens[:, :special_count], tokens[:, special_count:]
    field = patches.transpose(1, 2).reshape(tokens.shape[0], -1, *source)
    if mode == "area":
        field = F.adaptive_avg_pool2d(field, target)
    else:
        field = F.interpolate(field, size=target, mode=mode,
                              **({"align_corners": False} if mode in {"bilinear", "bicubic"} else {}))
    return torch.cat([special, field.flatten(2).transpose(1, 2)], dim=1)
