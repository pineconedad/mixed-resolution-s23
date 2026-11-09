import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_image(image, size, *, downsample):
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image[None], size=tuple(size), mode="bicubic", align_corners=False,
                         antialias=downsample)[0].clamp(0, 1)


def load_view(path, view, entry):
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        left, top, right, bottom = view["crop"]
        image = image.crop((round(left * width), round(top * height),
                            round(right * width), round(bottom * height)))
        original = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    if view["sharp"] or entry == "real":
        return resize_image(original, view["hi_hw"], downsample=True)
    small = resize_image(original, view["hw"], downsample=True)
    return resize_image(small, view["hi_hw"], downsample=False) if entry == "pixel" else small


def transform_intrinsics(intrinsics, original_hw, target_hw, crop):
    h, w = original_hw
    left, top, right, bottom = crop
    x0, y0, x1, y1 = round(left * w), round(top * h), round(right * w), round(bottom * h)
    k = np.array(intrinsics, dtype=np.float64, copy=True)
    k[0, 2] -= x0
    k[1, 2] -= y0
    k[0] *= target_hw[1] / (x1 - x0)
    k[1] *= target_hw[0] / (y1 - y0)
    return k
