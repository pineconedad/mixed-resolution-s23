"""DA3-Large adapter for its local prefix and conditional alternating backbone."""

from collections import defaultdict

import torch
from torch import nn

from mixres.rmfa.engine import frame_forward
from mixres.rmfa.positions import positions


class LocalBlock(nn.Module):
    def __init__(self, block, use_rope=True):
        super().__init__()
        self.block = block
        self.use_rope = use_rope

    @property
    def norm1(self):
        return self.block.norm1

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x, xpos):
        return self.block(x, pos=xpos if self.use_rope else None)


class GlobalBlock(LocalBlock):
    def forward(self, x, xpos):
        # DA3 intentionally removes patch-position differences in global RoPE.
        nodiff = torch.ones_like(xpos, dtype=torch.long)
        special = (xpos == 0).all(dim=-1, keepdim=True)
        nodiff = torch.where(special, torch.zeros_like(nodiff), nodiff)
        return self.block(x, pos=nodiff)


class IdentityBlock(nn.Module):
    def forward(self, x, xpos):
        return x


def _homogeneous_inverse(w2c):
    homogeneous = torch.eye(4, device=w2c.device, dtype=w2c.dtype).expand(*w2c.shape[:-2], 4, 4).clone()
    homogeneous[..., :3, :] = w2c
    return torch.linalg.inv(homogeneous)


def _depth_points(depth, intrinsics, c2w):
    height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                          torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij")
    local = torch.stack([(x - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
                         (y - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
                         depth], dim=-1)
    homogeneous = torch.cat([local, torch.ones_like(local[..., :1])], dim=-1)
    world = torch.einsum("ij,hwj->hwi", c2w, homogeneous)[..., :3]
    return local, world


class DA3Adapter:
    patch_size = 14
    embed_dim = 1024
    mlp_ratio = 4
    prefix_frame_blocks = 9
    identity_frame_pairs = (0,)

    def __init__(self, model):
        self.model = model.eval()
        backbone = self.model.backbone.pretrained
        if len(backbone.blocks) != 24 or backbone.alt_start != 8 or backbone.rope_start != 8:
            raise ValueError("Only the pinned DA3-Large layout is supported")
        if backbone.num_register_tokens != 0 or not backbone.cat_token:
            raise ValueError("Unexpected DA3-Large special-token or head layout")
        self.backbone = backbone
        self.special_count = 1
        self.prefix_blocks = [LocalBlock(block, use_rope=index >= backbone.rope_start)
                              for index, block in enumerate(backbone.blocks[:9])]
        # Block 8 is the local state paired with the first global block 9.
        self.frame_blocks = [IdentityBlock()] + [LocalBlock(backbone.blocks[index])
                                                 for index in range(10, 24, 2)]
        self.global_blocks = [GlobalBlock(backbone.blocks[index]) for index in range(9, 24, 2)]
        self.cached_pairs = (1, 3, 5, 7)
        self.head_layers = tuple(layer for pair in self.cached_pairs for layer in (2 * pair, 2 * pair + 1))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer = {"mean": mean, "std": std}

    def embed(self, images):
        groups = defaultdict(list)
        for index, image in enumerate(images):
            groups[tuple(image.shape)].append(index)
        output = [None] * len(images)
        mean = self.register_buffer["mean"].to(images[0])
        std = self.register_buffer["std"].to(images[0])
        for indices in groups.values():
            pixels = (torch.stack([images[index] for index in indices]) - mean) / std
            tokens = self.backbone.prepare_tokens_with_masks(pixels[None])[0]
            for index, hidden in zip(indices, tokens.split(1)):
                output[index] = hidden
        return output

    def preprocess(self, state, grids, views, observer=None):
        for index, block in enumerate(self.prefix_blocks):
            if index == self.backbone.alt_start:
                camera = torch.cat([
                    self.backbone.camera_token[:, :1],
                    self.backbone.camera_token[:, 1:].expand(1, len(state) - 1, -1),
                ], dim=1)[0]
                state = [torch.cat([camera[view_index][None, None], hidden[:, 1:]], dim=1)
                         for view_index, hidden in enumerate(state)]
            view_positions = None
            if index >= self.backbone.rope_start:
                view_positions = [positions(grid, grid, self.special_count, "native", hidden.device)
                                  for hidden, grid in zip(state, grids)]
            state = frame_forward(block, state, grids, self.special_count,
                                  observer=observer, layer=index, view_positions=view_positions)
        return state

    def _results_from_output(self, output):
        results = []
        c2w = _homogeneous_inverse(output.extrinsics[0])
        for index in range(c2w.shape[0]):
            depth = output.depth[0, index]
            intrinsics = output.intrinsics[0, index]
            local, world = _depth_points(depth, intrinsics, c2w[index])
            results.append({"camera_poses": c2w[index], "points": world,
                            "local_points": local, "conf": output.depth_conf[0, index][..., None],
                            "depth": depth, "depth_conf": output.depth_conf[0, index][..., None],
                            "intrinsics": intrinsics})
        return results

    def reference(self, images):
        pixels = torch.stack(images)
        mean = self.register_buffer["mean"].to(pixels)
        std = self.register_buffer["std"].to(pixels)
        native = self.model(((pixels - mean) / std)[None], ref_view_strategy="first")
        return self._results_from_output(native)

    def run_heads(self, cache, grids, head_positions=None):
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
        from depth_anything_3.utils.geometry import affine_inverse

        last_pair = self.cached_pairs[-1]
        camera_tokens = torch.cat([
            torch.cat([cache[2 * last_pair][index][:, :1],
                       cache[2 * last_pair + 1][index][:, :1]], dim=-1)
            for index in range(len(grids))
        ], dim=1)
        with torch.autocast(device_type=camera_tokens.device.type, enabled=False):
            pose_encoding = self.model.cam_dec(camera_tokens.float())
        results = [None] * len(grids)
        groups = defaultdict(list)
        for index, grid in enumerate(grids):
            groups[grid].append(index)
        for grid, indices in groups.items():
            feats = []
            for pair in self.cached_pairs:
                local = torch.cat([cache[2 * pair][index][:, 1:] for index in indices], dim=0)[None]
                global_state = torch.cat([cache[2 * pair + 1][index][:, 1:] for index in indices], dim=0)[None]
                patches = torch.cat([local, self.backbone.norm(global_state)], dim=-1)
                cameras = torch.cat([
                    torch.cat([cache[2 * pair][index][:, :1],
                               cache[2 * pair + 1][index][:, :1]], dim=-1)
                    for index in indices
                ], dim=1)
                feats.append((patches, cameras))
            height, width = (side * self.patch_size for side in grid)
            with torch.autocast(device_type=patches.device.type, enabled=False):
                output = self.model.head([(patches.float(), cameras.float())
                                          for patches, cameras in feats],
                                         height, width, patch_start_idx=0)
                c2w, intrinsics = pose_encoding_to_extri_intri(pose_encoding[:, indices], (height, width))
                output.extrinsics = affine_inverse(c2w)
                output.intrinsics = intrinsics
                output = self.model._process_mono_sky_estimation(output)
                group_results = self._results_from_output(output)
            for index, result in zip(indices, group_results):
                results[index] = result
        return results


def load_da3(checkpoint, device):
    import json
    from pathlib import Path

    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object

    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint / "config.json").read_text())
    model = create_object(OmegaConf.create(saved["config"]))
    weights = load_file(str(checkpoint / "model.safetensors"))
    prefix = "model."
    if not all(key.startswith(prefix) for key in weights):
        raise RuntimeError("Unexpected DA3 checkpoint key prefix")
    missing, unexpected = model.load_state_dict(
        {key[len(prefix):]: value for key, value in weights.items()}, strict=False)
    allowed_missing = {
        f"head.scratch.output_conv2_aux.{level}.2.{field}"
        for level in (1, 2, 3) for field in ("weight", "bias")
    }
    # The auxiliary levels share these LayerNorm parameters with level 0.
    # Only accept omitted aliases when their loaded counterpart really is shared.
    parameters = dict(model.named_parameters(remove_duplicate=False))
    shared_missing = all(
        key in allowed_missing
        and parameters[key] is parameters[f"head.scratch.output_conv2_aux.0.2.{key.rsplit('.', 1)[1]}"]
        for key in missing
    )
    if not shared_missing or unexpected:
        raise RuntimeError(f"Unexpected DA3 state mismatch: missing={missing}, unexpected={unexpected}")
    return DA3Adapter(model.to(device).eval())
