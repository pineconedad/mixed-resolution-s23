"""VGGT's alternating aggregator, joint camera head and grouped dense heads."""
from collections import defaultdict

import torch
from torch import nn

from mixres.rmfa.positions import ContinuousRoPE


class PositionBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @property
    def norm1(self):
        return self.block.norm1

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x, xpos):
        return self.block(x, pos=xpos)


class VGGTAdapter:
    patch_size = 14

    def __init__(self, model):
        self.model = model.eval()
        aggregator = model.aggregator
        if aggregator.patch_size != self.patch_size or any(head is None for head in
                                                         (model.camera_head, model.point_head, model.depth_head)):
            raise ValueError("Study requires patch-14 VGGT with camera, point and depth heads")
        if aggregator.aa_block_size != 1 or aggregator.aa_order != ["frame", "global"]:
            raise ValueError("Only the pinned alternating VGGT layout is supported")
        self.special_count = aggregator.patch_start_idx
        for block in list(aggregator.frame_blocks) + list(aggregator.global_blocks):
            if block.attn.rope is not None and not isinstance(block.attn.rope, ContinuousRoPE):
                block.attn.rope = ContinuousRoPE(block.attn.rope, base=block.attn.rope.base_frequency)
        self.frame_blocks = [PositionBlock(block) for block in aggregator.frame_blocks]
        self.global_blocks = [PositionBlock(block) for block in aggregator.global_blocks]
        self.cached_pairs = sorted(aggregator.cached_layer_indices)
        required = set(model.point_head.intermediate_layer_idx) | set(model.depth_head.intermediate_layer_idx)
        if not required <= set(self.cached_pairs):
            raise ValueError("Aggregator does not cache the dense heads' required layers")
        self.head_layers = tuple(layer for pair in self.cached_pairs for layer in (2 * pair, 2 * pair + 1))

    def embed(self, images):
        from vggt.models.aggregator import slice_expand_and_flatten

        aggregator = self.model.aggregator
        count = len(images)
        camera = slice_expand_and_flatten(aggregator.camera_token, 1, count)
        registers = slice_expand_and_flatten(aggregator.register_token, 1, count)
        groups = defaultdict(list)
        for index, image in enumerate(images):
            groups[tuple(image.shape)].append(index)
        tokens = [None] * count
        for indices in groups.values():
            pixels = torch.stack([images[i] for i in indices])
            pixels = (pixels - aggregator._resnet_mean[0]) / aggregator._resnet_std[0]
            patches = aggregator.patch_embed(pixels)
            if isinstance(patches, dict):
                patches = patches["x_norm_patchtokens"]
            # Select first-view tokens by the original index, not by grid group.
            hidden = torch.cat([camera[indices], registers[indices], patches], dim=1)
            for index, view in zip(indices, hidden.split(1)):
                tokens[index] = view
        return tokens

    def reference(self, images):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        native = self.model(torch.stack(images)[None])
        extrinsics, _ = pose_encoding_to_extri_intri(native["pose_enc"], build_intrinsics=False)
        w2c = torch.eye(4, device=extrinsics.device).repeat(len(images), 1, 1)
        w2c[:, :3] = extrinsics[0]
        c2w = torch.linalg.inv(w2c)
        return [{"camera_poses": c2w[i], "pose_encoding": native["pose_enc"][0, i],
                 "points": native["world_points"][0, i], "conf": native["world_points_conf"][0, i, ..., None],
                 "depth": native["depth"][0, i], "depth_conf": native["depth_conf"][0, i, ..., None]}
                for i in range(len(images))]

    def run_heads(self, cache, grids, head_positions=None):
        # VGGT's camera head has no RoPE and its DPT head already uses normalized
        # image UV coordinates. Only its aggregator receives pi_hr coordinates.
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = self.model
        last = len(self.frame_blocks) - 1
        camera_tokens = torch.stack([
            torch.cat([cache[2 * last][i][:, :1], cache[2 * last + 1][i][:, :1]], dim=-1)[0]
            for i in range(len(grids))], dim=0)[None]
        # CameraHead reads only the last layer's camera tokens and jointly attends
        # across views. Grouping it by image size would change the model.
        with torch.autocast(device_type="cuda", enabled=False):
            pose_encoding = model.camera_head([camera_tokens])[-1]
        results = [None] * len(grids)
        groups = defaultdict(list)
        for index, grid in enumerate(grids):
            groups[grid].append(index)
        for grid, indices in groups.items():
            aggregated = [None] * len(self.frame_blocks)
            for pair in self.cached_pairs:
                aggregated[pair] = torch.stack([
                    torch.cat([cache[2 * pair][i], cache[2 * pair + 1][i]], dim=-1)[0]
                    for i in indices], dim=0)[None]
            h, w = (side * self.patch_size for side in grid)
            # At the pinned commit DPTHead uses images only for dimensions/chunk
            # metadata, not pixel values. Its head grid may differ from entry size.
            images_metadata = torch.empty(1, len(indices), 3, h, w, device=camera_tokens.device)
            with torch.autocast(device_type="cuda", enabled=False):
                world, confidence = model.point_head(aggregated, images_metadata, self.special_count)
                depth, depth_conf = model.depth_head(aggregated, images_metadata, self.special_count)
                encoded = pose_encoding[:, indices]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(encoded, (h, w))
                w2c = torch.eye(4, device=extrinsics.device).repeat(len(indices), 1, 1)
                w2c[:, :3] = extrinsics[0]
                c2w = torch.linalg.inv(w2c)
                homogeneous = torch.cat([world[0], torch.ones_like(world[0][..., :1])], dim=-1)
                local = torch.einsum("bij,bhwj->bhwi", w2c, homogeneous)[..., :3]
            for j, index in enumerate(indices):
                results[index] = {"camera_poses": c2w[j], "pose_encoding": encoded[0, j],
                                  "points": world[0, j], "local_points": local[j],
                                  "conf": confidence[0, j, ..., None], "depth": depth[0, j],
                                  "depth_conf": depth_conf[0, j, ..., None], "intrinsics": intrinsics[0, j]}
        return results


def load_vggt(checkpoint, device):
    from pathlib import Path
    from vggt.models.vggt import VGGT

    model = VGGT()
    if Path(checkpoint).suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(checkpoint))
    else:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return VGGTAdapter(model.to(device).eval())
