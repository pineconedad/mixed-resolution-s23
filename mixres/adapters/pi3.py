from collections import defaultdict

import torch

from mixres.rmfa.positions import ContinuousRoPE, positions


class Pi3Adapter:
    patch_size = 14

    def __init__(self, model):
        self.model = model.eval()
        self.special_count = model.patch_start_idx
        self.frame_blocks = list(model.decoder[::2])
        self.global_blocks = list(model.decoder[1::2])
        self.head_layers = (len(model.decoder) - 2, len(model.decoder) - 1)
        for module in list(model.modules()):
            if hasattr(module, "rope") and module.rope is not None and not isinstance(module.rope, ContinuousRoPE):
                module.rope = ContinuousRoPE(module.rope)

    def embed(self, images):
        grouped = defaultdict(list)
        for i, image in enumerate(images):
            grouped[tuple(image.shape)].append(i)
        output = [None] * len(images)
        for indices in grouped.values():
            imgs = torch.stack([images[i] for i in indices])
            hidden = self.model.encoder((imgs - self.model.image_mean) / self.model.image_std, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            registers = self.model.register_token[0].expand(len(indices), -1, -1)
            hidden = torch.cat([registers, hidden], 1)
            for i, x in zip(indices, hidden.split(1)):
                output[i] = x
        return output

    def reference(self, images):
        native = self.model(torch.stack(images)[None])
        return [{key: native[key][0, i] for key in ("camera_poses", "points", "local_points", "conf")}
                for i in range(len(images))]

    def run_heads(self, cache, grids, head_positions=None):
        model = self.model
        result = [None] * len(grids)
        groups = defaultdict(list)
        for i, grid in enumerate(grids):
            groups[grid].append(i)
        for grid, indices in groups.items():
            hidden = torch.cat([torch.cat([cache[layer][i] for layer in self.head_layers], -1)
                                for i in indices], 0)
            pos = (positions(grid, grid, self.special_count, "native", hidden.device).expand(len(indices), -1, -1)
                   if head_positions is None else torch.cat([head_positions[i] for i in indices], dim=0))
            point_hidden = model.point_decoder(hidden, xpos=pos)
            conf_hidden = model.conf_decoder(hidden, xpos=pos)
            camera_hidden = model.camera_decoder(hidden, xpos=pos)
            h, w = (v * self.patch_size for v in grid)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                ret = model.point_head([point_hidden.float()[:, self.special_count:]], (h, w))
                xy, z = ret.split([2, 1], -1)
                z = z.exp()
                local = torch.cat([xy * z, z], -1)
                conf = model.conf_head([conf_hidden.float()[:, self.special_count:]], (h, w))
                poses = model.camera_head(camera_hidden.float()[:, self.special_count:], *grid)
                homogeneous = torch.cat([local, torch.ones_like(local[..., :1])], -1)
                points = torch.einsum("bij,bhwj->bhwi", poses, homogeneous)[..., :3]
            for j, index in enumerate(indices):
                result[index] = {"camera_poses": poses[j], "points": points[j],
                                 "local_points": local[j], "conf": conf[j]}
        return result


def load_pi3(checkpoint, device):
    from pi3.models.pi3 import Pi3
    from safetensors.torch import load_file

    model = Pi3()
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return Pi3Adapter(model.to(device).eval())
