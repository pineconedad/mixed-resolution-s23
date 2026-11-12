import torch
import torch.nn.functional as F

from mixres.rmfa.arms import ARMS, RMFA_ARMS
from mixres.rmfa.calibration import (CALIBRATIONS, LOGIT_BIASES, attention_bias,
                                     calibrate_groups, key_bias)
from mixres.rmfa.grid import patch_grid, resize_tokens
from mixres.rmfa.positions import positions


def frame_forward(block, tokens, grids, special, *, observer=None, layer=0, view_positions=None,
                  bias_hi=None):
    output = [None] * len(tokens)
    groups = {}
    for i, grid in enumerate(grids):
        groups.setdefault(grid, []).append(i)
    for grid, indices in groups.items():
        x = torch.cat([tokens[i] for i in indices], 0)
        pos = (positions(grid, grid, special, "native", x.device).expand(len(indices), -1, -1)
               if view_positions is None else torch.cat([view_positions[i] for i in indices], dim=0))
        if observer:
            observer.frame(layer, block, x, pos, indices, grid)
        bias = None
        if bias_hi is not None:
            if len({bias_hi[i] for i in indices}) != 1:
                raise ValueError("Views sharing a frame grid must share a sharp grid for logit bias")
            bias = key_bias(grid, bias_hi[indices[0]], special, x.device)
        with attention_bias(block, bias):
            y = block(x, xpos=pos)
        for i, view_output in zip(indices, y.split(1)):
            output[i] = view_output
    return output


def run_engine(adapter, images, views, arm_name, *, pool_op="area", lift_op="bilinear",
               token_interp="bilinear", head_grid="lo", rmfa_layers="all", observer=None,
               alignment_scope="global", position_scale="actual", calibrate="none",
               logit_bias="none"):
    if arm_name not in ARMS or head_grid not in {"lo", "hi"}:
        raise ValueError("Unsupported arm/head grid")
    if calibrate not in CALIBRATIONS or logit_bias not in LOGIT_BIASES:
        raise ValueError("Unknown calibration/logit-bias option")
    arm = ARMS[arm_name]
    if alignment_scope not in {"global", "all"} or position_scale not in {"actual", "nominal"}:
        raise ValueError("Unknown position-alignment protocol")
    hi = [patch_grid(v["hi_hw"], adapter.patch_size) for v in views]
    lo = [patch_grid(v["hw"], adapter.patch_size) for v in views]
    embedded = adapter.embed(images)
    source = [patch_grid(image.shape[-2:], adapter.patch_size) for image in images]
    special = adapter.special_count
    state_grids = hi if arm.frame == "hi" else lo
    state = [resize_tokens(x, s, t, special, token_interp if arm.entry == "token" else pool_op)
             for x, s, t in zip(embedded, source, state_grids)]
    if hasattr(adapter, "preprocess"):
        state = adapter.preprocess(state, state_grids, views, observer=observer)
    layers = len(adapter.frame_blocks)
    active_count = {"all": layers, "first_half": layers // 2,
                    "first_quarter": layers // 4}.get(rmfa_layers)
    if active_count is None:
        raise ValueError(rmfa_layers)
    cache = {}
    trace = []
    for pair, (frame, global_block) in enumerate(zip(adapter.frame_blocks, adapter.global_blocks)):
        use_rmfa = pair < active_count or arm_name not in RMFA_ARMS
        target_state = state_grids if use_rmfa else lo
        if not use_rmfa:
            state = [resize_tokens(x, g, t, special, pool_op) for x, g, t in zip(state, state_grids, lo)]
            state_grids = lo
        frame_kind = arm.positions if alignment_scope == "all" else "native"
        frame_pos = [positions(g, h, special, frame_kind, x.device,
                               v["rho"] if position_scale == "nominal" else None)
                     for x, g, h, v in zip(state, target_state, hi, views)]
        state = frame_forward(frame, state, target_state, special, observer=observer,
                              layer=2 * pair, view_positions=frame_pos,
                              bias_hi=hi if logit_bias in {"frame", "both"} else None)
        selected_head_grids = (hi if head_grid == "hi" else lo) if arm_name in RMFA_ARMS | {"A6"} else target_state
        if 2 * pair in adapter.head_layers:
            cache[2 * pair] = [resize_tokens(x, g, t, special, pool_op if t[0] <= g[0] else lift_op)
                              for x, g, t in zip(state, target_state, selected_head_grids)]
        global_grids = hi if arm.global_grid == "hi" else lo
        before = [resize_tokens(x, s, t, special, pool_op if t[0] <= s[0] else lift_op)
                  for x, s, t in zip(state, target_state, global_grids)]
        before, shift = calibrate_groups(before, global_grids, hi, special, calibrate)
        pos = [positions(g, h, special, arm.positions, x.device,
                         v["rho"] if position_scale == "nominal" else None)
               for x, g, h, v in zip(before, global_grids, hi, views)]
        sequence = torch.cat(before, 1)
        lengths = [x.shape[1] for x in before]
        expected_patches = sum(g[0] * g[1] for g in global_grids)
        if sequence.shape[1] != expected_patches + len(views) * special:
            raise AssertionError("Global token count mismatch")
        if observer:
            observer.global_input(2 * pair + 1, before, global_grids)
        global_bias = None
        if logit_bias in {"global", "both"}:
            parts = [key_bias(g, h, special, sequence.device) for g, h in zip(global_grids, hi)]
            if any(part is not None for part in parts):
                global_bias = torch.cat([torch.zeros(n, device=sequence.device) if part is None else part
                                         for part, n in zip(parts, lengths)])
        with attention_bias(global_block, global_bias):
            after = list(global_block(sequence, xpos=torch.cat(pos, 1)).split(lengths, dim=1))
        updated = []
        for x, low_in, low_out, s, t in zip(state, before, after, target_state, global_grids):
            if s == t and calibrate == "moment_transient" and low_in is not x:
                # Calibration only shapes what global attention reads; keep the native state.
                updated.append(torch.cat([low_out[:, :special],
                                          x[:, special:] + (low_out - low_in)[:, special:]], dim=1))
            elif s == t:
                updated.append(low_out)  # Avoid x + (out - x) rounding drift in T1.
            else:
                delta = resize_tokens(low_out - low_in, t, s, special,
                                      pool_op if s[0] < t[0] else lift_op)
                patches = x[:, special:] + delta[:, special:]
                updated.append(torch.cat([low_out[:, :special], patches], dim=1))
        state = updated
        if 2 * pair + 1 in adapter.head_layers:
            persistent_heads = head_grid == "hi" and arm_name in RMFA_ARMS
            head_state = state if persistent_heads else after
            head_source = target_state if persistent_heads else global_grids
            cache[2 * pair + 1] = [resize_tokens(x, g, t, special, pool_op if t[0] <= g[0] else lift_op)
                                  for x, g, t in zip(head_state, head_source, selected_head_grids)]
        if observer:
            observer.tokens(2 * pair + 1, after, global_grids, lo, special)
        trace.append({"layer": 2 * pair + 1, "patch_tokens": expected_patches,
                      "frame_block_count": (int(pair not in getattr(adapter, "identity_frame_pairs", ()))
                                            + (getattr(adapter, "prefix_frame_blocks", 0) if pair == 0 else 0)),
                      "special_tokens": len(views) * special, "total_tokens": sequence.shape[1],
                      "group_shift": shift,
                      "frame_total_tokens": sum(x.shape[1] for x in state),
                      "frame_attention_quadratic": sum(x.shape[1] ** 2 for x in state),
                      "global_attention_quadratic": sequence.shape[1] ** 2})
    if alignment_scope == "all":
        head_pos = [positions(g, h, special, arm.positions, x.device,
                              v["rho"] if position_scale == "nominal" else None)
                    for x, g, h, v in zip(state, selected_head_grids, hi, views)]
        predictions = adapter.run_heads(cache, selected_head_grids, head_positions=head_pos)
    else:
        predictions = adapter.run_heads(cache, selected_head_grids)
    if head_grid == "hi" and arm_name in RMFA_ARMS | {"A6"}:
        for prediction, source_grid, target_grid in zip(predictions, selected_head_grids, lo):
            if source_grid != target_grid:
                size = tuple(side * adapter.patch_size for side in target_grid)
                for key in ("points", "local_points", "conf", "depth", "depth_conf"):
                    if key not in prediction:
                        continue
                    field = prediction[key]
                    scalar = field.ndim == 2
                    field = (field[None] if scalar else field.permute(2, 0, 1))[None]
                    resized = F.interpolate(field, size=size, mode="bilinear", align_corners=False)[0]
                    prediction[key] = resized[0] if scalar else resized.permute(1, 2, 0)
                if "intrinsics" in prediction:
                    intrinsics = prediction["intrinsics"].clone()
                    intrinsics[0] *= target_grid[1] / source_grid[1]
                    intrinsics[1] *= target_grid[0] / source_grid[0]
                    prediction["intrinsics"] = intrinsics
    return predictions, trace
