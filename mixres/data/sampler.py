import numpy as np

from mixres.rmfa.grid import budget_sizes


def farthest_points(centres, count):
    centres = np.asarray(centres, dtype=np.float64)
    if not 0 < count <= len(centres) or not np.isfinite(centres).all():
        raise ValueError("Invalid FPS centres or count")
    chosen = [0]
    distances = np.full(len(centres), np.inf)
    while len(chosen) < count:
        distances = np.minimum(distances, ((centres - centres[chosen[-1]]) ** 2).sum(-1))
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return chosen


def sample_scene(frame_ids, camera_to_world, *, seed, scene_index, n_views, n_sharp,
                 rho, budget, aspect=4 / 3, excluded=()):
    available = [i for i, frame in enumerate(frame_ids) if frame not in set(excluded)]
    if len(available) < n_views:
        raise ValueError("Too few eligible frames; never silently lower the view count")
    indices = np.random.default_rng(seed + scene_index).choice(available, n_views, replace=False)
    sharp_indices = set(farthest_points(np.asarray(camera_to_world)[indices, :3, 3], n_sharp))
    hi, lo = budget_sizes(budget, n_sharp, n_views - n_sharp, rho, aspect)
    views = []
    for i, index in enumerate(indices):
        sharp = i in sharp_indices
        views.append({"frame": str(frame_ids[index]), "sharp": sharp,
                      "hw": list(hi if sharp else lo), "hi_hw": list(hi),
                      "rho": 1.0 if sharp else rho, "crop": [0.0, 0.0, 1.0, 1.0]})
    return views
