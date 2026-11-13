"""Fuse several pose predictions of the same views (e.g. one per input size).

Each prediction lives in its own world frame and scale, so every run is first
aligned to a common frame by a similarity transform, then rotations are
averaged on SO(3) (chordal mean) and camera centres arithmetically.
All poses are camera-to-world, shape [runs, views, 4, 4].
"""
import numpy as np


def project_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def align(run, reference):
    """Similarity-align `run` to `reference`; returns aligned rotations and centres."""
    rotations, centres = run[:, :3, :3], run[:, :3, 3]
    gauge = project_rotation(np.einsum("nij,nkj->ik", reference[:, :3, :3], rotations))
    rotated = centres @ gauge.T
    source = rotated - rotated.mean(0)
    target = reference[:, :3, 3] - reference[:, :3, 3].mean(0)
    denominator = (source ** 2).sum()
    scale = (source * target).sum() / denominator if denominator > 1e-12 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return gauge @ rotations, scale * source + reference[:, :3, 3].mean(0)


def fuse(runs, iterations=3):
    """Mean pose per view over runs. Returns (fused [views,4,4], per-run rotation spread in degrees)."""
    runs = np.asarray(runs, dtype=np.float64)
    if runs.ndim != 4 or runs.shape[2:] != (4, 4) or not np.isfinite(runs).all():
        raise ValueError("Expected finite poses of shape [runs, views, 4, 4]")
    reference = runs[0]
    for _ in range(iterations):
        aligned = [align(run, reference) for run in runs]
        rotations = np.stack([project_rotation(sum(r[i] for r, _ in aligned)) for i in range(runs.shape[1])])
        centres = np.mean([c for _, c in aligned], axis=0)
        reference = np.tile(np.eye(4), (runs.shape[1], 1, 1))
        reference[:, :3, :3], reference[:, :3, 3] = rotations, centres
    spread = []
    for r, _ in aligned:
        cosine = (np.einsum("nij,nij->n", r, rotations) - 1) / 2
        spread.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()))
    return reference, spread
