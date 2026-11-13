import numpy as np

from mixres.eval.ensemble import fuse, project_rotation
from mixres.eval.pose import relative_errors


def random_rotation(rng, degrees=180.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees) * rng.uniform(-1, 1)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def scene(rng, views=12):
    poses = np.tile(np.eye(4), (views, 1, 1))
    for pose in poses:
        pose[:3, :3], pose[:3, 3] = random_rotation(rng), rng.normal(size=3)
    return poses


def regauge(rng, poses, noise=0.0):
    world, scale, shift = random_rotation(rng), rng.uniform(0.3, 3), rng.normal(size=3)
    out = poses.copy()
    for pose in out:
        pose[:3, :3] = world @ pose[:3, :3] @ random_rotation(rng, noise)
        pose[:3, 3] = scale * world @ (pose[:3, 3] + rng.normal(size=3) * noise / 30) + shift
    return out


def test_identical_runs_in_different_frames_fuse_to_the_same_relative_poses():
    rng = np.random.default_rng(0)
    truth = scene(rng)
    fused, spread = fuse([regauge(rng, truth) for _ in range(3)])
    rotation, translation = relative_errors(fused, truth)
    assert rotation.max() < 1e-5 and translation.max() < 1e-4 and max(spread) < 1e-5


def test_fusing_independent_noisy_runs_beats_each_single_run():
    rng = np.random.default_rng(1)
    truth = scene(rng)
    runs = [regauge(rng, truth, noise=6.0) for _ in range(4)]
    fused, _ = fuse(runs)
    single = [np.maximum(*relative_errors(run, truth)).mean() for run in runs]
    assert np.maximum(*relative_errors(fused, truth)).mean() < min(single)
    assert np.allclose(project_rotation(fused[0, :3, :3]), fused[0, :3, :3], atol=1e-8)
