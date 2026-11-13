import numpy as np
import pytest

from mixres.eval.pose import relative_errors
from mixres.eval.refine import detect, match, pair_from_matches, refine
from test_ensemble import random_rotation

K = np.array([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]])


def looking_scene(rng, views=6):
    """Cameras on a ring looking at a point cloud near the origin."""
    points = rng.uniform(-1, 1, size=(400, 3))
    poses = np.tile(np.eye(4), (views, 1, 1))
    for index, pose in enumerate(poses):
        angle = 0.5 * index + rng.uniform(-0.1, 0.1)
        centre = np.array([4 * np.sin(angle), rng.uniform(-0.3, 0.3), -4 * np.cos(angle)])
        forward = -centre / np.linalg.norm(centre)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        pose[:3, :3], pose[:3, 3] = np.stack([right, np.cross(forward, right), forward], 1), centre
    return points, poses


def project(points, pose):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    return camera[:, :2] / camera[:, 2:] * K[0, 0] + K[:2, 2]


def perturb(rng, poses, degrees, shift):
    out = poses.copy()
    for pose in out:
        pose[:3, :3] = pose[:3, :3] @ random_rotation(rng, degrees)
        pose[:3, 3] += rng.normal(size=3) * shift
    return out


def exact_pairs(points, poses, noise=0.0, rng=None):
    pairs = {}
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            a, b = project(points, poses[i]), project(points, poses[j])
            if noise:
                a, b = a + rng.normal(size=a.shape) * noise, b + rng.normal(size=b.shape) * noise
            pairs[(i, j)] = pair_from_matches(a, b, K, K)
    return pairs


def test_two_view_estimate_follows_the_metric_convention():
    rng = np.random.default_rng(0)
    points, poses = looking_scene(rng)
    rotation, translation, inliers = exact_pairs(points, poses)[(0, 2)]
    truth = np.linalg.inv(poses[2]) @ poses[0]
    assert inliers > 300
    assert np.allclose(rotation, truth[:3, :3], atol=1e-5)
    direction = truth[:3, 3] / np.linalg.norm(truth[:3, 3])
    assert abs(translation @ direction) > 1 - 1e-8


def test_refinement_fixes_coarse_poses_and_is_identity_without_pairs():
    rng = np.random.default_rng(1)
    points, poses = looking_scene(rng)
    coarse = perturb(rng, poses, degrees=5.0, shift=0.15)
    untouched, accepted = refine(coarse, {})
    assert accepted == 0 and np.array_equal(untouched, coarse)
    refined, accepted = refine(coarse, exact_pairs(points, poses, noise=0.3, rng=rng))
    assert accepted == 15
    before = np.maximum(*relative_errors(coarse, poses)).mean()
    after = np.maximum(*relative_errors(refined, poses)).mean()
    assert after < 0.25 * before and after < 1.0


def test_wrong_or_weak_pairs_are_rejected():
    rng = np.random.default_rng(2)
    points, poses = looking_scene(rng)
    coarse = perturb(rng, poses, degrees=2.0, shift=0.05)
    pairs = exact_pairs(points, poses)
    bad = {key: (random_rotation(rng) @ value[0], value[1], value[2]) for key, value in pairs.items()}
    weak = {key: (value[0], value[1], 10) for key, value in pairs.items()}
    for rejected in (bad, weak):
        refined, accepted = refine(coarse, rejected)
        assert accepted <= 1 and np.maximum(*relative_errors(refined, poses)).mean() < 1.5 * \
            np.maximum(*relative_errors(coarse, poses)).mean()


def test_sift_matching_runs_on_real_pixels():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    image = cv2.GaussianBlur((rng.uniform(size=(480, 640)) * 255).astype(np.uint8), (0, 0), 2)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    shifted = np.roll(image, (7, 12), axis=(0, 1))
    a, b = match(detect(image), detect(shifted))
    assert len(a) > 50 and np.median(b - a, axis=0) == pytest.approx([12, 7], abs=0.5)
