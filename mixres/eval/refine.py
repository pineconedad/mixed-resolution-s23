"""Coarse tokens, fine geometry: refine feed-forward poses with two-view geometry.

A model gives coarse camera-to-world poses for all views. For every pair of
views, classical matching on the full-resolution images gives a precise relative
rotation and translation direction. Pairs that pass a quality check replace the
model's relative pose; all pairs are then fused back into one set of global
poses. With no accepted pair the output equals the coarse poses.

Conventions: poses are camera-to-world. The relative transform i->j is
inv(C_j) @ C_i, i.e. x_j = R_ij x_i + t_ij, matching mixres.eval.pose.
"""
import numpy as np

from mixres.eval.ensemble import project_rotation

# Fixed before any real image was processed. Do not tune on results.
SIFT_FEATURES = 4000
RATIO = 0.8
RANSAC_PIXELS = 1.0
MIN_INLIERS = 30
MAX_ROTATION_DISAGREEMENT_DEG = 15.0
REFINED_WEIGHT, COARSE_WEIGHT = 3.0, 1.0


def detect(gray):
    import cv2
    keypoints, descriptors = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(gray, None)
    points = np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2)
    return points, descriptors


def match(features_i, features_j):
    """Ratio-test matches that are also mutual nearest neighbours."""
    import cv2
    (points_i, desc_i), (points_j, desc_j) = features_i, features_j
    if desc_i is None or desc_j is None or len(desc_i) < 2 or len(desc_j) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(desc_i, desc_j, k=2)
               if m.distance < RATIO * n.distance}
    backward = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(desc_j, desc_i, k=2)
                if m.distance < RATIO * n.distance}
    pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    index = np.array(pairs)
    return points_i[index[:, 0]], points_j[index[:, 1]]


def pair_from_matches(points_i, points_j, k_i, k_j):
    """Relative pose i->j from pixel matches. Returns (R, unit t, inliers) or None."""
    import cv2
    if len(points_i) < 8:
        return None
    norm_i = cv2.undistortPoints(points_i.reshape(-1, 1, 2), k_i, None).reshape(-1, 2)
    norm_j = cv2.undistortPoints(points_j.reshape(-1, 1, 2), k_j, None).reshape(-1, 2)
    focal = 0.5 * (k_i[0, 0] + k_j[0, 0])
    essential, mask = cv2.findEssentialMat(norm_i, norm_j, np.eye(3), method=cv2.RANSAC, prob=0.9999,
                                           threshold=RANSAC_PIXELS / focal)
    if essential is None or essential.shape != (3, 3):
        return None
    inliers, rotation, translation, _ = cv2.recoverPose(essential, norm_i, norm_j, np.eye(3), mask=mask)
    norm = np.linalg.norm(translation)
    if inliers < 8 or norm < 1e-9 or not np.isfinite(rotation).all():
        return None
    return rotation, translation.ravel() / norm, int(inliers)


def rotation_angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def refine(coarse, pairs):
    """Fuse accepted pair estimates with the coarse poses.

    coarse: [views, 4, 4] camera-to-world. pairs: {(i, j): (R_ij, t_ij unit, inliers)} with i < j.
    Returns (refined [views, 4, 4], number of accepted pairs).
    """
    coarse = np.asarray(coarse, dtype=np.float64)
    count = len(coarse)
    rotations, centres = coarse[:, :3, :3].copy(), coarse[:, :3, 3].copy()
    edges, accepted = [], 0
    for i in range(count):
        for j in range(i + 1, count):
            relative = rotations[j].T @ rotations[i]
            offset = rotations[j].T @ (centres[i] - centres[j])
            baseline = np.linalg.norm(offset)
            direction = offset / baseline if baseline > 1e-12 else np.zeros(3)
            weight = COARSE_WEIGHT
            estimate = pairs.get((i, j))
            if (estimate is not None and estimate[2] >= MIN_INLIERS
                    and rotation_angle(estimate[0], relative) <= MAX_ROTATION_DISAGREEMENT_DEG):
                # Two-view translation has no scale and an arbitrary sign: keep the coarse length and side.
                sign = 1.0 if estimate[1] @ direction >= 0 else -1.0
                relative, direction, weight = estimate[0], sign * estimate[1], REFINED_WEIGHT
                accepted += 1
            edges.append((i, j, relative, direction, baseline, weight, weight == REFINED_WEIGHT))
    if not accepted:
        return coarse.copy(), 0
    for _ in range(100):
        for view in range(count):
            total = np.zeros((3, 3))
            for i, j, relative, _, _, weight, _ in edges:
                if i == view:
                    total += weight * rotations[j] @ relative
                elif j == view:
                    total += weight * rotations[i] @ relative.T
            rotations[view] = project_rotation(total)
    # Centres. A coarse edge fixes the full offset c_i - c_j = baseline * R_j d_ij. A refined edge
    # trusts only its direction (two-view geometry has no scale): the offset must be parallel to
    # R_j d_ij, with the coarse length kept as a weak term so the edge cannot flip or collapse.
    rows, targets = [], []

    def add(i, j, matrix, target, weight):
        block = np.zeros((3, 3 * count))
        block[:, 3 * i:3 * i + 3], block[:, 3 * j:3 * j + 3] = weight * matrix, -weight * matrix
        rows.extend(block)
        targets.extend(weight * target)

    for i, j, _, direction, baseline, weight, refined_edge in edges:
        world = rotations[j] @ direction
        if refined_edge:
            cross = np.array([[0, -world[2], world[1]], [world[2], 0, -world[0]], [-world[1], world[0], 0]])
            add(i, j, cross, np.zeros(3), weight)
            add(i, j, np.eye(3), baseline * world, 0.1 * weight)
        else:
            add(i, j, np.eye(3), baseline * world, weight)
    anchor = 1e-3
    rows.extend(anchor * np.eye(3 * count))
    targets.extend(anchor * coarse[:, :3, 3].ravel())
    solution = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)[0]
    refined = np.tile(np.eye(4), (count, 1, 1))
    refined[:, :3, :3], refined[:, :3, 3] = rotations, solution.reshape(count, 3)
    return refined, accepted
