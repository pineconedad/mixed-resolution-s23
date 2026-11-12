"""Standard relative-pose AUC for the independent quick test, not legacy T9.

All inputs are camera-to-world. Relative transform i->j is inv(C_j) @ C_i.
Unlike DA3's histogram AUC, this integrates the exact empirical error CDF.
The protocol identifier is stored in every result to prevent mixing them.
"""
import numpy as np

PROTOCOL = "relative_c2w_exact_auc_v1"
DA3_PROTOCOL = "da3_pairwise_histogram_c2w_v1"


def validate_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Expected at least two finite 4x4 camera poses")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous camera pose")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Camera rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("Camera rotation is not in SO(3)")
    return poses


def relative_errors(predicted, ground_truth):
    predicted, ground_truth = validate_poses(predicted), validate_poses(ground_truth)
    if predicted.shape != ground_truth.shape:
        raise ValueError("Pose count mismatch")
    rotation, translation = [], []
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            pred = np.linalg.inv(predicted[j]) @ predicted[i]
            gt = np.linalg.inv(ground_truth[j]) @ ground_truth[i]
            if np.linalg.norm(gt[:3, 3]) < 1e-9:
                continue  # Undefined GT translation direction, identically excluded in every arm.
            cosine = (np.trace(pred[:3, :3] @ gt[:3, :3].T) - 1) / 2
            rotation.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
            p, g = pred[:3, 3], gt[:3, 3]
            norm = np.linalg.norm(p) * np.linalg.norm(g)
            translation.append(90.0 if norm < 1e-12 else float(np.degrees(np.arccos(np.clip(abs(p @ g / norm), 0, 1)))))
    if not rotation:
        raise ValueError("No nondegenerate camera pairs")
    return np.array(rotation), np.array(translation)


def error_auc(errors, threshold):
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    if len(errors) == 0 or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Expected finite, nonnegative errors")
    inside = errors[errors < threshold]
    x = np.r_[0, inside, threshold]
    recall = np.r_[0, np.arange(1, len(inside) + 1) / len(errors), len(inside) / len(errors)]
    return float(np.trapz(recall, x) / threshold * 100)


def score_poses(predicted, ground_truth):
    r, t = relative_errors(predicted, ground_truth)
    return {**{f"auc{threshold}": error_auc(np.maximum(r, t), threshold) for threshold in (3, 5, 30)},
            "rotation_errors_deg": r.tolist(), "translation_errors_deg": t.tolist(),
             "pair_count": len(r)}


def score_poses_da3(predicted, ground_truth):
    """Match DA3-BENCH's pairwise integer-bin AUC, accepting c2w inputs."""
    predicted, ground_truth = validate_poses(predicted), validate_poses(ground_truth)
    if predicted.shape != ground_truth.shape:
        raise ValueError("Pose count mismatch")
    pred_w2c = np.linalg.inv(predicted)
    gt_w2c = np.linalg.inv(ground_truth)
    pred_w2c = pred_w2c @ np.linalg.inv(pred_w2c[0])
    gt_w2c = gt_w2c @ np.linalg.inv(gt_w2c[0])
    rotation, translation = [], []
    for i in range(len(pred_w2c)):
        for j in range(i + 1, len(pred_w2c)):
            pred = np.linalg.inv(pred_w2c[i]) @ pred_w2c[j]
            gt = np.linalg.inv(gt_w2c[i]) @ gt_w2c[j]
            cosine = (np.trace(pred[:3, :3] @ gt[:3, :3].T) - 1) / 2
            rotation.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
            p, g = pred[:3, 3], gt[:3, 3]
            denominator = np.linalg.norm(p) * np.linalg.norm(g)
            translation.append(90.0 if denominator < 1e-15 else
                               float(np.degrees(np.arccos(np.clip(abs(p @ g / denominator), 0, 1)))))
    maximum = np.maximum(rotation, translation)
    metrics = {}
    for threshold in (3, 5, 15, 30):
        histogram, _ = np.histogram(maximum, bins=np.arange(threshold + 1))
        metrics[f"auc{threshold}"] = float(np.mean(np.cumsum(histogram / len(maximum))) * 100)
    return {**metrics, "rotation_errors_deg": rotation, "translation_errors_deg": translation,
            "pair_count": len(maximum)}
