import numpy as np
from scipy.spatial import cKDTree


def point_metrics(predicted, ground_truth, threshold):
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if (predicted.ndim != 2 or ground_truth.ndim != 2 or predicted.shape[1] != 3
            or ground_truth.shape[1] != 3 or not np.isfinite(predicted).all()
            or not np.isfinite(ground_truth).all() or not len(predicted) or not len(ground_truth)):
        raise ValueError("Expected nonempty finite Nx3 point clouds")
    pred_to_gt = cKDTree(ground_truth).query(predicted, workers=-1)[0]
    gt_to_pred = cKDTree(predicted).query(ground_truth, workers=-1)[0]
    precision = float(np.mean(pred_to_gt < threshold))
    recall = float(np.mean(gt_to_pred < threshold))
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"accuracy": float(pred_to_gt.mean()), "completeness": float(gt_to_pred.mean()),
            "precision": precision, "recall": recall, "fscore": fscore,
            "threshold": threshold}


def chamfer_mm(predicted, ground_truth):
    metrics = point_metrics(predicted, ground_truth, threshold=np.inf)
    return {"accuracy_mm": metrics["accuracy"] * 1000,
            "completeness_mm": metrics["completeness"] * 1000,
            "chamfer_mm": (metrics["accuracy"] + metrics["completeness"]) * 500}
