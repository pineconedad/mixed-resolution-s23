import io
import json
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mixres.data.remote_zip import StoredMemberReader
from mixres.eval.pose import error_auc, relative_errors, score_poses
from mixres.io import atomic_json, digest, read_json
from mixres.stats.paired import holm, paired_bootstrap, recovery


def trajectory():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, :3, :3] = Rotation.from_euler("xyz", np.arange(15).reshape(5, 3) * 2, degrees=True).as_matrix()
    poses[:, :3, 3] = np.arange(15).reshape(5, 3) ** 1.2
    return poses


def test_perfect_pose_and_global_similarity_invariance():
    gt = trajectory()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [30, 12, -15], degrees=True).as_matrix()
    transform[:3, 3] = [4, 6, -10]
    predicted = gt.copy()
    predicted[:, :3, 3] *= 7
    predicted = transform[None] @ predicted
    metrics = score_poses(predicted, gt)
    assert metrics["auc3"] > 99.999
    assert metrics["pair_count"] == 10


def test_wrong_rotation_and_translation_reduce_auc():
    gt = trajectory()
    pred = gt.copy()
    pred[2, :3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    pred[3, :3, 3] = [-6, 10, 12]
    assert score_poses(pred, gt)["auc3"] < 60


def test_exact_auc_known_values():
    assert error_auc([0, 0], 3) == 100
    assert error_auc([4, 5], 3) == 0
    assert error_auc([1], 3) == pytest.approx(5 / 6 * 100)


def test_invalid_poses_and_all_stationary_fail():
    gt = trajectory()
    with pytest.raises(ValueError):
        score_poses(gt * np.nan, gt)
    stationary = np.repeat(np.eye(4)[None], 3, axis=0)
    with pytest.raises(ValueError, match="nondegenerate"):
        relative_errors(stationary, stationary)


def test_bootstrap_pairs_and_ratio_guard():
    result = paired_bootstrap([1, 1, 1])
    assert result["ci95"] == [1, 1]
    assert recovery([10, 20, 30], [20, 30, 40], [15, 25, 35])["R"] == 0.5
    assert recovery([10, 20, 30], [10, 20, 30], [15, 25, 35])["R"] is None
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.01], family_size=30), [0.3])


def test_atomic_json_failure_preserves_previous_result(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"complete": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"broken": float("nan")})
    assert read_json(path) == {"complete": True}
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_stored_nested_zip_and_crc():
    nested = io.BytesIO()
    with ZipFile(nested, "w", ZIP_DEFLATED) as inner:
        inner.writestr("example.txt", "correct data")
    outer_file = io.BytesIO()
    with ZipFile(outer_file, "w", ZIP_STORED) as outer:
        outer.writestr("seq-01.zip", nested.getvalue())
    with ZipFile(outer_file) as outer:
        with ZipFile(StoredMemberReader(outer_file, outer.getinfo("seq-01.zip"))) as inner:
            assert inner.read("example.txt") == b"correct data"
