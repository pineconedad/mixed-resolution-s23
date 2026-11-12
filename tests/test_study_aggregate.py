import copy
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mixres.io import atomic_json, file_hash, read_json
from mixres.stats.paired import holm, paired_bootstrap, paired_sign_flip, stratified_recovery


spec = importlib.util.spec_from_file_location(
    "study_aggregate", Path(__file__).resolve().parents[1] / "scripts/05_aggregate.py")
aggregate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate_module)


def write_cell(root, arm, values, dataset="eth3d", seed=1000, change=None):
    result = {
        "study": "s23", "model": "pi3", "dataset": dataset, "budget": 4096,
        "arm": arm, "seed": seed, "status": "completed", "baseline_protocol": "fresh_v1",
        "metric_protocol": {"name": "pose-v1"}, "auc_units": "percent",
        "scenes_expected": [f"scene{i}" for i in range(len(values))],
        "provenance": {
            "source_hash": "source", "git_commit": "commit", "upstream_commit": "upstream",
            "checkpoint": "checkpoint", "checkpoint_revision": "revision",
            "checkpoint_sha256": "weights", "model_config_hash": "model-config",
            "precision": "bf16", "versions": {"torch": "2.5", "numpy": "2.0"},
            "cuda_version": "12.4", "views_hash": f"{dataset}-{seed}",
            "config_hash": f"config-{arm}", "cell_hash": f"cell-{arm}",
            "frozen_hash": f"frozen-{arm}", "gpu_name": f"gpu-{arm}",
        },
        "per_scene": [{"scene": f"scene{i}", "status": "completed", "auc3": value}
                      for i, value in enumerate(values)],
    }
    if change:
        change(result)
    path = root / "pi3" / dataset / "4096" / arm / f"seed{seed}" / "result.json"
    atomic_json(path, result)
    return path


def csv_rows(output, name):
    with (output / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_stratified_bootstrap_preserves_equal_dataset_weights():
    # Pooled scene weighting gives 9.1/2 instead of the equal-dataset ratio 5.5/2.
    small = np.array([[10, 12, 11]])
    large = np.tile([10, 12, 20], (9, 1))
    stats = stratified_recovery([small, large], resamples=2000)
    assert stats["R"] == pytest.approx(2.75)
    assert stats["gain"]["mean"] == 5.5
    # Constant within-stratum values must give a point interval, even with unequal n.
    assert stats["ci95"] == [2.75, 2.75]


def test_stratified_bootstrap_matches_direct_paired_resampling():
    strata = [np.array([[0, 2, 1], [0, 4, 3]]),
              np.array([[10, 16, 11], [20, 28, 24], [30, 40, 36]])]
    rng = np.random.default_rng(7)
    draws = np.zeros((500, 2))
    for rows in strata:
        differences = rows[:, 1:] - rows[:, :1]
        idx = rng.integers(0, len(rows), size=(500, len(rows)))
        draws += differences[idx].mean(1) / 2
    stats = stratified_recovery(strata, resamples=500, seed=7)
    assert stats["ci95"] == pytest.approx(np.quantile(draws[:, 1] / draws[:, 0], [.025, .975]))
    assert stratified_recovery([[[1, 1, 2]]])["R"] is None
    assert stratified_recovery([[[2, 1, 3]]])["R"] is None


@pytest.mark.parametrize("strata", [[], [[]], [[[1, np.nan, 2]]], [[[1, 2]]]])
def test_stratified_bootstrap_rejects_invalid_input(strata):
    with pytest.raises(ValueError):
        stratified_recovery(strata)


def test_sign_flip_exact_not_bootstrap_zero_tail():
    assert paired_sign_flip([1, 1, 1])["p_two_sided"] == 0.25
    assert paired_sign_flip([1])["p_two_sided"] == 1
    assert paired_sign_flip([0, 0])["p_two_sided"] == 1
    assert paired_sign_flip([1, -1])["p_two_sided"] == 1
    assert paired_sign_flip([1, 1, 1, 0])["p_two_sided"] == 0.25
    assert paired_sign_flip([-1, -1, -1])["p_two_sided"] == 0.25
    assert paired_sign_flip([1, 1, 1])["p_method"] == "sign_flip_exact"
    # Archived exploratory output remains unchanged.
    assert paired_bootstrap([1, 1, 1])["p_two_sided"] == 0


def test_sign_flip_monte_carlo_is_reproducible_and_nonzero():
    result = paired_sign_flip(np.ones(30), resamples=500, seed=9)
    assert result == paired_sign_flip(np.ones(30), resamples=500, seed=9)
    assert result["p_method"] == "sign_flip_monte_carlo"
    assert 0 < result["p_two_sided"] <= 1
    assert result["p_two_sided"] >= 1 / 501
    with pytest.raises(ValueError):
        paired_sign_flip([np.nan])
    with pytest.raises(ValueError):
        holm([np.nan], family_size=30)


def test_empty_study_is_not_complete(tmp_path):
    output = tmp_path / "summary"
    audit = aggregate_module.aggregate_s23(tmp_path / "absent", output)
    assert audit["runs_expected"] == 420
    assert audit["runs_found"] == audit["complete"] == 0
    assert not audit["matrix_complete"]
    assert len(audit["missing"]) == 420
    checks = (output / "predictions_check.md").read_text()
    for hypothesis in ("H1", "H2", "H3", "H4"):
        assert f"{hypothesis}: insufficient" in checks
    assert ": held" not in checks
    assert csv_rows(output, "paired.csv") == []


def test_aggregate_stratifies_and_uses_fixed_holm_family(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "summary"
    for dataset, n, gain in (("eth3d", 1, 1), ("7scenes", 9, 10)):
        for arm, value in (("A2", 10), ("A3", 12), ("A4", 10 + gain)):
            write_cell(root, arm, [value] * n, dataset)
    audit = aggregate_module.aggregate_s23(root, output)
    rows = csv_rows(output, "recovery.csv")
    assert len(rows) == 1
    assert float(rows[0]["R"]) == pytest.approx(2.75)
    assert float(rows[0]["ci_low"]) == float(rows[0]["ci_high"]) == 2.75
    assert rows[0]["complete"] == "False"
    primary = [row for row in csv_rows(output, "paired.csv") if row["pair"] == "A4-A2"]
    raw = [float(row["p_raw"]) for row in primary]
    assert [float(row["p_holm"]) for row in primary] == pytest.approx(holm(raw, family_size=30))
    assert all(row["p_method"] == "sign_flip_exact" for row in primary)
    assert audit["complete"] == 6
    assert not audit["matrix_complete"]
    assert "H1: insufficient" in (output / "predictions_check.md").read_text()


@pytest.mark.parametrize("field", ["source_hash", "git_commit", "upstream_commit", "checkpoint_sha256",
                                   "checkpoint_revision", "model_config_hash", "precision",
                                   "versions", "cuda_version", "views_hash"])
def test_aggregate_rejects_mixed_scientific_provenance(tmp_path, field):
    root = tmp_path / "runs"
    write_cell(root, "A2", [1, 2])
    write_cell(root, "A4", [2, 3], change=lambda row: row["provenance"].update({field: "changed"}))
    with pytest.raises(ValueError, match="Mixed"):
        aggregate_module.aggregate_s23(root, tmp_path / "summary")


def test_aggregate_rejects_metric_and_scene_provenance_changes(tmp_path):
    root = tmp_path / "runs"
    write_cell(root, "A2", [1, 2])
    path = write_cell(root, "A4", [2, 3], change=lambda row: row.update(metric_protocol="different"))
    with pytest.raises(ValueError, match="metric"):
        aggregate_module.aggregate_s23(root, tmp_path / "summary")

    def mixed_scene(row):
        meta = copy.deepcopy(row["provenance"])
        meta["versions"] = {"torch": "other"}
        row["per_scene"][0]["provenance"] = meta

    write_cell(root, "A4", [2, 3], change=mixed_scene)
    assert path.exists()
    with pytest.raises(ValueError, match="scene provenance"):
        aggregate_module.aggregate_s23(root, tmp_path / "summary")


def test_false_completed_status_and_partial_pairing(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "summary"
    write_cell(root, "A2", [1, 2, 3])

    def fail_scene(row):
        row["per_scene"][1].update(status="failed", auc3=None)

    write_cell(root, "A4", [2, 3, 4], change=fail_scene)
    audit = aggregate_module.aggregate_s23(root, output)
    assert audit["complete"] == 1
    assert audit["partial_or_failed"][0]["missing_or_failed"] == ["scene1"]
    pair = csv_rows(output, "paired.csv")[0]
    assert pair["n"] == "2"
    assert pair["complete"] == "False"
    assert float(pair["p_raw"]) == 0.5
    assert read_json(output / "audit.json")["matrix_complete"] is False


def test_empty_scene_list_and_missing_provenance_are_not_accepted_as_complete(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "summary"
    write_cell(root, "A2", [])
    assert aggregate_module.aggregate_s23(root, output)["complete"] == 0
    write_cell(root, "A2", [1], change=lambda row: row["provenance"].pop("versions"))
    with pytest.raises(ValueError, match="Missing scientific provenance"):
        aggregate_module.aggregate_s23(root, output)


def test_duplicate_scene_is_rejected(tmp_path):
    root = tmp_path / "runs"
    write_cell(root, "A2", [1, 2], change=lambda row: row["per_scene"].append(row["per_scene"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_module.aggregate_s23(root, tmp_path / "summary")


def write_reconstruction(path, records, **changes):
    report = {"protocol": "da3_native_recon_unposed", "dataset": read_json(path)["dataset"],
              "source_result": str(path), "source_sha256": file_hash(path),
              "status": "completed", "per_scene": records}
    report.update(changes)
    atomic_json(path.with_name("recon.json"), report)
    return report


def test_recon_merging_native_metrics_by_scene_without_mutation(tmp_path, monkeypatch):
    root, output = tmp_path / "runs", tmp_path / "summary"
    path = write_cell(root, "A4", [10, 20])
    other = write_cell(root, "A2", [1, 2])
    predictions = tmp_path / "predictions.npz"
    atomic_json(predictions, {"synthetic": "identity only, no inference"})
    result = read_json(path)
    for scene in result["per_scene"]:
        scene["predictions_file"] = str(predictions)
    atomic_json(path, result)
    before = path.read_bytes()
    report = write_reconstruction(path, [
        {"scene": "scene1", "status": "completed", "predictions_sha256": file_hash(predictions),
         "metrics": {"fscore": .4, "precision": .5, "recall": 1 / 3,
                     "acc": .002, "comp": .006, "overall": .004}},
        {"scene": "scene0", "status": "completed", "predictions_sha256": file_hash(predictions),
         "metrics": {"F1": .8, "precision": 1., "recall": 2 / 3,
                     "acc": .004, "comp": .008, "overall": .006}},
    ])
    # Keep references to the actual loaded objects to detect in-memory mutation too.
    documents = {path: result, path.with_name("recon.json"): report, other: read_json(other)}
    original = copy.deepcopy(documents)
    monkeypatch.setattr(aggregate_module, "read_json", lambda name: documents[Path(name)])
    merged, _ = aggregate_module._merge_reconstruction(result, path)
    assert merged["per_scene"][0]["fscore"] == .8
    assert merged["per_scene"][1]["fscore"] == .4
    audit = aggregate_module.aggregate_s23(root, output)
    row = next(row for row in csv_rows(output, "cells.csv") if row["arm"] == "A4")
    assert float(row["auc3"]) == 15
    assert float(row["fscore"]) == pytest.approx(.6)
    assert float(row["precision"]) == .75
    assert float(row["recall"]) == .5
    assert float(row["accuracy_mm"]) == 3
    assert float(row["completeness_mm"]) == 7
    assert float(row["chamfer_mm"]) == 5
    assert row["fscore_units"] == "fraction"
    assert row["distance_units"] == "mm"
    assert row["recon_n"] == "2"
    assert audit["complete"] == 2
    assert audit["reconstruction_incomplete_count"] == 1
    assert documents == original
    assert path.read_bytes() == before


@pytest.mark.parametrize("dataset,scale", [("eth3d", 1000), ("scannetpp", 1000),
                                         ("7scenes", 1000), ("hiroom", 1000), ("dtu", 1)])
def test_recon_merging_native_distance_units_and_dtu_direction_labels(tmp_path, dataset, scale):
    path = write_cell(tmp_path / "runs", "A4", [10], dataset=dataset)
    write_reconstruction(path, [{"scene": "scene0", "status": "completed",
                                 "metrics": {"acc": 2., "comp": 4., "overall": 3.}}])
    output = tmp_path / "summary"
    aggregate_module.aggregate_s23(tmp_path / "runs", output)
    row = csv_rows(output, "cells.csv")[0]
    assert float(row["chamfer_mm"]) == 3 * scale
    assert float(row["accuracy_mm"]) == (4 if dataset == "dtu" else 2) * scale
    assert float(row["completeness_mm"]) == (2 if dataset == "dtu" else 4) * scale
    assert row["fscore"] == row["precision"] == row["recall"] == ""


@pytest.mark.parametrize("mode", ["missing", "scene_failed", "report_failed", "report_error",
                                 "pose_failed", "empty_metrics", "unverifiable_prediction"])
def test_recon_merging_excludes_missing_failed_and_unverified(tmp_path, mode):
    root, output = tmp_path / "runs", tmp_path / "summary"
    path = write_cell(root, "A4", [10])
    result = read_json(path)
    # Stale inline reconstruction must not fill in absent/failed bridge output.
    result["per_scene"][0].update(fscore=.99, chamfer_mm=99)
    if mode == "pose_failed":
        result["per_scene"][0]["status"] = "failed"
    atomic_json(path, result)
    record = {"scene": "scene0", "status": "completed", "metrics": {"fscore": .8}}
    changes = {}
    if mode == "scene_failed":
        record["status"] = "failed"
    elif mode == "report_failed":
        changes["status"] = "failed"
    elif mode == "report_error":
        changes["error"] = "native loader failed"
    elif mode == "empty_metrics":
        record["metrics"] = {}
    elif mode == "unverifiable_prediction":
        record["predictions_sha256"] = "missing-file-hash"
    if mode != "missing":
        write_reconstruction(path, [record], **changes)
    audit = aggregate_module.aggregate_s23(root, output)
    row = csv_rows(output, "cells.csv")[0]
    assert row["fscore"] == row["chamfer_mm"] == ""
    assert row["recon_n"] == "0"
    assert audit["complete"] == (0 if mode == "pose_failed" else 1)
    assert audit["reconstruction_incomplete_count"] == 1
    assert audit["reconstruction"][0]["missing_or_failed"] == ["scene0"]


def test_recon_merging_partial_report_keeps_only_completed_pose_scenes(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "summary"
    path = write_cell(root, "A4", [10, 20, 30, 40])
    result = read_json(path)
    result["per_scene"][2]["status"] = "failed"
    atomic_json(path, result)
    write_reconstruction(path, [
        {"scene": "scene2", "status": "completed", "metrics": {"fscore": .9}},
        {"scene": "scene1", "status": "failed", "metrics": {"fscore": .8}},
        {"scene": "scene0", "status": "completed", "metrics": {"fscore": .4}},
    ], status="partial_or_failed")
    audit = aggregate_module.aggregate_s23(root, output)
    row = csv_rows(output, "cells.csv")[0]
    assert row["n"] == "3"
    assert row["recon_n"] == "1"
    assert float(row["fscore"]) == .4
    assert audit["reconstruction"][0]["completed_scenes"] == ["scene0"]
    assert audit["reconstruction"][0]["missing_or_failed"] == ["scene1", "scene2", "scene3"]


def test_recon_merging_recorded_prediction_identity_without_local_file(tmp_path):
    root, output = tmp_path / "runs", tmp_path / "summary"
    path = write_cell(root, "A4", [10], change=lambda row: row["per_scene"][0].update(
        predictions_sha256="saved-identity", predictions_file="absent-prediction.npz"))
    write_reconstruction(path, [{"scene": "scene0", "status": "completed",
                                 "predictions_sha256": "saved-identity", "metrics": {"fscore": .4}}])
    audit = aggregate_module.aggregate_s23(root, output)
    assert audit["reconstruction_incomplete_count"] == 0
    assert audit["reconstruction"][0]["source_hash_verified"]
    assert audit["reconstruction"][0]["unverified_predictions"] == []


def test_recon_merging_absent_optional_hashes_are_audited(tmp_path):
    path = write_cell(tmp_path / "runs", "A4", [10])
    report = write_reconstruction(path, [{"scene": "scene0", "status": "completed",
                                          "metrics": {"fscore": .4}}])
    report.pop("source_sha256")
    atomic_json(path.with_name("recon.json"), report)
    merged, audit = aggregate_module._merge_reconstruction(read_json(path), path)
    assert merged["per_scene"][0]["fscore"] == .4
    assert audit["source_hash_verified"] is False
    assert audit["missing_prediction_hashes"] == ["scene0"]


def test_recon_merging_rejects_source_changed_after_reconstruction(tmp_path):
    path = write_cell(tmp_path / "runs", "A4", [10])
    write_reconstruction(path, [{"scene": "scene0", "status": "completed",
                                 "metrics": {"fscore": .4}}])
    result = read_json(path)
    result["per_scene"][0]["auc3"] = 11
    atomic_json(path, result)
    with pytest.raises(ValueError, match="source hash mismatch"):
        aggregate_module.aggregate_s23(tmp_path / "runs", tmp_path / "summary")


@pytest.mark.parametrize("mode", ["source_hash", "prediction_file", "prediction_record",
                                 "dataset", "cell", "protocol", "duplicate", "unexpected"])
def test_recon_merging_rejects_identity_and_scene_mismatches(tmp_path, mode):
    root = tmp_path / "runs"
    path = write_cell(root, "A4", [10])
    result = read_json(path)
    prediction = tmp_path / "prediction.npz"
    atomic_json(prediction, {"identity": "only"})
    result["per_scene"][0]["predictions_file"] = str(prediction)
    if mode == "prediction_record":
        result["per_scene"][0]["predictions_sha256"] = "different"
    atomic_json(path, result)
    record = {"scene": "scene0", "status": "completed", "metrics": {"fscore": .4},
              "predictions_sha256": file_hash(prediction)}
    records, changes = [record], {}
    if mode == "source_hash":
        changes["source_sha256"] = "different"
    elif mode == "prediction_file":
        record["predictions_sha256"] = "different"
    elif mode == "dataset":
        changes["dataset"] = "dtu"
    elif mode == "cell":
        changes["arm"] = "A2"
    elif mode == "protocol":
        changes["protocol"] = "unknown_units"
    elif mode == "duplicate":
        records.append(copy.deepcopy(record))
    elif mode == "unexpected":
        record["scene"] = "other"
    write_reconstruction(path, records, **changes)
    with pytest.raises(ValueError, match="Reconstruction|reconstruction"):
        aggregate_module.aggregate_s23(root, tmp_path / "summary")


@pytest.mark.parametrize("metrics", [{"F1": 80}, {"fscore": .5, "F1": .6},
                                     {"precision": -1}, {"overall": -1}, {"acc": "0.1"},
                                     {"recall": True}])
def test_recon_merging_rejects_invalid_metrics(tmp_path, metrics):
    path = write_cell(tmp_path / "runs", "A4", [10])
    write_reconstruction(path, [{"scene": "scene0", "status": "completed", "metrics": metrics}])
    with pytest.raises(ValueError, match="Invalid reconstruction"):
        aggregate_module.aggregate_s23(tmp_path / "runs", tmp_path / "summary")
