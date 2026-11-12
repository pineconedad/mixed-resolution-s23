import numpy as np

from mixres.eval.pose import score_poses_da3
from mixres.eval.recon import point_metrics
from mixres.probes.cost import transformer_flops


def test_da3_pose_identity_is_perfect():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0, 1, 2]
    scores = score_poses_da3(poses, poses)
    assert scores["auc3"] == 100.0


def test_point_and_flop_metrics():
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert point_metrics(points, points, 0.01)["fscore"] == 1.0
    flops = transformer_flops([{"frame_total_tokens": 4, "frame_attention_quadratic": 8,
                                "total_tokens": 8, "global_attention_quadratic": 64}], 16)
    assert flops["global"] > flops["frame"] > 0
