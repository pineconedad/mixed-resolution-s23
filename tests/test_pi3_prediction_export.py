import numpy as np
import torch

from mixres.study_runner import atomic_predictions


def test_pi3_prediction_geometry_roundtrip(tmp_path):
    height, width = 9, 13
    # Off-center principal point and unequal focal lengths exercise native fitting.
    intrinsics = torch.tensor([[17.0, 0.0, 4.25], [0.0, 23.0, 5.5], [0.0, 0.0, 1.0]])
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    depth = 1.0 + x.float() / width + y.float() / height
    local_points = torch.stack(((x - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
                                (y - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
                                depth), dim=-1)
    camera = torch.eye(4)
    camera[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    confidence = torch.linspace(0.0, 1.0, height * width).reshape(height, width, 1)
    prediction = dict(local_points=local_points, points=local_points + camera[:3, 3],
                      camera_poses=camera, conf=confidence)
    path = tmp_path / "predictions.npz"
    atomic_predictions(path, [prediction])

    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["intrinsics_0"], intrinsics.numpy(), atol=2e-5)
        np.testing.assert_array_equal(saved["depth_0"], depth.numpy())
        np.testing.assert_array_equal(saved["conf_0"], confidence.numpy())
        np.testing.assert_array_equal(saved["camera_poses_0"], camera.numpy())
        source = saved["intrinsics_source_0"].item()
        assert "recover_intrinsic_from_rays_d" in source
        assert "predicted local_points" in source and "no GT K" in source
        assert "points_0" not in saved and "local_points_0" not in saved
        pixels = np.stack((x.numpy(), y.numpy(), np.ones((height, width))), axis=-1)
        recovered = (pixels @ np.linalg.inv(saved["intrinsics_0"]).T
                     * saved["depth_0"][..., None])
        np.testing.assert_allclose(recovered, local_points.numpy(), atol=2e-6)

    assert "depth" not in prediction and "intrinsics" not in prediction
    assert not path.with_suffix(".tmp.npz").exists()
