from pathlib import Path

import numpy as np
from PIL import Image

from mixres.data import benchmark


def test_7scenes_manifest_is_seeded_and_excludes_development(tmp_path, monkeypatch):
    dataset = tmp_path / "7scenes"
    scene = dataset / "7Scenes/chess/seq-01"
    scene.mkdir(parents=True)
    for index in range(24):
        prefix = f"frame-{index:06d}"
        Image.new("RGB", (640, 480), color=(index, 0, 0)).save(scene / f"{prefix}.color.png")
        Image.new("I;16", (640, 480), color=index + 1).save(scene / f"{prefix}.depth.png")
        pose = np.eye(4)
        pose[0, 3] = index
        np.savetxt(scene / f"{prefix}.pose.txt", pose)
    monkeypatch.setattr(benchmark, "SEVEN_SCENES", ("chess",))
    first = benchmark.make_manifest("7scenes", dataset, tmp_path, 42, [4096])
    repeat = benchmark.make_manifest("7scenes", dataset, tmp_path, 42, [4096])
    assert first == repeat
    frames = {view["frame"] for view in first["scenes"][0]["views"]}
    second = benchmark.make_manifest("7scenes", dataset, tmp_path, 1000, [4096],
                                     excluded={"chess": frames})
    assert frames.isdisjoint(view["frame"] for view in second["scenes"][0]["views"])
    assert sum(view["sharp"] for view in first["scenes"][0]["views"]) == 4
    assert all(view["sizes"]["4096"]["hi_hw"] == [322, 420]
               for view in first["scenes"][0]["views"])
    assert all(view["sizes"]["4096"]["uniform_hw"] == [224, 294]
               for view in first["scenes"][0]["views"])
