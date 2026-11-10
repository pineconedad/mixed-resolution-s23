# Data preparation

Only the independent quick-test development subset has been opened.

- Dataset: official Microsoft 7-Scenes release.
- Source: <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>.
- License: non-commercial research, as linked by the official source.
- Selected scenes, decided before metrics: chess, fire, heads, office, pumpkin.
- Sequence: seq-01 in every selected scene, matching DA3's benchmark loader.
- Input sizes: original 640x480 RGB PNGs. No preprocessing saved over originals.
- Candidate set: sorted color frames in the official sequence archive.
- Sampling: 12 frames without replacement, NumPy default_rng(42 + sorted scene index).
- Sharp views: four FPS camera centres, starting from the first drawn frame.
- Poses: official camera-to-world matrices; both c2w and inverse w2c saved in the view JSON.
- Intrinsics: DA3 benchmark constants fx=fy=585, cx=320, cy=240. These are nominal Kinect intrinsics, not a claim that RGB is calibrated.
- Pose/geometry projection checks for full reconstruction are pending. The quick test uses pose only.
- Files: 60 original RGB images and their pose files, with per-file SHA-256 and source ETag receipts under data/7scenes.
- Archives: only seq-01 is fetched from each outer scene ZIP. Some nested ZIPs are recompressed, so those sequence ZIPs are retained locally under data/7scenes/_archives. They are not needed on the pod.
- Download dates and byte counts are recorded in each receipt.json.

ETH3D, ScanNet++, DTU and HiRoom have not been downloaded or evaluated. Frozen and held-out views have not been generated or opened.
