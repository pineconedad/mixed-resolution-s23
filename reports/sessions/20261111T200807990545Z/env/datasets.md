# Data preparation

The independent quick-test subset and the authorized ETH3D/7-Scenes development archives have been opened. No held-out archive or result has been opened.

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

DA3-BENCH development archives are pinned to dataset revision cb98821d5d4f721704f11bafde0403826b82b638. ETH3D is 15101568883 bytes with SHA-256 2a12b961e72566e194d707db02f2be65686ff12ad5f4347cb16b4c266b2d6917. 7-Scenes is 3553806786 bytes with SHA-256 4fc12afe9922c975068abd8918b897580926a932b718ad7646f6822205ef5b88. Both archives passed their integrity checks and were extracted locally in the layout consumed by the pinned benchmark loaders.

Fresh S23 manifests exist for seeds 42, 1000, 2000 and 3000 at nominal 4k and 8k. Generating metadata does not open model results. Every frozen seed excludes the seed-42 frames. ETH3D `pipes`, `relief` and `relief_2` are dropped before inference because their 14, 19 and 20 eligible frames cannot supply disjoint 12-view development and frozen sets. Native filename-suffix exclusions are applied; superseded metadata is retained. See dropped_scenes.md.

ScanNet++ remains absent pending acceptance of its original terms. DTU/DTU-64 and HiRoom remain absent and sealed until the development sweep, frozen configs and predictions are committed. No model evaluation has run on the newly extracted benchmark data yet. Pose/geometry projection checks and contact sheets for the full development archives remain pending.
