# Dropped scenes

The fresh S23 manifests require 12 seed-42 views and a disjoint 12-view pool for frozen seeds. Drops are decided from frame counts before inference.

| Dataset | Scene | Reason |
| --- | --- | --- |
| ETH3D | pipes | 14 eligible frames cannot supply two disjoint 12-view sets |
| ETH3D | relief | 19 frames after native filename exclusions cannot supply two disjoint 12-view sets |
| ETH3D | relief_2 | 20 frames after native filename exclusions cannot supply two disjoint 12-view sets |

No 7-Scenes scene is dropped. ScanNet++, DTU and HiRoom have not been indexed, so this file makes no claim about their eventual drops.
