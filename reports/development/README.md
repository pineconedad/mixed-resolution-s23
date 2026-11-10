# Development results, not frozen paper evidence

All five scenes use the same committed seed-42 views. All eight arms complete on both models, with no missing or excluded scenes. Every task metric is independently recomputed from the saved camera matrices. The full-checkpoint fp32 identity gates match the native models exactly on both checked scenes.

| Model | A1 mix | A2 global alignment | A4 RMFA | A3 upsample | A8 real pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Pi3 | 1.488 | 1.626 | 7.675 | 12.876 | 13.223 |
| VGGT-1B | 0.606 | 0.844 | 1.291 | 4.447 | 7.861 |

Values are mean pose AUC@3 in percent, under the independently rebuilt standard relative-pose protocol. The alignment operation in these preserved runs is global-only with effective grid ratios, not a verified reconstruction of historical S5. The full-scope nominal-rho interpretation is separately specified before its results are collected. Do not silently relabel or pool the two protocols.

Pi3's real-pixel-versus-upsampled gap is small on these scenes. VGGT's gap is +3.413 AUC points, with paired bootstrap interval [1.639, 4.889]. VGGT nevertheless also benefits from upsampling versus plain mix; the stronger expectation that only real pixels help it is not supported here. Its A4-A2 interval includes zero, so no reliable RMFA gain is established for VGGT in this protocol. Ratios with undefined bootstrap denominators are suppressed rather than forced into a clean story.

Hardware/timing provenance remains separate. Pi3 ran on RTX A4000; VGGT ran on L40S. Do not compare their runtimes as if measured on the same GPU. No reconstruction, frozen seeds, held-out generalization or DA3 method result is claimed.

See comparison.json/csv and architecture-controls.png, plus the complete session directories and control logs under reports/. The user-supplied GPU was stopped after verified backup; its host-local workspace and unattached network volume remain independently billable until the user authorizes their deletion.
