# Full-scope positional-alignment controls

This is the latest development protocol. It scales all applicable reused patch coordinates by the configured nominal rho, following the native Pi3 position tensor through frame attention, global attention and the head decoders. VGGT has RoPE only in the aggregator. The earlier global-only/effective-grid-ratio results remain preserved separately and are not silently relabelled as this protocol.

All eight arms complete on both models, five original 7-Scenes scenes, seed 42, nominal 4k budget, identical committed views. These are exploratory results, not the frozen main study.

| Model | A1 mix | A2 full position alignment | A4 RMFA-pixel | A5 RMFA-token | A3 upsample | A8 real pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pi3 | 1.477 | 5.602 | 8.336 | 8.415 | 12.874 | 13.219 |
| VGGT | 0.600 | 3.150 | 1.314 | 2.038 | 4.447 | 7.861 |

Mean pose AUC@3 is in percent. The Pi3 recovery ratio is 0.375964, 95% bootstrap interval [0.192650, 0.695516]. Its point estimate is below the plan's 0.5 method-framing threshold. The earlier 53.8% recovery used the weaker global-only baseline and is not the appropriate headline for full-scope position alignment.

VGGT A4-A2 is -1.835963 AUC points, interval [-3.747889, 0.027155]. It does not establish a transferable RMFA benefit. Its recovery ratio is suppressed because the upsampling-minus-position-alignment interval includes zero. The real-pixel-minus-upsampled gap remains positive for VGGT.

Both native fp32 identity gates pass exactly. All forty task evaluations in each session complete without failed or dropped scenes. The archives, every periodic snapshot, raw logs, actual environments, camera matrices, summaries and audits are preserved under reports/sessions and reports/control.

No frozen seeds, held-out generalization, DA3 method result, development sweep, reconstruction score or historical T9 reproduction is claimed. Default RMFA does not yet support a method-paper headline. Provisional framing follows the plan's diagnosis option until further development and frozen evidence justify a change.
