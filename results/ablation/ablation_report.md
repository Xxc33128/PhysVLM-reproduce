# S-P Map Ablation Report

## Overall Accuracy

| Depth Mode | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| normal | 1279 | 1600 | 79.94% |
| black | 1239 | 1600 | 77.44% |
| mismatch | 1270 | 1600 | 79.37% |

## Per-Robot Accuracy

| Depth Mode | Robot | Correct | Total | Accuracy |
| --- | --- | --- | --- | --- |
| normal | CR5 | 325 | 400 | 81.25% |
| normal | FR5 | 335 | 400 | 83.75% |
| normal | PANDA | 321 | 400 | 80.25% |
| normal | UR5 | 298 | 400 | 74.50% |
| black | CR5 | 308 | 400 | 77.00% |
| black | FR5 | 333 | 400 | 83.25% |
| black | PANDA | 311 | 400 | 77.75% |
| black | UR5 | 287 | 400 | 71.75% |
| mismatch | CR5 | 320 | 400 | 80.00% |
| mismatch | FR5 | 335 | 400 | 83.75% |
| mismatch | PANDA | 320 | 400 | 80.00% |
| mismatch | UR5 | 295 | 400 | 73.75% |

## Error Direction

| Depth Mode | False Positive (No -> Yes) | False Negative (Yes -> No) |
| --- | --- | --- |
| normal | 319 | 2 |
| black | 360 | 1 |
| mismatch | 327 | 3 |

## Interpretation Guide

- `normal` uses the original RGB + aligned S-P Map input.
- `black` keeps the `<depth>` token but replaces the S-P Map with an all-black placeholder.
- `mismatch` keeps the RGB image but uses the next scene's S-P Map from the same robot split.
- If `normal` is stronger than `black`, the S-P Map contributes useful physical information.
- If `mismatch` drops or changes the confusion pattern, the model is sensitive to incorrect physical context rather than only RGB or language priors.
