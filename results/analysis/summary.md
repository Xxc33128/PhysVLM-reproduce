# PhysVLM Reproduction Result Summary

## Run Scope

- Benchmark file: `phys_bench_sim_qas.json`
- Total examples: 1600
- Robots: CR5, FR5, PANDA, UR5
- Runtime: Colab A100 fp16, no 4-bit quantization
- Evaluation rule: first-token Yes/No match after stripping special tokens such as `<|endoftext|>`.

## Accuracy

| Robot | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| ALL | 1279 | 1600 | 79.94% |
| CR5 | 325 | 400 | 81.25% |
| FR5 | 335 | 400 | 83.75% |
| PANDA | 321 | 400 | 80.25% |
| UR5 | 298 | 400 | 74.50% |

## Question Type Accuracy

| Question Type | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| direct_pick | 668 | 800 | 83.50% |
| reachable_space | 611 | 800 | 76.38% |

## Overall Yes/No Confusion

Rows are labels, columns are predictions.

| Label | Pred Yes | Pred No |
| --- | --- | --- |
| Yes | 972 | 2 |
| No | 319 | 307 |

## Observations

- Overall accuracy is 79.94% across 1600 examples.
- FR5 is the strongest robot split at 83.75%; UR5 is the weakest at 74.50%.
- The model predicts `Yes` more often than the label distribution: 1291 Yes predictions vs 974 Yes labels.
- Most errors are false positives: label `No`, prediction `Yes` accounts for 319 of 321 errors.

## Generated Files

- `metrics_summary.csv`: per-robot accuracy table.
- `question_type_summary.csv`: accuracy by robot and question template.
- `confusion_by_robot.csv`: Yes/No confusion counts per robot.
- `prediction_distribution.csv`: output bias summary.
- `wrong_cases_top50.csv`: first 50 error cases for manual inspection.
- `object_error_summary.csv`: object-level error rates.
