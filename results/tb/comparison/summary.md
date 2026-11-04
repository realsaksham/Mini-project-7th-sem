# 4-way comparison

Seeds: 42, 43, 44 (mean ± std over seeds; fixed data split). Test set metrics unless noted.

| model | accuracy | precision | recall | F1 | ROC-AUC | val F1 | params | train time (s) |
|---|---|---|---|---|---|---|---|---|
| CNN + classical | 0.9804 ± 0.0056 | 0.9326 ± 0.0286 | 0.9524 ± 0.0381 | 0.9418 ± 0.0172 | 0.9975 ± 0.0004 | 0.9438 ± 0.0154 | 132,385 | 16 |
| CNN + quantum | 0.9841 ± 0.0032 | 0.9612 ± 0.0094 | 0.9429 ± 0.0165 | 0.9519 ± 0.0098 | 0.9980 ± 0.0014 | 0.9576 ± 0.0034 | 132,513 | 46 |
| ResNet18 + classical | 0.9921 ± 0.0057 | 0.9875 ± 0.0216 | 0.9651 ± 0.0361 | 0.9758 ± 0.0178 | 0.9991 ± 0.0006 | 0.9745 ± 0.0229 | 11,192,961 | 314 |
| ResNet18 + quantum | 0.9947 ± 0.0018 | 0.9936 ± 0.0056 | 0.9746 ± 0.0145 | 0.9839 ± 0.0057 | 0.9961 ± 0.0046 | 0.9904 ± 0.0049 | 11,193,089 | 346 |

## Error analysis (seed 42, 630 test images)

| model | errors | missed TB (false negatives) | false alarms (false positives) |
|---|---|---|---|
| CNN + classical | 16 | 9 | 7 |
| CNN + quantum | 10 | 7 | 3 |
| ResNet18 + classical | 1 | 1 | 0 |
| ResNet18 + quantum | 2 | 1 | 1 |

Images misclassified by at least one model: 22; by all four models: 1.
