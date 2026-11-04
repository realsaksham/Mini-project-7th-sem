# Hybrid study: classical vs. C->Q vs. Q->C

Seeds 42, 43, 44; mean ± std over seeds; same fixed validation/test split for every run.

## 2% of training data (58 images)

| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |
|---|---|---|---|---|---|---|
| Classical CNN | 0.8995 ± 0.0217 | 0.4952 ± 0.2089 | 0.6011 ± 0.1698 | 0.9107 ± 0.0269 | 132,385 (33) | 5 |
| CNN + matched classical head | 0.8333 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9176 ± 0.0265 | 132,509 (157) | 5 |
| C->Q hybrid (CNN + quantum head) | 0.7376 ± 0.2284 | 0.4698 ± 0.4706 | 0.3408 ± 0.2500 | 0.9273 ± 0.0252 | 132,513 (161) | 14 |
| Q->C hybrid (quanvolution + CNN) | 0.9101 ± 0.0024 | 0.5683 ± 0.0525 | 0.6772 ± 0.0199 | 0.9070 ± 0.0361 | 34,297 (33) | 1 |
| Random patch + CNN (control) | 0.8333 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8080 ± 0.0379 | 34,297 (33) | 1 |
| Avg-pool + CNN (control) | 0.9101 ± 0.0184 | 0.5810 ± 0.2076 | 0.6663 ± 0.1383 | 0.9067 ± 0.0322 | 34,081 (33) | 1 |
| Learned 2x2 conv + CNN (control) | 0.8720 ± 0.0128 | 0.3206 ± 0.0799 | 0.4509 ± 0.0762 | 0.8857 ± 0.0315 | 34,317 (33) | 2 |

## 5% of training data (147 images)

| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |
|---|---|---|---|---|---|---|
| Classical CNN | 0.9434 ± 0.0064 | 0.7524 ± 0.0530 | 0.8152 ± 0.0268 | 0.9519 ± 0.0100 | 132,385 (33) | 6 |
| CNN + matched classical head | 0.8688 ± 0.0614 | 0.2254 ± 0.3904 | 0.2630 ± 0.4555 | 0.9086 ± 0.0721 | 132,509 (157) | 6 |
| C->Q hybrid (CNN + quantum head) | 0.9397 ± 0.0130 | 0.6857 ± 0.1193 | 0.7873 ± 0.0638 | 0.9652 ± 0.0065 | 132,513 (161) | 17 |
| Q->C hybrid (quanvolution + CNN) | 0.9275 ± 0.0198 | 0.6889 ± 0.1433 | 0.7544 ± 0.0930 | 0.9471 ± 0.0166 | 34,297 (33) | 2 |
| Random patch + CNN (control) | 0.8720 ± 0.0075 | 0.2794 ± 0.0770 | 0.4163 ± 0.0857 | 0.8925 ± 0.0052 | 34,297 (33) | 2 |
| Avg-pool + CNN (control) | 0.9275 ± 0.0225 | 0.6667 ± 0.1350 | 0.7491 ± 0.0987 | 0.9421 ± 0.0165 | 34,081 (33) | 2 |
| Learned 2x2 conv + CNN (control) | 0.9185 ± 0.0336 | 0.6159 ± 0.1891 | 0.7053 ± 0.1555 | 0.9232 ± 0.0289 | 34,317 (33) | 2 |

## 10% of training data (294 images)

| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |
|---|---|---|---|---|---|---|
| Classical CNN | 0.9619 ± 0.0027 | 0.8381 ± 0.0252 | 0.8800 ± 0.0091 | 0.9767 ± 0.0022 | 132,385 (33) | 8 |
| CNN + matched classical head | 0.9614 ± 0.0033 | 0.8159 ± 0.0334 | 0.8756 ± 0.0098 | 0.9748 ± 0.0041 | 132,509 (157) | 8 |
| C->Q hybrid (CNN + quantum head) | 0.9534 ± 0.0219 | 0.8000 ± 0.0990 | 0.8496 ± 0.0767 | 0.9699 ± 0.0090 | 132,513 (161) | 22 |
| Q->C hybrid (quanvolution + CNN) | 0.9476 ± 0.0069 | 0.7937 ± 0.0542 | 0.8341 ± 0.0280 | 0.9680 ± 0.0046 | 34,297 (33) | 2 |
| Random patch + CNN (control) | 0.8720 ± 0.0048 | 0.3270 ± 0.0702 | 0.4564 ± 0.0654 | 0.9007 ± 0.0050 | 34,297 (33) | 2 |
| Avg-pool + CNN (control) | 0.9577 ± 0.0060 | 0.7968 ± 0.0220 | 0.8625 ± 0.0201 | 0.9700 ± 0.0040 | 34,081 (33) | 2 |
| Learned 2x2 conv + CNN (control) | 0.9328 ± 0.0239 | 0.6825 ± 0.1239 | 0.7685 ± 0.0923 | 0.9585 ± 0.0173 | 34,317 (33) | 3 |

## 25% of training data (735 images)

| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |
|---|---|---|---|---|---|---|
| Classical CNN | 0.9704 ± 0.0033 | 0.8857 ± 0.0165 | 0.9089 ± 0.0082 | 0.9855 ± 0.0013 | 132,385 (33) | 5 |
| CNN + matched classical head | 0.8810 ± 0.0825 | 0.3016 ± 0.5224 | 0.3089 ± 0.5351 | 0.9302 ± 0.0972 | 132,509 (157) | 5 |
| C->Q hybrid (CNN + quantum head) | 0.9614 ± 0.0211 | 0.8095 ± 0.1260 | 0.8712 ± 0.0798 | 0.9849 ± 0.0131 | 132,513 (161) | 14 |
| Q->C hybrid (quanvolution + CNN) | 0.9630 ± 0.0046 | 0.8476 ± 0.0190 | 0.8841 ± 0.0147 | 0.9813 ± 0.0045 | 34,297 (33) | 2 |
| Random patch + CNN (control) | 0.8783 ± 0.0033 | 0.3556 ± 0.0721 | 0.4902 ± 0.0516 | 0.9065 ± 0.0048 | 34,297 (33) | 1 |
| Avg-pool + CNN (control) | 0.9640 ± 0.0040 | 0.8889 ± 0.0145 | 0.8918 ± 0.0113 | 0.9802 ± 0.0036 | 34,081 (33) | 2 |
| Learned 2x2 conv + CNN (control) | 0.9534 ± 0.0243 | 0.8381 ± 0.0744 | 0.8571 ± 0.0748 | 0.9719 ± 0.0101 | 34,317 (33) | 2 |

## 100% of training data (2940 images)

| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |
|---|---|---|---|---|---|---|
| Classical CNN | 0.9767 ± 0.0033 | 0.9238 ± 0.0095 | 0.9298 ± 0.0094 | 0.9966 ± 0.0009 | 132,385 (33) | 15 |
| CNN + matched classical head | 0.9772 ± 0.0040 | 0.9079 ± 0.0145 | 0.9301 ± 0.0119 | 0.9959 ± 0.0011 | 132,509 (157) | 17 |
| C->Q hybrid (CNN + quantum head) | 0.9841 ± 0.0032 | 0.9429 ± 0.0165 | 0.9519 ± 0.0098 | 0.9980 ± 0.0014 | 132,513 (161) | 43 |
| Q->C hybrid (quanvolution + CNN) | 0.9788 ± 0.0033 | 0.9270 ± 0.0145 | 0.9359 ± 0.0096 | 0.9954 ± 0.0017 | 34,297 (33) | 5 |
| Random patch + CNN (control) | 0.9053 ± 0.0087 | 0.5206 ± 0.0715 | 0.6449 ± 0.0543 | 0.9336 ± 0.0035 | 34,297 (33) | 4 |
| Avg-pool + CNN (control) | 0.9815 ± 0.0009 | 0.9556 ± 0.0110 | 0.9451 ± 0.0020 | 0.9956 ± 0.0008 | 34,081 (33) | 4 |
| Learned 2x2 conv + CNN (control) | 0.9831 ± 0.0072 | 0.9524 ± 0.0000 | 0.9497 ± 0.0204 | 0.9956 ± 0.0033 | 34,317 (33) | 7 |

## Significance at 100% data (McNemar exact test, paired test images)

p < 0.05 means the two models' error patterns differ beyond chance for that seed.

| hybrid | vs. | p (seed 42) | p (seed 43) | p (seed 44) |
|---|---|---|---|---|
| C->Q hybrid (CNN + quantum head) | CNN + matched classical head | 0.092 | 0.146 | 1.000 |
| Q->C hybrid (quanvolution + CNN) | Random patch + CNN (control) | 0.000 | 0.000 | 0.000 |
| Q->C hybrid (quanvolution + CNN) | Avg-pool + CNN (control) | 1.000 | 0.344 | 0.791 |
| Q->C hybrid (quanvolution + CNN) | Learned 2x2 conv + CNN (control) | 1.000 | 1.000 | 0.057 |
| C->Q hybrid (CNN + quantum head) | Classical CNN | 0.424 | 0.227 | 0.267 |
| Q->C hybrid (quanvolution + CNN) | Classical CNN | 0.453 | 0.804 | 0.581 |
