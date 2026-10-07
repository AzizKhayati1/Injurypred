# Class-imbalance strategies (validation period)

| strategy | ROC-AUC | PR-AUC | mean raw p | Brier raw | mean calibrated p | Brier calibrated | base rate |
|---|---|---|---|---|---|---|---|
| lightgbm:scale_pos_weight=none | 0.681 | 0.245 | 0.069 | 0.0707 | 0.078 | 0.0659 | 0.078 |
| lightgbm:scale_pos_weight=sqrt | 0.676 | 0.225 | 0.087 | 0.0704 | 0.078 | 0.0667 | 0.078 |
| lightgbm:scale_pos_weight=auto | 0.672 | 0.214 | 0.088 | 0.0713 | 0.078 | 0.0675 | 0.078 |
| logistic_regression:class_weight=None | 0.646 | 0.221 | 0.062 | 0.0675 | 0.078 | 0.0676 | 0.078 |
| logistic_regression:class_weight=balanced | 0.649 | 0.213 | 0.434 | 0.2101 | 0.078 | 0.0677 | 0.078 |
