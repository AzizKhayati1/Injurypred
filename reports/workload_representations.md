# Workload representation experiment (validation period)

Same chronological split for every run; calibrated probabilities; 95% CI = athlete cluster bootstrap (LightGBM ROC-AUC).

| family | view | n features | LR ROC-AUC | LGBM ROC-AUC (95% CI) | LR PR-AUC | LGBM PR-AUC |
|---|---|---|---|---|---|---|
| absolute_rolling_loads | alone | 20 | 0.629 | 0.614 (0.563–0.674) | 0.129 | 0.126 |
| acwr_rolling | alone | 4 | 0.615 | 0.617 (0.572–0.658) | 0.122 | 0.125 |
| acwr_uncoupled | alone | 1 | 0.606 | 0.621 (0.577–0.664) | 0.133 | 0.128 |
| athlete_baseline_deviation | alone | 15 | 0.533 | 0.615 (0.566–0.670) | 0.115 | 0.159 |
| ewma | alone | 5 | 0.621 | 0.623 (0.580–0.667) | 0.136 | 0.141 |
| week_change_monotony | alone | 6 | 0.563 | 0.603 (0.570–0.640) | 0.133 | 0.111 |
| absolute_rolling_loads | with_context | 36 | 0.662 | 0.667 (0.608–0.714) | 0.196 | 0.215 |
| acwr_rolling | with_context | 20 | 0.657 | 0.677 (0.620–0.730) | 0.202 | 0.224 |
| acwr_uncoupled | with_context | 17 | 0.650 | 0.682 (0.630–0.727) | 0.206 | 0.205 |
| all_features | with_context | 76 | 0.649 | 0.676 (0.623–0.725) | 0.213 | 0.225 |
| athlete_baseline_deviation | with_context | 31 | 0.666 | 0.680 (0.620–0.731) | 0.219 | 0.207 |
| context_only | with_context | 16 | 0.643 | 0.667 (0.610–0.715) | 0.186 | 0.210 |
| ewma | with_context | 21 | 0.666 | 0.678 (0.621–0.734) | 0.212 | 0.234 |
| week_change_monotony | with_context | 22 | 0.658 | 0.666 (0.608–0.717) | 0.197 | 0.223 |

Read with the CIs: differences smaller than the interval width are not evidence that one representation is better.
