# Model card – `playerload-risk-availability-7d` v1.0.0

| | |
|---|---|
| **Task** | Probability that an *available* athlete starts a time-loss absence (injury or illness) in the 7 days after the prediction date |
| **Intended use** | Decision *support* for performance/medical staff: prioritising conversations, load reviews and monitoring; portfolio/teaching artefact |
| **Not intended for** | Diagnosis, return-to-play clearance, selection or contract decisions, any automated decision about an individual |
| **Training data** | Synthetic GPS-style data (see `docs/data.md`) – season 2024-25 |
| **Model** | Champion selected on validation PR-AUC among logistic regression, random forest, LightGBM, XGBoost and a discrete-time hazard model; Platt-calibrated on the validation period |
| **Inputs** | 76 leakage-free workload, recovery, history, athlete-baseline and context features (`GET /features`) |
| **Outputs** | Calibrated probability, risk category (relative to base rate), percentile, SHAP risk/protective factors, baseline comparison, calibration context |

## Performance (bundled synthetic data)

See `reports/evaluation.md` for the full report. Champion (XGBoost), test period Jan–May 2026:

* ROC-AUC **0.649** (athlete cluster-bootstrap 95 % CI 0.58–0.70), PR-AUC **0.196** vs base rate 0.066
* Brier 0.058 (Brier skill +0.06), expected calibration error 0.013
* Top 10 % of athlete-days by predicted risk capture 33 % of events (lift 3.3×)
* Forward validation (8 monthly folds): ROC-AUC 0.638 ± 0.038
* Unseen athletes (hold-out): ROC-AUC 0.688

These numbers are in the range reported for injury forecasting from workload data and show
that **individual prediction is weak**: most flagged athlete-days do not result in an absence,
and most absences are not preceded by a high-risk flag.

## Known limitations

* Synthetic data: performance on real squads is unknown until validated on real data.
* The model partially recovers hidden injury susceptibility (Spearman ρ ≈ 0.29 between an
  athlete's mean predicted risk and the hidden trait) but **not** individual workload tolerance
  (ρ ≈ 0.03) — personal tolerance is hard to learn from a handful of events per athlete.
* Associations, not causes: the what-if simulator shows model sensitivity, not treatment effects.
* Contact injuries and illnesses are largely unpredictable from load and cap achievable accuracy.
* Monitoring flags drift in the current season (intentional HSR/sprint emphasis in two teams,
  plus pre-season rows absent from season-1 training due to burn-in).

## Ethical considerations

Workload and injury data are health data. Use requires informed consent, purpose limitation,
access control, and human review. Risk scores must not be used punitively (selection,
contracts) or shown to athletes without context. Categories are relative to a base rate and
communicated with uncertainty; explanations avoid deterministic language by design (tested).
