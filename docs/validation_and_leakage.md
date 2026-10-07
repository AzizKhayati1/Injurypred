# Temporal validation & leakage prevention

Athlete-day rows are strongly autocorrelated: the 28-day load on Tuesday and on Wednesday share
27 days. A random train/test split would therefore test the model on near-copies of its training
rows, and any label window that crosses the split boundary would hand it the answer. This
document lists every leakage path we considered and the control that closes it.

## Prediction timestamp

A prediction for date **t** is made at the **end of day t**: every session dated ≤ t is known;
nothing dated > t is. The label covers **(t, t+7]**.

## Splits (`configs/config.yaml → split`)

| split | prediction dates | used for |
|---|---|---|
| train | 2024-07-01 → 2025-05-31 (season 1) | fitting model weights |
| validation | 2025-07-01 → 2025-12-31 (early season 2) | early stopping, calibration, model selection, risk thresholds |
| test | 2026-01-01 → 2026-05-31 (late season 2) | reporting only |
| production | 2026-07-01 → today (current season) | monitoring; labels mature over time |

Additional checks in `training/evaluate.py`:

* **Forward (rolling-origin) validation** – 8 monthly folds; for each fold the model is retrained
  on rows whose label window ends *before* the fold starts.
* **Athlete hold-out** – ~20 % of athletes (hash-based, deterministic) never appear in training;
  the model is evaluated on their late-season rows.
* **Cluster bootstrap** – confidence intervals resample *athletes*, not rows.

## Leakage paths and controls

| # | leakage path | control | where / test |
|---|---|---|---|
| 1 | **Future workload in features** (centred windows, `shift(-k)`, full-history normalisation) | all windows are trailing and end at t; normalisation inside the model pipeline is fitted on train only | `src/features/build.py`; `tests/test_leakage.py::test_features_invariant_to_truncating_the_future` recomputes features from data *truncated at t* and asserts bit-identical values |
| 2 | **Future data perturbation** | multiplying all post-t loads by 10 must not change any feature ≤ t | `test_features_invariant_to_perturbing_future_sessions` |
| 3 | **Label window crossing split boundaries** | *purging*: a row belongs to a split only if t + 7 is inside that split; validation/test starts after the last training label day | `src/models/splitting.py::temporal_split` (asserts at runtime); `test_split_purging_prevents_label_overlap` |
| 4 | **Partially observed labels** (window extends past the data end or the season end) | `observable_7d` flag; such rows are scored but never trained/evaluated on | `src/features/labels.py`; `test_rows_only_where_label_window_observable`, `test_observability_respects_season_end` |
| 5 | **Target leakage via status** (predicting on the injury day, or for already-injured athletes) | only *available* athletes are eligible; the onset day itself is unavailable | `test_labels_depend_on_future_only` |
| 6 | **Return date of an ongoing absence** (known only after the fact) | features use the absence flag for days ≤ t only; `Dataset.truncate` censors return dates | `src/data/repository.py::Dataset.truncate` |
| 7 | **Athlete baseline contaminated by the current spike** | baseline window ends 7 days before t | `tests/test_features.py::test_athlete_baseline_deviation` |
| 8 | **Population statistics leaking across time/athletes** (e.g. imputing with whole-dataset means) | missing baselines stay missing; imputers live inside the sklearn pipeline (fit on train) | `src/models/estimators.py` |
| 9 | **Latent ground truth** (synthetic traits) | never loaded by the feature pipeline; stored in a separate, labelled file | `test_schema_matches_catalogue_and_has_no_latent_traits` |
| 10 | **Selection on test** | champion chosen on validation PR-AUC; test metrics are computed but never used for decisions | `training/train.py::_select` |
| 11 | **Calibration on test** | calibrator fitted on validation only | `src/models/estimators.py::fit_model` |
| 12 | **What-if simulator peeking at the future** | simulation runs on `Dataset.truncate(t)` | `tests/test_simulation.py::test_simulation_ignores_future_data` |
| 13 | **Fixture calendar** (forward-looking) | allowed deliberately: fixtures are published months in advance and contain no athlete-level future information; can be disabled with `features.use_fixture_calendar: false` | documented here |

## Burn-in

The first 28 days of an athlete's tracked history are not eligible for prediction: chronic
(28-day) windows and baselines would be incomplete. A side effect, surfaced by the monitoring
report: the first season has no pre-season rows in training, while later seasons do.
