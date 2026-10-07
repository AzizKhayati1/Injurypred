# Methodology

This document explains *what* is modelled, *how* workload is represented, and *why* certain
popular sports-science constructs are treated as hypotheses rather than facts.

> PlayerLoad-Risk estimates statistical availability risk from workload patterns.
> It is not a medical diagnostic system.

---

## 1. Problem framing

For every athlete *a* who is **available** at the end of day *t*, estimate

```
P( a new time-loss absence (injury or illness) begins in (t, t+7] | data dated ≤ t )
```

This is the primary target `availability_risk_7d`. Secondary targets (`injury_event_next_7d`,
`injury_event_next_14d`, `availability_days_lost`, `event_next_1d`) are built by the same code.

Why "availability" rather than "injury"?

* Practitioners plan around *who can train/play*, not around diagnoses.
* Injury definitions vary (time-loss vs medical-attention; Fuller et al., 2006). A time-loss
  definition is objective and is what a GPS + availability log can support.
* Illness absences are partly load-related (Foster, 1998) but mostly not — including them
  makes the target honest about how much of availability is *not* explained by workload.

## 2. Workload representations

All features are computed on a complete **daily calendar** per athlete (rest days = 0 load;
days of absence = 0 load) with *trailing* windows that end at *t*.

| family | examples | notes |
|---|---|---|
| Acute load (7 d) | `total_distance_7d`, `sprint_distance_7d`, `player_load_7d`, `srpe_load_7d` | external (GPS/accelerometer) and internal (session-RPE, Foster et al., 2001) load |
| Mid / chronic load (14 / 28 d) | `player_load_14d`, `player_load_28d`, `sprint_distance_28d` | "fitness" proxy |
| Coupled rolling ACWR | `acwr_total_load`, `acwr_sprint_load`, `acwr_high_speed_load` | Hulin et al., 2014; Gabbett, 2016 |
| Uncoupled ACWR | `acwr_total_load_uncoupled` | chronic window excludes the acute week (Lolli et al., 2019) |
| EWMA | `ewma_acwr_total_load`, `ewma_acute_player_load`, … | λ = 2/(N+1) (Williams et al., 2017) |
| Change & variability | `weekly_load_change`, `monotony_7d`, `strain_7d` | Foster, 1998 |
| Recovery | `rest_days_7d`, `consecutive_training_days`, `consecutive_high_load_days`, `days_since_last_match` | |
| Speed exposure | `max_speed_pct_7d`, `days_since_near_max_speed` | near-maximal speed exposure |
| History | `previous_injury_count`, `days_since_previous_injury`, `days_since_return`, `previous_90d_injury_burden` | previous injury is one of the most consistent risk factors (Hägglund et al., 2006) |
| **Athlete baseline** | `sprint_distance_vs_baseline_pct`, `…_z`, `…_baseline_weekly` | see §4 |
| Context | `scheduled_matches_next_7d` | published fixture calendar (only forward-looking input) |

## 3. The acute:chronic workload ratio — concept, limitations, and how this project treats it

**Concept.** ACWR compares recent load ("fatigue", typically 7 days) with longer-term load
("fitness", typically 28 days). Ratios around 0.8–1.3 were proposed as a "sweet spot", with
spikes > 1.5 associated with higher injury rates in several cohorts (Hulin et al., 2014;
Gabbett, 2016).

**Why it is contested.**

1. *Mathematical coupling.* In the conventional rolling ACWR the acute week is part of the
   chronic window, so numerator and denominator are not independent; this can create
   spurious correlations (Lolli et al., 2019).
2. *Ratio problems.* Ratios conflate changes in numerator and denominator, behave badly when
   chronic load is small (e.g., return from injury, early pre-season), and their statistical
   properties make "zones" fragile (Impellizzeri et al., 2020).
3. *Causal interpretation.* Observational associations do not show that *manipulating* the
   ratio changes injury risk; the theoretical role of chronic load as "protective" has been
   questioned (Impellizzeri et al., 2021).
4. *Arbitrary choices.* Window lengths, rolling vs EWMA (Williams et al., 2017), coupled vs
   uncoupled, and the load metric all change the value.
5. *Prediction ≠ association.* Even robust associations rarely translate into accurate
   individual predictions (Bahr, 2016).

**How this project handles it.**

* ACWR is **one candidate representation among many**: coupled, uncoupled and EWMA versions
  are computed for several load metrics *alongside* absolute loads, week-over-week change,
  monotony/strain, recovery and athlete-baseline deviations.
* No ACWR "zone" is hard-coded into labels, thresholds or the risk engine.
* The synthetic ground truth is deliberately **not** an ACWR formula: injuries are generated
  from a fitness–fatigue state with athlete-specific tolerance, sprint spikes *and*
  under-exposure, insufficient recovery, re-injury windows and load-independent contact
  injuries. The model must *discover* which representation is informative.
* `training/experiments/workload_representations.py` compares feature families head-to-head on
  the validation period with athlete-cluster bootstrap CIs. Result on the bundled data
  (`reports/workload_representations.md`): every workload family alone reaches ROC-AUC
  ≈ 0.60–0.62; on top of a shared context (athlete + history + fixtures) EWMA, uncoupled ACWR,
  baseline deviations and coupled ACWR all land within ≈ 0.67–0.68 with overlapping CIs.
  **No representation clearly dominates.** Descriptively, observed risk does rise with ACWR in
  this data (`reports/figures/acwr_bands.png`) — unsurprising, because the generator's
  fitness–fatigue state is itself an acute-vs-chronic construct — but once other
  representations are available, coupled ACWR adds little *incremental* information, and the
  EWMA / uncoupled variants carry the same signal without the coupling artefact. An association
  across bands is also not a validated individual decision rule (Bahr, 2016; Impellizzeri et
  al., 2020).
* The dashboard shows the commonly cited 0.8–1.3 band as *context only*, labelled "contested".

## 4. Athlete-specific baselines

A global average hides that a winger's "normal" sprint week is several times a centre-back's.
For every athlete and day we compute a **personal weekly baseline**:

* window: trailing 84 days (12 weeks) **ending 7 days before t** — the current acute week is
  excluded, so a spike cannot inflate its own baseline;
* only **available** days enter the baseline (an injury lay-off does not drag "normal" down);
* at least 28 valid days are required, otherwise the feature is missing (not imputed with a
  population value, which would leak information across athletes and time);
* variability = SD of the athlete's weekly totals over the same window (floored at 5 % of the
  baseline to avoid division by ~0).

Outputs: `<metric>_baseline_weekly`, `<metric>_vs_baseline_pct` and `<metric>_vs_baseline_z`
for player load, total distance, HSR, sprint distance, session minutes and accelerations.
Example: baseline sprint 850 m/week, current 1,350 m → `+58.8 %`.

## 5. Models

| model | role | imbalance handling |
|---|---|---|
| Logistic regression (median-impute + standardise, L2) | transparent baseline | `class_weight=balanced` |
| Random forest | non-linear baseline | `balanced_subsample` |
| LightGBM | gradient boosting | `scale_pos_weight = sqrt(n_neg/n_pos)` |
| XGBoost | gradient boosting | `scale_pos_weight = sqrt(n_neg/n_pos)` |
| Discrete-time hazard (pooled logistic on next-day onsets) | time-to-event model; P(7 d) = 1 − (1 − h)^7 | balanced |

Boosting models early-stop on **validation ROC-AUC** (calibration-invariant) with small trees
and strong regularisation — the signal is weak, and deep trees overfit athlete idiosyncrasies.

**Calibration.** Re-weighting the minority class inflates raw probabilities (balanced logistic
regression: mean raw prediction 0.43 vs a 0.078 base rate). Every model is therefore
re-calibrated with Platt scaling on the logit, fitted on the **validation period only**.
`training/experiments/imbalance_strategies.py` shows that re-weighting does not improve
discrimination here, while calibration is what makes the probabilities usable.

**Selection.** Champion = highest validation PR-AUC (Brier as tie-breaker). The test period
is used only to report performance.

**Deep learning** was not used: ~60k correlated athlete-days with ~4k events is a small-data,
high-noise problem; sequence models would add variance and opacity without evidence of gain.

## 6. Explainability

* **Local**: SHAP (`TreeExplainer` for tree models, `LinearExplainer` for linear models) on the
  model's raw output (log-odds), before the monotone calibration step. Contributions are split
  into risk-increasing and protective factors and rendered as sentences with athlete context
  (value vs 7 days ago, value vs personal baseline).
* **Global**: mean |SHAP| per feature and per feature family (`reports/evaluation.md`).
* **Language policy**: explanations describe *associations with historical risk*; phrases such
  as "will get injured" are forbidden and checked by tests.
* **Calibration context** per prediction: the observed event rate among held-out test
  athlete-days in the same predicted-risk decile.

## 7. Risk engine

Weekly absence probabilities are low (≈ 7–8 % base rate here; similar orders of magnitude are
reported in professional football, Ekstrand et al., 2011). Absolute cut points like 20/40/70 %
would label almost everyone LOW. The default **relative** mode therefore defines categories as
multiples of the validation base rate (MODERATE ≥ 1.5×, HIGH ≥ 2.5×, VERY HIGH ≥ 4×); absolute
mode is one config switch away. The resolved thresholds are stored in model metadata
(versioned with the model) and their sensitivity/precision on the test period is reported.

## 8. What-if simulation

The simulator appends a hypothetical session for *t+1* to the data known at *t*, applies the
scenario (±% total load, sprint, HSR, duration, or a recovery day), recomputes features with the
production pipeline and re-scores. It answers "how does the **model's estimate** respond?" —
not "what will happen to the athlete". A single session is ~1/7 of the acute window, and tree
models are piecewise-constant, so typical athletes move little while athletes in a loaded
state can move substantially (e.g., a recovery day: 49 % → 39 % for a VERY HIGH athlete).

## References

* Bahr R. (2016). Why screening tests to predict injury do not work—and probably never will…: a critical review. *Br J Sports Med*, 50(13), 776–780.
* Banister E.W., Calvert T.W., Savage M.V., Bach T. (1975). A systems model of training for athletic performance. *Aust J Sports Med*, 7, 57–61.
* Ekstrand J., Hägglund M., Waldén M. (2011). Injury incidence and injury patterns in professional football: the UEFA injury study. *Br J Sports Med*, 45(7), 553–558.
* Foster C. (1998). Monitoring training in athletes with reference to overtraining syndrome. *Med Sci Sports Exerc*, 30(7), 1164–1168.
* Foster C., Florhaug J.A., Franklin J., et al. (2001). A new approach to monitoring exercise training. *J Strength Cond Res*, 15(1), 109–115.
* Fuller C.W., Ekstrand J., Junge A., et al. (2006). Consensus statement on injury definitions and data collection procedures in studies of football (soccer) injuries. *Br J Sports Med*, 40(3), 193–201.
* Gabbett T.J. (2016). The training–injury prevention paradox: should athletes be training smarter and harder? *Br J Sports Med*, 50(5), 273–280.
* Hägglund M., Waldén M., Ekstrand J. (2006). Previous injury as a risk factor for injury in elite football: a prospective study over two consecutive seasons. *Br J Sports Med*, 40(9), 767–772.
* Hulin B.T., Gabbett T.J., Blanch P., et al. (2014). Spikes in acute workload are associated with increased injury risk in elite cricket fast bowlers. *Br J Sports Med*, 48(8), 708–712.
* Impellizzeri F.M., Tenan M.S., Kempton T., Novak A., Coutts A.J. (2020). Acute:chronic workload ratio: conceptual issues and fundamental pitfalls. *Int J Sports Physiol Perform*, 15(6), 907–913.
* Impellizzeri F.M., Woodcock S., Coutts A.J., et al. (2021). What role do chronic workloads play in the acute to chronic workload ratio? Time to dismiss ACWR and its underlying theory. *Sports Med*, 51(3), 581–592.
* Lolli L., Batterham A.M., Hawkins R., et al. (2019). Mathematical coupling causes spurious correlation within the conventional acute-to-chronic workload ratio calculations. *Br J Sports Med*, 53(15), 921–922.
* Lundberg S.M., Lee S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
* Rossi A., Pappalardo L., Cintia P., et al. (2018). Effective injury forecasting in soccer with GPS training data and machine learning. *PLoS ONE*, 13(7), e0201264.
* Williams S., West S., Cross M.J., Stokes K.A. (2017). Better way to determine the acute:chronic workload ratio? *Br J Sports Med*, 51(3), 209–210.
* Windt J., Gabbett T.J. (2017). How do training and competition workloads relate to injury? The workload–injury aetiology model. *Br J Sports Med*, 51(5), 428–435.
