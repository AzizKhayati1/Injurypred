"""Evidence-based prevention recommendations (decision support, not prescriptions).

Transparent rules map an athlete's *current* workload features and *recorded*
injury history to prevention measures with published evidence. Every
recommendation states why it was triggered (the athlete's own numbers), what
to consider doing, and the evidence behind it. Final decisions belong to
qualified medical and performance staff who know the athlete.

The rules are deliberately simple and auditable: the ML model estimates *how
much* risk is elevated; these rules suggest *what kind* of preventive work is
commonly supported for the patterns observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

DISCLAIMER = ("Decision support only. Recommendations summarise published prevention evidence matched to workload and "
              "injury-history patterns; they are not a medical prescription. Discuss with the medical/performance team.")

REFERENCES = {
    "van_dyk_2019": "van Dyk N., Behan F.P., Whiteley R. (2019). Including the Nordic hamstring exercise in injury "
                    "prevention programmes halves the rate of hamstring injuries: a systematic review and meta-analysis "
                    "of 8459 athletes. Br J Sports Med, 53(21), 1362–1370.",
    "van_der_horst_2015": "van der Horst N., Smits D.W., Petersen J., Goedhart E.A., Backx F.J. (2015). The preventive "
                          "effect of the Nordic hamstring exercise on hamstring injuries in amateur soccer players: a "
                          "randomized controlled trial. Am J Sports Med, 43(6), 1316–1323.",
    "duhig_2016": "Duhig S., Shield A.J., Opar D., et al. (2016). Effect of high-speed running on hamstring strain "
                  "injury risk. Br J Sports Med, 50(24), 1536–1540.",
    "haroy_2019": "Harøy J., Clarsen B., Wiger E.G., et al. (2019). The Adductor Strengthening Programme prevents groin "
                  "problems among male football players: a cluster-randomised controlled trial. Br J Sports Med, "
                  "53(3), 150–157.",
    "hupperets_2009": "Hupperets M.D.W., Verhagen E.A.L.M., van Mechelen W. (2009). Effect of unsupervised home based "
                      "proprioceptive training on recurrences of ankle sprain: randomised controlled trial. BMJ, 339, "
                      "b2684.",
    "soligard_2008": "Soligard T., Myklebust G., Steffen K., et al. (2008). Comprehensive warm-up programme to prevent "
                     "injuries in young female footballers: cluster randomised controlled trial. BMJ, 337, a2469.",
    "ardern_2016": "Ardern C.L., Glasgow P., Schneiders A., et al. (2016). 2016 Consensus statement on return to "
                   "sport from the First World Congress in Sports Physical Therapy, Bern. Br J Sports Med, 50(14), "
                   "853–864.",
    "malone_2017": "Malone S., Roe M., Doran D.A., Gabbett T.J., Collins K. (2017). High chronic training loads and "
                   "exposure to bouts of maximal velocity running reduce injury risk in elite Gaelic football. J Sci "
                   "Med Sport, 20(3), 250–254.",
    "gabbett_2016": "Gabbett T.J. (2016). The training–injury prevention paradox: should athletes be training smarter "
                    "and harder? Br J Sports Med, 50(5), 273–280.",
    "impellizzeri_2020": "Impellizzeri F.M., Tenan M.S., Kempton T., Novak A., Coutts A.J. (2020). Acute:chronic "
                         "workload ratio: conceptual issues and fundamental pitfalls. Int J Sports Physiol Perform, "
                         "15(6), 907–913.",
    "saw_2016": "Saw A.E., Main L.C., Gastin P.B. (2016). Monitoring the athlete training response: subjective "
                "self-reported measures trump commonly used objective measures: a systematic review. Br J Sports Med, "
                "50(5), 281–291.",
    "patricios_2023": "Patricios J.S., Schneider K.J., Dvorak J., et al. (2023). Consensus statement on concussion in "
                      "sport: the 6th International Conference on Concussion in Sport–Amsterdam, October 2022. Br J "
                      "Sports Med, 57(11), 695–711.",
}


@dataclass
class Recommendation:
    id: str
    title: str
    category: str              # Load management | Strength | Recovery | Return to play | Speed exposure | Warm-up
    priority: str              # High | Medium | Low
    target: str                # injury type / outcome addressed
    why: str                   # athlete-specific trigger
    actions: list[str]
    evidence: str              # one-line evidence summary
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["references"] = [{"key": k, "citation": REFERENCES[k]} for k in self.references]
        return d


def _v(row: pd.Series | dict, key: str, default=None):
    v = row.get(key, default) if row is not None else default
    return default if v is None or (isinstance(v, float) and pd.isna(v)) else v


def recommend(row: pd.Series | dict | None, history: dict, position: str, risk_category: str | None = None,
              as_of=None) -> list[dict]:
    """Return prioritised recommendations for one athlete.

    ``row`` – current feature row (``None`` if the athlete is unavailable),
    ``history`` – output of :func:`src.clinical.injuries.player_injury_summary`.
    """
    recs: list[Recommendation] = []
    elevated = risk_category in ("HIGH", "VERY_HIGH")
    hi = "High" if elevated else "Medium"
    inj = [r for r in history.get("records", []) if r.get("absence_type") == "injury"]
    codes = {r.get("diagnosis_code") for r in inj}
    cutoff = (pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.max) - pd.Timedelta(days=365)
    recent_codes = {r.get("diagnosis_code") for r in inj if as_of is None or pd.Timestamp(r["onset_date"]) > cutoff}

    def hist_priority(code_set: set) -> str:
        return "High" if recent_codes & code_set else "Medium"
    recent = [r for r in inj if r.get("status") == "Resolved"]
    current = history.get("current")

    if current is not None:
        recs.append(Recommendation(
            "rtp", "Criteria-based return-to-play pathway", "Return to play", "High",
            current.get("diagnosis") or "current absence",
            f"Currently unavailable: {current.get('diagnosis', 'absence')} since {current['onset_date']}.",
            ["Progress through rehabilitation stages using objective criteria, not time alone",
             "Rebuild chronic running load and high-speed exposure before full team training",
             "Shared decision on return between medical staff, coach and athlete"],
            "Consensus recommends a continuum from return to participation → sport → performance, guided by criteria.",
            ["ardern_2016"] + (["patricios_2023"] if current.get("diagnosis_code") == "CON" else [])))
        return [r.to_dict() for r in recs]

    if row is not None:
        sprint_dev = _v(row, "sprint_distance_vs_baseline_pct")
        acwr_sprint = _v(row, "acwr_sprint_load")
        if (sprint_dev is not None and sprint_dev >= 50) or (acwr_sprint is not None and acwr_sprint >= 1.5):
            recs.append(Recommendation(
                "hsr_spike", "Smooth the high-speed running progression", "Load management", hi, "Hamstring injury",
                f"Sprint distance is {sprint_dev:+.0f}% vs the athlete's normal week"
                + (f" (sprint ACWR {acwr_sprint:.2f})" if acwr_sprint is not None else "") + ".",
                ["Spread sprint volume across the week instead of concentrating it in one session",
                 "Cap additional top-up sprint work until exposure is back near the personal baseline",
                 "Monitor hamstring complaints and session RPE closely this week"],
                "Large or rapid increases in high-speed running have been associated with hamstring strain risk; "
                "ratio-based thresholds themselves are debated.", ["duhig_2016", "gabbett_2016", "impellizzeri_2020"]))
        wl = _v(row, "weekly_load_change")
        acwr = _v(row, "ewma_acwr_total_load")
        if (wl is not None and wl >= 0.30) or (acwr is not None and acwr >= 1.5):
            recs.append(Recommendation(
                "load_spike", "Avoid abrupt week-to-week load increases", "Load management", hi, "All injuries",
                (f"Player load changed {wl * 100:+.0f}% week-over-week" if wl is not None else "Recent load spike")
                + (f"; EWMA acute:chronic ratio {acwr:.2f}." if acwr is not None else "."),
                ["Plan the next microcycle closer to the athlete's recent chronic load",
                 "Prefer gradual progressions over large single-week jumps",
                 "Re-check the estimate after the next sessions"],
                "Spikes in load relative to what an athlete is prepared for have been associated with injury; "
                "the magnitude of safe change is individual and the ACWR's interpretation is contested.",
                ["gabbett_2016", "impellizzeri_2020"]))
        rest = _v(row, "rest_days_7d")
        consec = _v(row, "consecutive_training_days")
        if (rest is not None and rest == 0) or (consec is not None and consec >= 6):
            recs.append(Recommendation(
                "recovery", "Schedule a recovery day", "Recovery", hi if elevated else "Medium", "Fatigue-related injury",
                f"{int(consec or 0)} consecutive days with a session; {int(rest or 0)} rest day(s) in the last 7 days.",
                ["Plan a full rest or low-intensity recovery day in the next 48 h",
                 "Check sleep and subjective wellness before high-intensity work"],
                "Insufficient recovery between demanding sessions is a modifiable component of the training load–"
                "injury relationship.", ["gabbett_2016", "saw_2016"]))
        rpe_trend = _v(row, "rpe_trend")
        if rpe_trend is not None and rpe_trend >= 1.0:
            recs.append(Recommendation(
                "wellness", "Check subjective fatigue and wellness", "Recovery", "Medium", "Fatigue / illness",
                f"Session RPE is {rpe_trend:+.1f} points above the athlete's 4-week average for similar work.",
                ["Use a short wellness questionnaire (sleep, soreness, mood, stress)",
                 "Consider reducing intensity if wellness scores are also poor"],
                "Subjective self-reported measures respond to training load more sensitively than many objective "
                "markers.", ["saw_2016"]))
        dsr = _v(row, "days_since_return")
        if dsr is not None and dsr < 28:
            recs.append(Recommendation(
                "post_return", "Protect the first weeks after return", "Return to play", hi,
                "Re-injury",
                f"Returned from an absence {int(dsr)} day(s) ago.",
                ["Keep sprint and high-speed running exposure progressing towards (not beyond) the personal baseline",
                 "Continue the rehabilitation strength programme alongside team training",
                 "Limit minutes in congested fixture periods where possible"],
                "Re-injury risk is highest soon after return; criteria-based progression is recommended.",
                ["ardern_2016"]))
        dmax = _v(row, "days_since_near_max_speed")
        if dmax is not None and dmax >= 14 and position != "GK":
            recs.append(Recommendation(
                "max_speed", "Reintroduce controlled near-maximal sprinting", "Speed exposure", "Medium",
                "Hamstring injury / under-preparation",
                f"No running at ≥ 90 % of the athlete's top speed for {int(dmax)} days.",
                ["Include short, well-recovered sprints at 90–95 % of top speed in the next sessions",
                 "Build exposure gradually before matches that demand repeated sprinting"],
                "Regular exposure to near-maximal velocity was associated with lower injury risk in field-sport "
                "athletes.", ["malone_2017"]))

    if "HAM" in codes or position in ("W", "FB", "ST"):
        reason = ("Previous hamstring injury on record." if "HAM" in codes
                  else "Sprint-heavy position with high hamstring exposure.")
        recs.append(Recommendation(
            "nordic", "Nordic hamstring exercise programme", "Strength", hist_priority({"HAM"}) if "HAM" in codes
            else "Medium",
            "Hamstring injury", reason,
            ["Progressive Nordic hamstring exercise programme (in-season maintenance 1–2 sessions/week)",
             "Combine with hip-extension strength and sprint-mechanics work"],
            "Programmes including the Nordic hamstring exercise roughly halved hamstring injury rates in "
            "meta-analysis.", ["van_dyk_2019", "van_der_horst_2015"]))
    if codes & {"ADD", "GRO"}:
        recs.append(Recommendation(
            "copenhagen", "Copenhagen adduction programme", "Strength", hist_priority({"ADD", "GRO"}), "Groin injury",
            "Previous adductor / groin problem on record.",
            ["Progressive Copenhagen adduction exercise, 2–3 times per week in pre-season, 1 per week in-season"],
            "The Adductor Strengthening Programme reduced groin problems in a cluster-randomised trial.",
            ["haroy_2019"]))
    if codes & {"ANK", "SYN"}:
        recs.append(Recommendation(
            "balance", "Balance & proprioceptive training", "Strength", hist_priority({"ANK", "SYN"}),
            "Ankle sprain recurrence",
            "Previous ankle sprain on record.",
            ["8–12 weeks of progressive balance/proprioceptive exercises after an ankle sprain",
             "Consider external support (brace/tape) during return phase"],
            "Proprioceptive training reduced ankle sprain recurrences in a randomised controlled trial.",
            ["hupperets_2009"]))
    if "CON" in codes:
        recs.append(Recommendation(
            "concussion", "Concussion history: follow graduated protocols", "Return to play", "Medium",
            "Concussion", "Previous concussion on record.",
            ["Ensure baseline and follow-up assessments are up to date",
             "Any suspected recurrence: remove from play and follow the graduated return-to-sport strategy"],
            "International consensus guidance on recognition and graduated return after sport-related concussion.",
            ["patricios_2023"]))
    if any(r.get("recurrence") for r in recent):
        recs.append(Recommendation(
            "recurrence", "Address recurrent injury risk factors", "Return to play",
            "High" if any(r.get("recurrence") and pd.Timestamp(r["onset_date"]) > cutoff for r in recent) else "Medium",
            "Re-injury",
            "Recurrent injury of the same type on record.",
            ["Review rehabilitation completeness and return criteria used previously",
             "Extend targeted strength work beyond the return date"],
            "Recurrences often follow incomplete rehabilitation; criteria-based progression is recommended.",
            ["ardern_2016"]))
    recs.append(Recommendation(
        "warmup", "Structured injury-prevention warm-up", "Warm-up", "Low", "Lower-limb injuries",
        "Recommended for all players.",
        ["Use a structured neuromuscular warm-up (e.g. FIFA 11+) at least twice a week"],
        "The FIFA 11+ programme reduced injuries in a cluster-randomised trial in youth football.",
        ["soligard_2008"]))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return [r.to_dict() for r in sorted(recs, key=lambda r: order[r.priority])]
