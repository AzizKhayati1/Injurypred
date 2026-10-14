"""Synthetic GPS-style workload & availability data generator.

!!! ALL DATA PRODUCED HERE IS SYNTHETIC. It does not describe real athletes. !!!

Design goals
------------
* **Temporal realism** – team schedules follow football micro-cycles (MD-4 … MD+1),
  pre-season ramps, congested fixture weeks, winter breaks and an off-season.
* **Athlete heterogeneity** – position-specific locomotor profiles, fitness,
  maximal speed, selection status (starter vs. squad player), age.
* **Latent ground truth** – each athlete has *hidden* traits (workload
  tolerance, injury susceptibility, recovery capacity). They drive a daily
  injury hazard but are **never** exposed to the model; they are saved to a
  separate, clearly-labelled file only so we can audit what the model recovers.
* **Not ACWR by construction** – the hazard is driven by a fitness–fatigue
  (Banister-style) state with athlete-specific tolerance, sprint-exposure
  spikes *and* under-exposure, insufficient recovery, re-injury windows and a
  load-independent contact-injury component. The model therefore has to
  *discover* which workload representation is informative instead of having a
  textbook formula baked into the labels.

The hazard parameters were chosen so that incidence is in the range reported
for professional football (≈3–4 /1000 h training, ≈20–30 /1000 h match play;
Ekstrand et al., 2011) and roughly 2 time-loss injuries + ~0.8 illnesses per
player-season.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

DATA_ORIGIN = "synthetic"

# Per-90-minute match demands by position (order of magnitude from published
# match-analysis literature; exact values are illustrative).
POSITION_PROFILES: dict[str, dict[str, float]] = {
    "GK": dict(n=3, dist=5600, hsr=150, sprint=20, acc=30, dec=32, vmax=27.5),
    "CB": dict(n=5, dist=9900, hsr=600, sprint=170, acc=45, dec=55, vmax=31.5),
    "FB": dict(n=4, dist=10600, hsr=950, sprint=320, acc=60, dec=68, vmax=33.0),
    "CM": dict(n=6, dist=11400, hsr=850, sprint=190, acc=58, dec=64, vmax=31.0),
    "W": dict(n=4, dist=10900, hsr=1050, sprint=360, acc=66, dec=70, vmax=33.8),
    "ST": dict(n=3, dist=10300, hsr=900, sprint=320, acc=62, dec=66, vmax=33.3),
}

# Session templates: duration (min), intensity per minute as a fraction of the
# athlete's match intensity, base RPE (CR-10) and fraction of max speed reached.
SESSION_TEMPLATES: dict[str, dict[str, float]] = {
    "match": dict(dur=90, dist=1.00, hsr=1.00, sprint=1.00, acc=1.00, rpe=8.0, speed=0.96),
    "match_top_up": dict(dur=25, dist=0.70, hsr=0.80, sprint=0.70, acc=0.90, rpe=6.0, speed=0.88),
    "MD+1_recovery": dict(dur=40, dist=0.45, hsr=0.05, sprint=0.00, acc=0.35, rpe=2.5, speed=0.60),
    "MD+1_compensatory": dict(dur=60, dist=0.70, hsr=0.75, sprint=0.70, acc=0.90, rpe=6.5, speed=0.90),
    "MD-4_strength": dict(dur=75, dist=0.55, hsr=0.30, sprint=0.20, acc=1.25, rpe=6.0, speed=0.80),
    "MD-3_endurance": dict(dur=85, dist=0.62, hsr=0.55, sprint=0.45, acc=0.95, rpe=7.0, speed=0.88),
    "MD-2_tactical": dict(dur=65, dist=0.52, hsr=0.25, sprint=0.20, acc=0.75, rpe=4.5, speed=0.80),
    "MD-1_activation": dict(dur=50, dist=0.45, hsr=0.20, sprint=0.35, acc=0.60, rpe=3.0, speed=0.93),
    "preseason_conditioning": dict(dur=95, dist=0.68, hsr=0.65, sprint=0.45, acc=1.00, rpe=7.5, speed=0.88),
    "preseason_technical": dict(dur=70, dist=0.52, hsr=0.25, sprint=0.15, acc=0.85, rpe=5.0, speed=0.78),
    "rehab_modified": dict(dur=40, dist=0.35, hsr=0.10, sprint=0.02, acc=0.35, rpe=3.5, speed=0.70),
    "individual_program": dict(dur=45, dist=0.45, hsr=0.20, sprint=0.10, acc=0.50, rpe=4.0, speed=0.80),
}

# Ground-truth hazard constants (per hour of exposure).
H_TRAIN_PER_HOUR = 2.6 / 1000
H_MATCH_PER_HOUR = 21.0 / 1000
MATCH_CONTACT_SHARE = 0.35          # load-independent share of match injuries
H_NON_SPORT_PER_DAY = 0.00015
ILLNESS_PER_DAY = 0.0024


@dataclass
class AthleteTraits:
    athlete_id: str
    team_id: str
    position: str
    date_of_birth: str
    prior_injury_count: int
    # ---- latent (never given to the model) ----
    fitness: float
    vmax: float
    workload_tolerance: float
    injury_susceptibility: float
    recovery_capacity: float
    selection_propensity: float


@dataclass
class TeamTraits:
    team_id: str
    intensity: float
    midweek_prob: float
    spike_prob: float


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------
def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_team_schedule(team: TeamTraits, seasons: list[dict], data_end: date, winter_break_days: int,
                        rng: np.random.Generator) -> tuple[dict[date, dict], list[dict]]:
    """Return {date: plan} for one team and the fixture list.

    plan = {"season", "phase", "type", "is_match", "match_type", "spike"}
    """
    plan: dict[date, dict] = {}
    fixtures: list[dict] = []
    for season in seasons:
        s_start = date.fromisoformat(season["start"])
        s_league = date.fromisoformat(season["league_start"])
        s_end = date.fromisoformat(season["end"])
        if s_start > data_end:
            continue
        # winter break centred on 24 Dec
        wb_start = date(s_start.year, 12, 22)
        wb_end = wb_start + timedelta(days=winter_break_days - 1)

        # --- match days ---------------------------------------------------
        match_days: dict[date, str] = {}
        d = s_start + timedelta(days=(5 - s_start.weekday()) % 7)  # first Saturday
        while d <= s_end - timedelta(days=6):
            if d < s_league:
                if d >= s_start + timedelta(days=13):
                    match_days[d] = "friendly"
            elif not (wb_start <= d <= wb_end + timedelta(days=2)):
                match_days[d] = "league"
                wed = d + timedelta(days=4)
                if (wed <= s_end - timedelta(days=10) and not (wb_start - timedelta(days=3) <= wed <= wb_end + timedelta(days=3))
                        and wed > s_league + timedelta(days=14) and rng.random() < team.midweek_prob):
                    match_days[wed] = "cup"
            d += timedelta(days=7)
        match_list = sorted(match_days)
        last_match = match_list[-1]

        # --- intensification ("spike") weeks ----------------------------
        spike_weeks = set()
        for wk_start in _daterange(s_league, s_end):
            if wk_start.weekday() == 0 and rng.random() < team.spike_prob:
                spike_weeks.add(wk_start)

        # The fixture calendar is published for the whole season (also beyond data_end).
        for md in match_list:
            fixtures.append({"team_id": team.team_id, "date": md.isoformat(), "season": season["name"],
                             "match_type": match_days[md]})

        for d in _daterange(s_start, min(s_end, data_end)):
            entry = {"season": season["name"], "spike": 1.0, "is_match": False, "match_type": None}
            if d in match_days:
                entry.update(phase="preseason" if d < s_league else "inseason", type="match",
                             is_match=True, match_type=match_days[d])
                plan[d] = entry
                continue
            prev = [m for m in match_list if m < d]
            nxt = [m for m in match_list if m > d]
            since = (d - prev[-1]).days if prev else 99
            to_next = (nxt[0] - d).days if nxt else 99

            if d < s_league:
                entry["phase"] = "preseason"
                days_in = (d - s_start).days
                if d.weekday() == 6 or since == 2 and days_in > 14:
                    entry["type"] = "rest"
                elif since == 1:
                    entry["type"] = "MD+1_recovery"
                elif to_next == 1:
                    entry["type"] = "MD-1_activation"
                else:
                    # classic early pre-season ramp: conditioning-heavy weeks
                    entry["type"] = "preseason_conditioning" if d.weekday() in (0, 2, 3) else "preseason_technical"
                    entry["spike"] = 1.0 + 0.10 * (days_in < 14)
            elif wb_start <= d <= wb_end:
                entry["phase"] = "winter_break"
                entry["type"] = "individual_program" if d.weekday() in (1, 3, 5) else "rest"
            elif d > last_match:
                entry["phase"] = "season_end"
                entry["type"] = "MD+1_recovery" if since == 1 else ("individual_program" if since in (3, 5) else "rest")
            else:
                entry["phase"] = "inseason"
                if since == 1:
                    entry["type"] = "MD+1_recovery"
                elif to_next == 1:
                    entry["type"] = "MD-1_activation"
                elif since == 2 and to_next >= 3:
                    entry["type"] = "rest"
                elif to_next == 2:
                    entry["type"] = "MD-2_tactical"
                elif to_next == 3:
                    entry["type"] = "MD-3_endurance"
                elif to_next >= 4 and d.weekday() != 6:
                    entry["type"] = "MD-4_strength"
                else:
                    entry["type"] = "rest"
                monday = d - timedelta(days=d.weekday())
                if monday in spike_weeks and entry["type"] in ("MD-4_strength", "MD-3_endurance"):
                    entry["spike"] = float(rng.uniform(1.30, 1.55))
            plan[d] = entry
    return plan, fixtures


# -----------------------------------------------------------------------------
# Athletes
# -----------------------------------------------------------------------------
def make_athletes(n_teams: int, per_team: int, rng: np.random.Generator, ref_date: date
                  ) -> tuple[list[AthleteTraits], list[TeamTraits]]:
    teams = [TeamTraits(team_id=f"T{t + 1}", intensity=float(rng.uniform(0.93, 1.07)),
                        midweek_prob=float(rng.uniform(0.15, 0.32)), spike_prob=0.12)
             for t in range(n_teams)]
    base_counts = {p: int(v["n"]) for p, v in POSITION_PROFILES.items()}
    athletes: list[AthleteTraits] = []
    idx = 101
    for team in teams:
        counts = dict(base_counts)
        # adjust squad size to per_team by adding/removing outfield players
        diff = per_team - sum(counts.values())
        outfield = ["CM", "CB", "FB", "W", "ST"]
        i = 0
        while diff != 0:
            p = outfield[i % len(outfield)]
            if diff > 0:
                counts[p] += 1
                diff -= 1
            elif counts[p] > 1:
                counts[p] -= 1
                diff += 1
            i += 1
        for pos, n in counts.items():
            for depth in range(n):
                prof = POSITION_PROFILES[pos]
                age = float(np.clip(rng.normal(25.5, 4.0), 18, 35))
                dob = ref_date - timedelta(days=int(age * 365.25))
                if pos == "GK":
                    sel = [0.92, 0.08, 0.03][min(depth, 2)]
                else:
                    starters = {"CB": 2, "FB": 2, "CM": 3, "W": 2, "ST": 1}.get(pos, 1)
                    if depth < starters:
                        sel = rng.uniform(0.78, 0.95)
                    elif depth < starters + 2:
                        sel = rng.uniform(0.30, 0.60)
                    else:
                        sel = rng.uniform(0.08, 0.25)
                susc = float(np.exp(rng.normal(-0.10, 0.45)))
                prior = int(rng.poisson(1.1 * susc * max(age - 17, 1) / 8))
                athletes.append(AthleteTraits(
                    athlete_id=f"A{idx}", team_id=team.team_id, position=pos, date_of_birth=dob.isoformat(),
                    prior_injury_count=prior,
                    fitness=float(np.clip(rng.normal(1.0, 0.05), 0.88, 1.12)),
                    vmax=float(prof["vmax"] + rng.normal(0, 0.9) - 0.08 * max(0.0, age - 28)),
                    workload_tolerance=float(np.clip(rng.normal(1.30, 0.12), 1.0, 1.65)),
                    injury_susceptibility=susc,
                    recovery_capacity=float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3)),
                    selection_propensity=float(sel),
                ))
                idx += 1
    return athletes, teams


# -----------------------------------------------------------------------------
# Simulation of one athlete
# -----------------------------------------------------------------------------
def _session_metrics(ath: AthleteTraits, stype: str, duration: float, mult: dict[str, float],
                     rng: np.random.Generator) -> dict[str, float]:
    prof = POSITION_PROFILES[ath.position]
    tpl = SESSION_TEMPLATES[stype]
    m = mult.get("all", 1.0)
    noise = lambda sd: float(np.exp(rng.normal(0, sd)))  # noqa: E731
    dist = prof["dist"] / 90 * ath.fitness * tpl["dist"] * duration * m * noise(0.08)
    hsr = prof["hsr"] / 90 * tpl["hsr"] * duration * m * mult.get("hsr", 1.0) * noise(0.18)
    sprint = prof["sprint"] / 90 * tpl["sprint"] * duration * m * mult.get("sprint", 1.0) * noise(0.30)
    if tpl["sprint"] == 0:
        sprint = 0.0
    acc = prof["acc"] / 90 * tpl["acc"] * duration * m * noise(0.15)
    dec = prof["dec"] / 90 * tpl["acc"] * duration * m * noise(0.15)
    sprint_count = int(round(sprint / rng.uniform(14, 20))) if sprint > 0 else 0
    if sprint > 0 and sprint_count == 0:
        sprint_count = 1
    speed_frac = tpl["speed"] * float(np.clip(rng.normal(1.0, 0.025), 0.9, 1.04))
    if sprint > 0:
        speed_frac = max(speed_frac, 25.3 / ath.vmax)  # had to exceed sprint threshold
    max_speed = min(ath.vmax * speed_frac, ath.vmax * 1.015)
    if sprint == 0:
        max_speed = min(max_speed, 25.1)
    player_load = (dist / 1000 * 88 + acc * 1.5 + dec * 1.2) * noise(0.05)
    return dict(total_distance=dist, high_speed_running_distance=min(hsr, dist * 0.25),
                sprint_distance=min(sprint, hsr) if hsr > 0 else 0.0, sprint_count=sprint_count,
                accelerations=int(round(acc)), decelerations=int(round(dec)), max_speed=max_speed,
                player_load=player_load)


def simulate_athlete(ath: AthleteTraits, team: TeamTraits, plan: dict[date, dict], start: date, end: date,
                     drift: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    sessions: list[dict] = []
    absences: list[dict] = []

    # latent state (normalised so that one full match ≈ 1.0 unit of internal load)
    prof = POSITION_PROFILES[ath.position]
    match_pl = prof["dist"] / 1000 * 88 + prof["acc"] * 1.5 + prof["dec"] * 1.2
    tau_f = 42.0
    tau_a = float(np.clip(7.0 / ath.recovery_capacity, 5.0, 10.0))
    fitness, fatigue = 0.30, 0.15
    s_acute, s_chronic = 0.05, 0.20
    sprint_norm = 0.22   # typical chronic sprint exposure in matches/units
    fitness_norm = 0.55

    unavailable_until: date | None = None
    rtp_days_left, rtp_total = 0, 0
    last_return: date | None = None
    consecutive = 0
    played_yesterday = 0.0
    dob = date.fromisoformat(ath.date_of_birth)
    injury_n = 0
    last_injury_onset: date | None = None

    for d in _daterange(start, end):
        entry = plan.get(d)
        age = (d - dob).days / 365.25
        prev_onset, prev_n = last_injury_onset, injury_n  # history known *before* today
        u = 0.0
        sprint_u = 0.0
        session_row = None

        if unavailable_until is not None and d < unavailable_until:
            # absent (injured / ill): no tracked session; latent state decays
            pass
        else:
            if unavailable_until is not None and d >= unavailable_until:
                unavailable_until = None
                last_return = d
            if entry is None:
                # off-season: untracked individual activity, keeps a little fitness
                u = 0.10 if rng.random() < 0.4 else 0.0
            else:
                stype = entry["type"]
                duration = None
                minutes_played = 0.0
                if rtp_days_left > 0 and stype != "rest":
                    stype = "rehab_modified"
                    progress = 1.0 - rtp_days_left / max(rtp_total, 1)
                    duration = SESSION_TEMPLATES[stype]["dur"] * (0.8 + 0.6 * progress)
                    rtp_days_left -= 1
                elif rtp_days_left > 0:
                    rtp_days_left -= 1
                if stype == "match":
                    recently_back = last_return is not None and (d - last_return).days < 14
                    p_start = ath.selection_propensity * (0.5 if recently_back else 1.0)
                    if entry["match_type"] == "friendly":
                        minutes_played = float(rng.choice([45, 60, 70])) if rng.random() < 0.8 else 0.0
                    elif rng.random() < p_start:
                        minutes_played = 90.0 if rng.random() < 0.75 else float(rng.integers(60, 86))
                    elif rng.random() < 0.40:
                        minutes_played = float(rng.integers(8, 31))
                    if minutes_played > 0:
                        duration = minutes_played
                    else:
                        stype, duration = "match_top_up", None
                elif stype == "MD+1_recovery" and played_yesterday < 45:
                    stype = "MD+1_compensatory"
                if stype != "rest" and stype != "match" and rng.random() < 0.03:
                    stype = "rest"  # individually managed / missed session

                if stype != "rest":
                    tpl = SESSION_TEMPLATES[stype]
                    if duration is None:
                        duration = float(np.clip(tpl["dur"] + rng.normal(0, 7), 15, 130))
                    mult = {"all": team.intensity * entry.get("spike", 1.0) if stype != "match" else 1.0}
                    if (drift.get("enabled") and entry["season"] == drift["season"]
                            and team.team_id in drift["teams"] and stype != "match"):
                        mult["hsr"] = drift["hsr_multiplier"]
                        mult["sprint"] = drift["sprint_multiplier"]
                    met = _session_metrics(ath, stype, duration, mult, rng)
                    u = met["player_load"] / match_pl
                    sprint_u = met["sprint_distance"] / max(prof["sprint"], 1)
                    session_row = dict(stype=stype, duration=duration, minutes_played=minutes_played,
                                       entry=entry, **met)
            played_yesterday = 0.0
            if session_row is not None and session_row["stype"] == "match":
                played_yesterday = session_row["minutes_played"]

        # ---- latent state update (Banister-style fitness/fatigue) -----------
        fitness += (u - fitness) / tau_f
        fatigue += (u - fatigue) / tau_a
        s_acute += (sprint_u - s_acute) / 7.0
        s_chronic += (sprint_u - s_chronic) / 28.0
        consecutive = consecutive + 1 if session_row is not None else 0

        # ---- ground-truth hazard (only for available athletes in season) ----
        available = unavailable_until is None
        if available and entry is not None:
            ratio = fatigue / max(fitness, 0.15)
            load_mult = min(np.exp(1.6 * max(0.0, ratio - ath.workload_tolerance)), 6.0)
            rec_mult = 1.0 + 0.10 * max(0, consecutive - 4)
            reinj_mult = 1.0
            if last_return is not None:
                reinj_mult += 1.8 * np.exp(-(d - last_return).days / 21.0)
            age_mult = np.exp(0.035 * (age - 26))
            hazard = H_NON_SPORT_PER_DAY
            if session_row is not None:
                hours = session_row["duration"] / 60.0
                is_match = session_row["stype"] == "match"
                sprint_mult = 1.0
                if session_row["sprint_distance"] > 0:
                    sprint_mult = np.exp(1.3 * max(0.0, s_acute / max(s_chronic, 0.03) - 1.35))
                    sprint_mult *= 1.0 + 0.7 * max(0.0, 0.5 - s_chronic / sprint_norm)
                    sprint_mult = min(sprint_mult, 5.0)
                under_prep = np.exp(2.0 * max(0.0, 0.7 - fitness / fitness_norm)) if is_match else 1.0
                base = (H_MATCH_PER_HOUR * (1 - MATCH_CONTACT_SHARE)) if is_match else H_TRAIN_PER_HOUR
                hazard += (base * hours * ath.injury_susceptibility * load_mult * sprint_mult * rec_mult
                           * reinj_mult * age_mult * under_prep)
                if is_match:
                    hazard += H_MATCH_PER_HOUR * MATCH_CONTACT_SHARE * hours * ath.injury_susceptibility ** 0.3
            p_inj = 1.0 - np.exp(-hazard)
            p_ill = ILLNESS_PER_DAY * (1.0 + 0.8 * max(0.0, ratio - 1.2))

            if rng.random() < p_inj:
                injury_n += 1
                if session_row is not None and session_row["stype"] == "match" and rng.random() < 0.35:
                    category, med, sd = "contact_trauma", 7, 1.0
                else:
                    r = rng.random()
                    category, med, sd = (("muscle", 9, 0.85) if r < 0.6 else
                                         ("ligament_joint", 18, 0.9) if r < 0.8 else ("tendon_overuse", 6, 0.7))
                recent = last_return is not None and (d - last_return).days < 60
                days_lost = int(np.clip(np.exp(rng.normal(np.log(med), sd)) * (1.3 if recent else 1.0), 1, 150))
                occurred = ("match" if session_row is not None and session_row["stype"] == "match" else
                            "training" if session_row is not None else "non_sport")
                absences.append(dict(athlete_id=ath.athlete_id, onset_date=d, days_lost=days_lost,
                                     absence_type="injury", category=category, occurred_in=occurred))
                unavailable_until = d + timedelta(days=days_lost)
                rtp_total = int(np.clip(days_lost // 4, 1, 10))
                rtp_days_left = rtp_total
                last_injury_onset = d
                if session_row is not None:
                    session_row["truncate"] = float(rng.uniform(0.3, 1.0))
            elif rng.random() < p_ill:
                days_lost = int(rng.integers(1, 7))
                absences.append(dict(athlete_id=ath.athlete_id, onset_date=d, days_lost=days_lost,
                                     absence_type="illness", category="illness", occurred_in="non_sport"))
                unavailable_until = d + timedelta(days=days_lost)
                rtp_total, rtp_days_left = 0, 0

        if session_row is not None:
            frac = session_row.pop("truncate", 1.0)
            entry = session_row.pop("entry")
            stype = session_row.pop("stype")
            dur = session_row.pop("duration") * frac
            mp = session_row.pop("minutes_played") * frac
            for k in ("total_distance", "high_speed_running_distance", "sprint_distance", "player_load"):
                session_row[k] *= frac
            for k in ("sprint_count", "accelerations", "decelerations"):
                session_row[k] = int(round(session_row[k] * frac))
            base_rpe = SESSION_TEMPLATES[stype]["rpe"]
            ratio = fatigue / max(fitness, 0.15)
            rpe = int(np.clip(round(base_rpe + 1.8 * (ratio - 1.0) + rng.normal(0, 0.7)), 1, 10))
            sessions.append(dict(
                athlete_id=ath.athlete_id, team_id=ath.team_id, date=d, season=entry["season"],
                session_type=stype, match_or_training="match" if stype == "match" else "training",
                duration_minutes=round(dur, 1), minutes_played=round(mp, 1),
                total_distance=round(session_row["total_distance"], 1),
                high_speed_running_distance=round(session_row["high_speed_running_distance"], 1),
                sprint_distance=round(session_row["sprint_distance"], 1),
                sprint_count=session_row["sprint_count"], accelerations=session_row["accelerations"],
                decelerations=session_row["decelerations"], max_speed=round(session_row["max_speed"], 2),
                player_load=round(session_row["player_load"], 1), perceived_exertion=rpe,
                training_load=round(rpe * dur, 1), position=ath.position, age=round(age, 2),
                previous_injury=bool(ath.prior_injury_count > 0 or prev_n > 0),
                days_since_last_injury=None if prev_onset is None else (d - prev_onset).days,
            ))
    return sessions, absences


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def generate(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    sc = cfg["synthetic"]
    rng = np.random.default_rng(sc["seed"])
    seasons = cfg["calendar"]["seasons"]
    start = date.fromisoformat(seasons[0]["start"])
    data_end = date.fromisoformat(sc["data_end"])

    athletes, teams = make_athletes(sc["n_teams"], sc["athletes_per_team"], rng, start)
    team_by_id = {t.team_id: t for t in teams}
    plans, fixtures = {}, []
    for team in teams:
        plans[team.team_id], fx = build_team_schedule(team, seasons, data_end, sc["winter_break_days"], rng)
        fixtures.extend(fx)

    all_sessions, all_absences = [], []
    for ath in athletes:
        s, a = simulate_athlete(ath, team_by_id[ath.team_id], plans[ath.team_id], start, data_end,
                                sc.get("drift_scenario", {}), rng)
        all_sessions.extend(s)
        all_absences.extend(a)

    sessions = pd.DataFrame(all_sessions).sort_values(["date", "athlete_id"]).reset_index(drop=True)
    sessions["date"] = pd.to_datetime(sessions["date"])
    sessions.insert(0, "session_id", [f"S{i:07d}" for i in range(1, len(sessions) + 1)])
    sessions["data_origin"] = DATA_ORIGIN

    absences = pd.DataFrame(all_absences)
    absences["onset_date"] = pd.to_datetime(absences["onset_date"])
    absences["return_date"] = absences["onset_date"] + pd.to_timedelta(absences["days_lost"], unit="D")
    absences = absences.sort_values(["onset_date", "athlete_id"]).reset_index(drop=True)
    absences.insert(0, "absence_id", [f"X{i:05d}" for i in range(1, len(absences) + 1)])
    absences["data_origin"] = DATA_ORIGIN
    absences = absences[["absence_id", "athlete_id", "onset_date", "return_date", "days_lost",
                         "absence_type", "category", "occurred_in", "data_origin"]]

    athletes_df = pd.DataFrame([{k: v for k, v in asdict(a).items()
                                 if k in ("athlete_id", "team_id", "position", "date_of_birth", "prior_injury_count")}
                                for a in athletes])
    athletes_df["data_origin"] = DATA_ORIGIN
    latent_df = pd.DataFrame([asdict(a) for a in athletes])
    fixtures_df = pd.DataFrame(fixtures).sort_values(["date", "team_id"]).reset_index(drop=True)
    fixtures_df["data_origin"] = DATA_ORIGIN
    return {"athletes": athletes_df, "sessions": sessions, "absences": absences,
            "fixtures": fixtures_df, "latent_athlete_traits": latent_df}


def save(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        compressed = name == "sessions"
        fname = f"{name}.csv.gz" if compressed else f"{name}.csv"
        p = out / fname
        df_out = df.copy()
        for c in df_out.columns:
            if pd.api.types.is_datetime64_any_dtype(df_out[c]):
                df_out[c] = df_out[c].dt.strftime("%Y-%m-%d")
        if compressed:
            df_out.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
        else:
            df_out.to_csv(p, index=False)
        paths[name] = str(p)
    return paths


def summarise(tables: dict[str, pd.DataFrame]) -> dict:
    s, a = tables["sessions"], tables["absences"]
    n_ath = tables["athletes"].shape[0]
    seasons_obs = s.groupby("athlete_id")["season"].nunique().mean()
    return {
        "athletes": int(n_ath),
        "sessions": int(len(s)),
        "date_range": [str(s["date"].min().date()), str(s["date"].max().date())],
        "injuries": int((a["absence_type"] == "injury").sum()),
        "illnesses": int((a["absence_type"] == "illness").sum()),
        "injuries_per_athlete_season": round(float((a["absence_type"] == "injury").sum() / n_ath / seasons_obs), 2),
        "median_days_lost_injury": float(a.loc[a["absence_type"] == "injury", "days_lost"].median()),
        "match_sessions": int((s["match_or_training"] == "match").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic PlayerLoad-Risk dataset.")
    parser.add_argument("--out", default=None, help="output directory (default: paths.synthetic_dir)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    tables = generate(cfg)
    paths = save(tables, args.out or cfg["paths"]["synthetic_dir"])
    from src.data.profiles import write_profiles

    write_profiles(args.out or cfg["paths"]["synthetic_dir"])
    summary = summarise(tables)
    logger.info("Synthetic data written: %s", json.dumps(paths, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
