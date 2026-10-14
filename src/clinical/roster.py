"""Player cards, season statistics, availability status and the league injury report."""

from __future__ import annotations

import pandas as pd

from src.clinical.injuries import CATALOGUE, player_injury_summary
from src.data.repository import Dataset

DEFAULT_CLUB_COLORS = ("#33415c", "#ffffff")


def club_records(ds: Dataset) -> dict[str, dict]:
    if ds.clubs is not None and len(ds.clubs):
        return {r["team_id"]: {k: (None if pd.isna(v) else v) for k, v in r.items()}
                for r in ds.clubs.to_dict("records")}
    return {t: {"team_id": t, "name": f"Team {t}", "short_name": t, "city": None,
                "primary_color": DEFAULT_CLUB_COLORS[0], "secondary_color": DEFAULT_CLUB_COLORS[1]}
            for t in sorted(ds.athletes["team_id"].unique())}


def player_bio(ds: Dataset, athlete_id: str, as_of) -> dict:
    a = ds.athletes.set_index("athlete_id").loc[athlete_id]
    bio = {"athlete_id": athlete_id, "name": athlete_id, "first_name": "", "last_name": athlete_id,
           "position": a["position"], "team_id": a["team_id"],
           "age": round((pd.Timestamp(as_of) - pd.Timestamp(a["date_of_birth"])).days / 365.25, 1),
           "date_of_birth": str(pd.Timestamp(a["date_of_birth"]).date())}
    if ds.players is not None and athlete_id in set(ds.players["athlete_id"]):
        p = ds.players.set_index("athlete_id").loc[athlete_id]
        for k in ("name", "first_name", "last_name", "nationality", "nationality_code", "shirt_number", "height_cm",
                  "weight_kg", "preferred_foot", "joined", "photo"):
            if k in p.index and not pd.isna(p[k]) and p[k] != "":
                bio[k] = p[k].item() if hasattr(p[k], "item") else p[k]
    return bio


def season_of(cfg: dict, date) -> dict | None:
    date = pd.Timestamp(date)
    for s in cfg["calendar"]["seasons"]:
        if pd.Timestamp(s["start"]) <= date <= pd.Timestamp(s["end"]):
            return s
    return None


def player_stats(ds: Dataset, cfg: dict, athlete_id: str, as_of) -> dict:
    as_of = pd.Timestamp(as_of)
    s = ds.sessions[(ds.sessions["athlete_id"] == athlete_id) & (ds.sessions["date"] <= as_of)]
    season = season_of(cfg, as_of)

    def block(df):
        m = df[(df["match_or_training"] == "match") & (df["minutes_played"] > 0)]
        return {"appearances": int(len(m)), "starts": int((m["minutes_played"] >= 60).sum()),
                "minutes": int(m["minutes_played"].sum()), "sessions": int(len(df)),
                "distance_km": round(float(df["total_distance"].sum()) / 1000, 1),
                "top_speed_kmh": round(float(df["max_speed"].max()), 1) if len(df) else None,
                "sprint_km": round(float(df["sprint_distance"].sum()) / 1000, 1)}

    out = {"career": block(s)}
    if season:
        out["season"] = {"name": season["name"], **block(s[s["date"] >= pd.Timestamp(season["start"])])}
    return out


def availability_status(summary: dict, as_of) -> dict:
    """ESPN-style status: Available · Day-to-day (expected back ≤ 7 days) · Out."""
    cur = summary.get("current")
    if not cur:
        return {"status": "Available", "detail": None}
    est = pd.Timestamp(cur["return_date"])
    days = (est - pd.Timestamp(as_of)).days
    return {"status": "Day-to-day" if days <= 7 else "Out", "detail": cur.get("diagnosis"),
            "since": cur["onset_date"], "expected_return": cur["return_date"], "days_to_return": int(days)}


def injury_report(ds: Dataset, as_of, recent_days: int = 21) -> dict:
    as_of = pd.Timestamp(as_of)
    ab = ds.absences
    det = ds.injury_details.set_index("absence_id") if ds.injury_details is not None else None
    clubs = club_records(ds)

    def enrich(r):
        d = det.loc[r.absence_id] if det is not None and r.absence_id in det.index else None
        bio = player_bio(ds, r.athlete_id, as_of)
        return {"athlete_id": r.athlete_id, "name": bio["name"], "position": bio["position"],
                "team_id": bio["team_id"], "club": clubs[bio["team_id"]]["name"],
                "shirt_number": bio.get("shirt_number"),
                "diagnosis": None if d is None else d["diagnosis"],
                "diagnosis_code": None if d is None else d["diagnosis_code"],
                "body_region": None if d is None else d["body_region"],
                "side": None if d is None or pd.isna(d["side"]) else d["side"],
                "severity": None if d is None else d["severity"],
                "recurrence": None if d is None or pd.isna(d["recurrence"]) else d["recurrence"],
                "absence_type": r.absence_type, "onset_date": str(r.onset_date.date()),
                "return_date": str(r.return_date.date()), "days_lost": int(r.days_lost)}

    current = ab[(ab["onset_date"] <= as_of) & (ab["return_date"] > as_of)]
    cur = []
    for r in current.sort_values("onset_date", ascending=False).itertuples(index=False):
        e = enrich(r)
        e["days_out"] = int((as_of - r.onset_date).days) + 1
        e["days_to_return"] = int((r.return_date - as_of).days)
        e["status"] = "Day-to-day" if e["days_to_return"] <= 7 else "Out"
        cur.append(e)
    window = as_of - pd.Timedelta(days=recent_days)
    news = []
    for r in ab[(ab["onset_date"] > window) & (ab["onset_date"] <= as_of)].itertuples(index=False):
        e = enrich(r)
        news.append({**e, "event": "injury" if r.absence_type == "injury" else "illness", "date": e["onset_date"]})
    for r in ab[(ab["return_date"] > window) & (ab["return_date"] <= as_of)].itertuples(index=False):
        e = enrich(r)
        news.append({**e, "event": "return", "date": e["return_date"]})
    news.sort(key=lambda x: x["date"], reverse=True)

    hist = ab[ab["onset_date"] <= as_of]
    stats = []
    if det is not None:
        j = hist.join(det[["diagnosis_code", "body_region"]], on="absence_id")
        j = j[j["absence_type"] == "injury"]
        for code, g in j.groupby("diagnosis_code"):
            stats.append({"diagnosis_code": code, "label": CATALOGUE[code].label if code in CATALOGUE else code,
                          "body_region": g["body_region"].iloc[0], "count": int(len(g)),
                          "median_days_lost": float(g["days_lost"].median()), "total_days_lost": int(g["days_lost"].sum())})
        stats.sort(key=lambda x: -x["count"])
    by_club = {t: {"club": c["name"], "out": sum(1 for x in cur if x["team_id"] == t)} for t, c in clubs.items()}
    return {"as_of": str(as_of.date()), "current": cur, "news": news[:40], "by_diagnosis": stats, "by_club": by_club,
            "n_current": len(cur)}


def player_injuries(ds: Dataset, athlete_id: str, as_of) -> dict:
    return player_injury_summary(ds.injury_details, ds.absences, athlete_id, as_of)
