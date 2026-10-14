"""Injury classification and clinical injury records.

The system does **not** diagnose. In a real deployment the diagnosis, body
region and mechanism are entered by medical staff; this module defines the
taxonomy used to store and display them, classifies severity and recurrence
with the football consensus definitions (Fuller et al., 2006), and — for the
synthetic demo only — assigns plausible diagnoses to the generated absences.

Severity (days lost, Fuller et al., 2006): slight 0–3 · minimal 4–7 ·
mild 8–28 · moderate/severe > 28 (we split > 28 into "moderate" 29–90 and
"severe" > 90 for display).

Recurrence: same diagnosis (and side) after return from the index injury —
*early* recurrence within 2 months of return, *late* within 12 months.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Diagnosis:
    code: str
    label: str
    region: str           # body region shown to users
    zone: str             # body-map zone id (see web/js/bodymap.js)
    tissue: str           # muscle | ligament | tendon | bone/joint | contusion | brain | systemic
    category: str         # matches absences.category
    mechanisms: tuple
    sided: bool = True
    description: str = ""


CATALOGUE: dict[str, Diagnosis] = {d.code: d for d in [
    Diagnosis("HAM", "Hamstring muscle injury", "Posterior thigh", "thigh_back", "muscle", "muscle",
              ("high-speed running", "sprinting", "overstretching"),
              description="Strain of the hamstring muscle group, most often during high-speed running."),
    Diagnosis("ADD", "Adductor muscle injury", "Groin / adductors", "groin", "muscle", "muscle",
              ("change of direction", "kicking", "overstretching"),
              description="Strain of the adductor (groin) muscles, typically with kicking or change of direction."),
    Diagnosis("QUA", "Quadriceps muscle injury", "Anterior thigh", "thigh_front", "muscle", "muscle",
              ("kicking", "sprinting", "acceleration"),
              description="Strain of the quadriceps, commonly rectus femoris during kicking or sprint starts."),
    Diagnosis("CAL", "Calf muscle injury", "Calf", "calf", "muscle", "muscle",
              ("acceleration", "jumping", "sprinting"),
              description="Strain of gastrocnemius or soleus, often during acceleration or late in sessions."),
    Diagnosis("ANK", "Ankle sprain (lateral ligament)", "Ankle", "ankle", "ligament", "ligament_joint",
              ("landing", "change of direction", "contact"),
              description="Inversion sprain of the lateral ankle ligaments."),
    Diagnosis("SYN", "Ankle syndesmosis sprain", "Ankle", "ankle", "ligament", "ligament_joint",
              ("contact", "external rotation"),
              description="'High' ankle sprain of the syndesmosis; usually longer recovery than lateral sprains."),
    Diagnosis("MCL", "Knee MCL sprain", "Knee", "knee", "ligament", "ligament_joint",
              ("contact", "valgus loading", "change of direction"),
              description="Sprain of the medial collateral ligament of the knee."),
    Diagnosis("MEN", "Knee meniscal injury", "Knee", "knee", "bone/joint", "ligament_joint",
              ("twisting", "change of direction"),
              description="Tear or irritation of the knee meniscus."),
    Diagnosis("PAT", "Patellar tendinopathy", "Knee", "knee", "tendon", "tendon_overuse",
              ("gradual onset",), description="Load-related pain of the patellar tendon."),
    Diagnosis("ACH", "Achilles tendinopathy", "Achilles", "achilles", "tendon", "tendon_overuse",
              ("gradual onset",), description="Load-related pain of the Achilles tendon."),
    Diagnosis("GRO", "Adductor-related groin pain", "Groin / adductors", "groin", "tendon", "tendon_overuse",
              ("gradual onset",), description="Longstanding adductor-related groin pain (Doha agreement entity)."),
    Diagnosis("TCO", "Thigh contusion", "Anterior thigh", "thigh_front", "contusion", "contact_trauma",
              ("collision", "tackle"), description="'Dead leg' from a direct knee/body impact."),
    Diagnosis("ACO", "Ankle/foot contusion", "Foot / ankle", "foot", "contusion", "contact_trauma",
              ("tackle", "stamp"), description="Direct-impact bruising of the foot or ankle."),
    Diagnosis("CON", "Concussion", "Head", "head", "brain", "contact_trauma",
              ("head collision", "fall"), sided=False,
              description="Sport-related concussion; managed with a graduated return-to-sport protocol."),
    Diagnosis("RIB", "Rib/chest contusion", "Chest", "chest", "contusion", "contact_trauma",
              ("collision",), description="Bruising of the ribs or chest wall from contact."),
    Diagnosis("URT", "Upper respiratory tract infection", "Systemic", "systemic", "systemic", "illness",
              ("illness",), sided=False, description="Common cold / respiratory infection."),
    Diagnosis("GAS", "Gastroenteritis", "Systemic", "systemic", "systemic", "illness",
              ("illness",), sided=False, description="Stomach/intestinal infection."),
    Diagnosis("VIR", "Viral illness / fever", "Systemic", "systemic", "systemic", "illness",
              ("illness",), sided=False, description="Non-specific viral illness with fever."),
]}

# Synthetic-demo assignment probabilities by recorded category (muscle mix follows the
# distribution reported by Ekstrand et al., 2011: hamstring > adductor > quadriceps > calf).
_MIX = {
    "muscle": [("HAM", 0.37), ("ADD", 0.23), ("QUA", 0.19), ("CAL", 0.21)],
    "ligament_joint": [("ANK", 0.45), ("MCL", 0.30), ("SYN", 0.12), ("MEN", 0.13)],
    "tendon_overuse": [("PAT", 0.35), ("ACH", 0.30), ("GRO", 0.35)],
    "contact_trauma": [("TCO", 0.40), ("ACO", 0.30), ("CON", 0.15), ("RIB", 0.15)],
    "illness": [("URT", 0.55), ("GAS", 0.30), ("VIR", 0.15)],
}


def severity(days_lost: int) -> str:
    if days_lost <= 3:
        return "Slight"
    if days_lost <= 7:
        return "Minimal"
    if days_lost <= 28:
        return "Mild"
    if days_lost <= 90:
        return "Moderate"
    return "Severe"


def muscle_grade(days_lost: int) -> str | None:
    """Indicative grade from time loss (display only; real grading is clinical/imaging-based)."""
    return "Grade 1" if days_lost <= 10 else "Grade 2" if days_lost <= 35 else "Grade 3"


def enrich_absences(absences: pd.DataFrame, players: pd.DataFrame | None = None, seed: int = 23) -> pd.DataFrame:
    """Synthetic demo: give each recorded absence a diagnosis, side, mechanism and recurrence status."""
    rng = np.random.default_rng(seed)
    foot = players.set_index("athlete_id")["preferred_foot"] if players is not None else pd.Series(dtype=object)
    ab = absences.sort_values(["athlete_id", "onset_date"]).reset_index(drop=True)
    rows = []
    for r in ab.itertuples(index=False):
        mix = _MIX.get(r.category, _MIX["illness"])
        codes, probs = zip(*mix)
        code = codes[rng.choice(len(codes), p=np.array(probs) / sum(probs))]
        # long lay-offs are more plausibly the more serious entity within a category
        if r.category == "ligament_joint" and r.days_lost > 40 and code == "ANK":
            code = "SYN"
        if code == "MEN" and r.days_lost < 14:
            code = "MCL"
        d = CATALOGUE[code]
        side = None
        if d.sided:
            kicking = foot.get(r.athlete_id, "Right")
            side = (kicking if kicking in ("Left", "Right") and rng.random() < 0.6
                    else ("Left" if rng.random() < 0.5 else "Right"))
        label = d.label
        if d.tissue == "muscle":
            label = f"{d.label} ({muscle_grade(int(r.days_lost)).lower()})"
        rows.append({"absence_id": r.absence_id, "athlete_id": r.athlete_id, "diagnosis_code": code,
                     "diagnosis": label, "body_region": d.region, "body_zone": d.zone, "side": side,
                     "tissue": d.tissue, "mechanism": d.mechanisms[rng.integers(len(d.mechanisms))],
                     "severity": severity(int(r.days_lost)),
                     "onset_date": r.onset_date, "return_date": r.return_date})
    out = pd.DataFrame(rows)
    out["recurrence"] = _recurrence(out)
    out["data_origin"] = "synthetic"
    for c in ("onset_date", "return_date"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    return out


def _recurrence(df: pd.DataFrame) -> list[str | None]:
    """Early (≤ 2 months after return) or late (≤ 12 months) recurrence of the same diagnosis and side."""
    out = []
    onset = pd.to_datetime(df["onset_date"])
    ret = pd.to_datetime(df["return_date"])
    for i, r in df.iterrows():
        prev = df[(df["athlete_id"] == r["athlete_id"]) & (df["diagnosis_code"] == r["diagnosis_code"])
                  & (df["side"].fillna("-") == (r["side"] or "-")) & (ret <= onset[i]) & (df.index != i)]
        if prev.empty or r["tissue"] == "systemic":
            out.append(None)
            continue
        gap = (onset[i] - ret[prev.index].max()).days
        out.append("Early recurrence" if gap <= 61 else "Late recurrence" if gap <= 365 else None)
    return out


def catalogue_records() -> list[dict]:
    return [asdict(d) for d in CATALOGUE.values()]


def player_injury_summary(details: pd.DataFrame, absences: pd.DataFrame, athlete_id: str, as_of) -> dict:
    """Injury history summary for one player as known at ``as_of``."""
    as_of = pd.Timestamp(as_of)
    ab = absences[(absences["athlete_id"] == athlete_id) & (absences["onset_date"] <= as_of)]
    det = details.set_index("absence_id") if details is not None and len(details) else None
    records = []
    for r in ab.sort_values("onset_date", ascending=False).itertuples(index=False):
        ongoing = r.return_date > as_of
        rec = {"absence_id": r.absence_id, "onset_date": str(r.onset_date.date()),
               "return_date": str(r.return_date.date()), "days_lost": int(r.days_lost) if not ongoing
               else int((as_of - r.onset_date).days) + 1, "absence_type": r.absence_type, "category": r.category,
               "occurred_in": getattr(r, "occurred_in", None), "status": "Ongoing" if ongoing else "Resolved",
               "expected_days_lost": int(r.days_lost)}
        if det is not None and r.absence_id in det.index:
            d = det.loc[r.absence_id]
            rec.update({k: (None if pd.isna(d[k]) else d[k]) for k in
                        ("diagnosis_code", "diagnosis", "body_region", "body_zone", "side", "tissue", "mechanism",
                         "severity", "recurrence")})
        else:
            rec.update({"diagnosis": r.category.replace("_", " ").capitalize(), "severity": severity(int(r.days_lost))})
        records.append(rec)
    inj = [r for r in records if r["absence_type"] == "injury"]
    zones: dict[str, int] = {}
    for r in inj:
        if r.get("body_zone"):
            key = f"{r['body_zone']}:{(r.get('side') or 'C')[0]}"
            zones[key] = zones.get(key, 0) + 1
    by_dx: dict[str, int] = {}
    for r in inj:
        by_dx[r.get("diagnosis_code") or r["category"]] = by_dx.get(r.get("diagnosis_code") or r["category"], 0) + 1
    last_year = [r for r in inj if pd.Timestamp(r["onset_date"]) > as_of - pd.Timedelta(days=365)]
    return {
        "records": records,
        "n_injuries": len(inj), "n_illnesses": len(records) - len(inj),
        "days_lost_injury": int(sum(r["days_lost"] for r in inj)),
        "injuries_last_365d": len(last_year),
        "days_lost_last_365d": int(sum(r["days_lost"] for r in last_year)),
        "recurrences": sum(1 for r in inj if r.get("recurrence")),
        "most_common": max(by_dx, key=by_dx.get) if by_dx else None,
        "body_map": zones,
        "current": next((r for r in records if r["status"] == "Ongoing"), None),
    }
