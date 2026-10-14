# Data card

## ⚠️ All bundled data is synthetic

`data/synthetic/` is produced by `src/data/synthetic.py` (seeded, deterministic). It does **not**
describe real athletes, teams or injuries, and no real athlete data was used to fit its
parameters beyond order-of-magnitude values from published literature. Every table carries a
`data_origin = "synthetic"` column and the API returns `"data_origin": "synthetic"`.

## Why synthetic?

GPS/injury data from professional clubs is proprietary, personal health data. Public datasets
exist — e.g. *SoccerMon* (Midoglu et al., 2024, *Scientific Data*: subjective wellness,
training load and GPS from two elite women's teams) — but licensing, coverage and format vary.
The project therefore ships a realistic generator so it runs out of the box, and keeps
ingestion behind a validated interface (`src/data/schemas.py`, `src/data/validation.py`,
`src/data/repository.py`) so a real dataset can be mapped onto the same tables.

## Tables

| file | grain | key columns |
|---|---|---|
| `athletes.csv` | athlete | `athlete_id`, `team_id`, `position`, `date_of_birth`, `prior_injury_count` |
| `sessions.csv.gz` | session | `session_id`, `athlete_id`, `date`, `session_type`, `match_or_training`, `duration_minutes`, `minutes_played`, `total_distance`, `high_speed_running_distance` (>19.8 km/h), `sprint_distance` (>25.2 km/h), `sprint_count`, `accelerations`, `decelerations`, `max_speed` (km/h), `player_load` (AU), `perceived_exertion` (CR-10), `training_load` (sRPE = RPE × min), `position`, `age`, `previous_injury`, `days_since_last_injury` |
| `absences.csv` | absence | `absence_id`, `athlete_id`, `onset_date`, `return_date` (first day available again), `days_lost`, `absence_type` (injury/illness), `category`, `occurred_in` |
| `fixtures.csv` | team match | `team_id`, `date`, `match_type` (friendly/league/cup) — the published calendar, including future fixtures |
| `latent_athlete_traits.csv` | athlete | **hidden generator traits — never used by the model**; kept only for auditing what the model recovers |

## Generator design

* 4 teams × 25 athletes (GK, CB, FB, CM, W, ST), seasons 2024-25 and 2025-26 plus the current
  2026-27 season up to 2026-10-06; June is the off-season (untracked).
* Football micro-cycles (MD+1 recovery/compensatory, MD-4 strength, MD-3 endurance, MD-2
  tactical, MD-1 activation), pre-season ramps and friendlies, midweek cup fixtures, winter break.
* Position-specific locomotor profiles; athlete fitness, top speed, selection status (starter vs
  squad player), age; random session-to-session noise; team "intensification" weeks (load spikes);
  return-to-play modified training after absences.
* **Hidden traits**: workload tolerance, injury susceptibility, recovery capacity.
* **Ground-truth hazard** (per day, available athletes): exposure-based (≈ 2.6 /1000 h training,
  ≈ 21 /1000 h match; cf. Ekstrand et al., 2011) × susceptibility × fitness–fatigue overload
  above the athlete's tolerance × sprint spike / sprint under-exposure × consecutive days without
  rest × re-injury window after return × age × under-preparation for matches; plus a
  load-independent contact-injury share in matches and non-load illness.
* **Intentional drift**: in 2026-27, teams T1 and T3 adopt a high-speed-running emphasis
  (HSR × 1.25, sprint × 1.30 in training) so the monitoring module has something real to detect.

Resulting dataset: 100 athletes, 55,953 sessions, 553 injuries (≈ 1.8 per athlete-season,
median 10 days lost), 161 illnesses; 7-day availability event rate ≈ 7 %.

## Validation rules

Implemented in `src/data/validation.py` (table-level) and `src/data/schemas.py` (record-level,
Pydantic): required columns, missing values, unparseable/future timestamps, negative distances
or counts, impossible speeds (outside 0–45 km/h), invalid durations (≤ 0 or > 240 min), HSR >
total distance, sprint > HSR, RPE outside CR-10, duplicate session IDs and duplicate sessions,
chronological ordering, unknown athlete IDs, position mismatches, overlapping absences,
sessions recorded during an absence. Errors block training and serving.

## Using real data

Map your export onto the four tables above (CSV files in `paths.synthetic_dir`, or tables in
the database pointed to by `DATABASE_URL`), run `make validate`, then `make train`. Set the
season calendar in `configs/config.yaml → calendar`.

## Player database, clubs and clinical injury records

Three descriptive tables sit next to the workload data. They are **not model inputs** (excluded
from the data fingerprint), so they never change predictions.

| file | key columns |
|---|---|
| `players.csv` | `athlete_id`, `name`, `first_name`, `last_name`, `nationality`, `nationality_code`, `shirt_number`, `height_cm`, `weight_kg`, `preferred_foot`, `joined`, `photo` (optional URL or `/img/players/<id>.jpg`) |
| `clubs.csv` | `team_id`, `name`, `short_name`, `city`, `primary_color`, `secondary_color`, `founded`, `league` |
| `injury_details.csv` | `absence_id`, `athlete_id`, `diagnosis_code`, `diagnosis`, `body_region`, `body_zone`, `side`, `tissue`, `mechanism`, `severity`, `recurrence` |

In the bundled demo all three are generated by `src/data/profiles.py` (`python -m src.data.profiles`):
**every player, club and diagnosis is fictional.** Diagnoses are drawn per recorded absence
category (muscle-injury mix following Ekstrand et al., 2011); severity uses the Fuller et al.
(2006) time-loss bands; recurrences are the same diagnosis and side within 2 months (early) or
12 months (late) of return. The diagnosis taxonomy lives in `src/clinical/injuries.py`
(`GET /injury-catalogue`).

**PlayerLoad-Risk does not diagnose.** In a real deployment the diagnosis, side and mechanism are
entered by medical staff; the system stores, summarises and visualises them (body map, injury
log, injury report) and uses them for prevention suggestions.

## Bring your own (real) data

1. Export your workload and absence data into `athletes.csv`, `sessions.csv(.gz)`, `absences.csv`
   and `fixtures.csv` (columns above) and, optionally, `players.csv`, `clubs.csv` and
   `injury_details.csv`. Use `diagnosis_code` values from the catalogue so the body map and
   prevention rules work.
2. Photos (optional): put `<athlete_id>.jpg|png|webp` in `web/img/players/` or set `photo` to a URL.
3. Set the season calendar in `configs/config.yaml`, then run `make validate`, `make train` and
   `make web`.

Before doing this with real people: workload and injury records are **health data**. You need a
lawful basis and informed consent (e.g. GDPR Art. 9 special-category data), access control,
purpose limitation and retention rules — and you must **not** publish identifiable players'
injury records or risk scores on a public website. Deploy real data only to an access-controlled
environment, and only publish photos you have the rights to use.
