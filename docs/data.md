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
