"""Data access layer.

The rest of the code base works with an in-memory :class:`Dataset`. Where it
comes from is decided here:

* ``DATABASE_URL`` set (PostgreSQL in Docker, or ``sqlite:///…`` locally) →
  tables are read from the database; an empty database is seeded from the
  synthetic CSV files on first use.
* otherwise → CSV files in ``paths.synthetic_dir`` (generated on demand).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

TABLES = ("athletes", "sessions", "absences", "fixtures")
# Optional descriptive tables (player bios, clubs, clinical injury details). They are not
# model inputs and are excluded from the data fingerprint.
EXTRA_TABLES = ("players", "clubs", "injury_details")
DATE_COLUMNS = {
    "athletes": ["date_of_birth"],
    "sessions": ["date"],
    "absences": ["onset_date", "return_date"],
    "fixtures": ["date"],
}


@dataclass
class Dataset:
    athletes: pd.DataFrame
    sessions: pd.DataFrame
    absences: pd.DataFrame
    fixtures: pd.DataFrame
    players: pd.DataFrame | None = None
    clubs: pd.DataFrame | None = None
    injury_details: pd.DataFrame | None = None

    @property
    def data_end(self) -> pd.Timestamp:
        return self.sessions["date"].max()

    @property
    def data_start(self) -> pd.Timestamp:
        return self.sessions["date"].min()

    def fingerprint(self) -> str:
        """Content hash used to tie a trained model to the exact data it saw."""
        h = hashlib.sha256()
        for name in TABLES:
            df = getattr(self, name)
            h.update(name.encode())
            h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
        return h.hexdigest()[:16]

    def subset(self, athlete_ids) -> Dataset:
        ids = set(athlete_ids)
        teams = set(self.athletes.loc[self.athletes["athlete_id"].isin(ids), "team_id"])
        return Dataset(
            athletes=self.athletes[self.athletes["athlete_id"].isin(ids)].reset_index(drop=True),
            sessions=self.sessions[self.sessions["athlete_id"].isin(ids)].reset_index(drop=True),
            absences=self.absences[self.absences["athlete_id"].isin(ids)].reset_index(drop=True),
            fixtures=self.fixtures[self.fixtures["team_id"].isin(teams)].reset_index(drop=True),
            players=None if self.players is None else self.players[self.players["athlete_id"].isin(ids)],
            clubs=self.clubs,
            injury_details=None if self.injury_details is None
            else self.injury_details[self.injury_details["athlete_id"].isin(ids)],
        )

    def truncate(self, end) -> Dataset:
        """Return the dataset as it would have looked at the end of day ``end``.

        Absences that started on/before ``end`` are kept but their return date
        is unknown at ``end`` if it lies in the future, so it is censored to
        ``end + 1`` (still absent). Fixtures are kept: the calendar is public.
        """
        end = pd.Timestamp(end)
        ab = self.absences[self.absences["onset_date"] <= end].copy()
        ab.loc[ab["return_date"] > end, "return_date"] = end + pd.Timedelta(days=1)
        det = None if self.injury_details is None else \
            self.injury_details[self.injury_details["absence_id"].isin(ab["absence_id"])]
        return Dataset(self.athletes.copy(), self.sessions[self.sessions["date"] <= end].copy(), ab,
                       self.fixtures.copy(), self.players, self.clubs, det)


def _parse(df: pd.DataFrame, table: str) -> pd.DataFrame:
    for c in DATE_COLUMNS.get(table, []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if table == "sessions":
        df = df.sort_values(["athlete_id", "date", "session_id"]).reset_index(drop=True)
    return df


def _csv_path(directory: Path, table: str) -> Path:
    gz = directory / f"{table}.csv.gz"
    return gz if gz.exists() else directory / f"{table}.csv"


def load_from_csv(directory: str | Path) -> Dataset:
    directory = Path(directory)
    frames = {}
    for t in TABLES:
        p = _csv_path(directory, t)
        frames[t] = _parse(pd.read_csv(p), t) if p.exists() else pd.DataFrame()
    for t in EXTRA_TABLES:
        p = _csv_path(directory, t)
        frames[t] = pd.read_csv(p, keep_default_na=True) if p.exists() else None
    return Dataset(**frames)


def ensure_synthetic_data(cfg: dict) -> Path:
    directory = Path(cfg["paths"]["synthetic_dir"])
    if not _csv_path(directory, "sessions").exists():
        from src.data.synthetic import generate, save

        logger.info("No data found in %s – generating synthetic dataset", directory)
        save(generate(cfg), directory)
    return directory


# --------------------------------------------------------------------------- database
def get_engine(url: str | None = None):
    from sqlalchemy import create_engine

    url = url or os.environ.get("DATABASE_URL")
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True)


def database_has_data(engine) -> bool:
    from sqlalchemy import inspect, text

    if not inspect(engine).has_table("sessions"):
        return False
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM sessions")).scalar() > 0


def seed_database(engine, dataset: Dataset) -> None:
    for t in TABLES + EXTRA_TABLES:
        df = getattr(dataset, t)
        if df is None:
            continue
        df.to_sql(t, engine, if_exists="replace", index=False, chunksize=5000, method="multi")
        logger.info("Seeded table %s (%d rows)", t, len(df))
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sessions_athlete_date ON sessions (athlete_id, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_absences_athlete ON absences (athlete_id)"))


def load_from_database(engine) -> Dataset:
    from sqlalchemy import inspect

    frames = {t: _parse(pd.read_sql_table(t, engine), t) for t in TABLES}
    names = set(inspect(engine).get_table_names())
    frames.update({t: pd.read_sql_table(t, engine) if t in names else None for t in EXTRA_TABLES})
    return Dataset(**frames)


def load_dataset(cfg: dict | None = None, prefer_database: bool = True) -> Dataset:
    cfg = cfg or load_config()
    directory = ensure_synthetic_data(cfg)
    engine = get_engine() if prefer_database else None
    if engine is not None:
        try:
            if not database_has_data(engine):
                seed_database(engine, load_from_csv(directory))
            ds = load_from_database(engine)
            logger.info("Loaded dataset from database (%d sessions)", len(ds.sessions))
            return ds
        except Exception as exc:  # pragma: no cover - depends on infrastructure
            logger.warning("Database unavailable (%s) – falling back to CSV files", exc)
    ds = load_from_csv(directory)
    logger.info("Loaded dataset from %s (%d sessions)", directory, len(ds.sessions))
    return ds


def log_predictions(rows: list[dict]) -> None:
    """Append served predictions to the ``prediction_log`` table (best effort)."""
    engine = get_engine()
    if engine is None or not rows:
        return
    try:
        pd.DataFrame(rows).to_sql("prediction_log", engine, if_exists="append", index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not write prediction log: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Database utilities")
    parser.add_argument("command", choices=["seed", "status"])
    parser.add_argument("--force", action="store_true", help="re-seed even if data exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    engine = get_engine()
    if engine is None:
        raise SystemExit("DATABASE_URL is not set")
    if args.command == "seed":
        if args.force or not database_has_data(engine):
            seed_database(engine, load_from_csv(ensure_synthetic_data(cfg)))
        else:
            logger.info("Database already seeded – use --force to overwrite")
    else:
        logger.info("Database has data: %s", database_has_data(engine))


if __name__ == "__main__":
    main()
