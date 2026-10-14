"""Fictional player & club database layered on top of the synthetic workload data.

!!! Every name, club and biography produced here is FICTIONAL. !!!
They make the demo read like a real squad-management product without
attaching (synthetic) injury records or risk scores to real people.

* ``clubs.csv``   – club name, short name, city, kit colours (used for crests/portraits)
* ``players.csv`` – name, nationality, shirt number, height, weight, preferred foot,
                    joined date, ``photo`` (optional path/URL – empty → generated portrait)

The generator uses its own seeded RNG, so it never changes the workload data or the model.
To use real data instead, provide your own ``players.csv`` / ``clubs.csv`` with the same
columns (see docs/data.md) — only with the necessary rights and consent.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

CLUBS = [
    # team_id, name, short, city, primary, secondary, founded
    ("T1", "Northbridge Athletic", "NBR", "Northbridge", "#14284b", "#e8b923", 1889),
    ("T2", "Riverside Rovers", "RIV", "Riverside", "#0b6e4f", "#ffffff", 1902),
    ("T3", "Harbour City FC", "HCF", "Harbour City", "#5fa8e0", "#0a2240", 1921),
    ("T4", "Kingsford United", "KIN", "Kingsford", "#7a1f3d", "#9fc5e8", 1894),
]

# Nationality → (code, first names, surnames). Generic, common names; combinations are random and fictional.
NAMES = {
    "England": ("ENG", ["Jack", "Callum", "Harvey", "Mason", "Owen", "Reece", "Toby", "Kieran", "Lewis", "Dominic"],
                ["Ashworth", "Brennan", "Caldwell", "Dawson", "Fenwick", "Hartley", "Pembury", "Rowntree", "Seddon", "Thorne"]),
    "Spain": ("ESP", ["Álvaro", "Iker", "Javier", "Mario", "Rubén", "Sergio", "Unai", "Adrián", "Pablo", "Hugo"],
              ["Arrieta", "Bermejo", "Campos", "Duarte", "Echeverría", "Llorente", "Montero", "Navas", "Olano", "Prieto"]),
    "France": ("FRA", ["Théo", "Lucas", "Mathis", "Hugo", "Yanis", "Bastien", "Romain", "Clément", "Enzo", "Maxime"],
               ["Aubert", "Barbier", "Chevalier", "Dumont", "Fabre", "Garnier", "Lemaire", "Marchand", "Perrin", "Roussel"]),
    "Germany": ("GER", ["Lukas", "Jonas", "Felix", "Niklas", "Tim", "Moritz", "Jannik", "Florian", "Leon", "Paul"],
                ["Albers", "Brandt", "Engel", "Hartmann", "Kessler", "Lindner", "Ostermann", "Ritter", "Seidel", "Vogt"]),
    "Brazil": ("BRA", ["Rafael", "Thiago", "Gabriel", "Matheus", "Vinícius", "Caio", "Igor", "Lucas", "Bruno", "Diego"],
               ["Alencar", "Barros", "Cardoso", "Fontes", "Guimarães", "Macedo", "Nogueira", "Paiva", "Rezende", "Teixeira"]),
    "Netherlands": ("NED", ["Daan", "Sem", "Bram", "Thijs", "Jesse", "Ruben", "Luuk", "Stijn", "Joris", "Mees"],
                    ["Bakker", "de Groot", "Hendriks", "Jansen", "Kuipers", "Meijer", "Postma", "Smits", "van Dam", "Visser"]),
    "Portugal": ("POR", ["Tiago", "Rúben", "Gonçalo", "Diogo", "Afonso", "Duarte", "Nuno", "Rodrigo", "Vasco", "Martim"],
                 ["Amaral", "Batista", "Correia", "Fonseca", "Lobo", "Magalhães", "Pinto", "Quintela", "Soares", "Valente"]),
    "Nigeria": ("NGA", ["Chidi", "Emeka", "Tunde", "Ikenna", "Seun", "Femi", "Obinna", "Kelechi", "Uche", "Ade"],
                ["Adeyemi", "Balogun", "Chukwu", "Eze", "Ibekwe", "Nwosu", "Obi", "Okafor", "Olawale", "Uzor"]),
    "Senegal": ("SEN", ["Moussa", "Cheikh", "Pape", "Abdou", "Ibrahima", "Lamine", "Ousmane", "Babacar", "Saliou", "Mamadou"],
                ["Ba", "Cissé", "Diallo", "Faye", "Gueye", "Mbaye", "Ndiaye", "Sarr", "Seck", "Thiam"]),
    "Norway": ("NOR", ["Magnus", "Sander", "Henrik", "Eirik", "Sindre", "Vegard", "Jonas", "Kristian", "Tobias", "Mathias"],
               ["Berntsen", "Dahl", "Eide", "Haugen", "Lie", "Moen", "Nygård", "Solberg", "Strand", "Vik"]),
    "Japan": ("JPN", ["Haruto", "Ren", "Sota", "Yuto", "Daiki", "Kaito", "Riku", "Takumi", "Kenta", "Shun"],
              ["Aoki", "Fujita", "Hayashi", "Ishida", "Kondo", "Morita", "Nakano", "Ogawa", "Sakai", "Ueda"]),
    "USA": ("USA", ["Tyler", "Brandon", "Jordan", "Cole", "Austin", "Logan", "Chase", "Dylan", "Hunter", "Wyatt"],
            ["Bennett", "Carver", "Donnelly", "Fletcher", "Harlow", "Kendrick", "Lawson", "Mercer", "Prescott", "Whitaker"]),
    "Argentina": ("ARG", ["Facundo", "Lautaro", "Nicolás", "Gonzalo", "Matías", "Franco", "Agustín", "Tomás", "Joaquín",
                          "Ezequiel"],
                  ["Acosta", "Benítez", "Castaño", "Ferreyra", "Godoy", "Ledesma", "Medina", "Quiroga", "Sosa", "Villalba"]),
    "Denmark": ("DEN", ["Mads", "Rasmus", "Mikkel", "Frederik", "Anders", "Kasper", "Emil", "Oliver", "Jeppe", "Nikolaj"],
                ["Bach", "Dam", "Holm", "Juhl", "Krogh", "Lund", "Mørk", "Rask", "Skov", "Winther"]),
    "Morocco": ("MAR", ["Youssef", "Hamza", "Anas", "Ayoub", "Ilias", "Zakaria", "Soufiane", "Mehdi", "Adam", "Rayane"],
                ["Alaoui", "Benali", "Chraibi", "El Amrani", "Fassi", "Haddad", "Idrissi", "Lahlou", "Mansouri", "Tazi"]),
    "Croatia": ("CRO", ["Luka", "Ivan", "Mateo", "Ante", "Josip", "Marko", "Filip", "Dario", "Domagoj", "Toni"],
                ["Babić", "Horvat", "Jurić", "Kovač", "Marić", "Novak", "Pavlović", "Šimić", "Tomić", "Vuković"]),
}
HOME_WEIGHT = 0.30  # share of players from the "home" nation (England)

SHIRT_NUMBERS = {"GK": [1, 13, 31, 40], "CB": [4, 5, 6, 15, 26, 33, 44, 35], "FB": [2, 3, 12, 22, 27, 29, 32],
                 "CM": [8, 6, 16, 18, 20, 21, 24, 25, 28], "W": [7, 11, 17, 19, 23, 37], "ST": [9, 10, 14, 30, 39, 99]}
HEIGHT = {"GK": (191, 4), "CB": (188, 4), "FB": (178, 5), "CM": (179, 5), "W": (175, 5), "ST": (183, 6)}


def generate_profiles(athletes: pd.DataFrame, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clubs = pd.DataFrame(CLUBS, columns=["team_id", "name", "short_name", "city", "primary_color", "secondary_color",
                                         "founded"])
    clubs = clubs[clubs["team_id"].isin(athletes["team_id"].unique())].reset_index(drop=True)
    clubs["league"] = "PlayerLoad League (fictional)"
    clubs["data_origin"] = "synthetic"

    nations = list(NAMES)
    weights = np.array([HOME_WEIGHT] + [(1 - HOME_WEIGHT) / (len(nations) - 1)] * (len(nations) - 1))
    used_names, rows = set(), []
    used_numbers: dict[str, set] = {}
    for a in athletes.sort_values("athlete_id").itertuples(index=False):
        nation = nations[rng.choice(len(nations), p=weights)]
        code, firsts, lasts = NAMES[nation]
        for _ in range(50):
            name = f"{firsts[rng.integers(len(firsts))]} {lasts[rng.integers(len(lasts))]}"
            if name not in used_names:
                break
        used_names.add(name)
        taken = used_numbers.setdefault(a.team_id, set())
        pool = [n for n in SHIRT_NUMBERS[a.position] if n not in taken] or \
               [n for n in range(2, 60) if n not in taken]
        number = int(pool[0] if rng.random() < 0.6 else rng.choice(pool))
        taken.add(number)
        mu, sd = HEIGHT[a.position]
        height = int(round(rng.normal(mu, sd)))
        weight = int(round(rng.normal(23.2, 1.1) * (height / 100) ** 2))
        if a.position in ("FB", "W"):
            foot = "Left" if rng.random() < 0.45 else "Right"
        else:
            foot = rng.choice(["Right", "Left", "Both"], p=[0.74, 0.22, 0.04])
        joined = pd.Timestamp("2024-07-01") - pd.Timedelta(days=int(rng.integers(0, 6 * 365)))
        rows.append({"athlete_id": a.athlete_id, "name": name, "first_name": name.split(" ", 1)[0],
                     "last_name": name.split(" ", 1)[1], "nationality": nation, "nationality_code": code,
                     "shirt_number": number, "height_cm": height, "weight_kg": weight, "preferred_foot": foot,
                     "joined": joined.date().isoformat(), "photo": "", "data_origin": "synthetic"})
    return pd.DataFrame(rows), clubs


def write_profiles(directory: str | Path) -> None:
    """(Re)generate players.csv, clubs.csv and injury_details.csv next to the synthetic workload data."""
    from src.clinical.injuries import enrich_absences

    d = Path(directory)
    athletes = pd.read_csv(d / "athletes.csv")
    players, clubs = generate_profiles(athletes)
    players.to_csv(d / "players.csv", index=False)
    clubs.to_csv(d / "clubs.csv", index=False)
    absences = pd.read_csv(d / "absences.csv", parse_dates=["onset_date", "return_date"])
    details = enrich_absences(absences, players)
    details.to_csv(d / "injury_details.csv", index=False)
    logger.info("Wrote %d players, %d clubs, %d injury records to %s", len(players), len(clubs), len(details), d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fictional player/club database and injury details")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    write_profiles(load_config()["paths"]["synthetic_dir"])


if __name__ == "__main__":
    main()
