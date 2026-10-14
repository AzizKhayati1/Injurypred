"""Static web bundle (Netlify): routing config and exported data are consistent."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_netlify_config_spa_fallback():
    cfg = tomllib.loads((ROOT / "netlify.toml").read_text())
    assert cfg["build"]["base"] == "web" and cfg["build"]["publish"] == "."
    assert not (WEB / "requirements.txt").exists()  # keeps Netlify from installing the Python stack
    assert tomllib.loads((WEB / "netlify.toml").read_text()) == cfg
    redirects = cfg["redirects"]
    assert redirects[-1] == {"from": "/*", "to": "/index.html", "status": 200}
    # /data misses must 404 (never the HTML shell) and come before the catch-all
    assert redirects[0]["from"] == "/data/*" and redirects[0]["status"] == 404
    assert "/*        /index.html   200" in (WEB / "_redirects").read_text()


def test_exported_data_is_complete():
    squad = json.loads((WEB / "data" / "squad.json").read_text())
    players = json.loads((WEB / "data" / "players.json").read_text())["players"]
    assert len(players) == len(squad["athletes"]) and all(p["name"] and p["club"]["name"] for p in players)
    for f in ("clubs.json", "injuries.json"):
        assert (WEB / "data" / f).exists()
    meta = json.loads((WEB / "data" / "meta.json").read_text())
    assert meta["model"]["data_origin"] == "synthetic"
    for a in squad["athletes"]:
        d = json.loads((WEB / "data" / "athletes" / f"{a['athlete_id']}.json").read_text())
        assert {"profile", "player", "prediction", "risk_history", "workload", "simulation", "injuries",
                "prevention"} <= set(d)
        assert d["player"]["name"] and d["prevention"]["recommendations"]
        if a["available"]:
            assert d["simulation"] and 0 <= d["prediction"]["risk_probability"] <= 1


def test_every_nav_link_has_a_route():
    html = (WEB / "index.html").read_text()
    app = (WEB / "js" / "app.js").read_text()
    routes = set(re.findall(r'route\("([^"]+)"', app))
    for href in re.findall(r'href="(/[a-z-]*)" data-link', html):
        assert href in routes, href


def test_assets_are_local():
    html = (WEB / "index.html").read_text()
    assert "http://" not in html and "https://" not in html  # no CDN dependency
    assert (WEB / "vendor" / "chart.umd.min.js").exists()
