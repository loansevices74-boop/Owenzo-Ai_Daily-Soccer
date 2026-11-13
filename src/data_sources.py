from __future__ import annotations
from io import StringIO
from datetime import date
import pandas as pd
import requests

LEAGUES = {
    "England Premier": ("eng.1", "E0"), "England Championship": ("eng.2", "E1"), "England League One": ("eng.3", "E2"), "England League Two": ("eng.4", "E3"),
    "Spain La Liga": ("esp.1", "SP1"), "Spain Segunda": ("esp.2", "SP2"), "Italy Serie A": ("ita.1", "I1"), "Italy Serie B": ("ita.2", "I2"),
    "Germany Bundesliga": ("ger.1", "D1"), "Germany 2 Bundesliga": ("ger.2", "D2"), "France Ligue 1": ("fra.1", "F1"), "France Ligue 2": ("fra.2", "F2"),
    "Netherlands Eredivisie": ("ned.1", "N1"), "Belgium First Division": ("bel.1", "B1"), "Portugal Liga": ("por.1", "P1"), "Turkey Super Lig": ("tur.1", "T1"),
    "Scotland Premiership": ("sco.1", "SC0"), "Greece Super League": ("gre.1", "G1"), "USA MLS": ("usa.1", None), "Brazil Serie A": ("bra.1", None),
    "Argentina Liga Profesional": ("arg.1", None), "UEFA Champions League": ("uefa.champions", None), "UEFA Europa League": ("uefa.europa", None),
}
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
EMPTY = ["fixture_id", "date", "home_team", "away_team", "league", "status", "status_detail"]

def _event_row(event, league):
    competition = (event.get("competitions") or [{}])[0]
    teams = competition.get("competitors") or []
    home = next((item for item in teams if item.get("homeAway") == "home"), None)
    away = next((item for item in teams if item.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    status = event.get("status", {}).get("type", {})
    return {"fixture_id": str(event.get("id", "")), "date": pd.to_datetime(event.get("date"), utc=True, errors="coerce"), "home_team": home.get("team", {}).get("displayName", ""), "away_team": away.get("team", {}).get("displayName", ""), "league": league, "status": status.get("name", "STATUS_UNKNOWN"), "status_detail": status.get("detail", "")}

def load_fixtures_with_status(leagues, start: date, end: date | None = None, timeout: int = 20):
    end = end or start
    rows, source_status = [], {}
    for league in leagues:
        slug = LEAGUES[league][0]
        try:
            response = requests.get(f"{BASE}/{slug}/scoreboard", params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 500}, timeout=timeout)
            response.raise_for_status()
            events = response.json().get("events", [])
            source_status[league] = {"ok": True, "fixtures": len(events), "error": ""}
            rows.extend(row for row in (_event_row(event, league) for event in events) if row)
        except (requests.RequestException, ValueError, TypeError) as exc:
            source_status[league] = {"ok": False, "fixtures": 0, "error": str(exc)}
    if not rows:
        frame = pd.DataFrame(columns=EMPTY)
    else:
        frame = pd.DataFrame(rows).drop_duplicates("fixture_id").sort_values("date").reset_index(drop=True)
    return frame, source_status

def load_fixtures(leagues, start, end=None, timeout=20):
    return load_fixtures_with_status(leagues, start, end, timeout)[0]

def load_league_history(league, season="2526", timeout=20):
    code = LEAGUES[league][1]
    if not code:
        return pd.DataFrame()
    response = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv", timeout=timeout)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.content.decode("latin1")))
    return pd.DataFrame({"date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"), "home_team": raw["HomeTeam"], "away_team": raw["AwayTeam"], "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"), "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"), "league": league}).dropna()

def load_history(leagues, season="2526"):
    frames = []
    for league in leagues:
        try:
            frames.append(load_league_history(league, season))
        except (requests.RequestException, ValueError, KeyError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals", "league"])
