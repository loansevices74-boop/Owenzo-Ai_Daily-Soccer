"""Public, keyless football-data.co.uk CSV loader."""
from __future__ import annotations
from io import StringIO
import pandas as pd
import requests

LEAGUES = {"England Premier":"E0","England Championship":"E1","England League One":"E2","England League Two":"E3","Scotland Premiership":"SC0","Scotland Championship":"SC1","Germany Bundesliga":"D1","Germany 2 Bundesliga":"D2","Italy Serie A":"I1","Italy Serie B":"I2","Spain La Liga":"SP1","Spain Segunda":"SP2","France Ligue 1":"F1","France Ligue 2":"F2","Netherlands Eredivisie":"N1","Belgium First Division":"B1","Portugal Liga":"P1","Turkey Super Lig":"T1","Greece Super League":"G1"}

def load_league(league: str, season: str = "2526", timeout: int = 20) -> pd.DataFrame:
    response = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/{LEAGUES[league]}.csv", timeout=timeout)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.content.decode("latin1")))
    frame = pd.DataFrame({"date":pd.to_datetime(raw["Date"],dayfirst=True,errors="coerce"),"home_team":raw["HomeTeam"],"away_team":raw["AwayTeam"],"home_goals":pd.to_numeric(raw["FTHG"],errors="coerce"),"away_goals":pd.to_numeric(raw["FTAG"],errors="coerce"),"league":league})
    return frame.dropna()

def load_many(leagues: list[str], season: str = "2526") -> pd.DataFrame:
    frames=[]
    for league in leagues:
        try: frames.append(load_league(league, season))
        except (requests.RequestException, ValueError, KeyError): pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["date","home_team","away_team","home_goals","away_goals","league"])
