"""Deterministic, explainable match probabilities from public results."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import math
import pandas as pd

@dataclass
class TeamStats:
    attack: float = 1.0
    defense: float = 1.0
    matches: int = 0

class SoccerModel:
    def __init__(self, home_advantage: float = 0.22, decay: float = 0.003):
        self.home_advantage, self.decay = home_advantage, decay
        self.stats: dict[str, TeamStats] = defaultdict(TeamStats)
        self.league_avg = 1.35
        self.as_of = None

    def fit(self, matches: pd.DataFrame) -> "SoccerModel":
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = matches.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=list(required)).sort_values("date")
        if frame.empty: return self
        self.as_of = frame["date"].max().date()
        total_goals = frame["home_goals"].sum() + frame["away_goals"].sum()
        self.league_avg = max(0.5, float(total_goals / (2 * len(frame))))
        weighted = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "w": 0.0})
        for row in frame.itertuples():
            age = max(0, (self.as_of - row.date.date()).days)
            weight = math.exp(-self.decay * age)
            for team, gf, ga in ((row.home_team, row.home_goals, row.away_goals), (row.away_team, row.away_goals, row.home_goals)):
                weighted[team]["gf"] += gf * weight; weighted[team]["ga"] += ga * weight; weighted[team]["w"] += weight
        for team, values in weighted.items():
            w = max(values["w"], 1e-9)
            self.stats[team] = TeamStats(max(0.35, values["gf"] / w / self.league_avg), max(0.35, values["ga"] / w / self.league_avg), int(w))
        return self

    @staticmethod
    def _poisson(lam: float, max_goals: int = 8) -> list[float]:
        values = [math.exp(-lam)]
        for k in range(1, max_goals + 1): values.append(values[-1] * lam / k)
        total = sum(values)
        return [v / total for v in values]

    def predict(self, home: str, away: str) -> dict[str, float | str]:
        h, a = self.stats[home], self.stats[away]
        home_xg = max(0.15, self.league_avg * h.attack * a.defense * math.exp(self.home_advantage))
        away_xg = max(0.15, self.league_avg * a.attack * h.defense)
        hp, ap = self._poisson(home_xg), self._poisson(away_xg)
        home_win = draw = away_win = 0.0
        for hg, ph in enumerate(hp):
            for ag, pa in enumerate(ap):
                value = ph * pa
                if hg > ag: home_win += value
                elif hg == ag: draw += value
                else: away_win += value
        probabilities = {"Home": home_win, "Draw": draw, "Away": away_win}
        pick = max(probabilities, key=probabilities.get)
        return {"home_team": home, "away_team": away, "pick": pick, "home_prob": home_win, "draw_prob": draw, "away_prob": away_win, "confidence": probabilities[pick], "home_xg": home_xg, "away_xg": away_xg}
