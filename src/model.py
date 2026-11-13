"""Explainable pre-match model with Poisson, Elo, form and market analysis."""
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
    home_matches: int = 0
    away_matches: int = 0
    form: float = 0.5
    home_form: float = 0.5
    away_form: float = 0.5
    elo: float = 1500.0

class SoccerModel:
    def __init__(self, home_advantage: float = 0.22, decay: float = 0.003, min_matches: int = 8):
        self.home_advantage, self.decay, self.min_matches = home_advantage, decay, min_matches
        self.stats: dict[str, TeamStats] = defaultdict(TeamStats)
        self.league_avg, self.as_of, self.training_matches = 1.35, None, 0

    def fit(self, matches: pd.DataFrame) -> "SoccerModel":
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = matches.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["home_goals", "away_goals"]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(required)).sort_values("date")
        if frame.empty: return self
        self.as_of, self.training_matches = frame["date"].max().date(), len(frame)
        total_goals = frame["home_goals"].sum() + frame["away_goals"].sum()
        self.league_avg = max(0.5, float(total_goals / (2 * len(frame))))
        weighted = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "w": 0.0, "home": 0, "away": 0, "points": [], "home_points": [], "away_points": []})
        ratings = defaultdict(lambda: 1500.0)
        for row in frame.itertuples():
            age = max(0, (self.as_of - row.date.date()).days)
            weight = math.exp(-self.decay * age)
            result_home = 3 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 0
            result_away = 3 - result_home if result_home != 1 else 1
            for team, gf, ga, points, venue in ((row.home_team, row.home_goals, row.away_goals, result_home, "home"), (row.away_team, row.away_goals, row.home_goals, result_away, "away")):
                item = weighted[team]; item["gf"] += gf * weight; item["ga"] += ga * weight; item["w"] += weight; item[venue] += 1; item["points"].append(points); item[f"{venue}_points"].append(points)
            expected = 1 / (1 + 10 ** ((ratings[row.away_team] - ratings[row.home_team] - 55) / 400))
            actual = 1.0 if result_home == 3 else 0.5 if result_home == 1 else 0.0
            change = 20 * (actual - expected)
            ratings[row.home_team] += change; ratings[row.away_team] -= change
        for team, values in weighted.items():
            w = max(values["w"], 1e-9)
            self.stats[team] = TeamStats(
                attack=max(0.35, values["gf"] / w / self.league_avg), defense=max(0.35, values["ga"] / w / self.league_avg), matches=sum(1 for _ in values["points"]),
                home_matches=values["home"], away_matches=values["away"], form=sum(values["points"][-10:]) / max(1, 3 * len(values["points"][-10:])),
                home_form=sum(values["home_points"][-8:]) / max(1, 3 * len(values["home_points"][-8:])), away_form=sum(values["away_points"][-8:]) / max(1, 3 * len(values["away_points"][-8:])), elo=ratings[team])
        return self

    @staticmethod
    def _poisson(lam: float, max_goals: int = 8) -> list[float]:
        values = [math.exp(-lam)]
        for k in range(1, max_goals + 1): values.append(values[-1] * lam / k)
        total = sum(values)
        return [value / total for value in values]

    def _score_matrix(self, home_xg: float, away_xg: float):
        hp, ap = self._poisson(home_xg), self._poisson(away_xg)
        matrix = {(hg, ag): ph * pa for hg, ph in enumerate(hp) for ag, pa in enumerate(ap)}
        return matrix

    def predict(self, home: str, away: str) -> dict[str, float | str]:
        h, a = self.stats[home], self.stats[away]
        home_xg = max(0.15, self.league_avg * h.attack * a.defense * math.exp(self.home_advantage))
        away_xg = max(0.15, self.league_avg * a.attack * h.defense)
        matrix = self._score_matrix(home_xg, away_xg)
        home_win = sum(p for (hg, ag), p in matrix.items() if hg > ag)
        draw = sum(p for (hg, ag), p in matrix.items() if hg == ag)
        away_win = sum(p for (hg, ag), p in matrix.items() if hg < ag)
        probabilities = {"Home": home_win, "Draw": draw, "Away": away_win}
        pick = max(probabilities, key=probabilities.get)
        return {"home_team": home, "away_team": away, "pick": pick, "home_prob": home_win, "draw_prob": draw, "away_prob": away_win, "confidence": probabilities[pick], "home_xg": home_xg, "away_xg": away_xg}

    def analyze(self, home: str, away: str) -> dict:
        base = self.predict(home, away); h, a = self.stats[home], self.stats[away]
        matrix = self._score_matrix(base["home_xg"], base["away_xg"])
        total_goals = sum(p for (hg, ag), p in matrix.items() if hg + ag > 1.5)
        under_35 = sum(p for (hg, ag), p in matrix.items() if hg + ag < 3.5)
        btts = sum(p for (hg, ag), p in matrix.items() if hg > 0 and ag > 0)
        markets = {
            "Home": base["home_prob"], "Draw": base["draw_prob"], "Away": base["away_prob"],
            "Home or Draw": base["home_prob"] + base["draw_prob"], "Away or Draw": base["away_prob"] + base["draw_prob"],
            "Over 1.5 Goals": total_goals, "Under 3.5 Goals": under_35, "Both Teams To Score": btts,
        }
        quality = "High" if min(h.matches, a.matches) >= 20 else "Medium" if min(h.matches, a.matches) >= self.min_matches else "Low"
        uncertainty = max(0.0, (self.min_matches - min(h.matches, a.matches)) / max(1, self.min_matches)) * 0.10
        ranked = sorted(markets.items(), key=lambda item: item[1], reverse=True)
        market, probability = ranked[0]
        adjusted = max(0.0, probability - uncertainty)
        tier = "Elite" if adjusted >= 0.80 else "Strong" if adjusted >= 0.75 else "Qualified" if adjusted >= 0.70 else "Watchlist" if adjusted >= 0.60 else "Avoid"
        reasons = [f"{home} attack {h.attack:.2f}x and {away} defence {a.defense:.2f}x league average", f"Elo: {h.elo:.0f} vs {a.elo:.0f}", f"Recent form: {h.form:.0%} vs {a.form:.0%}"]
        if quality == "Low": reasons.append("Limited historical sample; confidence penalized")
        return {**base, "recommended_market": market, "market_probability": probability, "adjusted_probability": adjusted, "fair_odds": 1 / probability if probability > 0 else None, "tier": tier, "data_quality": quality, "uncertainty_penalty": uncertainty, "home_elo": h.elo, "away_elo": a.elo, "home_form": h.form, "away_form": a.form, "home_matches": h.matches, "away_matches": a.matches, "reasons": reasons, "markets": markets}
