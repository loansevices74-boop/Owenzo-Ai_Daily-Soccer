"""Leakage-safe walk-forward evaluation and calibration metrics."""
from __future__ import annotations
import math
import pandas as pd
from src.model import SoccerModel


def _outcome(row, market):
    home, away = row.home_goals, row.away_goals
    return {
        "Home": home > away, "Draw": home == away, "Away": away > home,
        "Home or Draw": home >= away, "Away or Draw": away >= home,
        "Over 1.5 Goals": home + away >= 2, "Under 3.5 Goals": home + away <= 3,
        "Both Teams To Score": home > 0 and away > 0,
    }.get(market, False)


def walk_forward(matches: pd.DataFrame, min_training_matches: int = 30, max_days: int | None = 365) -> pd.DataFrame:
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.dropna(subset=list(required)).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    if max_days:
        cutoff = frame["date"].max() - pd.Timedelta(days=max_days)
        prediction_dates = frame.loc[frame["date"] >= cutoff, "date"].dt.date.unique()
    else:
        prediction_dates = frame["date"].dt.date.unique()
    records = []
    for prediction_date in prediction_dates:
        train = frame[frame["date"].dt.date < prediction_date]
        test = frame[frame["date"].dt.date == prediction_date]
        if len(train) < min_training_matches:
            continue
        model = SoccerModel().fit(train)
        for row in test.itertuples():
            result = model.analyze(row.home_team, row.away_team)
            market = result["recommended_market"]
            probability = float(result["market_probability"])
            records.append({"date": prediction_date, "league": getattr(row, "league", "Unknown"), "home_team": row.home_team, "away_team": row.away_team, "market": market, "probability": probability, "outcome": int(_outcome(row, market)), "brier": (probability - int(_outcome(row, market))) ** 2, "model_version": "poisson-elo-form-v2"})
    return pd.DataFrame(records)


def reliability_summary(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"count": 0, "accuracy": None, "brier": None, "log_loss": None, "calibration_gap": None}
    p = predictions["probability"].clip(0.001, 0.999)
    y = predictions["outcome"].astype(float)
    return {"count": len(predictions), "accuracy": float(y.mean()), "brier": float(((p - y) ** 2).mean()), "log_loss": float(-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log)).mean()), "calibration_gap": float(y.mean() - p.mean()), "average_probability": float(p.mean())}


def reliability_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["Band", "Predictions", "Average probability", "Actual hit rate", "Calibration gap"])
    frame = predictions.copy()
    frame["Band"] = pd.cut(frame["probability"], [0.0, 0.6, 0.7, 0.8, 1.0], labels=["<60%", "60–69%", "70–79%", "80%+"])
    return frame.groupby("Band", observed=False).agg(Predictions=("outcome", "size"), **{"Average probability": ("probability", "mean"), "Actual hit rate": ("outcome", "mean")}).reset_index().assign(**{"Calibration gap": lambda x: x["Actual hit rate"] - x["Average probability"]})
