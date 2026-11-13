import pandas as pd
from src.model import SoccerModel
from src.accumulator import build_six_leg_accumulator

def test_model_probabilities_sum_to_one():
    data=pd.DataFrame([{"date":"2025-01-01","home_team":"A","away_team":"B","home_goals":2,"away_goals":0},{"date":"2025-01-02","home_team":"B","away_team":"A","home_goals":1,"away_goals":1}])
    result=SoccerModel().fit(data).predict("A","B")
    assert abs(result["home_prob"]+result["draw_prob"]+result["away_prob"]-1)<1e-6

def test_accumulator_requires_six_legs():
    picks=[{"home_team":str(i),"away_team":"X","pick":"Home","confidence":0.72} for i in range(5)]
    assert build_six_leg_accumulator(picks) is None
