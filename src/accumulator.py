from __future__ import annotations
from itertools import combinations

def build_six_leg_accumulator(predictions: list[dict], target_odds: float = 10.0, legs: int = 6) -> dict | None:
    candidates=[p.copy() for p in predictions if p.get("confidence",0)>=0.70]
    if len(candidates)<legs:return None
    for p in candidates:p["estimated_odds"]=round(1/p["confidence"],2)
    best=None
    for combo in combinations(sorted(candidates,key=lambda p:p["confidence"],reverse=True)[:14],legs):
        combined=1
        for p in combo:combined*=p["estimated_odds"]
        if best is None or abs(combined-target_odds)<best[0]:best=(abs(combined-target_odds),combo,combined)
    _,combo,combined=best
    return {"legs":list(combo),"estimated_combined_odds":round(combined,2),"estimated_probability":round(1/combined,4),"target_odds":target_odds}
