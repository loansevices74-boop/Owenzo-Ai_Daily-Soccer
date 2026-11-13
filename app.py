import pandas as pd
import streamlit as st
from datetime import date, timedelta
from src.data_sources import LEAGUES, load_fixtures, load_history
from src.model import SoccerModel
from src.accumulator import build_six_leg_accumulator

st.set_page_config(page_title="Owenzo AI Daily Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Owenzo AI Daily Soccer")
st.caption("Only real scheduled fixtures from public ESPN scoreboards. No simulated fixtures are created.")
with st.sidebar:
    selected = st.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES)[:10])
    threshold = st.slider("Confidence band", 0.70, 0.80, (0.70, 0.80), 0.01)
    view = st.radio("View", ["Today", "Next 7 days"])
    refresh = st.button("Refresh live fixture data", type="primary")

@st.cache_data(ttl=300, show_spinner=False)
def fetch(selected_leagues, start, end):
    return load_fixtures(list(selected_leagues), start, end)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(selected_leagues):
    return load_history(list(selected_leagues))

start = date.today()
end = start if view == "Today" else start + timedelta(days=6)
if refresh or "fixtures" not in st.session_state or st.session_state.get("fixture_key") != (tuple(selected), start, end):
    with st.spinner("Refreshing real public fixtures…"):
        st.session_state.fixtures = fetch(tuple(selected), start, end)
        st.session_state.fixture_key = (tuple(selected), start, end)

fixtures = st.session_state.fixtures
if fixtures.empty:
    st.warning("No real fixtures were returned for the selected leagues and dates. No simulated selections are shown.")
    st.stop()

history = fetch_history(tuple(selected))
model = SoccerModel().fit(history) if not history.empty else SoccerModel()
fixtures["fixture_day"] = fixtures["date"].dt.date
st.metric("Real fixtures found", len(fixtures))
st.caption(f"Showing {start.isoformat()} through {end.isoformat()} UTC; refreshed from public scoreboards.")

for fixture_day, day_frame in fixtures.groupby("fixture_day", sort=True):
    st.subheader(str(fixture_day))
    rows = []
    for fixture in day_frame.itertuples():
        prediction = model.predict(fixture.home_team, fixture.away_team)
        if threshold[0] <= prediction["confidence"] <= threshold[1]:
            prediction.update({"league": fixture.league, "kickoff": fixture.date.strftime("%H:%M UTC"), "status": fixture.status_detail, "fixture_id": fixture.fixture_id})
            rows.append(prediction)
    if rows:
        table = pd.DataFrame([{**p, "confidence": f"{p['confidence']:.1%}", "home_prob": f"{p['home_prob']:.1%}", "draw_prob": f"{p['draw_prob']:.1%}", "away_prob": f"{p['away_prob']:.1%}"} for p in rows])
        st.dataframe(table[["league", "kickoff", "home_team", "away_team", "pick", "confidence", "home_prob", "draw_prob", "away_prob", "status"]], use_container_width=True, hide_index=True)
    else:
        st.info("No real fixture in this day falls within the selected 70–80% band.")

st.divider()
st.subheader("Daily 10.00 estimated-odds accumulator")
if view == "Today":
    all_rows = []
    for fixture in fixtures.itertuples():
        p = model.predict(fixture.home_team, fixture.away_team)
        if threshold[0] <= p["confidence"] <= threshold[1]:
            p.update({"league": fixture.league, "fixture_id": fixture.fixture_id})
            all_rows.append(p)
    accumulator = build_six_leg_accumulator(all_rows)
    if accumulator:
        st.warning("Six real fixtures only. Combined odds are model estimates, not bookmaker prices.")
        st.write(f"Estimated combined odds: **{accumulator['estimated_combined_odds']:.2f}** | Estimated probability: **{accumulator['estimated_probability']:.1%}**")
        st.dataframe(pd.DataFrame([{"Match": f"{p['home_team']} vs {p['away_team']}", "Pick": p["pick"], "Probability": f"{p['confidence']:.1%}", "Estimated odds": p["estimated_odds"]} for p in accumulator["legs"]]), use_container_width=True, hide_index=True)
    else:
        st.info("No accumulator shown: at least six real daily fixtures in the selected probability band are required.")
else:
    st.info("The accumulator is available only in Today view and uses only today's real fixtures.")
