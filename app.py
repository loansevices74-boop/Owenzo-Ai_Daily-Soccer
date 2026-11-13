import pandas as pd
import streamlit as st
from datetime import date, timedelta
from src.data_sources import LEAGUES, load_fixtures_with_status, load_history
from src.model import SoccerModel
from src.accumulator import build_six_leg_accumulator

st.set_page_config(page_title="Owenzo AI Daily Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Owenzo AI Daily Soccer")
st.caption("Real scheduled/live fixtures only. Predictions are estimates, not guarantees.")

with st.sidebar:
    selected = st.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES)[:12])
    threshold = st.slider("Selection probability band", 0.70, 0.80, (0.70, 0.80), 0.01)
    view = st.radio("Fixture window", ["Today", "Next 7 days"])
    refresh = st.button("Clear cache and refresh", type="primary")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixtures(selected_leagues, start, end):
    return load_fixtures_with_status(list(selected_leagues), start, end)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(selected_leagues):
    return load_history(list(selected_leagues))

if refresh:
    st.cache_data.clear()
    st.session_state.pop("fixture_key", None)

start = date.today()
end = start if view == "Today" else start + timedelta(days=6)
key = (tuple(selected), start.isoformat(), end.isoformat())
if "fixture_key" not in st.session_state or st.session_state.fixture_key != key:
    with st.spinner("Loading real public fixtures…"):
        st.session_state.fixtures, st.session_state.source_status = fetch_fixtures(tuple(selected), start, end)
        st.session_state.fixture_key = key

fixtures = st.session_state.fixtures
source_status = st.session_state.source_status
st.metric("Real fixtures received", len(fixtures))
st.caption(f"UTC window: {start.isoformat()} to {end.isoformat()} | Source refresh cache: 5 minutes")

with st.expander("Source diagnostics"):
    diagnostic_rows = [{"League": league, "Status": "OK" if info["ok"] else "ERROR", "Events returned": info["fixtures"], "Error": info["error"]} for league, info in source_status.items()]
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

if fixtures.empty:
    st.warning("No fixtures were returned by the selected public scoreboard sources. No simulated fixtures or selections are shown.")
    st.info("Try selecting more leagues, switch to Next 7 days, or use Clear cache and refresh.")
    st.link_button("Open LiveScore", "https://www.livescore.com/en/football/")
    st.link_button("Open Forebet", "https://www.forebet.com/en/football-predictions")
    st.stop()

st.subheader("Real fixtures")
real_table = fixtures.copy()
real_table["Kickoff UTC"] = real_table["date"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(real_table[["Kickoff UTC", "league", "home_team", "away_team", "status_detail"]].rename(columns={"league":"League", "home_team":"Home", "away_team":"Away", "status_detail":"Status"}), use_container_width=True, hide_index=True)

history = fetch_history(tuple(selected))
model = SoccerModel().fit(history) if not history.empty else SoccerModel()
st.subheader("Model selections: 70–80% only")
selection_rows = []
for fixture in fixtures.itertuples():
    prediction = model.predict(fixture.home_team, fixture.away_team)
    if threshold[0] <= prediction["confidence"] <= threshold[1]:
        prediction.update({"league": fixture.league, "kickoff": fixture.date.strftime("%Y-%m-%d %H:%M UTC"), "status": fixture.status_detail, "fixture_id": fixture.fixture_id})
        selection_rows.append(prediction)
if selection_rows:
    table = pd.DataFrame([{**p, "confidence": f"{p['confidence']:.1%}", "home_prob": f"{p['home_prob']:.1%}", "draw_prob": f"{p['draw_prob']:.1%}", "away_prob": f"{p['away_prob']:.1%}"} for p in selection_rows])
    st.dataframe(table[["league", "kickoff", "home_team", "away_team", "pick", "confidence", "home_prob", "draw_prob", "away_prob", "status"]], use_container_width=True, hide_index=True)
else:
    st.info("Real fixtures are available, but none currently fall within the selected 70–80% probability band.")

st.divider()
st.subheader("Daily six-leg accumulator")
if view == "Today":
    accumulator = build_six_leg_accumulator(selection_rows)
    if accumulator:
        st.warning("Uses six real fixtures. Prices are model-implied estimates, not live bookmaker odds.")
        st.write(f"Estimated combined odds: **{accumulator['estimated_combined_odds']:.2f}** | Estimated probability: **{accumulator['estimated_probability']:.1%}**")
        st.dataframe(pd.DataFrame([{"Match": f"{p['home_team']} vs {p['away_team']}", "Pick": p["pick"], "Probability": f"{p['confidence']:.1%}", "Estimated odds": p["estimated_odds"]} for p in accumulator["legs"]]), use_container_width=True, hide_index=True)
    else:
        st.info("No accumulator shown: at least six real daily selections are required.")
else:
    st.info("The accumulator is available only in Today view and uses only today's real fixtures.")

st.caption("Additional reference sites: LiveScore and Forebet are linked for manual verification; their protected pages are not scraped.")
