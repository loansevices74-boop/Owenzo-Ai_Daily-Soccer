import pandas as pd
import streamlit as st
from datetime import date, timedelta
from src.data_sources import LEAGUES, load_fixtures_with_status, load_history
from src.model import SoccerModel
from src.accumulator import build_six_leg_accumulator

st.set_page_config(page_title="Owenzo AI Daily Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Owenzo AI Daily Soccer")
st.caption("Real fixtures only. Selections combine Poisson scoring, Elo strength, form and data-quality checks.")
with st.sidebar:
    selected = st.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES)[:12])
    threshold = st.slider("Minimum adjusted probability", 0.60, 0.80, 0.70, 0.01)
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
start = date.today(); end = start if view == "Today" else start + timedelta(days=6)
key = (tuple(selected), start.isoformat(), end.isoformat())
if st.session_state.get("fixture_key") != key:
    with st.spinner("Loading real public fixtures and historical results…"):
        st.session_state.fixtures, st.session_state.source_status = fetch_fixtures(tuple(selected), start, end)
        st.session_state.fixture_key = key
fixtures = st.session_state.fixtures
st.metric("Real fixtures received", len(fixtures))
st.caption(f"UTC window: {start.isoformat()} to {end.isoformat()} | Fixture cache: 5 minutes")
with st.expander("Source diagnostics"):
    diagnostic_rows = [{"League": league, "Status": "OK" if info["ok"] else "ERROR", "Events returned": info["fixtures"], "Error": info["error"]} for league, info in st.session_state.source_status.items()]
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
if fixtures.empty:
    st.warning("No real fixtures were returned. No simulated fixtures or selections are shown.")
    st.stop()

real_table = fixtures.copy(); real_table["Kickoff UTC"] = real_table["date"].dt.strftime("%Y-%m-%d %H:%M")
st.subheader("Real fixtures")
st.dataframe(real_table[["Kickoff UTC", "league", "home_team", "away_team", "status_detail"]].rename(columns={"league":"League", "home_team":"Home", "away_team":"Away", "status_detail":"Status"}), use_container_width=True, hide_index=True)

history = fetch_history(tuple(selected)); model = SoccerModel().fit(history) if not history.empty else SoccerModel()
analysis = []
for fixture in fixtures.itertuples():
    item = model.analyze(fixture.home_team, fixture.away_team)
    item.update({"league": fixture.league, "kickoff": fixture.date.strftime("%Y-%m-%d %H:%M UTC"), "status": fixture.status_detail, "fixture_id": fixture.fixture_id})
    analysis.append(item)

st.subheader("Best model selections")
st.caption("A selection is shown only when its adjusted probability clears the threshold. The model can recommend safer markets such as double chance or goals.")
qualified = [item for item in analysis if item["adjusted_probability"] >= threshold and item["tier"] != "Avoid"]
if qualified:
    table = pd.DataFrame([{**p, "Probability": f"{p['adjusted_probability']:.1%}", "Fair odds": f"{p['fair_odds']:.2f}", "Model 1X2": f"{p['confidence']:.1%}"} for p in qualified])
    st.dataframe(table[["league", "kickoff", "home_team", "away_team", "recommended_market", "Probability", "Fair odds", "tier", "data_quality", "Model 1X2", "status"]].rename(columns={"league":"League", "kickoff":"Kickoff", "home_team":"Home", "away_team":"Away", "recommended_market":"Market", "tier":"Tier", "data_quality":"Data quality", "status":"Status"}), use_container_width=True, hide_index=True)
else:
    st.info("No real fixture clears the current threshold. No selection is invented.")

with st.expander("Selection details and reasons"):
    for item in sorted(qualified, key=lambda x: x["adjusted_probability"], reverse=True):
        st.markdown(f"**{item['home_team']} vs {item['away_team']} — {item['recommended_market']} ({item['adjusted_probability']:.1%})**")
        st.write(f"Tier: {item['tier']} | Data quality: {item['data_quality']} | Historical matches: {item['home_matches']}–{item['away_matches']} | xG: {item['home_xg']:.2f}–{item['away_xg']:.2f}")
        st.write(f"Home form: {item['home_form']:.1%} | Away form: {item['away_form']:.1%} | Elo: {item['home_elo']:.0f}–{item['away_elo']:.0f}")
        for reason in item["reasons"]: st.write(f"- {reason}")
        st.write("Market probabilities: " + " | ".join(f"{name}: {prob:.1%}" for name, prob in item["markets"].items()))

st.subheader("Watchlist")
watchlist = [item for item in analysis if 0.60 <= item["adjusted_probability"] < threshold]
if watchlist:
    st.dataframe(pd.DataFrame([{ "Match": f"{p['home_team']} vs {p['away_team']}", "Market": p["recommended_market"], "Probability": f"{p['adjusted_probability']:.1%}", "Tier": p["tier"], "Data quality": p["data_quality"] } for p in watchlist]), use_container_width=True, hide_index=True)
else: st.info("No watchlist matches in the current window.")

st.subheader("Daily six-leg accumulator")
if view == "Today":
    accumulator = build_six_leg_accumulator(qualified)
    if accumulator:
        st.warning("Six real fixtures only. Combined prices are model-implied estimates, not live bookmaker odds.")
        st.write(f"Estimated combined odds: **{accumulator['estimated_combined_odds']:.2f}** | Estimated probability: **{accumulator['estimated_probability']:.1%}**")
        st.dataframe(pd.DataFrame([{ "Match": f"{p['home_team']} vs {p['away_team']}", "Market": p.get("recommended_market", p.get("pick")), "Probability": f"{p['adjusted_probability']:.1%}", "Fair odds": round(p['fair_odds'], 2) } for p in accumulator["legs"]]), use_container_width=True, hide_index=True)
    else: st.info("No accumulator shown: at least six real qualified fixtures are required.")
else: st.info("The accumulator is available only in Today view and uses only today's real fixtures.")
st.caption("LiveScore and Forebet remain external reference links; protected pages are not scraped.")
