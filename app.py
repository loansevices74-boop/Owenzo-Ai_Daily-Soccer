import pandas as pd
import streamlit as st
from datetime import date, timedelta
from src.data_sources import LEAGUES, load_fixtures_with_status, load_history
from src.model import SoccerModel
from src.reliability import walk_forward, reliability_summary, reliability_table
from src.accumulator import build_six_leg_accumulator

st.set_page_config(page_title="Owenzo AI Daily Soccer", page_icon="⚽", layout="wide")
st.title("⚽ Owenzo AI Daily Soccer")
st.caption("Real fixtures only. Predictions are estimates; reliability is measured separately from probability.")
with st.sidebar:
    selected = st.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES)[:12])
    threshold = st.slider("Minimum adjusted probability", 0.60, 0.80, 0.70, 0.01)
    view = st.radio("Fixture window", ["Today", "Next 7 days"])
    backtest_days = st.selectbox("Reliability window", [90, 180, 365], index=2)
    refresh = st.button("Clear cache and refresh", type="primary")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixtures(selected_leagues, start, end):
    return load_fixtures_with_status(list(selected_leagues), start, end)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(selected_leagues):
    return load_history(list(selected_leagues))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_backtest(selected_leagues, days):
    history = load_history(list(selected_leagues))
    return walk_forward(history, min_training_matches=30, max_days=days)

if refresh:
    st.cache_data.clear(); st.session_state.pop("fixture_key", None)
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
    rows = [{"League": k, "Status": "OK" if v["ok"] else "ERROR", "Events returned": v["fixtures"], "Error": v["error"]} for k, v in st.session_state.source_status.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
st.caption("Selections require the probability threshold. Reliability below is historical and does not guarantee future results.")
qualified = [x for x in analysis if x["adjusted_probability"] >= threshold and x["tier"] != "Avoid"]
if qualified:
    table = pd.DataFrame([{**p, "Probability": f"{p['adjusted_probability']:.1%}", "Fair odds": f"{p['fair_odds']:.2f}"} for p in qualified])
    st.dataframe(table[["league", "kickoff", "home_team", "away_team", "recommended_market", "Probability", "Fair odds", "tier", "data_quality", "status"]].rename(columns={"league":"League", "kickoff":"Kickoff", "home_team":"Home", "away_team":"Away", "recommended_market":"Market", "tier":"Tier", "data_quality":"Data quality", "status":"Status"}), use_container_width=True, hide_index=True)
else: st.info("No real fixture clears the current threshold. No selection is invented.")

with st.expander("Selection details and reasons"):
    for item in sorted(qualified, key=lambda x: x["adjusted_probability"], reverse=True):
        st.markdown(f"**{item['home_team']} vs {item['away_team']} — {item['recommended_market']} ({item['adjusted_probability']:.1%})**")
        st.write(f"Tier: {item['tier']} | Data quality: {item['data_quality']} | Historical matches: {item['home_matches']}–{item['away_matches']} | xG: {item['home_xg']:.2f}–{item['away_xg']:.2f}")
        st.write(f"Home form: {item['home_form']:.1%} | Away form: {item['away_form']:.1%} | Elo: {item['home_elo']:.0f}–{item['away_elo']:.0f}")
        for reason in item["reasons"]: st.write(f"- {reason}")
        st.write("Market probabilities: " + " | ".join(f"{n}: {p:.1%}" for n, p in item["markets"].items()))

st.subheader("Watchlist")
watchlist = [x for x in analysis if 0.60 <= x["adjusted_probability"] < threshold]
if watchlist:
    st.dataframe(pd.DataFrame([{ "Match": f"{p['home_team']} vs {p['away_team']}", "Market": p["recommended_market"], "Probability": f"{p['adjusted_probability']:.1%}", "Tier": p["tier"], "Data quality": p["data_quality"] } for p in watchlist]), use_container_width=True, hide_index=True)
else: st.info("No watchlist matches in the current window.")

st.divider(); st.subheader("Model reliability")
st.caption("Walk-forward evaluation trains only on matches before each historical prediction date. Results are descriptive, not predictive.")
if history.empty:
    st.info("Reliability unavailable because historical results could not be loaded.")
else:
    with st.spinner("Running leakage-safe historical evaluation…"):
        backtest = fetch_backtest(tuple(selected), backtest_days)
    summary = reliability_summary(backtest)
    if summary["count"] < 50:
        st.warning(f"Only {summary['count']} walk-forward predictions are available. Reliability is not established until the sample is larger.")
    else:
        cols = st.columns(5)
        cols[0].metric("Predictions", summary["count"])
        cols[1].metric("Hit rate", f"{summary['accuracy']:.1%}")
        cols[2].metric("Brier score", f"{summary['brier']:.3f}")
        cols[3].metric("Log loss", f"{summary['log_loss']:.3f}")
        cols[4].metric("Calibration gap", f"{summary['calibration_gap']:+.1%}")
        st.dataframe(reliability_table(backtest).style.format({"Average probability":"{:.1%}", "Actual hit rate":"{:.1%}", "Calibration gap":"{:+.1%}"}), use_container_width=True, hide_index=True)
        by_market = backtest.groupby("market").agg(Predictions=("outcome", "size"), HitRate=("outcome", "mean"), Brier=("brier", "mean")).reset_index()
        st.dataframe(by_market.style.format({"HitRate":"{:.1%}", "Brier":"{:.3f}"}), use_container_width=True, hide_index=True)
        if summary["calibration_gap"] < -0.05: st.error("Recent evaluation indicates model overconfidence. Selections should be treated cautiously.")
        elif summary["calibration_gap"] > 0.05: st.info("The model is conservative in this evaluation window.")

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
