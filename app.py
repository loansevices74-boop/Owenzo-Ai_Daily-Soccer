import pandas as pd
import streamlit as st
from src.data_sources import LEAGUES, load_many
from src.model import SoccerModel
from src.accumulator import build_six_leg_accumulator

st.set_page_config(page_title="Owenzo AI Daily Soccer",page_icon="⚽",layout="wide")
st.title("⚽ Owenzo AI Daily Soccer")
st.caption("Keyless public-data model. Estimates are not guarantees or betting advice.")
with st.sidebar:
    season=st.text_input("Football-data season","2526")
    selected=st.multiselect("Leagues",list(LEAGUES),default=list(LEAGUES)[:10])
    threshold=st.slider("Confidence band",0.70,0.80,(0.70,0.80),0.01)
    run=st.button("Refresh public data",type="primary")
if run or "matches" not in st.session_state:
    with st.spinner("Loading public CSVs and fitting model…"):st.session_state.matches=load_many(selected,season) if selected else pd.DataFrame()
matches=st.session_state.matches
if matches.empty:st.warning("No public data loaded. Try another season or more leagues.");st.stop()
model=SoccerModel().fit(matches); latest=matches["date"].max().date(); rows=[]
for fixture in matches[matches["date"].dt.date==latest].itertuples():
    p=model.predict(fixture.home_team,fixture.away_team);p.update({"league":fixture.league,"date":str(latest)})
    if threshold[0]<=p["confidence"]<=threshold[1]:rows.append(p)
st.metric("Training through",str(model.as_of));st.metric("Matches in band",len(rows))
if rows:
    st.dataframe(pd.DataFrame([{**p,"confidence":f"{p['confidence']:.1%}","home_prob":f"{p['home_prob']:.1%}","draw_prob":f"{p['draw_prob']:.1%}","away_prob":f"{p['away_prob']:.1%}"} for p in rows])[["league","home_team","away_team","pick","confidence","home_prob","draw_prob","away_prob"]],use_container_width=True,hide_index=True)
else:st.info("No matches fall in the selected 70–80% band. No picks are invented.")
st.divider();st.subheader("Daily 10.00 estimated-odds accumulator")
acc=build_six_leg_accumulator(rows)
if acc:
    st.warning("Model-priced six-leg accumulator; public results do not provide live bookmaker odds.")
    st.write(f"Estimated combined odds: **{acc['estimated_combined_odds']:.2f}** | Estimated probability: **{acc['estimated_probability']:.1%}**")
    st.dataframe(pd.DataFrame([{"Match":f"{p['home_team']} vs {p['away_team']}","Pick":p["pick"],"Probability":f"{p['confidence']:.1%}","Estimated odds":p["estimated_odds"]} for p in acc["legs"]]),use_container_width=True,hide_index=True)
else:st.info("At least six qualifying matches are required for the accumulator.")
