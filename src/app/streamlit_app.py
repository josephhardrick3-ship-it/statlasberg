"""Streamlit dashboard for March Madness Bot."""
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Statlasberg", layout="wide")
st.title("Statlasberg")

SCORES_PATH = "data/outputs/team_scores.csv"
CHAMPS_PATH = "data/outputs/simulation_results.csv"

@st.cache_data
def load_data():
    scores = pd.read_csv(SCORES_PATH) if os.path.exists(SCORES_PATH) else pd.DataFrame()
    champs = pd.read_csv(CHAMPS_PATH) if os.path.exists(CHAMPS_PATH) else pd.DataFrame()
    return scores, champs

scores, champs = load_data()

if len(scores) == 0:
    st.warning("No scores found. Run `python build_features.py && python score_teams.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Team Rankings", "Team Profiles", "Bracket Simulation"])

with tab1:
    st.subheader("Contender Rankings")
    display_cols = ["team","contender_score","upset_risk_score","archetype","expected_round",
                    "experience_score","defense_score","guard_play_score","clutch_score","net_rank"]
    available = [c for c in display_cols if c in scores.columns]
    st.dataframe(scores.sort_values("contender_score", ascending=False)[available], use_container_width=True)

with tab2:
    st.subheader("Team Deep Dive")
    team = st.selectbox("Select team", sorted(scores["team"].unique()))
    row = scores[scores["team"] == team].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Contender Score", f"{row.get('contender_score', 0)}/100")
    col2.metric("Upset Risk", f"{row.get('upset_risk_score', 0)}/100")
    col3.metric("Archetype", row.get("archetype", "N/A"))
    if "explanation_summary" in row.index:
        st.write(row["explanation_summary"])

with tab3:
    st.subheader("Championship Probabilities")
    if len(champs) > 0:
        st.bar_chart(champs.set_index("team")["championship_pct"].head(15))
    else:
        st.info("Run bracket simulation first: `python simulate_bracket_cli.py`")
