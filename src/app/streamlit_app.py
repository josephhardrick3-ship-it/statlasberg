"""Statlasberg — 2026 March Madness Intelligence Platform"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import os
import json
import math
import time as _time
from datetime import datetime
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

st.set_page_config(
    page_title="Statlasberg",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Center + constrain content */
    .block-container {
        max-width: 1380px !important;
        margin: 0 auto !important;
        padding-top: 0.6rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Main background */
    .stApp { background-color: #0e1117; }
    body { color: #f1f5f9; }

    /* Metric cards — compact */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="metric-container"] label {
        color: #b0bbd0 !important;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.4rem;
        font-weight: 800;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
        font-size: 0.72rem;
    }

    /* Section headers */
    .section-header {
        color: #ff8c3a;
        font-size: 1.05rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 5px 0;
        border-bottom: 2px solid #f97316;
        margin-bottom: 10px;
    }

    /* Team card */
    .team-card {
        background: linear-gradient(135deg, #1a1f2e, #1e2536);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 5px 0;
    }
    .team-card:hover { border-color: #f97316; }
    .team-rank  { color: #f97316; font-size: 1.3rem; font-weight: 800; }
    .team-name  { color: #f1f5f9; font-size: 1.05rem; font-weight: 700; }
    .team-score { color: #4ade80; font-size: 0.95rem; font-weight: 600; }

    /* Matchup card — compact bracket card */
    .matchup-card {
        background: linear-gradient(135deg, #131820, #1a2132);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 8px 10px;
        margin: 4px 0;
    }
    .matchup-card:hover { border-color: #f97316; }

    /* Seed badge */
    .seed-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #f97316;
        color: white;
        border-radius: 50%;
        width: 24px; height: 24px;
        font-weight: 800;
        font-size: 0.72rem;
        margin-right: 7px;
        flex-shrink: 0;
    }

    /* Flag badges */
    .flag-cinderella { background:#d97706; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
    .flag-fraud      { background:#dc2626; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
    .flag-dark       { background:#7c3aed; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
    .flag-upset      { background:#ea580c; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
    .flag-hot        { background:#16a34a; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }

    /* Progress bar override */
    .stProgress > div > div { background-color: #f97316; }

    /* Tabs — high contrast */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e;
        border-radius: 8px 8px 0 0;
        color: #b0bbd0;
        font-weight: 600;
        padding: 7px 18px;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #f97316 !important;
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #2d3748; }

    /* Dataframe text */
    .stDataFrame { color: #f1f5f9; }

    /* Header text overrides */
    h1, h2, h3 { color: #f1f5f9 !important; }
    p, li { color: #d1d9e6; }
    small, .stCaption { color: #94a3b8 !important; }

    /* Info/warning boxes */
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions (module-level) ───────────────────────────────────────────
HIST_SEED_WIN_PCT = {
    (1,16): 0.987, (16,1): 0.013,
    (8,9):  0.513, (9,8):  0.487,
    (5,12): 0.649, (12,5): 0.351,
    (4,13): 0.793, (13,4): 0.207,
    (6,11): 0.630, (11,6): 0.370,
    (3,14): 0.851, (14,3): 0.149,
    (7,10): 0.601, (10,7): 0.399,
    (2,15): 0.940, (15,2): 0.060,
}

def win_prob_sigmoid(score_a, score_b):
    """Logistic win probability based on contender score gap."""
    diff = float(score_a or 50) - float(score_b or 50)
    return 1 / (1 + np.exp(-diff / 8))

def american_line(p, round5=True):
    """Convert probability to American moneyline odds string."""
    p = max(0.01, min(0.99, float(p)))
    if p >= 0.5:
        raw = int(p / (1 - p) * 100)
        return f"-{int(round(raw/5)*5)}" if round5 else f"-{raw}"
    else:
        raw = int((1 - p) / p * 100)
        return f"+{int(round(raw/5)*5)}" if round5 else f"+{raw}"

def hot_label(row_s):
    """Return momentum label based on last10 stats."""
    try:
        l10   = float(row_s.get("last10_win_pct", 0.5) or 0.5)
        l10m  = float(row_s.get("last10_adj_margin", 0) or 0)
        seasn = float(row_s.get("adj_margin", 0) or 0)
    except Exception:
        return ""
    if l10 >= 0.9:             return "🔥 HOT"
    if l10 >= 0.8:             return "🔥 Hot"
    if l10m > seasn + 5:       return "📈 Trending"
    if l10 <= 0.3:             return "❄️ Cold"
    return ""

def style_matchup_insight(t1, t2):
    """One-line style matchup narrative."""
    def fv(r, k, d=0.0):
        try: return float(r.get(k, d) or d)
        except: return d
    pace1, pace2 = fv(t1,"tempo",68), fv(t2,"tempo",68)
    thr_off2_raw = fv(t2,"three_pt_pct",33)
    thr_off2 = thr_off2_raw if thr_off2_raw > 1 else thr_off2_raw*100
    thr_def1 = fv(t1,"opp_three_pt_pct",33)
    guard1, guard2 = fv(t1,"guard_play_score",50), fv(t2,"guard_play_score",50)
    def1, def2 = fv(t1,"defense_score",50), fv(t2,"defense_score",50)
    n1, n2 = t1.get("team","Team A"), t2.get("team","Team B")

    if abs(pace1 - pace2) > 5:
        faster = n1 if pace1 > pace2 else n2
        slower = n2 if pace1 > pace2 else n1
        return f"⚡ Pace war — {faster} wants to run, {slower} wants to grind"
    if thr_off2 > 38 and thr_def1 < 32:
        return f"🎯 {n2}'s 3PT attack ({thr_off2:.0f}%) exploits {n1}'s porous perimeter D"
    if abs(guard1 - guard2) > 12:
        edge = n1 if guard1 > guard2 else n2
        return f"🏀 Clear backcourt edge for {edge} — guard play decides this one"
    if abs(def1 - def2) > 15:
        def_team = n1 if def1 > def2 else n2
        return f"🛡️ {def_team}'s elite defense controls the tempo"
    return "⚖️ Balanced matchup — execution and momentum win this one"

def safe_f(v, d=0.0):
    try: return float(v) if pd.notna(v) and v != '' else d
    except: return d

def safe_i(v, d=0):
    try: return int(float(v)) if pd.notna(v) and v != '' else d
    except: return d

def simulate_bracket_full(bkt_df):
    """Deterministic bracket simulation using win_prob_sigmoid on contender scores.
    Returns (sim_rounds, s16_teams, e8_teams, ff_teams, champion, runner_up).
    Each team gets exactly one exit label — no duplicates, no phantom Final Fours.
    """
    score_lkp = {r["team"]: safe_f(r.get("contender_score", 50)) for _, r in bkt_df.iterrows()}

    # Each region pod: two R1 pairs collapse into one R2 winner → S16 candidate
    PODS = [
        ([(1,16),(8,9)],  [(5,12),(4,13)]),   # top half of region
        ([(6,11),(3,14)], [(7,10),(2,15)]),   # bottom half of region
    ]

    def pgame(t1, t2):
        """Returns (winner, loser). Empty string if opponent is missing."""
        if not t1: return t2, ""
        if not t2: return t1, ""
        p = win_prob_sigmoid(score_lkp.get(t1, 50), score_lkp.get(t2, 50))
        return (t1, t2) if p >= 0.5 else (t2, t1)

    sim_rounds = {}
    s16_teams, e8_teams, ff_teams = [], [], []

    for region in ["East", "South", "West", "Midwest"]:
        reg = bkt_df[bkt_df["region"] == region]
        s2t = {int(r["seed"]): r["team"] for _, r in reg.iterrows()}

        region_s16 = []   # 2 S16 winners per region → play E8 game

        for (pod_a_pairs, pod_b_pairs) in PODS:
            def pod_winner(pairs):
                r1w = []
                for s1, s2 in pairs:
                    t1, t2 = s2t.get(s1,""), s2t.get(s2,"")
                    if t1 or t2:
                        w, l = pgame(t1, t2)
                        if l: sim_rounds[l] = "First Round"
                        if w: r1w.append(w)
                if len(r1w) == 2:
                    w, l = pgame(r1w[0], r1w[1])
                    if l: sim_rounds[l] = "Second Round"
                    return w
                return r1w[0] if r1w else ""

            pod_a_s16 = pod_winner(pod_a_pairs)
            pod_b_s16 = pod_winner(pod_b_pairs)

            # S16 game: pod A winner vs pod B winner
            if pod_a_s16 and pod_b_s16:
                s16_teams.extend([pod_a_s16, pod_b_s16])
                w, l = pgame(pod_a_s16, pod_b_s16)
                if l: sim_rounds[l] = "Sweet 16"
                if w: region_s16.append(w)
            elif pod_a_s16:
                s16_teams.append(pod_a_s16); region_s16.append(pod_a_s16)
            elif pod_b_s16:
                s16_teams.append(pod_b_s16); region_s16.append(pod_b_s16)

        # Elite 8 game (regional final)
        if len(region_s16) == 2:
            e8_teams.extend(region_s16)
            w, l = pgame(region_s16[0], region_s16[1])
            if l: sim_rounds[l] = "Elite 8"
            if w:
                ff_teams.append(w)
                sim_rounds[w] = "Final Four"   # tentative; may become Champion/Runner-Up
        elif region_s16:
            ff_teams.append(region_s16[0])
            sim_rounds[region_s16[0]] = "Final Four"

    # Final Four (East vs West, South vs Midwest — standard bracket pairing)
    champ_game_teams = []
    ff_pairs = [(0, 2), (1, 3)]  # East(0) vs West(2), South(1) vs Midwest(3)
    for ia, ib in ff_pairs:
        ta = ff_teams[ia] if ia < len(ff_teams) else ""
        tb = ff_teams[ib] if ib < len(ff_teams) else ""
        if ta and tb:
            w, l = pgame(ta, tb)
            if l: sim_rounds[l] = "Final Four"
            if w: champ_game_teams.append(w)
        elif ta: champ_game_teams.append(ta)
        elif tb: champ_game_teams.append(tb)

    # Championship
    champion, runner_up = "", ""
    if len(champ_game_teams) == 2:
        w, l = pgame(champ_game_teams[0], champ_game_teams[1])
        champion, runner_up = w, l
        sim_rounds[runner_up] = "Runner-Up"
        sim_rounds[champion]  = "Champion"
    elif champ_game_teams:
        champion = champ_game_teams[0]
        sim_rounds[champion] = "Champion"

    return sim_rounds, s16_teams, e8_teams, ff_teams, champion, runner_up

# ── Data loading ──────────────────────────────────────────────────────────────
SCORES_PATH  = "data/outputs/team_scores.csv"
CHAMPS_PATH  = "data/outputs/simulation_results.csv"
BRACKET_PATH = "data/brackets/bracket_2026.csv"
COACHES_PATH = "data/coaches_2026.csv"
RESULTS_PATH = "data/tournament_2026/results.csv"

COACH_COLS = ["team","coach","coach_years_at_school","coach_ncaa_games",
              "coach_sweet16s","coach_finalfours","first_year_coach_flag"]

@st.cache_data
def load_data():
    scores  = pd.read_csv(SCORES_PATH)  if os.path.exists(SCORES_PATH)  else pd.DataFrame()
    champs  = pd.read_csv(CHAMPS_PATH)  if os.path.exists(CHAMPS_PATH)  else pd.DataFrame()
    bracket = pd.read_csv(BRACKET_PATH) if os.path.exists(BRACKET_PATH) else pd.DataFrame()
    # Patch in 2026 coach data (scraper doesn't pull coaches)
    if os.path.exists(COACHES_PATH) and len(scores) > 0:
        coaches = pd.read_csv(COACHES_PATH)
        # Drop existing coach columns so we can replace them cleanly
        drop_cols = [c for c in COACH_COLS[1:] if c in scores.columns]
        if drop_cols:
            scores = scores.drop(columns=drop_cols)
        scores = scores.merge(coaches[COACH_COLS], on="team", how="left")
    # Deduplicate — some schools (Miami, Loyola) appear multiple times; keep the
    # row with the highest contender_score so we always get the right D1 team.
    if "contender_score" in scores.columns and len(scores) > 0:
        scores = (scores.sort_values("contender_score", ascending=False)
                        .drop_duplicates(subset="team", keep="first")
                        .reset_index(drop=True))
    return scores, champs, bracket

scores, champs, bracket = load_data()

if len(scores) == 0:
    st.error("⚠️ No scores found. Run the pipeline first.")
    st.stop()

# Merge bracket seeds into scores
if len(bracket) > 0:
    bracket_info = bracket[["team","region","seed"]].copy()
    scores = scores.merge(bracket_info, on="team", how="left")
    in_bracket = scores[scores["seed"].notna()].copy()
    in_bracket["seed"] = in_bracket["seed"].astype(int)
else:
    in_bracket = scores.copy()

# ── Deterministic bracket simulation ─────────────────────────────────────────
# Replace the pipeline's probabilistic expected_round with exact bracket picks.
# Exactly: 32 first-round winners, 16 Sweet 16, 8 Elite 8, 4 Final Four, 1 champ.
if len(in_bracket) > 0 and "region" in in_bracket.columns:
    sim_rounds, sim_s16, sim_e8, sim_ff, sim_champion, sim_runner_up = \
        simulate_bracket_full(in_bracket)
    in_bracket = in_bracket.copy()
    in_bracket["sim_round"] = in_bracket["team"].map(sim_rounds).fillna("First Round")
else:
    sim_rounds, sim_s16, sim_e8, sim_ff, sim_champion, sim_runner_up = {}, [], [], [], "", ""
    in_bracket["sim_round"] = "First Round"

# Results tracker (persistent via session state + CSV)
if "results" not in st.session_state:
    if os.path.exists(RESULTS_PATH):
        st.session_state.results = pd.read_csv(RESULTS_PATH).to_dict("records")
    else:
        st.session_state.results = []
results_dict = {r["matchup"]: r for r in st.session_state.results}

# Session state for deep-dive navigation and chat
if "dive_team" not in st.session_state:
    st.session_state.dive_team = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 Statlasberg")
    st.markdown("*2026 March Madness Intelligence*")
    st.divider()
    bracket_teams = sorted(in_bracket["team"].tolist()) if len(in_bracket) > 0 else sorted(scores["team"].tolist())
    selected_team = st.selectbox("🔍 Team Deep Dive", bracket_teams, key="team_selectbox")
    st.divider()
    if len(in_bracket) > 0:
        st.markdown("**📊 Model Summary**")
        n1s = in_bracket[in_bracket["seed"]==1]["contender_score"].mean() if "seed" in in_bracket else 0
        st.metric("Avg #1 Seed Score",  f"{n1s:.1f}" if n1s else "—")
        top_up = in_bracket[(in_bracket["seed"]>=10)&(in_bracket["seed"]<=12)].sort_values("contender_score", ascending=False)
        if len(top_up) > 0: st.metric("Best Upset Threat", top_up.iloc[0]["team"])
        st.metric("No. 1 Overall", in_bracket.sort_values("contender_score", ascending=False).iloc[0]["team"])
    # Model record
    total_res = len(st.session_state.results)
    if total_res > 0:
        st.divider()
        correct_res = sum(1 for r in st.session_state.results if r.get("model_pick") == r.get("winner"))
        acc = correct_res / total_res * 100
        st.metric("📊 Picks Record", f"{correct_res}-{total_res-correct_res}", f"{acc:.0f}% accuracy")
    st.divider()
    st.caption("Built on Selection Sunday · Mar 15, 2026")

# ── Compact Header ────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
with col_h1:
    st.metric("🏆 Model Favourite",
              champs.iloc[0]["team"] if len(champs)>0 else "—",
              f"{champs.iloc[0]['championship_pct']:.1f}%" if len(champs)>0 else "")
with col_h2:
    top = in_bracket.sort_values("contender_score", ascending=False).iloc[0] if len(in_bracket)>0 else None
    st.metric("📈 Top Score", f"{top['contender_score']:.1f}" if top is not None else "—", top["team"] if top is not None else "")
with col_h3:
    snubs_df = scores[(scores["contender_score"]>=65) & (~scores["team"].isin(bracket["team"]))] if len(bracket)>0 else pd.DataFrame()
    st.metric("🚨 Big Snubs", len(snubs_df), "model loves, left out")
with col_h4:
    upsets_df = in_bracket[(in_bracket["seed"]>=10) & (in_bracket["contender_score"]>=65)] if "seed" in in_bracket.columns else pd.DataFrame()
    st.metric("💥 Upset Alerts", len(upsets_df), "double-digit seeds w/ top scores")
with col_h5:
    avg_risk = in_bracket[in_bracket["seed"]==1]["upset_risk_score"].mean() if "seed" in in_bracket.columns else 0
    st.metric("⚠️ Avg #1 Seed Risk", f"{avg_risk:.1f}" if avg_risk else "—", "lower = safer")
st.markdown("---")

# ── Navigation banner after bracket deep-dive button clicked ──────────────────
if st.session_state.dive_team:
    st.info(f"📌 **{st.session_state.dive_team}** selected — click the **🔍 Team Deep Dive** tab above to view their full analysis.", icon="👆")
    if st.button("✖ Clear selection", key="clear_dive"):
        st.session_state.dive_team = None
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏅 Rankings", "🎯 Sweet 16 Picks", "🔍 Team Deep Dive",
    "📊 Model vs Committee", "🎲 Championship Odds", "🏆 Bracket", "📺 Live",
    "🤖 Ask Statlasberg"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">2026 Tournament Field — Model Rankings</div>', unsafe_allow_html=True)

    display_df = in_bracket.sort_values("contender_score", ascending=False).reset_index(drop=True)
    display_df.index += 1

    # Add hot/trend column
    if "last10_win_pct" in display_df.columns:
        display_df["trend"] = display_df.apply(hot_label, axis=1)

    def score_color(v):
        if v >= 70: return "background-color: #14532d; color: #86efac; font-weight:700"
        if v >= 60: return "background-color: #1a3a1a; color: #4ade80; font-weight:700"
        if v >= 50: return "background-color: #1c2a1c; color: #86efac"
        return "background-color: #1a1f2e; color: #cbd5e1"

    def risk_color(v):
        if v >= 40: return "background-color: #450a0a; color: #fca5a5; font-weight:700"
        if v >= 30: return "background-color: #431407; color: #fdba74; font-weight:700"
        return "background-color: #1a1f2e; color: #86efac"

    cols_show = ["team","region","seed","trend","contender_score","upset_risk_score",
                 "sim_round","defense_score","clutch_score","guard_play_score","archetype"]
    cols_avail = [c for c in cols_show if c in display_df.columns]
    fmt_cols = {c: "{:.1f}" for c in ["contender_score","upset_risk_score","defense_score","clutch_score","guard_play_score"] if c in cols_avail}

    styler = display_df[cols_avail].style
    if "contender_score" in cols_avail: styler = styler.map(score_color, subset=["contender_score"])
    if "upset_risk_score" in cols_avail: styler = styler.map(risk_color, subset=["upset_risk_score"])
    styled = styler.format(fmt_cols).set_properties(**{"background-color":"#0e1117","color":"#f1f5f9","border":"1px solid #374151"})
    st.dataframe(styled, use_container_width=True, height=560)

    # Flags summary
    st.markdown("---")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown('<div class="section-header">💥 Cinderella Candidates</div>', unsafe_allow_html=True)
        cinder = in_bracket[in_bracket["cinderella_flag"]==True] if "cinderella_flag" in in_bracket.columns else pd.DataFrame()
        if len(cinder) > 0:
            for _, r in cinder.iterrows():
                st.markdown(f'<span class="seed-badge">{int(r.get("seed","?"))}</span> <strong style="color:#f1f5f9">{r["team"]}</strong> <span style="color:#4ade80">{r["contender_score"]:.1f}</span>', unsafe_allow_html=True)
        else:
            st.caption("None flagged")
    with fc2:
        st.markdown('<div class="section-header">🃏 Fraud Favorites</div>', unsafe_allow_html=True)
        frauds = in_bracket[in_bracket["fraud_favorite_flag"]==True] if "fraud_favorite_flag" in in_bracket.columns else pd.DataFrame()
        if len(frauds) > 0:
            for _, r in frauds.iterrows():
                st.markdown(f'<span class="seed-badge">{int(r.get("seed","?"))}</span> <strong style="color:#f1f5f9">{r["team"]}</strong> <span style="color:#f87171">{r["contender_score"]:.1f}</span>', unsafe_allow_html=True)
        else:
            st.caption("None flagged")
    with fc3:
        st.markdown('<div class="section-header">🌑 Darkhorses</div>', unsafe_allow_html=True)
        dark = in_bracket[in_bracket["title_darkhorse_flag"]==True] if "title_darkhorse_flag" in in_bracket.columns else pd.DataFrame()
        if len(dark) > 0:
            for _, r in dark.head(5).iterrows():
                st.markdown(f'<span class="seed-badge">{int(r.get("seed","?"))}</span> <strong style="color:#f1f5f9">{r["team"]}</strong> <span style="color:#c084fc">{r["contender_score"]:.1f}</span>', unsafe_allow_html=True)
        else:
            st.caption("None flagged")
    with fc4:
        st.markdown('<div class="section-header">🔥 Hot Teams</div>', unsafe_allow_html=True)
        if "last10_win_pct" in in_bracket.columns:
            hot_teams = in_bracket[in_bracket["last10_win_pct"] >= 0.8].sort_values("last10_win_pct", ascending=False).head(6)
            for _, r in hot_teams.iterrows():
                l10 = safe_f(r.get("last10_win_pct"))
                st.markdown(f'<span class="seed-badge" style="background:#16a34a">{int(r.get("seed","?"))}</span> <strong style="color:#f1f5f9">{r["team"]}</strong> <span style="color:#4ade80">{l10*100:.0f}% L10</span>', unsafe_allow_html=True)
        else:
            st.caption("No trend data available")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SWEET 16 PICKS
# ─────────────────────────────────────────────────────────────────────────────
def path_weight(seed):
    if seed <= 4: return 1.0
    if seed <= 6: return 0.88
    if seed <= 8: return 0.75
    if seed <= 11: return 0.60
    return 0.45

with tab2:
    st.markdown('<div class="section-header">Model\'s Full Bracket Simulation</div>', unsafe_allow_html=True)
    st.caption("The model plays out every game using contender scores. Upsets happen when a lower seed out-scores their opponent. Exactly 16 Sweet 16, 8 Elite 8, 4 Final Four, 1 champion.")

    if "seed" not in in_bracket.columns or len(in_bracket) == 0 or not sim_s16:
        st.warning("Bracket simulation not available — check bracket data.")
    else:
        # ── Sweet 16 ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Sweet 16 — Model\'s 16 Picks</div>', unsafe_allow_html=True)
        regions = ["East","South","West","Midwest"]
        reg_cols = st.columns(4)
        all_s16_picks = []
        upset_teams_s16 = set()  # track any seeds ≥ 9 that made it

        for i, region in enumerate(regions):
            reg_s16 = in_bracket[in_bracket["team"].isin(sim_s16) & (in_bracket["region"]==region)]
            all_s16_picks.extend(reg_s16["team"].tolist())
            with reg_cols[i]:
                st.markdown(f'<div class="section-header">{region}</div>', unsafe_allow_html=True)
                for _, row in reg_s16.sort_values("contender_score", ascending=False).iterrows():
                    seed_val = int(row["seed"])
                    score = safe_f(row.get("contender_score", 50))
                    risk  = safe_f(row.get("upset_risk_score", 25))
                    hot   = hot_label(row)
                    flags = []
                    if seed_val >= 9:
                        flags.append("💥 Upset!")
                        upset_teams_s16.add(row["team"])
                    if row.get("cinderella_flag", False): flags.append("🪄")
                    if row.get("title_darkhorse_flag", False): flags.append("🌑")
                    if hot: flags.append(hot)
                    flag_span = f"<span style='margin-left:6px;font-size:0.78rem'>{' '.join(flags)}</span>" if flags else ""
                    badge_bg = "#dc2626" if seed_val >= 9 else "#374151"
                    bar_pct  = min(100, int(score))
                    bar_col  = "#4ade80" if score >= 65 else "#f59e0b" if score >= 55 else "#ef4444"
                    st.markdown(
                        f'<div class="team-card">'
                        f'<span class="seed-badge" style="background:{badge_bg}">{seed_val}</span> '
                        f'<strong class="team-name">{row["team"]}</strong>{flag_span}'
                        f'<br/><small style="color:#94a3b8">Score: {score:.1f} · Risk: {risk:.1f}</small>'
                        f'<div style="background:#2d3748;border-radius:4px;margin-top:5px;height:5px">'
                        f'<div style="width:{bar_pct}%;background:{bar_col};height:5px;border-radius:4px"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True)

        if upset_teams_s16:
            st.info(f"🚨 **Simulated upsets reaching Sweet 16:** {', '.join(sorted(upset_teams_s16))}")
        st.markdown(f"**Model's Sweet 16:** {', '.join(all_s16_picks)}")

        # ── Elite 8 ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🏆 Elite 8 — Model\'s Final 8</div>', unsafe_allow_html=True)
        st.caption("2 teams per region survive the Sweet 16 games. Updates automatically as results are logged.")

        eliminated = set()
        for res_r in st.session_state.get("results", []):
            w = res_r.get("winner", "")
            for t in res_r.get("teams", []):
                if t and t != w: eliminated.add(t)

        e8_cols = st.columns(4)
        all_e8_picks = []
        for ei, (region, ecol) in enumerate(zip(regions, e8_cols)):
            reg_e8 = in_bracket[in_bracket["team"].isin(sim_e8) & (in_bracket["region"]==region)]
            # If some E8 teams have been eliminated by logged results, replace with next-best S16 team
            active_e8 = reg_e8[~reg_e8["team"].isin(eliminated)]
            if len(active_e8) < 2:
                # Fall back to remaining S16 teams not yet eliminated
                reg_s16_df = in_bracket[in_bracket["team"].isin(sim_s16) & (in_bracket["region"]==region)]
                extras = reg_s16_df[~reg_s16_df["team"].isin(eliminated)].sort_values("contender_score", ascending=False)
                active_e8 = pd.concat([active_e8, extras]).drop_duplicates(subset="team").head(2)
            all_e8_picks.extend(active_e8["team"].tolist())
            with ecol:
                st.markdown(f'<div class="section-header">{region}</div>', unsafe_allow_html=True)
                if len(active_e8) == 0:
                    st.caption("All teams eliminated")
                for _, erow in active_e8.sort_values("contender_score", ascending=False).iterrows():
                    eseed = int(erow["seed"])
                    escore = safe_f(erow.get("contender_score", 50))
                    e_hot = hot_label(erow)
                    e_hot_span = f"<span style='color:#4ade80;font-size:0.8rem;margin-left:4px'>{e_hot}</span>" if e_hot else ""
                    e_badge = "#7c3aed" if eseed <= 4 else "#dc2626" if eseed >= 9 else "#475569"
                    st.markdown(
                        f'<div class="team-card">'
                        f'<span class="seed-badge" style="background:{e_badge}">{eseed}</span> '
                        f'<strong class="team-name">{erow["team"]}</strong>{e_hot_span}'
                        f'<br/><small style="color:#94a3b8">Score: {escore:.1f}</small>'
                        f'</div>',
                        unsafe_allow_html=True)

        if eliminated:
            st.caption(f"⚡ Eliminated so far: {', '.join(sorted(eliminated))}")
        st.markdown(f"**Model's Elite 8:** {', '.join(all_e8_picks)}")

        # ── Final Four + Champion ─────────────────────────────────────────────
        st.markdown("---")
        ff_s16a, ff_s16b = st.columns(2)
        with ff_s16a:
            st.markdown('<div class="section-header">🔥 Final Four</div>', unsafe_allow_html=True)
            for ff_t in sim_ff:
                row_ff = in_bracket[in_bracket["team"]==ff_t]
                if len(row_ff) > 0:
                    r = row_ff.iloc[0]
                    st.markdown(
                        f'<div class="team-card">'
                        f'<span class="seed-badge" style="background:#b45309">#{int(r["seed"])}</span> '
                        f'<strong class="team-name">{ff_t}</strong> '
                        f'<span style="color:#94a3b8;font-size:0.8rem">({r["region"]})</span>'
                        f'</div>',
                        unsafe_allow_html=True)
        with ff_s16b:
            st.markdown('<div class="section-header">🏆 Model\'s Champion</div>', unsafe_allow_html=True)
            if sim_champion:
                row_ch = in_bracket[in_bracket["team"]==sim_champion]
                ch_score = safe_f(row_ch.iloc[0]["contender_score"] if len(row_ch) > 0 else 70)
                ch_seed  = int(row_ch.iloc[0]["seed"] if len(row_ch) > 0 else 1)
                ch_region = row_ch.iloc[0]["region"] if len(row_ch) > 0 else ""
                st.markdown(
                    f'<div class="team-card" style="border:2px solid #f97316">'
                    f'<div style="font-size:1.5rem;text-align:center">🏆</div>'
                    f'<div style="text-align:center"><span class="seed-badge" style="background:#f97316">#{ch_seed}</span>'
                    f' <strong style="color:#f1f5f9;font-size:1.1rem">{sim_champion}</strong></div>'
                    f'<div style="text-align:center;color:#94a3b8;font-size:0.8rem">{ch_region} · Score: {ch_score:.1f}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            if sim_runner_up:
                st.caption(f"Runner-Up: {sim_runner_up}")

        # ── Best First-Round Upset Picks ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">💥 Model\'s First-Round Upset Picks (Seeds 9-13 the model picks to WIN)</div>', unsafe_allow_html=True)
        # Find lower seeds the simulation has WINNING their R1 game
        r1_upset_picks = in_bracket[
            (in_bracket["seed"] >= 9) &
            (in_bracket["seed"] <= 13) &
            (in_bracket["sim_round"] != "First Round")  # survived R1
        ].sort_values("contender_score", ascending=False)
        if len(r1_upset_picks) == 0:
            st.caption("Model doesn't project any seed 9-13 upsets in Round 1 this year.")
        else:
            up_cols = st.columns(min(3, len(r1_upset_picks)))
            for idx, (_, row) in enumerate(r1_upset_picks.head(6).iterrows()):
                with up_cols[idx % 3]:
                    matchup_seed = 17 - int(row["seed"])
                    opp = in_bracket[(in_bracket["region"]==row.get("region","")) & (in_bracket["seed"]==matchup_seed)]
                    opp_name = opp.iloc[0]["team"] if len(opp) > 0 else f"#{matchup_seed} seed"
                    wp = win_prob_sigmoid(row["contender_score"], safe_f(opp.iloc[0]["contender_score"] if len(opp)>0 else 60))
                    hl_up = hot_label(row)
                    hl_up_span = f" &nbsp;<span style='font-size:0.8rem'>{hl_up}</span>" if hl_up else ""
                    st.markdown(
                        f'<div class="team-card" style="border-color:#dc2626">'
                        f'<span style="color:#f97316;font-weight:800">#{int(row["seed"])} {row["team"]}</span>'
                        f'<span style="color:#94a3b8"> over </span>'
                        f'<span style="color:#f1f5f9">#{matchup_seed} {opp_name}</span>{hl_up_span}'
                        f'<br/><small style="color:#4ade80">Score: {row["contender_score"]:.1f} · Model win%: {wp*100:.0f}%</small>'
                        f'</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TEAM DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    row_data = scores[scores["team"] == selected_team]
    if len(row_data) == 0:
        st.warning(f"No data for {selected_team}")
        st.stop()
    row = row_data.iloc[0]

    try:
        seed_str = f"#{int(row['seed'])} Seed · {row.get('region','')}" if "seed" in row.index and pd.notna(row.get("seed")) else "Model Analysis"
    except Exception:
        seed_str = "Model Analysis"

    # Hot/trend label
    hl = hot_label(row)
    hot_badge = f' <span class="flag-hot">{hl}</span>' if hl else ""
    st.markdown(f'## {selected_team}{hot_badge}', unsafe_allow_html=True)
    st.markdown(f"*{seed_str} · {row.get('conference','')}"
                + (f" · Expected: **{row['expected_round']}**" if 'expected_round' in row else "") + "*")

    # Recent form bar
    if "last10_win_pct" in row.index:
        l10 = safe_f(row.get("last10_win_pct"))
        l10m = safe_f(row.get("last10_adj_margin"))
        season_m = safe_f(row.get("adj_margin"))
        trend_dir = "📈" if l10m > season_m + 3 else "📉" if l10m < season_m - 3 else "➡️"
        st.markdown(f"""
        <div style="background:#1a1f2e;border-radius:8px;padding:8px 14px;margin-bottom:10px;border:1px solid #374151">
            <small style="color:#b0bbd0;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Recent Form</small>
            <div style="display:flex;align-items:center;gap:16px;margin-top:4px">
                <span style="color:#f1f5f9;font-weight:700">Last 10 Win%: <span style="color:{'#4ade80' if l10>=0.8 else '#f59e0b' if l10>=0.6 else '#f87171'}">{l10*100:.0f}%</span></span>
                <span style="color:#f1f5f9;font-weight:700">L10 Margin: {trend_dir} <span style="color:{'#4ade80' if l10m>0 else '#f87171'}">{l10m:+.1f}</span></span>
                <span style="color:#94a3b8">Season avg: {season_m:+.1f}</span>
            </div>
            <div style="background:#2d3748;border-radius:3px;height:4px;margin-top:6px">
                <div style="width:{min(100,int(l10*100))}%;background:{'#4ade80' if l10>=0.7 else '#f59e0b'};height:4px;border-radius:3px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Top metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Contender Score",  f"{safe_f(row.get('contender_score')):.1f}/100")
    m2.metric("Upset Risk",       f"{safe_f(row.get('upset_risk_score')):.1f}/100")
    m3.metric("Record",           f"{safe_i(row.get('wins'))}-{safe_i(row.get('losses'))}")
    m4.metric("NET Rank",         f"#{safe_i(row.get('net_rank'))}" if pd.notna(row.get('net_rank')) else "N/A")
    m5.metric("Adj. Margin",      f"+{safe_f(row.get('adj_margin')):.1f}" if safe_f(row.get('adj_margin'))>0 else f"{safe_f(row.get('adj_margin')):.1f}")
    m6.metric("Archetype",        row.get("archetype","—") or "—")

    st.markdown("---")
    left_col, right_col = st.columns([1, 1])

    # ── Shot Zone Chart ───────────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="section-header">🎯 Shooting Zone Profile</div>', unsafe_allow_html=True)
        st.caption("Zone colors = efficiency vs D1 avg. Green = elite, Red = below average.")

        fig, ax = plt.subplots(figsize=(6, 5.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1f2e")
        ax.set_xlim(-25, 25); ax.set_ylim(-3, 42)
        ax.set_aspect("equal"); ax.axis("off")

        def eff_color(pct, low=30, high=40):
            norm = max(0, min(1, (pct - low) / (high - low)))
            r = int(255 * (1 - norm)); g = int(200 * norm + 55)
            return f"#{r:02x}{g:02x}55"

        paint_eff = safe_f(row.get("eff_fg_pct", 50)) * 0.8
        paint = mpatches.FancyBboxPatch((-8,0),16,19,boxstyle="round,pad=0.5",
            facecolor=eff_color(paint_eff,35,55), edgecolor="#e2e8f0", linewidth=1.5, alpha=0.85)
        ax.add_patch(paint)
        ax.text(0,9.5,f"PAINT\n{paint_eff:.0f}%",color="white",ha="center",va="center",fontsize=9,fontweight="bold")

        ft_pct_val = safe_f(row.get("ft_pct",70))
        ft_disp = ft_pct_val if ft_pct_val > 1 else ft_pct_val * 100
        ft_color = eff_color(ft_disp - 30, 0, 25)
        ax.plot([-8,8],[19,19],color="#e2e8f0",linewidth=1.5,linestyle="--")
        ax.add_patch(Arc((0,19),16,12,angle=0,theta1=0,theta2=180,color="#e2e8f0",linewidth=1.5))
        ax.add_patch(mpatches.FancyBboxPatch((-6,18),12,4,boxstyle="round,pad=0.3",facecolor=ft_color,edgecolor="none",alpha=0.7))
        ax.text(0,20.5,f"FT: {ft_disp:.0f}%",color="white",ha="center",fontsize=8)

        mid_eff = safe_f(row.get("eff_fg_pct",50)) * 0.85
        mid_col = eff_color(mid_eff,33,50)
        ax.add_patch(mpatches.FancyBboxPatch((-22,8),14,16,boxstyle="round,pad=0.3",facecolor=mid_col,edgecolor="#e2e8f0",linewidth=1,alpha=0.75))
        ax.text(-15,15.5,f"MID\n{mid_eff:.0f}%",color="white",ha="center",fontsize=7.5,fontweight="bold")
        ax.add_patch(mpatches.FancyBboxPatch((8,8),14,16,boxstyle="round,pad=0.3",facecolor=mid_col,edgecolor="#e2e8f0",linewidth=1,alpha=0.75))
        ax.text(15,15.5,f"MID\n{mid_eff:.0f}%",color="white",ha="center",fontsize=7.5,fontweight="bold")

        three_pct = safe_f(row.get("three_pt_pct",0.33))
        if three_pct > 1: three_pct /= 100
        three_col = eff_color(three_pct*100,32,40)
        ax.add_patch(Arc((0,5),44,44,angle=0,theta1=12,theta2=168,color="#e2e8f0",linewidth=2))
        ax.add_patch(mpatches.FancyBboxPatch((-25,0),7,8,boxstyle="round,pad=0.3",facecolor=three_col,edgecolor="#e2e8f0",linewidth=1,alpha=0.85))
        ax.text(-21.5,3.5,f"C3\n{three_pct*100:.1f}%",color="white",ha="center",fontsize=7,fontweight="bold")
        ax.add_patch(mpatches.FancyBboxPatch((18,0),7,8,boxstyle="round,pad=0.3",facecolor=three_col,edgecolor="#e2e8f0",linewidth=1,alpha=0.85))
        ax.text(21.5,3.5,f"C3\n{three_pct*100:.1f}%",color="white",ha="center",fontsize=7,fontweight="bold")
        ax.add_patch(mpatches.FancyBboxPatch((-14,26),28,10,boxstyle="round,pad=0.3",facecolor=three_col,edgecolor="#e2e8f0",linewidth=1,alpha=0.85))
        three_rate_raw = safe_f(row.get("three_pa_rate",0.35))
        three_rate = three_rate_raw/100 if three_rate_raw > 1 else three_rate_raw
        ax.text(0,31,f"3PT: {three_pct*100:.1f}%  Rate: {three_rate*100:.0f}%",color="white",ha="center",fontsize=8.5,fontweight="bold")

        hoop = plt.Circle((0,5),0.75,color="#ff6600",fill=False,linewidth=2.5)
        ax.add_patch(hoop)
        ax.add_patch(FancyArrowPatch((-3,4),(3,4),color="#e2e8f0",linewidth=2.5))
        ax.set_title(f"{selected_team} — Shooting Zones",color="#e2e8f0",fontsize=11,fontweight="bold",pad=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Radar / DNA Chart ─────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="section-header">🧬 Team DNA — Strength Radar</div>', unsafe_allow_html=True)
        cats = ["Defense","Offense","Clutch","Guard Play","Rebounding","Consistency"]
        vals = [safe_f(row.get("defense_score",50)), safe_f(row.get("efficiency_score",50)),
                safe_f(row.get("clutch_score",50)),  safe_f(row.get("guard_play_score",50)),
                safe_f(row.get("rebounding_score",50)), safe_f(row.get("consistency_score",50))]
        N = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        vals_plot = vals + [vals[0]]

        fig2, ax2 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#0e1117"); ax2.set_facecolor("#1a1f2e")
        ax2.plot(angles, vals_plot, linewidth=2.5, color="#f97316")
        ax2.fill(angles, vals_plot, alpha=0.25, color="#f97316")
        for ring in [25,50,75,100]:
            ax2.plot(angles, [ring]*N+[ring], linewidth=0.5, linestyle="--", color="#3d4a5c", alpha=0.7)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(cats, color="#f1f5f9", fontsize=9.5, fontweight="bold")
        ax2.set_yticklabels([]); ax2.set_ylim(0,100)
        ax2.spines["polar"].set_color("#374151"); ax2.grid(color="#374151", linewidth=0.5)
        ax2.set_title("Sub-Score Profile", color="#f1f5f9", fontsize=11, fontweight="bold", pad=20)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # Key Stats
        st.markdown('<div class="section-header">📋 Key Stats</div>', unsafe_allow_html=True)
        ks1, ks2 = st.columns(2)
        with ks1:
            st.metric("PPG",     f"{safe_f(row.get('points_per_game')):.1f}")
            st.metric("Opp PPG", f"{safe_f(row.get('points_allowed_per_game')):.1f}")
            tpp = safe_f(row.get('three_pt_pct'))
            st.metric("3P%",  f"{tpp*100:.1f}%" if tpp < 1 else f"{tpp:.1f}%")
            ftp = safe_f(row.get('ft_pct'))
            st.metric("FT%",  f"{ftp*100:.1f}%" if ftp < 1 else f"{ftp:.1f}%")
        with ks2:
            st.metric("Adj. Offense", f"{safe_f(row.get('adj_offense')):.1f}")
            st.metric("Adj. Defense", f"{safe_f(row.get('adj_defense')):.1f}")
            tempo = safe_f(row.get('tempo'))
            st.metric("Tempo", f"{tempo:.1f} poss/g" if tempo > 0 else "—")
            q1wp = safe_f(row.get('q1_win_pct'))
            st.metric("Q1 Win%", f"{q1wp*100:.0f}%" if q1wp < 1 else f"{q1wp:.0f}%")

    # Coach Profile + Flags
    st.markdown("---")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown('<div class="section-header">👨‍💼 Coach Profile</div>', unsafe_allow_html=True)
        st.markdown(f"**{row.get('coach','') or 'Unknown'}**")
        st.metric("Years at School",       f"{safe_i(row.get('coach_years_at_school'))}")
        st.metric("NCAA Tournament Games",  f"{safe_i(row.get('coach_ncaa_games'))}")
    with cc2:
        st.markdown('<div class="section-header">🏆 Tournament Résumé</div>', unsafe_allow_html=True)
        st.metric("Sweet 16 Appearances",   f"{safe_i(row.get('coach_sweet16s'))}")
        st.metric("Final Four Appearances",  f"{safe_i(row.get('coach_finalfours'))}")
        if row.get("first_year_coach_flag", 0): st.warning("⚠️ First-year coach at this school")
    with cc3:
        st.markdown('<div class="section-header">🚩 Model Flags</div>', unsafe_allow_html=True)
        flags_map = {
            "🪄 Cinderella":   row.get("cinderella_flag", False),
            "🌑 Darkhorse":    row.get("title_darkhorse_flag", False),
            "🃏 Fraud Fav.":   row.get("fraud_favorite_flag", False),
            "💥 Upset Alert":  row.get("dangerous_low_seed_flag", False),
            "📈 Underseeded":  row.get("underseeded_flag", False),
            "📉 Overseeded":   row.get("overseeded_flag", False),
            "⚡ Foul Depend.": row.get("high_foul_dependence_flag", False),
        }
        active = [k for k, v in flags_map.items() if v]
        if active:
            for f in active: st.markdown(f'<strong style="color:#f1f5f9">{f}</strong>', unsafe_allow_html=True)
        else:
            st.caption("No special flags")

    # Clutch/GW factor callout
    clutch = safe_f(row.get("clutch_score",50))
    cwp = safe_f(row.get("close_win_pct",0.5))
    cwp_disp = cwp*100 if cwp < 1 else cwp
    if clutch >= 65 or cwp_disp >= 65:
        st.markdown(f"""
        <div style="background:#1a2a1a;border:1px solid #16a34a;border-radius:8px;padding:8px 14px;margin-top:8px">
            <strong style="color:#4ade80">⚡ Clutch/Close Game Factor</strong><br/>
            <small style="color:#d1fae5">Clutch Score: {clutch:.1f}/100 · Close Win%: {cwp_disp:.0f}%
            — GW shots &amp; buzzer beaters are a <em>small</em> factor (~3-5%) in model confidence.
            This team shows genuine clutch ability beyond lucky finishes.</small>
        </div>
        """, unsafe_allow_html=True)

    if "explanation_summary" in row.index and pd.notna(row.get("explanation_summary")):
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Model Analysis</div>', unsafe_allow_html=True)
        st.info(row["explanation_summary"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL vs COMMITTEE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Model vs. Selection Committee — Seed Disagreement Analysis</div>', unsafe_allow_html=True)

    if "seed" not in in_bracket.columns:
        st.warning("Bracket/seed data not loaded.")
    else:
        # ── Compute model's predicted seed within each region ─────────────────
        # Method: rank every team in the region by contender_score (highest = #1 seed).
        # The gap (actual_seed − model_seed) tells us who the committee under/over-valued.
        model_seeded_parts = []
        for region in ["East","South","West","Midwest"]:
            reg = in_bracket[in_bracket["region"]==region].copy()
            reg = reg.sort_values("contender_score", ascending=False).reset_index(drop=True)
            reg["model_seed"] = reg.index + 1   # rank 1 = strongest score in region
            model_seeded_parts.append(reg)
        model_seeded_df = pd.concat(model_seeded_parts, ignore_index=True)
        # positive gap = actual seed higher number than model seed = UNDERSEEDED (model loves them more)
        model_seeded_df["seed_gap"] = model_seeded_df["seed"] - model_seeded_df["model_seed"]

        st.info("**How the model derives its predicted seed:** Within each region, every team is ranked 1–16 by their *contender score* (composite of adjusted margin, SOS, offensive/defensive efficiency, clutch, and recent form). The team ranked #1 in a region by the model is the model's predicted #1 seed. A positive gap means the committee seeded that team *lower* than the model would — they are underseeded in the model's view. **First-round win predictions are shown for all underseeded teams.**")

        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown('<div class="section-header">🟢 Underseeded — Model Ranks Higher Than Committee</div>', unsafe_allow_html=True)
            st.caption("Committee gave them a tougher draw than they deserve.")
            underseeded = model_seeded_df[model_seeded_df["seed_gap"] > 0].sort_values("seed_gap", ascending=False).head(10)
            for _, r in underseeded.iterrows():
                actual_seed = int(r["seed"])
                mseed = int(r["model_seed"])
                gap = int(r["seed_gap"])
                bar_w = min(100, gap * 8)
                # First-round matchup opponent (seed that sums to 17 with actual_seed)
                opp_seed = 17 - actual_seed
                opp_rows = in_bracket[(in_bracket["region"]==r["region"]) & (in_bracket["seed"]==opp_seed)]
                opp_name = opp_rows.iloc[0]["team"] if len(opp_rows) > 0 else f"#{opp_seed} seed"
                opp_score = safe_f(opp_rows.iloc[0]["contender_score"] if len(opp_rows) > 0 else 55)
                wp = win_prob_sigmoid(r["contender_score"], opp_score)
                win_verdict = f'<span style="color:#4ade80;font-weight:700">✅ Model picks {r["team"]} to WIN R1 ({wp*100:.0f}%)</span>' if wp > 0.5 \
                              else f'<span style="color:#f87171;font-weight:700">⚠️ Model still picks #{opp_seed} {opp_name} ({(1-wp)*100:.0f}%)</span>'
                score_drivers = []
                if safe_f(r.get("adj_margin",0)) > 12: score_drivers.append("strong margin")
                if safe_f(r.get("strength_of_schedule",0)) > 5: score_drivers.append("tough schedule")
                if safe_f(r.get("last10_win_pct",0.5)) >= 0.8: score_drivers.append("hot streak")
                if safe_f(r.get("clutch_score",50)) > 60: score_drivers.append("clutch performer")
                why = f" ({', '.join(score_drivers)})" if score_drivers else ""
                st.markdown(f"""
                <div class="team-card">
                    <span class="seed-badge" style="background:#374151">#{actual_seed}</span>
                    <strong class="team-name">{r['team']}</strong>
                    <span style="color:#4ade80;font-weight:700;font-size:0.88rem"> → Model: #{mseed} seed (+{gap} spots)</span>
                    <br/><small style="color:#94a3b8">Score: {r['contender_score']:.1f}{why} · Risk: {r['upset_risk_score']:.1f}</small>
                    <div style="background:#2d3748;border-radius:4px;margin-top:4px;height:5px">
                        <div style="width:{bar_w}%;background:#4ade80;height:5px;border-radius:4px"></div>
                    </div>
                    <small style="color:#64748b">R1 vs #{opp_seed} {opp_name}:</small> {win_verdict}
                </div>
                """, unsafe_allow_html=True)

        with mv2:
            st.markdown('<div class="section-header">🔴 Overseeded — Committee Ranks Higher Than Model</div>', unsafe_allow_html=True)
            st.caption("Committee gave them a better draw than the model thinks they deserve.")
            overseeded = model_seeded_df[model_seeded_df["seed_gap"] < 0].sort_values("seed_gap").head(10)
            for _, r in overseeded.iterrows():
                actual_seed = int(r["seed"])
                mseed = int(r["model_seed"])
                gap = abs(int(r["seed_gap"]))
                bar_w = min(100, gap * 8)
                opp_seed = 17 - actual_seed
                opp_rows = in_bracket[(in_bracket["region"]==r["region"]) & (in_bracket["seed"]==opp_seed)]
                opp_name = opp_rows.iloc[0]["team"] if len(opp_rows) > 0 else f"#{opp_seed} seed"
                opp_score = safe_f(opp_rows.iloc[0]["contender_score"] if len(opp_rows) > 0 else 55)
                wp = win_prob_sigmoid(r["contender_score"], opp_score)
                upset_risk = f'<span style="color:#f87171;font-weight:700">⚠️ Upset risk: #{opp_seed} {opp_name} ({(1-wp)*100:.0f}% per model)</span>' if (1-wp) > 0.38 \
                             else f'<span style="color:#94a3b8">Model still picks {r["team"]} to win R1 ({wp*100:.0f}%)</span>'
                st.markdown(f"""
                <div class="team-card">
                    <span class="seed-badge" style="background:#7f1d1d">#{actual_seed}</span>
                    <strong class="team-name">{r['team']}</strong>
                    <span style="color:#f87171;font-weight:700;font-size:0.88rem"> → Model: #{mseed} seed (−{gap} spots)</span>
                    <br/><small style="color:#94a3b8">Score: {r['contender_score']:.1f} · Risk: {r['upset_risk_score']:.1f}</small>
                    <div style="background:#2d3748;border-radius:4px;margin-top:4px;height:5px">
                        <div style="width:{bar_w}%;background:#ef4444;height:5px;border-radius:4px"></div>
                    </div>
                    <small style="color:#64748b">R1 vs #{opp_seed} {opp_name}:</small> {upset_risk}
                </div>
                """, unsafe_allow_html=True)

        if len(bracket) > 0:
            snubs = scores[(scores["contender_score"]>=63) & (~scores["team"].isin(bracket["team"]))].sort_values("contender_score", ascending=False)
            if len(snubs) > 0:
                st.markdown("---")
                st.markdown('<div class="section-header">🚨 Biggest Snubs (model 63+ score, not in bracket)</div>', unsafe_allow_html=True)
                snub_cols = st.columns(min(4, len(snubs)))
                for idx, (_, r) in enumerate(snubs.head(4).iterrows()):
                    with snub_cols[idx]:
                        st.markdown(f"""
                        <div class="team-card">
                            <div class="team-name">{r['team']}</div>
                            <div class="team-score">Score: {r['contender_score']:.1f}</div>
                            <small style="color:#f87171">Left out · Model rank among all teams: top {int(scores.sort_values('contender_score',ascending=False).reset_index(drop=True).index[scores['team']==r['team']].tolist()[0]+1) if len(scores[scores['team']==r['team']])>0 else '?'}</small>
                        </div>
                        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CHAMPIONSHIP ODDS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Championship Probabilities — 10,000 Bracket Simulations</div>', unsafe_allow_html=True)

    if len(champs) == 0:
        st.warning("Run simulation first.")
    else:
        champs_disp = champs.merge(bracket[["team","region","seed"]], on="team", how="left") if len(bracket)>0 else champs
        champs_disp = champs_disp.sort_values("championship_pct", ascending=False).head(20)

        pod1, pod2, pod3 = st.columns(3)
        for i, (_, r) in enumerate(champs_disp.head(3).iterrows()):
            seed_disp = f"#{int(r['seed'])} seed · {r['region']}" if pd.notna(r.get('seed')) else ""
            border_col = '#f59e0b' if i==0 else '#94a3b8' if i==1 else '#cd7f32'
            with [pod1,pod2,pod3][i]:
                st.markdown(f"""
                <div class="team-card" style="text-align:center;border-color:{border_col};border-width:2px">
                    <div style="font-size:2rem">{"🥇🥈🥉"[i]}</div>
                    <div class="team-name" style="font-size:1.3rem">{r['team']}</div>
                    <div style="color:#f97316;font-size:1.7rem;font-weight:800">{r['championship_pct']:.1f}%</div>
                    <div style="color:#94a3b8;font-size:0.8rem">{seed_disp}</div>
                    <div style="color:#b0bbd0;font-size:0.78rem">Model line: {american_line(r['championship_pct']/100 + 0.5)}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        fig3.patch.set_facecolor("#0e1117"); ax3.set_facecolor("#1a1f2e")
        teams_c = champs_disp["team"].tolist()
        probs_c = champs_disp["championship_pct"].tolist()
        colors_c = ["#f97316" if i==0 else "#3b82f6" if i<4 else "#4b5563" for i in range(len(teams_c))]
        bars = ax3.barh(teams_c[::-1], probs_c[::-1], color=colors_c[::-1], edgecolor="none", height=0.65)
        for bar, pct in zip(bars, probs_c[::-1]):
            ax3.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f"{pct:.1f}%", va="center", color="#ffffff", fontsize=8.5, fontweight="bold")
        ax3.set_xlabel("Championship Probability (%)", color="#b0bbd0", fontsize=9)
        ax3.tick_params(colors="#f1f5f9", labelsize=9)
        for spine in ["top","right"]: ax3.spines[spine].set_visible(False)
        ax3.spines["bottom"].set_color("#374151"); ax3.spines["left"].set_color("#374151")
        ax3.set_title("Simulated Championship Probabilities (10,000 runs)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.markdown("---")
        st.markdown('<div class="section-header">Championship Probability by Region</div>', unsafe_allow_html=True)
        if "region" in champs_disp.columns:
            reg_probs = champs_disp.groupby("region")["championship_pct"].sum().sort_values(ascending=False)
            rcols = st.columns(4)
            for i, (reg, pct) in enumerate(reg_probs.items()):
                if i < 4:
                    with rcols[i]: st.metric(f"{reg} Region", f"{pct:.1f}%", "of championships")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — INTERACTIVE BRACKET
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">🏆 2026 NCAA Tournament — Interactive Bracket</div>', unsafe_allow_html=True)
    st.caption("Model line vs. historical seed-based public line. Click any matchup → full analysis. Log results to track model accuracy.")

    BRACKET_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    # Model accuracy tracker
    total_res = len(st.session_state.results)
    if total_res > 0:
        correct_res = sum(1 for r in st.session_state.results if r.get("model_pick") == r.get("winner"))
        acc_pct = correct_res / total_res * 100
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("📊 Model Record", f"{correct_res}-{total_res-correct_res}", f"{acc_pct:.0f}% accuracy")
        recent5 = st.session_state.results[-5:]
        recent_correct = sum(1 for r in recent5 if r.get("model_pick") == r.get("winner"))
        ac2.metric("🔄 Last 5 Picks", f"{recent_correct}/5", "recent form")
        conf_label = "HIGH 🎯" if acc_pct > 75 else "MODERATE" if acc_pct > 60 else "RECALIBRATING ⚠️"
        ac3.metric("🤖 Confidence", conf_label)
        # Adaptive note: which seed ranges are working
        if total_res >= 4:
            fav_correct = sum(1 for r in st.session_state.results if r.get("model_pick") == r.get("winner") and r.get("model_pick") == r.get("fav_team"))
            ac4.metric("💡 Favored Pick Acc.", f"{fav_correct}/{total_res}", "model favorites")
        st.markdown("---")

    if "region" not in in_bracket.columns or len(in_bracket) == 0:
        st.warning("Bracket data not loaded.")
    else:
        reg_cols_br = st.columns(4)
        for reg_i, (region, rcol) in enumerate(zip(["East","South","West","Midwest"], reg_cols_br)):
            region_data = in_bracket[in_bracket["region"]==region].copy()
            top1_rows = region_data[region_data["seed"]==1]
            top1_name = top1_rows.iloc[0]["team"] if len(top1_rows)>0 else "TBD"

            with rcol:
                st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:1rem;border-bottom:2px solid #f97316;padding-bottom:3px;margin-bottom:8px;text-transform:uppercase">{region} · #1 {top1_name}</div>', unsafe_allow_html=True)

                for s1, s2 in BRACKET_PAIRS:
                    t1_rows = region_data[region_data["seed"]==s1]
                    t2_rows = region_data[region_data["seed"]==s2]
                    if len(t1_rows)==0 or len(t2_rows)==0: continue

                    t1 = t1_rows.iloc[0]
                    t2 = t2_rows.iloc[0]
                    c1 = safe_f(t1.get("contender_score",50))
                    c2 = safe_f(t2.get("contender_score",50))
                    p1 = win_prob_sigmoid(c1, c2)
                    p2 = 1 - p1

                    # Public/historical line
                    hist_p1 = HIST_SEED_WIN_PCT.get((s1, s2), 0.5)

                    fav_s  = s1 if p1 >= p2 else s2
                    dog_s  = s2 if p1 >= p2 else s1
                    fav    = t1 if p1 >= p2 else t2
                    dog    = t2 if p1 >= p2 else t1
                    fav_p  = max(p1, p2)
                    dog_p  = min(p1, p2)
                    fav_c  = c1 if p1 >= p2 else c2
                    dog_c  = c2 if p1 >= p2 else c1

                    matchup_key = f"{region}_{s1}v{s2}"
                    result_rec  = results_dict.get(matchup_key, {})
                    result_winner = result_rec.get("winner")

                    hl1 = hot_label(t1)
                    hl2 = hot_label(t2)

                    # Model vs public edge
                    edge = abs(p1 - hist_p1)
                    edge_tag = ""
                    if p1 > hist_p1 + 0.12:
                        edge_tag = f"<span style='color:#4ade80;font-size:0.7rem;font-weight:700'> 💎 Model favors #{s1}</span>"
                    elif p1 < hist_p1 - 0.12:
                        edge_tag = f"<span style='color:#f59e0b;font-size:0.7rem;font-weight:700'> 💎 Model favors #{s2}</span>"

                    result_badge = ""
                    if result_winner:
                        correct = result_winner == fav["team"]
                        result_badge = " ✅" if correct else " ❌"

                    with st.expander(f"#{s1} vs #{s2}{result_badge}", expanded=False):
                        # Matchup header
                        mh1, mh2 = st.columns(2)
                        with mh1:
                            hl1_span = f" <span style='color:#4ade80;font-size:0.72rem'>{hl1}</span>" if hl1 else ""
                            seed1_bg = "#f97316" if fav_s == s1 else "#475569"
                            bar1_bg  = "#4ade80" if p1 > p2 else "#64748b"
                            txt1_col = "#4ade80" if p1 > p2 else "#94a3b8"
                            st.markdown(
                                f'<div style="padding:8px;background:#131820;border-radius:6px">'
                                f'<span style="background:{seed1_bg};color:white;border-radius:50%;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:0.68rem;font-weight:800;margin-right:6px">{s1}</span>'
                                f'<strong style="color:#f1f5f9;font-size:0.95rem">{t1["team"]}</strong>{hl1_span}'
                                f'<br/><small style="color:#94a3b8">Score: {c1:.1f}</small>'
                                f'<div style="background:#1e293b;border-radius:3px;height:4px;margin:4px 0">'
                                f'<div style="width:{int(p1*100)}%;background:{bar1_bg};height:4px;border-radius:3px"></div></div>'
                                f'<span style="color:{txt1_col};font-weight:700;font-size:0.9rem">{p1*100:.0f}% · {american_line(p1)}</span>'
                                f'</div>',
                                unsafe_allow_html=True)
                            if st.button(f"🔍 {t1['team']} Deep Dive", key=f"dd1_{matchup_key}", use_container_width=True):
                                st.session_state["team_selectbox"] = t1["team"]
                                st.session_state.dive_team = t1["team"]
                                st.rerun()
                        with mh2:
                            hl2_span = f" <span style='color:#4ade80;font-size:0.72rem'>{hl2}</span>" if hl2 else ""
                            seed2_bg = "#f97316" if fav_s == s2 else "#475569"
                            bar2_bg  = "#4ade80" if p2 > p1 else "#64748b"
                            txt2_col = "#4ade80" if p2 > p1 else "#94a3b8"
                            st.markdown(
                                f'<div style="padding:8px;background:#131820;border-radius:6px">'
                                f'<span style="background:{seed2_bg};color:white;border-radius:50%;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:0.68rem;font-weight:800;margin-right:6px">{s2}</span>'
                                f'<strong style="color:#f1f5f9;font-size:0.95rem">{t2["team"]}</strong>{hl2_span}'
                                f'<br/><small style="color:#94a3b8">Score: {c2:.1f}</small>'
                                f'<div style="background:#1e293b;border-radius:3px;height:4px;margin:4px 0">'
                                f'<div style="width:{int(p2*100)}%;background:{bar2_bg};height:4px;border-radius:3px"></div></div>'
                                f'<span style="color:{txt2_col};font-weight:700;font-size:0.9rem">{p2*100:.0f}% · {american_line(p2)}</span>'
                                f'</div>',
                                unsafe_allow_html=True)
                            if st.button(f"🔍 {t2['team']} Deep Dive", key=f"dd2_{matchup_key}", use_container_width=True):
                                st.session_state["team_selectbox"] = t2["team"]
                                st.session_state.dive_team = t2["team"]
                                st.rerun()

                        # Public vs model line comparison
                        st.markdown(f"""
                        <div style="background:#1e293b;border-radius:6px;padding:6px 10px;margin:5px 0;font-size:0.8rem">
                            <strong style="color:#b0bbd0">📊 Public Line (hist. seed odds):</strong>
                            <span style="color:#f1f5f9">#{s1} {american_line(hist_p1)} &nbsp;/&nbsp; #{s2} {american_line(1-hist_p1)}</span>
                            &nbsp;·&nbsp;
                            <strong style="color:#b0bbd0">Model Line:</strong>
                            <span style="color:#f1f5f9">#{s1} {american_line(p1)} / #{s2} {american_line(p2)}</span>
                            {edge_tag}
                        </div>
                        """, unsafe_allow_html=True)

                        # Style matchup insight
                        insight = style_matchup_insight(t1, t2)
                        st.markdown(f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;padding:6px 10px;margin:5px 0;color:#d1fae5;font-size:0.82rem">💡 {insight}</div>', unsafe_allow_html=True)

                        # Clutch factor
                        fav_cl = safe_f(fav.get("clutch_score",50))
                        dog_cl = safe_f(dog.get("clutch_score",50))
                        fav_cwp = safe_f(fav.get("close_win_pct",0.5))
                        dog_cwp = safe_f(dog.get("close_win_pct",0.5))
                        if abs(fav_cl - dog_cl) > 8:
                            cl_edge = fav["team"] if fav_cl > dog_cl else dog["team"]
                            st.markdown(f'<small style="color:#fbbf24">⚡ GW/Clutch edge: <strong>{cl_edge}</strong> (clutch score diff: {abs(fav_cl-dog_cl):.0f} pts) — ~3-5% factor in close games</small>', unsafe_allow_html=True)

                        # Model flags
                        flag_msgs = []
                        if fav.get("fraud_favorite_flag", False): flag_msgs.append(f"⚠️ {fav['team']} is a Fraud Favorite — upset risk elevated")
                        if dog.get("dangerous_low_seed_flag", False): flag_msgs.append(f"💥 {dog['team']} is a Dangerous Low Seed")
                        if dog.get("cinderella_flag", False): flag_msgs.append(f"🪄 {dog['team']} has Cinderella traits — don't sleep on them")
                        if dog_p > 0.38: flag_msgs.append(f"💎 {dog['team']} at {american_line(dog_p)} may offer value — model gives {dog_p*100:.0f}%")
                        for msg in flag_msgs:
                            st.markdown(f'<small style="color:#f87171">{msg}</small>', unsafe_allow_html=True)

                        # Log result buttons
                        if not result_winner:
                            st.markdown("---")
                            st.markdown('<small style="color:#94a3b8">Log result to track model accuracy:</small>', unsafe_allow_html=True)
                            rb1, rb2 = st.columns(2)
                            if rb1.button(f"✅ {t1['team']} won", key=f"r1_{matchup_key}"):
                                rec = {"matchup": matchup_key, "winner": t1["team"],
                                       "teams": [t1["team"], t2["team"]],
                                       "model_pick": fav["team"],
                                       "fav_team": fav["team"], "timestamp": str(datetime.now().date())}
                                st.session_state.results.append(rec)
                                os.makedirs("data/tournament_2026", exist_ok=True)
                                pd.DataFrame(st.session_state.results).to_csv(RESULTS_PATH, index=False)
                                st.rerun()
                            if rb2.button(f"✅ {t2['team']} won", key=f"r2_{matchup_key}"):
                                rec = {"matchup": matchup_key, "winner": t2["team"],
                                       "teams": [t1["team"], t2["team"]],
                                       "model_pick": fav["team"],
                                       "fav_team": fav["team"], "timestamp": str(datetime.now().date())}
                                st.session_state.results.append(rec)
                                os.makedirs("data/tournament_2026", exist_ok=True)
                                pd.DataFrame(st.session_state.results).to_csv(RESULTS_PATH, index=False)
                                st.rerun()
                        else:
                            correct_str = "✅ Correct call!" if result_winner == fav["team"] else "❌ Upset occurred"
                            st.markdown(f'<div style="background:#1e293b;border-radius:6px;padding:6px 10px;margin-top:6px"><strong style="color:#f1f5f9">Result logged: {result_winner} won — {correct_str}</strong></div>', unsafe_allow_html=True)
                            if st.button("↩ Clear result", key=f"clr_{matchup_key}"):
                                st.session_state.results = [r for r in st.session_state.results if r["matchup"] != matchup_key]
                                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — LIVE GAMES  (ESPN public API, no key required)
# ─────────────────────────────────────────────────────────────────────────────
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?groups=100&limit=50"
)
ESPN_PLAYBYPLAY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary?event={event_id}"
)


def fetch_live_games():
    """Pull current NCAA tournament games from ESPN's public scoreboard API.
    Returns list of game dicts or empty list on failure."""
    if not _REQUESTS_OK:
        return []
    try:
        resp = _requests.get(ESPN_SCOREBOARD, timeout=6,
                             headers={"User-Agent": "statlasberg/1.0"})
        resp.raise_for_status()
        data = resp.json()
        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            comps = comp.get("competitors", [])
            if len(comps) < 2:
                continue
            # Identify home/away — NCAA tournament is neutral so both are "away"
            t1 = comps[0]; t2 = comps[1]
            status_obj = event.get("status", {})
            status_type = status_obj.get("type", {})
            state = status_type.get("state", "pre")      # pre / in / post
            display_clock = status_obj.get("displayClock", "0:00")
            period = status_obj.get("period", 0)
            completed = status_type.get("completed", False)

            def team_info(t):
                score_str = t.get("score", "0")
                try:
                    score = int(score_str)
                except Exception:
                    score = 0
                return {
                    "name":    t.get("team", {}).get("displayName", "Unknown"),
                    "abbr":    t.get("team", {}).get("abbreviation", "???"),
                    "score":   score,
                    "record":  t.get("records", [{}])[0].get("summary", ""),
                }

            g = {
                "event_id":  event.get("id", ""),
                "name":      event.get("name", ""),
                "state":     state,
                "completed": completed,
                "period":    period,
                "clock":     display_clock,
                "team1":     team_info(t1),
                "team2":     team_info(t2),
                "headline":  comp.get("notes", [{}])[0].get("headline", ""),
            }
            games.append(g)
        return games
    except Exception:
        return []


def fetch_play_by_play(event_id):
    """Return last N plays and current scoring run for a game."""
    if not _REQUESTS_OK or not event_id:
        return [], ""
    try:
        url = ESPN_PLAYBYPLAY.format(event_id=event_id)
        resp = _requests.get(url, timeout=6,
                             headers={"User-Agent": "statlasberg/1.0"})
        resp.raise_for_status()
        data = resp.json()
        plays_raw = []
        for period_data in data.get("plays", []):
            # ESPN returns plays inside period arrays
            if isinstance(period_data, dict):
                plays_raw.append(period_data)
        # Get last 10 plays
        last_plays = plays_raw[-10:] if len(plays_raw) >= 10 else plays_raw
        play_texts = [p.get("text", "") for p in reversed(last_plays) if p.get("text")]

        # Detect scoring run: last 6 scoring plays
        score_plays = [p for p in reversed(plays_raw) if p.get("scoringPlay", False)][:6]
        run_team = ""
        run_count = 0
        if score_plays:
            first_scorer = score_plays[0].get("team", {}).get("displayName", "")
            run = 0
            for sp in score_plays:
                if sp.get("team", {}).get("displayName", "") == first_scorer:
                    run += 1
                else:
                    break
            if run >= 3:
                run_team = first_scorer
                run_count = run

        momentum = f"🔥 {run_team} on a {run_count}-0 run!" if run_count >= 3 else ""
        return play_texts, momentum
    except Exception:
        return [], ""


def live_win_prob(pregame_p, score_diff, period, clock_str, total_periods=2):
    """Blend model pre-game probability with in-game state.
    score_diff = team1_score - team2_score (positive = team1 leading).
    Returns (p_team1_wins, p_team2_wins).
    """
    # Parse clock to seconds remaining in current period
    try:
        parts = clock_str.split(":")
        clock_secs = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0
    except Exception:
        clock_secs = 0

    total_game_secs = total_periods * 20 * 60  # 2×20 min halves = 2400s
    period_secs = 20 * 60  # 20 min per half
    elapsed_secs = max(0, (period - 1) * period_secs + (period_secs - clock_secs))
    game_pct = min(1.0, elapsed_secs / total_game_secs)  # 0=start, 1=end

    # In-game win probability: score diff matters more as game ends
    # Sensitivity: at end of game a 1-point lead is ~85% win; early it's ~52%
    k = 3.0 + game_pct * 18.0   # scale factor grows from 3 (start) to 21 (end)
    in_game_p = 1.0 / (1.0 + math.exp(-score_diff / k))

    # Blend: pre-game model dominates early, in-game dominates late
    blended = (1 - game_pct) * pregame_p + game_pct * in_game_p
    return round(blended, 3), round(1 - blended, 3)


def model_pregame_prob(team1_name, team2_name, bkt_df):
    """Look up pre-game win probability from contender scores."""
    def get_score(name):
        rows = bkt_df[bkt_df["team"].str.lower() == name.lower()]
        if len(rows) == 0:
            # fuzzy: find closest team name
            for _, r in bkt_df.iterrows():
                if name.lower() in r["team"].lower() or r["team"].lower() in name.lower():
                    return safe_f(r.get("contender_score", 55))
        return safe_f(rows.iloc[0].get("contender_score", 55)) if len(rows) > 0 else 55.0
    s1 = get_score(team1_name)
    s2 = get_score(team2_name)
    return win_prob_sigmoid(s1, s2), s1, s2


with tab7:
    st.markdown('<div class="section-header">📺 Live — NCAA Tournament Games</div>', unsafe_allow_html=True)
    st.caption("Live scores via ESPN · In-game win probability blends model pre-game score with live game state. Refreshes on demand.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
    with col_ctrl1:
        auto_ref = st.toggle("⏱ Auto-refresh (60s)", value=False, key="live_auto_refresh")
    with col_ctrl2:
        if st.button("🔄 Refresh now", key="live_refresh_btn"):
            st.cache_data.clear()

    if auto_ref:
        st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    if not _REQUESTS_OK:
        st.warning("⚠️ `requests` library not installed. Run `pip install requests` and restart.")
    else:
        with st.spinner("Fetching live games from ESPN…"):
            live_games = fetch_live_games()

        # Filter to tournament-relevant games (status=in or post from today)
        live_now   = [g for g in live_games if g["state"] == "in"]
        recent_fin = [g for g in live_games if g["state"] == "post"][:8]
        upcoming   = [g for g in live_games if g["state"] == "pre"][:8]

        if not live_games:
            st.info("🏀 No NCAA tournament games found right now. Games appear here once tipoff. Check back during tournament play (March 20 – April 7, 2026).")
        else:
            # ── LIVE GAMES ────────────────────────────────────────────────────
            if live_now:
                st.markdown(f'<div class="section-header">🔴 LIVE NOW — {len(live_now)} game(s)</div>', unsafe_allow_html=True)
                for g in live_now:
                    t1 = g["team1"]; t2 = g["team2"]
                    sd = t1["score"] - t2["score"]
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    p1, p2 = live_win_prob(pregame_p, sd, g["period"], g["clock"])

                    # Win probability bar color
                    lead_team = t1["name"] if sd >= 0 else t2["name"]
                    lead_p    = max(p1, p2)
                    bar_col   = "#4ade80" if lead_p >= 0.7 else "#f59e0b" if lead_p >= 0.55 else "#94a3b8"

                    period_str = f"{'1st' if g['period']==1 else '2nd' if g['period']==2 else 'OT'} · {g['clock']}"

                    with st.expander(
                        f"🔴 {t1['name']} {t1['score']} — {t2['score']} {t2['name']}  |  {period_str}",
                        expanded=True
                    ):
                        lcol1, lcol2 = st.columns(2)
                        for col, team, p, my_score, opp_score in [
                            (lcol1, t1, p1, s1, s2),
                            (lcol2, t2, p2, s2, s1),
                        ]:
                            with col:
                                is_leader = (team["score"] >= (t2["score"] if team is t1 else t1["score"]))
                                border_c = "#f97316" if is_leader else "#374151"
                                st.markdown(
                                    f'<div style="background:#131820;border:2px solid {border_c};border-radius:8px;padding:10px">'
                                    f'<div style="font-size:1.1rem;font-weight:800;color:#f1f5f9">{team["name"]}</div>'
                                    f'<div style="font-size:2.5rem;font-weight:900;color:{"#4ade80" if is_leader else "#f1f5f9"}">{team["score"]}</div>'
                                    f'<div style="color:#94a3b8;font-size:0.78rem">{team["record"]}</div>'
                                    f'<div style="margin-top:8px">'
                                    f'<div style="background:#2d3748;border-radius:4px;height:8px">'
                                    f'<div style="width:{int(p*100)}%;background:{bar_col};height:8px;border-radius:4px"></div></div>'
                                    f'<div style="color:{bar_col};font-weight:700;font-size:1.05rem;margin-top:3px">'
                                    f'{p*100:.0f}% live win prob</div>'
                                    f'<small style="color:#64748b">Pre-game model score: {my_score:.1f} · '
                                    f'{american_line(p)} live line</small>'
                                    f'</div></div>',
                                    unsafe_allow_html=True)

                        # Play-by-play + momentum
                        plays, momentum = fetch_play_by_play(g["event_id"])
                        if momentum:
                            st.markdown(f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;padding:6px 10px;border-radius:4px;margin-top:6px;color:#d1fae5;font-weight:700">{momentum}</div>', unsafe_allow_html=True)
                        if plays:
                            st.markdown('<small style="color:#64748b">Recent plays:</small>', unsafe_allow_html=True)
                            for play_txt in plays[:5]:
                                st.markdown(f'<div style="color:#94a3b8;font-size:0.78rem;padding:2px 0">• {play_txt}</div>', unsafe_allow_html=True)

                        # Model note
                        model_pick = t1["name"] if pregame_p >= 0.5 else t2["name"]
                        model_conf = max(pregame_p, 1-pregame_p)
                        agreement  = "✅ Aligns" if (p1 >= 0.5) == (pregame_p >= 0.5) else "⚠️ Diverging"
                        st.markdown(
                            f'<small style="color:#94a3b8">📊 Model pre-game pick: <strong style="color:#f1f5f9">{model_pick}</strong>'
                            f' ({model_conf*100:.0f}% pre-game) — {agreement} from live state</small>',
                            unsafe_allow_html=True)

            # ── RECENTLY FINISHED ─────────────────────────────────────────────
            if recent_fin:
                st.markdown("---")
                st.markdown('<div class="section-header">✅ Final Scores</div>', unsafe_allow_html=True)
                fin_cols = st.columns(min(4, len(recent_fin)))
                for idx, g in enumerate(recent_fin[:4]):
                    t1 = g["team1"]; t2 = g["team2"]
                    winner = t1 if t1["score"] > t2["score"] else t2
                    loser  = t2 if t1["score"] > t2["score"] else t1
                    pregame_p, _, _ = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    model_got_it = (pregame_p >= 0.5) == (t1["score"] > t2["score"])
                    verdict = "✅ Model correct" if model_got_it else "❌ Upset — model wrong"
                    with fin_cols[idx % 4]:
                        st.markdown(
                            f'<div class="team-card">'
                            f'<strong style="color:#4ade80">{winner["name"]} {winner["score"]}</strong>'
                            f'<span style="color:#64748b"> def. </span>'
                            f'{loser["name"]} {loser["score"]}'
                            f'<br/><small style="color:#94a3b8">{verdict}</small>'
                            f'</div>',
                            unsafe_allow_html=True)

            # ── UPCOMING ─────────────────────────────────────────────────────
            if upcoming:
                st.markdown("---")
                st.markdown('<div class="section-header">🕐 Upcoming Games — Pre-Game Model Lines</div>', unsafe_allow_html=True)
                up_cols = st.columns(min(3, len(upcoming)))
                for idx, g in enumerate(upcoming[:6]):
                    t1 = g["team1"]; t2 = g["team2"]
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    fav   = t1["name"] if pregame_p >= 0.5 else t2["name"]
                    fav_p = max(pregame_p, 1-pregame_p)
                    with up_cols[idx % 3]:
                        st.markdown(
                            f'<div class="team-card">'
                            f'<strong style="color:#f1f5f9">{t1["name"]}</strong>'
                            f'<span style="color:#64748b"> vs </span>'
                            f'<strong style="color:#f1f5f9">{t2["name"]}</strong>'
                            f'<br/><small style="color:#94a3b8">Model picks: <strong style="color:#4ade80">{fav}</strong> '
                            f'({fav_p*100:.0f}%) · {american_line(fav_p if fav==t1["name"] else 1-fav_p)}</small>'
                            f'<br/><small style="color:#64748b">Scores: {t1["name"]} {s1:.1f} vs {t2["name"]} {s2:.1f}</small>'
                            f'</div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Data: ESPN public API · Refresh rate: manual or 60s auto · In-game probability blends model pre-game score (dominates early) with live score differential (dominates late). Model principles stay constant — late-game score doesn't override the model, it updates it.")


# ─────────────────────────────────────────────────────────────────────────────
# STATLASBERG Q&A ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def statlasberg_qa(q, bkt_df, s16, e8, ff, champion, champs_df):
    """Rule-based Q&A engine for Statlasberg.  Returns a markdown string."""
    import re as _re
    q_low = q.lower().strip()

    # ── Champion ──────────────────────────────────────────────────────────────
    if any(w in q_low for w in ["champion", "win it all", "cut the nets", "national title",
                                 "going all the way", "best pick", "winner"]):
        champ_pct_str = ""
        if len(champs_df) > 0 and champion in champs_df["team"].values:
            pct = float(champs_df[champs_df["team"] == champion]["championship_pct"].iloc[0])
            champ_pct_str = f" ({pct:.1f}% in 500k simulations)"
        row = bkt_df[bkt_df["team"] == champion]
        score_str = f" — contender score **{safe_f(row.iloc[0].get('contender_score',50)):.1f}/100**" if len(row) else ""
        return (f"🏆 My champion pick is **{champion}**{champ_pct_str}{score_str}.\n\n"
                f"Runner-up: **{sim_runner_up}**.")

    # ── Final Four ────────────────────────────────────────────────────────────
    if "final four" in q_low or "final 4" in q_low or "semifinal" in q_low:
        ff_str = " · ".join(f"**{t}**" for t in ff) if ff else "—"
        return (f"🏀 My Final Four picks: {ff_str}\n\n"
                f"*(Determined by deterministic bracket simulation on contender scores)*")

    # ── Region winner ─────────────────────────────────────────────────────────
    for region in ["east", "west", "south", "midwest"]:
        if region in q_low and any(w in q_low for w in ["win", "who", "pick", "region", "represent"]):
            region_teams = bkt_df[bkt_df["region"].str.lower() == region]["team"].tolist()
            ff_from = [t for t in ff if t in region_teams]
            if ff_from:
                seed_row = bkt_df[bkt_df["team"] == ff_from[0]]
                seed_str = f" (#{int(seed_row.iloc[0]['seed'])} seed)" if len(seed_row) and "seed" in seed_row.columns else ""
                return (f"🏆 **{ff_from[0]}**{seed_str} is my pick to represent the "
                        f"**{region.capitalize()}** in the Final Four.")
            return f"🤔 No Final Four team found for the **{region.capitalize()}** in current simulation."

    # ── Sweet 16 ──────────────────────────────────────────────────────────────
    if "sweet 16" in q_low or "sweet sixteen" in q_low or "second week" in q_low:
        if s16:
            by_region = {}
            for t in s16:
                r = bkt_df[bkt_df["team"] == t]["region"].values
                reg = r[0] if len(r) else "Unknown"
                by_region.setdefault(reg, []).append(t)
            lines = [f"**{reg}**: {', '.join(teams)}" for reg, teams in sorted(by_region.items())]
            return "🎯 My Sweet 16 picks:\n\n" + "\n\n".join(lines)
        return "Sweet 16 data not available — run pipeline first."

    # ── Elite Eight ───────────────────────────────────────────────────────────
    if "elite eight" in q_low or "elite 8" in q_low or "quarterfinal" in q_low:
        if e8:
            e8_str = " · ".join(f"**{t}**" for t in e8)
            return f"🎯 My Elite Eight picks: {e8_str}"
        return "Elite Eight data not available."

    # ── Upset / Cinderella ────────────────────────────────────────────────────
    if any(w in q_low for w in ["upset", "cinderella", "darkhorse", "dark horse", "sleeper", "surprise"]):
        upsets = [(row["team"], int(row["seed"])) for _, row in bkt_df.iterrows()
                  if row.get("seed", 1) >= 10 and row["team"] in (s16 or [])]
        if upsets:
            upsets_sorted = sorted(upsets, key=lambda x: x[1], reverse=True)
            lines = [f"• **#{s} {t}** — model sends them to the Sweet 16" for t, s in upsets_sorted]
            return "💥 Upset picks to make the Sweet 16:\n\n" + "\n".join(lines)
        # Fallback: high-seeded teams with high contender scores
        threats = bkt_df[bkt_df["seed"] >= 10].sort_values("contender_score", ascending=False).head(5)
        if len(threats):
            lines = [f"• **#{int(r['seed'])} {r['team']}** (score: {safe_f(r.get('contender_score',50)):.1f})"
                     for _, r in threats.iterrows()]
            return "💥 Best upset threats this year:\n\n" + "\n".join(lines)
        return "No major upsets predicted — chalk looks strong."

    # ── Fraud / Overrated ─────────────────────────────────────────────────────
    if any(w in q_low for w in ["fraud", "overrated", "avoid", "fade", "trap team", "don't trust"]):
        fraud_col = "fraud_favorite_flag"
        if fraud_col in bkt_df.columns:
            frauds = bkt_df[bkt_df[fraud_col] == True].sort_values("seed")
            if len(frauds):
                lines = [f"• **#{int(r['seed'])} {r['team']}** (score: {safe_f(r.get('contender_score',50)):.1f}, "
                         f"risk: {safe_f(r.get('upset_risk_score',50)):.0f})"
                         for _, r in frauds.iterrows()]
                return "⚠️ **Fraud Favorites** — high seed, model doesn't believe in them:\n\n" + "\n".join(lines)
        return "No clear fraud favorites flagged this year — seeds look legitimate."

    # ── Top / Best Team ───────────────────────────────────────────────────────
    if any(w in q_low for w in ["best team", "top team", "number one", "#1", "strongest", "highest score"]):
        top = bkt_df.sort_values("contender_score", ascending=False).iloc[0]
        return (f"📈 **{top['team']}** has the highest contender score: "
                f"**{safe_f(top.get('contender_score',50)):.1f}/100** "
                f"(#{int(top.get('seed', 0))} seed, {top.get('archetype', '—')})")

    # ── Compare two teams ─────────────────────────────────────────────────────
    vs_match = _re.search(r'(.+?)\s+(?:vs\.?|versus|against|or|beat)\s+(.+)', q_low)
    if vs_match:
        t1_q = vs_match.group(1).strip()
        t2_q = vs_match.group(2).strip().rstrip("?")

        def find_team_row(query):
            for _, row in bkt_df.iterrows():
                if query == row["team"].lower() or query in row["team"].lower() or row["team"].lower() in query:
                    return row
            return None

        r1 = find_team_row(t1_q)
        r2 = find_team_row(t2_q)
        if r1 is not None and r2 is not None:
            sc1 = safe_f(r1.get("contender_score", 50))
            sc2 = safe_f(r2.get("contender_score", 50))
            p1  = win_prob_sigmoid(sc1, sc2)
            fav = r1["team"] if p1 >= 0.5 else r2["team"]
            fav_p = max(p1, 1 - p1)
            dog_p = min(p1, 1 - p1)
            return (f"⚔️ **{r1['team']}** ({sc1:.1f}) vs **{r2['team']}** ({sc2:.1f})\n\n"
                    f"Model favors **{fav}** at **{fav_p*100:.0f}%** "
                    f"({american_line(fav_p)} · opponent {american_line(dog_p)})\n\n"
                    f"*Based on contender scores — doesn't account for bracket seeding position.*")

    # ── Team lookup ───────────────────────────────────────────────────────────
    matched = None
    words = [w for w in q_low.split() if len(w) >= 4]
    for _, row in bkt_df.iterrows():
        t_low = row["team"].lower()
        if t_low in q_low or any(w in t_low for w in words):
            matched = row
            break
    if matched is not None:
        score  = safe_f(matched.get("contender_score", 50))
        risk   = safe_f(matched.get("upset_risk_score", 25))
        arch   = matched.get("archetype", "Unknown")
        seed   = int(matched["seed"]) if matched.get("seed") and str(matched.get("seed")) != "nan" else "—"
        sim_r  = matched.get("sim_round", "First Round")
        def_s  = safe_f(matched.get("defense_score", 50))
        cl_s   = safe_f(matched.get("clutch_score", 50))
        champ_pct_row = ""
        if len(champs_df) and matched["team"] in champs_df["team"].values:
            pct = float(champs_df[champs_df["team"] == matched["team"]]["championship_pct"].iloc[0])
            champ_pct_row = f"\n• Championship probability: **{pct:.1f}%**"
        return (f"📋 **{matched['team']}** — #{seed} seed · {arch}\n\n"
                f"• Contender Score: **{score:.1f}/100**\n"
                f"• Upset Risk: **{risk:.1f}/100**\n"
                f"• Defense Score: **{def_s:.1f}** · Clutch Score: **{cl_s:.1f}**\n"
                f"• Predicted round: **{sim_r}**{champ_pct_row}")

    # ── Default ───────────────────────────────────────────────────────────────
    return ("🤔 I didn't catch that. Try one of these:\n\n"
            "- *Who is your champion pick?*\n"
            "- *Who wins the East?*\n"
            "- *What's your Final Four?*\n"
            "- *Best upset picks?*\n"
            "- *Who are the fraud favorites?*\n"
            "- *Duke vs Kentucky* (head-to-head)\n"
            "- *Tell me about Auburn*")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — ASK STATLASBERG  (Q&A + Model Comparison)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<div class="section-header">🤖 Ask Statlasberg</div>', unsafe_allow_html=True)
    st.caption("Ask about picks, matchups, upsets, or any team — rule-based engine powered by our contender model.")

    # ── Model Comparison ──────────────────────────────────────────────────────
    with st.expander("📊 Model Comparison — Statlasberg vs KenPom-Proxy vs Seed Baseline", expanded=True):
        if len(in_bracket) > 0:
            comp_df = in_bracket.copy()

            # Statlasberg rank: by contender_score descending
            comp_df["statl_rank"] = comp_df["contender_score"].rank(ascending=False, method="min").astype(int)

            # KenPom-proxy: rank by adj_margin if available, else adj_offense - adj_defense
            if "adj_margin" in comp_df.columns and comp_df["adj_margin"].notna().any():
                comp_df["kp_rank"] = comp_df["adj_margin"].rank(ascending=False, method="min").fillna(99).astype(int)
                kp_label = "KenPom-Proxy\n(adj_margin rank)"
            elif "adj_offense" in comp_df.columns and "adj_defense" in comp_df.columns:
                comp_df["adj_net"] = comp_df["adj_offense"].fillna(0) - comp_df["adj_defense"].fillna(0)
                comp_df["kp_rank"] = comp_df["adj_net"].rank(ascending=False, method="min").fillna(99).astype(int)
                kp_label = "KenPom-Proxy\n(adj net rank)"
            else:
                comp_df["kp_rank"] = comp_df["seed"].astype(int)
                kp_label = "KenPom-Proxy\n(seed — no adj data)"

            # Seed baseline rank: within-region seed (1 best per region)
            comp_df["seed_rank"] = comp_df.groupby("region")["seed"].rank(method="min").astype(int)

            # Compute Final Four for each model
            # Statlasberg FF already in sim_ff
            # KenPom-proxy FF: top team by kp_rank in each region
            kp_ff = []
            for reg in comp_df["region"].unique():
                reg_df = comp_df[comp_df["region"] == reg].sort_values("kp_rank")
                if len(reg_df):
                    kp_ff.append(reg_df.iloc[0]["team"])
            # Seed baseline FF: #1 seed in each region
            seed_ff = []
            for reg in comp_df["region"].unique():
                reg_df = comp_df[(comp_df["region"] == reg) & (comp_df["seed"] == 1)]
                if len(reg_df):
                    seed_ff.append(reg_df.iloc[0]["team"])

            # ── Final Four Comparison ─────────────────────────────────────────
            st.markdown("#### 🏀 Final Four Predictions")
            ff_col1, ff_col2, ff_col3 = st.columns(3)
            with ff_col1:
                st.markdown("**🤖 Statlasberg**")
                for t in (sim_ff or []):
                    row = comp_df[comp_df["team"] == t]
                    seed_str = f" (#{int(row.iloc[0]['seed'])})" if len(row) else ""
                    st.markdown(f"• {t}{seed_str}")
                st.markdown(f"**Champion:** {sim_champion}")
            with ff_col2:
                st.markdown(f"**📊 {kp_label.split(chr(10))[0]}**")
                for t in kp_ff:
                    row = comp_df[comp_df["team"] == t]
                    seed_str = f" (#{int(row.iloc[0]['seed'])})" if len(row) else ""
                    st.markdown(f"• {t}{seed_str}")
                st.markdown(f"**Champion:** {kp_ff[0] if kp_ff else '—'}")
            with ff_col3:
                st.markdown("**🎲 Seed Baseline (#1 seeds)**")
                for t in seed_ff:
                    st.markdown(f"• {t} (#1)")
                st.markdown(f"**Champion:** {seed_ff[0] if seed_ff else '—'}")

            st.markdown("---")

            # ── Rankings comparison table ─────────────────────────────────────
            st.markdown("#### 📋 Top 20 Teams — Cross-Model Rankings")
            top20 = comp_df.sort_values("statl_rank").head(20)[
                ["team", "region", "seed", "statl_rank", "kp_rank", "contender_score", "sim_round"]
            ].copy()
            top20.columns = ["Team", "Region", "Seed", "Statlasberg Rank", kp_label.replace("\n", " "), "Score", "Predicted Round"]

            # Color-code rank differences
            def highlight_gap(row):
                try:
                    gap = int(row["Statlasberg Rank"]) - int(row[kp_label.replace(chr(10), " ")])
                    if gap <= -5:   return [""] * len(row) + []
                except Exception:
                    pass
                return [""] * len(row)

            st.dataframe(
                top20.set_index("Team"),
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                    "Statlasberg Rank": st.column_config.NumberColumn("Statl. Rank", format="%d"),
                    kp_label.replace("\n", " "): st.column_config.NumberColumn("KP-Proxy Rank", format="%d"),
                }
            )

            # ── Big disagreements ─────────────────────────────────────────────
            comp_df["rank_gap"] = (comp_df["statl_rank"] - comp_df["kp_rank"]).abs()
            disagree = comp_df[comp_df["rank_gap"] >= 8].sort_values("rank_gap", ascending=False).head(8)
            if len(disagree):
                st.markdown("#### ⚡ Biggest Disagreements Between Models")
                for _, row in disagree.iterrows():
                    gap_dir = "📈 Statlasberg **higher** on" if row["statl_rank"] < row["kp_rank"] else "📉 Statlasberg **lower** on"
                    st.markdown(
                        f'<div style="background:#1e293b;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:0.85rem">'
                        f'{gap_dir} <strong style="color:#f1f5f9">{row["team"]}</strong> '
                        f'<span style="color:#94a3b8">(#{int(row["seed"])} seed)</span> — '
                        f'Statl. rank <strong style="color:#4ade80">#{int(row["statl_rank"])}</strong> · '
                        f'KP-Proxy rank <strong style="color:#f87171">#{int(row["kp_rank"])}</strong>'
                        f'</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Load bracket data to see model comparisons.")

    st.markdown("---")

    # ── Chat Q&A ──────────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask Me Anything About the 2026 Tournament")

    # Suggestion chips
    suggestions = [
        "Who is your champion pick?",
        "What's your Final Four?",
        "Who wins the East?",
        "Best upset picks?",
        "Who are the fraud favorites?",
        "Tell me about Auburn",
    ]
    chip_cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(sug, key=f"chip_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                resp = statlasberg_qa(sug, in_bracket, sim_s16, sim_e8, sim_ff,
                                      sim_champion, champs)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input
    if user_q := st.chat_input("Ask about teams, matchups, upsets, the champion pick…"):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("Thinking…"):
            response = statlasberg_qa(user_q, in_bracket, sim_s16, sim_e8, sim_ff,
                                      sim_champion, champs)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption("Statlasberg Q&A is rule-based — it answers from the contender score model, not an LLM. Responses reflect this bracket's simulation only.")
