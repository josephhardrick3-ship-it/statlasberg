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
import random as _rng
import re as _re
from datetime import datetime
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.models.classify_archetypes import (
        classify_vulnerabilities, classify_strengths, get_matchup_arc
    )
    _ARCHETYPES_OK = True
except Exception:
    _ARCHETYPES_OK = False
    def classify_vulnerabilities(_row): return []
    def classify_strengths(_row):       return []
    def get_matchup_arc(_r1, _r2):      return ""

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
    """Archetype-driven matchup narrative — delegates to the full matchup matrix."""
    arc = get_matchup_arc(dict(t1), dict(t2))
    if arc:
        return arc
    # bare sub-score fallback if archetypes not loaded
    def fv(r, k, d=0.0):
        try: return float(r.get(k, d) or d)
        except: return d
    guard1, guard2 = fv(t1,"guard_play_score",50), fv(t2,"guard_play_score",50)
    def1,   def2   = fv(t1,"defense_score",50),    fv(t2,"defense_score",50)
    n1, n2 = t1.get("team","Team A"), t2.get("team","Team B")
    if abs(def1 - def2) > 15:
        return f"🛡 {'Defense' if def1>def2 else n2+' defense'} controls the pace here"
    if abs(guard1 - guard2) > 12:
        edge = n1 if guard1 > guard2 else n2
        return f"🏀 Clear backcourt edge for {edge} — guard play decides this one"
    return "⚖️ Evenly matched — execution and one momentum run decides this"

def style_matchup_insight_by_name(name1, name2, bkt_df):
    """Look up teams by name and return style matchup insight, or None."""
    r1 = bkt_df[bkt_df["team"] == name1]
    r2 = bkt_df[bkt_df["team"] == name2]
    if len(r1) == 0 or len(r2) == 0:
        return None
    return style_matchup_insight(r1.iloc[0], r2.iloc[0])

def safe_f(v, d=0.0):
    try: return float(v) if pd.notna(v) and v != '' else d
    except: return d

def safe_i(v, d=0):
    try: return int(float(v)) if pd.notna(v) and v != '' else d
    except: return d

_UBP_PODS = [
    ([(1,16),(8,9)], [(5,12),(4,13)], "p0"),
    ([(6,11),(3,14)], [(7,10),(2,15)], "p1"),
]
_UBP_REGIONS = ["East", "South", "West", "Midwest"]


def compute_user_bracket(bkt_df, picks):
    """Compute all bracket matchups based on the user's picks so far.

    Returns a dict with keys R64, R32, S16, E8, FF, Championship.
    Each value is a list of dicts:
        {t1, t2, winner (None if not yet picked), key, region, s1, s2}
    Matches with unresolved prerequisites have t1/t2 == None.
    """
    res = {"R64": [], "R32": [], "S16": [], "E8": [], "FF": [], "Championship": []}

    # Build seed→team lookup per region
    s2t = {}
    for reg in _UBP_REGIONS:
        reg_df = bkt_df[bkt_df["region"] == reg]
        s2t[reg] = {int(r["seed"]): str(r["team"])
                    for _, r in reg_df.iterrows() if pd.notna(r.get("seed"))}

    e8_slots = {reg: [] for reg in _UBP_REGIONS}

    for region in _UBP_REGIONS:
        for pod_a_pairs, pod_b_pairs, pod_label in _UBP_PODS:
            # R64
            for pairs_side, pairs in [("a", pod_a_pairs), ("b", pod_b_pairs)]:
                for s1, s2 in pairs:
                    t1 = s2t[region].get(s1, "")
                    t2 = s2t[region].get(s2, "")
                    if not (t1 and t2):
                        continue
                    key = f"ubp_R64_{region}_{s1}v{s2}"
                    res["R64"].append({"t1": t1, "t2": t2, "winner": picks.get(key),
                                       "key": key, "region": region, "s1": s1, "s2": s2})

            # R32 — depends on R64
            def _r64w(pairs_list):
                return [picks.get(f"ubp_R64_{region}_{s1}v{s2}")
                        for s1, s2 in pairs_list]

            wa = _r64w(pod_a_pairs)
            wb = _r64w(pod_b_pairs)

            for side, side_winners in [("a", wa), ("b", wb)]:
                r32_key = f"ubp_R32_{region}_{pod_label}{side}"
                if all(side_winners) and len(side_winners) == 2:
                    res["R32"].append({"t1": side_winners[0], "t2": side_winners[1],
                                       "winner": picks.get(r32_key), "key": r32_key, "region": region})
                else:
                    res["R32"].append({"t1": None, "t2": None, "winner": None,
                                       "key": r32_key, "region": region})

            # S16 — depends on R32
            r32_w_a = picks.get(f"ubp_R32_{region}_{pod_label}a")
            r32_w_b = picks.get(f"ubp_R32_{region}_{pod_label}b")
            s16_key = f"ubp_S16_{region}_{pod_label}"
            if r32_w_a and r32_w_b:
                res["S16"].append({"t1": r32_w_a, "t2": r32_w_b,
                                   "winner": picks.get(s16_key), "key": s16_key, "region": region})
            else:
                res["S16"].append({"t1": None, "t2": None, "winner": None,
                                   "key": s16_key, "region": region})
            e8_slots[region].append(picks.get(s16_key))

        # E8 — depends on both S16 winners in the region
        e8_key = f"ubp_E8_{region}"
        e8_t1, e8_t2 = (e8_slots[region] + [None, None])[:2]
        if e8_t1 and e8_t2:
            res["E8"].append({"t1": e8_t1, "t2": e8_t2,
                               "winner": picks.get(e8_key), "key": e8_key, "region": region})
        else:
            res["E8"].append({"t1": None, "t2": None, "winner": None,
                              "key": e8_key, "region": region})

    # FF — East/Midwest and South/West (mirroring simulate_bracket_full)
    e8w = [picks.get(f"ubp_E8_{reg}") for reg in _UBP_REGIONS]  # East, South, West, Midwest
    ff_pairs = [(e8w[0], e8w[3], "East/Midwest"), (e8w[1], e8w[2], "South/West")]
    for ff_idx, (ff_t1, ff_t2, ff_label) in enumerate(ff_pairs):
        ff_key = f"ubp_FF_{ff_idx}"
        if ff_t1 and ff_t2:
            res["FF"].append({"t1": ff_t1, "t2": ff_t2,
                               "winner": picks.get(ff_key), "key": ff_key, "region": ff_label})
        else:
            res["FF"].append({"t1": None, "t2": None, "winner": None,
                              "key": ff_key, "region": ff_label})

    # Championship
    champ_key = "ubp_Champ"
    ff_w = [picks.get(f"ubp_FF_{i}") for i in range(2)]
    if ff_w[0] and ff_w[1]:
        res["Championship"].append({"t1": ff_w[0], "t2": ff_w[1],
                                    "winner": picks.get(champ_key), "key": champ_key, "region": "National"})
    else:
        res["Championship"].append({"t1": None, "t2": None, "winner": None,
                                    "key": champ_key, "region": "National"})
    return res


def build_round_matchups(bkt_df):
    """Return matchups for every round as a dict: round → list of (t1, t2, winner, loser, region)."""
    rounds = {"R64": [], "R32": [], "S16": [], "E8": [], "FF": [], "Championship": []}
    score_lkp = {r["team"]: safe_f(r.get("contender_score", 50)) for _, r in bkt_df.iterrows()}

    def pwin(t1, t2):
        if not t1: return t2, t1
        if not t2: return t1, t2
        p = win_prob_sigmoid(score_lkp.get(t1, 50), score_lkp.get(t2, 50))
        return (t1, t2) if p >= 0.5 else (t2, t1)

    PODS = [
        ([(1,16),(8,9)],  [(5,12),(4,13)]),
        ([(6,11),(3,14)], [(7,10),(2,15)]),
    ]
    ff_teams = []
    for region in ["East", "South", "West", "Midwest"]:
        reg = bkt_df[bkt_df["region"] == region]
        s2t = {int(r["seed"]): r["team"] for _, r in reg.iterrows() if pd.notna(r.get("seed"))}
        region_e8 = []
        for pod_a_pairs, pod_b_pairs in PODS:
            r1w_a, r1w_b = [], []
            for s1, s2 in pod_a_pairs:
                t1n, t2n = s2t.get(s1, ""), s2t.get(s2, "")
                if t1n or t2n:
                    w, l = pwin(t1n, t2n)
                    rounds["R64"].append((t1n, t2n, w, l, region))
                    if w: r1w_a.append(w)
            for s1, s2 in pod_b_pairs:
                t1n, t2n = s2t.get(s1, ""), s2t.get(s2, "")
                if t1n or t2n:
                    w, l = pwin(t1n, t2n)
                    rounds["R64"].append((t1n, t2n, w, l, region))
                    if w: r1w_b.append(w)
            # R32
            pod_a_s16 = ""
            if len(r1w_a) == 2:
                w, l = pwin(r1w_a[0], r1w_a[1])
                rounds["R32"].append((r1w_a[0], r1w_a[1], w, l, region))
                pod_a_s16 = w
            elif r1w_a:
                pod_a_s16 = r1w_a[0]
            pod_b_s16 = ""
            if len(r1w_b) == 2:
                w, l = pwin(r1w_b[0], r1w_b[1])
                rounds["R32"].append((r1w_b[0], r1w_b[1], w, l, region))
                pod_b_s16 = w
            elif r1w_b:
                pod_b_s16 = r1w_b[0]
            # S16
            if pod_a_s16 and pod_b_s16:
                w, l = pwin(pod_a_s16, pod_b_s16)
                rounds["S16"].append((pod_a_s16, pod_b_s16, w, l, region))
                region_e8.append(w)
        # E8
        if len(region_e8) == 2:
            w, l = pwin(region_e8[0], region_e8[1])
            rounds["E8"].append((region_e8[0], region_e8[1], w, l, region))
            ff_teams.append(w)
    # Final Four
    champ_teams = []
    ff_pairs = [(ff_teams[0], ff_teams[3]), (ff_teams[1], ff_teams[2])] if len(ff_teams) >= 4 else []
    for t1n, t2n in ff_pairs:
        w, l = pwin(t1n, t2n)
        rounds["FF"].append((t1n, t2n, w, l, "National"))
        champ_teams.append(w)
    # Championship
    if len(champ_teams) == 2:
        w, l = pwin(champ_teams[0], champ_teams[1])
        rounds["Championship"].append((champ_teams[0], champ_teams[1], w, l, "National"))
    return rounds


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
        s2t = {int(r["seed"]): r["team"] for _, r in reg.iterrows() if pd.notna(r.get("seed"))}

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
SCORES_PATH   = "data/outputs/team_scores.csv"
CHAMPS_PATH   = "data/outputs/simulation_results.csv"
BRACKET_PATH  = "data/brackets/bracket_2026.csv"
COACHES_PATH  = "data/coaches_2026.csv"
RESULTS_PATH  = "data/tournament_2026/results.csv"
BACKTEST_PATH = "data/outputs/backtest_results.csv"

COACH_COLS = ["team","coach","coach_years_at_school","coach_ncaa_games",
              "coach_sweet16s","coach_finalfours","first_year_coach_flag"]

@st.cache_data(ttl=60)  # re-read files every 60 s so pipeline updates show immediately
def load_data():
    scores  = pd.read_csv(SCORES_PATH)  if os.path.exists(SCORES_PATH)  else pd.DataFrame()
    champs  = pd.read_csv(CHAMPS_PATH)  if os.path.exists(CHAMPS_PATH)  else pd.DataFrame()
    bracket = pd.read_csv(BRACKET_PATH) if os.path.exists(BRACKET_PATH) else pd.DataFrame()

    # ── Normalize bracket team names to match team_scores canonical names ──
    # When using real Sports-Reference data, team names already match between
    # bracket and scores (both use full names like "Brigham Young", "Pittsburgh").
    # When using sample data, abbreviated names need mapping.
    # Smart: only apply normalization if the original name ISN'T in scores but
    # the normalized alias IS — so real data passes through untouched.
    _BRACKET_NORM = {
        # Sample-data aliases → full SR names (no-ops on real data)
        "BYU": "Brigham Young",
        "TCU": "Texas Christian",
        "Saint Marys CA": "Saint Mary's",
        "Miami FL": "Miami",
        "VCU": "Virginia Commonwealth",
        "SMU": "Southern Methodist",
        "St Johns NY": "St. John's",
        "Pitt": "Pittsburgh",
        "UMBC": "Maryland-Baltimore County",
        "LIU": "Long Island University",
        # Also handle common display variants that may appear in custom brackets
        "St. Mary's": "Saint Mary's",
        "Saint John's": "St. John's",
        "NC State": "North Carolina State",
        "N.C. State": "North Carolina State",
        "Miami (FL)": "Miami",
        "McNeese State": "McNeese",
        "Prairie View": "Prairie View A&M",
    }
    if len(bracket) > 0 and "team" in bracket.columns:
        score_teams = set(scores["team"].tolist()) if len(scores) > 0 else set()
        def _smart_norm(t: str) -> str:
            if t in score_teams or not score_teams:
                return t   # already matches scores — no change needed
            normalized = _BRACKET_NORM.get(t, t)
            return normalized if normalized in score_teams else t
        bracket["team"] = bracket["team"].map(_smart_norm)
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

# Session state for deep-dive navigation, chat, and eye-test notes
if "dive_team" not in st.session_state:
    st.session_state.dive_team = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eye_test_notes" not in st.session_state:
    st.session_state.eye_test_notes = {}
if "user_bracket_picks" not in st.session_state:
    st.session_state.user_bracket_picks = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 Statlasberg")
    st.markdown("*2026 March Madness Intelligence*")
    st.divider()
    bracket_teams = sorted(in_bracket["team"].tolist()) if len(in_bracket) > 0 else sorted(scores["team"].tolist())
    # Show currently viewed team as sidebar info (selectbox lives inside the tab)
    _cur_team = st.session_state.get("team_selectbox", bracket_teams[0] if bracket_teams else "—")
    st.markdown(f"**🔍 Viewing:** {_cur_team}")
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏅 Rankings", "🎯 Sweet 16 Picks", "🔍 Team Deep Dive",
    "📊 Model vs Committee", "🎲 Championship Odds", "🏆 Bracket", "📺 Live",
    "📊 Model Comparison", "📈 Model Accuracy"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">2026 Tournament Field — Model Rankings</div>', unsafe_allow_html=True)

    display_df = in_bracket.sort_values("contender_score", ascending=False).reset_index(drop=True)

    # Add hot/trend column
    if "last10_win_pct" in display_df.columns:
        display_df["trend"] = display_df.apply(hot_label, axis=1)

    # ── CSS: make rank-row buttons look like text links ────────────────────
    st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="secondary"].rank-btn {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    text-align: left !important;
    cursor: pointer !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
}
div[data-testid="stButton"] > button[kind="secondary"].rank-btn:hover {
    color: #f97316 !important;
}
</style>""", unsafe_allow_html=True)

    st.caption("👆 Click any team name to load their full breakdown in Team Deep Dive")

    # ── Column header row ──────────────────────────────────────────────────
    hc = st.columns([0.4, 2.2, 0.55, 0.75, 1.0, 1.0, 2.2, 1.5])
    for hdr, col in zip(["#", "TEAM", "SEED", "REGION", "SCORE", "RISK", "ARCHETYPE", "ROUND"], hc):
        col.markdown(f"<span style='color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em'>{hdr}</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:2px 0 4px;border-color:#1e293b'>", unsafe_allow_html=True)

    # ── Scrollable ranked rows ─────────────────────────────────────────────
    for idx, row in display_df.iterrows():
        rank    = idx + 1
        team    = row["team"]
        seed_v  = int(row["seed"]) if "seed" in display_df.columns and not pd.isna(row.get("seed")) else "—"
        region  = row.get("region", "—")
        score   = safe_f(row.get("contender_score", 50))
        risk    = safe_f(row.get("upset_risk_score", 25))
        arch    = row.get("archetype", "—")
        sim_r   = row.get("sim_round", "—")
        trend   = row.get("trend", "")

        # Color logic
        score_col = "#86efac" if score >= 70 else "#4ade80" if score >= 60 else "#94a3b8"
        risk_col  = "#fca5a5" if risk >= 40 else "#fdba74" if risk >= 30 else "#86efac"
        seed_bg   = "#7c3aed" if seed_v == 1 else "#1e3a8a" if isinstance(seed_v, int) and seed_v <= 4 else "#374151"
        round_col = "#f97316" if "Champion" in str(sim_r) else "#c084fc" if "Final Four" in str(sim_r) else "#4ade80" if "Elite" in str(sim_r) else "#94a3b8"

        rc = st.columns([0.4, 2.2, 0.55, 0.75, 1.0, 1.0, 2.2, 1.5])
        with rc[0]:
            st.markdown(f"<span style='color:#475569;font-size:0.8rem'>{rank}</span>", unsafe_allow_html=True)
        with rc[1]:
            trend_html = f" <span style='font-size:0.72rem;color:#4ade80'>{trend}</span>" if trend else ""
            # Button styled as clickable link
            if st.button(f"🔍 {team}", key=f"rank_team_{rank}", help=f"Open {team} in Team Deep Dive",
                         use_container_width=True, type="secondary"):
                st.session_state["team_selectbox"] = team
                st.session_state.dive_team = team
                st.rerun()
        with rc[2]:
            st.markdown(f"<span style='background:{seed_bg};color:#fff;padding:1px 5px;border-radius:3px;font-size:0.8rem;font-weight:700'>{seed_v}</span>", unsafe_allow_html=True)
        with rc[3]:
            st.markdown(f"<span style='color:#94a3b8;font-size:0.82rem'>{str(region)[:4]}</span>", unsafe_allow_html=True)
        with rc[4]:
            st.markdown(
                f"<div style='background:#1a2230;border-radius:4px;overflow:hidden;height:20px'>"
                f"<div style='width:{min(score,100):.0f}%;background:{score_col};height:100%;display:flex;align-items:center;padding-left:4px'>"
                f"<span style='color:#0f172a;font-size:0.75rem;font-weight:700'>{score:.1f}</span></div></div>",
                unsafe_allow_html=True)
        with rc[5]:
            st.markdown(f"<span style='color:{risk_col};font-size:0.85rem;font-weight:700'>{risk:.0f}</span>", unsafe_allow_html=True)
        with rc[6]:
            st.markdown(f"<span style='color:#94a3b8;font-size:0.8rem'>{arch}</span>", unsafe_allow_html=True)
        with rc[7]:
            st.markdown(f"<span style='color:{round_col};font-size:0.8rem'>{sim_r}</span>", unsafe_allow_html=True)

        # Thin divider
        if rank % 4 == 0:
            st.markdown("<hr style='margin:2px 0;border-color:#1e293b;opacity:.4'>", unsafe_allow_html=True)

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
                    seed_val = int(row["seed"]) if pd.notna(row.get("seed")) else 0
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
                        f'<span class="seed-badge" style="background:#b45309">#{int(r["seed"]) if pd.notna(r.get("seed")) else "?"}</span> '
                        f'<strong class="team-name">{ff_t}</strong> '
                        f'<span style="color:#94a3b8;font-size:0.8rem">({r["region"]})</span>'
                        f'</div>',
                        unsafe_allow_html=True)
        with ff_s16b:
            st.markdown('<div class="section-header">🏆 Model\'s Champion</div>', unsafe_allow_html=True)
            if sim_champion:
                row_ch = in_bracket[in_bracket["team"]==sim_champion]
                ch_score = safe_f(row_ch.iloc[0]["contender_score"] if len(row_ch) > 0 else 70)
                ch_seed  = int(row_ch.iloc[0]["seed"]) if len(row_ch) > 0 and pd.notna(row_ch.iloc[0].get("seed")) else 1
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
                    matchup_seed = 17 - (int(row["seed"]) if pd.notna(row.get("seed")) else 8)
                    opp = in_bracket[(in_bracket["region"]==row.get("region","")) & (in_bracket["seed"]==matchup_seed)]
                    opp_name = opp.iloc[0]["team"] if len(opp) > 0 else f"#{matchup_seed} seed"
                    wp = win_prob_sigmoid(row["contender_score"], safe_f(opp.iloc[0]["contender_score"] if len(opp)>0 else 60))
                    hl_up = hot_label(row)
                    hl_up_span = f" &nbsp;<span style='font-size:0.8rem'>{hl_up}</span>" if hl_up else ""
                    st.markdown(
                        f'<div class="team-card" style="border-color:#dc2626">'
                        f'<span style="color:#f97316;font-weight:800">#{int(row["seed"]) if pd.notna(row.get("seed")) else "?"} {row["team"]}</span>'
                        f'<span style="color:#94a3b8"> over </span>'
                        f'<span style="color:#f1f5f9">#{matchup_seed} {opp_name}</span>{hl_up_span}'
                        f'<br/><small style="color:#4ade80">Score: {row["contender_score"]:.1f} · Model win%: {wp*100:.0f}%</small>'
                        f'</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TEAM DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    # ── Team Picker — prominent in-tab selector ───────────────────────────────
    _all_teams   = sorted(scores["team"].tolist())
    _show_all    = st.toggle("Show all D-I teams (not just bracket)", key="dd_show_all", value=False)
    _pick_pool   = _all_teams if _show_all else bracket_teams

    # Build seed-annotated labels: "#12 Akron (Midwest)" / "Akron" if not in bracket
    _seed_map = {}
    if len(in_bracket) > 0 and "seed" in in_bracket.columns:
        for _, _br in in_bracket.iterrows():
            _seed_map[_br["team"]] = (int(_br["seed"]), _br.get("region", ""))
    _pick_labels = [
        f"#{_seed_map[t][0]} {t} · {_seed_map[t][1]}" if t in _seed_map else t
        for t in _pick_pool
    ]
    _label_to_team = dict(zip(_pick_labels, _pick_pool))

    # Resolve default index — honours Rankings clickthrough via session state.
    # Guard against stale session state holding a label from "all D-I" mode
    # that no longer exists in the current (bracket-only) pool.
    _cur = st.session_state.get("team_selectbox", _pick_pool[0] if _pick_pool else "")
    # _cur could be either a label string or a raw team name — normalise to team name
    _cur_team = _label_to_team.get(_cur, _cur)   # if _cur is already a label, resolve it
    if _cur_team not in _pick_pool:              # team not in current pool → reset to first
        _cur_team = _pick_pool[0] if _pick_pool else ""
    _cur_label = next((lbl for lbl, tm in _label_to_team.items() if tm == _cur_team),
                      _pick_labels[0] if _pick_labels else "")
    _default_idx = _pick_labels.index(_cur_label) if _cur_label in _pick_labels else 0

    _pick_col, _nav_col = st.columns([5, 1])
    with _pick_col:
        _selected_label = st.selectbox(
            "🔍 Select team",
            _pick_labels,
            index=_default_idx,
            key="team_selectbox",
            label_visibility="collapsed",
        )
    with _nav_col:
        _tidx = _pick_labels.index(_selected_label) if _selected_label in _pick_labels else 0
        _col_prev, _col_next = st.columns(2)
        with _col_prev:
            if st.button("◀", key="dd_prev", help="Previous team",
                         disabled=_tidx == 0):
                st.session_state["team_selectbox"] = _pick_labels[_tidx - 1]
                st.rerun()
        with _col_next:
            if st.button("▶", key="dd_next", help="Next team",
                         disabled=_tidx == len(_pick_labels) - 1):
                st.session_state["team_selectbox"] = _pick_labels[_tidx + 1]
                st.rerun()

    selected_team = _label_to_team.get(_selected_label, _pick_pool[0] if _pick_pool else "")
    st.markdown("---")

    row_data = scores[scores["team"] == selected_team]
    if len(row_data) == 0:
        st.warning(f"⚠️ No data found for **{selected_team}**. Select another team above.")
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
    m6.metric("Pred. Round",      row.get("sim_round", row.get("expected_round","—")) or "—")

    # ── Advanced Stats Row ────────────────────────────────────────────────────
    adv_cols = st.columns(5)
    adj_off_v  = safe_f(row.get("adj_offense"))
    adj_def_v  = safe_f(row.get("adj_defense"))
    tempo_v    = safe_f(row.get("tempo"))
    three_pa_v = safe_f(row.get("three_pa_rate"))
    ft_rate_v  = safe_f(row.get("ft_rate"))
    to_pct_v   = safe_f(row.get("turnover_pct"))
    adv_cols[0].metric("Adj. Offense",   f"{adj_off_v:.1f}" if adj_off_v > 0 else "—")
    adv_cols[1].metric("Adj. Defense",   f"{adj_def_v:.1f}" if adj_def_v > 0 else "—",
                       help="Lower = better; D1 avg ≈ 100")
    adv_cols[2].metric("Tempo (poss/g)", f"{tempo_v:.0f}" if tempo_v > 55 else "—")
    adv_cols[3].metric("3PA Rate",       f"{three_pa_v:.1f}/g" if three_pa_v > 0 else "—")
    adv_cols[4].metric("TO Rate",        f"{to_pct_v:.1f} TO/g" if to_pct_v > 0 else "—",
                       help="Turnovers per game (D1 avg ≈ 13-14)")

    # ── Archetype Banner ──────────────────────────────────────────────────────
    arch       = row.get("archetype", "Solid Tournament Team") or "Solid Tournament Team"
    row_dict   = dict(row)
    _vtags = row.get("vuln_tags", ""); _vtags = str(_vtags) if pd.notna(_vtags) else ""
    _stags = row.get("strength_tags", ""); _stags = str(_stags) if pd.notna(_stags) else ""
    vuln_list  = [t for t in (_vtags or "").split(" | ") if t.strip()] or classify_vulnerabilities(row_dict)
    str_list   = [t for t in (_stags or "").split(" | ") if t.strip()] or classify_strengths(row_dict)

    ARCH_COLORS = {
        "Blue-Blood Dominant":   ("#7c3aed", "#ede9fe"),
        "Grind-It-Out Defense":  ("#0284c7", "#e0f2fe"),
        "Veteran Control":       ("#0891b2", "#ecfeff"),
        "Pace-and-Space Gunners":("#ea580c", "#fff7ed"),
        "Glass & Paint":         ("#4d7c0f", "#f7fee7"),
        "Cinderella Profile":    ("#db2777", "#fdf2f8"),
        "One-Man Show":          ("#d97706", "#fffbeb"),
        "Freshman Loaded":       ("#dc2626", "#fef2f2"),
        "Résumé Builder":        ("#9ca3af", "#f9fafb"),
    }
    a_bg, a_txt = ARCH_COLORS.get(arch, ("#374151", "#f1f5f9"))
    str_html  = "".join(f'<span style="background:#14532d;color:#86efac;border-radius:4px;padding:2px 8px;font-size:0.78rem;margin:2px">{s}</span>' for s in str_list)
    vuln_html = "".join(f'<span style="background:#450a0a;color:#fca5a5;border-radius:4px;padding:2px 8px;font-size:0.78rem;margin:2px">{v}</span>' for v in vuln_list)

    st.markdown(
        f'<div style="background:#1a1f2e;border:1px solid #374151;border-radius:10px;padding:12px 16px;margin:8px 0">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<span style="background:{a_bg};color:{a_txt};border-radius:6px;padding:4px 12px;font-weight:800;font-size:0.95rem">{arch}</span>'
        f'</div>'
        f'<div style="margin-bottom:6px">'
        f'<span style="color:#4ade80;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em">WHAT THEY BRING &nbsp;</span>{str_html}'
        f'</div>'
        f'<div>'
        f'<span style="color:#f87171;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em">WATCH OUT FOR &nbsp;</span>{vuln_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True)

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
        st.pyplot(fig)
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
        st.pyplot(fig2)
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
                actual_seed = int(r["seed"]) if pd.notna(r.get("seed")) else 0
                mseed = int(r["model_seed"]) if pd.notna(r.get("model_seed")) else 0
                gap = int(r["seed_gap"]) if pd.notna(r.get("seed_gap")) else 0
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
                actual_seed = int(r["seed"]) if pd.notna(r.get("seed")) else 0
                mseed = int(r["model_seed"]) if pd.notna(r.get("model_seed")) else 0
                gap = abs(int(r["seed_gap"])) if pd.notna(r.get("seed_gap")) else 0
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
    st.markdown('<div class="section-header">Championship Probabilities — 500,000 Bracket Simulations</div>', unsafe_allow_html=True)

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
                    <div style="color:#b0bbd0;font-size:0.78rem">Model line: {american_line(r['championship_pct']/100)}</div>
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
        ax3.set_title("Simulated Championship Probabilities (500,000 runs)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
        st.pyplot(fig3)
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
        all_round_matchups = build_round_matchups(in_bracket)
        ubp = compute_user_bracket(in_bracket, st.session_state.user_bracket_picks)

        # Top-level mode selector
        bkt_mode = st.radio(
            "Bracket mode",
            ["🎯 My Picks", "🤖 Model Bracket", "⚔️ Compare"],
            horizontal=True,
            label_visibility="collapsed",
            key="bkt_mode_radio"
        )
        st.markdown("---")

        # ── helper: render model-bracket matchup expander ──────────────────────
        def _render_matchup_exp(t1_name, t2_name, winner, loser, region, rnd_key):
            """Render a matchup expander for R32+ rounds."""
            if not t1_name or not t2_name:
                return
            t1_row = in_bracket[in_bracket["team"] == t1_name]
            t2_row = in_bracket[in_bracket["team"] == t2_name]
            t1 = t1_row.iloc[0] if len(t1_row) else None
            t2 = t2_row.iloc[0] if len(t2_row) else None
            if t1 is None or t2 is None:
                return
            c1 = safe_f(t1.get("contender_score", 50))
            c2 = safe_f(t2.get("contender_score", 50))
            p1 = win_prob_sigmoid(c1, c2)
            p2 = 1 - p1
            s1 = int(t1.get("seed")) if pd.notna(t1.get("seed")) else 0
            s2 = int(t2.get("seed")) if pd.notna(t2.get("seed")) else 0
            with st.expander(
                f"#{s1} {t1_name}  vs  #{s2} {t2_name}  —  Projected: **{winner}**",
                expanded=False
            ):
                mh1, mh2 = st.columns(2)
                for col, team, c, p, seed_n in [(mh1, t1, c1, p1, s1), (mh2, t2, c2, p2, s2)]:
                    with col:
                        is_w = (team["team"] == winner)
                        bdr = "#f97316" if is_w else "#374151"
                        bar_bg = "#4ade80" if is_w else "#64748b"
                        tname = team["team"]
                        st.markdown(
                            f'<div style="padding:8px;background:#131820;border:2px solid {bdr};border-radius:8px">'
                            f'<span style="background:{"#f97316" if is_w else "#475569"};color:white;border-radius:50%;'
                            f'width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;'
                            f'font-size:0.68rem;font-weight:800;margin-right:6px">{seed_n}</span>'
                            f'<strong style="color:#f1f5f9;font-size:0.95rem">{tname}</strong>'
                            f'{"  🏆 Projected Winner" if is_w else ""}'
                            f'<br/><small style="color:#94a3b8">Score: {c:.1f}</small>'
                            f'<div style="background:#1e293b;border-radius:3px;height:4px;margin:4px 0">'
                            f'<div style="width:{int(p*100)}%;background:{bar_bg};height:4px;border-radius:3px"></div></div>'
                            f'<span style="color:{"#4ade80" if is_w else "#94a3b8"};font-weight:700;font-size:0.9rem">'
                            f'{p*100:.0f}% · {american_line(p)}</span>'
                            f'</div>', unsafe_allow_html=True)
                        if st.button(f"🔍 {tname} Deep Dive", key=f"dd_{rnd_key}_{tname}", use_container_width=True):
                            st.session_state["team_selectbox"] = tname
                            st.session_state.dive_team = tname
                            st.rerun()
                insight = style_matchup_insight(t1, t2)
                st.markdown(f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;padding:6px 10px;margin:5px 0;color:#d1fae5;font-size:0.82rem">💡 {insight}</div>', unsafe_allow_html=True)
                fav_row = t1 if p1 >= p2 else t2
                dog_row = t2 if p1 >= p2 else t1
                fav_cl = safe_f(fav_row.get("clutch_score", 50))
                dog_cl = safe_f(dog_row.get("clutch_score", 50))
                if abs(fav_cl - dog_cl) > 8:
                    cl_edge = fav_row["team"] if fav_cl > dog_cl else dog_row["team"]
                    st.markdown(f'<small style="color:#fbbf24">⚡ Clutch edge: <strong>{cl_edge}</strong> (diff: {abs(fav_cl-dog_cl):.0f} pts)</small>', unsafe_allow_html=True)
                flag_msgs = []
                if fav_row.get("fraud_favorite_flag", False): flag_msgs.append(f"⚠️ {fav_row['team']} is a Fraud Favorite")
                if dog_row.get("dangerous_low_seed_flag", False): flag_msgs.append(f"💥 {dog_row['team']} is a Dangerous Low Seed")
                if dog_row.get("cinderella_flag", False): flag_msgs.append(f"🪄 {dog_row['team']} has Cinderella traits")
                for msg in flag_msgs:
                    st.markdown(f'<small style="color:#f87171">{msg}</small>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:4px">📍 {region} Region</div>', unsafe_allow_html=True)

        # ── helper: render a My Picks matchup card (pick buttons) ───────────────
        def _render_pick_card(m, picks, score_lkp, model_winner):
            """Show a pick card for one matchup. m is a dict from compute_user_bracket."""
            t1, t2 = m["t1"], m["t2"]
            if not t1 or not t2:
                st.markdown('<small style="color:#475569">TBD — complete earlier rounds first</small>', unsafe_allow_html=True)
                return
            key = m["key"]
            current_pick = m["winner"]
            c1 = score_lkp.get(t1, 50)
            c2 = score_lkp.get(t2, 50)
            p1 = win_prob_sigmoid(c1, c2)
            p2 = 1 - p1
            model_fav = t1 if p1 >= p2 else t2

            pc1, pc2 = st.columns(2)
            for col, team, p, cs in [(pc1, t1, p1, c1), (pc2, t2, p2, c2)]:
                with col:
                    picked = (current_pick == team)
                    is_model = (team == model_fav)
                    bdr = "#4ade80" if picked else ("#f97316" if is_model else "#374151")
                    label = f"{'✅ ' if picked else ''}{team}"
                    st.markdown(
                        f'<div style="background:#131820;border:2px solid {bdr};border-radius:8px;padding:8px;margin-bottom:4px">'
                        f'<div style="font-size:0.9rem;font-weight:700;color:#f1f5f9">{label}</div>'
                        f'<div style="color:#64748b;font-size:0.72rem">Score: {cs:.1f} · Model: {p*100:.0f}%</div>'
                        f'</div>', unsafe_allow_html=True)
                    tname = team
                    if st.button(f"Pick {tname}", key=f"ubpick_{key}_{tname}", use_container_width=True,
                                 type="primary" if picked else "secondary"):
                        st.session_state.user_bracket_picks[key] = tname
                        st.rerun()

            if current_pick:
                st.markdown(
                    f'<small style="color:{"#4ade80" if current_pick == model_fav else "#f87171"}">'
                    f'Your pick: <strong>{current_pick}</strong>'
                    f'{" ✓ agrees with model" if current_pick == model_fav else f" — model had {model_fav}"}'
                    f'</small>', unsafe_allow_html=True)
                if st.button("↩ Change", key=f"ubp_chg_{key}", use_container_width=False):
                    st.session_state.user_bracket_picks.pop(key, None)
                    st.rerun()
            else:
                st.markdown(f'<small style="color:#475569">Model pick: <strong>{model_fav}</strong> ({max(p1,p2)*100:.0f}%)</small>', unsafe_allow_html=True)

        # Score lookup for pick cards
        _score_lkp = {str(r["team"]): safe_f(r.get("contender_score", 50)) for _, r in in_bracket.iterrows()}

        # ── MODEL BRACKET MODE ──────────────────────────────────────────────────
        if bkt_mode == "🤖 Model Bracket":
            br_r64, br_r32, br_s16, br_e8, br_ff, br_champ = st.tabs([
                "🎯 Round of 64", "⚡ Round of 32", "🔥 Sweet 16", "💎 Elite 8", "🏅 Final Four", "🏆 Championship"
            ])

            with br_r64:
                st.caption("First-round matchups by region. Model line vs. seed baseline. Log results to track model accuracy.")
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

                            # Model vs seed-baseline edge (NOT live sportsbook lines)
                            edge = abs(p1 - hist_p1)
                            edge_tag = ""
                            if p1 > hist_p1 + 0.12:
                                edge_tag = f"<span style='color:#4ade80;font-size:0.7rem;font-weight:700'> 📐 Model diverges — higher on #{s1}</span>"
                            elif p1 < hist_p1 - 0.12:
                                edge_tag = f"<span style='color:#f59e0b;font-size:0.7rem;font-weight:700'> 📐 Model diverges — higher on #{s2}</span>"

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
                                    t1_name_str = t1["team"]
                                    if st.button(f"🔍 {t1_name_str} Deep Dive", key=f"dd1_{matchup_key}", use_container_width=True):
                                        st.session_state["team_selectbox"] = t1_name_str
                                        st.session_state.dive_team = t1_name_str
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
                                    t2_name_str = t2["team"]
                                    if st.button(f"🔍 {t2_name_str} Deep Dive", key=f"dd2_{matchup_key}", use_container_width=True):
                                        st.session_state["team_selectbox"] = t2_name_str
                                        st.session_state.dive_team = t2_name_str
                                        st.rerun()

                                # Model line vs seed baseline (not live sportsbook lines)
                                st.markdown(
                                    f'<div style="background:#1e293b;border-radius:6px;padding:6px 10px;margin:5px 0;font-size:0.8rem">'
                                    f'<strong style="color:#b0bbd0">🎯 Statlasberg Line:</strong> '
                                    f'<span style="color:#f1f5f9">#{s1} <strong style="color:#4ade80">{american_line(p1)}</strong> &nbsp;/&nbsp; '
                                    f'#{s2} <strong style="color:#4ade80">{american_line(p2)}</strong></span>'
                                    f'&nbsp;&nbsp;<span style="color:#475569;font-size:0.72rem">Seed baseline: #{s1} {american_line(hist_p1)} / #{s2} {american_line(1-hist_p1)}</span>'
                                    f'{edge_tag}'
                                    f'</div>'
                                    f'<div style="font-size:0.68rem;color:#475569;padding:2px 10px">For live sportsbook lines check DraftKings · FanDuel · ESPN Bet</div>',
                                    unsafe_allow_html=True)

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

                                # ── Log result + user pick ───────────────────────────────────
                                if not result_winner:
                                    st.markdown("---")
                                    st.markdown('<small style="color:#94a3b8">Log result:</small>', unsafe_allow_html=True)
                                    rb1, rb2 = st.columns(2)

                                    # Optional: user's own pick (if they disagreed with model)
                                    user_disagree = st.checkbox(
                                        f"🙅 I had the other team",
                                        key=f"dis_{matchup_key}",
                                        help="Check if you disagreed with Statlasberg's pick")
                                    user_pick_val = None
                                    user_note_val = ""
                                    if user_disagree:
                                        dog_team = t2["team"] if fav["team"] == t1["team"] else t1["team"]
                                        user_pick_val = dog_team
                                        user_note_val = st.text_input(
                                            "Why? (optional — Statlasberg will analyze it after):",
                                            key=f"unote_{matchup_key}",
                                            placeholder="e.g. better guard matchup, they're peaking at the right time…")

                                    def _log_result(winner_team):
                                        rec = {
                                            "matchup":    matchup_key,
                                            "winner":     winner_team,
                                            "teams":      [t1["team"], t2["team"]],
                                            "model_pick": fav["team"],
                                            "fav_team":   fav["team"],
                                            "user_pick":  user_pick_val,
                                            "user_note":  user_note_val,
                                            "timestamp":  str(datetime.now().date())}
                                        st.session_state.results.append(rec)
                                        os.makedirs("data/tournament_2026", exist_ok=True)
                                        pd.DataFrame(st.session_state.results).to_csv(RESULTS_PATH, index=False)
                                        st.rerun()

                                    t1n_btn = t1["team"]
                                    t2n_btn = t2["team"]
                                    if rb1.button(f"✅ {t1n_btn} won", key=f"r1_{matchup_key}"):
                                        _log_result(t1n_btn)
                                    if rb2.button(f"✅ {t2n_btn} won", key=f"r2_{matchup_key}"):
                                        _log_result(t2n_btn)

                                else:
                                    correct_str   = "✅ Called it." if result_winner == fav["team"] else "❌ Wrong."
                                    user_pick_rec = result_rec.get("user_pick")
                                    user_note_rec = result_rec.get("user_note", "")

                                    # Model verdict line
                                    st.markdown(
                                        f'<div style="background:#1e293b;border-radius:6px;padding:6px 10px;margin-top:6px">'
                                        f'<strong style="color:#f1f5f9">Result: {result_winner} won — Statlasberg {correct_str}</strong>'
                                        f'</div>', unsafe_allow_html=True)

                                    # User pick verdict (shown when they logged a disagreement)
                                    if user_pick_rec:
                                        user_correct = (user_pick_rec == result_winner)
                                        u_icon = "✅" if user_correct else "❌"
                                        if user_correct and result_winner != fav["team"]:
                                            u_verdict = "You were right, I was wrong. Good call."
                                        elif not user_correct and result_winner == fav["team"]:
                                            u_verdict = "I was right, you were wrong. Trust the model."
                                        else:
                                            u_verdict = "We were both wrong. March Madness things."
                                        note_line = f'<br/><span style="color:#94a3b8;font-size:0.78rem">Your reasoning: <em>"{user_note_rec}"</em></span>' if user_note_rec else ""
                                        st.markdown(
                                            f'<div style="background:#172233;border-left:3px solid {"#4ade80" if user_correct else "#f87171"};'
                                            f'border-radius:4px;padding:6px 10px;margin-top:4px;font-size:0.85rem">'
                                            f'{u_icon} <strong style="color:#f1f5f9">Your pick: {user_pick_rec}</strong> — {u_verdict}'
                                            f'{note_line}</div>', unsafe_allow_html=True)

                                    if st.button("↩ Clear result", key=f"clr_{matchup_key}"):
                                        st.session_state.results = [r for r in st.session_state.results if r["matchup"] != matchup_key]
                                        st.rerun()

            with br_r32:
                st.caption(f"Projected Round of 32 — model picks based on contender scores")
                if not all_round_matchups["R32"]:
                    st.info("Bracket data needed to project Round of 32.")
                else:
                    for reg in ["East", "South", "West", "Midwest"]:
                        reg_matchups = [(t1,t2,w,l,r) for t1,t2,w,l,r in all_round_matchups["R32"] if r == reg]
                        if reg_matchups:
                            st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.95rem;border-bottom:2px solid #f97316;padding-bottom:3px;margin-bottom:10px;text-transform:uppercase">{reg} Region</div>', unsafe_allow_html=True)
                            for idx, (t1n, t2n, w, l, r) in enumerate(reg_matchups):
                                _render_matchup_exp(t1n, t2n, w, l, r, f"r32_{reg}_{idx}")
                            st.markdown("")

            with br_s16:
                st.caption("Projected Sweet 16 — regional semifinal matchups")
                if not all_round_matchups["S16"]:
                    st.info("Bracket data needed to project Sweet 16.")
                else:
                    for reg in ["East", "South", "West", "Midwest"]:
                        reg_matchups = [(t1,t2,w,l,r) for t1,t2,w,l,r in all_round_matchups["S16"] if r == reg]
                        if reg_matchups:
                            st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.95rem;border-bottom:2px solid #f97316;padding-bottom:3px;margin-bottom:10px;text-transform:uppercase">{reg} Region</div>', unsafe_allow_html=True)
                            for idx, (t1n, t2n, w, l, r) in enumerate(reg_matchups):
                                _render_matchup_exp(t1n, t2n, w, l, r, f"s16_{reg}_{idx}")
                            st.markdown("")

            with br_e8:
                st.caption("Projected Elite 8 — regional final matchups")
                if not all_round_matchups["E8"]:
                    st.info("Bracket data needed to project Elite 8.")
                else:
                    e8_cols = st.columns(2)
                    for idx, (t1n, t2n, w, l, r) in enumerate(all_round_matchups["E8"]):
                        with e8_cols[idx % 2]:
                            st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.95rem;border-bottom:2px solid #f97316;padding-bottom:3px;margin-bottom:10px;text-transform:uppercase">{r} Region Championship</div>', unsafe_allow_html=True)
                            _render_matchup_exp(t1n, t2n, w, l, r, f"e8_{r}_{idx}")

            with br_ff:
                st.caption("Projected Final Four — national semifinal matchups")
                if not all_round_matchups["FF"]:
                    st.info("Bracket data needed to project Final Four.")
                else:
                    ff_cols = st.columns(2)
                    for idx, (t1n, t2n, w, l, r) in enumerate(all_round_matchups["FF"]):
                        with ff_cols[idx]:
                            st.markdown(f'<div style="color:#fbbf24;font-weight:800;font-size:1rem;border-bottom:2px solid #f59e0b;padding-bottom:3px;margin-bottom:10px">🏅 Semifinal {idx+1}</div>', unsafe_allow_html=True)
                            _render_matchup_exp(t1n, t2n, w, l, "National", f"ff_{idx}")

            with br_champ:
                st.caption("Projected Championship Game")
                if not all_round_matchups["Championship"]:
                    st.info("Bracket data needed to project Championship.")
                else:
                    for idx, (t1n, t2n, w, l, r) in enumerate(all_round_matchups["Championship"]):
                        st.markdown(f'<div style="text-align:center;font-size:1.5rem;font-weight:900;color:#fbbf24;margin:20px 0">🏆 Championship Game</div>', unsafe_allow_html=True)
                        _render_matchup_exp(t1n, t2n, w, l, "National", "champ_0")
                        t1_row = in_bracket[in_bracket["team"] == t1n]
                        t2_row = in_bracket[in_bracket["team"] == t2n]
                        if len(t1_row) and len(t2_row):
                            w_row = in_bracket[in_bracket["team"] == w]
                            if len(w_row):
                                wp = w_row.iloc[0]
                                w_cs = safe_f(wp.get("contender_score", 50))
                                w_seed = int(wp.get("seed")) if pd.notna(wp.get("seed")) else 0
                                st.markdown(
                                    f'<div style="text-align:center;background:linear-gradient(135deg,#1a2a1a,#0f2b0f);border:2px solid #4ade80;'
                                    f'border-radius:12px;padding:20px;margin-top:16px">'
                                    f'<div style="font-size:2rem">🏆</div>'
                                    f'<div style="color:#4ade80;font-weight:900;font-size:1.4rem;margin:8px 0">{w}</div>'
                                    f'<div style="color:#94a3b8;font-size:0.9rem">#{w_seed} seed · Score: {w_cs:.1f}</div>'
                                    f'<div style="color:#fbbf24;font-size:0.85rem;margin-top:6px">Statlasberg\'s 2026 National Champion</div>'
                                    f'</div>',
                                    unsafe_allow_html=True)

        # ── MY PICKS MODE ───────────────────────────────────────────────────────
        elif bkt_mode == "🎯 My Picks":
            # Progress tracker
            total_picks = len(st.session_state.user_bracket_picks)
            _pick_progress = min(total_picks / 63, 1.0)
            st.progress(_pick_progress, text=f"Bracket progress: {total_picks}/63 picks")

            _reset_col, _ = st.columns([1, 5])
            with _reset_col:
                if st.button("🗑 Reset My Picks", key="ubp_reset_all"):
                    st.session_state.user_bracket_picks = {}
                    st.rerun()

            # Compute model winner for each matchup key (for hints)
            _model_picks = {}
            for rnd_name, rnd_list in all_round_matchups.items():
                for t1n, t2n, w, l, reg in rnd_list:
                    # build the ubp key for this matchup to cross-reference
                    pass  # we'll just pass model_winner as the projected winner per round

            up_r64, up_r32, up_s16, up_e8, up_ff, up_champ = st.tabs([
                "🎯 Round of 64", "⚡ Round of 32", "🔥 Sweet 16", "💎 Elite 8", "🏅 Final Four", "🏆 Championship"
            ])

            with up_r64:
                r64_done = sum(1 for m in ubp["R64"] if m["winner"])
                st.caption(f"{r64_done}/32 first-round picks made · Click a team to pick them")
                reg_pcols = st.columns(4)
                for reg_pi, region in enumerate(["East","South","West","Midwest"]):
                    with reg_pcols[reg_pi]:
                        top1_rows = in_bracket[(in_bracket["region"]==region) & (in_bracket["seed"]==1)]
                        top1_name = top1_rows.iloc[0]["team"] if len(top1_rows)>0 else "TBD"
                        st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.85rem;border-bottom:2px solid #f97316;padding-bottom:2px;margin-bottom:8px;text-transform:uppercase">{region} · #1 {top1_name}</div>', unsafe_allow_html=True)
                        region_r64 = [m for m in ubp["R64"] if m["region"] == region]
                        for m in region_r64:
                            s1 = m.get("s1", 0); s2 = m.get("s2", 0)
                            st.markdown(f'<div style="color:#64748b;font-size:0.68rem;margin-top:6px;font-weight:600">#{s1} vs #{s2}</div>', unsafe_allow_html=True)
                            model_w = ""
                            c1 = _score_lkp.get(m["t1"],50); c2 = _score_lkp.get(m["t2"],50)
                            model_w = m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else m["t2"]
                            _render_pick_card(m, st.session_state.user_bracket_picks, _score_lkp, model_w)

            with up_r32:
                r32_done = sum(1 for m in ubp["R32"] if m["winner"])
                r32_avail = sum(1 for m in ubp["R32"] if m["t1"] and m["t2"])
                st.caption(f"{r32_done}/{r32_avail} Round of 32 picks made · Complete Round of 64 first to unlock all matchups")
                for region in ["East","South","West","Midwest"]:
                    region_r32 = [m for m in ubp["R32"] if m["region"] == region]
                    if not region_r32: continue
                    st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.85rem;border-bottom:1px solid #f97316;padding-bottom:2px;margin:10px 0 6px 0;text-transform:uppercase">{region} Region</div>', unsafe_allow_html=True)
                    rr_cols = st.columns(2)
                    for mi, m in enumerate(region_r32):
                        with rr_cols[mi % 2]:
                            model_w = ""
                            if m["t1"] and m["t2"]:
                                c1 = _score_lkp.get(m["t1"],50); c2 = _score_lkp.get(m["t2"],50)
                                model_w = m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else m["t2"]
                            _render_pick_card(m, st.session_state.user_bracket_picks, _score_lkp, model_w)

            with up_s16:
                s16_done = sum(1 for m in ubp["S16"] if m["winner"])
                s16_avail = sum(1 for m in ubp["S16"] if m["t1"] and m["t2"])
                st.caption(f"{s16_done}/{s16_avail} Sweet 16 picks made")
                for region in ["East","South","West","Midwest"]:
                    region_s16 = [m for m in ubp["S16"] if m["region"] == region]
                    if not region_s16: continue
                    st.markdown(f'<div style="color:#ff8c3a;font-weight:800;font-size:0.85rem;border-bottom:1px solid #f97316;padding-bottom:2px;margin:10px 0 6px 0;text-transform:uppercase">{region} Region</div>', unsafe_allow_html=True)
                    ss_cols = st.columns(2)
                    for mi, m in enumerate(region_s16):
                        with ss_cols[mi % 2]:
                            model_w = ""
                            if m["t1"] and m["t2"]:
                                c1 = _score_lkp.get(m["t1"],50); c2 = _score_lkp.get(m["t2"],50)
                                model_w = m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else m["t2"]
                            _render_pick_card(m, st.session_state.user_bracket_picks, _score_lkp, model_w)

            with up_e8:
                e8_done = sum(1 for m in ubp["E8"] if m["winner"])
                e8_avail = sum(1 for m in ubp["E8"] if m["t1"] and m["t2"])
                st.caption(f"{e8_done}/{e8_avail} Elite 8 picks made")
                e8_cols = st.columns(2)
                for ei, m in enumerate(ubp["E8"]):
                    with e8_cols[ei % 2]:
                        st.markdown(f'<div style="color:#ff8c3a;font-weight:700;font-size:0.85rem;margin-bottom:4px">{m["region"]} Region Championship</div>', unsafe_allow_html=True)
                        model_w = ""
                        if m["t1"] and m["t2"]:
                            c1 = _score_lkp.get(m["t1"],50); c2 = _score_lkp.get(m["t2"],50)
                            model_w = m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else m["t2"]
                        _render_pick_card(m, st.session_state.user_bracket_picks, _score_lkp, model_w)

            with up_ff:
                ff_done = sum(1 for m in ubp["FF"] if m["winner"])
                ff_avail = sum(1 for m in ubp["FF"] if m["t1"] and m["t2"])
                st.caption(f"{ff_done}/{ff_avail} Final Four picks made")
                ff_cols = st.columns(2)
                for fi, m in enumerate(ubp["FF"]):
                    with ff_cols[fi]:
                        st.markdown(f'<div style="color:#fbbf24;font-weight:700;font-size:0.95rem;margin-bottom:4px">🏅 Semifinal {fi+1} ({m["region"]})</div>', unsafe_allow_html=True)
                        model_w = ""
                        if m["t1"] and m["t2"]:
                            c1 = _score_lkp.get(m["t1"],50); c2 = _score_lkp.get(m["t2"],50)
                            model_w = m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else m["t2"]
                        _render_pick_card(m, st.session_state.user_bracket_picks, _score_lkp, model_w)

            with up_champ:
                st.caption("Championship Game")
                champ_m = ubp["Championship"][0] if ubp["Championship"] else None
                if champ_m:
                    st.markdown('<div style="text-align:center;font-size:1.3rem;font-weight:900;color:#fbbf24;margin:16px 0">🏆 Championship Game</div>', unsafe_allow_html=True)
                    model_w = ""
                    if champ_m["t1"] and champ_m["t2"]:
                        c1 = _score_lkp.get(champ_m["t1"],50); c2 = _score_lkp.get(champ_m["t2"],50)
                        model_w = champ_m["t1"] if win_prob_sigmoid(c1,c2) >= 0.5 else champ_m["t2"]
                    champ_cols = st.columns([1,2,1])
                    with champ_cols[1]:
                        _render_pick_card(champ_m, st.session_state.user_bracket_picks, _score_lkp, model_w)
                    if champ_m["winner"]:
                        ur = champ_m["winner"]
                        ur_row = in_bracket[in_bracket["team"] == ur]
                        ur_seed = int(ur_row.iloc[0]["seed"]) if len(ur_row) and pd.notna(ur_row.iloc[0].get("seed")) else 0
                        st.markdown(
                            f'<div style="text-align:center;background:linear-gradient(135deg,#2a1a00,#1a0f00);border:2px solid #fbbf24;'
                            f'border-radius:12px;padding:20px;margin-top:16px">'
                            f'<div style="font-size:2rem">🏆</div>'
                            f'<div style="color:#fbbf24;font-weight:900;font-size:1.4rem;margin:8px 0">{ur}</div>'
                            f'<div style="color:#94a3b8;font-size:0.9rem">#{ur_seed} seed</div>'
                            f'<div style="color:#f97316;font-size:0.85rem;margin-top:6px">Your 2026 National Champion Pick</div>'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.info("Complete earlier rounds to unlock the Championship.")

        # ── COMPARE MODE ────────────────────────────────────────────────────────
        else:  # Compare mode
            my_champ = ubp["Championship"][0]["winner"] if ubp["Championship"] else None
            model_champ = sim_champion

            # Summary header
            comp_summary_cols = st.columns(3)
            my_total = len(st.session_state.user_bracket_picks)
            model_match = sum(
                1 for m in ubp["R64"] + ubp["R32"] + ubp["S16"] + ubp["E8"] + ubp["FF"] + ubp["Championship"]
                if m["winner"] and m["t1"] and m["t2"] and
                (m["winner"] == (m["t1"] if win_prob_sigmoid(_score_lkp.get(m["t1"],50), _score_lkp.get(m["t2"],50)) >= 0.5 else m["t2"]))
            )
            with comp_summary_cols[0]:
                st.metric("My Champion Pick", my_champ or "Not yet picked")
            with comp_summary_cols[1]:
                st.metric("Model Champion", model_champ or "—")
            with comp_summary_cols[2]:
                if my_total > 0:
                    st.metric("Agreement with Model", f"{model_match}/{my_total}", f"{model_match/my_total*100:.0f}%")
                else:
                    st.metric("Agreement with Model", "—", "make picks first")

            if my_total == 0:
                st.info("Switch to **🎯 My Picks** to make your bracket picks, then come back here to compare with the model.")
            else:
                st.markdown("---")
                # Per-round comparison
                round_order = [("R64", ubp["R64"], "Round of 64"),
                               ("R32", ubp["R32"], "Round of 32"),
                               ("S16", ubp["S16"], "Sweet 16"),
                               ("E8",  ubp["E8"],  "Elite 8"),
                               ("FF",  ubp["FF"],  "Final Four"),
                               ("Championship", ubp["Championship"], "Championship")]
                for rnd_key, rnd_list, rnd_label in round_order:
                    picked_matchups = [m for m in rnd_list if m["winner"] and m["t1"] and m["t2"]]
                    if not picked_matchups:
                        continue
                    agree = sum(1 for m in picked_matchups
                                if m["winner"] == (m["t1"] if win_prob_sigmoid(_score_lkp.get(m["t1"],50), _score_lkp.get(m["t2"],50)) >= 0.5 else m["t2"]))
                    disagree = len(picked_matchups) - agree
                    with st.expander(f"**{rnd_label}** — {agree} agree, {disagree} disagree vs model", expanded=(disagree > 0)):
                        for m in picked_matchups:
                            c1 = _score_lkp.get(m["t1"], 50); c2 = _score_lkp.get(m["t2"], 50)
                            model_w = m["t1"] if win_prob_sigmoid(c1, c2) >= 0.5 else m["t2"]
                            you_agree = (m["winner"] == model_w)
                            icon = "✅" if you_agree else "⚡"
                            clash = "" if you_agree else f" · Model: **{model_w}**"
                            st.markdown(
                                f'<div style="background:#1e293b;border-left:3px solid {"#4ade80" if you_agree else "#f59e0b"};'
                                f'border-radius:4px;padding:5px 10px;margin:3px 0;font-size:0.82rem">'
                                f'{icon} <strong style="color:#f1f5f9">{m["t1"]}</strong>'
                                f'<span style="color:#475569"> vs </span>'
                                f'<strong style="color:#f1f5f9">{m["t2"]}</strong> — '
                                f'Your pick: <strong style="color:{"#4ade80" if you_agree else "#f87171"}">{m["winner"]}</strong>'
                                f'{clash}'
                                f'</div>', unsafe_allow_html=True)


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
                for g in recent_fin[:8]:
                    t1 = g["team1"]; t2 = g["team2"]
                    winner = t1 if t1["score"] > t2["score"] else t2
                    loser  = t2 if t1["score"] > t2["score"] else t1
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    model_got_it = (pregame_p >= 0.5) == (t1["score"] > t2["score"])
                    verdict = "✅ Model correct" if model_got_it else "❌ Upset — model wrong"
                    model_fav = t1["name"] if pregame_p >= 0.5 else t2["name"]
                    model_fav_p = max(pregame_p, 1 - pregame_p)
                    with st.expander(
                        f"{'✅' if model_got_it else '❌'} {winner['name']} {winner['score']} def. {loser['name']} {loser['score']}  |  {verdict}",
                        expanded=False
                    ):
                        fc1, fc2 = st.columns(2)
                        for col, team, score, is_winner_flag in [
                            (fc1, t1, t1["score"], t1["score"] > t2["score"]),
                            (fc2, t2, t2["score"], t2["score"] > t1["score"])
                        ]:
                            with col:
                                cs = s1 if team is t1 else s2
                                bdr = "#4ade80" if is_winner_flag else "#374151"
                                st.markdown(
                                    f'<div style="background:#131820;border:2px solid {bdr};border-radius:8px;padding:10px">'
                                    f'<div style="font-size:1.1rem;font-weight:800;color:#f1f5f9">{team["name"]}</div>'
                                    f'<div style="font-size:2rem;font-weight:900;color:{"#4ade80" if is_winner_flag else "#f1f5f9"}">{score}</div>'
                                    f'<div style="color:#94a3b8;font-size:0.78rem">{team.get("record","")}</div>'
                                    f'<div style="color:#64748b;font-size:0.78rem;margin-top:4px">Model score: {cs:.1f}</div>'
                                    f'</div>', unsafe_allow_html=True)
                                tname = team["name"]
                                if st.button(f"🔍 {tname} Deep Dive", key=f"fin_dd_{g['event_id']}_{tname}", use_container_width=True):
                                    st.session_state["team_selectbox"] = tname
                                    st.session_state.dive_team = tname
                                    st.rerun()
                        insight = style_matchup_insight_by_name(t1["name"], t2["name"], in_bracket)
                        if insight:
                            st.markdown(f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;padding:6px 10px;margin:8px 0;color:#d1fae5;font-size:0.82rem">💡 Pre-game model insight: {insight}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="background:#1e293b;border-radius:6px;padding:8px 12px;margin-top:6px;font-size:0.85rem">'
                            f'📊 Statlasberg pre-game pick: <strong style="color:#f1f5f9">{model_fav}</strong> '
                            f'({model_fav_p*100:.0f}%) — {verdict}'
                            f'</div>', unsafe_allow_html=True)

            # ── UPCOMING ─────────────────────────────────────────────────────
            if upcoming:
                st.markdown("---")
                st.markdown('<div class="section-header">🕐 Upcoming Games — Pre-Game Model Lines</div>', unsafe_allow_html=True)
                for g in upcoming[:8]:
                    t1 = g["team1"]; t2 = g["team2"]
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    fav_name = t1["name"] if pregame_p >= 0.5 else t2["name"]
                    fav_p = max(pregame_p, 1 - pregame_p)
                    with st.expander(
                        f"🕐 {t1['name']} vs {t2['name']}  —  Model pick: {fav_name} ({fav_p*100:.0f}%)",
                        expanded=False
                    ):
                        uc1, uc2 = st.columns(2)
                        for col, team, cs, p in [(uc1, t1, s1, pregame_p), (uc2, t2, s2, 1-pregame_p)]:
                            with col:
                                is_fav = team["name"] == fav_name
                                bar_col = "#4ade80" if is_fav else "#64748b"
                                st.markdown(
                                    f'<div style="background:#131820;border:2px solid {"#f97316" if is_fav else "#374151"};border-radius:8px;padding:10px">'
                                    f'<div style="font-size:1rem;font-weight:800;color:#f1f5f9">{team["name"]} {"⭐ Model Pick" if is_fav else ""}</div>'
                                    f'<div style="background:#1e293b;border-radius:3px;height:6px;margin:6px 0">'
                                    f'<div style="width:{int(p*100)}%;background:{bar_col};height:6px;border-radius:3px"></div></div>'
                                    f'<div style="color:{bar_col};font-weight:700;font-size:1rem">{p*100:.0f}% · {american_line(p)}</div>'
                                    f'<div style="color:#64748b;font-size:0.78rem;margin-top:4px">Model score: {cs:.1f}</div>'
                                    f'</div>', unsafe_allow_html=True)
                                tname = team["name"]
                                if st.button(f"🔍 {tname} Deep Dive", key=f"up_dd_{g['event_id']}_{tname}", use_container_width=True):
                                    st.session_state["team_selectbox"] = tname
                                    st.session_state.dive_team = tname
                                    st.rerun()
                        insight = style_matchup_insight_by_name(t1["name"], t2["name"], in_bracket)
                        if insight:
                            st.markdown(f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;padding:6px 10px;margin:8px 0;color:#d1fae5;font-size:0.82rem">💡 {insight}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Data: ESPN public API · Refresh rate: manual or 60s auto · In-game probability blends model pre-game score (dominates early) with live score differential (dominates late). Model principles stay constant — late-game score doesn't override the model, it updates it.")


# ─────────────────────────────────────────────────────────────────────────────
# STATLASBERG Q&A ENGINE  — speaks hoop, stays honest
# ─────────────────────────────────────────────────────────────────────────────

def _team_strengths(row):
    """Return a plain-english list of what makes this team dangerous."""
    tags = []
    if safe_f(row.get("defense_score", 50)) >= 72:
        tags.append("lockdown defense")
    elif safe_f(row.get("defense_score", 50)) >= 62:
        tags.append("solid D")
    if safe_f(row.get("clutch_score", 50)) >= 72:
        tags.append("ice in their veins in close games")
    elif safe_f(row.get("clutch_score", 50)) >= 62:
        tags.append("clutch enough")
    if safe_f(row.get("guard_play_score", 50)) >= 72:
        tags.append("guards that'll cook you off the dribble")
    elif safe_f(row.get("guard_play_score", 50)) >= 62:
        tags.append("solid backcourt")
    if safe_f(row.get("backcourt_experience_score", row.get("experience_score", 50))) >= 72:
        tags.append("the most experienced roster on the floor")
    if safe_f(row.get("rebounding_score", 50)) >= 72:
        tags.append("owning the glass")
    return tags


def _kp_verdict(pick_row, actual_row):
    """Was KenPom-proxy on the right side? Returns a verdict string."""
    def kp_val(r):
        if "adj_margin" in r.index and not pd.isna(r.get("adj_margin", float("nan"))):
            return safe_f(r.get("adj_margin", 0))
        if "adj_offense" in r.index and "adj_defense" in r.index:
            return safe_f(r.get("adj_offense", 50)) - safe_f(r.get("adj_defense", 50))
        return None

    kp_pick   = kp_val(pick_row)
    kp_actual = kp_val(actual_row)
    if kp_pick is None or kp_actual is None:
        return None
    if kp_actual > kp_pick:
        return f"KenPom-proxy **liked {actual_row['team']} more** than I did — I should've listened"
    return f"KenPom-proxy also had {pick_row['team']} — neither of us saw it coming"


def _diagnose_miss(pick_team, actual_winner, bkt_df):
    """
    Post-game autopsy. Returns (verdict_label, explanation_md).
    verdict_label: "BLIND SPOT" | "OVER-ZEALOUS" | "CHAOS"
    """
    pick_row   = bkt_df[bkt_df["team"] == pick_team]
    actual_row = bkt_df[bkt_df["team"] == actual_winner]

    if pick_row.empty:
        return "CHAOS", f"{actual_winner} just balled out. Not enough data to break it down further."

    p_score = safe_f(pick_row.iloc[0].get("contender_score", 50))
    a_score = safe_f(actual_row.iloc[0].get("contender_score", 50)) if not actual_row.empty else 50
    p_risk  = safe_f(pick_row.iloc[0].get("upset_risk_score", 25))
    p_seed  = int(pick_row.iloc[0].get("seed", 1))
    a_seed  = int(actual_row.iloc[0].get("seed", 16)) if not actual_row.empty else 16

    kp_line = _kp_verdict(pick_row.iloc[0], actual_row.iloc[0]) if not actual_row.empty else None
    kp_md   = f"\n\n**Comp check:** {kp_line}." if kp_line else ""

    seed_gap = a_seed - p_seed  # positive = underdog won

    # CHAOS: model score gap was small, anyone could've called it
    if abs(p_score - a_score) < 6:
        return ("CHAOS",
                f"Honestly? On paper these two were close — **{pick_team}** at {p_score:.1f} vs "
                f"**{actual_winner}** at {a_score:.1f}. "
                f"That's a coin flip. Model wasn't wrong, ball just didn't bounce right.{kp_md}")

    # OVER-ZEALOUS: model loved the pick, upset risk was flagged but ignored
    if p_risk >= 38 and p_score > a_score + 6:
        return ("OVER-ZEALOUS",
                f"The warning was right there. **{pick_team}**'s upset risk was **{p_risk:.0f}** — "
                f"model flagged the vulnerability and I still rode them hard. "
                f"Over-indexed on the contender score ({p_score:.1f}) and didn't respect "
                f"the **{actual_winner}** matchup. That's on me.{kp_md}")

    # BLIND SPOT: model loved pick AND didn't see risk — something structural was missing
    if p_score > a_score + 8 and p_risk < 35:
        blind_reason = ""
        if seed_gap >= 6:
            blind_reason = (f"A #{a_seed} taking down a #{p_seed} — "
                            f"something about {actual_winner} doesn't show up in the box score. "
                            f"Coaching edge, specific schematic matchup, or they just got unconscious.")
        else:
            blind_reason = (f"Model had {pick_team} clearly better and didn't see the upset risk coming. "
                            f"Classic blind spot — probably missing momentum, scheme, or a matchup-specific factor.")
        return ("BLIND SPOT", f"{blind_reason}{kp_md}")

    # Default: moderate miss
    return ("TOUGH BREAK",
            f"**{pick_team}** ({p_score:.1f}) vs **{actual_winner}** ({a_score:.1f}) — "
            f"model had the right side directionally but {actual_winner} made plays when it mattered. "
            f"Upset risk was **{p_risk:.0f}** — not totally blindsided, just got outcompeted.{kp_md}")


def statlasberg_qa(q, bkt_df, s16, e8, ff, champion, champs_df, results_list=None):
    """Statlasberg speaks hoop — confident, reactive, back-and-forth basketball banter."""
    # _re and _rng imported at module level
    results_list = results_list or []
    q_low = q.lower().strip()

    wrong_picks = [r for r in results_list if r.get("model_pick") != r.get("winner")]
    right_picks = [r for r in results_list if r.get("model_pick") == r.get("winner")]
    n_total     = len(results_list)

    # ── Conversation context (session state carries last team/topic discussed) ──
    import streamlit as _st
    _ctx = _st.session_state.get("chat_context", {})
    _last_team   = _ctx.get("last_team")
    _last_topic  = _ctx.get("last_topic", "")
    _last_stance = _ctx.get("last_stance", "")   # "liked" | "faded" | "neutral"
    _turns       = _ctx.get("turns", 0)

    def _save_ctx(team=None, topic=None, stance=None):
        _st.session_state["chat_context"] = {
            "last_team":   team  or _last_team,
            "last_topic":  topic or _last_topic,
            "last_stance": stance or _last_stance,
            "turns":       _turns + 1,
        }

    # ── Pronouns / context resolution ─────────────────────────────────────────
    # If user says "them", "they", "that team" — resolve to last mentioned team
    _pronoun_words = ["them", "they", "that team", "those guys", "that squad",
                      "their", "those guys", "this team", "these guys"]
    _is_pronoun_ref = any(w in q_low for w in _pronoun_words) and _last_team
    if _is_pronoun_ref:
        q_low = q_low  # keep original but resolve via _last_team in find_row fallback

    def find_row(query):
        """Fuzzy team name lookup — falls back to last-mentioned team for pronoun refs."""
        q_ = query.strip().lower()
        # Pronoun resolution — "them", "they", "that team" → last mentioned
        if any(w == q_.strip() or q_.strip().startswith(w) for w in _pronoun_words) and _last_team:
            hit = bkt_df[bkt_df["team"] == _last_team]
            if len(hit): return hit.iloc[0]
        for _, row in bkt_df.iterrows():
            t_ = row["team"].lower()
            if q_ == t_ or q_ in t_ or t_ in q_:
                return row
        # word-level fallback
        words = [w for w in q_.split() if len(w) >= 4]
        for _, row in bkt_df.iterrows():
            if any(w in row["team"].lower() for w in words):
                return row
        # last resort — last mentioned team if pronouns present in full original query
        if _is_pronoun_ref and _last_team:
            hit = bkt_df[bkt_df["team"] == _last_team]
            if len(hit): return hit.iloc[0]
        return None

    # ── Personality pools — vary responses so nothing feels scripted ───────────
    def _follow_up(team=None):
        """Return a short follow-up prompt to keep conversation going."""
        opts_team = [
            f"Who do you have them beating in the second round?",
            f"You riding with them all the way, or just first weekend?",
            f"You taking them in your bracket?",
            f"What's your gut saying about their path?",
        ]
        opts_gen = [
            "Who else you looking at?",
            "What's your bracket looking like overall?",
            "Any other teams you want to break down?",
            "Who you fading this year?",
            "Any sleepers you like that I haven't mentioned?",
        ]
        pool = opts_team if team else opts_gen
        return f"\n\n*{_rng.choice(pool)}*"

    def _pushback(team_name, score):
        """Give a short opinion-based pushback line."""
        if score >= 72:
            opts = [
                f"The numbers back it up — {team_name} is built for this.",
                f"I know you might not love it but {team_name} checks every box I care about.",
                f"This isn't a homer pick, this is just what the data says.",
            ]
        elif score >= 58:
            opts = [
                f"I hear you, but don't sleep on them — they're more dangerous than their seed suggests.",
                f"They're not flashy but they know how to win games that matter.",
                f"People are underrating them and that's exactly how upsets happen.",
            ]
        else:
            opts = [
                f"Yeah I'm not here to die on that hill either. They're risky.",
                f"Look, they could surprise someone but I wouldn't bet the house.",
                f"Fair criticism. Their numbers don't scream deep run.",
            ]
        return _rng.choice(opts)

    # ── Record / How are you doing ────────────────────────────────────────────
    if any(w in q_low for w in ["your record", "how many right", "accuracy", "how are your picks",
                                 "how'd you do", "how you doing", "hit rate", "track record"]):
        if n_total == 0:
            return ("No results logged yet. Head to the **🏆 Bracket** tab, watch some games, "
                    "and log the winners — then come back and I'll keep score.")
        acc = len(right_picks) / n_total * 100
        if acc >= 75:
            verdict = f"Running hot. {len(right_picks)}-{len(wrong_picks)}. Model's cooking."
        elif acc >= 55:
            verdict = (f"{len(right_picks)}-{len(wrong_picks)}. Holding up. "
                       "A few bad bounces but directionally locked in.")
        else:
            verdict = (f"{len(right_picks)}-{len(wrong_picks)}. Getting humbled. "
                       "Ask me 'where were you wrong?' and I'll break down the misses.")
        return f"📊 **{acc:.0f}% hit rate** — {verdict}"

    # ── How are MY picks doing (user vs model) ────────────────────────────────
    user_picks_logged = [r for r in results_list if r.get("user_pick")]
    if any(w in q_low for w in ["my picks", "my record", "how did i do", "my accuracy",
                                  "my hit rate", "am i right", "how am i doing"]):
        if not user_picks_logged:
            return ("You haven't logged any disagreements yet. "
                    "In the **🏆 Bracket** tab, check '🙅 I had the other team' when you disagree with me, "
                    "then log the result. I'll track your record vs mine.")
        u_right  = [r for r in user_picks_logged if r.get("user_pick") == r.get("winner")]
        u_wrong  = [r for r in user_picks_logged if r.get("user_pick") != r.get("winner")]
        u_beat_m = [r for r in user_picks_logged
                    if r.get("user_pick") == r.get("winner") and r.get("model_pick") != r.get("winner")]
        m_beat_u = [r for r in user_picks_logged
                    if r.get("model_pick") == r.get("winner") and r.get("user_pick") != r.get("winner")]
        u_acc    = len(u_right) / len(user_picks_logged) * 100
        lines    = [f"📊 **Your record on disagreements: {len(u_right)}-{len(u_wrong)} ({u_acc:.0f}%)**"]
        if u_beat_m:
            lines.append(f"\n✅ **You beat me on:** {', '.join(r['winner'] for r in u_beat_m)} — good reads.")
        if m_beat_u:
            lines.append(f"\n❌ **I beat you on:** {', '.join(r['winner'] for r in m_beat_u)} — trust the model.")
        if not u_beat_m and not m_beat_u:
            lines.append("\nNeither of us had it. Basketball.")
        return " ".join(lines)

    # ── Where did I beat you ──────────────────────────────────────────────────
    if any(w in q_low for w in ["where did i beat you", "where was i right", "my correct picks",
                                  "when was i right", "i was right"]):
        u_beat_m = [r for r in results_list
                    if r.get("user_pick") and
                    r.get("user_pick") == r.get("winner") and
                    r.get("model_pick") != r.get("winner")]
        if not u_beat_m:
            return "No logged disagreements where you were right and I was wrong — yet."
        lines = []
        for r in u_beat_m:
            label, explanation = _diagnose_miss(r["model_pick"], r["winner"], bkt_df)
            note = r.get("user_note", "")
            note_line = f'\n*Your reasoning: "{note}"* — ' if note else ""
            lines.append(
                f"✅ **{r['winner']}** (you had them, I had {r['model_pick']}) · `{label}`\n"
                f"{note_line}{explanation}")
        return (f"🎯 **{len(u_beat_m)} time(s) you saw it better than me:**\n\n"
                + "\n\n---\n\n".join(lines))

    # ── Analyze my reasoning on a specific game ───────────────────────────────
    if any(w in q_low for w in ["analyze my reasoning", "my reasoning", "why did i think",
                                  "was i right to think", "look at my pick on"]):
        # Find game with user note that matches a team name in the question
        for r in [x for x in results_list if x.get("user_note")]:
            for t in r.get("teams", []):
                if t.lower() in q_low or any(w in t.lower() for w in q_low.split() if len(w) >= 4):
                    winner   = r.get("winner", "")
                    u_pick   = r.get("user_pick", "")
                    m_pick   = r.get("model_pick", "")
                    note     = r.get("user_note", "")
                    u_right  = (u_pick == winner)
                    label, explanation = _diagnose_miss(m_pick, winner, bkt_df) if m_pick != winner else ("CORRECT", "")
                    verdict  = "✅ Right call." if u_right else "❌ Wrong call."
                    my_side  = f"I had **{m_pick}**. {'Called it too.' if m_pick == winner else f'I was wrong — {explanation}'}"
                    return (f"🔍 **Your reasoning on {t}**: *\"{note}\"*\n\n"
                            f"{verdict} **{winner}** won.\n\n"
                            f"{my_side}")
        return ("Give me a team name so I can find the right game — "
                "e.g. *Analyze my reasoning on Auburn*.")

    # ── Miss autopsy — where were you wrong ──────────────────────────────────
    if any(w in q_low for w in ["where were you wrong", "where'd you miss", "your misses",
                                  "bad picks", "wrong picks", "blew it", "got wrong"]):
        if not wrong_picks:
            if n_total == 0:
                return "Log some results first. No misses on record — yet."
            return "Perfect on logged results so far. Don't jinx it."

        lines = []
        for r in wrong_picks[:6]:
            pick   = r.get("model_pick", "?")
            actual = r.get("winner", "?")
            label, explanation = _diagnose_miss(pick, actual, bkt_df)
            lines.append(
                f"**{pick} → {actual} won** · `{label}`\n{explanation}")
        header = f"📉 **{len(wrong_picks)} miss{'es' if len(wrong_picks)>1 else ''}** so far. Real talk:\n\n"
        return header + "\n\n---\n\n".join(lines)

    # ── Single miss: "why were you wrong on X" ───────────────────────────────
    if any(w in q_low for w in ["why were you wrong", "why'd you miss", "why did you pick",
                                  "why did you get", "explain your miss"]):
        # Find the team name in question
        for r in wrong_picks:
            pick_l = r.get("model_pick", "").lower()
            if pick_l and pick_l in q_low:
                label, explanation = _diagnose_miss(r["model_pick"], r["winner"], bkt_df)
                return (f"📉 **{r['model_pick']} loss — `{label}`**\n\n{explanation}\n\n"
                        f"*Result logged: {r['winner']} won.*")
        # Fallback: no specific team found
        if wrong_picks:
            r = wrong_picks[-1]
            label, explanation = _diagnose_miss(r["model_pick"], r["winner"], bkt_df)
            return (f"Last miss — **{r['model_pick']} → {r['winner']} won** · `{label}`\n\n{explanation}")
        return "No misses logged yet."

    # ── Champion pick ─────────────────────────────────────────────────────────
    if any(w in q_low for w in ["champion", "win it all", "cut the nets", "national title",
                                 "going all the way", "who wins", "title game"]):
        # Did my champion already get bounced?
        champ_out = next(
            (r for r in results_list
             if champion in r.get("teams", []) and r.get("winner") != champion),
            None)
        if champ_out:
            label, explanation = _diagnose_miss(champion, champ_out["winner"], bkt_df)
            return (f"📉 Had **{champion}** going all the way. **{champ_out['winner']}** ended that. "
                    f"`{label}` — {explanation}")

        row = bkt_df[bkt_df["team"] == champion]
        score_val = safe_f(row.iloc[0].get("contender_score", 50)) if len(row) else 50.0
        champ_pct_str = ""
        if len(champs_df) > 0 and champion in champs_df["team"].values:
            pct = float(champs_df[champs_df["team"] == champion]["championship_pct"].iloc[0])
            champ_pct_str = f" — came out on top in **{pct:.1f}%** of 500k sims"

        strengths = _team_strengths(row.iloc[0]) if len(row) else []
        str_line  = (", ".join(strengths[:3]) + " — ") if strengths else ""
        seed      = int(row.iloc[0]["seed"]) if len(row) and not pd.isna(row.iloc[0].get("seed")) else "—"

        _save_ctx(team=champion, topic="champion", stance="liked")
        champ_takes = [
            f"🏆 **{champion}**. That's the pick and I'm not shopping around.{champ_pct_str}\n\n"
            f"#{seed} seed. {str_line}**{score_val:.1f}/100** — built for six rounds, not just one big game. "
            f"Runner-up: **{sim_runner_up}**, but I don't think it gets that far.\n\n"
            f"*You buying that or you think someone else cuts the nets?*",
            f"🏆 I'll say it clearly: **{champion}** wins it all.{champ_pct_str}\n\n"
            f"#{seed} seed, {str_line}score **{score_val:.1f}**. Every pillar checks — defense, clutch, guard play. "
            f"That's not a hot take, that's what the data says at the end of six rounds.\n\n"
            f"Runner-up is **{sim_runner_up}**. *Who you got?*",
        ]
        return _rng.choice(champ_takes)

    # ── Final Four ────────────────────────────────────────────────────────────
    if "final four" in q_low or "final 4" in q_low or "semifinal" in q_low:
        if not ff:
            return "Run the bracket simulation first — Final Four data isn't ready."
        ff_lines = []
        for t in ff:
            row = bkt_df[bkt_df["team"] == t]
            seed_ = f"#{int(row.iloc[0]['seed'])} seed" if len(row) and not pd.isna(row.iloc[0].get("seed")) else ""
            reg_  = row.iloc[0].get("region", "") if len(row) else ""
            ff_lines.append(f"• **{t}** — {seed_}, {reg_}")
        return ("🏀 My Final Four:\n\n" + "\n".join(ff_lines) + "\n\n"
                f"**Champion:** {champion} · **Runner-up:** {sim_runner_up}\n\n"
                "*Bracket simulation runs the full 6 rounds deterministically — no coin flips.*")

    # ── Region pick ───────────────────────────────────────────────────────────
    for region in ["east", "west", "south", "midwest"]:
        if region in q_low and any(w in q_low for w in ["win", "who", "pick", "represent", "coming out"]):
            region_teams = bkt_df[bkt_df["region"].str.lower() == region]["team"].tolist()
            ff_from = [t for t in ff if t in region_teams]
            if ff_from:
                pick_t = ff_from[0]
                row    = bkt_df[bkt_df["team"] == pick_t]
                seed_  = f"#{int(row.iloc[0]['seed'])}" if len(row) and not pd.isna(row.iloc[0].get("seed")) else ""
                score_ = safe_f(row.iloc[0].get("contender_score", 50)) if len(row) else 50.0
                strengths = _team_strengths(row.iloc[0]) if len(row) else []
                str_line  = (", ".join(strengths[:2]) + " — ") if strengths else ""
                # Did they already win/lose?
                pick_out = next((r for r in results_list
                                 if pick_t in r.get("teams", []) and r.get("winner") != pick_t), None)
                if pick_out:
                    label, explanation = _diagnose_miss(pick_t, pick_out["winner"], bkt_df)
                    return (f"📉 Had **{pick_t}** out of the **{region.capitalize()}** — "
                            f"**{pick_out['winner']}** bounced them. `{label}` — {explanation}")
                return (f"🏆 **{pick_t}** {seed_} out of the **{region.capitalize()}**. "
                        f"{str_line}score **{score_:.1f}**. "
                        f"That's the pick. Not changing it.")
            return f"Simulation didn't have a {region.capitalize()} team reach the Final Four — bracket may be incomplete."

    # ── Sweet 16 ──────────────────────────────────────────────────────────────
    if "sweet 16" in q_low or "sweet sixteen" in q_low or "second weekend" in q_low:
        if not s16:
            return "Sweet 16 data not ready. Run the pipeline."
        by_region: dict = {}
        for t in s16:
            r = bkt_df[bkt_df["team"] == t]
            reg = r.iloc[0].get("region", "Unknown") if len(r) else "Unknown"
            by_region.setdefault(reg, []).append(t)
        lines = []
        for reg, teams in sorted(by_region.items()):
            seeds_str = ", ".join(
                f"**{t}** (#{int(bkt_df[bkt_df['team']==t].iloc[0]['seed'])})" if len(bkt_df[bkt_df["team"]==t]) and pd.notna(bkt_df[bkt_df['team']==t].iloc[0].get('seed')) else f"**{t}**"
                for t in teams)
            lines.append(f"**{reg}:** {seeds_str}")
        return "🎯 Second weekend — teams I'm sending to the Sweet 16:\n\n" + "\n\n".join(lines)

    # ── Elite Eight ───────────────────────────────────────────────────────────
    if "elite eight" in q_low or "elite 8" in q_low or "quarterfinal" in q_low:
        if not e8:
            return "Elite Eight data not ready."
        by_region: dict = {}
        for t in e8:
            r = bkt_df[bkt_df["team"] == t]
            reg = r.iloc[0].get("region", "Unknown") if len(r) else "Unknown"
            by_region.setdefault(reg, []).append(t)
        lines = []
        for reg, teams in sorted(by_region.items()):
            lines.append(f"**{reg}:** {', '.join(f'**{t}**' for t in teams)}")
        return "🎯 My Elite Eight:\n\n" + "\n\n".join(lines)

    # ── Upsets / Cinderella ───────────────────────────────────────────────────
    if any(w in q_low for w in ["upset", "cinderella", "dark horse", "darkhorse", "sleeper",
                                  "double digit", "double-digit", "surprise me"]):
        s16_upsets = [(row["team"], int(row["seed"])) for _, row in bkt_df.iterrows()
                      if pd.notna(row.get("seed")) and row.get("seed", 1) >= 10 and row["team"] in (s16 or [])]
        e8_upsets  = [(row["team"], int(row["seed"])) for _, row in bkt_df.iterrows()
                      if pd.notna(row.get("seed")) and row.get("seed", 1) >= 8 and row["team"] in (e8 or [])]

        lines = []
        for t, s in sorted(e8_upsets, key=lambda x: x[1], reverse=True):
            row = bkt_df[bkt_df["team"] == t]
            score_ = safe_f(row.iloc[0].get("contender_score", 50)) if len(row) else 50.0
            lines.append(f"• 💣 **#{s} {t}** — Elite Eight. Score: {score_:.1f}. *Don't fade them.*")
        for t, s in sorted(s16_upsets, key=lambda x: x[1], reverse=True):
            if not any(t == line_t for line_t, _ in e8_upsets):
                row = bkt_df[bkt_df["team"] == t]
                score_ = safe_f(row.iloc[0].get("contender_score", 50)) if len(row) else 50.0
                lines.append(f"• **#{s} {t}** — Sweet 16 run. Score: {score_:.1f}.")
        if not lines:
            threats = bkt_df[bkt_df["seed"] >= 10].sort_values("contender_score", ascending=False).head(4)
            for _, r in threats.iterrows():
                _s = int(r['seed']) if pd.notna(r.get('seed')) else 0
                lines.append(f"• **#{_s} {r['team']}** — score: {safe_f(r.get('contender_score',50)):.1f}. One to watch.")
        if not lines:
            return "Chalk looks really strong this year. No glaring Cinderella picks."
        return "💥 Upset picks — teams the model actually believes in:\n\n" + "\n".join(lines)

    # ── Fraud / Fade ──────────────────────────────────────────────────────────
    if any(w in q_low for w in ["fraud", "overrated", "avoid", "fade", "trap team",
                                  "don't trust", "fake", "pretenders"]):
        frauds = []
        if "fraud_favorite_flag" in bkt_df.columns:
            frauds = bkt_df[bkt_df["fraud_favorite_flag"] == True].sort_values("seed").to_dict("records")
        if not frauds:
            # Fallback: high seed + high upset risk + exits early in sim
            candidates = bkt_df[
                (bkt_df["seed"] <= 5) &
                (bkt_df.get("upset_risk_score", pd.Series(0, index=bkt_df.index)) >= 40)
            ].sort_values("upset_risk_score", ascending=False).head(4)
            frauds = candidates.to_dict("records")

        if not frauds:
            return "Nobody jumps out as a fraud this year. Seeds look legitimate across the board."

        lines = []
        for r in frauds:
            risk_  = safe_f(r.get("upset_risk_score", 50))
            score_ = safe_f(r.get("contender_score", 50))
            sim_r_ = r.get("sim_round", "First Round")
            lines.append(
                f"• **#{int(r.get('seed','?'))} {r['team']}** — "
                f"contender score {score_:.1f}, upset risk **{risk_:.0f}**. "
                f"Model sends them home in **{sim_r_}**. Fade 'em.")
        return "⚠️ **Teams I'd fade** — overseeded, soft résumé, or hiding behind a weak schedule:\n\n" + "\n".join(lines)

    # ── Best/Top team ─────────────────────────────────────────────────────────
    if any(w in q_low for w in ["best team", "top team", "#1 overall", "number one overall",
                                  "highest rated", "most dangerous"]):
        top = bkt_df.sort_values("contender_score", ascending=False).iloc[0]
        strengths = _team_strengths(top)
        str_line  = (" — " + ", ".join(strengths)) if strengths else ""
        _top_seed = int(top.get('seed')) if pd.notna(top.get('seed')) else '?'
        return (f"📈 **{top['team']}** — #{_top_seed} seed, "
                f"contender score **{safe_f(top.get('contender_score',50)):.1f}/100**{str_line}. "
                f"Best team in the field. Period.")

    # ── Head-to-head comparison ───────────────────────────────────────────────
    vs_match = _re.search(r'(.+?)\s+(?:vs\.?|versus|against|or|beat|playing)\s+(.+)', q_low)
    if vs_match:
        t1_q = vs_match.group(1).strip().rstrip("?")
        t2_q = vs_match.group(2).strip().rstrip("?")
        r1   = find_row(t1_q)
        r2   = find_row(t2_q)
        if r1 is not None and r2 is not None:
            sc1   = safe_f(r1.get("contender_score", 50))
            sc2   = safe_f(r2.get("contender_score", 50))
            p1    = win_prob_sigmoid(sc1, sc2)
            fav   = r1["team"] if p1 >= 0.5 else r2["team"]
            dog   = r2["team"] if p1 >= 0.5 else r1["team"]
            fav_p = max(p1, 1 - p1)
            fav_s = _team_strengths(r1 if p1 >= 0.5 else r2)
            str_line = (", ".join(fav_s[:2]) + " — ") if fav_s else ""
            # Check if already played
            played = next((r for r in results_list
                           if {r1["team"], r2["team"]} == set(r.get("teams", []))), None)
            if played:
                actual_w = played["winner"]
                model_w  = played["model_pick"]
                if actual_w == model_w:
                    return (f"✅ Already played — **{actual_w}** won. Called it. "
                            f"Had them at **{fav_p*100:.0f}%** and they delivered.")
                else:
                    label, explanation = _diagnose_miss(model_w, actual_w, bkt_df)
                    return (f"📉 Already played — **{actual_w}** won, I had **{model_w}**. "
                            f"`{label}` — {explanation}")
            confidence = "Lock." if fav_p > 0.72 else "Lean." if fav_p > 0.60 else "Toss-up, but I'm taking"
            return (f"⚔️ **{r1['team']}** ({sc1:.1f}) vs **{r2['team']}** ({sc2:.1f})\n\n"
                    f"**{fav}** — {str_line}{fav_p*100:.0f}% win probability. {confidence}\n\n"
                    f"{american_line(fav_p)} / {american_line(1-fav_p)}")

    # ── Specific team lookup ───────────────────────────────────────────────────
    matched = find_row(q_low)
    if matched is not None:
        score   = safe_f(matched.get("contender_score", 50))
        risk    = safe_f(matched.get("upset_risk_score", 25))
        arch    = matched.get("archetype", "—")
        seed    = int(matched["seed"]) if matched.get("seed") and not pd.isna(matched.get("seed")) else "—"
        sim_r   = matched.get("sim_round", "First Round")
        def_s   = safe_f(matched.get("defense_score", 50))
        cl_s    = safe_f(matched.get("clutch_score", 50))
        t_name  = matched["team"]
        strengths = _team_strengths(matched)
        str_line  = ("\n\n**What makes them dangerous:** " + ", ".join(strengths)) if strengths else ""
        champ_pct_row = ""
        if len(champs_df) and t_name in champs_df["team"].values:
            pct = float(champs_df[champs_df["team"] == t_name]["championship_pct"].iloc[0])
            champ_pct_row = f" | **{pct:.1f}%** title odds"

        # Save to context so follow-up messages reference this team
        _was_my_champ = (t_name == champion)
        _save_ctx(team=t_name, topic="team_lookup",
                  stance="liked" if score >= 65 else "faded" if score < 50 else "neutral")

        # Check if they've played a result
        team_results = [r for r in results_list if t_name in r.get("teams", [])]
        result_lines = []
        for r in team_results:
            if r["winner"] == t_name:
                opp = [t for t in r.get("teams", []) if t != t_name]
                result_lines.append(f"✅ Beat **{opp[0] if opp else '?'}** — called it.")
            else:
                label, explanation = _diagnose_miss(r.get("model_pick",""), r["winner"], bkt_df)
                result_lines.append(f"📉 Lost to **{r['winner']}** — `{label}`: {explanation}")
        result_block = ("\n\n" + "\n".join(result_lines)) if result_lines else ""

        # Reactive voice — high/mid/low score + was this my champion pick?
        if score >= 72:
            intro_opts = [
                f"📋 **{t_name}** — this is one of my top picks.{' My champion, actually.' if _was_my_champ else ''}",
                f"📋 **{t_name}** — yeah, I rate them. Here's why:",
                f"📋 **{t_name}** — #{seed} seed and I think the committee may have undersold them.",
            ]
            risk_line = (f"Upset risk **{risk:.0f}** — they're {'vulnerable in the wrong matchup' if risk >= 38 else 'as safe as anyone in this bracket'}.")
            _follow_champ_q = "They're my champion pick — hard to argue me off this." if _was_my_champ else "Are you riding with them or do you see a weakness I'm missing?"
            follow = f"\n\n*{_follow_champ_q}*"
        elif score >= 58:
            intro_opts = [
                f"📋 **{t_name}** — solid team, real threat, not my top pick but I respect them.",
                f"📋 **{t_name}** — #{seed} seed. Good but not great in my model.",
                f"📋 **{t_name}** — here's the honest take:",
            ]
            risk_line = f"Upset risk **{risk:.0f}** — {'danger zone, one bad game could end their run' if risk >= 40 else 'manageable, but not untouchable'}."
            follow = f"\n\n*You got them going further than {sim_r} or do you think they're a first-weekend exit?*"
        else:
            intro_opts = [
                f"📋 **{t_name}** — look, I'm not a huge believer here. Real talk:",
                f"📋 **{t_name}** — they made the bracket but the model isn't impressed.",
                f"📋 **{t_name}** — #{seed} seed and honestly I think they're overseeded.",
            ]
            risk_line = f"Upset risk **{risk:.0f}** — that's a red flag. I'd be cautious betting on them past round 1."
            follow = f"\n\n*You a believer or are you agreeing with me on this one?*"

        intro = _rng.choice(intro_opts)
        return (f"{intro}\n\n"
                f"#{seed} seed · {arch} · going to: **{sim_r}**\n\n"
                f"Score **{score:.1f}/100** | Defense **{def_s:.1f}** | Clutch **{cl_s:.1f}**{champ_pct_row}\n\n"
                f"{risk_line}"
                f"{str_line}"
                f"{result_block}"
                f"{follow}")

    # ── Sports banter / trash talk / casual chat ──────────────────────────────

    # ── Short filler / continuations ("ok", "and?", "really?", "go on") ─────
    _filler = ["ok", "okay", "and", "go on", "really", "for real", "word", "facts",
               "true", "hm", "hmm", "interesting", "lol", "lmao", "haha", "wow",
               "damn", "sheesh", "fr", "fr fr", "no cap", "yep", "yup", "bet"]
    if q_low.strip().rstrip("?!.") in _filler or len(q_low.split()) <= 2 and q_low.strip().rstrip("?!.") in _filler:
        if _last_team:
            row_ctx = bkt_df[bkt_df["team"] == _last_team]
            if len(row_ctx):
                sc = safe_f(row_ctx.iloc[0].get("contender_score", 50))
                rk = safe_f(row_ctx.iloc[0].get("upset_risk_score", 30))
                continuations = [
                    f"Yeah I said it. {_last_team} — **{sc:.1f}** contender score. That's not a fluke. "
                    f"{'Low upset risk too, {:.0f}. Solid all the way through.'.format(rk) if rk < 33 else 'Upset risk is {:.0f} though — winnable but not automatic.'.format(rk)}",
                    f"I mean it. I've been watching the data on {_last_team} all season. "
                    f"The **{sc:.1f}** holds up on both sides of the ball.",
                    f"Not changing my position. {_last_team} is my pick and I'm living with it."
                    + _follow_up(_last_team),
                ]
                _save_ctx()
                return _rng.choice(continuations)
        comeback = [
            "Give me more to work with. Team? Matchup? Hit me.",
            "Say less... actually say more. Who are we talking about?",
            "I'm here, just need a direction. Who you got questions about?",
        ]
        _save_ctx()
        return _rng.choice(comeback)

    # ── Greetings / checking in ────────────────────────────────────────────
    if any(w in q_low for w in ["what's up", "whats up", "hey statlas", "yo statlas",
                                 "sup", "how you doing", "what's good", "wsg", "what it do"]):
        openers = [
            f"Tournament's here. I've got {champion} going all the way. What do you think? You buying that or you fading?",
            f"Bracket's locked. My champion pick is **{champion}** and I'm not moving off it. Convince me I'm wrong.",
            "March Madness season. This is literally the best month in sports. Who you trying to break down first?",
            f"Ready. Ask me anything. Fair warning — I've got {champion} winning it all and I'll defend that pick all week.",
            "Let's talk ball. I got the data, you got the eye test — between the two of us we might actually nail this bracket.",
        ]
        _save_ctx(topic="greeting")
        return _rng.choice(openers)

    # ── User trash-talking Statlasberg ─────────────────────────────────────
    if any(w in q_low for w in ["you suck", "you're trash", "you're terrible", "you're bad",
                                  "terrible model", "worst model", "bad picks", "you're wrong",
                                  "your picks suck", "you stink", "garbage picks", "bs picks",
                                  "stupid model", "dumb model", "ur bad", "ur trash"]):
        claps = [
            f"Bold words. My picks are logged, yours aren't. Point at a specific pick and let's actually debate it — "
            f"who do you think I'm wrong about? {'Last thing we talked about was ' + _last_team + '.' if _last_team else ''}",
            "Yeah yeah. Every model's trash until it's right, then it was 'obvious.' Classic. "
            "Tell me which team I have wrong and explain why. I can take the criticism.",
            f"Okay, {_rng.choice(['fair enough', 'I hear you', 'noted'])}. "
            f"Which pick specifically? I'd rather argue on specifics than fight in the abstract. "
            f"{'Is it ' + _last_team + '?' if _last_team else 'Name a team.'}",
            "You know what? Maybe. I'm not perfect. But I'm better than a dartboard. "
            "Who do you think I've got wrong? Let's hash it out.",
        ]
        _save_ctx(topic="trash_talk")
        return _rng.choice(claps)

    # ── Trash talking a specific team ─────────────────────────────────────
    trash_words = ["sucks", "terrible", "overrated", "trash", "fraud", "garbage",
                   "soft", "can't win", "won't win", "no chance", "bums", "done",
                   "cooked", "ain't winning", "won't do anything", "mid"]
    team_trash = None
    for word in trash_words:
        if word in q_low:
            # Try to find a team they're trash-talking
            words_in_q = q_low.replace(word, "").strip()
            candidate = find_row(words_in_q)
            if candidate is not None:
                team_trash = candidate
                break

    if team_trash is not None:
        t_name  = team_trash["team"]
        t_score = safe_f(team_trash.get("contender_score", 50))
        t_risk  = safe_f(team_trash.get("upset_risk_score", 25))
        t_seed  = int(team_trash.get("seed", 5)) if team_trash.get("seed") and not pd.isna(team_trash.get("seed")) else 5
        _save_ctx(team=t_name, stance="faded")
        if t_score >= 70:
            pushbacks = [
                f"Nah I'm not letting you do that. **{t_name}** — **{t_score:.1f}** contender score. "
                f"That's not a soft pick, that's the data talking. "
                f"What specifically are you seeing that you don't like? Their schedule? Backcourt? Give me something.",
                f"**{t_name}** at {t_score:.1f} — I get it, maybe they haven't *looked* great on TV, "
                f"but their underlying numbers are legit. "
                f"Tell me the eye test reason and I'll either agree or push back with numbers.",
                f"Bold take but I respect the conviction. **{t_name}**'s score is {t_score:.1f} — "
                f"that's one of the better marks in the bracket. "
                f"Upset risk is {t_risk:.0f} so they're not bulletproof. What round do you see them losing?",
            ]
            return _rng.choice(pushbacks) + _follow_up(t_name)
        elif t_score >= 55:
            agree_disagree = [
                f"Honestly? I'm not riding **{t_name}** hard either. {t_score:.1f} score, "
                f"upset risk at **{t_risk:.0f}** — there's a real vulnerability there. "
                f"I wouldn't call them trash but they're definitely beatable in the right matchup. Who would you have beating them?",
                f"Not gonna lie, the model's lukewarm on **{t_name}** too. {t_score:.1f} is middling for a team at their seed. "
                f"Upset risk {t_risk:.0f} means they could easily go home early. "
                f"The question is who has the profile to beat them — any specific matchup you're scared of?",
            ]
            return _rng.choice(agree_disagree)
        else:
            roast_options = [
                f"Yeah I'm with you. **{t_name}** at {t_score:.1f} — that's not inspiring. "
                f"{'Overseeded at #{}.'.format(t_seed) if t_seed <= 6 else 'No real reason to trust them.'} "
                f"I'd fade them first round without guilt. Who replaces them in your bracket?",
                f"Hard to defend **{t_name}**. Score {t_score:.1f}, risk {t_risk:.0f}, "
                f"and they haven't beaten anyone that scares me. "
                f"If they win a game I'll be shocked. Who do you have going instead?",
            ]
            return _rng.choice(roast_options)

    # ── Hyping a team ──────────────────────────────────────────────────────
    hype_words = ["love", "my guy", "the team", "they're winning", "they gonna win",
                  "going all the way", "best team", "real deal", "legit", "they different",
                  "they're different", "can't stop them", "unbeatable", "they're built",
                  "i got", "i have", "riding with", "rolling with", "taking", "i like",
                  "i trust", "believe in"]
    for word in hype_words:
        if word in q_low:
            candidate = find_row(q_low.replace(word, "").strip())
            if candidate is not None:
                t_name  = candidate["team"]
                t_score = safe_f(candidate.get("contender_score", 50))
                t_risk  = safe_f(candidate.get("upset_risk_score", 25))
                str_tags = _team_strengths(candidate)
                str_line = (", ".join(str_tags[:2]) + " — ") if str_tags else ""
                _save_ctx(team=t_name, stance="liked")
                if t_score >= 70:
                    _champ_note = "That's my champion pick too." if t_name == champion else "Not my champion but they could get there."
                    opts = [
                        f"**{t_name}** — that's a pick I can ride with. {str_line}score **{t_score:.1f}**. "
                        f"{_champ_note} How deep you going with them?" + _follow_up(t_name),
                        f"Okay I respect that. **{t_name}** at {t_score:.1f} is legit. {str_line}"
                        f"Upset risk {t_risk:.0f} so they're not invincible but the floor is high. "
                        f"What round do you have them exiting?" + _follow_up(t_name),
                    ]
                    return _rng.choice(opts)
                elif t_score >= 55:
                    opts = [
                        f"I can see why you like **{t_name}**. {str_line}Score {t_score:.1f} — real pieces there. "
                        f"But upset risk is **{t_risk:.0f}**, which means they need a clean draw. "
                        f"What's your path for them — who do they need to avoid?",
                        f"**{t_name}** is interesting. {t_score:.1f} in my model — not top tier but dangerous. "
                        f"Tell me what you're seeing specifically that makes you ride with them.",
                    ]
                    return _rng.choice(opts)
                else:
                    opts = [
                        f"Ride or die with **{t_name}**, I respect the commitment. "
                        f"I'm not gonna sugarcoat it — {t_score:.1f} is a rough score and risk is {t_risk:.0f}. "
                        f"What are you seeing in them that the numbers are missing?",
                        f"Bold. **{t_name}** at {t_score:.1f} — the model doesn't love them. "
                        f"But I've seen the eye test beat the algorithm before. Make your case.",
                    ]
                    return _rng.choice(opts)

    # ── User shares eye-test opinion (for learning) ────────────────────────
    if any(w in q_low for w in ["eye test", "i think", "i believe", "i feel like", "in my opinion",
                                  "imo", "to me", "the way i see it", "watch tape", "tape says",
                                  "film says", "they look", "they seem", "they play"]):
        # Find any team mentioned
        for word in q_low.split():
            candidate = find_row(word) if len(word) >= 4 else None
            if candidate is not None:
                t_name = candidate["team"]
                break
        else:
            candidate = None

        # Store the eye-test note if we can tie it to a team
        if candidate is not None:
            t_name  = candidate["team"]
            t_score = safe_f(candidate.get("contender_score", 50))
            # Log it as an informal note in session state
            if "eye_test_notes" not in st.session_state:
                st.session_state.eye_test_notes = {}
            st.session_state.eye_test_notes[t_name] = q
            gap_dir = "higher" if t_score >= 60 else "lower"
            return (f"👁️ **{t_name}** — eye test logged. I've got a **{t_score:.1f}** on them.\n\n"
                    f"You're {('on the same page as me' if gap_dir == 'higher' else 'seeing something I might not be measuring')}. "
                    f"Data can miss effort, scheme fit, and momentum. That's why your eye test matters. "
                    f"After games are played, ask me *'analyze my reasoning on {t_name}'* and "
                    f"we'll see if the tape was right.")
        return ("Eye test is real — data can't capture everything. "
                "Tell me which team and what you're seeing. "
                "I'll log it and we'll see if you were right when the dust settles.")

    # ── Jokes / banter about March Madness ────────────────────────────────
    if any(w in q_low for w in ["joke", "funny", "make me laugh", "something funny",
                                  "entertain me", "tell me something", "roast"]):
        jokes = [
            "A 12 seed walks into a bar. The 5 seed says 'you're not supposed to win.' "
            "The 12 says 'tell that to the committee.' Same story every March.",
            "What does a bracketologist's wife say to him in April? "
            "'How'd your picks do?' **He cries.**",
            "I told a 16 seed they had no chance. They had me blocked on all platforms by Sunday.",
            "Nobody loses faster than someone who picked all 1 seeds in a pool. "
            "Absolute cowards. Zero imagination.",
            "The best part of March Madness? Every casual fan who picked the cute mascot "
            "is beating the PhD statistician through the first weekend.",
        ]
        return _rng.choice(jokes)

    # ── User asking Statlasberg to roast a team ─────────────────────────────
    if any(w in q_low for w in ["roast", "drag", "clown", "talk trash", "trash talk",
                                  "say something bad", "rip on", "clown on"]):
        # Find team to roast
        candidate = find_row(q_low.replace("roast", "").replace("drag", "").replace("clown", "")
                                   .replace("talk trash", "").replace("trash talk", "").strip())
        if candidate is not None:
            t_name  = candidate["team"]
            t_score = safe_f(candidate.get("contender_score", 50))
            t_seed  = int(candidate.get("seed", 8)) if candidate.get("seed") and not pd.isna(candidate.get("seed")) else 8
            t_arch  = candidate.get("archetype", "")
            roasts = [
                (f"**{t_name}**. Contender score {t_score:.0f}. "
                 f"Their biggest March Madness achievement this year is making the bracket. "
                 f"Congratulations on that. "
                 f"They're gonna play great for 30 minutes and then completely forget "
                 f"how basketball works in the last two minutes. Watch."),
                (f"You want me to roast **{t_name}**? Alright — "
                 f"they're good enough to get everyone's hopes up and not good enough to do anything about it. "
                 f"That's the cruelest archetype in the tournament. "
                 f"Score of {t_score:.0f}. Exists to ruin brackets, not win rings."),
                (f"**{t_name}** has the energy of a team that peaked in practice. "
                 f"#{t_seed} seed, {t_score:.0f} score. "
                 f"They're going to look incredible until they don't. "
                 f"That's it. That's the whole scouting report."),
            ]
            return _rng.choice(roasts)
        teams_to_roast = in_bracket.sort_values("upset_risk_score", ascending=False).head(3)["team"].tolist()
        return (f"Give me a team name and I'll tear them apart. "
                f"Off the top of my head, if you want some easy targets: "
                f"{', '.join(teams_to_roast)} all have ugly upset numbers. "
                f"Take your pick.")

    # ── Profanity / expressive frustration ────────────────────────────────
    if any(w in q_low for w in ["wtf", "what the f", "what the hell", "holy sh", "what the sh",
                                  "no way", "are you kidding", "no shot", "impossible", "insane",
                                  "unbelievable", "how is this", "this is crazy", "wild"]):
        team_mentioned = find_row(q_low)
        if team_mentioned is not None:
            t_name  = team_mentioned["team"]
            t_score = safe_f(team_mentioned.get("contender_score", 50))
            return (f"I KNOW. **{t_name}** — chaos. "
                    f"Model had them at {t_score:.1f} which is {'respectable' if t_score >= 55 else 'nothing special'}, "
                    f"but March doesn't care about numbers at 2am on a Thursday. "
                    f"This is why you watch. This is literally why you watch.")
        banter_chaos = [
            "Yeah March Madness is absolutely unhinged and that's exactly why we're all here.",
            "I can't explain all of it. Nobody can. That's the point. The chaos is the product.",
            "The tournament has been doing this for 40 years. We should stop being surprised and just enjoy it.",
        ]
        return _rng.choice(banter_chaos)

    # ── Agreeing / disagreeing with Statlasberg ────────────────────────────
    if any(w in q_low for w in ["i agree", "you're right", "you were right", "fair enough",
                                  "can't argue", "that makes sense", "respect", "good point",
                                  "you got me", "you're not wrong"]):
        ctx_team = _last_team or champion
        responses = [
            f"Appreciate that. {'On ' + ctx_team + '? ' if ctx_team else ''}I knew you'd come around. "
            + _follow_up(ctx_team),
            f"That's all I need to hear. Now what else you want to break down?",
            f"Good. Now go tell your friends. {ctx_team + ' is real.' if ctx_team else 'Trust the process.'}",
            f"We're on the same page. Doesn't mean this bracket doesn't go sideways — March gonna March — "
            f"but at least we're starting from the right place." + _follow_up(),
        ]
        _save_ctx()
        return _rng.choice(responses)

    if any(w in q_low for w in ["i disagree", "i don't think so", "nah", "nope", "not buying it",
                                  "wrong", "doubt it", "no way", "disagree", "i doubt",
                                  "i don't believe", "respectfully disagree", "hard disagree",
                                  "disagree hard", "no i don't", "i don't buy"]):
        team_mentioned = find_row(q_low)
        t_target = (team_mentioned["team"] if team_mentioned is not None else _last_team)
        if t_target:
            row_d = bkt_df[bkt_df["team"] == t_target]
            sc_d  = safe_f(row_d.iloc[0].get("contender_score", 50)) if len(row_d) else 50
            _save_ctx(team=t_target)
            pushback_opts = [
                f"Disagree on **{t_target}** — alright, make your case. "
                f"What's the angle? Schedule? Injuries? A matchup you think breaks them? "
                f"I've got them at {sc_d:.1f} and I'll defend it but I'm listening.",
                f"Okay disagreeing on **{t_target}**. Tell me what you see that I'm missing. "
                f"I'm not married to the number — {sc_d:.1f} could move if you give me a real reason.",
                f"You'd rather fade **{t_target}**? Alright. How early you got them losing? "
                f"First round, second round? Tell me the team and I'll run the matchup numbers.",
            ]
            return _rng.choice(pushback_opts)
        _save_ctx()
        return _rng.choice([
            "Alright, tell me what pick specifically. I'm not gonna just sit here taking heat "
            "without knowing what game we're talking about.",
            f"Disagree with {'what I said about ' + _last_team if _last_team else 'what?'} "
            "Give me the specific pick and your reasoning.",
            "Noted. But 'no' isn't an argument. Which team and why?",
        ])

    # ── Talking about the tournament broadly ──────────────────────────────
    if any(w in q_low for w in ["love march", "love the tournament", "best time of year",
                                  "tournament time", "march is", "best sport", "nothing better",
                                  "greatest tournament", "best bracket"]):
        return ("March Madness is the best sporting event on the planet. "
                "Don't @ me. 68 teams, 6 rounds, one champion, and somewhere in there "
                "an 11 seed is going to end someone's whole bracket and then the commentator "
                "is gonna say 'Cinderella story!' for the 40th year in a row. "
                "I love it. Let's break this thing down.")

    # ── Asking about data / methodology ────────────────────────────────────
    if any(w in q_low for w in ["how do you", "how does the model", "what are you based on",
                                  "what data", "methodology", "how is your score", "where do you get",
                                  "your formula", "your algorithm", "how do you calculate",
                                  "how do you work", "what factors"]):
        return ("Here's the honest breakdown:\n\n"
                "My **contender_score** is a weighted composite of:\n"
                "- **Defense** (25%) — stops you dead or leaks points\n"
                "- **Guard play** (22%) — senior guards win March, period\n"
                "- **Clutch** (20%) — one-possession game history\n"
                "- **Efficiency** (18%) — adjusted offense/defense margins\n"
                "- **Rebounding** (10%) — extra possessions = extra wins\n"
                "- **Consistency** (5%) — no random blowout-loss risk\n\n"
                "Archetypes capture *how* teams play (Four Factors). "
                "KenPom-proxy cross-checks the pick against adjusted efficiency. "
                "I'm not an LLM — I don't guess, I calculate. "
                "But I can still get it wrong. That's March.")

    # ── Asking about specific matchup eye-test ────────────────────────────
    if any(w in q_low for w in ["what do you think about", "thoughts on", "talk to me about",
                                  "give me your take", "what's your take", "break down"]):
        # Try to find team or matchup
        vs_match2 = _re.search(r'(.+?)\s+(?:vs\.?|versus|against|or)\s+(.+)', q_low)
        if vs_match2:
            t1_q = vs_match2.group(1).strip().rstrip("?")
            t2_q = vs_match2.group(2).strip().rstrip("?")
            r1, r2 = find_row(t1_q), find_row(t2_q)
            if r1 is not None and r2 is not None:
                # Reuse the vs matchup logic
                sc1, sc2 = safe_f(r1.get("contender_score",50)), safe_f(r2.get("contender_score",50))
                p1 = win_prob_sigmoid(sc1, sc2)
                fav = r1["team"] if p1 >= 0.5 else r2["team"]
                fav_p = max(p1, 1-p1)
                arc = get_matchup_arc(dict(r1), dict(r2)) if _ARCHETYPES_OK else ""
                return (f"📊 **{r1['team']}** ({sc1:.1f}) vs **{r2['team']}** ({sc2:.1f})\n\n"
                        f"**{fav}** wins — {fav_p*100:.0f}% probability.\n\n"
                        + (f"*Matchup read:* {arc}" if arc else ""))
        candidate = find_row(q_low.replace("what do you think about","").replace("thoughts on","")
                                   .replace("talk to me about","").replace("give me your take","")
                                   .replace("what's your take on","").replace("break down","").strip())
        if candidate is not None:
            # Fall through to the team lookup handler above — re-trigger it
            t_name  = candidate["team"]
            t_score = safe_f(candidate.get("contender_score",50))
            t_risk  = safe_f(candidate.get("upset_risk_score",25))
            t_arch  = candidate.get("archetype","—")
            t_seed_v = int(candidate["seed"]) if candidate.get("seed") and not pd.isna(candidate.get("seed")) else "—"
            str_tags = _team_strengths(candidate)
            str_line = ("\n\n**What makes them dangerous:** " + ", ".join(str_tags)) if str_tags else ""
            return (f"📋 **{t_name}** — #{t_seed_v} seed · {t_arch}\n\n"
                    f"Contender score **{t_score:.1f}/100** | Upset risk **{t_risk:.0f}**\n\n"
                    + str_line)

    # ── Default — context-aware, never just a static menu ────────────────────
    _save_ctx()
    if _last_team:
        # Reference the last thing we discussed to stay in the conversation
        row_ctx = bkt_df[bkt_df["team"] == _last_team]
        sc_ctx  = safe_f(row_ctx.iloc[0].get("contender_score", 50)) if len(row_ctx) else 50
        defaults = [
            f"I'm not sure I followed that — were you still talking about **{_last_team}**? "
            f"If so, ask me about their matchup, their path, or why you think I'm wrong on them.",
            f"Lost me there. You still on **{_last_team}** or did we move to a new team?",
            f"Say that differently? We were just talking about {_last_team} ({sc_ctx:.0f}) — "
            f"if that's still the subject, try: *'Who beats them?'* or *'How far are they going?'*",
        ]
        return _rng.choice(defaults)
    # First interaction — give a punchy prompt, not a wall of commands
    first_time = [
        f"I'm locked in. Start with a team, a matchup, or just ask who I've got winning it all. "
        f"Hint: it's **{champion}**. Fight me.",
        f"Talk ball to me. Who's your pick to win it all? I'll tell you mine and we'll debate.",
        f"Ask me anything — who I'm fading, who's going deep, my Final Four, a specific matchup. "
        f"Or just say a team name and I'll break them down.",
        f"Tell me a team and I'll give you the full breakdown. Or ask *'who should I fade?'* "
        f"and I'll start the conversation.",
    ]
    return _rng.choice(first_time)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — ASK STATLASBERG  (Q&A + Model Comparison)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.caption("How does Statlasberg compare to KenPom-proxy and seed baseline? Disagreements often reveal where the model sees value.")

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
                    seed_str = f" (#{int(row.iloc[0]['seed'])})" if len(row) and pd.notna(row.iloc[0].get('seed')) else ""
                    st.markdown(f"• {t}{seed_str}")
                st.markdown(f"**Champion:** {sim_champion}")
            with ff_col2:
                st.markdown(f"**📊 {kp_label.split(chr(10))[0]}**")
                for t in kp_ff:
                    row = comp_df[comp_df["team"] == t]
                    seed_str = f" (#{int(row.iloc[0]['seed'])})" if len(row) and pd.notna(row.iloc[0].get('seed')) else ""
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
                        f'<span style="color:#94a3b8">(#{int(row["seed"]) if pd.notna(row.get("seed")) else "?"} seed)</span> — '
                        f'Statl. rank <strong style="color:#4ade80">#{int(row["statl_rank"])}</strong> · '
                        f'KP-Proxy rank <strong style="color:#f87171">#{int(row["kp_rank"])}</strong>'
                        f'</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Load bracket data to see model comparisons.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — MODEL ACCURACY  (Historical Backtest 2015–2025)
# ─────────────────────────────────────────────────────────────────────────────
with tab9:
    st.markdown('<div class="section-header">📈 Model Accuracy — Historical Backtest (2015–2025)</div>',
                unsafe_allow_html=True)
    st.caption("How well does Statlasberg's scoring model rank actual NCAA champions? "
               "Each season uses real Sports-Reference stats. No future information is used.")

    if not os.path.exists(BACKTEST_PATH):
        st.warning("⚠️ No backtest results found. Run `python run_backtest.py --data-dir data/raw/teams` first.")
    else:
        bt = pd.read_csv(BACKTEST_PATH)
        valid_bt = bt[bt["champion_rank"].notna() & (bt["champion_rank"] > 0)].copy()

        if len(valid_bt) > 0:
            # ── Summary KPIs ─────────────────────────────────────────────────
            avg_rank    = valid_bt["champion_rank"].mean()
            top3_pct    = (valid_bt["champion_rank"] <= 3).mean() * 100
            top5_pct    = (valid_bt["champion_rank"] <= 5).mean() * 100
            top10_pct   = (valid_bt["champion_rank"] <= 10).mean() * 100
            n_seasons   = len(valid_bt)

            st.markdown("### Overall Performance")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Champion Rank",  f"#{avg_rank:.1f}",
                      delta=None, help="Lower is better. Statlasberg's average ranking of the eventual champion.")
            k2.metric("Top-5 Rate",  f"{top5_pct:.0f}%",
                      help="% of seasons where the actual champion was ranked top-5 by the model.")
            k3.metric("Top-10 Rate", f"{top10_pct:.0f}%",
                      help="% of seasons the champion was in the model's top 10.")
            k4.metric("Seasons Backtested", f"{n_seasons}",
                      help="All seasons using real Sports-Reference data (2015–2025, skip 2020).")

            # Grade banner
            if avg_rank <= 5:
                grade_msg = "🏆 **Elite accuracy** — the model consistently identifies championship-caliber teams."
                grade_color = "#1a7f1a"
            elif avg_rank <= 10:
                grade_msg = "✅ **Strong accuracy** — champion is almost always in the model's top 10."
                grade_color = "#2468a0"
            else:
                grade_msg = "⚠️ **Room to improve** — model sometimes misses the eventual champion."
                grade_color = "#b85c00"
            st.markdown(
                f'<div style="background:{grade_color}22;border-left:4px solid {grade_color};'
                f'padding:10px 16px;border-radius:6px;margin:12px 0;">{grade_msg}</div>',
                unsafe_allow_html=True
            )

            # ── Rank Distribution Bar Chart ───────────────────────────────────
            st.markdown("### Champion Rank Each Season")

            import altair as alt
            chart_df = valid_bt[["season", "champion_rank", "actual_champion",
                                  "model_top_pick", "champion_contender_score"]].copy()
            chart_df["season"] = chart_df["season"].astype(str)
            chart_df["hit_label"] = chart_df["champion_rank"].apply(
                lambda r: "🟢 Top 5" if r <= 5 else ("🟡 Top 10" if r <= 10 else "🔴 Outside Top 10")
            )
            chart_df["bar_color"] = chart_df["champion_rank"].apply(
                lambda r: "#2ecc71" if r <= 5 else ("#f39c12" if r <= 10 else "#e74c3c")
            )

            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("season:N", title="Season", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("champion_rank:Q", title="Champion Rank (lower = better)",
                            scale=alt.Scale(domain=[0, max(25, int(chart_df["champion_rank"].max()) + 2)])),
                    color=alt.Color("hit_label:N",
                                    scale=alt.Scale(
                                        domain=["🟢 Top 5", "🟡 Top 10", "🔴 Outside Top 10"],
                                        range=["#2ecc71", "#f39c12", "#e74c3c"]
                                    ),
                                    legend=alt.Legend(title="Model Accuracy")),
                    tooltip=[
                        alt.Tooltip("season:N",                  title="Season"),
                        alt.Tooltip("actual_champion:N",         title="Champion"),
                        alt.Tooltip("champion_rank:Q",           title="Model Rank"),
                        alt.Tooltip("champion_contender_score:Q", title="Score", format=".1f"),
                        alt.Tooltip("model_top_pick:N",          title="Model's #1 Pick"),
                    ]
                )
                .properties(height=320, title="Champion Rank by Season")
            )

            # Add a reference line at y=5
            rule = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(
                color="#2ecc71", strokeDash=[6, 4], size=1.5
            ).encode(y="y:Q")

            st.altair_chart(bar_chart + rule, use_container_width=True)

            # ── Year-by-Year Table ────────────────────────────────────────────
            st.markdown("### Year-by-Year Results")

            display_bt = valid_bt[["season", "actual_champion", "champion_rank",
                                    "champion_contender_score", "model_top_pick",
                                    "top5_picks"]].copy()
            display_bt = display_bt.rename(columns={
                "season":                   "Year",
                "actual_champion":          "Champion",
                "champion_rank":            "Model Rank",
                "champion_contender_score": "Champion Score",
                "model_top_pick":           "Model's #1 Pick",
                "top5_picks":               "Model Top-5 Picks",
            })
            display_bt["Champion Score"] = display_bt["Champion Score"].round(1)

            def _rank_style(rank):
                if rank <= 3:  return "color:#2ecc71; font-weight:700"
                if rank <= 5:  return "color:#27ae60; font-weight:600"
                if rank <= 10: return "color:#f39c12; font-weight:600"
                return "color:#e74c3c; font-weight:600"

            # Render as styled HTML table
            rows_html = ""
            for _, row in display_bt.iterrows():
                rank = int(row["Model Rank"])
                style = _rank_style(rank)
                icon = "🟢" if rank <= 5 else ("🟡" if rank <= 10 else "🔴")
                top_pick = row["Model's #1 Pick"]
                champion = row["Champion"]
                hit = "✅ " if champion == top_pick else ""
                top5 = row["Model Top-5 Picks"]
                score = row["Champion Score"]
                year = int(row["Year"])
                rows_html += (
                    f"<tr>"
                    f"<td><b>{year}</b></td>"
                    f"<td>{champion}</td>"
                    f"<td style='{style}'>{icon} #{rank}</td>"
                    f"<td>{score:.1f}</td>"
                    f"<td>{hit}{top_pick}</td>"
                    f"<td style='font-size:0.82em;color:#aaa'>{top5}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>
                <thead><tr style='background:#1e2a3a;'>
                  <th style='padding:8px 10px;text-align:left'>Year</th>
                  <th style='padding:8px 10px;text-align:left'>Champion</th>
                  <th style='padding:8px 10px;text-align:left'>Model Rank</th>
                  <th style='padding:8px 10px;text-align:left'>Score</th>
                  <th style='padding:8px 10px;text-align:left'>Model's #1 Pick</th>
                  <th style='padding:8px 10px;text-align:left'>Top-5 Picks</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
                </table>""",
                unsafe_allow_html=True
            )

            # ── Methodology Note ──────────────────────────────────────────────
            st.markdown("---")
            with st.expander("ℹ️ How the backtest works", expanded=False):
                st.markdown("""
**Data**: Real team stats scraped from Sports-Reference.com for each season
(2015–2019, 2021–2025 — 2020 skipped, no tournament).

**Method**:
1. For each season, load that year's full D-I stats (offense, defense, SRS, etc.)
2. Run the same scoring pipeline used for 2026 predictions — no look-ahead data
3. Rank all D-I teams by `contender_score`
4. Record where the actual NCAA champion placed in that ranking

**What "Champion Rank" means**:
- Rank 1 = the model correctly identified the champion as its top pick
- Rank 5 = champion was in the top 5 but not #1 pick
- Rank 22 = worst case (2017 North Carolina — UNC's SRS was penalized by a tough ACC schedule)

**Limitations**:
- The model uses season-long stats, not tournament-specific adjustments
- Bracket matchups (who each team would face) are not factored into the backtest rank
- Injury overrides are NOT applied to historical seasons (they're 2026-specific)
- 2020 excluded — COVID cancelled the tournament
                """)

        else:
            st.info("No valid backtest data available. Run the backtest first.")
