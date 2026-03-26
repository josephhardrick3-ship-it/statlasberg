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

# Persistent results CSV — referenced by both Bracket tab and Recap tab
_RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "outputs", "game_results_2026.csv"
)
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


@st.cache_data(ttl=600, show_spinner=False)
def run_monte_carlo_sim(r32_pairs_frozen, score_lkp_frozen, n=500_000):
    """Vectorized 500K Monte Carlo simulation from R32.

    r32_pairs_frozen: tuple of 16 (t1, t2) pairs in region order
      [East×4, South×4, West×4, Midwest×4] with adjacent pairs forming S16 matchups.
    Returns dict: team → Sweet 16 advancement probability.
    """
    r32_pairs = list(r32_pairs_frozen)
    slkp      = dict(score_lkp_frozen)

    # Build flat team list and score/prob arrays
    team_set = []
    for t1, t2 in r32_pairs:
        if t1 and t1 not in team_set: team_set.append(t1)
        if t2 and t2 not in team_set: team_set.append(t2)
    n_teams = len(team_set)
    tidx    = {t: i for i, t in enumerate(team_set)}
    scores  = np.array([slkp.get(t, 50.0) for t in team_set])

    # Win-probability matrix: prob_mat[i,j] = P(team_i beats team_j)
    diff     = scores[:, None] - scores[None, :]          # (n_teams, n_teams)
    prob_mat = 1.0 / (1.0 + np.exp(-diff / 8.0))         # sigmoid

    # Convert r32 pairs to index arrays
    valid   = [(t1, t2) for t1, t2 in r32_pairs if t1 and t2]
    r32_t1  = np.array([tidx[t1] for t1, _ in valid])    # (16,)
    r32_t2  = np.array([tidx[t2] for _, t2 in valid])    # (16,)

    # ── R32 ──────────────────────────────────────────────────────────────────
    r32_p   = prob_mat[r32_t1, r32_t2]                   # (16,)
    r32_rng = np.random.random((n, len(valid)))
    r32_win = np.where(r32_rng < r32_p, r32_t1, r32_t2)  # (n, 16) - winners as ints

    # ── S16 advancement: count how often each team wins their R32 game ────────
    s16_counts = np.bincount(r32_win.flatten(), minlength=n_teams)
    return {team_set[i]: float(s16_counts[i]) / n for i in range(n_teams)}


@st.cache_data(ttl=600, show_spinner=False)
def run_full_bracket_monte_carlo(r32_frozen, score_lkp_frozen, n=500_000):
    """Vectorized full-bracket Monte Carlo: R32 through Championship.

    r32_frozen: tuple of 16 (t1, t2, completed, actual_winner) tuples
      Ordered: East×4, South×4, West×4, Midwest×4.
      Adjacent pairs form S16 matchups within each region.
    score_lkp_frozen: tuple of (team, adapted_score) pairs
    Returns dict: team → championship_pct (0–100)
    """
    r32_info = list(r32_frozen)
    slkp     = dict(score_lkp_frozen)

    # Build flat team list and index map
    team_set = []
    for t1, t2, _, _ in r32_info:
        if t1 and t1 not in team_set: team_set.append(t1)
        if t2 and t2 not in team_set: team_set.append(t2)
    n_teams = len(team_set)
    if n_teams == 0:
        return {}
    tidx    = {t: i for i, t in enumerate(team_set)}
    scores  = np.array([slkp.get(t, 50.0) for t in team_set])

    # Win-probability matrix: prob_mat[i,j] = P(team_i beats team_j)
    diff     = scores[:, None] - scores[None, :]
    prob_mat = 1.0 / (1.0 + np.exp(-diff / 8.0))

    # Convert R32 matchups to index arrays
    r32_t1 = np.array([tidx[t1] for t1, t2, _, _ in r32_info])  # (16,)
    r32_t2 = np.array([tidx[t2] for t1, t2, _, _ in r32_info])  # (16,)

    # ── R32 (16 games) ─────────────────────────────────────────────────────
    r32_p   = prob_mat[r32_t1, r32_t2]                           # (16,)
    r32_rng = np.random.random((n, 16))
    r32_win = np.where(r32_rng < r32_p, r32_t1, r32_t2)         # (n, 16)

    # Lock completed games — override random result with actual winner
    for i, (t1, t2, done, aw) in enumerate(r32_info):
        if done and aw and aw in tidx:
            r32_win[:, i] = tidx[aw]

    # ── S16 (8 games): adjacent R32 winner pairs ──────────────────────────
    s16_t1  = r32_win[:, 0::2]                                   # (n, 8)
    s16_t2  = r32_win[:, 1::2]                                   # (n, 8)
    s16_p   = prob_mat[s16_t1, s16_t2]
    s16_rng = np.random.random((n, 8))
    s16_win = np.where(s16_rng < s16_p, s16_t1, s16_t2)         # (n, 8)

    # ── E8 (4 games): adjacent S16 winner pairs per region ────────────────
    e8_t1   = s16_win[:, 0::2]                                   # (n, 4)
    e8_t2   = s16_win[:, 1::2]                                   # (n, 4)
    e8_p    = prob_mat[e8_t1, e8_t2]
    e8_rng  = np.random.random((n, 4))
    e8_win  = np.where(e8_rng < e8_p, e8_t1, e8_t2)             # (n, 4)

    # ── FF (2 games): East(0) vs Midwest(3), South(1) vs West(2) ──────────
    ff_t1   = e8_win[:, [0, 1]]                                  # (n, 2)
    ff_t2   = e8_win[:, [3, 2]]                                  # (n, 2)
    ff_p    = prob_mat[ff_t1, ff_t2]
    ff_rng  = np.random.random((n, 2))
    ff_win  = np.where(ff_rng < ff_p, ff_t1, ff_t2)             # (n, 2)

    # ── Championship (1 game) ─────────────────────────────────────────────
    ch_t1   = ff_win[:, 0]                                       # (n,)
    ch_t2   = ff_win[:, 1]                                       # (n,)
    ch_p    = prob_mat[ch_t1, ch_t2]
    ch_rng  = np.random.random(n)
    champs  = np.where(ch_rng < ch_p, ch_t1, ch_t2)             # (n,)

    # Count championships per team
    champ_counts = np.bincount(champs, minlength=n_teams)
    return {team_set[i]: float(champ_counts[i]) / n * 100 for i in range(n_teams)}


def _tossup_edge(t1_name, t2_name, bkt_df):
    """Return tossup-based score adjustment for close matchups.

    Positive = t1 edge, negative = t2 edge. Magnitude 0-3 pts.
    Uses efficiency margin, defense score, turnover discipline,
    off rebounding, clutch score, and guard play — the metrics
    that decide close tournament games.
    """
    r1 = bkt_df[bkt_df["team"] == t1_name]
    r2 = bkt_df[bkt_df["team"] == t2_name]
    if len(r1) == 0 or len(r2) == 0:
        return 0.0
    r1, r2 = r1.iloc[0], r2.iloc[0]

    edge = 0.0
    # Efficiency margin (adj_off - adj_def) — strongest predictor
    em1 = safe_f(r1.get("adj_offense")) - safe_f(r1.get("adj_defense"))
    em2 = safe_f(r2.get("adj_offense")) - safe_f(r2.get("adj_defense"))
    if em1 != em2:
        edge += 0.6 if em1 > em2 else -0.6

    # Defense score composite (higher = better) — replaces empty opp_three_pt_pct
    ds1 = safe_f(r1.get("defense_score"))
    ds2 = safe_f(r2.get("defense_score"))
    if ds1 != ds2:
        edge += 0.5 if ds1 > ds2 else -0.5

    # Turnover discipline (lower turnover_pct = fewer giveaways = better)
    tp1 = safe_f(r1.get("turnover_pct"))
    tp2 = safe_f(r2.get("turnover_pct"))
    if tp1 > 0 and tp2 > 0 and tp1 != tp2:
        edge += 0.4 if tp1 < tp2 else -0.4

    # Off rebounding (higher = better second-chance points)
    orb1 = safe_f(r1.get("off_rebound_pct"))
    orb2 = safe_f(r2.get("off_rebound_pct"))
    if orb1 != orb2:
        edge += 0.3 if orb1 > orb2 else -0.3

    # Clutch score (higher = better in close games)
    cl1 = safe_f(r1.get("clutch_score"))
    cl2 = safe_f(r2.get("clutch_score"))
    if cl1 != cl2:
        edge += 0.2 if cl1 > cl2 else -0.2

    # Guard play score (higher = better ball-handling under pressure)
    gp1 = safe_f(r1.get("guard_play_score"))
    gp2 = safe_f(r2.get("guard_play_score"))
    if gp1 != gp2:
        edge += 0.2 if gp1 > gp2 else -0.2

    return edge


def build_round_matchups(bkt_df):
    """Return matchups for every round as a dict: round → list of (t1, t2, winner, loser, region)."""
    rounds = {"R64": [], "R32": [], "S16": [], "E8": [], "FF": [], "Championship": []}
    score_lkp = {r["team"]: safe_f(r.get("contender_score", 50)) for _, r in bkt_df.iterrows()}

    def pwin(t1, t2):
        if not t1: return t2, t1
        if not t2: return t1, t2
        s1, s2 = score_lkp.get(t1, 50), score_lkp.get(t2, 50)
        p = win_prob_sigmoid(s1, s2)
        # For close games (50-68%), tossup metrics break the tie
        if 0.32 < p < 0.68:
            tossup_adj = _tossup_edge(t1, t2, bkt_df)
            # Add tossup as a score adjustment (each point of edge ≈ shift in win prob)
            adj_s1 = s1 + tossup_adj
            adj_s2 = s2 - tossup_adj
            p = win_prob_sigmoid(adj_s1, adj_s2)
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


def _compute_adapted_scores(bkt_df, recap_df):
    """Return contender score lookup adjusted for actual game results.

    When a team wins as a model underdog, boost their effective score toward
    their opponent's — making future projections more accurate as results come in.
    Winners of expected results get a small confidence bump.
    """
    base = {r["team"]: safe_f(r.get("contender_score", 50)) for _, r in bkt_df.iterrows()}
    adj = base.copy()
    if recap_df is None or recap_df.empty:
        return adj
    for _, row in recap_df.iterrows():
        w = str(row.get("winner", "") or "").strip()
        l = str(row.get("loser",  "") or "").strip()
        if not w or not l or w not in adj or l not in adj:
            continue
        ws = adj[w]; ls = adj[l]
        if ls > ws:                              # upset — winner had lower score
            adj[w] = ws + (ls - ws) * 0.45      # 45% pull toward beaten opponent's score
            adj[l] = ls - (ls - ws) * 0.20      # loser docked slightly
        else:                                    # expected result
            adj[w] = ws + 1.0                   # small confidence bump for winner
    return adj


def build_bracket_live(bkt_df, recap_df, adapted_scores=None):
    """Build round matchups overlaying actual results on top of model projections.

    Uses adapted_scores (Bayesian-updated contender scores) for future-round
    projections so the simulation improves as results come in.

    Returns dict: round -> list of matchup dicts with keys:
      t1, t2, s1, s2, winner, loser, region, winner_p, model_winner, completed, model_correct
    """
    # Use adapted scores for projections if provided, raw scores for model_winner baseline
    score_lkp = adapted_scores if adapted_scores is not None else \
                {r["team"]: safe_f(r.get("contender_score", 50)) for _, r in bkt_df.iterrows()}
    seed_lkp  = {r["team"]: (int(r["seed"]) if pd.notna(r.get("seed")) else 0) for _, r in bkt_df.iterrows()}

    # Build name normaliser: results CSV may use different variants (e.g. "Prairie View")
    # than the bracket (e.g. "Prairie View A&M").  Map any known alias → bracket canonical.
    _bracket_names = {r["team"] for _, r in bkt_df.iterrows()}
    _to_bkt = {n: n for n in _bracket_names}
    for _alias, _canon in _BRACKET_NORM.items():
        if _canon in _bracket_names:
            _to_bkt[_alias] = _canon
        if _alias in _bracket_names:
            _to_bkt[_canon] = _alias
    def _bn(name: str) -> str:
        return _to_bkt.get(name, name)

    actual = {}  # frozenset({winner, loser}) → (winner, model_pick_from_csv)
    if recap_df is not None and not recap_df.empty:
        for _, row in recap_df.iterrows():
            w = _bn(str(row.get("winner", "") or "").strip())
            l = _bn(str(row.get("loser",  "") or "").strip())
            mp = _bn(str(row.get("model_pick", "") or "").strip())
            if w and l:
                actual[frozenset({w, l})] = (w, mp)

    def resolve(t1, t2):
        if not t1 or not t2:
            w = t1 or t2
            return {"winner": w, "loser": "", "winner_p": 1.0, "model_winner": w,
                    "completed": False, "model_correct": None}
        c1 = score_lkp.get(t1, 50); c2 = score_lkp.get(t2, 50)
        p1 = win_prob_sigmoid(c1, c2)
        # For close games, add tossup edge to improve pick accuracy
        if 0.32 < p1 < 0.68:
            te = _tossup_edge(t1, t2, bkt_df)
            p1 = win_prob_sigmoid(c1 + te, c2 - te)
        mw  = t1 if p1 >= 0.5 else t2
        ml  = t2 if p1 >= 0.5 else t1
        mwp = max(p1, 1 - p1)
        key = frozenset({t1, t2})
        if key in actual:
            aw, csv_pick = actual[key]
            al = t2 if aw == t1 else t1
            # Use CSV model_pick for completed games (the pick the user actually saw)
            recorded_mw = csv_pick if csv_pick else mw
            return {"winner": aw, "loser": al, "winner_p": mwp, "model_winner": recorded_mw,
                    "completed": True, "model_correct": (aw == recorded_mw)}
        return {"winner": mw, "loser": ml, "winner_p": mwp, "model_winner": mw,
                "completed": False, "model_correct": None}

    rounds = {"R64": [], "R32": [], "S16": [], "E8": [], "FF": [], "Championship": []}
    PODS = [
        ([(1, 16), (8, 9)],  [(5, 12), (4, 13)]),
        ([(6, 11), (3, 14)], [(7, 10), (2, 15)]),
    ]
    ff_teams = []
    for region in ["East", "South", "West", "Midwest"]:
        reg = bkt_df[bkt_df["region"] == region]
        s2t = {int(r["seed"]): r["team"] for _, r in reg.iterrows() if pd.notna(r.get("seed"))}
        region_e8 = []
        for pod_a_pairs, pod_b_pairs in PODS:
            r1w_a = []; r1w_b = []
            for s1, s2 in pod_a_pairs:
                t1n, t2n = s2t.get(s1, ""), s2t.get(s2, "")
                if t1n or t2n:
                    res = resolve(t1n, t2n)
                    rounds["R64"].append({"t1": t1n, "t2": t2n, "s1": s1, "s2": s2, "region": region, **res})
                    if res["winner"]: r1w_a.append(res["winner"])
            for s1, s2 in pod_b_pairs:
                t1n, t2n = s2t.get(s1, ""), s2t.get(s2, "")
                if t1n or t2n:
                    res = resolve(t1n, t2n)
                    rounds["R64"].append({"t1": t1n, "t2": t2n, "s1": s1, "s2": s2, "region": region, **res})
                    if res["winner"]: r1w_b.append(res["winner"])
            pod_a_s16 = ""
            if len(r1w_a) == 2:
                res = resolve(r1w_a[0], r1w_a[1])
                rounds["R32"].append({"t1": r1w_a[0], "t2": r1w_a[1],
                                      "s1": seed_lkp.get(r1w_a[0], 0), "s2": seed_lkp.get(r1w_a[1], 0),
                                      "region": region, **res})
                pod_a_s16 = res["winner"]
            elif r1w_a:
                pod_a_s16 = r1w_a[0]
            pod_b_s16 = ""
            if len(r1w_b) == 2:
                res = resolve(r1w_b[0], r1w_b[1])
                rounds["R32"].append({"t1": r1w_b[0], "t2": r1w_b[1],
                                      "s1": seed_lkp.get(r1w_b[0], 0), "s2": seed_lkp.get(r1w_b[1], 0),
                                      "region": region, **res})
                pod_b_s16 = res["winner"]
            elif r1w_b:
                pod_b_s16 = r1w_b[0]
            if pod_a_s16 and pod_b_s16:
                res = resolve(pod_a_s16, pod_b_s16)
                rounds["S16"].append({"t1": pod_a_s16, "t2": pod_b_s16,
                                      "s1": seed_lkp.get(pod_a_s16, 0), "s2": seed_lkp.get(pod_b_s16, 0),
                                      "region": region, **res})
                region_e8.append(res["winner"])
        if len(region_e8) == 2:
            res = resolve(region_e8[0], region_e8[1])
            rounds["E8"].append({"t1": region_e8[0], "t2": region_e8[1],
                                  "s1": seed_lkp.get(region_e8[0], 0), "s2": seed_lkp.get(region_e8[1], 0),
                                  "region": region, **res})
            ff_teams.append(res["winner"])
    champ_teams = []
    ff_pairs = [(ff_teams[0], ff_teams[3]), (ff_teams[1], ff_teams[2])] if len(ff_teams) >= 4 else []
    for t1n, t2n in ff_pairs:
        res = resolve(t1n, t2n)
        rounds["FF"].append({"t1": t1n, "t2": t2n,
                              "s1": seed_lkp.get(t1n, 0), "s2": seed_lkp.get(t2n, 0),
                              "region": "National", **res})
        champ_teams.append(res["winner"])
    if len(champ_teams) == 2:
        res = resolve(champ_teams[0], champ_teams[1])
        rounds["Championship"].append({"t1": champ_teams[0], "t2": champ_teams[1],
                                        "s1": seed_lkp.get(champ_teams[0], 0), "s2": seed_lkp.get(champ_teams[1], 0),
                                        "region": "National", **res})
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

# Module-level so all functions (including cached ones) can reference it without scope issues
_BRACKET_NORM = {
    # Short/alternate names → team_scores_2026.csv canonical names
    # (used by _smart_norm in load_data to reconcile bracket → scores)
    "BYU": "Brigham Young",
    "TCU": "Texas Christian",
    "VCU": "Virginia Commonwealth",
    "Saint Marys CA": "Saint Mary's",
    "St. Johns": "St. John's",
    "St Johns": "St. John's",
    "St Johns NY": "St. John's",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "Miami FL": "Miami",
    "Miami (FL)": "Miami",
    "SMU": "Southern Methodist",
    "Pitt": "Pittsburgh",
    "UMBC": "Maryland-Baltimore County",
    "LIU": "Long Island University",
    "St. Mary's": "Saint Mary's",
    "Saint John's": "St. John's",
    "McNeese State": "McNeese State",
    "Prairie View": "Prairie View A&M",
    # ESPN display names → team_scores_2026.csv canonical names
    # All maps target team_scores so score_lkp lookups work correctly
    "TCU Horned Frogs": "Texas Christian",
    "BYU Cougars": "Brigham Young",
    "VCU Rams": "Virginia Commonwealth",
    "SMU Mustangs": "Southern Methodist",
    "Saint Mary's Gaels": "Saint Mary's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Duke Blue Devils": "Duke",
    "Michigan Wolverines": "Michigan",
    "Michigan State Spartans": "Michigan State",
    "Ohio State Buckeyes": "Ohio State",
    "Nebraska Cornhuskers": "Nebraska",
    "Arkansas Razorbacks": "Arkansas",
    "Wisconsin Badgers": "Wisconsin",
    "Vanderbilt Commodores": "Vanderbilt",
    "Louisville Cardinals": "Louisville",
    "North Carolina Tar Heels": "North Carolina",
    "Texas Longhorns": "Texas",
    "Texas A&M Aggies": "Texas A&M",
    "Georgia Bulldogs": "Georgia",
    "Iowa State Cyclones": "Iowa State",
    "Iowa Hawkeyes": "Iowa",
    "Illinois Fighting Illini": "Illinois",
    "Houston Cougars": "Houston",
    "Florida Gators": "Florida",
    "Arizona Wildcats": "Arizona",
    "Purdue Boilermakers": "Purdue",
    "Gonzaga Bulldogs": "Gonzaga",
    "Kansas Jayhawks": "Kansas",
    "Tennessee Volunteers": "Tennessee",
    "Virginia Cavaliers": "Virginia",
    "Alabama Crimson Tide": "Alabama",
    "Kentucky Wildcats": "Kentucky",
    "Texas Tech Red Raiders": "Texas Tech",
    "Connecticut Huskies": "Connecticut",
    "UConn Huskies": "Connecticut",
    "St. John's Red Storm": "St. John's",
    "St John's Red Storm": "St. John's",
    "UCLA Bruins": "UCLA",
    "Villanova Wildcats": "Villanova",
    "Miami Hurricanes": "Miami",
    "North Carolina State Wolfpack": "North Carolina State",
    "NC State Wolfpack": "North Carolina State",
    "Siena Saints": "Siena",
    "Howard Bison": "Howard",
    "North Dakota State Bison": "North Dakota State",
    "High Point Panthers": "High Point",
    "McNeese Cowboys": "McNeese State",
    "Troy Trojans": "Troy",
    "Pennsylvania Quakers": "Pennsylvania",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "Hawaii Rainbow Warriors": "Hawaii",
    "Idaho Vandals": "Idaho",
    "Kennesaw State Owls": "Kennesaw State",
    "Saint Louis Billikens": "Saint Louis",
    "Utah State Aggies": "Utah State",
    "Santa Clara Broncos": "Santa Clara",
    "Hofstra Pride": "Hofstra",
    "Akron Zips": "Akron",
    "Wright State Raiders": "Wright State",
    "Tennessee State Tigers": "Tennessee State",
    "Furman Paladins": "Furman",
    "Long Island University Sharks": "Long Island University",
    "Queens Royals": "Queens",
    "California Baptist Lancers": "California Baptist",
    "Prairie View A&M Panthers": "Prairie View",
    "Prairie View A&M":          "Prairie View",
    # UCF → Central Florida (canonical name in team_scores_2026.csv)
    "UCF":                       "Central Florida",
    "UCF Knights":               "Central Florida",
    # Explicit anti-fuzzy-match entries — prevent short names from
    # matching unrelated teams via substring (e.g. "South Florida" ≠ "Florida")
    "South Florida Bulls": "South Florida",
    "Central Florida Knights": "Central Florida",
    "Florida State Seminoles": "Florida State",
    "Florida Atlantic Owls": "Florida Atlantic",
    "Western Kentucky Hilltoppers": "Western Kentucky",
    "Miami (OH) RedHawks": "Miami OH",
    "Miami Ohio RedHawks": "Miami OH",
    "Miami Ohio": "Miami OH",
    "Miami (Ohio)": "Miami OH",
    "Miami Redhawks": "Miami OH",
    "Miami OH Redhawks": "Miami OH",
    "Northern Iowa Panthers": "Northern Iowa",
}

# Seed → estimated contender score for teams NOT in bracket_analysis_2026.csv.
# Calibrated to the score distribution of teams that ARE in the bracket.
_SEED_SCORE_FALLBACK = {
    1: 79, 2: 74, 3: 71, 4: 68, 5: 65,
    6: 63, 7: 61, 8: 58, 9: 56, 10: 54,
    11: 52, 12: 49, 13: 46, 14: 43, 15: 38, 16: 30,
}

@st.cache_data(ttl=60)  # re-read files every 60 s so pipeline updates show immediately
def load_data():
    scores  = pd.read_csv(SCORES_PATH)  if os.path.exists(SCORES_PATH)  else pd.DataFrame()
    champs  = pd.read_csv(CHAMPS_PATH)  if os.path.exists(CHAMPS_PATH)  else pd.DataFrame()
    bracket = pd.read_csv(BRACKET_PATH) if os.path.exists(BRACKET_PATH) else pd.DataFrame()

    # ── Normalize bracket team names to match team_scores canonical names ──
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
    # Add bracket teams not in team_scores (e.g. Miami OH, play-in winners)
    # so they appear in in_bracket with seed-based score estimates
    _already = set(scores["team"].tolist())
    _missing_rows = []
    for _, _br in bracket_info.iterrows():
        if _br["team"] not in _already:
            _seed_est = _SEED_SCORE_FALLBACK.get(int(_br["seed"]) if pd.notna(_br["seed"]) else 0, 50)
            _missing_rows.append({"team": _br["team"], "region": _br["region"],
                                   "seed": _br["seed"], "contender_score": _seed_est})
    if _missing_rows:
        scores = pd.concat([scores, pd.DataFrame(_missing_rows)], ignore_index=True)
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

# ── Update championship odds based on actual results ────────────────────────
# Teams that have been eliminated get 0% championship probability.
# Remaining teams' probabilities are renormalized.
# Preserve original pre-tournament odds for comparison before zeroing eliminated teams.
champs_original = champs.copy() if len(champs) > 0 else pd.DataFrame()
_actual_results_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "outputs", "game_results_2026.csv"
)
if len(champs) > 0 and "championship_pct" in champs.columns and os.path.exists(_actual_results_csv):
    try:
        _results_live = pd.read_csv(_actual_results_csv)
        if "loser" in _results_live.columns and not _results_live.empty:
            _eliminated = set(_results_live["loser"].dropna().str.strip().tolist())
            _mask = champs["team"].isin(_eliminated)
            if _mask.any():
                champs = champs.copy()
                champs.loc[_mask, "championship_pct"] = 0.0
                _total = champs["championship_pct"].sum()
                if _total > 0:
                    champs["championship_pct"] = champs["championship_pct"] / _total * 100
    except Exception:
        pass

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
if "predict_t1" not in st.session_state:
    st.session_state.predict_t1 = None
if "predict_t2" not in st.session_state:
    st.session_state.predict_t2 = None

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

if st.session_state.predict_t1 and st.session_state.predict_t2:
    st.info(f"🎯 **{st.session_state.predict_t1}** vs **{st.session_state.predict_t2}** — click the **🎯 Score Predictor** tab to see predicted final score.", icon="👆")
    if st.button("✖ Clear prediction", key="clear_predict"):
        st.session_state.predict_t1 = None
        st.session_state.predict_t2 = None
        st.rerun()

# ── Tossup Lab helpers ────────────────────────────────────────────────────────

_TOSSUP_METRICS = [
    ("Turnover Margin",  "to_margin",  True,   3),   # higher = better, weight 3
    ("3-Point Defense",  "opp_3pt",    False,  2),   # lower = better, weight 2
    ("Off. Rebounding",  "orb",        True,   2),   # higher = better
    ("Free Throw Rate",  "ftr",        True,   1),   # higher = better
    ("Tempo Control",    "tempo",      None,   1),   # gap matters, not direction
    ("Efficiency Margin","eff_margin", True,   2),   # higher = better
]

def compute_tossup_scorecard(t1_name, t2_name, bkt_df):
    """Compute the 6-metric tossup scorecard for two teams."""
    r1 = bkt_df[bkt_df["team"] == t1_name]
    r2 = bkt_df[bkt_df["team"] == t2_name]
    if len(r1) == 0 or len(r2) == 0:
        return None
    r1, r2 = r1.iloc[0], r2.iloc[0]
    # Compute raw values
    t1_vals = {
        "to_margin":  safe_f(r1.get("opp_turnover_pct")) - safe_f(r1.get("turnover_pct")),
        "opp_3pt":    safe_f(r1.get("opp_three_pt_pct")),
        "orb":        safe_f(r1.get("off_rebound_pct")),
        "ftr":        safe_f(r1.get("ft_rate")),
        "tempo":      safe_f(r1.get("tempo")),
        "eff_margin": safe_f(r1.get("adj_offense")) - safe_f(r1.get("adj_defense")),
    }
    t2_vals = {
        "to_margin":  safe_f(r2.get("opp_turnover_pct")) - safe_f(r2.get("turnover_pct")),
        "opp_3pt":    safe_f(r2.get("opp_three_pt_pct")),
        "orb":        safe_f(r2.get("off_rebound_pct")),
        "ftr":        safe_f(r2.get("ft_rate")),
        "tempo":      safe_f(r2.get("tempo")),
        "eff_margin": safe_f(r2.get("adj_offense")) - safe_f(r2.get("adj_defense")),
    }
    metrics = []
    t1_adv = 0; t2_adv = 0; t1_wt = 0; t2_wt = 0
    for name, key, higher_better, weight in _TOSSUP_METRICS:
        v1, v2 = t1_vals[key], t2_vals[key]
        if key == "tempo":
            # Tempo: the team closer to their preferred pace has the edge in a neutral game
            # — just show the gap as context
            edge_team = ""
            metrics.append({"name": name, "t1_val": v1, "t2_val": v2, "edge": abs(v1 - v2),
                           "edge_team": edge_team, "weight": weight, "note": f"Gap: {abs(v1-v2):.1f}"})
            continue
        if higher_better:
            edge_team = t1_name if v1 > v2 else (t2_name if v2 > v1 else "")
        else:
            edge_team = t1_name if v1 < v2 else (t2_name if v2 < v1 else "")
        if edge_team == t1_name:
            t1_adv += 1; t1_wt += weight
        elif edge_team == t2_name:
            t2_adv += 1; t2_wt += weight
        metrics.append({"name": name, "t1_val": v1, "t2_val": v2, "edge": abs(v1 - v2),
                       "edge_team": edge_team, "weight": weight})
    # Coach experience bonus
    c1 = safe_i(r1.get("coach_ncaa_games")); c2 = safe_i(r2.get("coach_ncaa_games"))
    coach_edge = t1_name if c1 > c2 + 5 else (t2_name if c2 > c1 + 5 else "")
    return {"metrics": metrics, "t1_adv": t1_adv, "t2_adv": t2_adv,
            "t1_wt": t1_wt, "t2_wt": t2_wt,
            "coach_edge": coach_edge, "t1_coach": c1, "t2_coach": c2,
            "t1_clutch": safe_f(r1.get("clutch_score")), "t2_clutch": safe_f(r2.get("clutch_score")),
            "t1_last10": safe_f(r1.get("last10_win_pct")), "t2_last10": safe_f(r2.get("last10_win_pct"))}

def generate_statlas_lean(t1, t2, scorecard, model_pick, model_prob, bkt_df):
    """Synthesize model + scorecard into a tossup recommendation."""
    if scorecard is None:
        return model_pick, "Insufficient data for scorecard — defaulting to model pick."
    sc_leader = t1 if scorecard["t1_wt"] > scorecard["t2_wt"] else (
                t2 if scorecard["t2_wt"] > scorecard["t1_wt"] else "")
    model_agrees = (model_pick == sc_leader) if sc_leader else False
    # Check for Dangerous flag
    r1 = bkt_df[bkt_df["team"] == t1]
    r2 = bkt_df[bkt_df["team"] == t2]
    t1_dangerous = bool(r1.iloc[0].get("dangerous_low_seed_flag")) if len(r1) else False
    t2_dangerous = bool(r2.iloc[0].get("dangerous_low_seed_flag")) if len(r2) else False
    # Strong scorecard (5-0 or better weighted)
    wt_gap = abs(scorecard["t1_wt"] - scorecard["t2_wt"])
    if wt_gap >= 6:
        conf = "HIGH"
        reason = (f"Scorecard strongly favors {sc_leader} ({scorecard['t1_adv'] if sc_leader==t1 else scorecard['t2_adv']}"
                  f"-{scorecard['t2_adv'] if sc_leader==t1 else scorecard['t1_adv']} on key metrics).")
    elif model_agrees and sc_leader:
        conf = "MODERATE"
        reason = f"Model ({model_prob*100:.0f}%) and scorecard both favor {sc_leader}."
    elif sc_leader and not model_agrees:
        conf = "SPLIT"
        reason = f"Model picks {model_pick} ({model_prob*100:.0f}%) but scorecard leans {sc_leader} — watch the film."
        # When signals disagree, flag it as a coin flip
        lean_team = sc_leader if wt_gap >= 3 else ""
        if not lean_team:
            return "COIN FLIP", reason
        return lean_team, reason
    else:
        return "COIN FLIP", "Model and metrics are dead even — this is a true tossup."
    # Dangerous flag warning
    underdog = t2 if model_pick == t1 else t1
    if (underdog == t1 and t1_dangerous) or (underdog == t2 and t2_dangerous):
        reason += f" ⚠️ {underdog} carries the Dangerous flag — upset risk elevated."
    lean_team = sc_leader if sc_leader else model_pick
    return lean_team, f"[{conf}] {reason}"

def _tossup_metric_bar_html(name, t1_val, t2_val, t1_name, t2_name, higher_better=True):
    """Horizontal comparison bar for a tossup metric."""
    if higher_better is None:
        # Tempo — just show values, no winner coloring
        return (f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e293b">'
                f'<span style="color:#94a3b8;width:140px">{name}</span>'
                f'<span style="color:#e2e8f0">{t1_val:.1f}</span>'
                f'<span style="color:#64748b;font-size:0.8rem">vs</span>'
                f'<span style="color:#e2e8f0">{t2_val:.1f}</span>'
                f'<span style="color:#64748b;width:60px;text-align:right">Gap {abs(t1_val-t2_val):.1f}</span>'
                f'</div>')
    if higher_better:
        t1_wins = t1_val > t2_val; t2_wins = t2_val > t1_val
    else:
        t1_wins = t1_val < t2_val; t2_wins = t2_val < t1_val
    c1 = "#4ade80" if t1_wins else ("#ef4444" if t2_wins else "#94a3b8")
    c2 = "#4ade80" if t2_wins else ("#ef4444" if t1_wins else "#94a3b8")
    edge_name = t1_name if t1_wins else (t2_name if t2_wins else "Even")
    return (f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e293b">'
            f'<span style="color:#94a3b8;width:140px">{name}</span>'
            f'<span style="color:{c1};font-weight:600">{t1_val:.1f}</span>'
            f'<span style="color:#64748b;font-size:0.8rem">vs</span>'
            f'<span style="color:{c2};font-weight:600">{t2_val:.1f}</span>'
            f'<span style="color:{c1 if t1_wins else c2};width:60px;text-align:right;font-size:0.8rem">'
            f'{"✓ " + edge_name if (t1_wins or t2_wins) else "—"}</span>'
            f'</div>')

# ── Score Predictor helpers ────────────────────────────────────────────────────

def predict_final_score(t1_name, t2_name, bkt_df, tournament_factor=0.975):
    """Predict final score: model picks the winner, regression sets the total.

    The proven contender_score model (81% winner accuracy) determines WHO wins
    and by how much (spread direction & magnitude). The calibrated regression
    estimates TOTAL points. Final scores allocate the regression total using
    the model-derived spread.

    This ensures the Score Predictor never contradicts the model's proven picks.
    """
    # --- Regression coefficients for TOTAL estimation ---
    # Per-team: score = PPG_COEF * ppg + PAG_COEF * opp_pag + DEF_COEF * opp_adj_def + INTERCEPT
    # Trained on team_scores.csv (rescaled 0-100 efficiency values)
    PPG_COEF = 1.504622
    PAG_COEF = 0.814834
    DEF_COEF = -0.083819
    TEAM_INTERCEPT = -98.014121
    # Total calibration: cal_total = TOT_COEF * raw_total + GAP_COEF * margin_gap + TOT_INTERCEPT
    TOT_COEF = 0.866839
    GAP_COEF = 0.655479
    TOT_INTERCEPT = 12.821787

    r1 = bkt_df[bkt_df["team"] == t1_name]
    r2 = bkt_df[bkt_df["team"] == t2_name]
    if len(r1) == 0 or len(r2) == 0:
        return None
    r1, r2 = r1.iloc[0], r2.iloc[0]

    ppg1 = safe_f(r1.get("points_per_game"), 75)
    ppg2 = safe_f(r2.get("points_per_game"), 75)
    pag1 = safe_f(r1.get("points_allowed_per_game"), 70)
    pag2 = safe_f(r2.get("points_allowed_per_game"), 70)
    off1 = safe_f(r1.get("adj_offense"), 0)
    def1 = safe_f(r1.get("adj_defense"), 0)
    off2 = safe_f(r2.get("adj_offense"), 0)
    def2 = safe_f(r2.get("adj_defense"), 0)
    tempo1 = safe_f(r1.get("tempo"), 0)
    tempo2 = safe_f(r2.get("tempo"), 0)
    cs1 = safe_f(r1.get("contender_score", 50))
    cs2 = safe_f(r2.get("contender_score", 50))

    has_efficiency = off1 > 0 and def1 > 0 and off2 > 0 and def2 > 0
    has_tempo = tempo1 > 0 and tempo2 > 0

    # --- Winner determination: proven contender_score model (81%) ---
    # The model accounts for SOS, conference quality, seeding, pedigree —
    # far more reliable than raw PPG for determining who wins.
    model_wp = win_prob_sigmoid(cs1, cs2)
    # Convert model win probability to implied point spread
    # k=0.15 is basketball-calibrated: 7pt ≈ 74%, 10pt ≈ 82%, 14pt ≈ 89%
    if 0.001 < model_wp < 0.999:
        model_spread = np.log(model_wp / (1 - model_wp)) / 0.15
    else:
        model_spread = 25.0 if model_wp >= 0.999 else -25.0
    model_spread = max(-25.0, min(25.0, model_spread))

    # --- Tossup tiebreaker: enforce minimum spread, never contradict model ---
    # When the model has ANY lean, respect its direction — the bracket already
    # shows that pick and the model is proven at 81%. Tossup only picks the
    # winner when contender_scores are exactly equal (true 50/50).
    if abs(model_spread) < 1.0:
        if model_spread == 0.0:
            # True tie (identical contender_scores): use tossup to pick winner
            tossup = compute_tossup_scorecard(t1_name, t2_name, bkt_df)
            if tossup:
                wt_diff = tossup["t1_wt"] - tossup["t2_wt"]
                lean = wt_diff
                if lean == 0:
                    last10_diff = tossup["t1_last10"] - tossup["t2_last10"]
                    clutch_diff = tossup["t1_clutch"] - tossup["t2_clutch"]
                    lean = (last10_diff * 10) + clutch_diff
                if lean == 0:
                    if tossup["coach_edge"] == t1_name:
                        lean = 1
                    elif tossup["coach_edge"] == t2_name:
                        lean = -1
                if lean == 0:
                    seed1 = safe_f(r1.get("seed", 16))
                    seed2 = safe_f(r2.get("seed", 16))
                    lean = 1 if seed1 < seed2 else -1
                model_spread = 1.0 if lean > 0 else -1.0
            else:
                model_spread = 1.0  # fallback
        else:
            # Model has a slight lean: keep its direction, just enforce 1-pt min
            if model_spread > 0:
                model_spread = max(1.0, model_spread)
            else:
                model_spread = min(-1.0, model_spread)

    # --- Total estimation: calibrated regression ---
    if has_efficiency:
        method = "regression_calibrated"
        # Per-team regression with opponent defense quality
        raw_t1 = PPG_COEF * ppg1 + PAG_COEF * pag2 + DEF_COEF * def2 + TEAM_INTERCEPT
        raw_t2 = PPG_COEF * ppg2 + PAG_COEF * pag1 + DEF_COEF * def1 + TEAM_INTERCEPT

        # Margin gap: quality differential between teams (drives blowout bonus)
        adj_margin1 = off1 - def1
        adj_margin2 = off2 - def2
        margin_gap = abs(adj_margin1 - adj_margin2)
    else:
        method = "ppg_average"
        # Fallback: simple PPG average when efficiency data unavailable
        raw_t1 = (ppg1 + pag2) / 2
        raw_t2 = (ppg2 + pag1) / 2
        margin_gap = abs(ppg1 - pag1 - (ppg2 - pag2))

    raw_total = raw_t1 + raw_t2
    if has_efficiency:
        # Calibrate total: corrects bias and adds blowout bonus
        cal_total = TOT_COEF * raw_total + GAP_COEF * margin_gap + TOT_INTERCEPT
    else:
        cal_total = raw_total  # No calibration without full data

    # --- Allocate calibrated total using MODEL spread (not regression) ---
    # This ensures the predicted winner always matches the model's pick.
    t1_sc = (cal_total + model_spread) / 2
    t2_sc = (cal_total - model_spread) / 2

    # Ensure non-negative scores
    t1_sc = max(t1_sc, 40)
    t2_sc = max(t2_sc, 40)

    # Convert to integers immediately to avoid double-rounding mismatches.
    # (round(74.5419, 1) → 74.5, then :.0f banker-rounds to "74" — causes phantom ties)
    t1_int = int(t1_sc + 0.5)  # always rounds .5 UP (no banker's rounding)
    t2_int = int(t2_sc + 0.5)

    # Anti-tie: tournament games can't end tied
    if t1_int == t2_int:
        # Give the winner 1 extra point (total shifts by 1 — negligible)
        if model_spread >= 0:
            t1_int += 1
        else:
            t2_int += 1

    possessions = (tempo1 + tempo2) / 2 if has_tempo else 67.5
    margin = abs(t1_int - t2_int)
    confidence_range = 6 if margin > 15 else (8 if margin > 8 else 10)

    return {
        "t1_score": t1_int, "t2_score": t2_int,
        "possessions": round(possessions, 1),
        "spread": t1_int - t2_int, "total": t1_int + t2_int,
        "t1_off": off1, "t1_def": def1, "t2_off": off2, "t2_def": def2,
        "t1_tempo": tempo1, "t2_tempo": tempo2,
        "confidence_range": confidence_range, "method": method,
    }


def compute_prediction_accuracy(bkt_df, results_df):
    """Compare predicted scores against actual game results for calibration."""
    if results_df is None or results_df.empty:
        return []
    calibration = []
    for _, row in results_df.iterrows():
        t1 = str(row.get("t1", "") or "").strip()
        t2 = str(row.get("t2", "") or "").strip()
        actual_t1 = safe_i(row.get("t1_score"))
        actual_t2 = safe_i(row.get("t2_score"))
        if not t1 or not t2 or actual_t1 == 0 or actual_t2 == 0:
            continue
        pred = predict_final_score(t1, t2, bkt_df)
        if pred is None:
            continue
        pred_winner = t1 if pred["t1_score"] > pred["t2_score"] else t2
        actual_winner = t1 if actual_t1 > actual_t2 else t2
        calibration.append({
            "t1": t1, "t2": t2,
            "pred_t1": pred["t1_score"], "pred_t2": pred["t2_score"],
            "actual_t1": actual_t1, "actual_t2": actual_t2,
            "pred_spread": pred["spread"],
            "actual_spread": actual_t1 - actual_t2,
            "spread_error": abs(pred["spread"] - (actual_t1 - actual_t2)),
            "total_error": abs(pred["total"] - (actual_t1 + actual_t2)),
            "pred_total": pred["total"], "actual_total": actual_t1 + actual_t2,
            "winner_correct": pred_winner == actual_winner,
            "round": str(row.get("round", "")),
        })
    return calibration


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏅 Rankings", "🎯 Picks", "🔍 Team Deep Dive",
    "🎲 Championship Odds", "🏆 Bracket", "📺 Live",
    "📈 Model Accuracy", "📋 Recap", "🔬 Tossup Lab", "🎯 Score Predictor"
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
# BRACKET DATA FOR CHAMPIONSHIP ODDS TAB
# Load recap from CSV (no ESPN fetch needed here — tab5 handles that)
# ─────────────────────────────────────────────────────────────────────────────
_tab4_recap = pd.DataFrame()
_tab4_adapted = {}
_tab4_live_rounds = {}
if len(in_bracket) > 0 and "region" in in_bracket.columns:
    try:
        if os.path.exists(_RESULTS_CSV):
            _tab4_recap = pd.read_csv(_RESULTS_CSV)
    except Exception:
        pass
    _tab4_adapted = _compute_adapted_scores(in_bracket, _tab4_recap)
    _tab4_live_rounds = build_bracket_live(in_bracket, _tab4_recap, _tab4_adapted)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CHAMPIONSHIP ODDS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    # ── Live Updated Odds — Full Bracket Resimulation ──────────────────────
    _live_champ_probs = {}
    _r32_games_completed = 0
    _r32_games_total = 0
    if _tab4_live_rounds and "R32" in _tab4_live_rounds and len(_tab4_live_rounds["R32"]) > 0:
        _r32_list = _tab4_live_rounds["R32"]
        _r32_games_total = len(_r32_list)
        _r32_tuples = []
        for m in _r32_list:
            t1 = m.get("t1", "")
            t2 = m.get("t2", "")
            done = m.get("completed", False)
            aw = m.get("winner", "") if done else ""
            _r32_tuples.append((t1, t2, done, aw))
            if done:
                _r32_games_completed += 1
        _r32_frozen = tuple(_r32_tuples)
        _score_frozen = tuple(sorted(_tab4_adapted.items()))
        if _tab4_adapted and len(_r32_tuples) == 16:
            _live_champ_probs = run_full_bracket_monte_carlo(_r32_frozen, _score_frozen, n=500_000)

    # Build original odds lookup from pre-tournament simulation
    _orig_odds_lkp = {}
    if len(champs_original) > 0 and "championship_pct" in champs_original.columns:
        for _, _r in champs_original.iterrows():
            _orig_odds_lkp[_r["team"]] = float(_r.get("championship_pct", 0))

    # ── DISPLAY: Live Updated Odds (shown first if available) ────────────
    if _live_champ_probs and any(v > 0 for v in _live_champ_probs.values()):
        _total_games = len(_tab4_recap) if not isinstance(_tab4_recap, type(None)) and hasattr(_tab4_recap, '__len__') else 0
        st.markdown(f'<div class="section-header">🔄 Live Updated Championship Odds — 500,000 Simulations</div>', unsafe_allow_html=True)
        st.caption(f"Re-simulated with adapted scores from {_total_games} completed games · {_r32_games_completed}/{_r32_games_total} R32 games locked")

        # Build display DataFrame
        _live_rows = []
        for team, pct in sorted(_live_champ_probs.items(), key=lambda x: x[1], reverse=True):
            if pct > 0.05:
                _live_rows.append({"team": team, "championship_pct": pct})
        _live_disp = pd.DataFrame(_live_rows)
        if len(bracket) > 0:
            _live_disp = _live_disp.merge(bracket[["team", "region", "seed"]], on="team", how="left")
        _live_disp = _live_disp.head(20)

        # Top 3 podium
        _lp1, _lp2, _lp3 = st.columns(3)
        for i, (_, r) in enumerate(_live_disp.head(3).iterrows()):
            seed_disp = f"#{int(r['seed'])} seed · {r['region']}" if pd.notna(r.get('seed')) else ""
            orig_pct = _orig_odds_lkp.get(r["team"], 0)
            delta = r["championship_pct"] - orig_pct
            if abs(delta) >= 0.1:
                delta_str = f'<span style="color:{("#4ade80" if delta > 0 else "#f87171")};font-size:0.85rem">{"↑" if delta > 0 else "↓"} {abs(delta):+.1f}%</span>'
            else:
                delta_str = '<span style="color:#6b7280;font-size:0.85rem">—</span>'
            border_col = '#f59e0b' if i == 0 else '#94a3b8' if i == 1 else '#cd7f32'
            with [_lp1, _lp2, _lp3][i]:
                st.markdown(f"""
                <div class="team-card" style="text-align:center;border-color:{border_col};border-width:2px">
                    <div style="font-size:2rem">{"🥇🥈🥉"[i]}</div>
                    <div class="team-name" style="font-size:1.3rem">{r['team']}</div>
                    <div style="color:#f97316;font-size:1.7rem;font-weight:800">{r['championship_pct']:.1f}%</div>
                    <div style="color:#94a3b8;font-size:0.8rem">{seed_disp}</div>
                    <div>{delta_str}</div>
                    <div style="color:#b0bbd0;font-size:0.78rem">Model line: {american_line(r['championship_pct']/100)}</div>
                </div>
                """, unsafe_allow_html=True)

        # Odds movement table
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Odds Movement — Pre-Tournament vs Live</div>', unsafe_allow_html=True)
        _move_rows = []
        for _, r in _live_disp.iterrows():
            team = r["team"]
            updated_pct = r["championship_pct"]
            orig_pct = _orig_odds_lkp.get(team, 0)
            delta = updated_pct - orig_pct
            seed_v = int(r["seed"]) if pd.notna(r.get("seed")) else ""
            if abs(delta) >= 0.1:
                if delta > 0:
                    delta_html = f'<span style="color:#4ade80;font-weight:{"800" if abs(delta)>2 else "600"}">↑ +{delta:.1f}%</span>'
                else:
                    delta_html = f'<span style="color:#f87171;font-weight:{"800" if abs(delta)>2 else "600"}">↓ {delta:.1f}%</span>'
            else:
                delta_html = '<span style="color:#6b7280">—</span>'
            bold = "font-weight:800;" if abs(delta) > 2 else ""
            _move_rows.append(f"""<tr style="{bold}">
                <td style="padding:6px 12px;border-bottom:1px solid #374151">{team}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #374151;text-align:center">{seed_v}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #374151;text-align:center;color:#94a3b8">{orig_pct:.1f}%</td>
                <td style="padding:6px 12px;border-bottom:1px solid #374151;text-align:center;color:#f97316">{updated_pct:.1f}%</td>
                <td style="padding:6px 12px;border-bottom:1px solid #374151;text-align:center">{delta_html}</td>
            </tr>""")
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;background:#1a1f2e;border-radius:8px;overflow:hidden">
            <thead>
                <tr style="background:#0f1318">
                    <th style="padding:8px 12px;text-align:left;color:#f1f5f9;border-bottom:2px solid #374151">Team</th>
                    <th style="padding:8px 12px;text-align:center;color:#f1f5f9;border-bottom:2px solid #374151">Seed</th>
                    <th style="padding:8px 12px;text-align:center;color:#94a3b8;border-bottom:2px solid #374151">Original %</th>
                    <th style="padding:8px 12px;text-align:center;color:#f97316;border-bottom:2px solid #374151">Updated %</th>
                    <th style="padding:8px 12px;text-align:center;color:#f1f5f9;border-bottom:2px solid #374151">Change</th>
                </tr>
            </thead>
            <tbody>
                {"".join(_move_rows)}
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        # Updated bar chart
        st.markdown("---")
        fig_live, ax_live = plt.subplots(figsize=(12, 5))
        fig_live.patch.set_facecolor("#0e1117"); ax_live.set_facecolor("#1a1f2e")
        teams_l = _live_disp["team"].tolist()
        probs_l = _live_disp["championship_pct"].tolist()
        colors_l = ["#f97316" if i == 0 else "#3b82f6" if i < 4 else "#4b5563" for i in range(len(teams_l))]
        bars_l = ax_live.barh(teams_l[::-1], probs_l[::-1], color=colors_l[::-1], edgecolor="none", height=0.65)
        for bar, pct in zip(bars_l, probs_l[::-1]):
            ax_live.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", color="#ffffff", fontsize=8.5, fontweight="bold")
        ax_live.set_xlabel("Championship Probability (%)", color="#b0bbd0", fontsize=9)
        ax_live.tick_params(colors="#f1f5f9", labelsize=9)
        for spine in ["top", "right"]: ax_live.spines[spine].set_visible(False)
        ax_live.spines["bottom"].set_color("#374151"); ax_live.spines["left"].set_color("#374151")
        ax_live.set_title("Live Updated Championship Probabilities (500,000 runs)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
        st.pyplot(fig_live)
        plt.close(fig_live)

        st.markdown("---")

    # ── DISPLAY: Original Pre-Tournament Odds ────────────────────────────
    st.markdown('<div class="section-header">📊 Pre-Tournament Championship Odds (Selection Sunday)</div>', unsafe_allow_html=True)

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
        ax3.set_title("Pre-Tournament Championship Probabilities (500,000 runs)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
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
# TAB 5 — INTERACTIVE BRACKET
# ─────────────────────────────────────────────────────────────────────────────

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?groups=100&limit=50"
)
ESPN_PLAYBYPLAY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary?event={event_id}"
)
# NCAA.com official scoreboard (authoritative, real-time)
_NCAA_SCOREBOARD = (
    "https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/"
    "{year}/{month}/{day}/scoreboard.json"
)


_TOURNEY_DATES = [
    "20260317","20260318",
    "20260319","20260320","20260321","20260322","20260323",
    "20260324","20260325","20260327","20260328","20260329",
    "20260330","20260404","20260405","20260407",
]


def fetch_game_box_score(event_id):
    """Extract actual game stats from ESPN summary endpoint the moment a game ends.
    Returns dict with per-team FG%, rebounds, turnovers, assists, half scores, etc.
    Keys: t1_name, t1_fg_pct, t1_rebounds, t1_turnovers, t1_h1, t1_h2, ... (t2_ same)
    All values are raw strings as ESPN returns them — caller converts to float as needed.
    Uses session_state cache (not st.cache_data) so empty/error results are NOT cached."""
    # Cache successful results in session_state so we don't re-fetch per rerun
    _bs_cache_key = f"_box_score_{event_id}"
    if _bs_cache_key in st.session_state:
        cached = st.session_state[_bs_cache_key]
        if cached:  # only return if non-empty (retry empty results)
            return cached
    if not _REQUESTS_OK or not event_id:
        return {}
    try:
        url = ESPN_PLAYBYPLAY.format(event_id=event_id)
        resp = _requests.get(url, timeout=8, headers={"User-Agent": "statlasberg/1.0"})
        resp.raise_for_status()
        data = resp.json()

        boxscore = data.get("boxscore", {})
        teams = boxscore.get("teams", [])
        if len(teams) < 2:
            return {}

        result = {}
        for i, td in enumerate(teams[:2]):
            pfx = f"t{i+1}_"
            result[f"{pfx}name"] = td.get("team", {}).get("displayName", "")
            stats = {s.get("name", ""): s.get("displayValue", "")
                     for s in td.get("statistics", [])}
            for espn_key, out_key in [
                ("fieldGoalPct",           "fg_pct"),
                ("threePointFieldGoalPct", "fg3_pct"),
                ("freeThrowPct",           "ft_pct"),
                ("totalRebounds",          "rebounds"),
                ("offensiveRebounds",      "off_reb"),
                ("turnovers",              "turnovers"),
                ("assists",                "assists"),
                ("steals",                 "steals"),
                ("blocks",                 "blocks"),
                ("fieldGoalsMade-fieldGoalsAttempted", "fg_made_att"),
            ]:
                result[f"{pfx}{out_key}"] = stats.get(espn_key, "")

        # Line scores (points per half/OT) from header competitions
        header = data.get("header", {})
        comps_list = header.get("competitions", [])
        if comps_list:
            competitors = comps_list[0].get("competitors", [])
            for i, comp in enumerate(competitors[:2]):
                pfx = f"t{i+1}_"
                ls = comp.get("linescores", [])
                result[f"{pfx}h1"] = ls[0].get("displayValue", "") if len(ls) > 0 else ""
                result[f"{pfx}h2"] = ls[1].get("displayValue", "") if len(ls) > 1 else ""

        # Cache only non-empty results
        if result:
            st.session_state[_bs_cache_key] = result
        return result
    except Exception as _bs_err:
        print(f"[Statlasberg] Box score fetch failed for event {event_id}: {_bs_err}")
        return {}


@st.cache_data(ttl=180)
def fetch_all_tournament_games(bracket_teams):
    """Fetch all completed 2026 tournament games from ESPN, querying every
    tournament date up to today. Returns dict keyed by event_id."""
    if not _REQUESTS_OK:
        return {}
    today_str = datetime.now().strftime("%Y%m%d")
    seen = {}

    def _norm(raw_name, bt_set):
        # 1. Explicit _BRACKET_NORM mapping always wins — skip bt_set check and fuzzy
        #    so "Miami (OH) RedHawks" → "Miami OH" even if bt_set has stale entries
        n = _BRACKET_NORM.get(raw_name)
        if n is not None:
            return n
        # 2. Exact match in bracket set
        if raw_name in bt_set:
            return raw_name
        # 3. Fuzzy substring match — prefer LONGEST matching bracket name to avoid
        #    "Michigan" matching "Michigan State Spartans" before "Michigan State"
        matches = [bt for bt in bt_set if bt.lower() in raw_name.lower() or raw_name.lower() in bt.lower()]
        if matches:
            return max(matches, key=len)  # longest = most specific
        return raw_name

    for d in _TOURNEY_DATES:
        if d > today_str:
            break
        # ── Network fetch: separate try so HTTP errors don't kill game parsing ──
        try:
            url = ESPN_SCOREBOARD + f"&dates={d}"
            resp = _requests.get(url, timeout=6, headers={"User-Agent": "statlasberg/1.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as _net_err:
            print(f"[Statlasberg] ESPN fetch failed for {d}: {_net_err}")
            continue
        # ── Parse each event individually so one bad game doesn't skip the date ──
        for event in data.get("events", []):
            try:
                eid = event.get("id", "")
                if eid in seen:
                    continue
                comp = event.get("competitions", [{}])[0]
                comps = comp.get("competitors", [])
                if len(comps) < 2:
                    continue
                status_type = event.get("status", {}).get("type", {})
                if not status_type.get("completed", False):
                    continue
                headline = comp.get("notes", [{}])[0].get("headline", "")
                if "NIT" in headline or "CBI" in headline or "CIT" in headline:
                    continue
                t1c = comps[0]; t2c = comps[1]
                t1_name = _norm(t1c.get("team", {}).get("displayName", ""), bracket_teams)
                t2_name = _norm(t2c.get("team", {}).get("displayName", ""), bracket_teams)
                try:
                    t1_score = int(t1c.get("score", 0))
                    t2_score = int(t2c.get("score", 0))
                except Exception:
                    t1_score = t2_score = 0
                # Extract seed from ESPN curatedRank (used as fallback for teams not in bracket)
                t1_espn_seed = t1c.get("curatedRank", {}).get("current", 0) or 0
                t2_espn_seed = t2c.get("curatedRank", {}).get("current", 0) or 0
                winner = t1_name if t1_score >= t2_score else t2_name
                loser  = t2_name if t1_score >= t2_score else t1_name
                box = fetch_game_box_score(eid)
                # ESPN summary may return teams in different order than scoreboard.
                # Use stored t1_name/t2_name to detect and swap if needed.
                box_t1 = _norm(box.get("t1_name", ""), bracket_teams)
                box_t2 = _norm(box.get("t2_name", ""), bracket_teams)
                if box_t1 and box_t2 and box_t1 == t2_name and box_t2 == t1_name:
                    swapped = {}
                    for k, v in box.items():
                        if k.startswith("t1_"):   swapped["t2_" + k[3:]] = v
                        elif k.startswith("t2_"): swapped["t1_" + k[3:]] = v
                        else:                     swapped[k] = v
                    box = swapped
                seen[eid] = {
                    "event_id": eid, "date": d,
                    "t1": t1_name, "t2": t2_name,
                    "winner": winner, "loser": loser,
                    "t1_score": t1_score, "t2_score": t2_score,
                    "t1_espn_seed": t1_espn_seed, "t2_espn_seed": t2_espn_seed,
                    "headline": headline,
                    **box,  # attach all box score fields (fg_pct, rebounds, etc.)
                }
            except Exception as _evt_err:
                print(f"[Statlasberg] Error parsing event {event.get('id','?')} on {d}: {_evt_err}")
                continue
    return seen


def _safe_pct(val_str):
    """Convert ESPN stat string like '52.3' or '52.3%' to float, or None."""
    try:
        return float(str(val_str).replace("%", "").strip())
    except Exception:
        return None


def game_narrative(row, bkt_df):
    """Introspective, fact-based recap — what the game stats reveal about basketball
    and what the model should take away. Margin-calibrated: coin-flips get humility,
    decisive games get real analysis. No causal claims, no hallucination."""
    winner  = str(row.get("winner", ""))
    loser   = str(row.get("loser", ""))
    correct = str(row.get("correct", "False")).strip().lower() in ("true", "1", "yes")
    conf    = float(row.get("model_conf", 0.5)) * 100
    w_seed  = int(float(row.get("winner_seed", 0) or 0))
    l_seed  = int(float(row.get("loser_seed",  0) or 0))
    w_flags = str(row.get("winner_flags", ""))
    l_flags = str(row.get("loser_flags",  ""))
    rnd     = str(row.get("round", ""))
    t1      = str(row.get("t1", ""))
    t1_sc   = int(float(row.get("t1_score", 0) or 0))
    t2_sc   = int(float(row.get("t2_score", 0) or 0))
    w_sc    = t1_sc if winner == t1 else t2_sc
    l_sc    = t2_sc if winner == t1 else t1_sc
    margin  = abs(w_sc - l_sc)

    t1_is_winner = (winner == t1)
    w_pfx = "t1_" if t1_is_winner else "t2_"
    l_pfx = "t2_" if t1_is_winner else "t1_"

    # Real game stats
    w_fg  = _safe_pct(row.get(f"{w_pfx}fg_pct"))
    l_fg  = _safe_pct(row.get(f"{l_pfx}fg_pct"))
    w_fg3 = _safe_pct(row.get(f"{w_pfx}fg3_pct"))
    l_fg3 = _safe_pct(row.get(f"{l_pfx}fg3_pct"))
    w_reb = _safe_pct(row.get(f"{w_pfx}rebounds"))
    l_reb = _safe_pct(row.get(f"{l_pfx}rebounds"))
    w_to  = _safe_pct(row.get(f"{w_pfx}turnovers"))
    l_to  = _safe_pct(row.get(f"{l_pfx}turnovers"))
    w_ast = _safe_pct(row.get(f"{w_pfx}assists"))
    l_ast = _safe_pct(row.get(f"{l_pfx}assists"))
    w_h1  = row.get(f"{w_pfx}h1", ""); l_h1 = row.get(f"{l_pfx}h1", "")
    w_h2  = row.get(f"{w_pfx}h2", ""); l_h2 = row.get(f"{l_pfx}h2", "")
    has_box = any(v is not None for v in [w_fg, l_fg, w_reb, l_reb])

    # Margin classification
    if margin <= 3:
        margin_tag = "coin-flip"
    elif margin <= 7:
        margin_tag = "close"
    elif margin <= 14:
        margin_tag = "clear"
    else:
        margin_tag = "decisive"

    # Pre-game model scores for context when no box available
    feat_lkp = {str(r["team"]): r for _, r in bkt_df.iterrows()}
    w_row2 = feat_lkp.get(winner, {}); l_row2 = feat_lkp.get(loser, {})
    w_cs = safe_f(w_row2.get("contender_score")); l_cs = safe_f(l_row2.get("contender_score"))
    w_am = safe_f(w_row2.get("adj_margin"));      l_am = safe_f(l_row2.get("adj_margin"))

    parts = []

    # ── Line 1: result + model verdict ────────────────────────────────────────
    seed_part = f" (#{w_seed} over #{l_seed})" if w_seed and l_seed else ""
    upset_note = " — UPSET" if w_seed and l_seed and w_seed > l_seed + 2 else ""
    parts.append(
        f"{'✅' if correct else '❌'} {winner} def. {loser} {w_sc}–{l_sc}{seed_part}{upset_note}. "
        f"Model picked {row.get('model_pick','?')} at {conf:.0f}% confidence — {'correct' if correct else 'missed'}."
    )

    # ── Line 2: basketball insight from real stats ─────────────────────────────
    if has_box:
        fg_gap = (w_fg - l_fg) if w_fg is not None and l_fg is not None else None
        reb_gap = (w_reb - l_reb) if w_reb is not None and l_reb is not None else None
        to_gap  = (l_to - w_to) if w_to is not None and l_to is not None else None  # positive = winner had fewer TOs

        if margin_tag in ("decisive",):
            # Find the most telling stat — biggest absolute gap
            insights = []
            if fg_gap is not None and abs(fg_gap) >= 8:
                if fg_gap > 0:
                    insights.append(
                        f"{winner} shot {w_fg:.0f}% vs {loser}'s {l_fg:.0f}% — "
                        f"a {fg_gap:.0f}-point FG% advantage almost always decides the final margin in tournament play."
                    )
                else:
                    insights.append(
                        f"{loser} actually shot better ({l_fg:.0f}% vs {w_fg:.0f}%) but still lost by {margin} — "
                        f"other factors (turnovers, rebounding) overcame the shooting gap."
                    )
            if reb_gap is not None and reb_gap >= 7 and not insights:
                insights.append(
                    f"{winner} dominated the glass {int(w_reb)}–{int(l_reb)} — "
                    f"in March, extra possessions from offensive rebounds compound into a scoring gap that's hard to erase."
                )
            if to_gap is not None and to_gap >= 5 and not insights:
                insights.append(
                    f"{loser} committed {int(l_to)} turnovers to {winner}'s {int(w_to)} — "
                    f"teams that give up that many extra possessions rarely advance in the tournament."
                )
            # Halftime story
            if w_h1 and l_h1 and w_h2 and l_h2:
                try:
                    wh1, lh1, wh2, lh2 = int(w_h1), int(l_h1), int(w_h2), int(l_h2)
                    if lh1 > wh1 and wh2 > lh2:
                        insights.append(
                            f"{winner} trailed at half ({wh1}–{lh1}) but outscored {loser} {wh2}–{lh2} in the second half — "
                            f"halftime adjustments are one of the most underrated edges in March."
                        )
                    elif wh1 > lh1 + 8:
                        insights.append(
                            f"Built a {wh1}–{lh1} halftime lead and closed it out — "
                            f"teams that go up big early in tournament games convert at a high rate."
                        )
                except Exception:
                    pass
            if insights:
                parts.append(insights[0])
            elif fg_gap is not None:
                parts.append(
                    f"Shooting: {winner} {w_fg:.0f}% vs {loser} {l_fg:.0f}%. "
                    f"A {margin}-point blowout — the stats matched the result."
                )

        elif margin_tag == "clear":
            stat_lines = []
            if fg_gap is not None:
                stat_lines.append(("fg", abs(fg_gap), "FG% edge (" + str(int(w_fg)) + "% vs " + str(int(l_fg)) + "%)"))
            if reb_gap is not None and reb_gap >= 4:
                stat_lines.append(("reb", reb_gap, "rebounding edge (" + str(int(w_reb)) + "-" + str(int(l_reb)) + ")"))
            if to_gap is not None and to_gap >= 3:
                stat_lines.append(("to", to_gap, "turnover advantage (" + str(int(w_to)) + " vs " + str(int(l_to)) + ")"))
            stat_lines.sort(key=lambda x: x[1], reverse=True)
            if stat_lines:
                primary = stat_lines[0][2]
                fallback = stat_lines[-1][2] if len(stat_lines) > 1 else "overall play"
                kind = "efficiency" if "FG" in primary else "advantage"
                outcome_read = "the model confidence was grounded in real quality" if correct else "the actual performance outpaced what the model saw pre-game"
                if fg_gap is not None and fg_gap < 0 and correct:
                    parts.append(
                        loser + " actually shot better (" + str(int(l_fg)) + "% vs " + str(int(w_fg)) + "%), "
                        "but " + winner + "'s " + fallback + " "
                        "told a different story. The model pre-game read held up despite the shooting numbers."
                    )
                else:
                    parts.append(
                        winner + "'s " + primary + " was the clearest separator — "
                        "a " + str(margin) + "-point tournament win with that kind of " + kind + " "
                        "suggests " + outcome_read + "."
                    )
            else:
                parts.append(
                    "A " + str(margin) + "-point win — clear enough to draw signal from, "
                    "but no single stat dominated the box score."
                )

        else:  # coin-flip or close
            if fg_gap is not None:
                if abs(fg_gap) <= 4:
                    close_read = "The model was right directionally, but the margin says this was anyones game." if correct else "A loss this close does not necessarily mean the models read was wrong."
                    parts.append(
                        "Both teams shot similarly (" + str(int(w_fg)) + "% vs " + str(int(l_fg)) + "%) and it came down to " + str(margin) + " points — "
                        "games this tight often turn on late free throws or a single possession. " + close_read
                    )
                else:
                    close_read2 = "something beyond shooting efficiency kept this close (turnovers, rebounding, late-game execution)." if correct else "the shooting edge was not enough to convert, which happens in close tournament games."
                    parts.append(
                        winner + " shot " + str(int(w_fg)) + "% vs " + loser + "'s " + str(int(l_fg)) + "%, yet it came down to " + str(margin) + " — " + close_read2
                    )
            else:
                var_read = "Do not over-correct from a close win; the signal is weak." if correct else "Do not over-correct from a close miss; the model principles may still be sound."
                parts.append(
                    "A " + str(margin) + "-point game — outcomes this close carry high variance. " + var_read
                )

    else:
        # No box score — use pre-game model scores
        if w_cs and l_cs and abs(w_cs - l_cs) >= 5:
            cs_gap = w_cs - l_cs
            confirmed = "confirmed" if correct else "contradicted"
            parts.append(
                "Pre-game Contender Scores: " + winner + " " + str(int(w_cs)) + " vs " + loser + " " + str(int(l_cs)) + " (gap " + (("+" if cs_gap > 0 else "") + str(int(cs_gap))) + "). "
                "Pre-game metrics suggested a gap — the result " + confirmed + " that read."
            )
        elif margin_tag in ("coin-flip", "close"):
            close_no_box = "too close to read strong signal into." if correct else "close enough that this could go either way on a different night."
            parts.append("A " + str(margin) + "-point result with no box score yet — " + close_no_box)

    # ── Line 3: what the model takes away ─────────────────────────────────────
    if margin_tag in ("clear", "decisive"):
        if correct:
            if conf >= 75:
                parts.append(
                    "High-confidence call (" + str(int(conf)) + "%) that held up by " + str(margin) + " — "
                    "when the model is this sure and the margin is this wide, it signals the underlying quality metrics are reading the matchup correctly."
                )
            else:
                parts.append(
                    "Model was only " + str(int(conf)) + "% here but won by " + str(margin) + " — "
                    "the actual performance was more one-sided than expected. Worth checking whether this team's metrics are underrated."
                )
        else:
            if conf >= 75:
                round_note = "the model may be overweighting seed or pre-game metrics in this round" if rnd in ("R64", "FF4") else "teams that advance to this stage often have qualities that regular-season metrics do not fully capture"
                parts.append(
                    "A " + str(int(conf)) + "%-confidence miss by " + str(margin) + " points is meaningful feedback — " + round_note + "."
                )
            else:
                flag_note = "An upset tag on " + loser + " was there pre-tournament." if ("Fraud" in l_flags or "Dangerous" in w_flags) else "This is the kind of game to revisit when reweighting features."
                parts.append(
                    "Missed at " + str(int(conf)) + "% confidence and lost by " + str(margin) + " — the result was not close. " + flag_note
                )
    elif margin_tag in ("coin-flip", "close"):
        if not correct:
            parts.append(
                "Lost by " + str(margin) + " — within the noise floor of tournament basketball. "
                "Do not over-correct. A game this tight on a different night could easily flip."
            )

    return "\n".join(p for p in parts if p)


def _round_from_headline(hl):
    """Map raw ESPN headline string to a standard round code."""
    hl = hl.lower()
    if "first four" in hl:                             return "FF4"
    if "1st round" in hl or "first round" in hl:       return "R64"
    if "second round" in hl or "2nd round" in hl:      return "R32"
    if "sweet 16" in hl:                               return "S16"
    if "elite 8" in hl:                                return "E8"
    if "final four" in hl and "first" not in hl:       return "FF"
    if "championship" in hl:                           return "Championship"
    return None


def load_or_update_results(bracket_teams, in_bracket, all_round_matchups):
    """Load persistent results from CSV, merge fresh ESPN data, save, return DataFrame.
    Uses session_state cache so this only runs once per Streamlit rerun cycle."""
    _cache_key = "_recap_df_cache"
    if _cache_key in st.session_state:
        return st.session_state[_cache_key]
    # Build lookup tables
    seed_lkp  = {}; flag_lkp = {}; score_lkp = {}; hl_lkp = {}
    for _, r in in_bracket.iterrows():
        t = str(r.get("team", ""))
        seed_lkp[t]  = int(r["seed"]) if pd.notna(r.get("seed")) else 0
        score_lkp[t] = safe_f(r.get("contender_score", 50), 50)
        flags = []
        if r.get("fraud_favorite_flag"):     flags.append("Fraud Fav")
        if r.get("cinderella_flag"):          flags.append("Cinderella")
        if r.get("dangerous_low_seed_flag"):  flags.append("Dangerous")
        if r.get("underseeded_flag"):         flags.append("Underseeded")
        flag_lkp[t] = ", ".join(flags)
        hl_lkp[t]   = hot_label(r)

    model_pick_lkp = {}
    for rnd_name, matchup_list in all_round_matchups.items():
        for t1n, t2n, w, l, reg in matchup_list:
            model_pick_lkp[(t1n, t2n)] = (w, l, rnd_name, reg)
            model_pick_lkp[(t2n, t1n)] = (w, l, rnd_name, reg)

    # Load existing CSV
    existing_ids = set()
    existing_rows = []
    if os.path.exists(_RESULTS_CSV):
        try:
            old_df = pd.read_csv(_RESULTS_CSV)
            # Normalize raw ESPN headline strings → standard round codes
            known_rounds = {"FF4", "R64", "R32", "S16", "E8", "FF", "Championship"}
            if "round" in old_df.columns:
                rl = old_df["round"].astype(str).str.lower()
                bad = ~old_df["round"].isin(known_rounds)
                old_df.loc[bad & rl.str.contains("first four",   na=False), "round"] = "FF4"
                old_df.loc[bad & rl.str.contains("1st round",    na=False), "round"] = "R64"
                old_df.loc[bad & rl.str.contains("second round", na=False), "round"] = "R32"
                old_df.loc[bad & rl.str.contains("2nd round",    na=False), "round"] = "R32"
                old_df.loc[bad & rl.str.contains("sweet 16",     na=False), "round"] = "S16"
                old_df.loc[bad & rl.str.contains("elite 8",      na=False), "round"] = "E8"
                old_df.loc[bad & rl.str.contains("final four",   na=False) &
                           ~rl.str.contains("first",             na=False), "round"] = "FF"
                old_df.loc[bad & rl.str.contains("championship", na=False), "round"] = "Championship"
                # Any still-unknown rows with typical R64 patterns (regional/1st round variants)
                bad2 = ~old_df["round"].isin(known_rounds)
                old_df.loc[bad2, "round"] = "R64"  # safest default for unclassified tournament games
            existing_ids = set(str(x) for x in old_df["event_id"].tolist())
            existing_rows = old_df.to_dict("records")
        except Exception as _csv_err:
            print(f"[Statlasberg] Error reading results CSV: {_csv_err}")

    # Build team-pair set from existing rows to prevent duplicates even with different event IDs
    existing_pairs = set()
    for er in existing_rows:
        t1e = str(er.get("t1", "") or "").strip()
        t2e = str(er.get("t2", "") or "").strip()
        if t1e and t2e:
            existing_pairs.add(frozenset({t1e, t2e}))

    # Fetch fresh from ESPN (multi-date)
    fresh_games = fetch_all_tournament_games(bracket_teams)
    new_rows = []
    for eid, g in fresh_games.items():
        if str(eid) in existing_ids:
            continue
        # Also skip if same team pair already exists (prevents duplicates from different event IDs)
        pair = frozenset({g["t1"], g["t2"]})
        if pair in existing_pairs:
            continue
        t1, t2 = g["t1"], g["t2"]
        winner, loser = g["winner"], g["loser"]
        # Check headline FIRST — First Four must override any bracket-based label
        headline = g.get("headline", "").lower()
        headline_round = _round_from_headline(headline)

        # Seed-aware score lookup: if team isn't in bracket, fall back to seed-based estimate
        def _score_with_seed_fallback(team, espn_seed):
            if team in score_lkp:
                return score_lkp[team]
            return _SEED_SCORE_FALLBACK.get(int(espn_seed or 0), 50)

        model_tuple = model_pick_lkp.get((t1, t2)) or model_pick_lkp.get((t2, t1))
        if headline_round == "FF4":
            # First Four always wins — never let bracket lookup label it R64
            s1 = _score_with_seed_fallback(t1, g.get("t1_espn_seed", 0))
            s2 = _score_with_seed_fallback(t2, g.get("t2_espn_seed", 0))
            model_winner = t1 if win_prob_sigmoid(s1, s2) >= 0.5 else t2
            model_loser  = t2 if model_winner == t1 else t1
            rnd = "FF4"; region = ""
        elif model_tuple:
            model_winner, model_loser, rnd, region = model_tuple
        else:
            s1 = _score_with_seed_fallback(t1, g.get("t1_espn_seed", 0))
            s2 = _score_with_seed_fallback(t2, g.get("t2_espn_seed", 0))
            model_winner = t1 if win_prob_sigmoid(s1, s2) >= 0.5 else t2
            model_loser  = t2 if model_winner == t1 else t1
            rnd = headline_round or "R64"
            region = ""
        correct    = (winner == model_winner)
        # Use seed fallback for confidence too
        mw_score = _score_with_seed_fallback(model_winner, g.get("t1_espn_seed" if model_winner == t1 else "t2_espn_seed", 0))
        ml_score = _score_with_seed_fallback(model_loser,  g.get("t2_espn_seed" if model_winner == t1 else "t1_espn_seed", 0))
        model_conf = win_prob_sigmoid(mw_score, ml_score)
        w_seed = seed_lkp.get(winner, 0); l_seed = seed_lkp.get(loser, 0)
        upset  = (w_seed > l_seed + 3) if w_seed and l_seed else False
        # Box score columns — stored directly from ESPN (already attached to g by fetch_all)
        box_cols = [
            "t1_fg_pct","t1_fg3_pct","t1_ft_pct","t1_rebounds","t1_off_reb",
            "t1_turnovers","t1_assists","t1_steals","t1_blocks","t1_h1","t1_h2",
            "t2_fg_pct","t2_fg3_pct","t2_ft_pct","t2_rebounds","t2_off_reb",
            "t2_turnovers","t2_assists","t2_steals","t2_blocks","t2_h1","t2_h2",
        ]
        row_dict = {
            "event_id":     eid,
            "date":         g.get("date", ""),
            "round":        rnd,
            "region":       region,
            "t1":           t1, "t2": t2,
            "winner":       winner, "loser": loser,
            "t1_score":     g["t1_score"], "t2_score": g["t2_score"],
            "model_pick":   model_winner,
            "model_conf":   round(model_conf, 3),
            "correct":      correct,
            "upset":        upset,
            "winner_seed":  w_seed, "loser_seed": l_seed,
            "winner_flags": flag_lkp.get(winner, ""),
            "loser_flags":  flag_lkp.get(loser, ""),
            "winner_hot":   hl_lkp.get(winner, ""),
            "loser_hot":    hl_lkp.get(loser, ""),
            "narrative":    "",
        }
        for bc in box_cols:
            row_dict[bc] = g.get(bc, "")
        row_dict["narrative"] = game_narrative(row_dict, in_bracket)
        new_rows.append(row_dict)

    if new_rows:
        st.toast(f"✅ Found {len(new_rows)} new game(s) from ESPN!", icon="🏀")
    elif st.session_state.get("_did_refresh"):
        st.toast(f"Up to date — all {len(existing_rows)} games already tracked", icon="✅")
    all_rows = existing_rows + new_rows
    if not all_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)

    # Normalize string columns — CSV round-trips turn "" into NaN
    for _col in ("winner_flags", "loser_flags", "winner_hot", "loser_hot", "narrative", "round", "region"):
        if _col in result_df.columns:
            result_df[_col] = result_df[_col].fillna("").astype(str)
    # Ensure date column is consistently str (prevents sort errors from int/str mix)
    if "date" in result_df.columns:
        result_df["date"] = result_df["date"].astype(str)

    # Always save — ensures normalized round labels are persisted back to disk
    try:
        os.makedirs(os.path.dirname(_RESULTS_CSV), exist_ok=True)
        result_df.to_csv(_RESULTS_CSV, index=False)
    except Exception as _save_err:
        print(f"[Statlasberg] Error saving results CSV: {_save_err}")

    # Cache in session_state so subsequent calls in the same rerun are instant
    st.session_state["_recap_df_cache"] = result_df
    return result_df



with tab5:
    st.markdown('<div class="section-header">🏆 2026 NCAA Tournament Bracket</div>', unsafe_allow_html=True)
    _bkt_hdr1, _bkt_hdr2 = st.columns([5, 1])
    with _bkt_hdr1:
        st.caption("Model predictions · Live results update as games complete · Simulations adjust automatically")
    with _bkt_hdr2:
        if st.button("🔄 Refresh", key="bracket_refresh"):
            st.cache_data.clear()
            st.session_state.pop("_recap_df_cache", None)
            # Also clear box_score caches so they re-fetch
            for k in list(st.session_state.keys()):
                if k.startswith("_box_score_"):
                    del st.session_state[k]
            st.session_state["_did_refresh"] = True
            st.session_state["_last_refresh"] = datetime.now().strftime("%I:%M %p")
            st.rerun()

    BRACKET_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    if "region" not in in_bracket.columns or len(in_bracket) == 0:
        st.warning("Bracket data not loaded.")
    else:
        # Compute bracket data for this tab (with ESPN fetch + CSV update)
        all_round_matchups = build_round_matchups(in_bracket)
        _bkt_team_set = frozenset(in_bracket["team"].tolist())
        _bkt_recap = load_or_update_results(_bkt_team_set, in_bracket, all_round_matchups)
        _adapted_scores = _compute_adapted_scores(in_bracket, _bkt_recap)
        live_rounds = build_bracket_live(in_bracket, _bkt_recap, _adapted_scores)

        # ── Record header ────────────────────────────────────────────────────
        if not _bkt_recap.empty and "correct" in _bkt_recap.columns:
            # Normalize date column to str so sort never hits int/str mix
            if "date" in _bkt_recap.columns:
                _bkt_recap["date"] = _bkt_recap["date"].astype(str)
            _total_bkt = len(_bkt_recap)
            _correct_bkt = int(_bkt_recap["correct"].sum())
            _pct_bkt = _correct_bkt / _total_bkt * 100 if _total_bkt else 0
            _streak_bkt = 0
            for _cv in _bkt_recap.sort_values("date", ascending=False)["correct"].tolist():
                if _cv:
                    _streak_bkt += 1
                else:
                    break
            _hc = st.columns(4)
            _hc[0].metric("🤖 Statlasberg Record", f"{_correct_bkt}–{_total_bkt - _correct_bkt}", f"{_pct_bkt:.0f}%")
            _hc[1].metric("🔥 Streak", f"W{_streak_bkt}" if _streak_bkt > 0 else "—")
            _rnd_strs_bkt = []
            for _rnd_bkt in ["R64", "R32", "S16", "E8", "FF"]:
                _rdf_bkt = _bkt_recap[_bkt_recap["round"] == _rnd_bkt]
                if len(_rdf_bkt):
                    _rnd_strs_bkt.append(f"{_rnd_bkt} {int(_rdf_bkt['correct'].sum())}/{len(_rdf_bkt)}")
            _hc[2].metric("📊 By Round", " · ".join(_rnd_strs_bkt) or "—")
            _hc[3].metric("🎯 Confidence", "HIGH" if _pct_bkt > 75 else "MODERATE" if _pct_bkt > 60 else "CALIBRATING")
            _last_ref = st.session_state.get("_last_refresh", "")
            _ref_note = f" · Last refresh: {_last_ref}" if _last_ref else ""
            st.caption(f"📡 Auto-synced with ESPN · {_total_bkt} games tracked{_ref_note}")
            st.markdown("---")
            # Clear the refresh flag after showing the result
            st.session_state.pop("_did_refresh", None)

        # ── Helper: render one matchup card ─────────────────────────────────
        def _bkt_card(m, round_tag=""):
            t1 = m.get("t1", ""); t2 = m.get("t2", "")
            s1 = m.get("s1", 0);  s2 = m.get("s2", 0)
            winner = m.get("winner", ""); loser = m.get("loser", "")
            mw     = m.get("model_winner", winner)
            wp     = m.get("winner_p", 0.5)
            done   = m.get("completed", False)
            ok     = m.get("model_correct")
            if not t1 and not t2:
                return

            if done:
                # ── completed game ────────────────────────────────────────
                w_seed = s1 if winner == t1 else s2
                l_seed = s2 if winner == t1 else s1
                verdict = "✅ Called it" if ok else f"❌ Missed — had {mw}"
                if ok:
                    bg     = "#0a1f0a"
                    border = "#22c55e"
                    w_color = "#4ade80"
                    v_color = "#4ade80"
                    icon    = "✅"
                else:
                    bg     = "#1f0a0a"
                    border = "#ef4444"
                    w_color = "#f87171"
                    v_color = "#f87171"
                    icon    = "❌"
                st.markdown(
                    f'<div style="background:{bg};border:2px solid {border};border-radius:8px;padding:8px 12px;margin:3px 0">'
                    f'<div style="color:{w_color};font-weight:800;font-size:1rem">{icon} #{w_seed} {winner}</div>'
                    f'<div style="color:#64748b;font-size:0.78rem;margin-top:2px">def. #{l_seed} {loser}</div>'
                    f'<div style="color:{v_color};font-size:0.72rem;margin-top:4px">{verdict} · {wp*100:.0f}% conf</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                # ── model projection ──────────────────────────────────────
                # Use the model_winner from resolve() (which includes tossup) for the pick,
                # but show contender_score-based probabilities for display
                t1_row = in_bracket[in_bracket["team"] == t1]
                t2_row = in_bracket[in_bracket["team"] == t2]
                c1 = safe_f(t1_row.iloc[0].get("contender_score", 50)) if len(t1_row) else 50
                c2 = safe_f(t2_row.iloc[0].get("contender_score", 50)) if len(t2_row) else 50
                p1 = win_prob_sigmoid(c1, c2); p2 = 1 - p1
                fav   = t1 if p1 >= p2 else t2
                fav_p = max(p1, p2)
                dog   = t2 if p1 >= p2 else t1
                dog_p = min(p1, p2)
                fs    = s1 if fav == t1 else s2
                ds    = s2 if fav == t1 else s1
                fav_row = t1_row.iloc[0] if len(t1_row) and fav == t1 else (t2_row.iloc[0] if len(t2_row) else None)
                hl = hot_label(fav_row) if fav_row is not None else ""
                hl_span = f' <span style="color:#4ade80;font-size:0.65rem">{hl}</span>' if hl else ""
                flags = []
                for _tn, _tr in [(t1, t1_row), (t2, t2_row)]:
                    if len(_tr):
                        r = _tr.iloc[0]
                        if r.get("fraud_favorite_flag"):      flags.append(f"⚠️ {_tn}")
                        if r.get("cinderella_flag"):           flags.append(f"🪄 {_tn}")
                        if r.get("dangerous_low_seed_flag"):   flags.append(f"💥 {_tn}")
                flag_line = (f'<div style="color:#f59e0b;font-size:0.68rem;margin-top:3px">'
                             f'{" · ".join(flags)}</div>') if flags else ""
                st.markdown(
                    f'<div style="background:#131820;border:2px solid #f97316;border-radius:8px;padding:8px 12px;margin:3px 0">'
                    f'<div style="color:#f1f5f9;font-weight:800;font-size:0.95rem">#{fs} {fav}{hl_span}</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:4px;margin:4px 0">'
                    f'<div style="width:{int(fav_p*100)}%;background:#f97316;height:4px;border-radius:3px"></div></div>'
                    f'<div style="color:#f97316;font-weight:700;font-size:0.85rem">{fav_p*100:.0f}% · {american_line(fav_p)}</div>'
                    f'<div style="color:#94a3b8;font-size:0.78rem;margin-top:4px">vs #{ds} {dog} ({dog_p*100:.0f}%)</div>'
                    f'{flag_line}'
                    f'</div>', unsafe_allow_html=True)

            # ── Predict Score button ──────────────────────────────────────
            if t1 and t2:
                _rtag = round_tag or m.get("region", "")
                if st.button("🎯 Score", key=f"pred_{t1}_{t2}_{_rtag}",
                             help=f"Predict {t1} vs {t2} final score",
                             use_container_width=True):
                    st.session_state.predict_t1 = t1
                    st.session_state.predict_t2 = t2
                    st.rerun()

        # ── 500K Monte Carlo Simulation ─────────────────────────────────────
        _r32_pairs = [(m["t1"], m["t2"]) for m in live_rounds["R32"] if m.get("t1") and m.get("t2")]
        if len(_r32_pairs) == 16:
            # Build score lookup from all 32 teams still in bracket (actual R64 winners)
            _mc_score_lkp = {}
            for _mm in live_rounds["R32"]:
                for _tt in [_mm.get("t1",""), _mm.get("t2","")]:
                    if _tt:
                        _mc_score_lkp[_tt] = safe_f(_adapted_scores.get(_tt, 50))
            _mc_probs = run_monte_carlo_sim(
                tuple(_r32_pairs),
                tuple(sorted(_mc_score_lkp.items())),
                n=500_000
            )
            if _mc_probs:
                _mc_sorted = sorted(_mc_probs.items(), key=lambda x: x[1], reverse=True)
                st.markdown('<div class="section-header">🎲 500,000-Game Simulation — Sweet 16 Advancement Odds</div>', unsafe_allow_html=True)
                st.caption("How often each team advances to the Sweet 16 across 500K simulations. Sorted by probability.")
                _mc_cols = st.columns(4)
                for _mci, (_mct, _mcp) in enumerate(_mc_sorted):
                    _col = _mc_cols[_mci % 4]
                    with _col:
                        _bar_w = int(_mcp * 100)
                        _seed_r = in_bracket[in_bracket["team"] == _mct]["seed"].values
                        _seed_d = int(_seed_r[0]) if len(_seed_r) else "?"
                        _bar_color = "#22c55e" if _mcp >= 0.5 else "#f97316" if _mcp >= 0.3 else "#64748b"
                        st.markdown(
                            f'<div style="background:#0f1419;border:1px solid #1e293b;border-radius:6px;padding:6px 10px;margin:3px 0">'
                            f'<div style="color:#f1f5f9;font-weight:700;font-size:0.82rem">#{_seed_d} {_mct}</div>'
                            f'<div style="background:#1e293b;border-radius:3px;height:5px;margin:4px 0">'
                            f'<div style="width:{_bar_w}%;background:{_bar_color};height:5px;border-radius:3px"></div></div>'
                            f'<div style="color:{_bar_color};font-size:0.78rem;font-weight:600">{_mcp*100:.1f}%</div>'
                            f'</div>', unsafe_allow_html=True)
                st.markdown("---")

        # ── Round tabs ───────────────────────────────────────────────────────
        br_r64, br_r32, br_s16, br_e8, br_ff, br_champ = st.tabs([
            "R64", "R32", "Sweet 16", "Elite 8", "Final Four", "🏆 Championship"
        ])

        with br_r64:
            _done_r64  = sum(1 for m in live_rounds["R64"] if m.get("completed"))
            _total_r64 = len(live_rounds["R64"])
            st.caption(f"Round of 64 · {_done_r64}/{_total_r64} games complete")
            _r64_cols = st.columns(4)
            for _ri, _region in enumerate(["East", "South", "West", "Midwest"]):
                with _r64_cols[_ri]:
                    _top1 = in_bracket[(in_bracket["region"] == _region) & (in_bracket["seed"] == 1)]
                    _top1_name = _top1.iloc[0]["team"] if len(_top1) else "TBD"
                    st.markdown(
                        f'<div style="color:#ff8c3a;font-weight:800;font-size:0.85rem;border-bottom:2px solid #f97316;'
                        f'padding-bottom:3px;margin-bottom:10px;text-transform:uppercase">{_region}<br/>'
                        f'<span style="color:#94a3b8;font-size:0.75rem;font-weight:400">#1 {_top1_name}</span></div>',
                        unsafe_allow_html=True)
                    for _m in [m for m in live_rounds["R64"] if m.get("region") == _region]:
                        _s1_h = _m.get("s1", 0); _s2_h = _m.get("s2", 0)
                        st.markdown(f'<div style="color:#475569;font-size:0.68rem;margin-top:6px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                        _bkt_card(_m, round_tag="R64")

        with br_r32:
            _done_r32  = sum(1 for m in live_rounds["R32"] if m.get("completed"))
            _total_r32 = len(live_rounds["R32"])
            st.caption(f"Round of 32 · {_done_r32}/{_total_r32} games complete")
            _r32_cols = st.columns(4)
            for _ri2, _region in enumerate(["East", "South", "West", "Midwest"]):
                with _r32_cols[_ri2]:
                    st.markdown(
                        f'<div style="color:#ff8c3a;font-weight:700;font-size:0.8rem;border-bottom:1px solid #f97316;'
                        f'padding-bottom:2px;margin-bottom:8px;text-transform:uppercase">{_region}</div>',
                        unsafe_allow_html=True)
                    for _m in [m for m in live_rounds["R32"] if m.get("region") == _region]:
                        _s1_h = _m.get("s1", 0); _s2_h = _m.get("s2", 0)
                        st.markdown(f'<div style="color:#475569;font-size:0.68rem;margin-top:6px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                        _bkt_card(_m, round_tag="R32")

        with br_s16:
            _done_s16 = sum(1 for m in live_rounds["S16"] if m.get("completed"))
            st.caption(f"Sweet 16 · {_done_s16}/{len(live_rounds['S16'])} games complete")
            _s16_cols = st.columns(2)
            for _si, _m in enumerate(live_rounds["S16"]):
                with _s16_cols[_si % 2]:
                    st.markdown(
                        f'<div style="color:#ff8c3a;font-weight:700;font-size:0.8rem;margin-bottom:6px;text-transform:uppercase">'
                        f'{_m.get("region", "")} Region</div>',
                        unsafe_allow_html=True)
                    _s1_h = _m.get("s1", 0); _s2_h = _m.get("s2", 0)
                    st.markdown(f'<div style="color:#475569;font-size:0.72rem;margin-bottom:4px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                    _bkt_card(_m, round_tag="S16")

        with br_e8:
            _done_e8 = sum(1 for m in live_rounds["E8"] if m.get("completed"))
            st.caption(f"Elite 8 · {_done_e8}/{len(live_rounds['E8'])} games complete")
            _e8_cols = st.columns(2)
            for _ei, _m in enumerate(live_rounds["E8"]):
                with _e8_cols[_ei % 2]:
                    st.markdown(
                        f'<div style="color:#ff8c3a;font-weight:700;font-size:0.9rem;margin-bottom:6px;text-transform:uppercase">'
                        f'🏆 {_m.get("region", "")} Region Championship</div>',
                        unsafe_allow_html=True)
                    _s1_h = _m.get("s1", 0); _s2_h = _m.get("s2", 0)
                    st.markdown(f'<div style="color:#475569;font-size:0.72rem;margin-bottom:4px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                    _bkt_card(_m, round_tag="E8")

        with br_ff:
            _done_ff = sum(1 for m in live_rounds["FF"] if m.get("completed"))
            st.caption(f"Final Four · {_done_ff}/{len(live_rounds['FF'])} games complete")
            _ff_cols = st.columns(2)
            for _fi, _m in enumerate(live_rounds["FF"]):
                with _ff_cols[_fi]:
                    st.markdown(
                        f'<div style="color:#fbbf24;font-weight:800;font-size:1rem;margin-bottom:8px">🏅 Semifinal {_fi + 1}</div>',
                        unsafe_allow_html=True)
                    _s1_h = _m.get("s1", 0); _s2_h = _m.get("s2", 0)
                    st.markdown(f'<div style="color:#475569;font-size:0.72rem;margin-bottom:4px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                    _bkt_card(_m, round_tag="FF")

        with br_champ:
            st.markdown('<div style="text-align:center;font-size:1.4rem;font-weight:900;color:#fbbf24;margin:16px 0">🏆 Championship Game</div>', unsafe_allow_html=True)
            if live_rounds["Championship"]:
                _cm = live_rounds["Championship"][0]
                _ch_cols = st.columns([1, 2, 1])
                with _ch_cols[1]:
                    _s1_h = _cm.get("s1", 0); _s2_h = _cm.get("s2", 0)
                    st.markdown(f'<div style="color:#475569;font-size:0.78rem;text-align:center;margin-bottom:6px">#{_s1_h} vs #{_s2_h}</div>', unsafe_allow_html=True)
                    _bkt_card(_cm, round_tag="CHAMP")
                if _cm.get("completed"):
                    _champ_w = _cm["winner"]
                    _champ_s = _cm.get("s1", 0) if _champ_w == _cm["t1"] else _cm.get("s2", 0)
                    st.markdown(
                        f'<div style="text-align:center;background:linear-gradient(135deg,#1a2a1a,#0f2b0f);'
                        f'border:2px solid #4ade80;border-radius:12px;padding:24px;margin-top:16px">'
                        f'<div style="font-size:2.5rem">🏆</div>'
                        f'<div style="color:#4ade80;font-weight:900;font-size:1.5rem;margin:8px 0">{_champ_w}</div>'
                        f'<div style="color:#94a3b8">#{_champ_s} seed · 2026 National Champion</div>'
                        f'</div>', unsafe_allow_html=True)
                elif _cm["t1"] and _cm["t2"]:
                    _proj_w = _cm.get("model_winner", _cm["winner"])
                    _proj_s = _cm.get("s1", 0) if _proj_w == _cm["t1"] else _cm.get("s2", 0)
                    _proj_p = _cm.get("winner_p", 0.5)
                    st.markdown(
                        f'<div style="text-align:center;background:linear-gradient(135deg,#1a1a2a,#0f0f2b);'
                        f'border:2px solid #f97316;border-radius:12px;padding:24px;margin-top:16px">'
                        f'<div style="font-size:2.5rem">🏆</div>'
                        f'<div style="color:#f97316;font-weight:900;font-size:1.5rem;margin:8px 0">{_proj_w}</div>'
                        f'<div style="color:#94a3b8">#{_proj_s} seed · Statlasberg\'s Projected Champion · {_proj_p*100:.0f}% conf</div>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.info("Championship matchup will be determined as the tournament progresses.")




# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — LIVE GAMES  (ESPN public API, no key required)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=25)
def fetch_ncaa_live_games():
    """Fetch live scores from NCAA.com official scoreboard — second source to
    cross-check ESPN and catch any games ESPN's filter misses."""
    if not _REQUESTS_OK:
        return []
    try:
        from datetime import datetime
        now = datetime.now()
        url = _NCAA_SCOREBOARD.format(
            year=now.year, month=f"{now.month:02d}", day=f"{now.day:02d}"
        )
        resp = _requests.get(url, timeout=6, headers={
            "User-Agent": "statlasberg/1.0", "Accept": "application/json"
        })
        resp.raise_for_status()
        data = resp.json()
        games = []
        for gw in data.get("games", []):
            g = gw.get("game", {})
            home = g.get("home", {}); away = g.get("away", {})
            state_raw = g.get("gameState", "pre").lower()
            state = "in" if state_raw in ("live", "in") else "post" if "final" in state_raw else "pre"
            period_raw = g.get("currentPeriod", "")
            period = 2 if any(x in period_raw for x in ("2H", "H2", "2nd")) else 1
            try:
                h_sc = int(home.get("score", 0) or 0)
                a_sc = int(away.get("score", 0) or 0)
            except Exception:
                h_sc = a_sc = 0
            games.append({
                "event_id": "ncaa_" + str(g.get("gameID", "")),
                "state": state, "period": period,
                "clock": g.get("contestClock", ""),
                "team1": {
                    "name":   home.get("names", {}).get("full", home.get("names", {}).get("short", "")),
                    "score":  h_sc,
                    "record": home.get("record", ""),
                },
                "team2": {
                    "name":   away.get("names", {}).get("full", away.get("names", {}).get("short", "")),
                    "score":  a_sc,
                    "record": away.get("record", ""),
                },
                "headline": g.get("network", ""),
                "source": "NCAA",
            })
        return games
    except Exception:
        return []


def _team_card_html(name, score_str, record, subtitle, border="#374151",
                    score_color="#f1f5f9", badge=""):
    """Shared team card block used across live, finished, and upcoming sections."""
    return (
        f'<div style="background:#131820;border:2px solid {border};'
        f'border-radius:8px;padding:10px;margin-bottom:4px">'
        f'<div style="font-size:1rem;font-weight:800;color:#f1f5f9">'
        f'{name}{(" " + badge) if badge else ""}</div>'
        f'<div style="font-size:2rem;font-weight:900;color:{score_color}">{score_str}</div>'
        f'<div style="color:#94a3b8;font-size:0.75rem">{record}</div>'
        f'<div style="color:#64748b;font-size:0.75rem;margin-top:3px">{subtitle}</div>'
        f'</div>'
    )


def _prob_bar_html(p, color):
    """Win probability fill bar."""
    return (
        f'<div style="background:#2d3748;border-radius:4px;height:7px;margin:5px 0">'
        f'<div style="width:{int(p*100)}%;background:{color};height:7px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="color:{color};font-weight:700;font-size:0.95rem">'
        f'{p*100:.0f}% win prob</div>'
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


# All 2026 NCAA tournament dates (First Four → Championship)
with tab6:
    st.markdown('<div class="section-header">📺 Live — NCAA Tournament</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, _ = st.columns([1, 1, 4])
    with ctrl1:
        auto_ref = st.toggle("⏱ Auto-refresh (60s)", value=False, key="live_auto_refresh")
    with ctrl2:
        if st.button("🔄 Refresh", key="live_refresh_btn"):
            st.cache_data.clear()
            st.session_state.pop("_recap_df_cache", None)
            for k in list(st.session_state.keys()):
                if k.startswith("_box_score_"):
                    del st.session_state[k]
            st.session_state["_did_refresh"] = True
            st.session_state["_last_refresh"] = datetime.now().strftime("%I:%M %p")
            st.rerun()
    if auto_ref:
        st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    if not _REQUESTS_OK:
        st.warning("⚠️ `requests` not installed. Run `pip install requests` and restart.")
    else:
        # Fetch from ESPN (primary) and NCAA.com (secondary), merge
        with st.spinner("Fetching scores…"):
            espn_games = fetch_live_games()
            ncaa_games = fetch_ncaa_live_games()

        # NCAA supplements ESPN — add games ESPN missed (match by team name pair)
        espn_pairs = {
            (g["team1"]["name"].lower(), g["team2"]["name"].lower())
            for g in espn_games
        }
        extra = [
            g for g in ncaa_games
            if (g["team1"]["name"].lower(), g["team2"]["name"].lower()) not in espn_pairs
            and (g["team2"]["name"].lower(), g["team1"]["name"].lower()) not in espn_pairs
        ]
        all_games = espn_games + extra

        live_now   = [g for g in all_games if g["state"] == "in"]
        recent_fin = [g for g in all_games if g["state"] == "post"][:8]
        upcoming   = [g for g in all_games if g["state"] == "pre"][:8]

        # Seed/flag helpers (used in Final Scores section)
        def _live_get_seed(team_name):
            norm = _BRACKET_NORM.get(team_name, team_name)
            rm = in_bracket[in_bracket["team"].str.lower() == norm.lower()]
            return int(rm.iloc[0].get("seed", 0) or 0) if not rm.empty else 0

        def _live_get_flags(team_name):
            norm = _BRACKET_NORM.get(team_name, team_name)
            rm = in_bracket[in_bracket["team"].str.lower() == norm.lower()]
            if not rm.empty:
                r = rm.iloc[0]
                flags = [fc.replace("_", " ").title()
                         for fc in ["fraud_favorite", "cinderella", "clutch_score", "hot"]
                         if r.get(fc)]
                return " | ".join(flags)
            return ""

        if not all_games:
            st.info("🏀 No tournament games right now. Games appear at tipoff (March 20 – April 7, 2026).")
        else:
            # ── LIVE NOW ──────────────────────────────────────────────────────
            if live_now:
                st.markdown(f'<div class="section-header">🔴 LIVE — {len(live_now)} game(s)</div>', unsafe_allow_html=True)
                for g in live_now:
                    t1 = g["team1"]; t2 = g["team2"]
                    sd = t1["score"] - t2["score"]
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    p1, p2 = live_win_prob(pregame_p, sd, g["period"], g["clock"])
                    lead_p  = max(p1, p2)
                    bar_col = "#4ade80" if lead_p >= 0.7 else "#f59e0b" if lead_p >= 0.55 else "#94a3b8"
                    period_str = ("1st" if g["period"] == 1 else "2nd" if g["period"] == 2 else "OT") + " · " + g["clock"]
                    src = g.get("source", "ESPN")
                    with st.expander(
                        f"🔴 {t1['name']} {t1['score']} — {t2['score']} {t2['name']}  |  {period_str}",
                        expanded=True
                    ):
                        lc1, lc2 = st.columns(2)
                        for col, team, p, cs in [(lc1, t1, p1, s1), (lc2, t2, p2, s2)]:
                            with col:
                                is_lead = team["score"] >= (t2["score"] if team is t1 else t1["score"])
                                st.markdown(
                                    _team_card_html(
                                        team["name"], str(team["score"]), team["record"],
                                        "Model score: " + f"{cs:.1f}",
                                        border="#f97316" if is_lead else "#374151",
                                        score_color="#4ade80" if is_lead else "#f1f5f9",
                                    )
                                    + _prob_bar_html(p, bar_col)
                                    + f'<div style="color:#64748b;font-size:0.73rem">{american_line(p)} live line</div>',
                                    unsafe_allow_html=True,
                                )
                                if st.button("🔍 Deep Dive", key=f"live_dd_{g['event_id']}_{team['name']}", use_container_width=True):
                                    st.session_state["team_selectbox"] = team["name"]
                                    st.session_state.dive_team = team["name"]
                                    st.rerun()

                        plays, momentum = fetch_play_by_play(g["event_id"])
                        if momentum:
                            st.markdown(
                                f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;padding:6px 10px;'
                                f'border-radius:4px;margin-top:6px;color:#d1fae5;font-weight:700">{momentum}</div>',
                                unsafe_allow_html=True)
                        if plays:
                            st.markdown('<small style="color:#64748b">Recent plays:</small>', unsafe_allow_html=True)
                            for play_txt in plays[:5]:
                                st.markdown(f'<div style="color:#94a3b8;font-size:0.78rem">• {play_txt}</div>', unsafe_allow_html=True)

                        mpick = t1["name"] if pregame_p >= 0.5 else t2["name"]
                        mconf = max(pregame_p, 1 - pregame_p)
                        agree = "✅ Aligns" if (p1 >= 0.5) == (pregame_p >= 0.5) else "⚠️ Diverging"
                        st.markdown(
                            f'<small style="color:#94a3b8">📊 Pre-game pick: <strong style="color:#f1f5f9">{mpick}</strong>'
                            f' ({mconf*100:.0f}%) — {agree} · Source: {src}</small>',
                            unsafe_allow_html=True)

            # ── FINAL SCORES ──────────────────────────────────────────────────
            if recent_fin:
                st.markdown("---")
                st.markdown('<div class="section-header">✅ Final Scores</div>', unsafe_allow_html=True)
                for g in recent_fin:
                    t1 = g["team1"]; t2 = g["team2"]
                    winner = t1 if t1["score"] > t2["score"] else t2
                    loser  = t2 if t1["score"] > t2["score"] else t1
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    model_got_it = (pregame_p >= 0.5) == (t1["score"] > t2["score"])
                    model_fav    = t1["name"] if pregame_p >= 0.5 else t2["name"]
                    model_fav_p  = max(pregame_p, 1 - pregame_p)
                    verdict      = "✅ Correct" if model_got_it else "❌ Upset"
                    verdict_icon = "✅" if model_got_it else "❌"
                    with st.expander(
                        f"{verdict_icon} {winner['name']} {winner['score']} def. {loser['name']} {loser['score']}  |  {verdict}",
                        expanded=False
                    ):
                        fc1, fc2 = st.columns(2)
                        for col, team, cs, is_win in [
                            (fc1, t1, s1, t1["score"] > t2["score"]),
                            (fc2, t2, s2, t2["score"] > t1["score"]),
                        ]:
                            with col:
                                st.markdown(
                                    _team_card_html(
                                        team["name"], str(team["score"]), team.get("record", ""),
                                        "Model score: " + f"{cs:.1f}",
                                        border="#4ade80" if is_win else "#374151",
                                        score_color="#4ade80" if is_win else "#f1f5f9",
                                    ),
                                    unsafe_allow_html=True)
                                if st.button("🔍 Deep Dive", key=f"fin_dd_{g['event_id']}_{team['name']}", use_container_width=True):
                                    st.session_state["team_selectbox"] = team["name"]
                                    st.session_state.dive_team = team["name"]
                                    st.rerun()

                        st.markdown(
                            f'<div style="background:#1e293b;border-radius:6px;padding:8px 12px;font-size:0.85rem">'
                            f'📊 Statlasberg: <strong style="color:#f1f5f9">{model_fav}</strong>'
                            f' ({model_fav_p*100:.0f}%) — {verdict}</div>',
                            unsafe_allow_html=True)

                        insight = style_matchup_insight_by_name(t1["name"], t2["name"], in_bracket)
                        if insight:
                            st.markdown(
                                f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;'
                                f'padding:6px 10px;color:#d1fae5;font-size:0.82rem;margin-top:6px">💡 {insight}</div>',
                                unsafe_allow_html=True)

                        # Instant recap
                        box = fetch_game_box_score(g["event_id"])
                        t1_is_winner = (t1["score"] > t2["score"])
                        row_dict = {
                            "t1": t1["name"], "t2": t2["name"],
                            "t1_score": t1["score"], "t2_score": t2["score"],
                            "winner": winner["name"], "loser": loser["name"],
                            "model_pick": model_fav, "model_conf": model_fav_p,
                            "correct": model_got_it,
                            "winner_seed": _live_get_seed(winner["name"]),
                            "loser_seed":  _live_get_seed(loser["name"]),
                            "winner_flags": _live_get_flags(winner["name"]),
                            "loser_flags":  _live_get_flags(loser["name"]),
                        }
                        for k, v in box.items():
                            row_dict[k] = v
                        narrative = game_narrative(row_dict, in_bracket)
                        if narrative:
                            border_c = "#4ade80" if model_got_it else "#f87171"
                            st.markdown(
                                f'<div style="background:#0f1f2e;border-left:3px solid {border_c};'
                                f'border-radius:4px;padding:8px 12px;margin-top:8px;color:#e2e8f0;font-size:0.82rem">'
                                f'<div style="font-weight:700;color:#94a3b8;margin-bottom:4px">📋 Statlas Recap</div>'
                                f'{narrative.replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True)

                        if box and any(box.get(f"t{i}_{s}") for i in [1, 2] for s in ["fg_pct", "rebounds", "turnovers"]):
                            w_pfx = "t1_" if t1_is_winner else "t2_"
                            l_pfx = "t2_" if t1_is_winner else "t1_"
                            stat_rows = [
                                {"Stat": lbl, winner["name"]: box.get(w_pfx + sk, ""), loser["name"]: box.get(l_pfx + sk, "")}
                                for sk, lbl in [("fg_pct", "FG%"), ("fg3_pct", "3PT%"), ("rebounds", "Reb"), ("turnovers", "TO"), ("assists", "AST")]
                                if box.get(w_pfx + sk) or box.get(l_pfx + sk)
                            ]
                            if stat_rows:
                                st.dataframe(pd.DataFrame(stat_rows).set_index("Stat"), use_container_width=True, hide_index=False)

            # ── UPCOMING ──────────────────────────────────────────────────────
            if upcoming:
                st.markdown("---")
                st.markdown('<div class="section-header">🕐 Upcoming — Pre-Game Lines</div>', unsafe_allow_html=True)
                for g in upcoming:
                    t1 = g["team1"]; t2 = g["team2"]
                    pregame_p, s1, s2 = model_pregame_prob(t1["name"], t2["name"], in_bracket)
                    fav_name = t1["name"] if pregame_p >= 0.5 else t2["name"]
                    fav_p    = max(pregame_p, 1 - pregame_p)
                    with st.expander(
                        f"🕐 {t1['name']} vs {t2['name']}  —  {fav_name} ({fav_p*100:.0f}%)",
                        expanded=False
                    ):
                        uc1, uc2 = st.columns(2)
                        for col, team, cs, p in [(uc1, t1, s1, pregame_p), (uc2, t2, s2, 1 - pregame_p)]:
                            with col:
                                is_fav  = team["name"] == fav_name
                                bar_col = "#4ade80" if is_fav else "#64748b"
                                st.markdown(
                                    _team_card_html(
                                        team["name"], american_line(p), team.get("record", ""),
                                        f"{p*100:.0f}% win prob  ·  Score: {cs:.1f}",
                                        border="#f97316" if is_fav else "#374151",
                                        badge="⭐ Pick" if is_fav else "",
                                    )
                                    + _prob_bar_html(p, bar_col),
                                    unsafe_allow_html=True)
                                if st.button("🔍 Deep Dive", key=f"up_dd_{g['event_id']}_{team['name']}", use_container_width=True):
                                    st.session_state["team_selectbox"] = team["name"]
                                    st.session_state.dive_team = team["name"]
                                    st.rerun()

                        insight = style_matchup_insight_by_name(t1["name"], t2["name"], in_bracket)
                        if insight:
                            st.markdown(
                                f'<div style="background:#1a2a1a;border-left:3px solid #16a34a;border-radius:4px;'
                                f'padding:6px 10px;color:#d1fae5;font-size:0.82rem">💡 {insight}</div>',
                                unsafe_allow_html=True)

        st.caption("Sources: ESPN + NCAA.com official · In-game probability blends pre-game model with live score differential.")


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
# TAB 7 — MODEL ACCURACY  (Historical Backtest 2015–2025)
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — RECAP  (persistent record + per-game narratives + round browser)
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown('<div class="section-header">📋 Statlasberg — 2026 Tournament Recap</div>', unsafe_allow_html=True)
    st.caption("Per-game model breakdown · Persistent record · Browse by round · Powered by ESPN data")

    rc1, rc2 = st.columns([5, 1])
    with rc2:
        if st.button("🔄 Refresh Results", key="recap_refresh"):
            st.cache_data.clear()
            st.session_state.pop("_recap_df_cache", None)
            st.rerun()

    bracket_team_set = frozenset(in_bracket["team"].tolist()) if len(in_bracket) > 0 else frozenset()
    recap_df = load_or_update_results(bracket_team_set, in_bracket, all_round_matchups)


    # ── Round label display map ────────────────────────────────────────────────
    _RND_LABELS = {
        "FF4": "First Four",
        "R64": "Round of 64", "R32": "Round of 32", "S16": "Sweet 16",
        "E8": "Elite 8", "FF": "Final Four", "Championship": "Championship",
    }
    _RND_ORDER = ["FF4", "R64", "R32", "S16", "E8", "FF", "Championship"]

    if recap_df.empty:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#64748b">'
            '<div style="font-size:3rem">🏀</div>'
            '<div style="font-size:1.2rem;font-weight:700;color:#94a3b8;margin-top:12px">No completed games yet</div>'
            '<div style="margin-top:8px">Check back once tournament games are underway.<br/>'
            'First Four tips off March 18 — Round of 64 begins March 19.</div>'
            '</div>', unsafe_allow_html=True)
    else:
        # ── Normalize dtypes from CSV (booleans come back as strings) ──────────
        def _to_bool(col):
            if col.dtype == object:
                return col.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))
            return col.astype(bool)
        recap_df["correct"]    = _to_bool(recap_df["correct"])
        if "upset" in recap_df.columns:
            recap_df["upset"] = _to_bool(recap_df["upset"])
        else:
            recap_df["upset"] = False
        recap_df["model_conf"]  = pd.to_numeric(recap_df["model_conf"], errors="coerce").fillna(0.5)
        for _sc in ("winner_seed", "loser_seed"):
            if _sc in recap_df.columns:
                recap_df[_sc] = pd.to_numeric(recap_df[_sc], errors="coerce").fillna(0).astype(int)
            else:
                recap_df[_sc] = 0

        total   = len(recap_df)
        correct = int(recap_df["correct"].sum())
        losses  = total - correct
        acc_pct = correct / total * 100 if total else 0
        upsets  = int(recap_df["upset"].sum())
        upsets_called = int(recap_df[recap_df["upset"] & recap_df["correct"]].shape[0])

        # Streak calculation
        streak_val = 0; streak_type = ""
        for c in recap_df["correct"].tolist()[::-1]:
            if streak_val == 0:
                streak_type = "W" if c else "L"
                streak_val = 1
            elif (c and streak_type == "W") or (not c and streak_type == "L"):
                streak_val += 1
            else:
                break
        streak_str = f"{streak_type}{streak_val}" if streak_val else "—"

        # ── SECTION A: Statlasberg Record ─────────────────────────────────────
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d1f35,#0a1628);border:2px solid #f97316;'
            f'border-radius:12px;padding:18px 24px;margin-bottom:16px">'
            f'<div style="color:#f97316;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px">Statlasberg 2026 Record</div>'
            f'<div style="display:flex;align-items:baseline;gap:12px;margin:6px 0">'
            f'<span style="color:#f1f5f9;font-size:2.2rem;font-weight:900">{correct}–{losses}</span>'
            f'<span style="color:{"#4ade80" if acc_pct>=65 else "#fbbf24" if acc_pct>=50 else "#f87171"};font-size:1.3rem;font-weight:700">{acc_pct:.0f}%</span>'
            f'<span style="color:#64748b;font-size:0.85rem">· Streak: <strong style="color:#fbbf24">{streak_str}</strong>'
            f' · Upsets called: {upsets_called}/{upsets}</span>'
            f'</div>'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">'
            + "".join([
                f'<span style="color:#94a3b8;font-size:0.78rem">'
                f'<strong style="color:#f1f5f9">{_RND_LABELS.get(r,r)}</strong>: '
                f'{int(recap_df[recap_df["round"]==r]["correct"].sum())}/{len(recap_df[recap_df["round"]==r])}'
                f'</span>'
                for r in _RND_ORDER if r in recap_df["round"].values
            ]) +
            f'</div></div>', unsafe_allow_html=True)

        # ── CONFIDENCE METER ───────────────────────────────────────────────────
        conf_buckets = [
            (0.5, 0.6, "50–60%"), (0.6, 0.7, "60–70%"),
            (0.7, 0.8, "70–80%"), (0.8, 1.01, "80%+"),
        ]
        conf_cols = st.columns(4)
        for ci, (lo, hi, label) in enumerate(conf_buckets):
            bdf = recap_df[(recap_df["model_conf"] >= lo) & (recap_df["model_conf"] < hi)]
            n = len(bdf)
            if n == 0:
                conf_cols[ci].markdown(
                    f'<div style="background:#131820;border:1px solid #1e293b;border-radius:8px;padding:12px;text-align:center">'
                    f'<div style="color:#475569;font-size:0.72rem;font-weight:700">{label} confidence</div>'
                    f'<div style="color:#475569;font-size:1.6rem;font-weight:900">—</div>'
                    f'<div style="color:#475569;font-size:0.68rem">no games yet</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                w = int(bdf["correct"].sum()); l = n - w
                actual = w / n * 100
                mid = (lo + hi) / 2 * 100
                diff = actual - mid
                color = "#4ade80" if diff >= -5 else "#fbbf24" if diff >= -15 else "#f87171"
                border = "#4ade80" if diff >= -5 else "#fbbf24" if diff >= -15 else "#f87171"
                conf_cols[ci].markdown(
                    f'<div style="background:#131820;border:1px solid {border};border-radius:8px;padding:12px;text-align:center">'
                    f'<div style="color:#94a3b8;font-size:0.72rem;font-weight:700">{label} confidence</div>'
                    f'<div style="color:{color};font-size:1.6rem;font-weight:900">{actual:.0f}%</div>'
                    f'<div style="color:#64748b;font-size:0.72rem">{w}–{l} · {n} games</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-top:6px">'
                    f'<div style="width:{actual:.0f}%;background:{color};height:5px;border-radius:3px"></div></div>'
                    f'</div>', unsafe_allow_html=True)

        # ── SECTION B: Browse by Round ─────────────────────────────────────────
        played_rounds = [r for r in _RND_ORDER if r in recap_df["round"].values]
        round_tab_labels = [_RND_LABELS.get(r, r) for r in played_rounds]
        if round_tab_labels:
            round_tabs = st.tabs(round_tab_labels)
            for ti, rnd in enumerate(played_rounds):
                with round_tabs[ti]:
                    rnd_df = recap_df[recap_df["round"] == rnd]
                    rnd_correct = int(rnd_df["correct"].sum())
                    rnd_total   = len(rnd_df)
                    rnd_acc     = rnd_correct / rnd_total * 100 if rnd_total else 0
                    acc_color   = "#4ade80" if rnd_acc >= 70 else "#fbbf24" if rnd_acc >= 50 else "#f87171"
                    st.markdown(
                        f'<div style="color:{acc_color};font-weight:700;font-size:1rem;margin-bottom:10px">'
                        f'Statlas went {rnd_correct}–{rnd_total - rnd_correct} in {_RND_LABELS.get(rnd,rnd)} ({rnd_acc:.0f}%)'
                        f'</div>', unsafe_allow_html=True)

                    for _, row in rnd_df.iterrows():
                        hit    = bool(row["correct"])
                        upset  = bool(row.get("upset", False))
                        border = "#4ade80" if hit else "#f87171"
                        icon   = "✅" if hit else "❌"
                        upset_badge = ' <span style="background:#f97316;color:#000;border-radius:3px;padding:1px 6px;font-size:0.68rem;font-weight:800">UPSET</span>' if upset else ""
                        t1_sc  = int(row.get("t1_score", 0)); t2_sc = int(row.get("t2_score", 0))
                        w_sc   = t1_sc if row["winner"] == row["t1"] else t2_sc
                        l_sc   = t2_sc if row["winner"] == row["t1"] else t1_sc
                        conf   = float(row["model_conf"]) * 100
                        narrative = str(row.get("narrative", ""))

                        expander_label = f"{icon} {row['winner']} def. {row['loser']}  ({w_sc}–{l_sc}){'  🚨 UPSET' if upset else ''}"
                        with st.expander(expander_label, expanded=False):
                            # Header
                            st.markdown(
                                f'<div style="background:#0f172a;border-left:4px solid {border};border-radius:6px;padding:10px 14px;margin-bottom:8px">'
                                f'<div style="color:#f1f5f9;font-weight:700;font-size:1rem">'
                                f'{icon} {row["winner"]} (#{int(row["winner_seed"])}) def. {row["loser"]} (#{int(row["loser_seed"])})'
                                f'  <span style="color:#64748b;font-size:0.85rem">{w_sc}–{l_sc}</span>'
                                f'{upset_badge}</div>'
                                f'<div style="color:#64748b;font-size:0.78rem;margin-top:4px">'
                                f'Model pick: <strong style="color:{"#4ade80" if hit else "#f87171"}">{row["model_pick"]}</strong>'
                                f' at {conf:.0f}% confidence'
                                f'{"  ·  " + str(row["loser_flags"]) + " flag" if row.get("loser_flags") and str(row.get("loser_flags")) not in ("", "nan") else ""}'
                                f'{"  ·  " + str(row["winner_flags"]) + " flag" if row.get("winner_flags") and str(row.get("winner_flags")) not in ("", "nan") and row.get("winner_flags") != row.get("loser_flags") else ""}'
                                f'</div></div>', unsafe_allow_html=True)

                            # Narrative
                            if narrative:
                                st.markdown(
                                    f'<div style="background:#131820;border-radius:6px;padding:10px 14px;margin-bottom:10px;'
                                    f'color:#e2e8f0;font-size:0.88rem;line-height:1.55;border-left:2px solid #f97316">'
                                    f'<div style="color:#f97316;font-size:0.68rem;font-weight:700;text-transform:uppercase;margin-bottom:5px">📖 What Statlas Learned</div>'
                                    f'{narrative}'
                                    f'</div>', unsafe_allow_html=True)

                            # Key stat comparison
                            stat_rows = [
                                ("contender_score", "Contender Score"),
                                ("clutch_score",    "Clutch Score"),
                                ("defense_score",   "Defense Score"),
                                ("adj_margin",      "Adj Margin"),
                            ]
                            feat_lkp2 = {str(r2["team"]): r2 for _, r2 in in_bracket.iterrows()}
                            w_row2 = feat_lkp2.get(str(row["winner"]), {})
                            l_row2 = feat_lkp2.get(str(row["loser"]), {})
                            stat_md = []
                            for sk, sl in stat_rows:
                                wv = safe_f(w_row2.get(sk)); lv = safe_f(l_row2.get(sk))
                                if wv or lv:
                                    delta = wv - lv
                                    color = "#4ade80" if delta > 0 else "#f87171"
                                    stat_md.append(
                                        f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b">'
                                        f'<span style="color:#94a3b8;font-size:0.78rem">{sl}</span>'
                                        f'<span style="font-size:0.78rem">'
                                        f'<span style="color:#4ade80">{wv:.1f}</span>'
                                        f' vs <span style="color:#f87171">{lv:.1f}</span>'
                                        f' <span style="color:{color};font-weight:700">({delta:+.1f})</span>'
                                        f'</span></div>'
                                    )
                            if stat_md:
                                st.markdown(
                                    f'<div style="margin-top:4px">'
                                    f'<div style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;margin-bottom:4px">'
                                    f'{row["winner"]} vs {row["loser"]} — Key Stats</div>'
                                    + "".join(stat_md) + '</div>', unsafe_allow_html=True)

        # ── SECTION C: Model Intelligence ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### Model Intelligence Report")

        with st.expander("🚩 Flag Report Card — Did the model's warnings come true?", expanded=False):
            flag_types = [
                ("Fraud Fav",  "Fraud Favorites",    "Overseeded teams flagged by Statlas — did they lose early?"),
                ("Cinderella", "Cinderella Teams",   "Low seeds with upset potential — did they deliver?"),
                ("Dangerous",  "Dangerous Low Seeds","Double-digit seeds with elite metrics — did they upset?"),
                ("Underseeded","Underseeded Teams",  "Teams seeded lower than metrics — did they outperform?"),
            ]
            any_shown = False
            for flag_key, flag_label, flag_desc in flag_types:
                flagged_w = recap_df[recap_df["winner_flags"].str.contains(flag_key, na=False)]
                flagged_l = recap_df[recap_df["loser_flags"].str.contains(flag_key, na=False)]
                n_total = len(set(flagged_w["winner"].tolist() + flagged_l["loser"].tolist()))
                if n_total == 0:
                    continue
                any_shown = True
                delivered = len(flagged_l) if flag_key == "Fraud Fav" else len(flagged_w)
                pct = delivered / n_total * 100 if n_total else 0
                bar_color = "#4ade80" if pct >= 60 else "#fbbf24" if pct >= 40 else "#f87171"
                st.markdown(
                    f'<div style="margin-bottom:12px">'
                    f'<div style="color:#f1f5f9;font-weight:700">{flag_label} '
                    f'<span style="color:#64748b;font-size:0.78rem">— {flag_desc}</span></div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:8px;margin:5px 0">'
                    f'<div style="width:{min(pct,100):.0f}%;background:{bar_color};height:8px;border-radius:4px"></div></div>'
                    f'<div style="color:{bar_color};font-size:0.8rem">{delivered}/{n_total} delivered ({pct:.0f}%)</div>'
                    f'</div>', unsafe_allow_html=True)
            if not any_shown:
                st.caption("Flag analysis will appear once flagged teams have played.")

        with st.expander("📊 What's Predictive — Feature Analysis", expanded=False):
            feat_cols = [
                ("contender_score", "Contender Score"),
                ("clutch_score",    "Clutch Score"),
                ("defense_score",   "Defense Score"),
                ("guard_play_score","Guard Play"),
                ("adj_margin",      "Adj. Margin"),
                ("last10_win_pct",  "Recent Form (L10)"),
                ("rebounding_score","Rebounding"),
            ]
            feat_diffs = []
            for fk, fl in feat_cols:
                if fk not in in_bracket.columns:
                    continue
                flkp = {str(r["team"]): safe_f(r.get(fk)) for _, r in in_bracket.iterrows()}
                diffs = [flkp.get(r["winner"], 0) - flkp.get(r["loser"], 0)
                         for _, r in recap_df.iterrows()
                         if flkp.get(r["winner"]) is not None and flkp.get(r["loser"]) is not None]
                if not diffs:
                    continue
                avg_d = sum(diffs) / len(diffs)
                pct_pos = sum(1 for d in diffs if d > 0) / len(diffs) * 100
                feat_diffs.append((fl, avg_d, pct_pos, len(diffs)))
            feat_diffs.sort(key=lambda x: x[2], reverse=True)
            for fl, avg_d, pct_pos, n in feat_diffs:
                bar_w = min(int(abs(pct_pos - 50) * 2), 100)
                bc = "#4ade80" if pct_pos >= 60 else "#f87171" if pct_pos <= 40 else "#94a3b8"
                st.markdown(
                    f'<div style="margin-bottom:7px">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<span style="color:#f1f5f9;font-size:0.83rem;font-weight:600">{fl}</span>'
                    f'<span style="color:{bc};font-size:0.77rem">{pct_pos:.0f}% winner higher · avg {avg_d:+.1f} (n={n})</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:3px;height:5px;margin-top:3px">'
                    f'<div style="width:{bar_w}%;background:{bc};height:5px;border-radius:3px"></div></div>'
                    f'</div>', unsafe_allow_html=True)


        with st.expander("💥 Biggest Misses", expanded=False):
            misses = recap_df[~recap_df["correct"]].sort_values("model_conf", ascending=False).head(5)
            if misses.empty:
                st.success("Statlas has been perfect so far!")
            else:
                for _, r in misses.iterrows():
                    st.markdown(
                        f'<div style="background:#1a0a0a;border-left:3px solid #f87171;border-radius:5px;padding:8px 12px;margin-bottom:5px">'
                        f'<div style="color:#f87171;font-weight:700">❌ {r["winner"]} def. {r["model_pick"]}{"  (UPSET)" if r["upset"] else ""}</div>'
                        f'<div style="color:#94a3b8;font-size:0.78rem">Model was {r["model_conf"]*100:.0f}% confident · #{int(r["winner_seed"])} over #{int(r["loser_seed"])}'
                        f'{"  ·  " + str(r["loser_flags"]) if r.get("loser_flags") and str(r.get("loser_flags")) not in ("", "nan") else ""}</div>'
                        f'<div style="color:#64748b;font-size:0.75rem;margin-top:3px">{str(r.get("narrative",""))}</div>'
                        f'</div>', unsafe_allow_html=True)

        with st.expander("🎯 Clutch Calls — Statlas Best Picks", expanded=False):
            hits = recap_df[recap_df["correct"]].sort_values("model_conf", ascending=False).head(5)
            if hits.empty:
                st.caption("No correct predictions logged yet.")
            else:
                for _, r in hits.iterrows():
                    st.markdown(
                        f'<div style="background:#0a1a0a;border-left:3px solid #4ade80;border-radius:5px;padding:8px 12px;margin-bottom:5px">'
                        f'<div style="color:#4ade80;font-weight:700">✅ {r["winner"]} def. {r["loser"]}</div>'
                        f'<div style="color:#94a3b8;font-size:0.78rem">Confidence: {r["model_conf"]*100:.0f}% · Seed {int(r["winner_seed"])} over {int(r["loser_seed"])}'
                        f'{"  ·  " + str(r["winner_hot"]) if r.get("winner_hot") and str(r.get("winner_hot")) not in ("", "nan") else ""}</div>'
                        f'<div style="color:#64748b;font-size:0.75rem;margin-top:3px">{str(r.get("narrative",""))}</div>'
                        f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — TOSSUP LAB
# ─────────────────────────────────────────────────────────────────────────────
with tab9:
    st.markdown('<div class="section-header">🔬 TOSSUP LAB — Close Game Intelligence</div>',
                unsafe_allow_html=True)
    _tl_hdr1, _tl_hdr2 = st.columns([5, 1])
    with _tl_hdr1:
        st.caption("The model is a blowout predictor — this lab helps you call the coin flips.")
    with _tl_hdr2:
        if st.button("🔄 Refresh", key="tossup_refresh"):
            st.cache_data.clear()
            st.session_state.pop("_recap_df_cache", None)
            st.rerun()

    # ── Recompute bracket data using cached functions ────────────────────────
    _tl_team_set = frozenset(in_bracket["team"].tolist())
    _tl_all_matchups = build_round_matchups(in_bracket)
    _tl_recap = load_or_update_results(_tl_team_set, in_bracket, _tl_all_matchups)
    _tl_adapted = _compute_adapted_scores(in_bracket, _tl_recap)
    _tl_rounds = build_bracket_live(in_bracket, _tl_recap, _tl_adapted)

    # ── Identify tossups (model confidence ≤ 65%) ────────────────────────────
    _TOSSUP_THRESH = 0.65
    _tl_upcoming = []
    _tl_completed = []
    for rnd_name, games in _tl_rounds.items():
        for g in games:
            wp = g.get("winner_p", 1.0)
            if wp > _TOSSUP_THRESH:
                continue
            g["_round"] = rnd_name
            if g.get("completed"):
                _tl_completed.append(g)
            else:
                _tl_upcoming.append(g)

    # ── Summary metrics ──────────────────────────────────────────────────────
    _tl_comp_correct = sum(1 for g in _tl_completed if g.get("model_correct"))
    _tl_comp_total = len(_tl_completed)
    _tl_comp_pct = (_tl_comp_correct / _tl_comp_total * 100) if _tl_comp_total else 0
    _mc1, _mc2, _mc3 = st.columns(3)
    _mc1.metric("🎯 Active Tossups", len(_tl_upcoming))
    _mc2.metric("✅ Completed Tossups", _tl_comp_total)
    _mc3.metric("📊 Tossup Accuracy", f"{_tl_comp_correct}-{_tl_comp_total - _tl_comp_correct} ({_tl_comp_pct:.0f}%)" if _tl_comp_total else "—")

    # ── Section A: Active Tossups ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Active Tossups")
    if not _tl_upcoming:
        st.info("No upcoming tossup matchups detected (all model picks > 65% confidence).")
    else:
        for _tg in sorted(_tl_upcoming, key=lambda x: x.get("winner_p", 0.5)):
            _t1, _t2 = _tg["t1"], _tg["t2"]
            _s1, _s2 = _tg.get("s1", 0), _tg.get("s2", 0)
            _wp = _tg.get("winner_p", 0.5)
            _mw = _tg.get("model_winner", _t1)
            _rnd = _tg.get("_round", "")
            _p1 = _wp if _mw == _t1 else (1 - _wp)

            with st.expander(f"**{_rnd}** — #{_s1} {_t1} vs #{_s2} {_t2}  ·  {_wp*100:.0f}% conf", expanded=True):
                # Team lookup
                _tr1 = in_bracket[in_bracket["team"] == _t1]
                _tr2 = in_bracket[in_bracket["team"] == _t2]
                _cs1 = safe_f(_tr1.iloc[0].get("contender_score")) if len(_tr1) else 0
                _cs2 = safe_f(_tr2.iloc[0].get("contender_score")) if len(_tr2) else 0

                # Flags
                _f1_parts = []
                _f2_parts = []
                if len(_tr1):
                    r = _tr1.iloc[0]
                    if r.get("dangerous_low_seed_flag"): _f1_parts.append("💥 Dangerous")
                    if r.get("fraud_favorite_flag"): _f1_parts.append("🃏 Fraud Fav")
                    if r.get("cinderella_flag"): _f1_parts.append("🪄 Cinderella")
                    if r.get("underseeded_flag"): _f1_parts.append("📈 Underseeded")
                if len(_tr2):
                    r = _tr2.iloc[0]
                    if r.get("dangerous_low_seed_flag"): _f2_parts.append("💥 Dangerous")
                    if r.get("fraud_favorite_flag"): _f2_parts.append("🃏 Fraud Fav")
                    if r.get("cinderella_flag"): _f2_parts.append("🪄 Cinderella")
                    if r.get("underseeded_flag"): _f2_parts.append("📈 Underseeded")

                # Team cards
                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    _hl1 = hot_label(_tr1.iloc[0]) if len(_tr1) else ""
                    _badge1 = "🤖 Model Pick" if _mw == _t1 else ""
                    st.markdown(_team_card_html(
                        f"#{_s1} {_t1}", f"{_cs1:.1f}", _hl1,
                        " · ".join(_f1_parts) if _f1_parts else "No flags",
                        border="#3b82f6" if _mw == _t1 else "#374151",
                        badge=_badge1
                    ), unsafe_allow_html=True)
                    st.markdown(_prob_bar_html(_p1, "#3b82f6"), unsafe_allow_html=True)
                    st.caption(f"Moneyline: {american_line(_p1)}")

                with _tc2:
                    _hl2 = hot_label(_tr2.iloc[0]) if len(_tr2) else ""
                    _badge2 = "🤖 Model Pick" if _mw == _t2 else ""
                    st.markdown(_team_card_html(
                        f"#{_s2} {_t2}", f"{_cs2:.1f}", _hl2,
                        " · ".join(_f2_parts) if _f2_parts else "No flags",
                        border="#3b82f6" if _mw == _t2 else "#374151",
                        badge=_badge2
                    ), unsafe_allow_html=True)
                    st.markdown(_prob_bar_html(1 - _p1, "#3b82f6"), unsafe_allow_html=True)
                    st.caption(f"Moneyline: {american_line(1 - _p1)}")

                # Tossup Scorecard
                st.markdown("#### 📊 Tossup Scorecard")
                _sc = compute_tossup_scorecard(_t1, _t2, in_bracket)
                if _sc is None:
                    st.warning("Could not compute scorecard — team data missing.")
                else:
                    _bar_html = ""
                    for m in _sc["metrics"]:
                        hb = None
                        for _nm, _key, _hb, _w in _TOSSUP_METRICS:
                            if _nm == m["name"]:
                                hb = _hb
                                break
                        _bar_html += _tossup_metric_bar_html(
                            m["name"], m["t1_val"], m["t2_val"], _t1, _t2, higher_better=hb
                        )
                    st.markdown(
                        f'<div style="background:#0f1419;border-radius:8px;padding:12px;margin:8px 0">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;padding-bottom:6px;border-bottom:2px solid #334155">'
                        f'<span style="color:#e2e8f0;font-weight:700">{_t1}</span>'
                        f'<span style="color:#64748b;font-size:0.85rem">Metric</span>'
                        f'<span style="color:#e2e8f0;font-weight:700">{_t2}</span>'
                        f'</div>'
                        f'{_bar_html}'
                        f'<div style="margin-top:8px;padding-top:6px;border-top:2px solid #334155;display:flex;justify-content:space-between">'
                        f'<span style="color:#4ade80;font-weight:700">{_t1}: {_sc["t1_adv"]} edges (wt {_sc["t1_wt"]})</span>'
                        f'<span style="color:#4ade80;font-weight:700">{_t2}: {_sc["t2_adv"]} edges (wt {_sc["t2_wt"]})</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Coach experience + clutch/form
                    _cx1, _cx2, _cx3 = st.columns(3)
                    with _cx1:
                        st.markdown("**🎓 Coach Exp**")
                        st.caption(f"{_t1}: {_sc['t1_coach']} NCAA games")
                        st.caption(f"{_t2}: {_sc['t2_coach']} NCAA games")
                        if _sc["coach_edge"]:
                            st.caption(f"Edge: {_sc['coach_edge']}")
                    with _cx2:
                        st.markdown("**🧊 Clutch Score**")
                        st.caption(f"{_t1}: {_sc['t1_clutch']:.1f}")
                        st.caption(f"{_t2}: {_sc['t2_clutch']:.1f}")
                    with _cx3:
                        st.markdown("**📈 Last 10 Win%**")
                        st.caption(f"{_t1}: {_sc['t1_last10']*100:.0f}%")
                        st.caption(f"{_t2}: {_sc['t2_last10']*100:.0f}%")

                    # Matchup insight
                    _mi = style_matchup_insight_by_name(_t1, _t2, in_bracket)
                    if _mi:
                        st.markdown(f"**🔍 Matchup Insight:** {_mi}")

                    # STATLAS LEAN
                    st.markdown("---")
                    _lean_team, _lean_reason = generate_statlas_lean(_t1, _t2, _sc, _mw, _wp, in_bracket)
                    if _lean_team == "COIN FLIP":
                        _lean_color = "#f59e0b"
                        _lean_icon = "🪙"
                    elif "HIGH" in _lean_reason:
                        _lean_color = "#4ade80"
                        _lean_icon = "🎯"
                    elif "SPLIT" in _lean_reason:
                        _lean_color = "#f97316"
                        _lean_icon = "⚠️"
                    else:
                        _lean_color = "#60a5fa"
                        _lean_icon = "📊"
                    st.markdown(
                        f'<div style="background:#0a1520;border:2px solid {_lean_color};border-radius:10px;padding:14px;margin-top:8px">'
                        f'<div style="color:{_lean_color};font-size:1.1rem;font-weight:800">'
                        f'{_lean_icon} STATLAS LEAN: {_lean_team}</div>'
                        f'<div style="color:#cbd5e1;margin-top:6px">{_lean_reason}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Section B: Completed Tossups ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Completed Tossups")
    if not _tl_completed:
        st.info("No completed tossup games yet.")
    else:
        st.markdown(f"**Record on tossups:** {_tl_comp_correct}-{_tl_comp_total - _tl_comp_correct}"
                    f" ({_tl_comp_pct:.0f}%)")
        for _cg in _tl_completed:
            _ct1, _ct2 = _cg["t1"], _cg["t2"]
            _cw = _cg.get("winner", "")
            _cmw = _cg.get("model_winner", "")
            _cwp = _cg.get("winner_p", 0.5)
            _cmc = _cg.get("model_correct", False)
            _crnd = _cg.get("_round", "")
            _cs1, _cs2 = _cg.get("s1", 0), _cg.get("s2", 0)
            _icon = "✅" if _cmc else "❌"
            _bg = "#0a1a0a" if _cmc else "#1a0a0a"
            _bc = "#4ade80" if _cmc else "#f87171"

            # Compute scorecard for completed game
            _csc = compute_tossup_scorecard(_ct1, _ct2, in_bracket)
            _sc_correct = False
            if _csc:
                _sc_leader = _ct1 if _csc["t1_wt"] > _csc["t2_wt"] else (
                    _ct2 if _csc["t2_wt"] > _csc["t1_wt"] else "")
                _sc_correct = (_sc_leader == _cw)

            st.markdown(
                f'<div style="background:{_bg};border-left:3px solid {_bc};border-radius:5px;padding:8px 12px;margin-bottom:5px">'
                f'<div style="color:{_bc};font-weight:700">{_icon} {_crnd}: #{_cs1} {_ct1} vs #{_cs2} {_ct2} — '
                f'Winner: {_cw}</div>'
                f'<div style="color:#94a3b8;font-size:0.78rem">'
                f'Model pick: {_cmw} ({_cwp*100:.0f}%) · '
                f'Scorecard {"✅ agreed" if _sc_correct else "❌ missed"}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Section C: Tossup Intelligence ───────────────────────────────────────
    if _tl_completed:
        with st.expander("🧠 Tossup Intelligence — Which Metrics Predicted Winners?", expanded=False):
            _metric_hits = {m[0]: 0 for m in _TOSSUP_METRICS if m[2] is not None}
            _metric_total = {m[0]: 0 for m in _TOSSUP_METRICS if m[2] is not None}
            for _cg in _tl_completed:
                _csc = compute_tossup_scorecard(_cg["t1"], _cg["t2"], in_bracket)
                if not _csc:
                    continue
                _cw = _cg.get("winner", "")
                for m in _csc["metrics"]:
                    if m["name"] not in _metric_hits:
                        continue
                    _metric_total[m["name"]] += 1
                    if m.get("edge_team") == _cw:
                        _metric_hits[m["name"]] += 1

            st.markdown("Which of the 6 scorecard metrics correctly predicted the winner in completed tossups:")
            for mname in _metric_hits:
                _mh = _metric_hits[mname]
                _mt = _metric_total[mname]
                _mpct = (_mh / _mt * 100) if _mt else 0
                _bar_w = int(_mpct)
                _bar_c = "#4ade80" if _mpct >= 60 else ("#f59e0b" if _mpct >= 40 else "#f87171")
                st.markdown(
                    f'<div style="margin-bottom:6px">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
                    f'<span style="color:#e2e8f0;font-weight:600">{mname}</span>'
                    f'<span style="color:{_bar_c};font-weight:700">{_mh}/{_mt} ({_mpct:.0f}%)</span>'
                    f'</div>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px">'
                    f'<div style="width:{_bar_w}%;background:{_bar_c};height:6px;border-radius:4px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 10 — SCORE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab10:
    st.markdown('<div class="section-header">🎯 SCORE PREDICTOR — Final Score Intelligence</div>',
                unsafe_allow_html=True)
    _sp_hdr1, _sp_hdr2 = st.columns([5, 1])
    with _sp_hdr1:
        st.caption("Model picks the winner · Regression sets the total · Click any matchup in the Bracket tab to predict scores")
    with _sp_hdr2:
        if st.button("🔄 Refresh", key="predict_refresh"):
            st.cache_data.clear()
            st.session_state.pop("_recap_df_cache", None)
            st.rerun()

    # ── Recompute bracket data ───────────────────────────────────────────
    _sp_team_set = frozenset(in_bracket["team"].tolist())
    _sp_all_matchups = build_round_matchups(in_bracket)
    _sp_recap = load_or_update_results(_sp_team_set, in_bracket, _sp_all_matchups)
    _sp_adapted = _compute_adapted_scores(in_bracket, _sp_recap)
    _sp_rounds = build_bracket_live(in_bracket, _sp_recap, _sp_adapted)

    # ── Team selection ───────────────────────────────────────────────────
    _sp_teams = sorted(in_bracket["team"].tolist())
    _sp_sel1, _sp_sel2 = st.columns(2)
    with _sp_sel1:
        _sp_def1 = _sp_teams.index(st.session_state.predict_t1) if st.session_state.predict_t1 in _sp_teams else 0
        _sp_t1 = st.selectbox("Team 1", _sp_teams, index=_sp_def1, key="sp_team1")
    with _sp_sel2:
        _sp_def2 = _sp_teams.index(st.session_state.predict_t2) if st.session_state.predict_t2 in _sp_teams else min(1, len(_sp_teams) - 1)
        _sp_t2 = st.selectbox("Team 2", _sp_teams, index=_sp_def2, key="sp_team2")

    # Clear navigation state after consumption
    if st.session_state.predict_t1:
        st.session_state.predict_t1 = None
        st.session_state.predict_t2 = None

    if _sp_t1 == _sp_t2:
        st.warning("Select two different teams to predict a score.")
    else:
        pred = predict_final_score(_sp_t1, _sp_t2, in_bracket)
        if pred is None:
            st.warning("Insufficient data for score prediction.")
        else:
            # ── Scoreboard display ───────────────────────────────────────
            _tr1 = in_bracket[in_bracket["team"] == _sp_t1]
            _tr2 = in_bracket[in_bracket["team"] == _sp_t2]
            _s1 = safe_i(_tr1.iloc[0].get("seed", 0)) if len(_tr1) else 0
            _s2 = safe_i(_tr2.iloc[0].get("seed", 0)) if len(_tr2) else 0
            _t1_sc = pred["t1_score"]
            _t2_sc = pred["t2_score"]
            _t1_color = "#4ade80" if _t1_sc > _t2_sc else ("#f87171" if _t1_sc < _t2_sc else "#f1f5f9")
            _t2_color = "#4ade80" if _t2_sc > _t1_sc else ("#f87171" if _t2_sc < _t1_sc else "#f1f5f9")

            st.markdown(
                f'<div style="background:#131820;border:2px solid #f97316;border-radius:12px;'
                f'padding:20px;text-align:center;margin:10px 0">'
                f'<div style="display:flex;justify-content:center;align-items:center;gap:40px">'
                f'<div>'
                f'<div style="color:#94a3b8;font-size:0.8rem">#{_s1} seed</div>'
                f'<div style="color:#f1f5f9;font-weight:800;font-size:1.2rem">{_sp_t1}</div>'
                f'<div style="color:{_t1_color};font-weight:900;font-size:2.5rem">{_t1_sc:.0f}</div>'
                f'</div>'
                f'<div style="color:#64748b;font-size:1.5rem;font-weight:300">—</div>'
                f'<div>'
                f'<div style="color:#94a3b8;font-size:0.8rem">#{_s2} seed</div>'
                f'<div style="color:#f1f5f9;font-weight:800;font-size:1.2rem">{_sp_t2}</div>'
                f'<div style="color:{_t2_color};font-weight:900;font-size:2.5rem">{_t2_sc:.0f}</div>'
                f'</div>'
                f'</div>'
                f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:12px">'
                f'Projected possessions: {pred["possessions"]:.0f} · '
                f'Method: {"Model + Regression" if pred["method"] == "regression_calibrated" else "Efficiency-Adjusted" if pred["method"] == "efficiency_adjusted" else "PPG Average"}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Betting lines & confidence ───────────────────────────────
            # Derive win probability from the predicted spread so that
            # spread, total, win-prob, and moneyline are all consistent.
            # k=0.15 is basketball-calibrated: 7-pt spread ≈ 74% win prob.
            _wp = 1.0 / (1.0 + np.exp(-pred["spread"] * 0.15))
            _spread = pred["spread"]
            if _spread > 0:
                _spread_str = f"{_sp_t1} -{abs(_spread):.1f}"
            elif _spread < 0:
                _spread_str = f"{_sp_t2} -{abs(_spread):.1f}"
            else:
                _spread_str = "PICK"

            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            with _sc1:
                st.metric("📊 Spread", _spread_str)
            with _sc2:
                st.metric("📈 Total (O/U)", f"{pred['total']:.1f}")
            with _sc3:
                _wp_team = _sp_t1 if _wp >= 0.5 else _sp_t2
                _wp_pct = _wp if _wp >= 0.5 else (1 - _wp)
                st.metric("🎲 Win Prob", f"{_wp_pct*100:.0f}% {_wp_team}")
            with _sc4:
                _ml_team = _sp_t1 if _wp >= 0.5 else _sp_t2
                _ml_val = american_line(_wp if _wp >= 0.5 else (1 - _wp))
                st.metric("💰 Moneyline", f"{_ml_team} {_ml_val}")

            # Confidence range
            cr = pred["confidence_range"]
            st.markdown(
                f'<div style="background:#1e293b;border-radius:8px;padding:10px;margin:10px 0">'
                f'<div style="color:#94a3b8;font-size:0.85rem;font-weight:600">Confidence Range (68%)</div>'
                f'<div style="color:#f1f5f9;font-size:0.9rem">'
                f'{_sp_t1}: {max(0, _t1_sc - cr):.0f} — {_t1_sc + cr:.0f} · '
                f'{_sp_t2}: {max(0, _t2_sc - cr):.0f} — {_t2_sc + cr:.0f}'
                f'</div></div>',
                unsafe_allow_html=True)

            # ── Efficiency breakdown ─────────────────────────────────────
            if pred["method"] in ("efficiency_adjusted", "regression_calibrated"):
                st.markdown("### 📊 Efficiency Breakdown")
                _ef1, _ef2 = st.columns(2)
                with _ef1:
                    st.markdown(f"**{_sp_t1}**")
                    _em1a, _em1b, _em1c = st.columns(3)
                    _em1a.metric("Adj. Offense", f"{pred['t1_off']:.1f}")
                    _em1b.metric("Adj. Defense", f"{pred['t1_def']:.1f}")
                    _em1c.metric("Tempo", f"{pred['t1_tempo']:.1f}" if pred["t1_tempo"] > 0 else "N/A")
                with _ef2:
                    st.markdown(f"**{_sp_t2}**")
                    _em2a, _em2b, _em2c = st.columns(3)
                    _em2a.metric("Adj. Offense", f"{pred['t2_off']:.1f}")
                    _em2b.metric("Adj. Defense", f"{pred['t2_def']:.1f}")
                    _em2c.metric("Tempo", f"{pred['t2_tempo']:.1f}" if pred["t2_tempo"] > 0 else "N/A")

            # ── Matchup factors comparison bars ──────────────────────────
            st.markdown("### 🔬 Matchup Factors")
            _factor_metrics = [
                ("3-Point %", "three_pt_pct", True),
                ("FT %", "ft_pct", True),
                ("Eff. FG%", "eff_fg_pct", True),
                ("Off. Reb %", "off_rebound_pct", True),
                ("Def. Reb %", "def_rebound_pct", True),
                ("Turnover %", "turnover_pct", False),
                ("Forced TO %", "opp_turnover_pct", True),
                ("Last 10 Win%", "last10_win_pct", True),
            ]
            _factor_html = ""
            for fname, fkey, higher_better in _factor_metrics:
                v1 = safe_f(_tr1.iloc[0].get(fkey)) if len(_tr1) else 0
                v2 = safe_f(_tr2.iloc[0].get(fkey)) if len(_tr2) else 0
                if v1 == 0 and v2 == 0:
                    continue
                _factor_html += _tossup_metric_bar_html(
                    fname, v1, v2, _sp_t1, _sp_t2, higher_better=higher_better)
            if _factor_html:
                st.markdown(
                    f'<div style="background:#0f1419;border-radius:8px;padding:12px;margin:8px 0">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;'
                    f'padding-bottom:6px;border-bottom:2px solid #334155">'
                    f'<span style="color:#e2e8f0;font-weight:700">{_sp_t1}</span>'
                    f'<span style="color:#64748b;font-size:0.85rem">Metric</span>'
                    f'<span style="color:#e2e8f0;font-weight:700">{_sp_t2}</span>'
                    f'</div>'
                    f'{_factor_html}'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── Style matchup insight ────────────────────────────────────
            _mi = style_matchup_insight_by_name(_sp_t1, _sp_t2, in_bracket)
            if _mi:
                st.markdown(
                    f'<div style="background:#1a2a1a;border-left:3px solid #f97316;padding:8px 12px;'
                    f'border-radius:4px;margin:10px 0;color:#f1f5f9;font-weight:600">'
                    f'🔍 {_mi}</div>',
                    unsafe_allow_html=True)

            # ── Flags & momentum ─────────────────────────────────────────
            _flag_parts = []
            for _tn, _tr in [(_sp_t1, _tr1), (_sp_t2, _tr2)]:
                if len(_tr):
                    r = _tr.iloc[0]
                    if r.get("dangerous_low_seed_flag"): _flag_parts.append(f"💥 {_tn}: Dangerous")
                    if r.get("fraud_favorite_flag"): _flag_parts.append(f"🃏 {_tn}: Fraud Fav")
                    if r.get("cinderella_flag"): _flag_parts.append(f"🪄 {_tn}: Cinderella")
                    hl = hot_label(r)
                    if hl: _flag_parts.append(f"{_tn}: {hl}")
            if _flag_parts:
                st.markdown(
                    f'<div style="background:#1e293b;border-radius:6px;padding:8px 12px;margin:8px 0;'
                    f'color:#f59e0b;font-size:0.85rem">{"  ·  ".join(_flag_parts)}</div>',
                    unsafe_allow_html=True)

    # ── Model Calibration: Predicted vs Actual ───────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Calibration — Predicted vs Actual")
    st.caption("How well the efficiency model predicts actual final scores in completed tournament games.")

    calibration = compute_prediction_accuracy(in_bracket, _sp_recap)
    if not calibration:
        st.info("No completed games yet for calibration.")
    else:
        _cal_correct = sum(1 for c in calibration if c["winner_correct"])
        _cal_total = len(calibration)
        _avg_spread_err = np.mean([c["spread_error"] for c in calibration])
        _avg_total_err = np.mean([c["total_error"] for c in calibration])

        _cm1, _cm2, _cm3, _cm4 = st.columns(4)
        _cm1.metric("Winner Correct", f"{_cal_correct}/{_cal_total}", f"{_cal_correct/_cal_total*100:.0f}%")
        _cm2.metric("Avg Spread Error", f"{_avg_spread_err:.1f} pts")
        _cm3.metric("Avg Total Error", f"{_avg_total_err:.1f} pts")
        _cm4.metric("Games Analyzed", _cal_total)

        with st.expander("📋 Game-by-Game Breakdown", expanded=False):
            for c in sorted(calibration, key=lambda x: x["spread_error"], reverse=True):
                _correct_icon = "✅" if c["winner_correct"] else "❌"
                _bg = "#0a1a0a" if c["winner_correct"] else "#1a0a0a"
                _bc = "#4ade80" if c["winner_correct"] else "#f87171"
                st.markdown(
                    f'<div style="background:{_bg};border-left:3px solid {_bc};border-radius:5px;'
                    f'padding:8px 12px;margin:4px 0">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<div>'
                    f'<div style="color:#f1f5f9;font-weight:700;font-size:0.85rem">'
                    f'{_correct_icon} {c["t1"]} vs {c["t2"]} ({c["round"]})</div>'
                    f'<div style="color:#94a3b8;font-size:0.78rem">'
                    f'Predicted: {c["pred_t1"]:.0f}-{c["pred_t2"]:.0f} · '
                    f'Actual: {c["actual_t1"]}-{c["actual_t2"]}</div>'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<div style="color:#f59e0b;font-weight:600;font-size:0.85rem">'
                    f'Spread err: {c["spread_error"]:.1f}</div>'
                    f'<div style="color:#64748b;font-size:0.75rem">'
                    f'Total err: {c["total_error"]:.1f}</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True)
