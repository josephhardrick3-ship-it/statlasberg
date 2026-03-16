"""Statlasberg — 2026 March Madness Intelligence Platform"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import os

st.set_page_config(
    page_title="Statlasberg",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #8892a4 !important; font-size: 0.75rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-size: 1.6rem; font-weight: 700;
    }

    /* Section headers */
    .section-header {
        color: #f97316;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 6px 0;
        border-bottom: 2px solid #f97316;
        margin-bottom: 12px;
    }

    /* Team card */
    .team-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
    }
    .team-rank { color: #f97316; font-size: 1.4rem; font-weight: 800; }
    .team-name { color: #e2e8f0; font-size: 1.1rem; font-weight: 700; }
    .team-score { color: #48bb78; font-size: 1.0rem; }

    /* Seed badge */
    .seed-badge {
        display: inline-block;
        background: #f97316;
        color: white;
        border-radius: 50%;
        width: 26px; height: 26px;
        text-align: center;
        line-height: 26px;
        font-weight: 800;
        font-size: 0.8rem;
        margin-right: 8px;
    }

    /* Flag badges */
    .flag-cinderella { background:#f59e0b; color:#000; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:700; }
    .flag-fraud      { background:#ef4444; color:#fff; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:700; }
    .flag-dark       { background:#8b5cf6; color:#fff; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:700; }
    .flag-upset      { background:#f97316; color:#fff; padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:700; }

    /* Progress bar override */
    .stProgress > div > div { background-color: #f97316; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e;
        border-radius: 8px 8px 0 0;
        color: #8892a4;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] { background: #f97316 !important; color: white !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #2d3748; }
    [data-testid="stSidebar"] .stSelectbox label { color: #8892a4; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
SCORES_PATH  = "data/outputs/team_scores.csv"
CHAMPS_PATH  = "data/outputs/simulation_results.csv"
BRACKET_PATH = "data/brackets/bracket_2026.csv"

@st.cache_data
def load_data():
    scores  = pd.read_csv(SCORES_PATH)  if os.path.exists(SCORES_PATH)  else pd.DataFrame()
    champs  = pd.read_csv(CHAMPS_PATH)  if os.path.exists(CHAMPS_PATH)  else pd.DataFrame()
    bracket = pd.read_csv(BRACKET_PATH) if os.path.exists(BRACKET_PATH) else pd.DataFrame()
    return scores, champs, bracket

scores, champs, bracket = load_data()

if len(scores) == 0:
    st.error("⚠️ No scores found. Run `python run_pipeline.py --csv data/raw/teams/team_stats_2026.csv --bracket data/brackets/bracket_2026.csv --simulate` first.")
    st.stop()

# Merge bracket seeds into scores
if len(bracket) > 0:
    bracket_info = bracket[["team","region","seed"]].copy()
    scores = scores.merge(bracket_info, on="team", how="left")
    in_bracket = scores[scores["seed"].notna()].copy()
    in_bracket["seed"] = in_bracket["seed"].astype(int)
else:
    in_bracket = scores.copy()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 Statlasberg")
    st.markdown("*2026 March Madness Intelligence*")
    st.divider()

    bracket_teams = sorted(in_bracket["team"].tolist()) if len(in_bracket) > 0 else sorted(scores["team"].tolist())
    selected_team = st.selectbox("🔍 Team Deep Dive", bracket_teams)
    st.divider()

    # Quick model stats
    if len(in_bracket) > 0:
        st.markdown("**📊 Model Summary**")
        n1seeds = in_bracket[in_bracket["seed"]==1]["contender_score"].mean() if "seed" in in_bracket else 0
        st.metric("Avg #1 Seed Score",  f"{n1seeds:.1f}" if n1seeds else "—")
        top_upset = in_bracket[(in_bracket["seed"]>=10) & (in_bracket["seed"]<=12)].sort_values("contender_score", ascending=False)
        if len(top_upset) > 0:
            st.metric("Best Upset Threat", top_upset.iloc[0]["team"])
        st.metric("No. 1 Overall", in_bracket.sort_values("contender_score", ascending=False).iloc[0]["team"])
    st.divider()
    st.caption("Built on Selection Sunday · Mar 15, 2026")


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
with col_h1:
    st.metric("🏆 Model Favourite", champs.iloc[0]["team"] if len(champs)>0 else "—",
              f"{champs.iloc[0]['championship_pct']:.1f}%" if len(champs)>0 else "")
with col_h2:
    top = in_bracket.sort_values("contender_score", ascending=False).iloc[0] if len(in_bracket) > 0 else None
    st.metric("📈 Top Contender Score", f"{top['contender_score']:.1f}" if top is not None else "—", top["team"] if top is not None else "")
with col_h3:
    snubs = scores[(scores["contender_score"]>=65) & (~scores["team"].isin(bracket["team"])) ] if len(bracket)>0 else pd.DataFrame()
    st.metric("🚨 Big Snubs", len(snubs), "teams model loves, left out" if len(snubs)>0 else "")
with col_h4:
    upsets = in_bracket[(in_bracket["seed"]>=10) & (in_bracket["contender_score"]>=65)] if "seed" in in_bracket.columns else pd.DataFrame()
    st.metric("💥 Upset Alerts", len(upsets), "double-digit seeds w/ top scores")
with col_h5:
    avg_risk_1seeds = in_bracket[in_bracket["seed"]==1]["upset_risk_score"].mean() if "seed" in in_bracket.columns else 0
    st.metric("⚠️ Avg #1 Seed Risk", f"{avg_risk_1seeds:.1f}" if avg_risk_1seeds else "—", "lower is safer")

st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏅 Rankings", "🎯 Sweet 16 Picks", "🔍 Team Deep Dive",
    "📊 Model vs Committee", "🎲 Championship Odds"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">2026 Tournament Field — Model Rankings</div>', unsafe_allow_html=True)

    display_df = in_bracket.sort_values("contender_score", ascending=False).reset_index(drop=True)
    display_df.index += 1

    # Color map for contender score
    def score_color(v):
        if v >= 70: return "background-color: #14532d; color: #86efac"
        if v >= 60: return "background-color: #1a3a1a; color: #4ade80"
        if v >= 50: return "background-color: #1c2a1c; color: #86efac"
        return "background-color: #1a1f2e; color: #94a3b8"

    def risk_color(v):
        if v >= 40: return "background-color: #450a0a; color: #fca5a5"
        if v >= 30: return "background-color: #431407; color: #fdba74"
        return "background-color: #1a1f2e; color: #86efac"

    cols_show = ["team","region","seed","contender_score","upset_risk_score","expected_round",
                 "defense_score","clutch_score","guard_play_score","archetype"]
    cols_avail = [c for c in cols_show if c in display_df.columns]

    fmt_cols = {c: "{:.1f}" for c in ["contender_score","upset_risk_score","defense_score","clutch_score","guard_play_score"] if c in cols_avail}
    score_cols = [c for c in ["contender_score"] if c in cols_avail]
    risk_cols  = [c for c in ["upset_risk_score"] if c in cols_avail]
    styler = display_df[cols_avail].style
    if score_cols:
        styler = styler.map(score_color, subset=score_cols)
    if risk_cols:
        styler = styler.map(risk_color, subset=risk_cols)
    styled = styler.format(fmt_cols) \
        .set_properties(**{"background-color":"#0e1117","color":"#e2e8f0","border":"1px solid #2d3748"})

    st.dataframe(styled, use_container_width=True, height=600)

    # Flags summary
    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown('<div class="section-header">💥 Cinderella Candidates</div>', unsafe_allow_html=True)
        cinder = in_bracket[in_bracket.get("cinderella_flag", pd.Series(False)) == True] if "cinderella_flag" in in_bracket.columns else pd.DataFrame()
        if len(cinder) > 0:
            for _, r in cinder.iterrows():
                st.markdown(f'<span class="seed-badge">{int(r["seed"]) if "seed" in r else "?"}</span> **{r["team"]}** — {r["contender_score"]:.1f} pts', unsafe_allow_html=True)
        else:
            st.caption("None flagged this year")
    with fc2:
        st.markdown('<div class="section-header">🃏 Fraud Favorites</div>', unsafe_allow_html=True)
        frauds = in_bracket[in_bracket.get("fraud_favorite_flag", pd.Series(False)) == True] if "fraud_favorite_flag" in in_bracket.columns else pd.DataFrame()
        if len(frauds) > 0:
            for _, r in frauds.iterrows():
                st.markdown(f'<span class="seed-badge">{int(r["seed"]) if "seed" in r else "?"}</span> **{r["team"]}** — score {r["contender_score"]:.1f}', unsafe_allow_html=True)
        else:
            st.caption("No clear frauds flagged")
    with fc3:
        st.markdown('<div class="section-header">🌑 Darkhorse Alerts</div>', unsafe_allow_html=True)
        dark = in_bracket[in_bracket.get("title_darkhorse_flag", pd.Series(False)) == True] if "title_darkhorse_flag" in in_bracket.columns else pd.DataFrame()
        if len(dark) > 0:
            for _, r in dark.head(6).iterrows():
                st.markdown(f'<span class="seed-badge">{int(r["seed"]) if "seed" in r else "?"}</span> **{r["team"]}** — {r["contender_score"]:.1f}', unsafe_allow_html=True)
        else:
            st.caption("None flagged")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SWEET 16 PICKS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Model\'s Predicted Sweet 16 — Targeting 75%+ Accuracy</div>', unsafe_allow_html=True)
    st.caption("Top 4 teams per region by contender score, adjusted for seed path and upset risk.")

    if "seed" not in in_bracket.columns or len(in_bracket) == 0:
        st.warning("Bracket data not available.")
    else:
        regions = ["East","South","West","Midwest"]
        reg_cols = st.columns(4)

        all_s16_picks = []
        for i, region in enumerate(regions):
            reg_df = in_bracket[in_bracket["region"]==region].copy()
            if len(reg_df) == 0:
                continue

            # Score each team: contender_score penalised by seed difficulty
            # Seeds 1-4 get neutral, 5-8 mild bonus for potential Cinderella path,
            # 9-16 high upset risk reduces expected advance probability
            def path_weight(seed):
                if seed <= 4: return 1.0
                if seed <= 6: return 0.88
                if seed <= 8: return 0.75
                if seed <= 11: return 0.60
                return 0.45

            reg_df["advance_score"] = reg_df.apply(
                lambda r: r["contender_score"] * path_weight(r["seed"]) - r["upset_risk_score"] * 0.15, axis=1
            )
            reg_df = reg_df.sort_values("advance_score", ascending=False)
            top4 = reg_df.head(4)
            all_s16_picks.extend(top4["team"].tolist())

            with reg_cols[i]:
                st.markdown(f'<div class="section-header">{region}</div>', unsafe_allow_html=True)
                for rank, (_, row) in enumerate(top4.iterrows(), 1):
                    seed_val = int(row["seed"])
                    conf = min(99, int(row["advance_score"]))

                    # Confidence color
                    if conf >= 55: bar_col = "#48bb78"
                    elif conf >= 45: bar_col = "#f59e0b"
                    else: bar_col = "#ef4444"

                    flags = []
                    if row.get("cinderella_flag", False): flags.append("🪄 Cinderella")
                    if row.get("title_darkhorse_flag", False): flags.append("🌑 Darkhorse")
                    if seed_val >= 10: flags.append("💥 Upset Alert")

                    flag_str = " ".join(flags)
                    st.markdown(f"""
                    <div class="team-card">
                        <span class="seed-badge">{seed_val}</span>
                        <span class="team-name">{row['team']}</span><br/>
                        <small style="color:#8892a4">Score: {row['contender_score']:.1f} · Risk: {row['upset_risk_score']:.1f}</small><br/>
                        <div style="background:#2d3748;border-radius:4px;margin-top:6px;height:6px">
                            <div style="width:{min(100,conf)}%;background:{bar_col};height:6px;border-radius:4px"></div>
                        </div>
                        <small style="color:{bar_col}">{conf}% advance confidence</small>
                        {"<br/><small>" + flag_str + "</small>" if flags else ""}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"**Model's 16 Sweet 16 Picks:** {', '.join(all_s16_picks)}")

        # First-round upset picks
        st.markdown('<div class="section-header">🔥 Best First-Round Upset Picks (Seeds 10-13)</div>', unsafe_allow_html=True)
        upset_picks = in_bracket[
            (in_bracket["seed"] >= 10) & (in_bracket["seed"] <= 13)
        ].sort_values("contender_score", ascending=False).head(6)
        up_cols = st.columns(min(3, len(upset_picks)))
        for idx, (_, row) in enumerate(upset_picks.iterrows()):
            with up_cols[idx % 3]:
                matchup_seed = 17 - int(row["seed"])
                opp = in_bracket[
                    (in_bracket["region"]==row.get("region","")) & (in_bracket["seed"]==matchup_seed)
                ]
                opp_name = opp.iloc[0]["team"] if len(opp) > 0 else f"#{matchup_seed} seed"
                st.markdown(f"""
                <div class="team-card">
                    <span style="color:#f97316;font-weight:800">#{int(row['seed'])} {row['team']}</span>
                    <span style="color:#8892a4"> over </span>
                    <span style="color:#e2e8f0">#{matchup_seed} {opp_name}</span><br/>
                    <small style="color:#4ade80">Model Score: {row['contender_score']:.1f} · Risk: {row['upset_risk_score']:.1f}</small>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TEAM DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    row_data = scores[scores["team"] == selected_team]
    if len(row_data) == 0:
        st.warning(f"No data for {selected_team}")
        st.stop()
    row = row_data.iloc[0]

    # Header
    try:
        seed_str = f"#{int(row['seed'])} Seed · {row.get('region','')}" if "seed" in row.index and pd.notna(row.get("seed")) else "Model Analysis"
    except Exception:
        seed_str = "Model Analysis"
    st.markdown(f"## {selected_team}")
    st.markdown(f"*{seed_str} · {row.get('conference','')}"
                + (f" · Expected: **{row['expected_round']}**" if 'expected_round' in row else "") + "*")

    # Top metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    def safe_float(v, default=0.0):
        try: return float(v) if pd.notna(v) else default
        except: return default
    def safe_int_val(v, default=0):
        try: return int(float(v)) if pd.notna(v) else default
        except: return default
    m1.metric("Contender Score",  f"{safe_float(row.get('contender_score')):.1f}/100")
    m2.metric("Upset Risk",       f"{safe_float(row.get('upset_risk_score')):.1f}/100")
    m3.metric("Record",           f"{safe_int_val(row.get('wins'))}-{safe_int_val(row.get('losses'))}")
    m4.metric("NET Rank",         f"#{safe_int_val(row.get('net_rank'))}" if pd.notna(row.get('net_rank')) else "N/A")
    m5.metric("Adj. Margin",      f"+{safe_float(row.get('adj_margin')):.1f}" if safe_float(row.get('adj_margin'))>0 else f"{safe_float(row.get('adj_margin')):.1f}")
    m6.metric("Archetype",        row.get("archetype","—") or "—")

    st.markdown("---")
    left_col, right_col = st.columns([1, 1])

    # ── Shot Zone Chart ───────────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="section-header">🎯 Shooting Zone Profile</div>', unsafe_allow_html=True)
        st.caption("Zone colors = efficiency percentile among tournament teams. Green = elite, Red = below avg.")

        fig, ax = plt.subplots(figsize=(6, 5.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1f2e")

        # Court lines (half court, top = hoop end)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-3, 42)
        ax.set_aspect("equal")
        ax.axis("off")

        def eff_color(pct, low=30, high=40):
            """Map a percentage to green-yellow-red colormap."""
            norm = max(0, min(1, (pct - low) / (high - low)))
            r = int(255 * (1 - norm))
            g = int(200 * norm + 55)
            return f"#{r:02x}{g:02x}55"

        # ─── Paint zone ───────────────────────────────────────────────────────
        paint_eff = row.get("eff_fg_pct", 50) * 0.8  # proxy for paint efficiency
        paint_color = eff_color(paint_eff, 35, 55)
        paint = mpatches.FancyBboxPatch((-8, 0), 16, 19, boxstyle="round,pad=0.5",
                                         facecolor=paint_color, edgecolor="#e2e8f0", linewidth=1.5, alpha=0.85)
        ax.add_patch(paint)
        ax.text(0, 9.5, f"PAINT\neff {paint_eff:.0f}%", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold")

        # ─── Free throw line ──────────────────────────────────────────────────
        ft_color = eff_color(row.get("ft_pct", 70) - 30, 0, 25)
        ax.plot([-8, 8], [19, 19], color="#e2e8f0", linewidth=1.5, linestyle="--")
        ft_arc = Arc((0, 19), 16, 12, angle=0, theta1=0, theta2=180, color="#e2e8f0", linewidth=1.5)
        ax.add_patch(ft_arc)
        ft_rect = mpatches.FancyBboxPatch((-6, 18), 12, 4, boxstyle="round,pad=0.3",
                                           facecolor=ft_color, edgecolor="none", alpha=0.7)
        ax.add_patch(ft_rect)
        ax.text(0, 20.5, f"FT: {row.get('ft_pct',0)*100:.0f}%" if row.get("ft_pct",0) < 1
                else f"FT: {row.get('ft_pct',0):.0f}%", color="white", ha="center", fontsize=8)

        # ─── Mid-range zones ──────────────────────────────────────────────────
        # Implied mid-range efficiency from eff_fg_pct and 3P contribution
        mid_eff = row.get("eff_fg_pct", 50) * 0.85
        mid_col = eff_color(mid_eff, 33, 50)
        # Left mid
        lm = mpatches.FancyBboxPatch((-22, 8), 14, 16, boxstyle="round,pad=0.3",
                                      facecolor=mid_col, edgecolor="#e2e8f0", linewidth=1, alpha=0.75)
        ax.add_patch(lm)
        ax.text(-15, 15.5, f"MID\n{mid_eff:.0f}%", color="white", ha="center", fontsize=7.5, fontweight="bold")
        # Right mid
        rm = mpatches.FancyBboxPatch((8, 8), 14, 16, boxstyle="round,pad=0.3",
                                      facecolor=mid_col, edgecolor="#e2e8f0", linewidth=1, alpha=0.75)
        ax.add_patch(rm)
        ax.text(15, 15.5, f"MID\n{mid_eff:.0f}%", color="white", ha="center", fontsize=7.5, fontweight="bold")

        # ─── Three-point arc ──────────────────────────────────────────────────
        three_pct = row.get("three_pt_pct", 0.33)
        if three_pct > 1: three_pct /= 100
        three_col = eff_color(three_pct * 100, 32, 40)
        three_arc = Arc((0, 5), 44, 44, angle=0, theta1=12, theta2=168,
                        color="#e2e8f0", linewidth=2)
        ax.add_patch(three_arc)

        # Left corner 3
        lc3 = mpatches.FancyBboxPatch((-25, 0), 7, 8, boxstyle="round,pad=0.3",
                                        facecolor=three_col, edgecolor="#e2e8f0", linewidth=1, alpha=0.85)
        ax.add_patch(lc3)
        ax.text(-21.5, 3.5, f"C3\n{three_pct*100:.1f}%", color="white", ha="center", fontsize=7, fontweight="bold")

        # Right corner 3
        rc3 = mpatches.FancyBboxPatch((18, 0), 7, 8, boxstyle="round,pad=0.3",
                                        facecolor=three_col, edgecolor="#e2e8f0", linewidth=1, alpha=0.85)
        ax.add_patch(rc3)
        ax.text(21.5, 3.5, f"C3\n{three_pct*100:.1f}%", color="white", ha="center", fontsize=7, fontweight="bold")

        # Above-break 3
        ab3 = mpatches.FancyBboxPatch((-14, 26), 28, 10, boxstyle="round,pad=0.3",
                                        facecolor=three_col, edgecolor="#e2e8f0", linewidth=1, alpha=0.85)
        ax.add_patch(ab3)
        three_rate_raw = row.get("three_pa_rate", 0.35)
        three_rate = three_rate_raw / 100 if three_rate_raw > 1 else three_rate_raw
        ax.text(0, 31, f"3PT: {three_pct*100:.1f}%  Rate: {three_rate*100:.0f}%",
                color="white", ha="center", fontsize=8.5, fontweight="bold")

        # Hoop
        hoop = plt.Circle((0, 5), 0.75, color="#ff6600", fill=False, linewidth=2.5)
        ax.add_patch(hoop)
        backboard = mpatches.FancyArrowPatch((-3, 4), (3, 4), color="#e2e8f0", linewidth=2.5)
        ax.add_patch(backboard)

        ax.set_title(f"{selected_team} — Shooting Zones", color="#e2e8f0",
                     fontsize=11, fontweight="bold", pad=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Radar / DNA Chart ─────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="section-header">🧬 Team DNA — Strength Radar</div>', unsafe_allow_html=True)

        cats = ["Defense", "Offense", "Clutch", "Guard Play", "Rebounding", "Consistency"]
        vals = [
            row.get("defense_score", 50),
            row.get("efficiency_score", 50),
            row.get("clutch_score", 50),
            row.get("guard_play_score", 50),
            row.get("rebounding_score", 50),
            row.get("consistency_score", 50),
        ]

        N = len(cats)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        vals_plot = vals + [vals[0]]
        angles += angles[:1]

        fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#1a1f2e")

        ax2.plot(angles, vals_plot, linewidth=2.5, linestyle="solid", color="#f97316")
        ax2.fill(angles, vals_plot, alpha=0.25, color="#f97316")

        # Draw concentric rings
        for ring in [25, 50, 75, 100]:
            ring_vals = [ring] * N + [ring]
            ax2.plot(angles, ring_vals, linewidth=0.5, linestyle="--", color="#3d4a5c", alpha=0.7)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(cats, color="#e2e8f0", fontsize=9.5, fontweight="bold")
        ax2.set_yticklabels([])
        ax2.set_ylim(0, 100)
        ax2.spines["polar"].set_color("#2d3748")
        ax2.grid(color="#2d3748", linewidth=0.5)

        ax2.set_title(f"Sub-Score Profile", color="#e2e8f0", fontsize=11, fontweight="bold", pad=20)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # ── Key Stats ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Key Stats</div>', unsafe_allow_html=True)
        ks1, ks2 = st.columns(2)
        def sf(v, d=0.0):
            try: return float(v) if pd.notna(v) and v != '' else d
            except: return d
        with ks1:
            st.metric("PPG",     f"{sf(row.get('points_per_game')):.1f}")
            st.metric("Opp PPG", f"{sf(row.get('points_allowed_per_game')):.1f}")
            tpp = sf(row.get('three_pt_pct'))
            st.metric("3P%", f"{tpp*100:.1f}%" if tpp < 1 else f"{tpp:.1f}%")
            ftp = sf(row.get('ft_pct'))
            st.metric("FT%", f"{ftp*100:.1f}%" if ftp < 1 else f"{ftp:.1f}%")
        with ks2:
            st.metric("Adj. Offense", f"{sf(row.get('adj_offense')):.1f}")
            st.metric("Adj. Defense", f"{sf(row.get('adj_defense')):.1f}")
            tempo = sf(row.get('tempo'))
            st.metric("Tempo", f"{tempo:.1f} poss/g" if tempo > 0 else "—")
            q1wp = sf(row.get('q1_win_pct'))
            st.metric("Q1 Win%", f"{q1wp*100:.0f}%" if q1wp < 1 else f"{q1wp:.0f}%")

    # ── Coach Profile ─────────────────────────────────────────────────────────
    st.markdown("---")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown('<div class="section-header">👨‍💼 Coach Profile</div>', unsafe_allow_html=True)
        coach_name = row.get('coach','') or "Unknown"
        st.markdown(f"**{coach_name}**")
        def safe_int(val, default=0):
            try: return int(float(val)) if pd.notna(val) and val != '' else default
            except: return default
        st.metric("Years at School",      f"{safe_int(row.get('coach_years_at_school'))}")
        st.metric("NCAA Tournament Games", f"{safe_int(row.get('coach_ncaa_games'))}")
    with cc2:
        st.markdown('<div class="section-header">🏆 Tournament Résumé</div>', unsafe_allow_html=True)
        st.metric("Sweet 16 Appearances",  f"{safe_int(row.get('coach_sweet16s'))}")
        st.metric("Final Four Appearances", f"{safe_int(row.get('coach_finalfours'))}")
        if row.get("first_year_coach_flag", 0):
            st.warning("⚠️ First-year coach at this school")
    with cc3:
        st.markdown('<div class="section-header">🚩 Model Flags</div>', unsafe_allow_html=True)
        flags = {
            "🪄 Cinderella":   row.get("cinderella_flag", False),
            "🌑 Darkhorse":    row.get("title_darkhorse_flag", False),
            "🃏 Fraud Fav.":   row.get("fraud_favorite_flag", False),
            "💥 Upset Alert":  row.get("dangerous_low_seed_flag", False),
            "📈 Underseeded":  row.get("underseeded_flag", False),
            "📉 Overseeded":   row.get("overseeded_flag", False),
            "⚡ Foul Depend.": row.get("high_foul_dependence_flag", False),
        }
        active_flags = [k for k, v in flags.items() if v]
        if active_flags:
            for f in active_flags:
                st.markdown(f"**{f}**")
        else:
            st.caption("No special flags")

    # ── Explanation ───────────────────────────────────────────────────────────
    if "explanation_summary" in row.index and pd.notna(row.get("explanation_summary")):
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Model Analysis</div>', unsafe_allow_html=True)
        st.info(row["explanation_summary"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL vs COMMITTEE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Model vs. Selection Committee — Seed Agreement Analysis</div>', unsafe_allow_html=True)
    st.caption("Positive gap = model ranks team HIGHER than committee. Negative = model ranks team LOWER.")

    if "seed" not in in_bracket.columns:
        st.warning("Bracket/seed data not loaded.")
    else:
        merged = in_bracket[in_bracket["seed"].notna()].copy()
        if "predicted_seed_line" in merged.columns:
            merged["gap"] = merged["predicted_seed_line"].fillna(merged["seed"]) - merged["seed"]
        else:
            merged["gap"] = 0

        # Most undervalued by committee
        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown('<div class="section-header">🟢 Model Loves (underseeded)</div>', unsafe_allow_html=True)
            under = merged.sort_values("gap").head(10)
            for _, r in under.iterrows():
                bar_w = min(100, int(abs(r["gap"]) * 8))
                st.markdown(f"""
                <div class="team-card">
                    <span class="seed-badge">{int(r['seed'])}</span>
                    <span class="team-name">{r['team']}</span>
                    <span style="color:#48bb78"> · Model says #{int(r.get('predicted_seed_line', r['seed']))} seed</span><br/>
                    <div style="background:#2d3748;border-radius:4px;margin-top:4px;height:5px">
                        <div style="width:{bar_w}%;background:#48bb78;height:5px;border-radius:4px"></div>
                    </div>
                    <small style="color:#8892a4">Contender: {r['contender_score']:.1f} · Risk: {r['upset_risk_score']:.1f}</small>
                </div>
                """, unsafe_allow_html=True)
        with mv2:
            st.markdown('<div class="section-header">🔴 Committee Loves (overseeded)</div>', unsafe_allow_html=True)
            over = merged.sort_values("gap", ascending=False).head(10)
            for _, r in over.iterrows():
                bar_w = min(100, int(abs(r["gap"]) * 8))
                st.markdown(f"""
                <div class="team-card">
                    <span class="seed-badge">{int(r['seed'])}</span>
                    <span class="team-name">{r['team']}</span>
                    <span style="color:#f87171"> · Model says #{int(r.get('predicted_seed_line', r['seed']))} seed</span><br/>
                    <div style="background:#2d3748;border-radius:4px;margin-top:4px;height:5px">
                        <div style="width:{bar_w}%;background:#ef4444;height:5px;border-radius:4px"></div>
                    </div>
                    <small style="color:#8892a4">Contender: {r['contender_score']:.1f} · Risk: {r['upset_risk_score']:.1f}</small>
                </div>
                """, unsafe_allow_html=True)

        # Snubs
        if len(bracket) > 0:
            snubs = scores[
                (scores["contender_score"] >= 63) & (~scores["team"].isin(bracket["team"]))
            ].sort_values("contender_score", ascending=False)
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
                            <small style="color:#f87171">Left out · Model: {r.get('expected_round','—')}</small>
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
        # Merge bracket info into champs
        champs_disp = champs.merge(bracket[["team","region","seed"]], on="team", how="left") if len(bracket)>0 else champs
        champs_disp = champs_disp.sort_values("championship_pct", ascending=False).head(20)

        # Top 3 podium
        pod1, pod2, pod3 = st.columns(3)
        podium_data = champs_disp.head(3)
        medals = ["🥇", "🥈", "🥉"]
        pod_cols = [pod1, pod2, pod3]
        for i, (_, r) in enumerate(podium_data.iterrows()):
            with pod_cols[i]:
                seed_disp = f"#{int(r['seed'])} seed · {r['region']}" if pd.notna(r.get('seed')) else ""
                st.markdown(f"""
                <div class="team-card" style="text-align:center;border-color:{'#f59e0b' if i==0 else '#94a3b8' if i==1 else '#cd7f32'}">
                    <div style="font-size:2rem">{medals[i]}</div>
                    <div class="team-name" style="font-size:1.3rem">{r['team']}</div>
                    <div style="color:#f97316;font-size:1.6rem;font-weight:800">{r['championship_pct']:.1f}%</div>
                    <div style="color:#8892a4;font-size:0.8rem">{seed_disp}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Bar chart for all
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#1a1f2e")

        teams_c  = champs_disp["team"].tolist()
        probs_c  = champs_disp["championship_pct"].tolist()
        colors_c = ["#f97316" if i == 0 else "#3b82f6" if i < 4 else "#4b5563" for i in range(len(teams_c))]

        bars = ax3.barh(teams_c[::-1], probs_c[::-1], color=colors_c[::-1], edgecolor="none", height=0.65)
        for bar, pct in zip(bars, probs_c[::-1]):
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f"{pct:.1f}%", va="center", color="#e2e8f0", fontsize=8.5, fontweight="bold")

        ax3.set_xlabel("Championship Probability (%)", color="#8892a4", fontsize=9)
        ax3.tick_params(colors="#e2e8f0", labelsize=9)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["bottom"].set_color("#2d3748")
        ax3.spines["left"].set_color("#2d3748")
        ax3.set_title("Simulated Championship Probabilities (10,000 runs)", color="#e2e8f0",
                      fontsize=11, fontweight="bold", pad=10)

        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # Region breakdown
        st.markdown("---")
        st.markdown('<div class="section-header">Championship Probability by Region</div>', unsafe_allow_html=True)
        if "region" in champs_disp.columns:
            reg_probs = champs_disp.groupby("region")["championship_pct"].sum().sort_values(ascending=False)
            rc1, rc2, rc3, rc4 = st.columns(4)
            rcols = [rc1, rc2, rc3, rc4]
            for i, (reg, pct) in enumerate(reg_probs.items()):
                if i < 4:
                    with rcols[i]:
                        st.metric(f"{reg} Region", f"{pct:.1f}%", "of championships")
