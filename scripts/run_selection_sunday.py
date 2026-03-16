#!/usr/bin/env python3
"""
run_selection_sunday.py
───────────────────────
Full bracket analysis for Selection Sunday — March 16, 2026.

USAGE:
    # Step 1: Generate/populate the bracket CSV
    python scripts/bracket_entry.py          # CBS template (then edit with real seedings)
    python scripts/bracket_entry.py --interactive   # enter teams one by one

    # Step 2: Run this analysis
    python scripts/run_selection_sunday.py

    # With custom sims count:
    python scripts/run_selection_sunday.py --sims 25000

WHAT IT PRODUCES:
    1. Fraud Favorites   — committee loves them, model is skeptical (overseeded)
    2. Darkhorse Threats — model loves them, committee underrated them (underseeded)
    3. Round-of-64 matchups — win probabilities for all 32 first-round games
    4. Top upset picks   — highest-probability upsets by seed line
    5. Monte Carlo simulation — championship %s for all 68 teams
    6. Final prediction summary

OUTPUTS (saved to data/outputs/):
    team_scores_2026.csv       — full model scores for all ~365 teams
    bracket_analysis_2026.csv  — 68 bracketed teams with all scores + flags
    simulation_results_2026.csv — champion probabilities
    upset_picks_2026.csv        — curated upset targets
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np

from src.ingest.fetch_team_stats      import fetch_from_sports_ref
from src.features.build_team_features import build_features
from src.features.compute_subscores   import compute_all_subscores
from src.models.baseline_rules        import score_all_teams, predicted_seed_line
from src.models.simulate_bracket      import simulate_bracket, _compute_win_prob
from src.utils.io                     import write_csv, ensure_dirs

SEASON      = 2026
BOT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRACKET_PATH = os.path.join(BOT_ROOT, "data", "brackets", "bracket_2026.csv")
STATS_PATH  = os.path.join(BOT_ROOT, "data", "raw", "teams", "team_stats_2026.csv")

W = 75   # display width


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    filled = int(round(val / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _pct_str(p: float) -> str:
    return f"{p:.1f}%"


def _flag_str(row) -> str:
    flags = []
    if row.get("fraud_favorite_flag") or row.get("overseeded_flag"):
        flags.append("⚠️  FRAUD")
    if row.get("title_darkhorse_flag") or row.get("underseeded_flag"):
        flags.append("🔥 DARKHORSE")
    if row.get("dangerous_low_seed_flag"):
        flags.append("💀 DANGEROUS")
    return "  ".join(flags)


def find_team(df: pd.DataFrame, name: str) -> dict:
    """Fuzzy-ish team lookup: exact → first-word → empty dict."""
    exact = df[df["team"] == name]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    first = name.split()[0]
    partial = df[df["team"].str.contains(first, case=False, na=False)]
    if not partial.empty:
        return partial.iloc[0].to_dict()
    return {"team": name, "contender_score": 40.0, "committee_alignment_score": 40.0,
            "efficiency_score": 40.0, "defense_score": 40.0, "clutch_score": 40.0,
            "model_vs_committee_gap": 0.0, "net_rank": float("nan"),
            "fraud_favorite_flag": False, "underseeded_flag": False,
            "predicted_seed_line": None}


# ─────────────────────────────────────────────────────────────────────────────
# Section printers
# ─────────────────────────────────────────────────────────────────────────────

def print_fraud_and_darkhorse(bracket_df: pd.DataFrame, scores_df: pd.DataFrame):
    print(f"\n{'='*W}")
    print("  FRAUD FAVORITES vs DARKHORSE THREATS")
    print(f"  (committee_alignment_score vs efficiency_score gap)")
    print(f"{'='*W}")
    print(f"  {'Team':<24} {'Seed':<7} {'CS':>6} {'CAS':>6} {'Gap':>6}  {'NET':>5}  {'Flag'}")
    print(f"  {'-'*(W-2)}")

    rows = []
    for _, brow in bracket_df.iterrows():
        seed_raw = brow.get("seed", "?")
        # skip play-in slots
        try:
            seed = int(str(seed_raw).replace("a","").replace("b",""))
        except ValueError:
            continue
        srow = find_team(scores_df, brow["team"])
        gap  = srow.get("model_vs_committee_gap", 0.0)
        rows.append({
            "team": brow["team"],
            "region": brow.get("region",""),
            "seed": seed,
            "seed_raw": seed_raw,
            "cs":  srow.get("contender_score", 40.0),
            "cas": srow.get("committee_alignment_score", 40.0),
            "gap": gap,
            "net": srow.get("net_rank", float("nan")),
            "fraud":      bool(srow.get("overseeded_flag") or srow.get("fraud_favorite_flag")),
            "darkhorse":  bool(srow.get("underseeded_flag") or srow.get("title_darkhorse_flag")),
        })

    # Sort: frauds (most negative gap) first, then darkhorses (most positive)
    frauds    = sorted([r for r in rows if r["fraud"]],     key=lambda x: x["gap"])
    darkhorse = sorted([r for r in rows if r["darkhorse"]], key=lambda x: -x["gap"])
    neutral   = [r for r in rows if not r["fraud"] and not r["darkhorse"]]

    def _print_row(r, label=""):
        net_s = f"#{int(r['net'])}" if not np.isnan(r["net"]) else " N/A"
        gap_s = f"{r['gap']:+.1f}"
        lbl   = f"⚠️ FRAUD" if r["fraud"] else ("🔥 DARK" if r["darkhorse"] else "")
        print(f"  {r['team']:<24} {r['region'][:3]}-{r['seed_raw']!s:<4} "
              f"{r['cs']:>6.1f} {r['cas']:>6.1f} {gap_s:>6}  {net_s:>5}  {lbl}")

    print(f"\n  ── FRAUD FAVORITES (committee trusts them more than the model does) ──")
    if frauds:
        for r in frauds:
            _print_row(r)
    else:
        print("    None flagged — unusual alignment this year!")

    print(f"\n  ── DARKHORSE THREATS (model trusts them more than the committee does) ──")
    if darkhorse:
        for r in darkhorse:
            _print_row(r)
    else:
        print("    None flagged.")

    # Summary of all 1-4 seeds
    print(f"\n  ── TOP SEEDS AT A GLANCE ──")
    top_seeds = sorted([r for r in rows if r["seed"] <= 4], key=lambda x: (x["seed"], x["region"]))
    for r in top_seeds:
        _print_row(r)

    return rows


def print_matchups(bracket_df: pd.DataFrame, scores_df: pd.DataFrame) -> list:
    """Print Round of 64 matchups. Returns list of upset alerts."""
    print(f"\n{'='*W}")
    print("  ROUND OF 64 — MATCHUP ANALYSIS")
    print(f"{'='*W}")

    # Standard first-round matchup pairings by seed
    SEED_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    regions    = ["East","Midwest","South","West"]

    upset_alerts = []

    for region in regions:
        rdf = bracket_df[bracket_df["region"] == region].copy()
        rdf["seed_num"] = pd.to_numeric(
            rdf["seed"].astype(str).str.replace(r"[ab]","", regex=True),
            errors="coerce"
        )
        rdf = rdf.dropna(subset=["seed_num"])
        seed_to_team = dict(zip(rdf["seed_num"].astype(int), rdf["team"]))

        print(f"\n  ── {region.upper()} ─────────────────────────────────────────────")
        print(f"  {'Higher Seed':<26} {'vs':<4} {'Lower Seed':<26} {'Win %':<8} {'Alert'}")
        print(f"  {'-'*(W-2)}")

        for s_hi, s_lo in SEED_PAIRS:
            team_hi = seed_to_team.get(s_hi, f"Seed {s_hi}")
            team_lo = seed_to_team.get(s_lo, f"Seed {s_lo}")

            fa = {**find_team(scores_df, team_hi), "seed": s_hi}
            fb = {**find_team(scores_df, team_lo), "seed": s_lo}

            # Use the same NaN-safe win probability as the Monte Carlo engine
            p_hi = _compute_win_prob(fa, fb)

            # Alert logic
            alert = ""
            if s_lo <= 12 and p_hi < 0.65:           # upset real threat
                alert = f"🚨 UPSET ALERT ({_pct_str(p_hi*100)} fav)"
                upset_alerts.append({
                    "region": region, "fav_seed": s_hi, "fav": team_hi,
                    "dog_seed": s_lo, "dog": team_lo,
                    "fav_win_pct": round(p_hi*100, 1),
                    "upset_pct": round((1-p_hi)*100, 1)
                })
            elif s_lo <= 5 and p_hi < 0.80:
                alert = f"⚠️  Close game ({_pct_str(p_hi*100)} fav)"
            elif s_lo == 12 and p_hi < 0.72:
                alert = f"💡 5-12 watch ({_pct_str(p_hi*100)} fav)"
                upset_alerts.append({
                    "region": region, "fav_seed": s_hi, "fav": team_hi,
                    "dog_seed": s_lo, "dog": team_lo,
                    "fav_win_pct": round(p_hi*100, 1),
                    "upset_pct": round((1-p_hi)*100, 1)
                })

            fraud_hi = bool(find_team(scores_df, team_hi).get("overseeded_flag"))
            if fraud_hi and s_hi <= 4:
                alert = (alert + " | FRAUD 1-seed") if alert else "⚠️  FRAUD 1-seed watch"

            bar = _bar(p_hi * 100, 10)
            print(f"  #{s_hi:<2} {team_hi:<22} vs  #{s_lo:<2} {team_lo:<22} "
                  f"{_pct_str(p_hi*100):<8} {alert}")

    return upset_alerts


def print_upset_summary(upset_alerts: list):
    if not upset_alerts:
        return
    print(f"\n{'='*W}")
    print("  TOP UPSET PICKS (sorted by upset probability)")
    print(f"{'='*W}")
    ua = sorted(upset_alerts, key=lambda x: -x["upset_pct"])
    print(f"  {'Dog':<26} {'Seed':<7} {'Upset %':<9} {'vs Fav':<26} {'Seed'}")
    print(f"  {'-'*(W-2)}")
    for u in ua[:12]:
        print(f"  {u['dog']:<26} #{u['dog_seed']:<5}  {u['upset_pct']:>6.1f}%   "
              f"{u['fav']:<26} #{u['fav_seed']}  ({u['region']})")


def print_simulation(champ_df: pd.DataFrame, n_sims: int):
    print(f"\n{'='*W}")
    print(f"  MONTE CARLO SIMULATION — {n_sims:,} runs")
    print(f"{'='*W}")
    print(f"  {'Team':<26} {'Champ %':>8}  {'Probability Bar'}")
    print(f"  {'-'*(W-2)}")
    for _, row in champ_df.head(20).iterrows():
        pct = row["championship_pct"]
        bar = _bar(pct * 4, 20)   # scale: 25% = full bar
        print(f"  {row['team']:<26} {_pct_str(pct):>8}  {bar}")
    # Show cumulative FF odds
    ff_pct = champ_df.head(4)["championship_pct"].sum()
    print(f"\n  Top-4 combined championship probability: {ff_pct:.1f}%")


def print_final_summary(bracket_df, scores_df, champ_df):
    print(f"\n{'='*W}")
    print("  2026 FINAL PREDICTION SUMMARY")
    print(f"{'='*W}")

    # Model's top 4 overall
    top4 = scores_df.sort_values("contender_score", ascending=False).head(4)
    print(f"\n  Model's top 4 teams:")
    for i, (_, r) in enumerate(top4.iterrows()):
        net = f"NET #{int(r['net_rank'])}" if "net_rank" in r and not pd.isna(r.get("net_rank")) else ""
        print(f"    #{i+1}  {r['team']:<28} CS:{r['contender_score']:.1f}  CAS:{r.get('committee_alignment_score',0):.1f}  {net}")

    # Committee's 1-seeds from bracket
    one_seeds = bracket_df[bracket_df["seed"].astype(str) == "1"]
    print(f"\n  Committee's 1-seeds:")
    for _, r in one_seeds.iterrows():
        srow = find_team(scores_df, r["team"])
        model_rank = scores_df.sort_values("contender_score", ascending=False).reset_index(drop=True)
        team_rank = model_rank[model_rank["team"] == r["team"]].index
        rank_str = f"(model #{team_rank[0]+1})" if len(team_rank) > 0 else ""
        fraud = "⚠️ FRAUD RISK" if srow.get("overseeded_flag") else ""
        print(f"    {r['region']:<10} {r['team']:<28} {rank_str}  {fraud}")

    if champ_df is not None and len(champ_df) > 0:
        print(f"\n  Model champion pick: {champ_df.iloc[0]['team']} "
              f"({champ_df.iloc[0]['championship_pct']:.1f}% probability)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--bracket", type=str, default=BRACKET_PATH)
    args = parser.parse_args()

    ensure_dirs()
    os.makedirs(os.path.join(BOT_ROOT, "data", "outputs"), exist_ok=True)

    # ── Load + score all teams ────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  MARCH MADNESS BOT — 2026 SELECTION SUNDAY ANALYSIS")
    print(f"{'='*W}")

    print(f"\nLoading team stats from {STATS_PATH}...")
    if not os.path.exists(STATS_PATH):
        print(f"  Stats file not found — fetching from Sports Reference...")
        from src.ingest.fetch_team_stats import fetch_from_sports_ref
        df_raw = fetch_from_sports_ref(SEASON)
    else:
        df_raw = pd.read_csv(STATS_PATH)
        print(f"  Loaded {len(df_raw)} teams")

    df = build_features(df_raw)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df = df.sort_values("contender_score", ascending=False).reset_index(drop=True)
    df["model_rank"] = df.index + 1

    write_csv(df, os.path.join(BOT_ROOT, "data", "outputs", "team_scores_2026.csv"))
    print(f"  Full model scores saved → data/outputs/team_scores_2026.csv")

    # ── Load bracket ──────────────────────────────────────────────────────────
    if not os.path.exists(args.bracket):
        print(f"\n⚠️  Bracket file not found: {args.bracket}")
        print("Run:  python scripts/bracket_entry.py   (then edit with real seedings)")
        print("Then re-run this script.\n")
        print("Showing full model ranking instead:\n")
        print(f"  {'#':<5} {'Team':<28} {'CS':>6} {'CAS':>6} {'NET':>6} {'Gap':>6}  {'Pred Seed'}")
        print(f"  {'-'*(W-2)}")
        for _, r in df.head(30).iterrows():
            net_s = f"#{int(r['net_rank'])}" if "net_rank" in r and not pd.isna(r.get("net_rank")) else " N/A"
            gap   = r.get("model_vs_committee_gap", 0)
            ps    = r.get("predicted_seed_line", "?")
            fraud = " ← FRAUD RISK" if r.get("overseeded_flag") else ""
            dark  = " ← UNDERSEEDED" if r.get("underseeded_flag") else ""
            print(f"  #{int(r['model_rank']):<4} {r['team']:<28} "
                  f"{r['contender_score']:>6.1f} "
                  f"{r.get('committee_alignment_score',0):>6.1f} "
                  f"{net_s:>6} {gap:>+6.1f}  ~{ps}{fraud}{dark}")
        return

    bracket_df = pd.read_csv(args.bracket, dtype={"seed": str})
    bracket_df = bracket_df[bracket_df["season"] == SEASON].copy()
    print(f"  Bracket loaded: {len(bracket_df)} entries across "
          f"{bracket_df['region'].nunique()} regions")

    # ── Enrich bracket with model scores ──────────────────────────────────────
    enriched = []
    for _, brow in bracket_df.iterrows():
        srow = find_team(df, brow["team"])
        enriched.append({
            **brow.to_dict(),
            "model_rank":               srow.get("model_rank", 999),
            "contender_score":          srow.get("contender_score", 40.0),
            "committee_alignment_score": srow.get("committee_alignment_score", 40.0),
            "efficiency_score":         srow.get("efficiency_score", 40.0),
            "defense_score":            srow.get("defense_score", 40.0),
            "clutch_score":             srow.get("clutch_score", 40.0),
            "model_vs_committee_gap":   srow.get("model_vs_committee_gap", 0.0),
            "net_rank":                 srow.get("net_rank", float("nan")),
            "predicted_seed_line":      srow.get("predicted_seed_line", None),
            "overseeded_flag":          bool(srow.get("overseeded_flag", False)),
            "underseeded_flag":         bool(srow.get("underseeded_flag", False)),
            "fraud_favorite_flag":      bool(srow.get("fraud_favorite_flag", False)),
            "title_darkhorse_flag":     bool(srow.get("title_darkhorse_flag", False)),
        })
    bracket_enriched = pd.DataFrame(enriched)
    write_csv(bracket_enriched,
              os.path.join(BOT_ROOT, "data", "outputs", "bracket_analysis_2026.csv"))

    # ── Section 1: Fraud & Darkhorse ─────────────────────────────────────────
    print_fraud_and_darkhorse(bracket_df, df)

    # ── Section 2: Round of 64 matchups ──────────────────────────────────────
    upset_alerts = print_matchups(bracket_df, df)

    # ── Section 3: Upset summary ──────────────────────────────────────────────
    print_upset_summary(upset_alerts)
    if upset_alerts:
        write_csv(pd.DataFrame(upset_alerts),
                  os.path.join(BOT_ROOT, "data", "outputs", "upset_picks_2026.csv"))

    # ── Section 4: Monte Carlo simulation ────────────────────────────────────
    # Filter bracket to numeric seeds only (exclude play-in a/b slots)
    sim_bracket = bracket_df[
        pd.to_numeric(bracket_df["seed"], errors="coerce").notna()
    ].copy()
    sim_bracket["seed"] = sim_bracket["seed"].astype(int)

    print(f"\nRunning {args.sims:,} Monte Carlo simulations...")
    champ_df = simulate_bracket(sim_bracket, df, n_sims=args.sims)
    write_csv(champ_df,
              os.path.join(BOT_ROOT, "data", "outputs", "simulation_results_2026.csv"))
    print_simulation(champ_df, args.sims)

    # ── Section 5: Final summary ──────────────────────────────────────────────
    print_final_summary(bracket_df, df, champ_df)

    print(f"\n{'='*W}")
    print("  All outputs saved to data/outputs/")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
