#!/usr/bin/env python3
"""
Evaluate whether the model correctly identifies the 4 overall #1 seeds
for each tournament year (2015–2026).

For historical years: compare model top-N vs actual #1 seeds.
For 2026: predict which 4 teams will be the #1 seeds.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.ingest.fetch_team_stats import fetch_from_sports_ref
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams

# ─────────────────────────────────────────────────────────────
# Actual #1 seeds by year and region
# Sources: NCAA.com official brackets
# ─────────────────────────────────────────────────────────────
ONE_SEEDS = {
    2015: {"East": "Villanova",       "Midwest": "Wisconsin",
           "South": "Kentucky",       "West":    "Duke"},
    2016: {"East": "North Carolina",  "Midwest": "Kansas",
           "South": "Virginia",       "West":    "Oregon"},
    2017: {"East": "Villanova",       "Midwest": "Kansas",
           "South": "North Carolina", "West":    "Gonzaga"},
    2018: {"East": "Villanova",       "Midwest": "Kansas",
           "South": "Virginia",       "West":    "Xavier"},
    2019: {"East": "Duke",            "Midwest": "North Carolina",
           "South": "Virginia",       "West":    "Gonzaga"},
    2021: {"East": "Michigan",        "Midwest": "Illinois",
           "South": "Baylor",         "West":    "Gonzaga"},
    2022: {"East": "Baylor",          "Midwest": "Kansas",
           "South": "Arizona",        "West":    "Gonzaga"},
    2023: {"East": "Purdue",          "Midwest": "Houston",
           "South": "Alabama",        "West":    "Kansas"},
    2024: {"East": "Connecticut",     "Midwest": "Purdue",
           "South": "Houston",        "West":    "North Carolina"},
    2025: {"East": "Duke",            "Midwest": "Auburn",
           "South": "Florida",        "West":    "Houston"},
}

# 2026 projected (CBS bracketology, March 15 — Selection Sunday eve)
ONE_SEEDS_2026_PROJECTED = {
    "East":    "Duke",
    "Midwest": "Michigan",
    "South":   "Florida",
    "West":    "Arizona",
}


def run_model(season: int) -> pd.DataFrame:
    df = fetch_from_sports_ref(season)
    df = build_features(df)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df = df.sort_values("contender_score", ascending=False).reset_index(drop=True)
    df["model_rank"] = df.index + 1
    return df


def find_rank(df: pd.DataFrame, team: str) -> int:
    """Return model rank for a team (partial match tolerated)."""
    exact = df[df["team"] == team]
    if not exact.empty:
        return int(exact.iloc[0]["model_rank"])
    partial = df[df["team"].str.contains(team.split()[0], case=False, na=False)]
    if not partial.empty:
        return int(partial.iloc[0]["model_rank"])
    return -1


def bar(rank: int, max_rank: int = 30) -> str:
    """Visual indicator for how high the model placed a 1-seed."""
    if rank == -1:
        return "NF   [team not found]"
    if rank <= 4:
        return f"#{rank:<3} ✅ IN TOP 4"
    if rank <= 8:
        return f"#{rank:<3} 🟡 IN TOP 8"
    if rank <= 15:
        return f"#{rank:<3} 🔶 IN TOP 15"
    return f"#{rank:<3} ❌ OUTSIDE TOP 15"


def main():
    print("=" * 75)
    print("  1-SEED PREDICTION ACCURACY — 2015–2025  (then 2026 forecast)")
    print("=" * 75)

    season_stats = []

    for season in sorted(ONE_SEEDS.keys()):
        seeds = ONE_SEEDS[season]
        seed_teams = list(seeds.values())

        df = run_model(season)

        ranks = {region: find_rank(df, team) for region, team in seeds.items()}
        top4_correct = sum(1 for r in ranks.values() if r != -1 and r <= 4)
        top8_correct = sum(1 for r in ranks.values() if r != -1 and r <= 8)

        print(f"\n{'─'*75}")
        print(f"  {season}  — 1-seeds in model top 4: {top4_correct}/4  |  top 8: {top8_correct}/4")
        print(f"{'─'*75}")
        for region, team in seeds.items():
            rank = ranks[region]
            score_row = df[df["model_rank"] == rank] if rank != -1 else pd.DataFrame()
            score = f"(score {score_row.iloc[0]['contender_score']:.1f})" if not score_row.empty else ""
            print(f"  {region:<9} {team:<26} {bar(rank)}  {score}")

        # Also show who the model DID put in top 4
        top4 = df.head(4)[["model_rank", "team", "contender_score"]].copy()
        top4_names = top4["team"].tolist()
        correct_names = [t for t in top4_names if any(
            t == s or s.split()[0] in t or t.split()[0] in s
            for s in seed_teams)]
        print(f"\n  Model top 4: {' | '.join(f'#{i+1} {n}' for i,n in enumerate(top4_names))}")
        if correct_names:
            print(f"  Overlap with actual 1-seeds: {correct_names}")

        season_stats.append({
            "season": season,
            "top4_correct": top4_correct,
            "top8_correct": top8_correct,
        })

    # ── Summary
    stats_df = pd.DataFrame(season_stats)
    print(f"\n{'='*75}")
    print("  HISTORICAL SUMMARY (10 seasons)")
    print(f"{'='*75}")
    print(f"  Seasons with all 4 #1 seeds in model top 8: "
          f"{(stats_df['top8_correct'] == 4).sum()}/10")
    print(f"  Seasons with ≥3 #1 seeds in model top 8:   "
          f"{(stats_df['top8_correct'] >= 3).sum()}/10")
    print(f"  Seasons with ≥2 #1 seeds in model top 4:   "
          f"{(stats_df['top4_correct'] >= 2).sum()}/10")
    print(f"  Avg #1 seeds in model top 4:  {stats_df['top4_correct'].mean():.1f}/4")
    print(f"  Avg #1 seeds in model top 8:  {stats_df['top8_correct'].mean():.1f}/4")

    # ── 2026 Prediction
    print(f"\n{'='*75}")
    print("  2026 BRACKET PREDICTION — 1-SEEDS")
    print(f"  CBS Bracketology (Mar 15):  Duke | Michigan | Florida | Arizona")
    print(f"{'='*75}")

    df26 = run_model(2026)

    # Show where projected 1-seeds land in the model
    print(f"\n  Where CBS projected 1-seeds rank in the model:")
    print(f"  {'Region':<10} {'Team':<22} {'CBS Seed':<10} {'Model Rank'}")
    print(f"  {'-'*60}")
    for region, team in ONE_SEEDS_2026_PROJECTED.items():
        rank = find_rank(df26, team)
        row  = df26[df26["model_rank"] == rank] if rank != -1 else pd.DataFrame()
        score = row.iloc[0]["contender_score"] if not row.empty else 0
        net   = row.iloc[0].get("net_rank", float("nan")) if not row.empty else float("nan")
        net_s = f"NET #{int(net)}" if not pd.isna(net) else "NET N/A"
        print(f"  {region:<10} {team:<22} {net_s:<10} {bar(rank)}  (score {score:.1f})")

    # Show who the MODEL thinks the 1-seeds should be (top 4)
    top4_26 = df26.head(4)
    print(f"\n  Model's top 4 (what IT would pick as 1-seeds):")
    print(f"  {'Rank':<6} {'Team':<26} {'Score':>6} {'NET':>6} {'Last10':>8} {'ClosePct':>9} {'Defense':>8}")
    print(f"  {'-'*75}")
    for _, r in top4_26.iterrows():
        net  = f"#{int(r['net_rank'])}" if not pd.isna(r.get("net_rank", float("nan"))) else "N/A"
        l10  = f"{r['last10_win_pct']:.0%}" if not pd.isna(r.get("last10_win_pct", float("nan"))) else "N/A"
        cpct = f"{r['close_game_record']:.0%}" if not pd.isna(r.get("close_game_record", float("nan"))) else "N/A"
        print(f"  #{int(r['model_rank']):<5} {r['team']:<26} {r['contender_score']:>6.1f} "
              f"{net:>6} {l10:>8} {cpct:>9} {r['defense_score']:>8.1f}")

    # Show full top 15 with flags for projected 1-seeds
    print(f"\n  Full top 15 with 1-seed flags:")
    print(f"  {'Rank':<6} {'Team':<26} {'Score':>6} {'NET':>6} {'1-seed?'}")
    print(f"  {'-'*60}")
    projected_1seeds = list(ONE_SEEDS_2026_PROJECTED.values())
    for _, r in df26.head(15).iterrows():
        net = f"#{int(r['net_rank'])}" if not pd.isna(r.get("net_rank", float("nan"))) else "N/A"
        is_1seed = any(r["team"] == s or s.split()[0] in r["team"] for s in projected_1seeds)
        flag = " ← CBS #1 SEED" if is_1seed else ""
        print(f"  #{int(r['model_rank']):<5} {r['team']:<26} {r['contender_score']:>6.1f} "
              f"{net:>6}{flag}")

    print(f"\n{'='*75}")
    print("  MODEL'S 2026 FINAL PREDICTION")
    print(f"{'='*75}")
    print(f"  Champion pick:      #{find_rank(df26, df26.iloc[0]['team'])}  {df26.iloc[0]['team']}")
    print(f"  Predicted 1-seeds:  {' | '.join(df26.head(4)['team'].tolist())}")
    print(f"  CBS 1-seeds:        Duke | Michigan | Florida | Arizona")
    disagreements = [t for t in df26.head(4)["team"] if not any(
        t == s or s.split()[0] in t for s in projected_1seeds)]
    if disagreements:
        print(f"\n  Model DISAGREES on: {disagreements}")
        print(f"  These CBS 1-seeds are OUTSIDE model top 4:")
        for s in projected_1seeds:
            r = find_rank(df26, s)
            if r > 4:
                print(f"    {s} → model #{r}")


if __name__ == "__main__":
    main()
