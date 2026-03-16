#!/usr/bin/env python3
"""
Evaluate how well the model predicts Final Four teams for each historical season.

Shows:
  - Rank of each actual Final Four team in the model's rankings
  - How many Final Four teams appear in model's top 4 / top 10 / top 20 / top 30
  - Champion rank (already in backtest, shown here per-year)

Usage:
    python scripts/eval_final_four.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest.fetch_team_stats import fetch_from_sports_ref
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams

# Historical Final Four teams (4 per year)
FINAL_FOURS = {
    2015: ["Duke", "Wisconsin", "Michigan State", "Kentucky"],
    2016: ["Villanova", "Oklahoma", "North Carolina", "Syracuse"],
    2017: ["North Carolina", "Oregon", "Gonzaga", "South Carolina"],
    2018: ["Villanova", "Michigan", "Kansas", "Loyola-Chicago"],
    2019: ["Virginia", "Texas Tech", "Auburn", "Michigan State"],
    2021: ["Baylor", "Houston", "UCLA", "Gonzaga"],
    2022: ["Kansas", "Villanova", "Duke", "North Carolina"],
    2023: ["Connecticut", "San Diego State", "Florida Atlantic", "Miami FL"],
    2024: ["Connecticut", "Alabama", "NC State", "Purdue"],
    2025: ["Duke", "Florida", "Auburn", "Houston"],
}

CHAMPIONS = {
    2015: "Duke", 2016: "Villanova", 2017: "North Carolina",
    2018: "Villanova", 2019: "Virginia", 2021: "Baylor",
    2022: "Kansas", 2023: "Connecticut", 2024: "Connecticut",
    2025: "Florida",
}


def find_team_rank(df_sorted: pd.DataFrame, team_name: str) -> int:
    """Find a team's rank by partial name match (handles aliases)."""
    # Exact match first
    mask = df_sorted["team"].str.lower() == team_name.lower()
    if mask.any():
        return int(df_sorted.loc[mask, "rank"].iloc[0])

    # Partial match
    mask2 = df_sorted["team"].str.contains(team_name, case=False, na=False)
    if mask2.any():
        return int(df_sorted.loc[mask2, "rank"].iloc[0])

    return -1  # Not found


def evaluate_season(year: int) -> dict:
    """Score all teams and evaluate Final Four prediction accuracy."""
    df = fetch_from_sports_ref(year)
    if df.empty:
        return None

    df = build_features(df)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df_sorted = df.sort_values("contender_score", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    ff_teams = FINAL_FOURS.get(year, [])
    champion = CHAMPIONS.get(year, "")

    results = {}
    ranks = []
    found = []
    for team in ff_teams:
        r = find_team_rank(df_sorted, team)
        results[team] = r
        if r > 0:
            ranks.append(r)
            found.append(team)

    champ_rank = find_team_rank(df_sorted, champion)

    # Count how many FF teams land in top N
    in_top4  = sum(1 for r in ranks if r <= 4)
    in_top10 = sum(1 for r in ranks if r <= 10)
    in_top20 = sum(1 for r in ranks if r <= 20)
    in_top30 = sum(1 for r in ranks if r <= 30)

    return {
        "year": year,
        "champion": champion,
        "champ_rank": champ_rank,
        "ff_ranks": results,
        "avg_ff_rank": round(np.mean(ranks), 1) if ranks else -1,
        "in_top4": in_top4,
        "in_top10": in_top10,
        "in_top20": in_top20,
        "in_top30": in_top30,
        "ranked_df": df_sorted,
    }


def main():
    years = sorted(FINAL_FOURS.keys())

    print("=" * 72)
    print("  FINAL FOUR PREDICTION ACCURACY — 2015–2025")
    print("=" * 72)

    all_champ_ranks = []
    all_ff_ranks    = []
    total_in_top4   = 0
    total_in_top10  = 0
    total_in_top20  = 0
    total_in_top30  = 0

    for year in years:
        res = evaluate_season(year)
        if not res:
            continue

        print(f"\n{year}  (Champion: {res['champion']}, rank #{res['champ_rank']})")
        print(f"  {'Team':<30} {'Model Rank':>10}")
        print(f"  {'-'*42}")
        for team, rank in res["ff_ranks"].items():
            marker = " ← CHAMP" if team == res["champion"] else ""
            rank_str = f"#{rank}" if rank > 0 else "NF"
            print(f"  {team:<30} {rank_str:>10}{marker}")

        print(f"  {'─'*42}")
        print(f"  Avg FF rank: {res['avg_ff_rank']:.1f}  |  "
              f"In top 4: {res['in_top4']}/4  |  "
              f"In top 10: {res['in_top10']}/4  |  "
              f"In top 20: {res['in_top20']}/4  |  "
              f"In top 30: {res['in_top30']}/4")

        all_champ_ranks.append(res["champ_rank"])
        all_ff_ranks.extend([r for r in res["ff_ranks"].values() if r > 0])
        total_in_top4  += res["in_top4"]
        total_in_top10 += res["in_top10"]
        total_in_top20 += res["in_top20"]
        total_in_top30 += res["in_top30"]

    n = len(years)
    max_ff = n * 4

    print("\n" + "=" * 72)
    print("  OVERALL SUMMARY (10 seasons, 40 Final Four slots)")
    print("=" * 72)
    print(f"  Avg champion rank:        {np.mean(all_champ_ranks):.1f}")
    print(f"  Avg Final Four rank:      {np.mean(all_ff_ranks):.1f}")
    print(f"  FF teams in model top 4:  {total_in_top4}/{max_ff} ({100*total_in_top4/max_ff:.0f}%)")
    print(f"  FF teams in model top 10: {total_in_top10}/{max_ff} ({100*total_in_top10/max_ff:.0f}%)")
    print(f"  FF teams in model top 20: {total_in_top20}/{max_ff} ({100*total_in_top20/max_ff:.0f}%)")
    print(f"  FF teams in model top 30: {total_in_top30}/{max_ff} ({100*total_in_top30/max_ff:.0f}%)")

    # Also print 2026 top 15 for reference
    print("\n" + "=" * 72)
    print("  2026 TOP 15 PREDICTED CONTENDERS")
    print("=" * 72)
    res26 = evaluate_season(2026) if 2026 not in FINAL_FOURS else None
    if res26 is None:
        df26 = fetch_from_sports_ref(2026)
        df26 = build_features(df26)
        df26 = compute_all_subscores(df26)
        df26 = score_all_teams(df26)
        df26_sorted = df26.sort_values("contender_score", ascending=False).reset_index(drop=True)
        df26_sorted["rank"] = df26_sorted.index + 1
    else:
        df26_sorted = res26["ranked_df"]

    # Filter to meaningful teams (adj_margin > 10 or top 15 overall)
    top15 = df26_sorted[df26_sorted["adj_margin"] >= 15].head(15)
    print(f"  {'Rank':<6} {'Team':<28} {'Score':>6} {'AdjMargin':>10} {'Barthag':>8} {'Last10':>7} {'ClosePct':>9}")
    print(f"  {'-'*76}")
    for _, row in top15.iterrows():
        l10 = f"{row['last10_win_pct']:.0%}" if pd.notna(row.get('last10_win_pct')) else "  N/A"
        cp  = f"{row['close_game_record']:.0%}" if pd.notna(row.get('close_game_record')) else "  N/A"
        brt = f"{row['barthag']:.3f}" if pd.notna(row.get('barthag')) else "  N/A"
        print(f"  #{int(row['rank']):<5} {row['team']:<28} {row['contender_score']:>6.1f} "
              f"{row['adj_margin']:>10.1f} {brt:>8} {l10:>7} {cp:>9}")


if __name__ == "__main__":
    main()
