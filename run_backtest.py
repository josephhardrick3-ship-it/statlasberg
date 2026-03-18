#!/usr/bin/env python3
"""
Run backtest across historical seasons (2010–2025, skip 2020).

For each season, the pipeline:
  1. Loads team stats for that season
  2. Scores teams
  3. Compares model predictions against actual tournament outcomes
  4. Aggregates accuracy metrics

Usage:
    python run_backtest.py
    python run_backtest.py --seasons 2022 2023 2024
    python run_backtest.py --data-dir data/raw/historical
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.ingest.fetch_team_stats import generate_sample_data
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams
from src.models.classify_archetypes import classify_all_teams
from src.utils.io import write_csv, ensure_dirs
from src.utils.logging_utils import get_logger

log = get_logger("backtest")

DEFAULT_SEASONS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                   2021, 2022, 2023, 2024, 2025]

# Historical champions for validation
CHAMPIONS = {
    2010: "Duke", 2011: "Connecticut", 2012: "Kentucky", 2013: "Louisville",
    2014: "Connecticut", 2015: "Duke", 2016: "Villanova", 2017: "North Carolina",
    2018: "Villanova", 2019: "Virginia", 2021: "Baylor", 2022: "Kansas",
    2023: "Connecticut", 2024: "Connecticut", 2025: "Florida",
}

BRACKETS_PATH = "data/raw/tournament_history/brackets.csv"
# First-round seed matchup pairs (higher seed vs lower seed)
_BRACKET_PODS = [
    ([(1, 16), (8, 9)],  [(5, 12), (4, 13)]),
    ([(6, 11), (3, 14)], [(7, 10), (2, 15)]),
]


def load_bracket_field(season, brackets_path=BRACKETS_PATH):
    """Load historical bracket field for a season.

    Returns dict mapping (region, seed) → team_name, or None if unavailable.
    """
    if not os.path.exists(brackets_path):
        return None
    df = pd.read_csv(brackets_path)
    season_df = df[df["season"] == season]
    if len(season_df) == 0:
        return None
    return {
        (str(row["region"]), int(row["seed"])): str(row["team"])
        for _, row in season_df.iterrows()
    }


def _win_prob_bt(c1, c2):
    """Sigmoid win probability. Same formula as the Streamlit app."""
    diff = float(c1) - float(c2)
    return 1.0 / (1.0 + np.exp(-diff / 10.0))


def simulate_bracket(scores_df, bracket_field):
    """Run a deterministic bracket simulation given seeded field and model scores.

    Returns (predicted_champion, predicted_ff_teams).
    """
    score_lkp = {
        str(row["team"]): float(row.get("contender_score", 50))
        for _, row in scores_df.iterrows()
    }

    def pwin(t1, t2):
        if not t1: return t2, t1
        if not t2: return t1, t2
        p = _win_prob_bt(score_lkp.get(t1, 50), score_lkp.get(t2, 50))
        return (t1, t2) if p >= 0.5 else (t2, t1)

    ff_teams = []
    for region in ["East", "South", "West", "Midwest"]:
        region_e8 = []
        for pod_a_pairs, pod_b_pairs in _BRACKET_PODS:
            r1w_a = []
            for s1, s2 in pod_a_pairs:
                t1 = bracket_field.get((region, s1), "")
                t2 = bracket_field.get((region, s2), "")
                if t1 or t2:
                    w, _ = pwin(t1, t2)
                    if w:
                        r1w_a.append(w)
            r1w_b = []
            for s1, s2 in pod_b_pairs:
                t1 = bracket_field.get((region, s1), "")
                t2 = bracket_field.get((region, s2), "")
                if t1 or t2:
                    w, _ = pwin(t1, t2)
                    if w:
                        r1w_b.append(w)
            # Round of 32
            pod_a_s16 = ""
            if len(r1w_a) == 2:
                w, _ = pwin(r1w_a[0], r1w_a[1])
                pod_a_s16 = w
            elif r1w_a:
                pod_a_s16 = r1w_a[0]
            pod_b_s16 = ""
            if len(r1w_b) == 2:
                w, _ = pwin(r1w_b[0], r1w_b[1])
                pod_b_s16 = w
            elif r1w_b:
                pod_b_s16 = r1w_b[0]
            # Sweet 16
            if pod_a_s16 and pod_b_s16:
                w, _ = pwin(pod_a_s16, pod_b_s16)
                region_e8.append(w)
        # Elite 8
        if len(region_e8) == 2:
            w, _ = pwin(region_e8[0], region_e8[1])
            ff_teams.append(w)

    # Final Four
    champ_teams = []
    ff_pairs = (
        [(ff_teams[0], ff_teams[3]), (ff_teams[1], ff_teams[2])]
        if len(ff_teams) >= 4 else []
    )
    for t1, t2 in ff_pairs:
        w, _ = pwin(t1, t2)
        champ_teams.append(w)

    # Championship
    champion = ""
    if len(champ_teams) == 2:
        w, _ = pwin(champ_teams[0], champ_teams[1])
        champion = w

    return champion, ff_teams


def run_single_season(season, data_dir=None):
    """Score one season. Returns scored DataFrame."""
    csv_path = os.path.join(data_dir, f"team_stats_{season}.csv") if data_dir else None

    using_real_data = csv_path and os.path.exists(csv_path)
    if using_real_data:
        df = pd.read_csv(csv_path)
        if "season" not in df.columns:
            df["season"] = season
    else:
        # Generate sample data (placeholder — install real historical data via
        # scripts/fetch_sports_ref.py for meaningful backtest accuracy)
        df = generate_sample_data(season=season, n_teams=68)

    df = build_features(df)
    df = compute_all_subscores(df)
    # Historical backtests: skip season-specific injury overrides (e.g. 2026
    # injuries should not penalise 2019 Duke in the historical ranking)
    df = score_all_teams(df, apply_availability=False)
    df = classify_all_teams(df)
    df["_using_real_data"] = using_real_data
    return df


def evaluate_season(scores_df, season):
    """Evaluate model for one season against known outcomes."""
    metrics = {"season": season}

    ranked = scores_df.sort_values("contender_score", ascending=False).reset_index(drop=True)

    # Model's top pick (rank 1)
    if len(ranked) > 0:
        metrics["model_top_pick"] = ranked.iloc[0]["team"]
        metrics["model_top_score"] = float(ranked.iloc[0]["contender_score"])
    else:
        metrics["model_top_pick"] = None
        metrics["model_top_score"] = None

    champion = CHAMPIONS.get(season)
    metrics["actual_champion"] = champion
    if champion:
        match = ranked[ranked["team"] == champion]
        if len(match) > 0:
            metrics["champion_rank"] = int(match.index[0]) + 1
            metrics["champion_contender_score"] = float(match.iloc[0]["contender_score"])
        else:
            metrics["champion_rank"] = -1
            metrics["champion_contender_score"] = None
    else:
        metrics["champion_rank"] = None
        metrics["champion_contender_score"] = None

    # Top-5 picks for that season
    top5 = ranked.head(5)["team"].tolist()
    metrics["top5_picks"] = ", ".join(top5)

    # Score statistics
    metrics["mean_contender"] = round(scores_df["contender_score"].mean(), 1)
    metrics["std_contender"] = round(scores_df["contender_score"].std(), 1)
    metrics["max_contender"] = round(scores_df["contender_score"].max(), 1)

    # Bracket simulation accuracy (uses historical bracket seedings if available)
    bracket_field = load_bracket_field(season)
    if bracket_field:
        bkt_champion, bkt_ff = simulate_bracket(scores_df, bracket_field)
        actual_champ = CHAMPIONS.get(season, "")
        metrics["bracket_sim_champion"] = bkt_champion
        metrics["bracket_sim_ff"] = ", ".join(t for t in bkt_ff if t)
        metrics["bracket_champion_correct"] = (bkt_champion == actual_champ) if actual_champ else None
        metrics["actual_in_bracket_ff"] = (actual_champ in bkt_ff) if actual_champ else None
    else:
        metrics["bracket_sim_champion"] = None
        metrics["bracket_sim_ff"] = None
        metrics["bracket_champion_correct"] = None
        metrics["actual_in_bracket_ff"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest March Madness model")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()

    all_metrics = []
    for season in args.seasons:
        log.info(f"Running backtest for {season}")
        scores = run_single_season(season, args.data_dir)
        metrics = evaluate_season(scores, season)
        # Carry over data-source flag so the summary can report it
        metrics["_using_real_data"] = bool(scores["_using_real_data"].iloc[0]) if "_using_real_data" in scores.columns else False
        all_metrics.append(metrics)

        rank = metrics.get("champion_rank")
        bkt_champ = metrics.get("bracket_sim_champion", "")
        bkt_correct = metrics.get("bracket_champion_correct")
        bkt_tag = ""
        if bkt_champ:
            bkt_tag = f"  │  Bracket sim: {bkt_champ} {'✓' if bkt_correct else '✗'}"
        if rank and rank > 0:
            print(f"  {season}: Champion ranked #{rank} (score: {metrics['champion_contender_score']}){bkt_tag}")
        else:
            print(f"  {season}: Champion not found in data{bkt_tag}")

    results = pd.DataFrame(all_metrics)
    # Drop internal helper column before saving
    save_cols = [c for c in results.columns if not c.startswith("_")]
    write_csv(results[save_cols], "data/outputs/backtest_results.csv")

    # Determine how many seasons used real vs. sample data
    real_seasons  = [m for m in all_metrics if m.get("_using_real_data")]
    sample_seasons = [m for m in all_metrics if not m.get("_using_real_data")]

    # Summary
    valid = results[results["champion_rank"].notna() & (results["champion_rank"] > 0)]
    if len(valid) > 0:
        print(f"\n{'=' * 60}")
        print(f"  BACKTEST SUMMARY ({len(valid)} seasons)")
        print(f"{'=' * 60}")
        print(f"  Avg champion rank:   {valid['champion_rank'].mean():.1f}")
        print(f"  Champions in top 3:  {(valid['champion_rank'] <= 3).sum()}/{len(valid)}")
        print(f"  Champions in top 5:  {(valid['champion_rank'] <= 5).sum()}/{len(valid)}")
        print(f"  Champions in top 10: {(valid['champion_rank'] <= 10).sum()}/{len(valid)}")
        print(f"  Worst rank:          {int(valid['champion_rank'].max())}")

        # Bracket simulation accuracy
        bkt_valid = results[results["bracket_champion_correct"].notna()].copy()
        if len(bkt_valid) > 0:
            bkt_correct_n = int(bkt_valid["bracket_champion_correct"].sum())
            bkt_ff_n = int(bkt_valid["actual_in_bracket_ff"].sum()) if "actual_in_bracket_ff" in bkt_valid else 0
            print(f"\n  Bracket sim (vs actual seedings):")
            print(f"  Champion predicted:  {bkt_correct_n}/{len(bkt_valid)} ({bkt_correct_n/len(bkt_valid)*100:.0f}%)")
            print(f"  Champion in sim FF:  {bkt_ff_n}/{len(bkt_valid)} ({bkt_ff_n/len(bkt_valid)*100:.0f}%)")

        if real_seasons:
            print(f"\n  ✅ Real Sports-Reference data: {len(real_seasons)} seasons")
        if sample_seasons:
            print(f"\n  ⚠️  Sample data used: {len(sample_seasons)} seasons")
            print(f"     Run scripts/fetch_sports_ref.py for those years.")


if __name__ == "__main__":
    main()
