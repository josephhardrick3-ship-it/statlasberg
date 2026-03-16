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


def run_single_season(season, data_dir=None):
    """Score one season. Returns scored DataFrame."""
    csv_path = os.path.join(data_dir, f"team_stats_{season}.csv") if data_dir else None

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "season" not in df.columns:
            df["season"] = season
    else:
        # Generate sample data (placeholder — replace with real historical data)
        df = generate_sample_data(season=season, n_teams=68)

    df = build_features(df)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df = classify_all_teams(df)
    return df


def evaluate_season(scores_df, season):
    """Evaluate model for one season against known outcomes."""
    metrics = {"season": season}

    champion = CHAMPIONS.get(season)
    if champion:
        ranked = scores_df.sort_values("contender_score", ascending=False).reset_index(drop=True)
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

    # Archetype distribution
    metrics["n_balanced_powerhouse"] = int((scores_df["archetype"] == "Balanced Powerhouse").sum())
    metrics["n_fraud_favorite"] = int((scores_df["archetype"] == "Fraud Favorite").sum())
    metrics["n_dangerous_low_seed"] = int((scores_df["archetype"] == "Dangerous Low Seed").sum())

    # Score statistics
    metrics["mean_contender"] = round(scores_df["contender_score"].mean(), 1)
    metrics["std_contender"] = round(scores_df["contender_score"].std(), 1)
    metrics["max_contender"] = round(scores_df["contender_score"].max(), 1)

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
        all_metrics.append(metrics)

        rank = metrics.get("champion_rank")
        if rank and rank > 0:
            print(f"  {season}: Champion ranked #{rank} (score: {metrics['champion_contender_score']})")
        else:
            print(f"  {season}: Champion not found in data")

    results = pd.DataFrame(all_metrics)
    write_csv(results, "data/outputs/backtest_results.csv")

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
        print(f"\n  NOTE: Using sample data. Plug in real historical CSVs for")
        print(f"  accurate backtest results via --data-dir.")


if __name__ == "__main__":
    main()
