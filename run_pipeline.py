#!/usr/bin/env python3
"""
March Madness Predictive Analysis Bot — Main Pipeline
=====================================================

Usage:
    python run_pipeline.py                     # Score teams (sample data)
    python run_pipeline.py --csv path/to/data  # Score teams from your CSV
    python run_pipeline.py --bracket bracket.csv --simulate  # Run bracket sim
    python run_pipeline.py --season 2025       # Specify season

Pipeline steps:
  1. Load raw team stats (CSV or sample data)
  2. Build derived features (percentiles, composites)
  3. Compute 0-100 sub-scores (defense, experience, guard play, clutch, rebounding)
  4. Run baseline scoring model (contender_score, upset_risk_score)
  5. Classify archetypes
  6. Generate text explanations
  7. Output team_scores.csv
  8. (Optional) Load bracket → simulate → output simulation_results.csv
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.ingest.fetch_team_stats import generate_sample_data, fetch_from_csv
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams
from src.models.classify_archetypes import classify_all_teams
from src.explain.generate_explanations import generate_all_explanations
from src.utils.io import write_csv, ensure_dirs
from src.utils.logging_utils import get_logger

log = get_logger("pipeline")


def load_data(args):
    """Step 1: Load raw team stats."""
    if args.csv and os.path.exists(args.csv):
        log.info(f"Loading team stats from {args.csv}")
        df = fetch_from_csv(args.csv)
        if "season" not in df.columns:
            df["season"] = args.season
    else:
        log.info(f"Generating sample data for {args.season} ({args.n_teams} teams)")
        df = generate_sample_data(season=args.season, n_teams=args.n_teams)
    log.info(f"Loaded {len(df)} teams")
    return df


def run_scoring_pipeline(df):
    """Steps 2-6: Features → Sub-scores → Model → Archetypes → Explanations."""
    # Step 2: Build derived features
    log.info("Step 2: Building derived features")
    df = build_features(df)

    # Step 3: Compute sub-scores
    log.info("Step 3: Computing sub-scores")
    df = compute_all_subscores(df)

    # Step 4: Run baseline scoring model
    log.info("Step 4: Running baseline scoring model")
    df = score_all_teams(df)

    # Step 5: Classify archetypes
    log.info("Step 5: Classifying archetypes")
    df = classify_all_teams(df)

    # Step 6: Generate explanations
    log.info("Step 6: Generating explanations")
    df = generate_all_explanations(df)

    return df


def run_bracket_simulation(scores_df, bracket_path, n_sims=500000):
    """Step 8: Load bracket and run Monte Carlo simulation."""
    from src.models.simulate_bracket import simulate_bracket

    log.info(f"Loading bracket from {bracket_path}")
    bracket = pd.read_csv(bracket_path)

    # Validate bracket
    required = {"region", "seed", "team"}
    if not required.issubset(set(bracket.columns)):
        log.error(f"Bracket CSV must have columns: {required}")
        return None

    log.info(f"Bracket loaded: {len(bracket)} teams across {bracket['region'].nunique()} regions")
    champ_df = simulate_bracket(bracket, scores_df, n_sims=n_sims)
    return champ_df


def main():
    parser = argparse.ArgumentParser(description="March Madness Predictive Analysis Bot")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to team stats CSV (if omitted, uses sample data)")
    parser.add_argument("--season", type=int, default=2026,
                        help="Season year (default: 2026)")
    parser.add_argument("--n-teams", type=int, default=68, dest="n_teams",
                        help="Number of teams for sample data (default: 68)")
    parser.add_argument("--bracket", type=str, default=None,
                        help="Path to bracket CSV (columns: region, seed, team)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run bracket simulation (requires --bracket)")
    parser.add_argument("--sims", type=int, default=500000,
                        help="Number of simulations (default: 500000)")
    parser.add_argument("--top", type=int, default=20,
                        help="Print top N teams (default: 20)")
    args = parser.parse_args()

    ensure_dirs()

    # ── Score all teams ──────────────────────────────
    raw_df = load_data(args)
    scores_df = run_scoring_pipeline(raw_df)

    # ── Save results ────────────────────────────────
    write_csv(scores_df, "data/outputs/team_scores.csv")
    log.info("Saved team_scores.csv")

    # ── Print summary ───────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MARCH MADNESS BOT — {args.season} TEAM RANKINGS")
    print("=" * 70)

    display_cols = ["team", "contender_score", "upset_risk_score", "archetype",
                    "expected_round", "defense_score", "experience_score",
                    "guard_play_score", "clutch_score"]
    available = [c for c in display_cols if c in scores_df.columns]
    top = scores_df.sort_values("contender_score", ascending=False).head(args.top)

    print(f"\nTop {args.top} Contenders:\n")
    print(top[available].to_string(index=False))

    # Flags
    darkhorses = scores_df[scores_df.get("title_darkhorse_flag", pd.Series(False)) == True]
    frauds = scores_df[scores_df.get("fraud_favorite_flag", pd.Series(False)) == True]
    dangerous = scores_df[scores_df.get("dangerous_low_seed_flag", pd.Series(False)) == True]

    if len(darkhorses) > 0:
        print(f"\nDARKHORSE ALERTS: {', '.join(darkhorses['team'].tolist())}")
    if len(frauds) > 0:
        print(f"FRAUD FAVORITES: {', '.join(frauds['team'].tolist())}")
    if len(dangerous) > 0:
        print(f"DANGEROUS LOW SEEDS: {', '.join(dangerous['team'].tolist())}")

    # Archetype distribution
    print(f"\nArchetype Distribution:")
    for arch, count in scores_df["archetype"].value_counts().items():
        print(f"  {arch}: {count}")

    # ── Bracket simulation ──────────────────────────
    if args.simulate:
        if not args.bracket:
            log.warning("--simulate requires --bracket. Generating sample bracket.")
            bracket_path = _generate_sample_bracket(scores_df, args.season)
            args.bracket = bracket_path

        champ_df = run_bracket_simulation(scores_df, args.bracket, n_sims=args.sims)
        if champ_df is not None and len(champ_df) > 0:
            write_csv(champ_df, "data/outputs/simulation_results.csv")
            print(f"\n{'=' * 70}")
            print(f"  BRACKET SIMULATION ({args.sims:,} runs)")
            print(f"{'=' * 70}\n")
            print("Championship Probabilities:")
            for _, row in champ_df.head(15).iterrows():
                bar = "█" * int(row["championship_pct"] / 2)
                print(f"  {row['team']:25s} {row['championship_pct']:6.2f}%  {bar}")

    print(f"\nOutputs saved to data/outputs/")
    return scores_df


def _generate_sample_bracket(scores_df, season):
    """Create a sample bracket from the top 64 teams."""
    top64 = scores_df.sort_values("contender_score", ascending=False).head(64).copy()
    regions = ["East", "West", "South", "Midwest"]
    seeds = list(range(1, 17))

    rows = []
    for i, region in enumerate(regions):
        chunk = top64.iloc[i::4].reset_index(drop=True)
        for j, seed in enumerate(seeds):
            if j < len(chunk):
                rows.append({
                    "season": season,
                    "region": region,
                    "seed": seed,
                    "team": chunk.iloc[j]["team"],
                })

    bracket_df = pd.DataFrame(rows)
    path = f"data/brackets/bracket_{season}_sample.csv"
    write_csv(bracket_df, path)
    return path


if __name__ == "__main__":
    main()
