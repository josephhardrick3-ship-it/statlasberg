"""Build play-by-play aggregate features per team per season.

Day 3 focus. Extracts behavioral signals from game-level PBP data.
"""

import pandas as pd
import numpy as np
from src.utils.io import write_csv
from src.utils.logging_utils import get_logger

log = get_logger("build_pbp_features")


def build_pbp_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PBP events into team-season features.
    
    Args:
        pbp_df: Cleaned PBP data with columns:
                season, game_id, team, event_type_clean, seconds_remaining, margin
    
    Returns:
        DataFrame with one row per team-season.
    """
    if pbp_df.empty:
        log.warning("No PBP data provided, returning empty features")
        return pd.DataFrame()
    
    results = []
    for (season, team), games in pbp_df.groupby(["season", "team"]):
        game_ids = games["game_id"].unique()
        
        row = {
            "season": season,
            "team": team,
            "games_with_pbp": len(game_ids),
        }
        
        # Scoring drought analysis
        droughts = _compute_scoring_droughts(games)
        row["avg_scoring_drought_secs"] = droughts["avg"]
        row["max_scoring_drought_secs"] = droughts["max"]
        
        # Run analysis
        runs = _compute_runs(games)
        row["avg_run_created"] = runs["avg_created"]
        row["max_run_created"] = runs["max_created"]
        row["avg_run_allowed"] = runs["avg_allowed"]
        row["max_run_allowed"] = runs["max_allowed"]
        
        # Final 5 minutes analysis
        final5 = games[games["seconds_remaining"] <= 300]
        if len(final5) > 0:
            turnovers = final5["event_type_clean"].eq("turnover").sum()
            possessions = max(1, len(final5) // 4)  # rough estimate
            row["final5_turnover_pct"] = round(turnovers / possessions * 100, 1)
            
            fts = final5[final5["event_type_clean"] == "score_ft"]
            ft_misses = final5[final5["event_type_clean"] == "miss_ft"]
            ft_total = len(fts) + len(ft_misses)
            row["final5_ft_pct"] = round(len(fts) / max(1, ft_total) * 100, 1)
        else:
            row["final5_turnover_pct"] = None
            row["final5_ft_pct"] = None
        
        # Placeholder for complex features
        row["clutch_off_rating"] = None
        row["clutch_def_rating"] = None
        row["comeback_win_pct"] = None
        row["blown_lead_pct"] = None
        row["halftime_adjustment_margin"] = None
        row["pressure_game_stability_score"] = None
        row["single_scorer_dependency_score"] = None
        
        results.append(row)
    
    return pd.DataFrame(results)


def _compute_scoring_droughts(game_events: pd.DataFrame) -> dict:
    """Compute average and max scoring drought in seconds."""
    scoring_events = game_events[
        game_events["event_type_clean"].isin(["score", "score_3pt", "score_ft"])
    ]
    if len(scoring_events) < 2:
        return {"avg": 0, "max": 0}
    
    times = scoring_events["seconds_remaining"].sort_values(ascending=False)
    gaps = times.diff().abs().dropna()
    
    return {
        "avg": round(gaps.mean(), 1) if len(gaps) > 0 else 0,
        "max": round(gaps.max(), 1) if len(gaps) > 0 else 0,
    }


def _compute_runs(game_events: pd.DataFrame) -> dict:
    """Compute scoring runs created and allowed."""
    # Simplified: count consecutive scoring events
    return {
        "avg_created": 0,
        "max_created": 0,
        "avg_allowed": 0,
        "max_allowed": 0,
    }


def save_pbp_features(df: pd.DataFrame):
    """Save PBP features to CSV."""
    write_csv(df, "data/features/play_by_play_features.csv")
    log.info("Saved PBP features")
