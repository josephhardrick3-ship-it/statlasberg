"""Build the team_season_features.csv from raw data.

This is the main feature table. One row per team per season.
Combines team stats, roster experience, and regional bias counts.
"""

import pandas as pd
import numpy as np
from src.utils.io import load_feature_weights, write_csv
from src.utils.logging_utils import get_logger

log = get_logger("build_team_features")


def build_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Build the full team feature table from raw team stats.
    
    Args:
        team_stats: DataFrame with team-level season stats.
                    Must have columns: season, team, adj_offense, adj_defense, etc.
    
    Returns:
        DataFrame with all computed features added.
    """
    df = team_stats.copy()
    log.info(f"Building features for {len(df)} teams")
    
    # Compute derived features
    df = _add_quality_percentiles(df)
    df = _add_experience_composites(df)
    df = _add_style_features(df)
    
    return df


def _add_quality_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile ranks for key quality metrics."""
    for col in ["adj_offense", "adj_defense", "adj_margin", "turnover_pct",
                "off_rebound_pct", "def_rebound_pct", "ft_pct"]:
        if col in df.columns:
            # For defense and turnover_pct, lower is better
            if col in ["adj_defense", "turnover_pct"]:
                df[f"{col}_pctile"] = df[col].rank(ascending=True, pct=True) * 100
            else:
                df[f"{col}_pctile"] = df[col].rank(ascending=False, pct=True) * 100
    return df


def _add_experience_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite experience scores."""
    # Backcourt experience: weighted combo of guard-related experience
    if "backcourt_experience_score" not in df.columns:
        df["backcourt_experience_score"] = (
            df.get("senior_minutes_pct", pd.Series(50, index=df.index)) * 0.4 +
            df.get("avg_age", pd.Series(20.5, index=df.index)).apply(
                lambda x: min(100, max(0, (x - 19) / 3 * 100))
            ) * 0.3 +
            df.get("returning_starters", pd.Series(2, index=df.index)).apply(
                lambda x: min(100, x / 5 * 100)
            ) * 0.3
        ).round(1)
    return df


def _add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add style-related composite features."""
    # Scoring balance (inverse of concentration - placeholder)
    if "bench_points_pct" in df.columns:
        df["scoring_balance"] = df["bench_points_pct"].clip(10, 50)
    
    # Road/neutral win %
    if "neutral_wins" in df.columns and "neutral_losses" in df.columns:
        total = df["neutral_wins"] + df["neutral_losses"]
        df["neutral_win_pct"] = np.where(total > 0, df["neutral_wins"] / total, 0.5)
    
    if "road_wins" in df.columns and "road_losses" in df.columns:
        total = df["road_wins"] + df["road_losses"]
        df["road_win_pct"] = np.where(total > 0, df["road_wins"] / total, 0.5)
    
    # Combined road/neutral
    if "road_win_pct" in df.columns and "neutral_win_pct" in df.columns:
        df["road_neutral_win_pct"] = (df["road_win_pct"] + df["neutral_win_pct"]) / 2
    
    return df


def save_features(df: pd.DataFrame, season: int = None):
    """Save the team features to CSV."""
    path = "data/features/team_season_features.csv"
    write_csv(df, path)
    log.info(f"Saved team features to {path}")
