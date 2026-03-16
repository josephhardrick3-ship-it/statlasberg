"""Clean play-by-play data for feature extraction.

This module standardizes raw PBP event data into a consistent format
for downstream feature building.
"""

import pandas as pd
from src.clean.normalize_team_names import normalize as norm_team


def clean_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw play-by-play data.
    
    Expected columns: season, game_id, team, opponent, period, time_remaining,
                      event_type, player, points_scored, team_score, opp_score
    """
    df = df.copy()
    
    # Normalize team names
    for col in ["team", "opponent"]:
        if col in df.columns:
            df[col] = df[col].apply(norm_team)
    
    # Standardize event types
    event_map = {
        "made shot": "score", "made layup": "score", "made dunk": "score",
        "made 3pt": "score_3pt", "made free throw": "score_ft",
        "missed shot": "miss", "missed 3pt": "miss_3pt",
        "missed free throw": "miss_ft",
        "turnover": "turnover", "steal": "steal",
        "offensive rebound": "oreb", "defensive rebound": "dreb",
        "foul": "foul", "personal foul": "foul",
        "timeout": "timeout", "substitution": "sub",
    }
    if "event_type" in df.columns:
        df["event_type_clean"] = df["event_type"].str.lower().map(event_map).fillna("other")
    
    # Parse time remaining to seconds
    if "time_remaining" in df.columns:
        df["seconds_remaining"] = df["time_remaining"].apply(_parse_time)
    
    # Calculate margin at each event
    if "team_score" in df.columns and "opp_score" in df.columns:
        df["margin"] = df["team_score"] - df["opp_score"]
    
    return df


def _parse_time(t) -> float:
    """Parse time string like '12:34' to seconds remaining."""
    if pd.isna(t):
        return None
    t = str(t).strip()
    parts = t.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None
