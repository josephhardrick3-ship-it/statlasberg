"""Build outcome labels for training data from tournament results."""

import pandas as pd


def add_tournament_labels(
    team_features: pd.DataFrame,
    tournament_results: pd.DataFrame
) -> pd.DataFrame:
    """Merge tournament outcome labels onto team features.
    
    Args:
        team_features: Team season features.
        tournament_results: DataFrame with season, team, and round columns
                           indicating how far each team advanced.
    
    Returns:
        team_features with label columns added.
    """
    df = team_features.copy()
    
    label_cols = [
        "made_tournament", "won_round64", "won_round32",
        "made_sweet16", "made_elite8", "made_final4", "won_title",
    ]
    
    for col in label_cols:
        if col not in df.columns:
            df[col] = 0
    
    if tournament_results.empty:
        return df
    
    # Map round reached to binary labels
    round_to_labels = {
        "Round of 64": ["made_tournament"],
        "Round of 32": ["made_tournament", "won_round64"],
        "Sweet 16": ["made_tournament", "won_round64", "won_round32", "made_sweet16"],
        "Elite 8": ["made_tournament", "won_round64", "won_round32", "made_sweet16", "made_elite8"],
        "Final Four": ["made_tournament", "won_round64", "won_round32", "made_sweet16", "made_elite8", "made_final4"],
        "Champion": ["made_tournament", "won_round64", "won_round32", "made_sweet16", "made_elite8", "made_final4", "won_title"],
    }
    
    for _, row in tournament_results.iterrows():
        team_mask = (df["season"] == row["season"]) & (df["team"] == row["team"])
        labels = round_to_labels.get(row.get("round_reached", ""), [])
        for label in labels:
            df.loc[team_mask, label] = 1
    
    return df
