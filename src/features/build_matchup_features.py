"""Build matchup features for head-to-head comparisons.

Used after the bracket is set to evaluate specific pairings.
"""

import pandas as pd
import numpy as np
from src.utils.io import load_feature_weights
from src.utils.logging_utils import get_logger

log = get_logger("build_matchup_features")


def build_matchup(team_a: pd.Series, team_b: pd.Series) -> dict:
    """Compare two teams and compute matchup edges.
    
    Args:
        team_a, team_b: Rows from team_season_features.
    
    Returns:
        Dict with edge scores and overall matchup score.
    """
    weights = load_feature_weights().get("matchup_score", {})
    
    edges = {}
    
    # Turnover edge: team with better (lower) turnover rate
    edges["turnover_edge"] = _compute_edge(
        team_a.get("turnover_pct", 16), team_b.get("turnover_pct", 16),
        lower_is_better=True
    )
    
    # Rebound edge
    edges["rebound_edge"] = _compute_edge(
        team_a.get("off_rebound_pct", 30), team_b.get("off_rebound_pct", 30),
        lower_is_better=False
    )
    
    # Guard experience edge
    edges["guard_experience_edge"] = _compute_edge(
        team_a.get("backcourt_experience_score", 50),
        team_b.get("backcourt_experience_score", 50),
        lower_is_better=False
    )
    
    # Defense edge (lower adj_defense is better)
    edges["defense_edge"] = _compute_edge(
        team_a.get("adj_defense", 100), team_b.get("adj_defense", 100),
        lower_is_better=True
    )
    
    # Tempo edge: team that controls pace better
    # If team_a is defensive-minded and team_b needs fast pace, advantage team_a
    edges["tempo_edge"] = _compute_edge(
        team_a.get("adj_defense", 100), team_b.get("tempo", 68),
        lower_is_better=True
    ) * 0.5  # Reduced weight - this is a rough proxy
    
    # Shooting edge
    edges["shooting_edge"] = _compute_edge(
        team_a.get("three_pt_pct", 34), team_b.get("three_pt_pct", 34),
        lower_is_better=False
    )
    
    # Compute weighted matchup score
    matchup_score = sum(
        edges.get(k, 0) * weights.get(k, 0)
        for k in weights
    )
    edges["matchup_score"] = round(matchup_score, 2)
    
    # Win probability (sigmoid of matchup score + seed difference)
    seed_a = team_a.get("seed", 8)
    seed_b = team_b.get("seed", 8)
    seed_diff = seed_b - seed_a  # Positive = team_a has better seed
    
    logit = matchup_score * 2 + seed_diff * 0.05
    win_prob = 1 / (1 + np.exp(-logit))
    edges["team_a_win_prob"] = round(win_prob, 3)
    
    return edges


def _compute_edge(val_a: float, val_b: float, lower_is_better: bool = False) -> float:
    """Compute normalized edge between two values.
    
    Returns positive if team_a has advantage, negative if team_b.
    Range roughly -1 to 1.
    """
    if lower_is_better:
        diff = val_b - val_a
    else:
        diff = val_a - val_b
    
    # Normalize to roughly -1 to 1
    scale = max(abs(val_a), abs(val_b), 1)
    return round(diff / scale, 3)


def build_all_matchups(bracket_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Build matchup features for all first-round bracket pairings."""
    matchups = []
    
    for region in bracket_df["region"].unique():
        region_teams = bracket_df[bracket_df["region"] == region].sort_values("seed")
        seeds = region_teams["seed"].tolist()
        teams = region_teams["team"].tolist()
        
        # Standard bracket pairings: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
        
        for i, j in pairings:
            if i < len(teams) and j < len(teams):
                team_a_name = teams[i]
                team_b_name = teams[j]
                
                a_row = features_df[features_df["team"] == team_a_name]
                b_row = features_df[features_df["team"] == team_b_name]
                
                if len(a_row) > 0 and len(b_row) > 0:
                    edges = build_matchup(a_row.iloc[0], b_row.iloc[0])
                    edges["team"] = team_a_name
                    edges["opponent"] = team_b_name
                    edges["team_seed"] = seeds[i] if i < len(seeds) else None
                    edges["opponent_seed"] = seeds[j] if j < len(seeds) else None
                    edges["region"] = region
                    matchups.append(edges)
    
    return pd.DataFrame(matchups)
