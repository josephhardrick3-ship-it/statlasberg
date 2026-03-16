"""Fetch historical tournament brackets and results.

Sources:
  - Sports Reference tournament pages
  - NCAA tournament history
  - Manual bracket CSVs in data/brackets/

Each bracket file should have: season, region, seed, team
Results should add: won_round64, won_round32, made_sweet16, etc.
"""

import os
import pandas as pd
from src.utils.io import PROJECT_ROOT


def load_bracket(season: int) -> pd.DataFrame:
    """Load a bracket CSV for a given season."""
    path = os.path.join(PROJECT_ROOT, f"data/brackets/{season}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [WARNING] No bracket file found for {season}")
    return pd.DataFrame(columns=["season", "region", "seed", "team"])


def load_all_brackets() -> pd.DataFrame:
    """Load all available bracket files."""
    bracket_dir = os.path.join(PROJECT_ROOT, "data/brackets")
    frames = []
    for f in sorted(os.listdir(bracket_dir)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(bracket_dir, f))
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["season", "region", "seed", "team"])


def generate_sample_bracket(season: int = 2026) -> pd.DataFrame:
    """Generate a sample 68-team bracket for testing."""
    regions = ["East", "West", "South", "Midwest"]
    seeds = list(range(1, 17))
    
    # Use sample teams from fetch_team_stats
    from src.ingest.fetch_team_stats import generate_sample_data
    teams_df = generate_sample_data(season, 68)
    team_list = teams_df["team"].tolist()
    
    rows = []
    idx = 0
    for region in regions:
        for seed in seeds:
            if idx < len(team_list):
                rows.append({
                    "season": season,
                    "region": region,
                    "seed": seed,
                    "team": team_list[idx],
                })
                idx += 1
    
    # Add play-in teams (4 extra)
    for i in range(min(4, len(team_list) - idx)):
        rows.append({
            "season": season,
            "region": regions[i],
            "seed": 16,  # Play-in seeds
            "team": team_list[idx + i] if idx + i < len(team_list) else f"PlayIn_{i}",
        })
    
    return pd.DataFrame(rows)
