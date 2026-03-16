"""Fetch roster data for tournament teams.

Sources:
  - ESPN team roster pages
  - Sports Reference player pages
  - Manual CSV input

For MVP, supports CSV loading and sample generation.
"""

import pandas as pd
import numpy as np


def fetch_from_csv(filepath: str) -> pd.DataFrame:
    """Load roster data from a pre-downloaded CSV."""
    return pd.read_csv(filepath)


def generate_sample_rosters(teams: list, season: int = 2026) -> pd.DataFrame:
    """Generate sample roster data for testing."""
    np.random.seed(99 + season)
    
    positions = ["PG", "SG", "SF", "PF", "C", "G", "F"]
    classes = ["FR", "SO", "JR", "SR"]
    states = ["IL", "NY", "IN", "TX", "CA", "GA", "FL", "OH", "NJ", "NC",
              "MI", "PA", "VA", "TN", "MD", "WA", "OR", "AL", "SC", "LA"]
    cities = {
        "IL": ["Chicago", "Naperville", "Aurora"],
        "NY": ["New York", "Brooklyn", "Queens"],
        "IN": ["Indianapolis", "Fort Wayne", "Gary"],
        "TX": ["Houston", "Dallas", "San Antonio"],
        "CA": ["Los Angeles", "Oakland", "San Diego"],
        "GA": ["Atlanta", "Savannah"],
        "FL": ["Miami", "Orlando", "Tampa"],
        "OH": ["Columbus", "Cleveland"],
        "NJ": ["Newark", "Jersey City"],
        "NC": ["Charlotte", "Raleigh"],
    }
    
    rows = []
    for team in teams:
        n_players = np.random.randint(10, 15)
        for j in range(n_players):
            state = np.random.choice(states)
            city_options = cities.get(state, [f"{state}_City"])
            rows.append({
                "season": season,
                "team": team,
                "player": f"Player_{team[:3]}_{j+1}",
                "position": np.random.choice(positions),
                "height": f"{np.random.randint(5,7)}-{np.random.randint(0,12)}",
                "class_year": np.random.choice(classes, p=[0.25, 0.25, 0.25, 0.25]),
                "starter_flag": int(j < 5),
                "minutes_pct": round(np.random.uniform(2, 25) if j >= 5 else np.random.uniform(20, 35), 1),
                "usage_pct": round(np.random.uniform(10, 30), 1),
                "home_city": np.random.choice(city_options),
                "home_state": state,
                "high_school": f"HS_{state}_{j}",
                "aau_program": "",
            })
    
    return pd.DataFrame(rows)
