"""Fetch play-by-play data for games.

Sources:
  - ESPN game pages (play-by-play tab)
  - NCAA game details
  
For MVP, supports CSV loading. Day 3 work adds PBP extraction.
"""

import pandas as pd


def fetch_from_csv(filepath: str) -> pd.DataFrame:
    """Load PBP data from CSV."""
    return pd.read_csv(filepath)


def fetch_from_espn(game_id: str) -> pd.DataFrame:
    """Fetch PBP from ESPN game page.
    
    TODO: Implement scraping from
    espn.com/mens-college-basketball/playbyplay/_/gameId/{game_id}
    """
    print(f"  [STUB] ESPN PBP for game {game_id}")
    return pd.DataFrame()
