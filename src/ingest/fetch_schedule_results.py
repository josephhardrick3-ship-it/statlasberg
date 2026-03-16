"""Fetch game schedules and results.

Sources:
  - ESPN scoreboards
  - Sports Reference game logs
  - Manual CSV

For MVP, supports CSV loading.
"""

import pandas as pd


def fetch_from_csv(filepath: str) -> pd.DataFrame:
    """Load schedule/results from CSV."""
    return pd.read_csv(filepath)


def fetch_from_sports_ref(team: str, season: int) -> pd.DataFrame:
    """Fetch game log from Sports Reference.
    
    TODO: Implement scraping from
    sports-reference.com/cbb/schools/{team}/{season}-gamelogs.html
    """
    print(f"  [STUB] Sports Ref schedule for {team} {season}")
    return pd.DataFrame()
