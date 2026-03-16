"""Fetch team-level season stats.

Data sources (in order of priority):
  1. Bart Torvik (barttorvik.com) - adjusted efficiency, tempo
  2. Sports Reference (sports-reference.com/cbb) - season summaries, historical
  3. NCAA stats (stats.ncaa.org) - official NET, team stats
  4. ESPN (espn.com) - supplementary

For the MVP, this module supports:
  - Loading from local CSV (manual download / pre-scraped)
  - Generating sample data for testing

Plug in real scraping/API adapters later.
"""

import pandas as pd
import numpy as np
import os
from src.utils.io import PROJECT_ROOT


def fetch_from_csv(filepath: str) -> pd.DataFrame:
    """Load team stats from a pre-downloaded CSV."""
    return pd.read_csv(filepath)


def fetch_from_torvik(season: int) -> pd.DataFrame:
    """Fetch team stats from Bart Torvik.
    
    TODO: Implement scraping from barttorvik.com/#
    The site exposes sortable team stat tables with:
    - Adjusted offense/defense/margin
    - Tempo
    - Record breakdowns
    - Various efficiency metrics
    
    For now, returns empty DataFrame with expected columns.
    """
    print(f"  [STUB] Torvik fetch for {season} - plug in real adapter")
    return pd.DataFrame(columns=[
        "team", "conference", "wins", "losses",
        "adj_offense", "adj_defense", "adj_margin", "tempo",
        "off_rebound_pct", "def_rebound_pct", "turnover_pct",
        "opp_turnover_pct", "ft_rate", "ft_pct",
        "three_pa_rate", "three_pt_pct", "opp_three_pt_pct",
    ])


def fetch_from_sports_ref(season: int) -> pd.DataFrame:
    """Fetch team stats from Sports Reference (school-stats page).

    Checks local cache first (data/raw/teams/team_stats_{season}.csv).
    Falls back to live scrape if not cached.
    """
    import sys
    import time

    cache_path = os.path.join(PROJECT_ROOT, "data", "raw", "teams", f"team_stats_{season}.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached Sports Reference data for {season}")
        return pd.read_csv(cache_path)

    print(f"  Scraping Sports Reference for {season}...")

    # Import the scraper logic (scripts/ is a sibling of src/)
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from fetch_sports_ref import scrape_year
        df = scrape_year(season)
        if not df.empty:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"  Cached → {cache_path}")
        return df
    except Exception as e:
        print(f"  ERROR fetching Sports Reference for {season}: {e}")
        return pd.DataFrame()


def generate_sample_data(season: int = 2026, n_teams: int = 68) -> pd.DataFrame:
    """Generate realistic sample team data for testing the pipeline.
    
    Uses plausible ranges for D1 basketball stats.
    """
    np.random.seed(42 + season)
    
    # Sample team names (top programs + mid-majors)
    teams = [
        "Houston", "Connecticut", "Purdue", "North Carolina", "Duke",
        "Kansas", "Auburn", "Tennessee", "Arizona", "Gonzaga",
        "Iowa State", "Marquette", "Creighton", "Baylor", "Kentucky",
        "Alabama", "Illinois", "Texas", "Michigan State", "Oregon",
        "San Diego State", "Saint Marys CA", "Nevada", "New Mexico",
        "Drake", "Vermont", "Yale", "Princeton", "Dayton", "Memphis",
        "Colorado State", "Utah State", "Clemson", "Wisconsin", "Florida",
        "Oklahoma", "Texas Tech", "TCU", "South Carolina", "Mississippi State",
        "BYU", "Pittsburgh", "Virginia", "Northwestern", "Indiana",
        "Miami FL", "Xavier", "Providence", "Villanova", "Seton Hall",
        "Michigan", "Ohio State", "UCLA", "Colorado", "Washington State",
        "Boise State", "Grand Canyon", "James Madison", "McNeese State",
        "Oakland", "Wagner", "Colgate", "Montana State", "Samford",
        "Morehead State", "Longwood", "Stetson", "Howard",
    ][:n_teams]
    
    conferences = {
        "Houston": "Big 12", "Connecticut": "Big East", "Purdue": "Big Ten",
        "North Carolina": "ACC", "Duke": "ACC", "Kansas": "Big 12",
        "Auburn": "SEC", "Tennessee": "SEC", "Arizona": "Big 12",
        "Gonzaga": "WCC", "Iowa State": "Big 12", "Marquette": "Big East",
        "Creighton": "Big East", "Baylor": "Big 12", "Kentucky": "SEC",
        "Alabama": "SEC", "Illinois": "Big Ten", "Texas": "SEC",
        "Michigan State": "Big Ten", "Oregon": "Big Ten",
    }
    
    data = []
    for i, team in enumerate(teams):
        # Quality tier determines stat ranges
        tier = i // 17  # 0=elite, 1=good, 2=mid, 3=low
        
        adj_off = np.random.normal(115 - tier * 5, 3)
        adj_def = np.random.normal(95 + tier * 4, 3)
        adj_margin = adj_off - adj_def
        
        wins = int(np.clip(np.random.normal(28 - tier * 5, 3), 10, 35))
        losses = int(np.clip(np.random.normal(5 + tier * 4, 2), 2, 20))
        
        data.append({
            "season": season,
            "team": team,
            "conference": conferences.get(team, "Other"),
            "coach": f"Coach_{team.replace(' ', '_')}",
            "coach_years_at_school": np.random.randint(1, 20),
            "coach_ncaa_games": np.random.randint(0, 50),
            "coach_sweet16s": np.random.randint(0, 8),
            "coach_finalfours": np.random.randint(0, 4),
            "wins": wins,
            "losses": losses,
            "win_pct": round(wins / (wins + losses), 3),
            "road_wins": int(wins * np.random.uniform(0.3, 0.5)),
            "road_losses": int(losses * np.random.uniform(0.4, 0.7)),
            "neutral_wins": np.random.randint(1, 6),
            "neutral_losses": np.random.randint(0, 3),
            "net_rank": i + 1 + np.random.randint(-5, 6),
            "strength_of_schedule": round(np.random.normal(0, 5), 1),
            "quad1_wins": np.random.randint(max(0, 8 - tier * 3), 12 - tier * 2),
            "quad2_wins": np.random.randint(2, 8),
            "adj_offense": round(adj_off, 1),
            "adj_defense": round(adj_def, 1),
            "adj_margin": round(adj_margin, 1),
            "tempo": round(np.random.normal(68, 3), 1),
            "turnover_pct": round(np.random.normal(16.5, 2), 1),
            "opp_turnover_pct": round(np.random.normal(17, 2), 1),
            "off_rebound_pct": round(np.random.normal(30, 3), 1),
            "def_rebound_pct": round(np.random.normal(72, 3), 1),
            "ft_rate": round(np.random.normal(32, 4), 1),
            "ft_pct": round(np.random.normal(73, 4), 1),
            "three_pa_rate": round(np.random.normal(37, 4), 1),
            "three_pt_pct": round(np.random.normal(34, 3), 1),
            "opp_three_pt_pct": round(np.random.normal(33, 2.5), 1),
            "points_per_game": round(adj_off * 0.68 + np.random.normal(0, 2), 1),
            "points_allowed_per_game": round(adj_def * 0.68 + np.random.normal(0, 2), 1),
            "close_game_record": round(np.random.uniform(0.3, 0.8), 3),
            "last10_win_pct": round(np.random.uniform(0.5, 1.0), 3),
            "last10_adj_margin": round(adj_margin + np.random.normal(0, 3), 1),
            "avg_age": round(np.random.normal(20.8, 0.8), 1),
            "underclass_minutes_pct": round(np.random.uniform(20, 60), 1),
            "freshman_minutes_pct": round(np.random.uniform(5, 40), 1),
            "freshman_guard_minutes_pct": round(np.random.uniform(0, 30), 1),
            "sophomore_minutes_pct": round(np.random.uniform(10, 35), 1),
            "junior_minutes_pct": round(np.random.uniform(15, 40), 1),
            "senior_minutes_pct": round(np.random.uniform(10, 50), 1),
            "returning_starters": np.random.randint(1, 5),
            "returning_minutes_pct": round(np.random.uniform(30, 85), 1),
            "primary_guard_experience_score": round(np.random.uniform(30, 95), 1),
            "backcourt_experience_score": round(np.random.uniform(30, 95), 1),
            "frontcourt_experience_score": round(np.random.uniform(30, 95), 1),
            "bench_minutes_pct": round(np.random.uniform(20, 45), 1),
            "bench_points_pct": round(np.random.uniform(15, 40), 1),
            "injury_flag": int(np.random.random() < 0.15),
            "star_player_flag": int(np.random.random() < 0.3),
            # Regional counts (will be overwritten by player origin pipeline)
            "chicago_guard_count": np.random.randint(0, 3),
            "nyc_guard_count": np.random.randint(0, 3),
            "indiana_guard_count": np.random.randint(0, 2),
            "texas_big_count": np.random.randint(0, 3),
            "southern_big_count": np.random.randint(0, 3),
            "west_coast_wing_count": np.random.randint(0, 3),
            "local_site_player_count": np.random.randint(0, 4),
            "host_state_player_count": np.random.randint(0, 3),
        })
    
    return pd.DataFrame(data)
