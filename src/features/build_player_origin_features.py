"""Build player origin features and regional bias counts.

Maps each player to a development region based on their hometown/state,
then aggregates counts per team for the regional bias layer.

Day 3 focus - plugs into team features as supplementary data.
"""

import pandas as pd
from src.utils.io import load_region_map, write_csv
from src.utils.logging_utils import get_logger

log = get_logger("build_player_origin")


def tag_player_regions(roster_df: pd.DataFrame) -> pd.DataFrame:
    """Tag each player with their regional development bias.
    
    Args:
        roster_df: Cleaned roster with home_state, position columns.
    
    Returns:
        roster_df with added region flag columns.
    """
    region_map = load_region_map()
    regions = region_map.get("regions", {})
    positions_map = region_map.get("positions", {})
    
    df = roster_df.copy()
    
    # Determine position bucket
    guard_pos = set(positions_map.get("guard", []))
    wing_pos = set(positions_map.get("wing", []))
    big_pos = set(positions_map.get("big", []))
    
    df["is_guard"] = df["position"].isin(guard_pos).astype(int)
    df["is_wing"] = df["position"].isin(wing_pos).astype(int)
    df["is_big"] = df["position"].isin(big_pos).astype(int)
    
    # Tag each region
    df["bias_region"] = "other"
    df["from_chicagoland"] = 0
    df["from_nyc"] = 0
    df["from_indiana"] = 0
    df["from_texas"] = 0
    df["from_south"] = 0
    df["from_west_coast"] = 0
    
    for region_key, region_info in regions.items():
        mask = df["home_state"].isin(region_info.get("states", []))
        col = f"from_{region_key}"
        if col in df.columns:
            df.loc[mask, col] = 1
            df.loc[mask, "bias_region"] = region_key
    
    return df


def aggregate_team_region_counts(roster_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player origin counts per team.
    
    Returns one row per team with regional counts.
    """
    region_map = load_region_map()
    regions = region_map.get("regions", {})
    
    df = roster_df.copy()
    
    # Only count players who match the region's position bias
    results = []
    for team, group in df.groupby(["season", "team"]):
        row = {"season": team[0], "team": team[1]}
        
        for region_key, region_info in regions.items():
            pos_bias = region_info.get("position_bias", "any")
            region_players = group[group[f"from_{region_key}"] == 1]
            
            if pos_bias == "guard":
                count = region_players["is_guard"].sum()
            elif pos_bias == "wing":
                count = region_players["is_wing"].sum()
            elif pos_bias == "big":
                count = region_players["is_big"].sum()
            else:
                count = len(region_players)
            
            col_map = {
                "chicagoland": "chicago_guard_count",
                "nyc": "nyc_guard_count",
                "indiana": "indiana_guard_count",
                "texas": "texas_big_count",
                "south": "southern_big_count",
                "west_coast": "west_coast_wing_count",
            }
            col = col_map.get(region_key, f"{region_key}_count")
            row[col] = int(count)
        
        results.append(row)
    
    return pd.DataFrame(results)


def save_player_origins(df: pd.DataFrame):
    """Save tagged player origins."""
    write_csv(df, "data/features/player_origin_features.csv")
    log.info("Saved player origin features")
