"""Clean and standardize roster data."""

import pandas as pd
from src.clean.normalize_team_names import normalize as norm_team
from src.clean.normalize_player_names import normalize as norm_player


def clean_roster(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw roster DataFrame.
    
    Expected columns: season, team, player, position, height, class_year,
                      home_city, home_state
    """
    df = df.copy()
    
    # Normalize names
    df["team"] = df["team"].apply(norm_team)
    df["player"] = df["player"].apply(norm_player)
    
    # Standardize class year
    class_map = {
        "Fr.": "FR", "Fr": "FR", "Freshman": "FR",
        "So.": "SO", "So": "SO", "Sophomore": "SO",
        "Jr.": "JR", "Jr": "JR", "Junior": "JR",
        "Sr.": "SR", "Sr": "SR", "Senior": "SR",
        "GR": "SR", "Gr": "SR", "Graduate": "SR",  # Grad students count as seniors
    }
    if "class_year" in df.columns:
        df["class_year"] = df["class_year"].map(class_map).fillna(df["class_year"])
    
    # Standardize position
    pos_map = {
        "Point Guard": "PG", "Shooting Guard": "SG", "Small Forward": "SF",
        "Power Forward": "PF", "Center": "C", "Guard": "G", "Forward": "F",
    }
    if "position" in df.columns:
        df["position"] = df["position"].map(pos_map).fillna(df["position"])
    
    # Parse height to inches
    if "height" in df.columns:
        df["height_inches"] = df["height"].apply(_parse_height)
    
    # Clean state names
    if "home_state" in df.columns:
        df["home_state"] = df["home_state"].str.strip().str.upper()
    
    return df


def _parse_height(h) -> float:
    """Parse height string like '6-4' or '6\'4\"' to inches."""
    if pd.isna(h):
        return None
    h = str(h).strip()
    
    # Try "6-4" format
    parts = h.replace("'", "-").replace('"', '').replace("–", "-").split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except ValueError:
            return None
    return None
