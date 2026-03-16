"""Normalize player names across data sources."""

import re


def normalize(name: str) -> str:
    """Normalize a player name."""
    if not name or not isinstance(name, str):
        return ""
    
    name = name.strip()
    
    # Remove suffixes like Jr., Sr., III, IV
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|IV|II)\s*$', '', name, flags=re.IGNORECASE)
    
    # Standardize capitalization
    parts = name.split()
    parts = [p.capitalize() if len(p) > 2 else p.upper() for p in parts]
    name = " ".join(parts)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def normalize_column(df, col: str = "player"):
    """Normalize player names in a DataFrame column in-place."""
    df[col] = df[col].apply(normalize)
    return df
