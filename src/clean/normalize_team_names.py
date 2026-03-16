"""Normalize team names across different data sources.

Team names vary wildly between ESPN, Sports Reference, NCAA, and Torvik.
This module provides a single canonical mapping.
"""

import re
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
from src.utils.constants import TEAM_ALIASES


# Master canonical names - add to this as you encounter new variations
_CANONICAL_NAMES = {}  # Will be populated from data


def normalize(name: str) -> str:
    """Normalize a team name to its canonical form.
    
    Steps:
    1. Strip whitespace
    2. Check exact alias match
    3. Clean common patterns (St./Saint, parenthetical)
    4. Fuzzy match against known names if needed
    """
    if not name or not isinstance(name, str):
        return ""
    
    name = name.strip()
    
    # Check aliases first
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    
    # Standard cleaning
    cleaned = _clean_name(name)
    
    # Check aliases again with cleaned version
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    
    return cleaned


def _clean_name(name: str) -> str:
    """Apply standard cleaning rules."""
    # Remove trailing seed numbers like "(1)" or "#1"
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    name = re.sub(r'\s*#\d+\s*', ' ', name)
    
    # Standardize St. / Saint
    name = re.sub(r"^St\.\s", "Saint ", name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def fuzzy_match(name: str, known_names: list, threshold: int = 85) -> str:
    """Fuzzy match a name against a list of known canonical names.
    
    Returns the best match above threshold, or the original name.
    Falls back to exact matching if rapidfuzz is not installed.
    """
    if not known_names:
        return name
    
    if HAS_RAPIDFUZZ:
        result = process.extractOne(name, known_names, scorer=fuzz.ratio)
        if result and result[1] >= threshold:
            return result[0]
    else:
        # Simple fallback: case-insensitive exact match
        lower_map = {n.lower(): n for n in known_names}
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return name


def build_canonical_map(team_lists: list) -> dict:
    """Build a mapping from all observed names to canonical names.
    
    Args:
        team_lists: List of lists of team names from different sources.
    
    Returns:
        Dict mapping each observed name to its canonical form.
    """
    all_names = set()
    for tl in team_lists:
        all_names.update(tl)
    
    canonical = {}
    for name in sorted(all_names):
        canonical[name] = normalize(name)
    
    return canonical


def normalize_column(df, col: str = "team"):
    """Normalize team names in a DataFrame column in-place."""
    df[col] = df[col].apply(normalize)
    return df
