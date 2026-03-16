#!/usr/bin/env python3
"""
Merge Torvik player class/experience data into per-year team_stats CSVs.

Input:  data/raw/teams/torvik_player_class_2015_2026.csv
        data/raw/teams/team_stats_{year}.csv (one per year)

Adds:   avg_age, freshman_minutes_pct, sophomore_minutes_pct,
        junior_minutes_pct, senior_minutes_pct, underclass_minutes_pct,
        returning_starters (approximated from senior share of starters)

Usage:
    python scripts/merge_player_stats.py
"""

import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ROOT   = os.path.dirname(SCRIPT_DIR)
TEAMS_DIR  = os.path.join(BOT_ROOT, "data", "raw", "teams")

PLAYER_CSV = os.path.join(TEAMS_DIR, "torvik_player_class_2015_2026.csv")
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Avg class → avg_age offset (Fr=1→~18.5, So=2→~19.5, Jr=3→~20.5, Sr=4→~21.5)
CLASS_TO_AGE_OFFSET = 17.5

# Team name normalizations from Torvik player data → bot schema
TEAM_ALIASES = {
    "UConn": "Connecticut", "UCONN": "Connecticut",
    "Ole Miss": "Mississippi", "SMU": "Southern Methodist",
    "LSU": "Louisiana State", "USC": "Southern California",
    "UCF": "Central Florida", "TCU": "Texas Christian",
    "Pitt": "Pittsburgh", "BYU": "Brigham Young",
    "Saint Mary's (CA)": "Saint Marys CA",
    "St. Mary's (CA)": "Saint Marys CA",
    "Miami (FL)": "Miami FL",
    "St. John's (NY)": "St Johns NY",
    "Saint John's": "St Johns NY",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "UNC": "North Carolina",
    "Ohio St.": "Ohio State",
    "Mich. St.": "Michigan State",
    "Fla.": "Florida",
    "Penn St.": "Penn State",
    "Okla. St.": "Oklahoma State",
    "Ariz. St.": "Arizona State",
    "Colo. St.": "Colorado State",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal St. Northridge": "Cal State Northridge",
    "UTSA": "UT San Antonio",
    "UIW": "Incarnate Word",
    "LIU": "Long Island University",
    "Jax. St.": "Jacksonville State",
    "SE Missouri St.": "Southeast Missouri State",
    "SIU Edwardsville": "SIU-Edwardsville",
    "Ga. Southern": "Georgia Southern",
    "Southern Miss.": "Southern Mississippi",
    "La. Tech": "Louisiana Tech",
    "Fla. Atlantic": "Florida Atlantic",
    "N. Illinois": "Northern Illinois",
    "E. Tennessee St.": "East Tennessee State",
    "Abilene Chr.": "Abilene Christian",
    "UNC Greensboro": "UNC Greensboro",
}


def normalize_torvik_team(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def fuzzy_match(name: str, choices: list, threshold: int = 80):
    result = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def merge_year(player_df: pd.DataFrame, year: int) -> None:
    sr_path = os.path.join(TEAMS_DIR, f"team_stats_{year}.csv")
    if not os.path.exists(sr_path):
        print(f"  [{year}] No team stats file, skipping")
        return

    sr = pd.read_csv(sr_path)
    py = player_df[player_df["season"] == year].copy()

    if py.empty:
        print(f"  [{year}] No player data, skipping")
        return

    # Normalize team names in player data
    py["team_norm"] = py["team"].apply(normalize_torvik_team)
    py_lookup = py.set_index("team_norm")

    sr_teams = sr["team"].tolist()
    torvik_teams = py_lookup.index.tolist()

    matched = fuzzy_matched = 0
    unmatched = []

    exp_cols = [
        "avg_age", "freshman_minutes_pct", "sophomore_minutes_pct",
        "junior_minutes_pct", "senior_minutes_pct", "underclass_minutes_pct",
        "returning_minutes_pct",
    ]
    for col in exp_cols:
        if col not in sr.columns:
            sr[col] = np.nan

    for idx, row in sr.iterrows():
        sr_name = row["team"]

        if sr_name in py_lookup.index:
            p_row = py_lookup.loc[sr_name]
            matched += 1
        else:
            best = fuzzy_match(sr_name, torvik_teams)
            if best:
                p_row = py_lookup.loc[best]
                fuzzy_matched += 1
            else:
                unmatched.append(sr_name)
                continue

        # Handle duplicate index (multiple rows for same team — take first)
        if isinstance(p_row, pd.DataFrame):
            p_row = p_row.iloc[0]

        avg_class = float(p_row["avg_class"])
        fr_pct  = float(p_row["freshman_min_pct"])
        so_pct  = float(p_row["sophomore_min_pct"])
        jr_pct  = float(p_row["junior_min_pct"])
        sr_pct  = float(p_row["senior_min_pct"])
        sr.at[idx, "avg_age"] = round(avg_class + CLASS_TO_AGE_OFFSET, 2)
        sr.at[idx, "freshman_minutes_pct"]   = fr_pct
        sr.at[idx, "sophomore_minutes_pct"]  = so_pct
        sr.at[idx, "junior_minutes_pct"]     = jr_pct
        sr.at[idx, "senior_minutes_pct"]     = sr_pct
        sr.at[idx, "underclass_minutes_pct"] = float(p_row["underclass_min_pct"])
        # returning_minutes_pct = all non-freshmen (played last season)
        sr.at[idx, "returning_minutes_pct"]  = round(so_pct + jr_pct + sr_pct, 1)

    print(f"  [{year}] exact: {matched} | fuzzy: {fuzzy_matched} | unmatched: {len(unmatched)}")
    if unmatched:
        print(f"          Unmatched: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    sr.to_csv(sr_path, index=False)


def main():
    print(f"Loading player class data from {PLAYER_CSV}")
    player_df = pd.read_csv(PLAYER_CSV)
    player_df["season"] = pd.to_numeric(player_df["season"], errors="coerce")
    print(f"  {len(player_df)} team-season rows, years: {sorted(player_df['season'].dropna().unique().astype(int).tolist())}")

    for year in YEARS:
        merge_year(player_df, year)

    # Rebuild combined CSV
    all_dfs = []
    for year in YEARS:
        path = os.path.join(TEAMS_DIR, f"team_stats_{year}.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(TEAMS_DIR, "team_stats_ALL_2015_2026.csv"), index=False)
        print(f"\nCombined CSV updated: {len(combined)} rows")
        print("Experience columns added: avg_age, senior_minutes_pct, freshman_minutes_pct, underclass_minutes_pct")

    print("\nDone. Re-run backtest:")
    print("  python run_backtest.py --seasons 2015 2016 2017 2018 2019 2021 2022 2023 2024 2025 --data-dir data/raw/teams")


if __name__ == "__main__":
    main()
