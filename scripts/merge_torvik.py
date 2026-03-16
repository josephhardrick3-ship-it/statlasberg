#!/usr/bin/env python3
"""
Merge Torvik efficiency data with Sports-Reference season stats.

Torvik provides:   AdjOE, AdjDE, Barthag, tempo, EFG%, TOR, TORD, ORB, DRB, FTR, 3P%, etc.
Sports-Ref provides: W/L, SRS, SOS, FT%, 3P%, raw counting stats

The merge fills in the critical efficiency columns that Sports-Reference lacks,
giving the bot's model much better signal (especially for adj_offense/adj_defense).

Usage:
    python scripts/merge_torvik.py
    # Outputs merged team_stats_{year}.csv files, overwriting the SR-only versions
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ROOT = os.path.dirname(SCRIPT_DIR)
TEAMS_DIR = os.path.join(BOT_ROOT, "data", "raw", "teams")

TORVIK_CSV = os.path.join(TEAMS_DIR, "torvik_all_2015_2026.csv")
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Torvik column → bot schema column
TORVIK_COL_MAP = {
    "AdjOE":  "adj_offense",
    "AdjDE":  "adj_defense",
    "AdjT":   "tempo",
    "TOR":    "turnover_pct",
    "TORD":   "opp_turnover_pct",
    "ORB":    "off_rebound_pct",
    "DRB":    "def_rebound_pct",
    "FTR":    "ft_rate",
    "3P%":    "three_pt_pct",
    "3P%D":   "opp_three_pt_pct",
    "3PR":    "three_pa_rate",
    "WAB":    "wab",           # Wins Above Bubble — useful signal
    "Barthag":"barthag",       # win probability vs avg D1 team
    "EFG%":   "eff_fg_pct",
    "EFGD%":  "opp_eff_fg_pct",
}

# Team name aliases for fuzzy-match fallback
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
    "VCU": "VCU", "UNLV": "UNLV",
}


def clean_torvik_team(name: str) -> str:
    """Strip tournament seed/result annotations from Torvik team names.

    Examples:
      'Villanova   2 seed, CHAMPS'  → 'Villanova'
      'Kansas   1 seed, Elite Eight' → 'Kansas'
      'Michigan'                    → 'Michigan'
    """
    if pd.isna(name):
        return name
    name = str(name).strip()
    # Strip seed/result annotation: "   N seed, ..." or "   N seed"
    name = re.sub(r'\s+\d+\s+seed.*$', '', name, flags=re.IGNORECASE).strip()
    # Strip any trailing parenthetical
    name = re.sub(r'\s*\(.*\)\s*$', '', name).strip()
    return TEAM_ALIASES.get(name, name)


def fuzzy_match(name: str, choices: list, threshold: int = 80):
    """Return best fuzzy match from choices, or None if below threshold."""
    result = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def merge_year(torvik_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Merge Torvik data into Sports-Reference CSV for one year."""
    sr_path = os.path.join(TEAMS_DIR, f"team_stats_{year}.csv")
    if not os.path.exists(sr_path):
        print(f"  [{year}] No SR file found, skipping")
        return pd.DataFrame()

    sr = pd.read_csv(sr_path)
    torvik_year = torvik_df[torvik_df["season"] == year].copy()

    if torvik_year.empty:
        print(f"  [{year}] No Torvik rows for this year")
        return sr

    # Clean Torvik team names
    torvik_year["team_clean"] = torvik_year["Team"].apply(clean_torvik_team)

    # Parse W/L from Torvik Rec column (e.g. "33–5" or "29–2 17–1")
    def parse_record(rec):
        if pd.isna(rec):
            return None, None
        m = re.match(r'(\d+)[–\-](\d+)', str(rec))
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    torvik_year[["torvik_w", "torvik_l"]] = torvik_year["Rec"].apply(
        lambda r: pd.Series(parse_record(r))
    )

    # Rename Torvik columns to bot schema
    rename_map = {k: v for k, v in TORVIK_COL_MAP.items() if k in torvik_year.columns}
    torvik_year = torvik_year.rename(columns=rename_map)

    # Build lookup: clean_name → row
    torvik_lookup = torvik_year.set_index("team_clean")
    sr_teams = sr["team"].tolist()
    torvik_teams = torvik_lookup.index.tolist()

    matched = 0
    fuzzy_matched = 0
    unmatched = []

    for col in TORVIK_COL_MAP.values():
        if col not in sr.columns:
            sr[col] = np.nan

    for idx, row in sr.iterrows():
        sr_name = row["team"]

        # Exact match first
        if sr_name in torvik_lookup.index:
            t_row = torvik_lookup.loc[sr_name]
            matched += 1
        else:
            # Fuzzy fallback
            best = fuzzy_match(sr_name, torvik_teams)
            if best:
                t_row = torvik_lookup.loc[best]
                fuzzy_matched += 1
            else:
                unmatched.append(sr_name)
                continue

        # Fill Torvik columns (Torvik values override SR derived values for efficiency)
        for bot_col in TORVIK_COL_MAP.values():
            if bot_col in t_row.index and pd.notna(t_row[bot_col]):
                try:
                    sr.at[idx, bot_col] = float(str(t_row[bot_col]).replace('+', ''))
                except (ValueError, TypeError):
                    pass

        # Also fill adj_margin from Torvik (AdjOE - AdjDE)
        try:
            adj_o = float(sr.at[idx, "adj_offense"])
            adj_d = float(sr.at[idx, "adj_defense"])
            sr.at[idx, "adj_margin"] = round(adj_o - adj_d, 1)
        except (ValueError, TypeError):
            pass

    print(f"  [{year}] {len(sr)} teams | exact: {matched} | fuzzy: {fuzzy_matched} | unmatched: {len(unmatched)}")
    if unmatched:
        print(f"          Unmatched: {unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    return sr


def main():
    print(f"Loading Torvik data from {TORVIK_CSV}")
    torvik = pd.read_csv(TORVIK_CSV)
    torvik["season"] = pd.to_numeric(torvik["season"], errors="coerce")
    print(f"  {len(torvik)} rows, years: {sorted(torvik['season'].dropna().unique().astype(int).tolist())}")

    all_merged = []
    for year in YEARS:
        merged = merge_year(torvik, year)
        if not merged.empty:
            merged.to_csv(os.path.join(TEAMS_DIR, f"team_stats_{year}.csv"), index=False)
            all_merged.append(merged)

    if all_merged:
        combined = pd.concat(all_merged, ignore_index=True)
        combined.to_csv(os.path.join(TEAMS_DIR, "team_stats_ALL_2015_2026.csv"), index=False)
        print(f"\nCombined CSV updated: {len(combined)} rows")
        print(f"Key columns filled by Torvik: adj_offense, adj_defense, adj_margin, tempo, turnover_pct, off_rebound_pct")

    print("\nDone. Re-run the backtest:")
    print("  python run_backtest.py --seasons 2015 2016 2017 2018 2019 2021 2022 2023 2024 2025 --data-dir data/raw/teams")


if __name__ == "__main__":
    main()
