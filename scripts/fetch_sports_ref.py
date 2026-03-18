#!/usr/bin/env python3
"""
Scrape NCAA Men's Basketball team stats from Sports-Reference (2015-2026).

Saves per-year CSVs to data/raw/teams/team_stats_{year}.csv and a combined
CSV to data/raw/teams/team_stats_ALL_2015_2026.csv.

Usage:
    python scripts/fetch_sports_ref.py
    python scripts/fetch_sports_ref.py --years 2024 2025 2026
    python scripts/fetch_sports_ref.py --dry-run   # print URLs only
"""

import argparse
import os
import sys
import time
from typing import Optional

import pandas as pd
import numpy as np

# Run from bot root so relative paths work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BOT_ROOT)

OUTPUT_DIR = os.path.join(BOT_ROOT, "data", "raw", "teams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://www.sports-reference.com/cbb/seasons/men"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DELAY_SECONDS = 3.5  # Sports-Reference rate limit buffer

# Years to scrape (skip 2020 — COVID, no tournament)
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Team name aliases to normalize to bot's expected names
TEAM_ALIASES = {
    "UConn": "Connecticut", "UCONN": "Connecticut",
    "UNC": "North Carolina", "Ole Miss": "Mississippi",
    "SMU": "Southern Methodist", "LSU": "Louisiana State",
    "USC": "Southern California", "UCF": "Central Florida",
    "TCU": "Texas Christian", "Pitt": "Pittsburgh",
    "Saint Mary's (CA)": "Saint Marys CA",
    "St. Mary's (CA)": "Saint Marys CA",
    "Miami (FL)": "Miami FL", "Miami FL": "Miami FL",
    "St. John's (NY)": "St Johns NY",
    "Saint John's": "St Johns NY",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "UNLV": "UNLV", "VCU": "VCU", "BYU": "Brigham Young",
    "LIU": "Long Island University",
    "UTSA": "UT San Antonio",
    "UIW": "Incarnate Word",
}


def normalize_team_name(name: str) -> str:
    """Standardize team names to match the bot's TEAM_ALIASES."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    # Strip Sports-Reference NCAA tournament marker (non-breaking space + NCAA)
    name = name.replace("\xa0NCAA", "").replace(" NCAA", "").strip()
    # Strip rank suffix like "Connecticut (1)"
    if "(" in name and name.endswith(")"):
        name = name[:name.rfind("(")].strip()
    return TEAM_ALIASES.get(name, name)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level column headers from Sports-Reference tables."""
    if isinstance(df.columns, pd.MultiIndex):
        # Join levels, drop redundant empty strings
        df.columns = [
            "_".join(str(c).strip() for c in col if str(c).strip() and str(c) != "Unnamed")
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
    return df


def scrape_year(year: int) -> pd.DataFrame:
    """Scrape school-stats page for one season. Returns mapped DataFrame."""
    url = f"{BASE_URL}/{year}-school-stats.html"
    print(f"  Fetching {year}: {url}")

    try:
        tables = pd.read_html(
            url,
            attrs={"id": "basic_school_stats"},
            storage_options={"User-Agent": HEADERS["User-Agent"]},
        )
    except Exception:
        # Fallback: read all tables, take the largest one
        try:
            tables = pd.read_html(
                url,
                storage_options={"User-Agent": HEADERS["User-Agent"]},
            )
            tables = [t for t in tables if len(t) > 50]  # filter out nav tables
        except Exception as e:
            print(f"  ERROR fetching {year}: {e}")
            return pd.DataFrame()

    if not tables:
        print(f"  No tables found for {year}")
        return pd.DataFrame()

    raw = tables[0].copy()
    raw = flatten_columns(raw)

    # Drop header repeat rows (Sports-Reference inserts them mid-table)
    # These are rows where the school column contains "School" or is NaN
    school_col = _find_col(raw, ["School", "school", "School_"])
    if school_col:
        raw = raw[raw[school_col].notna()]
        raw = raw[raw[school_col].astype(str).str.strip() != "School"]
        raw = raw[raw[school_col].astype(str).str.strip() != ""]
        raw = raw.reset_index(drop=True)

    return _map_columns(raw, year)


def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    # Partial match fallback
    for c in df.columns:
        for candidate in candidates:
            if candidate.lower() in c.lower():
                return c
    return None


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _map_columns(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Map Sports-Reference columns to bot schema."""
    df = pd.DataFrame(index=raw.index)
    df["season"] = year

    # Team name
    school_col = _find_col(raw, ["School"])
    df["team"] = raw[school_col].apply(normalize_team_name) if school_col else np.nan

    # Conference
    conf_col = _find_col(raw, ["Conf."])
    df["conference"] = raw[conf_col] if conf_col else np.nan

    # Record
    g_col = _find_col(raw, ["G"])
    w_col = _find_col(raw, ["W"])
    l_col = _find_col(raw, ["L"])
    wl_col = _find_col(raw, ["W-L%"])
    srs_col = _find_col(raw, ["SRS"])
    sos_col = _find_col(raw, ["SOS"])

    games = _safe_float(raw[g_col]) if g_col else np.nan
    df["wins"] = _safe_float(raw[w_col]) if w_col else np.nan
    df["win_pct"] = _safe_float(raw[wl_col]) if wl_col else np.nan

    # Derive losses from wins + win_pct (more reliable than parsing the "L" column,
    # which Sports-Reference renders as cumulative season totals that pandas sometimes
    # misreads when multi-level headers are flattened).
    valid_mask = df["wins"].notna() & df["win_pct"].notna() & (df["win_pct"] > 0)
    df["losses"] = np.nan
    df.loc[valid_mask, "losses"] = (
        (df.loc[valid_mask, "wins"] / df.loc[valid_mask, "win_pct"]).round().astype(int)
        - df.loc[valid_mask, "wins"].astype(int)
    )
    df["adj_margin"] = _safe_float(raw[srs_col]) if srs_col else np.nan
    df["strength_of_schedule"] = _safe_float(raw[sos_col]) if sos_col else np.nan

    # Scoring (per-game from totals)
    tm_col = _find_col(raw, ["Tm.", "Pts", "PTS"])
    opp_col = _find_col(raw, ["Opp."])
    tm_pts = _safe_float(raw[tm_col]) if tm_col else np.nan
    opp_pts = _safe_float(raw[opp_col]) if opp_col else np.nan
    df["points_per_game"] = (tm_pts / games).round(1) if g_col else np.nan
    df["points_allowed_per_game"] = (opp_pts / games).round(1) if g_col else np.nan

    # Shooting
    fg_pct_col = _find_col(raw, ["FG%"])
    three_pct_col = _find_col(raw, ["3P%"])
    ft_pct_col = _find_col(raw, ["FT%"])
    three_pa_col = _find_col(raw, ["3PA"])
    three_p_col = _find_col(raw, ["3P"])
    ft_col = _find_col(raw, ["FT"])
    fta_col = _find_col(raw, ["FTA"])

    df["ft_pct"] = _safe_float(raw[ft_pct_col]) * 100 if ft_pct_col else np.nan
    df["three_pt_pct"] = _safe_float(raw[three_pct_col]) * 100 if three_pct_col else np.nan

    # 3PA rate (3PA per 100 possessions approximation — use raw 3PA/G)
    three_pa = _safe_float(raw[three_pa_col]) if three_pa_col else np.nan
    df["three_pa_rate"] = (three_pa / games).round(1) if g_col else np.nan

    # FT rate (FTA per game)
    fta = _safe_float(raw[fta_col]) if fta_col else np.nan
    df["ft_rate"] = (fta / games).round(1) if g_col else np.nan

    # Rebounding
    orb_col = _find_col(raw, ["ORB"])
    trb_col = _find_col(raw, ["TRB"])
    orb = _safe_float(raw[orb_col]) if orb_col else np.nan
    trb = _safe_float(raw[trb_col]) if trb_col else np.nan
    df["off_rebound_pct"] = (orb / games).round(1) if g_col else np.nan
    df["def_rebound_pct"] = ((trb - orb) / games).round(1) if g_col else np.nan

    # Turnovers
    tov_col = _find_col(raw, ["TOV"])
    tov = _safe_float(raw[tov_col]) if tov_col else np.nan
    df["turnover_pct"] = (tov / games).round(1) if g_col else np.nan

    # Derived adjusted efficiency (SRS-based approximation)
    # adj_offense ≈ PPG + SOS * 0.3  |  adj_defense ≈ PAPG - SOS * 0.3
    sos = df["strength_of_schedule"].fillna(0)
    df["adj_offense"] = (df["points_per_game"] + sos * 0.3).round(1)
    df["adj_defense"] = (df["points_allowed_per_game"] - sos * 0.3).round(1)

    # Columns the bot expects but Sports-Reference doesn't have — leave as NaN
    # (bot's feature builder gracefully defaults these)
    for col in [
        "coach", "coach_years_at_school", "coach_ncaa_games",
        "coach_sweet16s", "coach_finalfours",
        "road_wins", "road_losses", "neutral_wins", "neutral_losses",
        "net_rank", "quad1_wins", "quad2_wins", "tempo",
        "opp_turnover_pct", "opp_three_pt_pct",
        "close_game_record", "last10_win_pct", "last10_adj_margin",
        "avg_age", "underclass_minutes_pct", "freshman_minutes_pct",
        "freshman_guard_minutes_pct", "sophomore_minutes_pct",
        "junior_minutes_pct", "senior_minutes_pct",
        "returning_starters", "returning_minutes_pct",
        "primary_guard_experience_score", "backcourt_experience_score",
        "frontcourt_experience_score", "bench_minutes_pct", "bench_points_pct",
        "injury_flag", "star_player_flag",
        "chicago_guard_count", "nyc_guard_count", "indiana_guard_count",
        "texas_big_count", "southern_big_count", "west_coast_wing_count",
        "local_site_player_count", "host_state_player_count",
    ]:
        df[col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape NCAA stats from Sports-Reference")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only, no download")
    args = parser.parse_args()

    all_dfs = []
    for i, year in enumerate(sorted(args.years)):
        if args.dry_run:
            print(f"Would fetch: {BASE_URL}/{year}-school-stats.html")
            continue

        df = scrape_year(year)

        if df.empty:
            print(f"  Skipping {year} — no data returned")
        else:
            out_path = os.path.join(OUTPUT_DIR, f"team_stats_{year}.csv")
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df)} teams → {out_path}")
            all_dfs.append(df)

        # Rate limit: sleep between requests (skip after last)
        if i < len(args.years) - 1 and not args.dry_run:
            print(f"  Waiting {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "team_stats_ALL_2015_2026.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined CSV: {len(combined)} rows → {combined_path}")
        print(f"Seasons: {sorted(combined['season'].unique().tolist())}")


if __name__ == "__main__":
    main()
