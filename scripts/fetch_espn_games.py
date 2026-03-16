#!/usr/bin/env python3
"""
Fetch all 2025-26 NCAA Men's Basketball game results from ESPN API.

Computes per-team:
  - close_game_record: W-L in games decided by ≤5 pts
  - last10_win_pct: win pct in last 10 games
  - last10_adj_margin: avg margin in last 10 games

Saves:
  - data/raw/teams/espn_game_log_2026.csv  (all games)
  - data/raw/teams/espn_team_records_2026.csv  (per-team aggregates)
  - Updates data/raw/teams/team_stats_2026.csv with these columns

Usage:
    python scripts/fetch_espn_games.py
"""

import os
import sys
import time
import json
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ROOT   = os.path.dirname(SCRIPT_DIR)
TEAMS_DIR  = os.path.join(BOT_ROOT, "data", "raw", "teams")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

SEASON_START = date(2025, 11, 1)
SEASON_END   = date(2026, 3, 15)  # through conference championship weekend

CLOSE_MARGIN = 5   # games decided by ≤5 pts are "close"

TEAM_ALIASES = {
    "UConn": "Connecticut", "UCONN": "Connecticut",
    "Ole Miss": "Mississippi", "SMU": "Southern Methodist",
    "LSU": "Louisiana State", "USC": "Southern California",
    "UCF": "Central Florida", "TCU": "Texas Christian",
    "Pitt": "Pittsburgh", "BYU": "Brigham Young",
    "Saint Mary's (CA)": "Saint Marys CA",
    "St. Mary's (CA)": "Saint Marys CA",
    "Miami (OH)": "Miami OH",
    "Miami (FL)": "Miami FL",
    "St. John's (NY)": "St Johns NY",
    "Saint John's (NY)": "St Johns NY",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "LIU": "Long Island University",
    "UTSA": "UT San Antonio",
    "UIW": "Incarnate Word",
    "UMKC": "Kansas City",
    "SIUE": "SIU-Edwardsville",
    "SIU Edwardsville": "SIU-Edwardsville",
    "Cal State Bakersfield": "Cal State Bakersfield",
    "Loyola Chicago": "Loyola-Chicago",
    "Loyola (IL)": "Loyola-Chicago",
    "Texas A&M-Commerce": "East Texas A&M",
}


# Common ESPN mascot suffixes to strip (sorted longest first for greedy match)
_MASCOT_SUFFIXES = [
    "Crimson Tide", "Volunteers", "Mountaineers", "Golden Eagles", "Golden Flashes",
    "Golden Grizzlies", "Golden Rams", "Scarlet Knights", "Cardinal and Gold",
    "Blue Raiders", "Blue Devils", "Blue Demons", "Blue Hens", "Blue Hawks",
    "Blue Jays", "Bluejays", "Bulldogs", "Cardinals", "Wildcats", "Tigers",
    "Panthers", "Bears", "Eagles", "Warriors", "Rams", "Bobcats", "Coyotes",
    "Falcons", "Hornets", "Ravens", "Owls", "Pirates", "Knights", "Flames",
    "Blazers", "Mavericks", "Bison", "Seawolves", "Penguins", "Roadrunners",
    "Red Wolves", "Red Storm", "Red Flash", "Privateers", "Colonels", "Bearcats",
    "Thunderbirds", "Buccaneers", "Terrapins", "Terriers", "Tritons", "Toreros",
    "Retrievers", "Mean Green", "Sun Devils", "Beachcombers", "Rainbow Warriors",
    "Aggies", "Anteaters", "Aztecs", "Chanticleers", "Cornhuskers", "Cyclones",
    "Ducks", "Flyers", "Gators", "Gaels", "Gorillas", "Greyhounds", "Hawkeyes",
    "Hoosiers", "Hurricanes", "Huskies", "Jackrabbits", "Jayhawks", "Lobos",
    "Longhorns", "Lumberjacks", "Mastodons", "Monarchs", "Mocs", "Muleriders",
    "Ospreys", "Peacocks", "Penguins", "Pioneers", "Pride", "Quakers", "Racers",
    "Rebels", "Rockets", "Salukis", "Seminoles", "Shockers", "Skyhawks", "Sooners",
    "Spartans", "Spiders", "Stags", "Sycamores", "Tar Heels", "Thoroughbreds",
    "Titans", "Trojans", "Utes", "Vaqueros", "Vikings", "Violets", "Warhawks",
    "Wolf Pack", "Wolves", "Wolverines", "Yellow Jackets", "Zips", "49ers",
    "Braves", "Broncos", "Bruins", "Buckeyes", "Camels", "Catamounts", "Cougars",
    "Crusaders", "Deacons", "Demon Deacons", "Engineers", "Flyers", "Governors",
    "Green Wave", "Grizzlies", "Hoyas", "Jaguars", "Jaspers", "Jets", "Lions",
    "Matadors", "Minutemen", "Musketeers", "Nanooks", "Nittany Lions",
    "Phoenix", "Ramblers", "Razorbacks", "Running Eagles", "Saints",
    "Sea Hawks", "Seahawks", "Sharks", "Skyhawks", "Thundering Herd",
    "Toppers", "Wasps", "Westerners", "Wolverines", "Zips",
]
_MASCOT_SUFFIXES_SORTED = sorted(_MASCOT_SUFFIXES, key=len, reverse=True)


def strip_mascot(name: str) -> str:
    """Remove ESPN mascot suffix from team display name."""
    name = name.strip()
    for mascot in _MASCOT_SUFFIXES_SORTED:
        if name.endswith(" " + mascot):
            return name[: -(len(mascot) + 1)].strip()
    return name


def normalize(name: str) -> str:
    if pd.isna(name):
        return name
    name = strip_mascot(name.strip())
    return TEAM_ALIASES.get(name, name)


_SR_TEAMS: list = []   # populated by set_sr_teams()

def set_sr_teams(teams: list) -> None:
    global _SR_TEAMS
    _SR_TEAMS = teams


def normalize_espn(espn_name: str) -> str:
    """
    Convert ESPN display name to bot team name.
    Strategy: strip known mascots, apply aliases, then try progressively
    shorter names until an exact match with SR teams is found.
    """
    name = espn_name.strip()
    # Apply alias first (catches UConn, St. John's, etc.)
    aliased = TEAM_ALIASES.get(name, None)
    if aliased:
        return aliased

    # Try stripping known mascot suffix
    stripped = strip_mascot(name)
    aliased2 = TEAM_ALIASES.get(stripped, stripped)
    if _SR_TEAMS and aliased2 in _SR_TEAMS:
        return aliased2

    # Progressive word-strip: try removing 1, 2 trailing words
    tokens = name.split()
    for n_remove in range(1, min(4, len(tokens))):
        candidate = " ".join(tokens[:-n_remove])
        candidate_aliased = TEAM_ALIASES.get(candidate, candidate)
        if _SR_TEAMS and candidate_aliased in _SR_TEAMS:
            return candidate_aliased

    # Fall back to best alias resolution of stripped name
    return aliased2


def fuzzy_match(name: str, choices: list, threshold: int = 80) -> Optional[str]:
    result = process.extractOne(name, choices, scorer=fuzz.token_set_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def fetch_games_for_date(d: date) -> list:
    """Fetch completed games for one date. Returns list of game dicts."""
    ds = d.strftime("%Y%m%d")
    params = {"dates": ds, "limit": 100}
    try:
        resp = requests.get(SCOREBOARD_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Error fetching {d}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})

        # Only include completed games
        if not status.get("completed", False):
            continue

        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue

        home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
        away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])

        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (ValueError, TypeError):
            continue

        margin = abs(home_score - away_score)
        home_win = home_score > away_score

        games.append({
            "date": d.isoformat(),
            "game_id": event.get("id"),
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "margin": margin,
            "home_win": home_win,
            "neutral_site": comp.get("neutralSite", False),
        })

    return games


def collect_all_games() -> pd.DataFrame:
    """Loop all dates and collect completed game results."""
    all_games = []
    current = SEASON_START
    total_days = (SEASON_END - SEASON_START).days + 1

    print(f"Fetching game results from {SEASON_START} to {SEASON_END}")
    day_count = 0
    while current <= SEASON_END:
        games = fetch_games_for_date(current)
        all_games.extend(games)
        if games:
            print(f"  {current}: {len(games)} games")
        current += timedelta(days=1)
        day_count += 1
        # Polite rate limiting
        if day_count % 10 == 0:
            time.sleep(0.5)

    df = pd.DataFrame(all_games)
    print(f"\nTotal completed games: {len(df)}")
    return df


def compute_team_records(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team aggregates from game log."""
    # Build per-team game list (each game appears twice: once for each team)
    records = []
    for _, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]
        home_win, away_win = g["home_win"], not g["home_win"]
        margin = g["margin"]

        records.append({
            "team": home,
            "date": g["date"],
            "win": home_win,
            "margin_signed": g["home_score"] - g["away_score"],
            "close": margin <= CLOSE_MARGIN,
        })
        records.append({
            "team": away,
            "date": g["date"],
            "win": away_win,
            "margin_signed": g["away_score"] - g["home_score"],
            "close": margin <= CLOSE_MARGIN,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"])

    result = []
    for team, grp in df.groupby("team"):
        grp = grp.reset_index(drop=True)
        last10 = grp.tail(10)
        close = grp[grp["close"]]

        close_w = int(close["win"].sum())
        close_l = int((~close["win"]).sum())
        close_total = close_w + close_l
        close_win_pct = round(close_w / close_total, 3) if close_total > 0 else 0.5

        result.append({
            "team": normalize_espn(team),
            "close_wins":   close_w,
            "close_losses": close_l,
            # close_game_record stored as numeric win % (what the model expects)
            "close_game_record": close_win_pct,
            "last10_win_pct": round(last10["win"].mean(), 3),
            "last10_adj_margin": round(last10["margin_signed"].mean(), 1),
        })

    return pd.DataFrame(result)


def merge_into_team_stats(records: pd.DataFrame) -> None:
    """Merge ESPN records into team_stats_2026.csv."""
    path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr = pd.read_csv(path)

    for col in ["close_game_record", "last10_win_pct", "last10_adj_margin"]:
        if col not in sr.columns:
            sr[col] = np.nan
        else:
            # Force numeric (old runs may have stored string "W-L" format)
            sr[col] = pd.to_numeric(sr[col], errors="coerce")

    espn_teams = records["team"].tolist()
    matched = fuzzy_matched = 0
    unmatched = []

    for idx, row in sr.iterrows():
        sr_name = row["team"]

        if sr_name in records["team"].values:
            r = records[records["team"] == sr_name].iloc[0]
            matched += 1
        else:
            best = fuzzy_match(sr_name, espn_teams)
            if best:
                r = records[records["team"] == best].iloc[0]
                fuzzy_matched += 1
            else:
                unmatched.append(sr_name)
                continue

        sr.at[idx, "close_game_record"]  = r["close_game_record"]
        sr.at[idx, "last10_win_pct"]     = r["last10_win_pct"]
        sr.at[idx, "last10_adj_margin"]  = r["last10_adj_margin"]

    print(f"Merged: exact={matched}, fuzzy={fuzzy_matched}, unmatched={len(unmatched)}")
    if unmatched:
        print(f"  Unmatched: {unmatched[:8]}")

    sr.to_csv(path, index=False)
    print(f"Updated {path}")


def main():
    # Load SR team names so normalize_espn can exact-match against them
    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    if os.path.exists(sr_path):
        sr_teams = pd.read_csv(sr_path)["team"].tolist()
        set_sr_teams(sr_teams)

    # Check if game log already downloaded; skip re-fetch if so
    game_log_path = os.path.join(TEAMS_DIR, "espn_game_log_2026.csv")
    if os.path.exists(game_log_path):
        print(f"Using cached game log: {game_log_path}")
        games = pd.read_csv(game_log_path)
        print(f"  {len(games)} games loaded")
    else:
        games = collect_all_games()
        if games.empty:
            print("No games found!")
            return
        games.to_csv(game_log_path, index=False)
        print(f"Game log saved: {game_log_path}")

    # Compute team records
    records = compute_team_records(games)
    records_path = os.path.join(TEAMS_DIR, "espn_team_records_2026.csv")
    records.to_csv(records_path, index=False)
    print(f"Team records saved: {records_path}")
    print(f"  Teams with data: {len(records)}")

    # Merge into team stats
    merge_into_team_stats(records)


if __name__ == "__main__":
    main()
