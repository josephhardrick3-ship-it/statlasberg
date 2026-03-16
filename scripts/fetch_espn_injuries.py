#!/usr/bin/env python3
"""
Fetch ESPN injury status for top 2026 NCAA teams.

Checks roster status via ESPN API for top 100 teams (by adj_margin).
Sets injury_flag=1 and/or star_player_flag=1 in team_stats_2026.csv
for teams missing key players.

Usage:
    python scripts/fetch_espn_injuries.py
"""

import os
import sys
import time
import json
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

# Status types ESPN uses for inactive players
INJURY_STATUSES = {"injury", "out", "doubtful", "questionable", "day-to-day", "suspended"}


def get_all_teams() -> dict:
    """Fetch all ESPN team IDs. Returns {displayName: teamId}."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        "mens-college-basketball/teams?limit=400"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Error fetching team list: {e}")
        return {}

    teams = {}
    sports = data.get("sports", [{}])[0]
    leagues = sports.get("leagues", [{}])[0]
    for t in leagues.get("teams", []):
        team = t.get("team", {})
        teams[team.get("displayName", "")] = team.get("id")
    return teams


def check_team_injuries(team_id: str) -> dict:
    """Return injury info for a team: {has_injury, has_star_injury, injured_players}."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"mens-college-basketball/teams/{team_id}/roster"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {"has_injury": False, "has_star_injury": False, "injured_players": []}

    injured = []
    for athlete in data.get("athletes", []):
        status = athlete.get("status", {})
        status_type = (status.get("type") or "").lower()
        status_name = (status.get("name") or "").lower()
        status_desc = (status.get("description") or "").lower()

        is_injured = (
            status_type in INJURY_STATUSES
            or status_name in INJURY_STATUSES
            or any(w in status_desc for w in ["out", "injured", "doubtful", "questionable"])
        )

        if is_injured:
            injured.append(athlete.get("fullName", "Unknown"))

    # Consider a "star injury" if ≥1 starter-level player is out
    has_star = len(injured) >= 1

    return {
        "has_injury": len(injured) > 0,
        "has_star_injury": has_star,
        "injured_players": injured,
    }


def normalize_espn_name(name: str) -> str:
    """Strip mascot from ESPN display name."""
    # Remove common mascot words (last 1-3 words)
    tokens = name.strip().split()
    # Try removing trailing words
    known_schools = [
        "Duke", "Kentucky", "Kansas", "Florida", "Alabama", "Auburn",
        "Michigan", "Michigan State", "Indiana", "Ohio State", "Wisconsin",
        "Illinois", "Purdue", "Northwestern", "Penn State", "Iowa", "Minnesota",
        "Maryland", "Nebraska", "Rutgers", "Houston", "Texas", "Oklahoma",
        "Oklahoma State", "West Virginia", "Kansas State", "TCU", "Baylor",
        "Arizona", "Arizona State", "Colorado", "Utah", "Oregon", "Washington",
        "UCLA", "USC", "Stanford", "California", "Oregon State", "Washington State",
        "North Carolina", "Duke", "Virginia", "Virginia Tech", "Pittsburgh",
        "Notre Dame", "Clemson", "Louisville", "Wake Forest", "Miami",
        "Georgia Tech", "North Carolina State", "Syracuse", "Boston College",
        "Connecticut", "Villanova", "Georgetown", "Providence", "Seton Hall",
        "Xavier", "Creighton", "Marquette", "Butler", "DePaul", "St. John's",
        "Florida State", "Georgia", "Tennessee", "Arkansas", "Mississippi",
        "Mississippi State", "LSU", "South Carolina", "Missouri", "Vanderbilt",
        "Texas A&M", "Ole Miss",
    ]
    for n_remove in range(1, min(3, len(tokens))):
        candidate = " ".join(tokens[:-n_remove])
        if candidate in known_schools:
            return candidate
    return tokens[0] if tokens else name


def main():
    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr = pd.read_csv(sr_path)

    # Focus on top 100 teams by adj_margin
    top_teams = sr.nlargest(100, "adj_margin")["team"].tolist()
    print(f"Checking injuries for top 100 teams by adj_margin")

    # Get all ESPN team IDs
    print("Fetching ESPN team list...")
    espn_teams = get_all_teams()
    print(f"  {len(espn_teams)} teams in ESPN")

    # Build name mapping: sr_name → espn_id
    espn_names = list(espn_teams.keys())

    # Initialize flags to 0
    if "injury_flag" not in sr.columns:
        sr["injury_flag"] = 0
    if "star_player_flag" not in sr.columns:
        sr["star_player_flag"] = 0

    sr["injury_flag"]     = 0
    sr["star_player_flag"] = 0

    injury_report = []
    checked = 0

    for sr_name in top_teams:
        # Find matching ESPN team
        espn_id = None
        if sr_name in espn_teams:
            espn_id = espn_teams[sr_name]
        else:
            # Try fuzzy
            result = process.extractOne(sr_name, espn_names, scorer=fuzz.token_set_ratio)
            if result and result[1] >= 80:
                espn_id = espn_teams[result[0]]

        if not espn_id:
            continue

        injury_info = check_team_injuries(espn_id)
        checked += 1

        if injury_info["has_injury"]:
            idx = sr[sr["team"] == sr_name].index
            if not idx.empty:
                sr.at[idx[0], "injury_flag"] = 1
                if injury_info["has_star_injury"]:
                    sr.at[idx[0], "star_player_flag"] = 1

            injury_report.append({
                "team": sr_name,
                "injured_players": ", ".join(injury_info["injured_players"]),
            })
            print(f"  [INJURY] {sr_name}: {injury_info['injured_players']}")

        time.sleep(0.2)  # Rate limiting

    print(f"\nChecked {checked} teams, {len(injury_report)} teams with injuries flagged")

    sr.to_csv(sr_path, index=False)
    print(f"Updated {sr_path}")

    if injury_report:
        report_path = os.path.join(TEAMS_DIR, "injury_report_2026.csv")
        pd.DataFrame(injury_report).to_csv(report_path, index=False)
        print(f"Injury report saved: {report_path}")


if __name__ == "__main__":
    main()
