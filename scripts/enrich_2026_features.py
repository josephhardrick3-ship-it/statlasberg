#!/usr/bin/env python3
"""
enrich_2026_features.py
───────────────────────
Adds advanced 2026-specific signals to team_stats_2026.csv:

  1. q1_win_pct    — win rate vs NET top-30 opponents (Quad 1)
  2. q1_games      — number of Q1 opportunities
  3. first_year_coach — True if team has a new head coach this season
  4. margin_std    — standard deviation of game-by-game scoring margins
                     (consistency proxy — high std = prone to wild swings)
  5. blowout_pct   — % of wins by 15+ points (dominance indicator)
  6. close_win_pct — % of games decided by ≤5 points that this team WON
  7. foul_dependence — reliance on free throws (ft_rate proxy for foul trouble)
  8. three_pt_variance — std of 3P% game-to-game (feast-or-famine 3pt teams)
                         *Note: approx only — computed from margin volatility*

Run once before selection sunday:
    python scripts/enrich_2026_features.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

BOT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH  = os.path.join(BOT_ROOT, "data", "raw", "teams", "team_stats_2026.csv")
Q1_PATH     = os.path.join(BOT_ROOT, "data", "experiments", "q1_performance_2026.csv")
COACH_PATH  = os.path.join(BOT_ROOT, "data", "experiments", "coach_changes_2026.csv")
GAMELOG_PATH= os.path.join(BOT_ROOT, "data", "raw", "teams", "espn_game_log_2026.csv")

# Explicit ESPN display name → SR data name overrides
ESPN_TO_SR = {
    "UConn Huskies":           "Connecticut",
    "Connecticut Huskies":     "Connecticut",
    "NC State Wolfpack":       "North Carolina State",
    "LSU Tigers":              "Louisiana State",
    "Iowa St. Cyclones":       "Iowa State",
    "St. John's Red Storm":    "St. John's",
    "Miami Hurricanes":        "Miami FL",
    "Michigan St. Spartans":   "Michigan State",
    "Ohio St. Buckeyes":       "Ohio State",
    "Penn St. Nittany Lions":  "Penn State",
    "Arizona St. Sun Devils":  "Arizona State",
    "Colorado St. Rams":       "Colorado State",
    "Kansas St. Wildcats":     "Kansas State",
    "BYU Cougars":             "Brigham Young",
    "SMU Mustangs":            "Southern Methodist",
    "TCU Horned Frogs":        "Texas Christian",
    "UCF Knights":             "Central Florida",
    "USF Bulls":               "South Florida",
    "VCU Rams":                "Virginia Commonwealth",
    "UNLV Rebels":             "Nevada-Las Vegas",
    "USC Trojans":             "Southern California",
    "FAU Owls":                "Florida Atlantic",
    "Loyola Chicago Ramblers": "Loyola-Chicago",
    "Saint Mary's (CA) Gaels": "Saint Marys CA",
    "Ole Miss Rebels":         "Mississippi",
    "Mississippi St. Bulldogs":"Mississippi State",
}


def espn_to_sr(name: str) -> str:
    """ESPN display name → SR team name.

    Strategy: explicit dict first (covers all major tournament teams),
    then strip the last word(s) that look like a mascot (title-case standalone word).
    """
    if name in ESPN_TO_SR:
        return ESPN_TO_SR[name]
    # Try progressively removing trailing words until we get a match
    # (ESPN names are typically "School Mascot" or "School State Mascot")
    parts = name.split()
    for drop in range(1, min(4, len(parts))):
        candidate = " ".join(parts[:-drop]).strip()
        if candidate:
            # Apply a few standard abbreviation expansions
            candidate = candidate.replace("St.", "State").replace("State State", "State")
            return candidate
    return name


# ─────────────────────────────────────────────────────────────────────────────

def merge_q1(sr: pd.DataFrame) -> pd.DataFrame:
    """Merge Q1 win rate from experiment file."""
    if not os.path.exists(Q1_PATH):
        print(f"  Q1 file not found: {Q1_PATH}")
        return sr
    q1 = pd.read_csv(Q1_PATH)
    q1 = q1.rename(columns={"q1_win_pct": "q1_win_pct_new"})
    # Merge on team name
    merged = sr.merge(q1[["team", "q1_wins", "q1_losses", "q1_games",
                            "q1_win_pct_new"]],
                      on="team", how="left")
    merged["q1_win_pct"]  = merged.pop("q1_win_pct_new")
    merged["q1_wins"]     = merged["q1_wins_y"] if "q1_wins_y" in merged else merged.get("q1_wins", np.nan)
    merged["q1_losses"]   = merged["q1_losses_y"] if "q1_losses_y" in merged else merged.get("q1_losses", np.nan)
    merged["q1_games"]    = merged["q1_games_y"] if "q1_games_y" in merged else merged.get("q1_games", np.nan)
    # Clean up duplicate _x/_y columns from merge
    for col in list(merged.columns):
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            if col.endswith("_y") and base in ["q1_wins", "q1_losses", "q1_games"]:
                merged[base] = merged[col]
            merged.drop(columns=[col], errors="ignore", inplace=True)
    print(f"  Q1 win rate: {merged['q1_win_pct'].notna().sum()} teams populated")
    return merged


def merge_coach_changes(sr: pd.DataFrame) -> pd.DataFrame:
    """Add first_year_coach boolean flag."""
    if not os.path.exists(COACH_PATH):
        print(f"  Coach file not found: {COACH_PATH}")
        sr["first_year_coach"] = False
        return sr
    coaches = pd.read_csv(COACH_PATH)
    new_coach_teams = set(coaches[coaches["is_new_coach"] == True]["team"].tolist())
    sr["first_year_coach"] = sr["team"].isin(new_coach_teams)
    n = sr["first_year_coach"].sum()
    print(f"  First-year coaches: {n} teams flagged → {sorted(new_coach_teams)}")
    return sr


def compute_game_log_features(sr: pd.DataFrame) -> pd.DataFrame:
    """Compute margin consistency, blowout %, and close win % from game log."""
    if not os.path.exists(GAMELOG_PATH):
        print(f"  Game log not found: {GAMELOG_PATH}")
        sr["margin_std"] = np.nan
        sr["blowout_pct"] = np.nan
        sr["close_win_pct"] = np.nan
        return sr

    log = pd.read_csv(GAMELOG_PATH)
    sr_names = set(sr["team"].tolist())

    # Map ESPN names to SR names
    log["home_sr"] = log["home_team"].apply(espn_to_sr)
    log["away_sr"] = log["away_team"].apply(espn_to_sr)

    stats = {}
    for _, row in log.iterrows():
        for side, other_side, is_home in [
            ("home_sr", "away_sr", True),
            ("away_sr", "home_sr", False)
        ]:
            team = row[side]
            if team not in sr_names:
                continue
            if team not in stats:
                stats[team] = {"margins": [], "wins": 0, "losses": 0,
                               "close_wins": 0, "close_losses": 0,
                               "blowout_wins": 0}
            # Margin from this team's perspective
            if is_home:
                margin = int(row["home_score"]) - int(row["away_score"])
                won    = row["home_win"]
            else:
                margin = int(row["away_score"]) - int(row["home_score"])
                won    = not row["home_win"]

            stats[team]["margins"].append(margin)
            if won:
                stats[team]["wins"] += 1
                if abs(margin) <= 5:
                    stats[team]["close_wins"] += 1
                if margin >= 15:
                    stats[team]["blowout_wins"] += 1
            else:
                stats[team]["losses"] += 1
                if abs(margin) <= 5:
                    stats[team]["close_losses"] += 1

    rows = []
    for team, s in stats.items():
        margins  = s["margins"]
        total    = s["wins"] + s["losses"]
        close_g  = s["close_wins"] + s["close_losses"]
        rows.append({
            "team":          team,
            "margin_std":    round(np.std(margins), 2) if len(margins) >= 5 else np.nan,
            "blowout_pct":   round(s["blowout_wins"] / total, 3) if total > 0 else np.nan,
            "close_win_pct": round(s["close_wins"] / close_g, 3) if close_g > 0 else np.nan,
        })
    feat_df = pd.DataFrame(rows)

    # Drop old versions if re-running
    for col in ["margin_std", "blowout_pct", "close_win_pct"]:
        if col in sr.columns:
            sr = sr.drop(columns=[col])

    sr = sr.merge(feat_df, on="team", how="left")
    print(f"  Game log features: {feat_df['margin_std'].notna().sum()} teams with margin data")
    return sr


def add_foul_dependence(sr: pd.DataFrame) -> pd.DataFrame:
    """
    foul_dependence: how reliant a team is on free throws.
    High ft_rate = more susceptible to early foul trouble.
    Proxy: ft_rate (FTA per FGA) — higher = more reliant on foul line.
    """
    if "ft_rate" in sr.columns:
        sr["foul_dependence"] = sr["ft_rate"]
        print(f"  foul_dependence: derived from ft_rate")
    else:
        sr["foul_dependence"] = np.nan
    return sr


def main():
    print("=" * 60)
    print("  Enriching 2026 team features")
    print("=" * 60)

    sr = pd.read_csv(STATS_PATH)
    print(f"  Loaded {len(sr)} teams from {STATS_PATH}")

    sr = merge_q1(sr)
    sr = merge_coach_changes(sr)
    sr = compute_game_log_features(sr)
    sr = add_foul_dependence(sr)

    sr.to_csv(STATS_PATH, index=False)
    print(f"\n  Saved enriched stats → {STATS_PATH}")
    print(f"  New columns: q1_win_pct, q1_games, first_year_coach, "
          f"margin_std, blowout_pct, close_win_pct, foul_dependence")

    # Preview top teams
    preview_cols = ["team", "q1_win_pct", "q1_games", "first_year_coach",
                    "margin_std", "blowout_pct"]
    avail = [c for c in preview_cols if c in sr.columns]
    top = sr[sr["net_rank"].notna()].sort_values("net_rank").head(15)
    print(f"\n  Top 15 by NET rank:")
    print(top[avail].to_string(index=False))


if __name__ == "__main__":
    main()
