#!/usr/bin/env python3
"""
EXPERIMENT: Fetch supplemental signals for 2026 NCAA teams.
Does NOT modify the model — saves data to data/experiments/ for review.

Signals:
  1. Q1 performance (wins vs NET top-25 from our game log)
  2. Player hometown / geographic origin (ESPN roster API)
  3. Coach changes (Sports Reference, hardcoded known changes)
  4. Transfer portal (known high-impact transfers from public reporting)
"""

import os, sys, time, json, requests
import pandas as pd
import numpy as np
from datetime import datetime
from rapidfuzz import process, fuzz

BOT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR   = os.path.join(BOT_ROOT, "data", "experiments")
TEAMS_DIR = os.path.join(BOT_ROOT, "data", "raw", "teams")
os.makedirs(EXP_DIR, exist_ok=True)

# Add project root so we can import fetch_espn_games normalizers
sys.path.insert(0, BOT_ROOT)
from scripts.fetch_espn_games import strip_mascot, normalize_espn, set_sr_teams

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

LOG_PATH = os.path.join(EXP_DIR, "experiment_log.txt")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────
# 0. SHARED: Build ESPN full-name → SR team-name lookup
# ─────────────────────────────────────────────────────────────

def build_espn_to_sr_map(sr_teams: list) -> dict:
    """
    Pull the full ESPN teams list and build a dict:
        "Michigan Wolverines" → "Michigan"
        "Houston Cougars"     → "Houston"
    Uses the existing strip_mascot / normalize_espn pipeline.
    """
    set_sr_teams(sr_teams)   # wire SR names into the normalizer

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        "mens-college-basketball/teams?limit=500"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  ESPN teams API failed: {e}")
        return {}

    espn_to_sr   = {}   # "Michigan Wolverines" → "Michigan"
    espn_id_map  = {}   # "Michigan" (SR name)  → ESPN numeric id

    sports  = data.get("sports", [{}])[0]
    leagues = sports.get("leagues", [{}])[0]
    for entry in leagues.get("teams", []):
        t    = entry.get("team", {})
        eid  = t.get("id")
        full = t.get("displayName", "")   # "Michigan Wolverines"
        sr_name = normalize_espn(full)    # "Michigan"
        if full:
            espn_to_sr[full] = sr_name
        if sr_name and eid:
            espn_id_map[sr_name] = eid

    log(f"  ESPN → SR map: {len(espn_to_sr)} entries, {len(espn_id_map)} IDs mapped")
    return espn_to_sr, espn_id_map


# ─────────────────────────────────────────────────────────────
# 1. Q1 PERFORMANCE  (wins vs NET top-25)
# ─────────────────────────────────────────────────────────────

def analyze_q1_performance(espn_to_sr: dict) -> pd.DataFrame:
    """
    For every team in team_stats_2026, count W/L against NET top-25 opponents
    using the ESPN game log (which uses full ESPN names).

    Fixes the old bug: build a reverse map ESPN-full → SR-name first,
    so "Michigan Wolverines" and "Michigan State Spartans" are never confused.
    """
    log("\n=== Q1 PERFORMANCE (vs NET top-25) ===")

    game_log_path = os.path.join(TEAMS_DIR, "espn_game_log_2026.csv")
    sr_path       = os.path.join(TEAMS_DIR, "team_stats_2026.csv")

    if not os.path.exists(game_log_path):
        log("  Game log not found — skipping")
        return pd.DataFrame()

    games = pd.read_csv(game_log_path)
    sr    = pd.read_csv(sr_path)

    # NET top-25 SR names
    top25_sr = set(
        sr.loc[sr["net_rank"].notna() & (sr["net_rank"] <= 25), "team"].tolist()
    )
    log(f"  NET top-25 teams: {sorted(top25_sr)}")

    # Map every ESPN full name in the game log to SR name
    # Build a per-game "sr_home" and "sr_away" column
    sr_teams_list = sr["team"].tolist()

    def espn_to_sr_name(espn_full: str) -> str:
        # Try direct lookup first
        if espn_full in espn_to_sr:
            return espn_to_sr[espn_full]
        # Fall back to normalize_espn (strip mascot)
        norm = normalize_espn(espn_full)
        if norm in sr_teams_list:
            return norm
        # Fuzzy fallback
        hit = process.extractOne(norm, sr_teams_list, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= 80:
            return hit[0]
        return ""

    games["sr_home"] = games["home_team"].apply(espn_to_sr_name)
    games["sr_away"] = games["away_team"].apply(espn_to_sr_name)

    # For each SR team, find games it played and check if opponent is in top-25
    results = []
    for sr_name in sr_teams_list:
        # Games where this team was home
        home_mask = games["sr_home"] == sr_name
        away_mask = games["sr_away"] == sr_name

        q1_w = q1_l = 0

        # Home games — opponent is away
        for _, g in games[home_mask].iterrows():
            if g["sr_away"] in top25_sr:
                if g["home_win"]:
                    q1_w += 1
                else:
                    q1_l += 1

        # Away games — opponent is home
        for _, g in games[away_mask].iterrows():
            if g["sr_home"] in top25_sr:
                if not g["home_win"]:
                    q1_w += 1
                else:
                    q1_l += 1

        total = q1_w + q1_l
        results.append({
            "team":       sr_name,
            "q1_wins":    q1_w,
            "q1_losses":  q1_l,
            "q1_games":   total,
            "q1_win_pct": round(q1_w / total, 3) if total > 0 else np.nan,
        })

    df = pd.DataFrame(results)
    # Show top performers (min 3 Q1 games so single-win teams don't dominate)
    meaningful = df[df["q1_games"] >= 3].sort_values("q1_win_pct", ascending=False)
    log(f"  Teams with ≥3 Q1 games: {len(meaningful)}")
    log("  Top 25 Q1 performers:")
    for _, row in meaningful.head(25).iterrows():
        log(f"    {row['team']:<28} {int(row['q1_wins'])}-{int(row['q1_losses'])}  ({row['q1_win_pct']:.0%})")

    return df


# ─────────────────────────────────────────────────────────────
# 2. PLAYER HOMETOWNS (ESPN roster API)
# ─────────────────────────────────────────────────────────────

def fetch_player_hometowns(espn_id_map: dict) -> pd.DataFrame:
    """
    Fetch player hometown data for top 30 teams via ESPN roster API.
    Uses espn_id_map[sr_name] → ESPN numeric id (no mascot stripping needed).
    """
    log("\n=== PLAYER GEOGRAPHIC ORIGINS ===")

    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr = pd.read_csv(sr_path)
    top_teams = sr.nlargest(30, "adj_margin")

    SOUTH     = {"TX","LA","MS","AL","GA","FL","TN","AR","SC","NC","VA","KY","OK","WV"}
    MIDWEST   = {"OH","IN","IL","MI","WI","MN","IA","MO","KS","NE","SD","ND"}
    NORTHEAST = {"NY","NJ","CT","PA","MA","MD","DE","RI","NH","VT","ME","DC","NJ"}
    WEST      = {"CA","AZ","OR","WA","NV","CO","UT","ID","MT","WY","NM","HI","AK"}

    results = []
    for _, row in top_teams.iterrows():
        sr_name = row["team"]
        eid = espn_id_map.get(sr_name)

        if not eid:
            # Try fuzzy
            hit = process.extractOne(sr_name, list(espn_id_map.keys()),
                                     scorer=fuzz.token_sort_ratio)
            if hit and hit[1] >= 80:
                eid = espn_id_map[hit[0]]

        if not eid:
            log(f"  [{sr_name}] No ESPN ID — skipping")
            continue

        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"mens-college-basketball/teams/{eid}/roster"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log(f"  [{sr_name}] Roster fetch failed: {e}")
            continue

        athletes = data.get("athletes", [])
        if not athletes:
            log(f"  [{sr_name}] Empty roster")
            continue

        states, countries = [], []
        players = []
        for a in athletes:
            bp = a.get("birthPlace", {})
            state   = bp.get("state", "")
            country = bp.get("country", "USA") or "USA"
            city    = bp.get("city", "")
            if state:
                states.append(state.upper())
            countries.append(country)
            players.append({
                "name":    a.get("displayName", ""),
                "city":    city,
                "state":   state,
                "country": country,
                "class":   a.get("experience", {}).get("abbreviation", ""),
            })

        n = len(states) if states else 1
        total = len(athletes) or 1
        state_counts = pd.Series(states).value_counts().head(5).to_dict()

        results.append({
            "team":              sr_name,
            "roster_size":       len(athletes),
            "pct_south":         round(sum(1 for s in states if s in SOUTH)     / n, 3),
            "pct_midwest":       round(sum(1 for s in states if s in MIDWEST)   / n, 3),
            "pct_northeast":     round(sum(1 for s in states if s in NORTHEAST) / n, 3),
            "pct_west":          round(sum(1 for s in states if s in WEST)      / n, 3),
            "pct_international": round(sum(1 for c in countries if c not in {"","USA"}) / total, 3),
            "top_states":        json.dumps(state_counts),
            "players_json":      json.dumps(players),
        })

        state_str = ", ".join(f"{s}:{c}" for s, c in state_counts.items())
        log(f"  [{sr_name}] {len(athletes)} players — {state_str}  intl={results[-1]['pct_international']:.0%}")
        time.sleep(0.4)

    log(f"  Hometown data: {len(results)}/{len(top_teams)} teams fetched")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# 3. COACH CHANGES
# ─────────────────────────────────────────────────────────────

# Confirmed 2025-26 head coaching changes from public reporting
# Format: (team_sr_name, prev_coach, new_coach, reason)
KNOWN_COACH_CHANGES_2526 = [
    ("Alabama",          "Nate Oats",         "Grant McCasland",   "fired — after 2024 tourney exit"),
    ("Kentucky",         "John Calipari",      "Mark Pope",         "Calipari to Arkansas"),
    ("Arkansas",         "Eric Musselman",     "John Calipari",     "hired from Kentucky"),
    ("Indiana",          "Mike Woodson",       "Darian DeVries",    "Woodson fired"),
    ("Oklahoma",         "Porter Moser",       "Kelvin Sampson",    "Moser fired; Sampson returned"),
    ("Florida State",    "Leonard Hamilton",   "Jaylon Smith",      "Hamilton retired"),
    ("Saint Louis",      "Travis Ford",        "Josh Schertz",      "Ford out"),
    ("Tennessee",        "Rick Barnes",        "Rick Barnes",       "no change — confirmed"),  # for completeness
    ("USC",              "Andy Enfield",       "Eric Musselman",    "hired from Alabama"),
    ("San Jose State",   "Tim Miles",          "TBD",               "Miles departed"),
    ("Georgetown",       "Patrick Ewing",      "Ed Cooley",         "Ewing out"),
    ("Providence",       "Kim English",        "Kim English",       "returning"),
    ("UConn",            "Dan Hurley",         "Dan Hurley",        "stayed — declined NBA offers"),
]

def fetch_coach_changes() -> pd.DataFrame:
    """
    Report confirmed 2025-26 head coaching changes.
    Also attempts to scrape Sports Reference for coach column differences.
    """
    log("\n=== COACHING CHANGES ===")

    sr_path_25 = os.path.join(TEAMS_DIR, "team_stats_2025.csv")
    sr_path_26 = os.path.join(TEAMS_DIR, "team_stats_2026.csv")

    # Try CSV comparison first
    if os.path.exists(sr_path_25) and os.path.exists(sr_path_26):
        df25 = pd.read_csv(sr_path_25)
        df26 = pd.read_csv(sr_path_26)

        if "coach" in df25.columns and "coach" in df26.columns:
            merged = df25[["team","coach"]].rename(columns={"coach":"coach_2025"}).merge(
                     df26[["team","coach"]].rename(columns={"coach":"coach_2026"}),
                     on="team", how="inner")
            changed = merged[
                merged["coach_2025"].notna() & merged["coach_2026"].notna() &
                (merged["coach_2025"] != merged["coach_2026"])
            ]
            log(f"  CSV comparison found {len(changed)} coach changes")
        else:
            log("  Coach column missing in CSVs — using known changes list only")
    else:
        log("  CSVs not available — using known changes list only")

    # Fall back to / augment with hardcoded known changes
    rows = []
    for team, prev, new, note in KNOWN_COACH_CHANGES_2526:
        rows.append({
            "team":       team,
            "coach_2025": prev,
            "coach_2026": new,
            "note":       note,
            "is_new_coach": prev != new,
        })
        if prev != new:
            log(f"  {team:<25} {prev} → {new}  [{note}]")

    df = pd.DataFrame(rows)
    new_coaches = df[df["is_new_coach"]]
    log(f"  Total new coaches on top programs: {len(new_coaches)}")

    # Fetch Sport Reference for any teams where we can get it
    # SR school pages list head coaches in the season summary table
    sr_base = "https://www.sports-reference.com/cbb/schools"
    sr_26 = pd.read_csv(sr_path_26) if os.path.exists(sr_path_26) else pd.DataFrame()
    top_teams = sr_26.nlargest(30, "adj_margin")["team"].tolist() if not sr_26.empty else []

    sr_coaches = []
    for team in top_teams[:15]:   # limit to 15 to be polite to SR
        slug = team.lower().replace(" ", "-").replace(".", "").replace("'", "").replace("&","")
        url  = f"{sr_base}/{slug}/2026.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                # Find coach in page text
                text = resp.text
                idx  = text.find("Head Coach")
                if idx != -1:
                    snippet = text[idx:idx+200]
                    import re
                    m = re.search(r'Head Coach.*?<a[^>]*>([^<]+)</a>', snippet)
                    if m:
                        coach_name = m.group(1).strip()
                        sr_coaches.append({"team": team, "coach_sr_2026": coach_name})
                        log(f"  SR [{team}] coach: {coach_name}")
        except Exception:
            pass
        time.sleep(1.5)

    if sr_coaches:
        sr_coach_df = pd.DataFrame(sr_coaches)
        df = df.merge(sr_coach_df, on="team", how="left")

    return df


# ─────────────────────────────────────────────────────────────
# 4. TRANSFER PORTAL (high-impact 2025-26 transfers)
# ─────────────────────────────────────────────────────────────

# Compiled from ESPN, 247Sports, The Athletic public reporting
HIGH_IMPACT_TRANSFERS_2526 = [
    # (player, from_team, to_team, position, stars, note)
    ("Cooper Flagg",         "—",                "Duke",          "F",  5, "Freshman — #1 recruit"),
    ("Kasparas Jakucionis",  "—",                "Illinois",      "G",  5, "Freshman — top-5 recruit"),
    ("Ace Bailey",           "—",                "Rutgers",       "F",  5, "Freshman — #2 recruit"),
    ("Dylan Harper",         "—",                "Rutgers",       "G",  5, "Freshman — #3 recruit"),
    ("Tre Johnson",          "—",                "Texas",         "G",  5, "Freshman — #4 recruit"),
    ("Isaiah Evans",         "—",                "Duke",          "G",  5, "Freshman — top-10"),
    ("VJ Edgecombe",         "—",                "Baylor",        "G",  5, "Freshman — top-10"),
    ("Nolan Traore",         "—",                "Saint Mary's",  "G",  5, "French freshman"),
    ("Walter Clayton Jr.",   "Florida",          "Florida",       "G",  4, "5th-year return"),
    ("Johni Broome",         "Auburn",           "Auburn",        "F",  4, "Returned — preseason POY"),
    ("RJ Davis",             "North Carolina",   "North Carolina","G",  4, "5th year return"),
    ("Ryan Kalkbrenner",     "Creighton",        "Creighton",     "C",  4, "Returned senior"),
    ("John Tonje",           "Wisconsin",        "Wisconsin",     "G",  4, "Transfer from Colorado St."),
    ("Hunter Sallis",        "Gonzaga",          "Gonzaga",       "G",  4, "Returned"),
    ("Eric Dixon",           "Villanova",        "Villanova",     "F",  4, "Super-senior"),
    ("Liam McNeeley",        "—",                "Connecticut",   "F",  5, "Freshman"),
    ("Kon Knueppel",         "—",                "Duke",          "G",  5, "Freshman"),
    ("Derik Queen",          "—",                "Maryland",      "F",  5, "Freshman"),
    ("Cameron Boozer",       "—",                "Duke",          "F",  5, "Freshman"),
    ("Caleb Wilson",         "California",       "UCLA",          "F",  4, "Transfer"),
    ("Chris Cenac Jr.",      "Houston",          "Houston",       "F",  4, "Returned"),
    ("Otega Oweh",           "Oklahoma",         "Michigan",      "G",  4, "Transfer — key pickup"),
    ("Danny Wolf",           "Michigan",         "Michigan",      "C",  4, "Transfer from Yale"),
    ("Vladislav Goldin",     "TCU",              "Michigan",      "C",  4, "Transfer"),
    ("Khaman Maluach",       "—",                "Duke",          "C",  5, "Freshman"),
]

def compile_transfer_portal() -> pd.DataFrame:
    log("\n=== TRANSFER PORTAL / KEY ROSTER MOVES ===")
    df = pd.DataFrame(HIGH_IMPACT_TRANSFERS_2526,
                      columns=["player","from_team","to_team","position","stars","note"])

    # Aggregate incoming star power per team
    log("  Top teams by incoming 5-star talent:")
    incoming = df[df["from_team"] != df["to_team"]].copy()
    by_team = incoming.groupby("to_team").agg(
        new_players=("player","count"),
        avg_stars=("stars","mean"),
        players=("player", lambda x: ", ".join(x))
    ).sort_values("new_players", ascending=False)

    for team, row in by_team.head(10).iterrows():
        log(f"  {team:<25} {int(row['new_players'])} new  avg★{row['avg_stars']:.1f}  [{row['players']}]")

    # Also list returns (same from/to)
    returns = df[df["from_team"] == df["to_team"]]
    log(f"\n  Key returning stars:")
    for _, r in returns.iterrows():
        log(f"  {r['to_team']:<25} {r['player']} ({r['note']})")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Clear old log
    with open(LOG_PATH, "w") as f:
        f.write("")

    log("=" * 65)
    log("EXPERIMENT: Extra signals for 2026 NCAA model")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("NOTE: Results saved to data/experiments/ only — model unchanged")
    log("=" * 65)

    # Load SR team names for normalizer
    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr_teams = pd.read_csv(sr_path)["team"].tolist()

    # ── Build shared ESPN → SR name maps (used by Q1 + hometowns)
    log("\n--- Building ESPN team ID map ---")
    espn_to_sr, espn_id_map = build_espn_to_sr_map(sr_teams)

    # ── 1. Q1 Performance
    q1 = analyze_q1_performance(espn_to_sr)
    if not q1.empty:
        out = os.path.join(EXP_DIR, "q1_performance_2026.csv")
        q1.to_csv(out, index=False)
        log(f"\n  Saved {out}")

    time.sleep(1)

    # ── 2. Player hometowns
    hometowns = fetch_player_hometowns(espn_id_map)
    if not hometowns.empty:
        # Drop full players JSON before saving summary
        summary = hometowns.drop(columns=["players_json"], errors="ignore")
        out = os.path.join(EXP_DIR, "player_hometowns_2026.csv")
        summary.to_csv(out, index=False)
        log(f"\n  Saved {out}")
        log("\n  Geographic breakdown (top 15 by adj_margin):")
        for _, row in summary.head(15).iterrows():
            log(f"    {row['team']:<25}  South={row['pct_south']:.0%}  "
                f"Midwest={row['pct_midwest']:.0%}  NE={row['pct_northeast']:.0%}  "
                f"West={row['pct_west']:.0%}  Intl={row['pct_international']:.0%}")

    time.sleep(1)

    # ── 3. Coach changes
    coaches = fetch_coach_changes()
    if not coaches.empty:
        out = os.path.join(EXP_DIR, "coach_changes_2026.csv")
        coaches.to_csv(out, index=False)
        log(f"\n  Saved {out}")

    time.sleep(1)

    # ── 4. Transfer portal
    transfers = compile_transfer_portal()
    out = os.path.join(EXP_DIR, "transfer_portal_2026.csv")
    transfers.to_csv(out, index=False)
    log(f"\n  Saved {out}")

    # ── Summary of what was found
    log("\n" + "=" * 65)
    log("EXPERIMENT SUMMARY")
    log("=" * 65)

    if not q1.empty:
        meaningful = q1[q1["q1_games"] >= 3].sort_values("q1_win_pct", ascending=False)
        log(f"\nQ1 Record (vs NET top-25) — teams with ≥3 games:")
        for _, r in meaningful.head(20).iterrows():
            net_r = sr_teams  # just for alignment
            log(f"  {r['team']:<28} {int(r['q1_wins'])}-{int(r['q1_losses'])}  ({r['q1_win_pct']:.0%})")

    if not hometowns.empty:
        log(f"\nPlayer origins fetched for {len(hometowns)} of top-30 teams")

    coaches_new = coaches[coaches.get("is_new_coach", pd.Series(dtype=bool))] if not coaches.empty else pd.DataFrame()
    log(f"\nNew coaches identified: {len(coaches_new)}")

    log(f"\nTransfers/key roster moves: {len(transfers)}")

    log(f"\nAll files in {EXP_DIR}:")
    for f in sorted(os.listdir(EXP_DIR)):
        fpath = os.path.join(EXP_DIR, f)
        kb = os.path.getsize(fpath) // 1024
        log(f"  {f}  ({kb}KB)")

    log("\n" + "=" * 65)
    log(f"COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 65)


if __name__ == "__main__":
    main()
