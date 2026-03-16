#!/usr/bin/env python3
"""
bracket_entry.py
────────────────
Generates data/brackets/bracket_2026.csv from the real Selection Sunday
seedings.

TWO MODES:
  1. Template mode (default):
     python scripts/bracket_entry.py
     → Writes a pre-populated CBS projection template.
       Open the CSV, update every team name with the REAL seedings,
       then save. That's it.

  2. Interactive mode:
     python scripts/bracket_entry.py --interactive
     → Walks you through each seed slot one at a time.

The output CSV is what run_selection_sunday.py and run_pipeline.py consume.

FORMAT EXPECTED:
  season, region, seed, team
  2026, East, 1, Duke
  2026, East, 2, Connecticut
  ...

FIRST FOUR (play-in games):
  Use seed 11a / 11b for the two at-large play-in slots,
  and 16a / 16b for the two auto-bid play-in slots.
  Example:
    2026, East,    16a, <team_A>
    2026, East,    16b, <team_B>
  The simulation treats play-in winners by their base seed (11 or 16).
"""

import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRACKET_DIR = os.path.join(BOT_ROOT, "data", "brackets")
OUT_PATH    = os.path.join(BRACKET_DIR, "bracket_2026.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CBS Bracketology projection — March 15, 2026 (Selection Sunday eve).
# SOURCE: CBS Sports / Jerry Palm bracketology last updated March 15.
#
# *** THIS IS A PROJECTION.  REPLACE EVERY TEAM WITH THE REAL SEEDINGS
#     AFTER THE SELECTION SHOW TONIGHT. ***
#
# Instructions:
#   1. Open data/brackets/bracket_2026.csv in any text editor or Excel.
#   2. As the bracket is revealed, update each team name.
#   3. Keep region / seed columns exactly as-is.
#   4. Save.  Then run:
#        python scripts/run_selection_sunday.py
# ─────────────────────────────────────────────────────────────────────────────

# fmt: (region, seed, team)
CBS_PROJECTED_BRACKET = [
    # ── EAST — Duke ──────────────────────────────────────────────────────────
    ("East",  1,   "Duke"),
    ("East",  2,   "Connecticut"),
    ("East",  3,   "Purdue"),
    ("East",  4,   "Illinois"),
    ("East",  5,   "Tennessee"),
    ("East",  6,   "Gonzaga"),
    ("East",  7,   "NC State"),
    ("East",  8,   "Indiana"),
    ("East",  9,   "Auburn"),
    ("East", 10,   "Vanderbilt"),
    ("East", 11,   "Texas"),           # UPDATE — may be play-in
    ("East", 12,   "Dayton"),
    ("East", 13,   "Princeton"),
    ("East", 14,   "Oral Roberts"),
    ("East", 15,   "Bryant"),
    ("East", 16,   "TBD_auto_East"),   # play-in winner — update tonight

    # ── MIDWEST — Michigan ────────────────────────────────────────────────────
    ("Midwest",  1,  "Michigan"),
    ("Midwest",  2,  "Iowa State"),
    ("Midwest",  3,  "Wisconsin"),
    ("Midwest",  4,  "Kansas"),
    ("Midwest",  5,  "BYU"),
    ("Midwest",  6,  "Michigan State"),
    ("Midwest",  7,  "Arkansas"),
    ("Midwest",  8,  "Marquette"),
    ("Midwest",  9,  "Virginia"),
    ("Midwest", 10,  "Missouri"),
    ("Midwest", 11,  "Minnesota"),      # UPDATE — may be play-in
    ("Midwest", 12,  "New Mexico"),
    ("Midwest", 13,  "High Point"),
    ("Midwest", 14,  "Morehead State"),
    ("Midwest", 15,  "McNeese State"),
    ("Midwest", 16,  "TBD_auto_Midwest"),

    # ── SOUTH — Florida ───────────────────────────────────────────────────────
    ("South",  1,  "Florida"),
    ("South",  2,  "Alabama"),
    ("South",  3,  "Texas A&M"),
    ("South",  4,  "North Carolina"),
    ("South",  5,  "Saint Marys CA"),
    ("South",  6,  "Texas Tech"),
    ("South",  7,  "Creighton"),
    ("South",  8,  "Kansas State"),
    ("South",  9,  "Colorado State"),
    ("South", 10,  "Stanford"),
    ("South", 11,  "Mississippi State"),  # UPDATE — may be play-in
    ("South", 12,  "Liberty"),
    ("South", 13,  "Akron"),
    ("South", 14,  "Grand Canyon"),
    ("South", 15,  "Lipscomb"),
    ("South", 16,  "TBD_auto_South"),

    # ── WEST — Arizona ────────────────────────────────────────────────────────
    ("West",  1,  "Arizona"),
    ("West",  2,  "Houston"),
    ("West",  3,  "St. Johns"),
    ("West",  4,  "Baylor"),
    ("West",  5,  "UCLA"),
    ("West",  6,  "Kentucky"),
    ("West",  7,  "Oregon"),
    ("West",  8,  "Georgia"),
    ("West",  9,  "Northwestern"),
    ("West", 10,  "San Diego State"),
    ("West", 11,  "Oklahoma"),           # UPDATE — may be play-in
    ("West", 12,  "Colorado State"),
    ("West", 13,  "Troy"),
    ("West", 14,  "Furman"),
    ("West", 15,  "Eastern Washington"),
    ("West", 16,  "TBD_auto_West"),
]

SEASON = 2026


def write_template():
    os.makedirs(BRACKET_DIR, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["season", "region", "seed", "team"])
        for region, seed, team in CBS_PROJECTED_BRACKET:
            writer.writerow([SEASON, region, seed, team])
    print(f"✅  Template written → {OUT_PATH}")
    print()
    print("Next steps:")
    print("  1. Open data/brackets/bracket_2026.csv")
    print("  2. Update every team name with the real Selection Sunday seedings")
    print("  3. For First Four play-in games, use seeds 11a/11b and 16a/16b")
    print("     (e.g. 'East', '16a', 'Farleigh Dickinson')")
    print("  4. Save the file")
    print("  5. Run:  python scripts/run_selection_sunday.py")
    print()
    print("Current projection (update tonight):")
    print(f"  {'Region':<10} {'Seed':<5} {'Team'}")
    print(f"  {'-'*40}")
    for region, seed, team in CBS_PROJECTED_BRACKET:
        flag = " ← UPDATE TONIGHT" if "TBD" in team else ""
        print(f"  {region:<10} {seed:<5} {team}{flag}")


def interactive_entry():
    """Walk through each seed slot interactively."""
    os.makedirs(BRACKET_DIR, exist_ok=True)
    rows = []
    regions = ["East", "Midwest", "South", "West"]
    seeds   = list(range(1, 17))

    print("=" * 60)
    print("  BRACKET ENTRY — 2026 NCAA Tournament")
    print("  Enter each team name as announced.")
    print("  Press Enter to keep the CBS projection (shown in brackets).")
    print("=" * 60)

    # Build lookup from CBS projection
    cbs_lookup = {(r, s): t for r, s, t in CBS_PROJECTED_BRACKET}

    for region in regions:
        print(f"\n── {region.upper()} REGION ──")
        for seed in seeds:
            cbs = cbs_lookup.get((region, seed), "")
            prompt = f"  {region} #{seed:>2}  [{cbs}]: "
            val = input(prompt).strip()
            team = val if val else cbs
            rows.append((SEASON, region, seed, team))

    # First Four
    print("\n── FIRST FOUR PLAY-IN GAMES ──")
    first_four_slots = [
        ("East",    "16a", "TBD_auto_East_A"),
        ("East",    "16b", "TBD_auto_East_B"),
        ("Midwest", "16a", "TBD_auto_Midwest_A"),
        ("Midwest", "16b", "TBD_auto_Midwest_B"),
        ("South",   "11a", "TBD_at_large_South_A"),
        ("South",   "11b", "TBD_at_large_South_B"),
        ("West",    "11a", "TBD_at_large_West_A"),
        ("West",    "11b", "TBD_at_large_West_B"),
    ]
    for region, seed, default in first_four_slots:
        val = input(f"  {region} {seed}: ").strip()
        team = val if val else default
        rows.append((SEASON, region, seed, team))

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["season", "region", "seed", "team"])
        for row in rows:
            writer.writerow(row)

    print(f"\n✅  Bracket saved → {OUT_PATH}")
    print("Run:  python scripts/run_selection_sunday.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interactive", action="store_true",
                   help="Enter bracket interactively team-by-team")
    args = p.parse_args()

    if args.interactive:
        interactive_entry()
    else:
        write_template()
