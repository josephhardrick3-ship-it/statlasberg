"""
availability_2026.py
────────────────────
Manual overrides for known player/coach absences heading into the 2026
NCAA Tournament.  The model's stats are based on the full regular season
and may OVERSTATE a team's strength if a key player is no longer available.

HOW TO USE:
  - Set the value to a negative float (points to subtract from contender_score).
  - Guideline:
      Star player gone (projected top-5 pick, 20+ ppg):  -18 to -22
      Key starter gone (projected top-20 pick, 15+ ppg): -10 to -15
      Important rotation piece gone:                      -5 to -8
      Sub / depth piece gone:                             -2 to -4
  - Leave commented out if status is uncertain.
  - Run `python scripts/run_selection_sunday.py` after updating.

CURRENT STATUS (as of March 16, 2026 — Selection Sunday):
  All entries below are CONFIRMED injuries as of tip-off weekend.

SOURCES (verified via ESPN / CBS Sports / Yahoo Sports):
  Duke Foster:   https://www.cbssports.com/college-basketball/news/caleb-foster-injury-duke-2026-ncaa-tournament-march-madness/
  NC Wilson:     https://www.cbssports.com/college-basketball/news/caleb-wilson-injury-north-carolina-out-for-season/
  TTech Toppin:  https://www.espn.com/mens-college-basketball/story/_/id/47968594/texas-tech-jt-toppin-torn-acl-season
  BYU Saunders:  via Yahoo Sports March Madness 2026 preview
  Michigan Cason: via Yahoo Sports March Madness 2026 preview
  Tennessee Ament: https://www.foxsports.com/articles/cbk/no-minutes-limit-no-problem-nate-aments-return-fuels-tennessees-comeback-over-auburn
  Villanova Hodge: https://www.espn.com/mens-college-basketball/story/_/id/48086353/villanova-says-matt-hodge-suffered-season-ending-acl-tear
  Kansas Peterson: https://www.cbssports.com/college-basketball/news/darryn-peterson-kansas-exit-oklahoma-state-bill-self/
"""

# ─────────────────────────────────────────────────────────────────────────────
# contender_score adjustments — negative = penalize, positive = boost
# ─────────────────────────────────────────────────────────────────────────────
CONTENDER_SCORE_ADJ: dict[str, float] = {

    # ── CONFIRMED SEASON-ENDING INJURIES ─────────────────────────────────────

    # North Carolina — Caleb Wilson (star freshman, projected top-5 pick)
    # Broke right thumb in practice on March 5 (non-contact drill).
    # OUT FOR SEASON. Was averaging 19.8 ppg / 9.4 rpg — best UNC freshman ever.
    # Without Wilson, UNC ceiling is Sweet 16 at best.
    "North Carolina": -20,

    # Texas Tech — JT Toppin (All-American forward)
    # Torn ACL in right knee vs. Arizona State. OUT FOR SEASON.
    # Was averaging 21.8 ppg / 10.8 rpg — one of only two players in D-I averaging
    # a double-double with 20+ ppg (alongside Cameron Boozer).
    # Selection committee already dropped TTech from 3-seed to 4-seed.
    "Texas Tech": -20,

    # Duke — Caleb Foster (starting point guard)
    # Fractured right foot, underwent surgery. Out until at least Final Four
    # (coach Scheyer: "we'd have to advance to a Final Four").
    # Was starting PG in 30/31 games, 8.5 ppg / 2.8 ast.
    # Patrick Ngongba II (foot soreness) is EXPECTED BACK for NCAA opener — no adj.
    "Duke": -12,

    # BYU — Richie Saunders (key shooter, torn ACL in February)
    #      + Dawson Baker (shooter, torn ACL in November)
    # Both are gone for the season. BYU's spacing is gutted — only 2 players
    # shooting above 35% from three. Dybantsa is healthy and playing well.
    "Brigham Young": -9,

    # Villanova — Matt Hodge (starting forward, torn ACL vs. St. John's)
    # 6-8 redshirt freshman started all 29 games (9.2 ppg, 36.8% 3P).
    # OUT FOR SEASON. Wildcats are a lower seed anyway — bounce-out risk elevated.
    "Villanova": -8,

    # Tennessee — Nate Ament (ankle + knee injury)
    # Injured vs. Alabama (Feb 28). Returned for SEC tournament but coach Rick Barnes
    # confirmed "he is NOT 100%." Playing through pain with limited explosiveness.
    # Projected top-10 pick, 17.4 ppg / 6.4 rpg. Partial effectiveness expected.
    "Tennessee": -4,

    # Michigan — L.J. Cason (backup guard, season-ending ACL)
    # Rotation piece gone — places added pressure on Elliot Cadeau to be consistent.
    # Michigan is still extremely strong; minor depth hit.
    "Michigan": -3,

    # Kansas — Darryn Peterson (chronic cramping / hamstring issues)
    # Missed 11 of 27 regular-season games with quad, hamstring, ankle injuries.
    # Returned for final 7 games and played through Big 12 tournament.
    # PROJECTED AVAILABLE but remains a game-to-game durability concern.
    # Partial adjustment to reflect real uncertainty.
    "Kansas": -3,

    # ── ADD ANY LATE-BREAKING INJURY / SUSPENSION NEWS HERE ──────────────────
    # Format: "Team Name As In SR Data": score_adjustment_float
    # SR team names: "Connecticut", "Michigan State", "Ohio State", etc.
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-team upset risk boosts — applied to upset_risk_score directly.
# Use for teams where key absences elevate first-round/second-round exit risk
# BEYOND what the contender_score drop already captures.
# ─────────────────────────────────────────────────────────────────────────────
UPSET_RISK_ADJ: dict[str, float] = {
    # North Carolina: without Wilson they've looked average. Wrong matchup exits them fast.
    "North Carolina": 8.0,
    # Texas Tech: without Toppin they have no dominant interior presence.
    "Texas Tech": 8.0,
    # Duke: without Foster they lack a reliable ball-handler under pressure.
    "Duke": 5.0,
    # BYU: perimeter shooting decimated — opponents can pack the paint vs. Dybantsa.
    "Brigham Young": 5.0,
    # Tennessee: Ament not 100% = clutch execution risk in close games.
    "Tennessee": 3.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Teams with confirmed HEALTHY key players returning from injury
# (small positive boost to contender_score)
# ─────────────────────────────────────────────────────────────────────────────
RETURN_FROM_INJURY: dict[str, float] = {
    # Duke's Patrick Ngongba II (foot soreness) is expected back for NCAA opener.
    # No boost needed — his return just restores the baseline, not an upgrade.
}

# ─────────────────────────────────────────────────────────────────────────────
# PROVEN COACH EXEMPTION
# ─────────────────────────────────────────────────────────────────────────────
# Teams listed here will NOT receive the first_year_coach_penalty (-3.5 pts)
# even if the coach is technically in their first year at this school.
#
# Criteria for exemption:
#   — National championship OR 3+ Final Four appearances as a head coach
#   — Track record of routinely advancing in the NCAA tournament
#
# For 2025-26: Most coaches with elite résumés (Calipari, Musselman, Pope)
# are already in Year 2+ and not flagged. This set is populated as a
# future-proof safety net and for any late additions.
#
# IMPORTANT: also update coach_changes_2026.csv to set is_new_coach = False
# for any coach who is actually in Year 2+ at their school (preferred approach).
# This PROVEN_COACH_EXEMPT set is the fallback for genuine Year-1 elite coaches.
PROVEN_COACH_EXEMPT: set = set()
# To add an exemption, replace set() with a set literal, e.g.:
# PROVEN_COACH_EXEMPT: set = {"Arkansas", "USC"}
#
# Examples for future years:
#   "Arkansas"   — John Calipari (1 nat'l title, 4 Final Fours)
#   "USC"        — Eric Musselman (2x Elite 8 at Arkansas)
#   "Kentucky"   — Mark Pope (BYU Final Four 2011 + KY Sweet 16 yr 1)
#   Add any Year-1 coach with national title / 3+ Final Fours
