"""Classify teams into tournament archetypes — style-first, matchup-aware.

The old archetype system was decorative: it just re-labeled teams based on
the same sub-scores that already drive contender_score.  This version adds
three layers of real analytical value:

  1. PRIMARY ARCHETYPE — HOW a team plays (style identity), not just how good
     they are.  Tells you what kind of basketball they run and what conditions
     they thrive in.  Driven by the FOUR FACTORS:
       - Shooting efficiency (eFG%)
       - Turnover rate
       - Offensive rebounding
       - Free throw rate / pace

  2. VULNERABILITY TAGS — what can beat this team.  Specific, actionable
     weaknesses in matchup terms.  "Hot-three-point team exploits their porous
     perimeter D" is more useful than a generic upset-risk number.

  3. STRENGTH TAGS — what this team brings that most opponents don't.
     Written the way a scout would actually say them.

Plus a MATCHUP MATRIX: head-to-head style-clash insights when two archetype
patterns create a meaningful tactical edge.
"""

import numpy as np
import pandas as pd
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _v(row, col: str, default: float = 50.0) -> float:
    """Safe float getter — returns default for NaN / missing."""
    val = row.get(col, default)
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _pct_norm(raw: float, lo: float, hi: float) -> float:
    """Normalise raw value into 0-1 between lo and hi."""
    return max(0.0, min(1.0, (raw - lo) / max(hi - lo, 1e-9)))


def _as_decimal(val: float, threshold: float = 1.0) -> float:
    """Ensure a percentage column is stored as decimal (0.35 not 35)."""
    return val / 100.0 if val > threshold else val


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRIMARY ARCHETYPE
#    Driven by how teams actually win games, not just overall rating.
# ─────────────────────────────────────────────────────────────────────────────

def classify_archetype(row) -> str:
    """
    Assign a primary play-style archetype.

    Layer 1 — sub-scores (0-100 percentile ranks, always available):
      defense_score, guard_play_score, clutch_score, rebounding_score,
      consistency_score, cinderella_score, efficiency_score, contender_score

    Layer 2 — advanced raw stats (graceful NaN fallbacks):
      tempo        : possessions/game   (~64-74 for D1; 0 if unknown)
      three_pa_rate: 3PA/game           (~5-10 for D1; 0 if unknown)
      three_pt_pct : 3P%                (decimal 0.33 or raw 33; normalized)
      opp_three_pt_pct: opp 3P%         (decimal or raw; normalized)
      ft_pct       : FT%                (decimal or raw; normalized)
      ft_rate      : FTA/FGA            (decimal; 0 if unknown)
      adj_offense  : KenPom adj off eff (~95-125)
      adj_defense  : KenPom adj def eff (~85-115; lower = better)
      avg_age      : roster avg age     (~19-23)
      turnover_pct : TO/possession      (~0.14-0.22)
    """
    # ── Sub-scores (always 0-100) ────────────────────────────────────────────
    contender   = _v(row, "contender_score")
    defense     = _v(row, "defense_score")
    guard       = _v(row, "guard_play_score")
    clutch      = _v(row, "clutch_score")
    reb         = _v(row, "rebounding_score")
    consistency = _v(row, "consistency_score")
    cinderella  = _v(row, "cinderella_score")
    efficiency  = _v(row, "efficiency_score")

    # ── Raw advanced stats ───────────────────────────────────────────────────
    avg_age   = _v(row, "avg_age", 21.0)
    tempo     = _v(row, "tempo", 0.0)

    # Three-point identity
    three_pa  = _v(row, "three_pa_rate", 0.0)   # attempts/game
    three_pct = _as_decimal(_v(row, "three_pt_pct", 0.0))

    # Adjusted efficiency — net margin proxy
    adj_off = _v(row, "adj_offense", 0.0)
    adj_def = _v(row, "adj_defense", 0.0)
    has_eff_data = adj_off > 85 and adj_def > 75
    adj_net = (adj_off - adj_def) if has_eff_data else 0.0  # > 0 = net positive

    # Free throw identity
    ft_rate = _v(row, "ft_rate", 0.0)   # FTA/FGA stored as %; > 40.0 = foul-dependent

    # Turnover proneness
    to_pct = _v(row, "turnover_pct", 0.0)   # turnover rate stored as %; > 19.5 = sloppy

    # Pace / style signals
    has_tempo    = tempo > 55
    fast_paced   = (has_tempo and tempo > 70) or (not has_tempo and three_pa > 8.0)
    slow_paced   = (has_tempo and tempo < 66) or (not has_tempo and three_pa < 5.5)
    three_heavy  = three_pa > 7.5 or (not has_tempo and guard >= 62 and contender >= 52)

    # ── Priority-ordered checks ───────────────────────────────────────────────

    # 1. Blue-Blood Dominant — elite on every axis, proven against anyone
    if contender >= 74 and defense >= 64 and guard >= 62 and clutch >= 60:
        return "Blue-Blood Dominant"

    # 2. Grind-It-Out Defense — identity is stopping you, not scoring on you
    #    Markers: elite defense, proven in close games, slow pace preference,
    #    good adj_defense if available
    adj_def_elite = has_eff_data and adj_def < 95   # top-tier KenPom defensive efficiency
    if defense >= 72 and clutch >= 56 and (slow_paced or reb >= 60 or adj_def_elite):
        return "Grind-It-Out Defense"

    # 3. Veteran Control — senior guards orchestrate everything
    #    Markers: experienced backcourt, clutch, consistent execution
    if guard >= 65 and clutch >= 65 and consistency >= 58:
        return "Veteran Control"

    # 4. Pace-and-Space Gunners — live and die from three, push pace
    #    Markers: three-point heavy, fast pace, guard-led offense
    if (three_heavy or fast_paced) and guard >= 60 and contender >= 50:
        return "Pace-and-Space Gunners"

    # 5. Glass & Paint — dominate the boards, pound inside, slow game down
    #    Markers: elite rebounding, defense holds, slower pace preference
    if reb >= 68 and defense >= 56 and (slow_paced or reb >= guard + 15):
        return "Glass & Paint"

    # 6. Cinderella Profile — underseeded, defense-first, built to cause upsets
    if cinderella >= 68 and defense >= 60 and contender >= 54:
        return "Cinderella Profile"

    # 7. One-Man Show — one guard carries everything; take him away and it's over
    if guard >= 70 and reb < 48 and defense < 54:
        return "One-Man Show"

    # 8. Freshman Loaded — talented but volatile; young roster under pressure
    if avg_age < 20.8 and efficiency < 56 and contender >= 46:
        return "Freshman Loaded"

    # 9. Résumé Builder — committee-friendly record, model sees through it
    if (row.get("fraud_favorite_flag", False) or
            (_v(row, "committee_alignment_score") >= 66 and efficiency < 52)):
        return "Résumé Builder"

    return "Solid Tournament Team"


# ─────────────────────────────────────────────────────────────────────────────
# 2. VULNERABILITY TAGS  — what can beat this team
# ─────────────────────────────────────────────────────────────────────────────

def classify_vulnerabilities(row) -> list:
    """
    Return up to 3 specific matchup weaknesses in plain basketball language.
    Uses both advanced raw stats AND sub-scores as signals.
    """
    tags = []

    def v(col, d=50.0): return _v(row, col, d)

    # ── Foul-dependent (lives at the stripe) ──────────────────────────────
    ft_rate = v("ft_rate", 0.0)
    if ft_rate > 40.0:   # stored as %, e.g. 42.0 = 42 FTAs per 100 FGAs
        tags.append("🚨 Foul-dependent — star gets in foul trouble, season over")

    # ── Porous perimeter defense ──────────────────────────────────────────
    opp3 = _as_decimal(v("opp_three_pt_pct", 0.0))
    adj_def = v("adj_defense", 0.0)
    perimeter_weak = (opp3 > 0.355) or (adj_def > 0 and adj_def > 105)
    if perimeter_weak:
        tags.append("🔓 Leaks from three — hot perimeter shooters will feast")
    elif v("defense_score") < 36 and len(tags) < 2:
        tags.append("🔓 Below-average defense — can be carved up by quality offenses")

    # ── Three-point feast-or-famine ────────────────────────────────────────
    three_pa  = v("three_pa_rate", 0.0)
    three_pct = _as_decimal(v("three_pt_pct", 0.0))
    if three_pa > 7.5 and (three_pct < 0.333 or three_pct == 0.0):
        tags.append("🧊 Shooter's curse — one cold night from three and they can't adjust")

    # ── Young / inexperienced under tournament pressure ────────────────────
    avg_age = v("avg_age", 21.0)
    if avg_age < 20.6 and v("clutch_score") < 48:
        tags.append("🌱 Young — tournament pressure may crack them in a tight one")

    # ── Turnover-prone ─────────────────────────────────────────────────────
    to_pct = v("turnover_pct", 0.0)
    if to_pct > 19.5:   # stored as %, e.g. 20.0 = 20% turnover rate
        tags.append("💸 Turnover-prone — a press or scramble defense will hurt them")

    # ── Star-dependent / one-man show ─────────────────────────────────────
    if v("guard_play_score") >= 72 and v("consistency_score") < 44:
        tags.append("🎭 Star-dependent — scheme the guard and the offense disappears")

    # ── First-year coach chaos ────────────────────────────────────────────
    if row.get("first_year_coach_flag", 0) and not row.get("first_year_coach_exempt", False):
        tags.append("🆕 First-year coach — March adjustment risk")

    # ── Gets sped up / uncomfortable up-tempo ────────────────────────────
    tempo = v("tempo", 0.0)
    if (tempo > 55 and tempo < 66) and v("guard_play_score") < 50:
        tags.append("🌊 Pace teams make them uncomfortable — prefers a crawl")

    # ── Net efficiency concern (only if real KenPom data available) ────────
    adj_off = v("adj_offense", 0.0)
    adj_def_raw = v("adj_defense", 0.0)
    if adj_off > 85 and adj_def_raw > 75:
        adj_net = adj_off - adj_def_raw
        if adj_net < 4:
            tags.append("📉 Thin efficiency margin — doesn't have much cushion against elite teams")

    return tags[:3]


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRENGTH TAGS  — what this team brings that most can't
# ─────────────────────────────────────────────────────────────────────────────

def classify_strengths(row) -> list:
    """
    Return up to 3 specific matchup advantages.
    Written the way a scout would actually say them.
    """
    tags = []

    def v(col, d=50.0): return _v(row, col, d)

    # ── Elite defense on neutral courts ───────────────────────────────────
    adj_def = v("adj_defense", 0.0)
    adj_def_elite = adj_def > 75 and adj_def < 96   # top-5% KenPom D
    if v("defense_score") >= 72 or adj_def_elite:
        tags.append("🔒 Defense travels — elite on any neutral floor")
    elif v("defense_score") >= 62:
        tags.append("🔒 Solid D — won't give you easy baskets")

    # ── Ice at the stripe in crunch time ──────────────────────────────────
    ft_pct = _as_decimal(v("ft_pct", 0.0))
    if ft_pct >= 0.77 and v("clutch_score") >= 65:
        tags.append("🎯 Ice at the stripe — won't blow leads at the line late")

    # ── Senior guards / veteran execution ────────────────────────────────
    if v("guard_play_score") >= 68 and v("clutch_score") >= 65:
        tags.append("🧠 Veteran guards — calmness is contagious in March")

    # ── Wins tight games / clutch identity ────────────────────────────────
    if v("clutch_score") >= 72:
        tags.append("💪 Built for one-possession games — doesn't panic")
    elif v("clutch_score") >= 62:
        tags.append("💪 Proven in close games — knows how to finish")

    # ── Offensive rebound and second-chance machine ────────────────────────
    orb_pct = v("off_rebound_pct", 0.0)
    if v("rebounding_score") >= 72 or orb_pct > 33.0:   # stored as %, e.g. 34.0 = 34% OREB rate
        tags.append("🪤 Trap the glass — extra possessions every single game")

    # ── Guard creation / bucket on demand ─────────────────────────────────
    if v("guard_play_score") >= 70 and v("clutch_score") >= 68:
        tags.append("⚡ Can create off the dribble — always has a bucket on demand")

    # ── Battle-tested vs tournament-caliber opponents ─────────────────────
    q1_wp = v("q1_win_pct", 0.0)
    if q1_wp >= 0.50 and v("efficiency_score") >= 68:
        tags.append("📈 Battle-tested — proven against tournament-level competition")

    # ── Adjusted net efficiency dominance ─────────────────────────────────
    adj_off = v("adj_offense", 0.0)
    adj_def_raw = v("adj_defense", 0.0)
    if adj_off > 85 and adj_def_raw > 75:
        adj_net = adj_off - adj_def_raw
        if adj_net >= 18:
            tags.append(f"📊 Elite efficiency gap (+{adj_net:.1f} adj net) — best in the field")

    # ── Consistent / never has a bad night ───────────────────────────────
    if v("consistency_score") >= 72:
        tags.append("📐 Consistent — no random bad nights; you know what you're getting")

    return tags[:3]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATCHUP MATRIX — head-to-head style clashes
# ─────────────────────────────────────────────────────────────────────────────

# Keys: (arch1_keyword, arch2_keyword) — checked via substring match.
# Values: template string.  {t1} = row1 team, {t2} = row2 team.
MATCHUP_MATRIX = {
    ("Pace-and-Space", "Grind-It-Out"):
        "{t2} will drain the shot clock and make this a street fight. "
        "{t1} wants to run — {t2} won't let them. "
        "Expect mid-60s and {t1} looking for a rhythm they never find.",

    ("Grind-It-Out", "Pace-and-Space"):
        "{t1} smothers pace teams — expect to hold {t2} 10-12 points below average. "
        "They'll pack the paint and dare {t2} to beat them from three all night.",

    ("Veteran", "Freshman"):
        "Experience eats youth. {t1}'s senior guards will have {t2}'s freshmen "
        "making exactly the decisions freshmen make in March. Second half is when it shows.",

    ("Freshman", "Veteran"):
        "{t2}'s experienced backcourt will be in {t1}'s head the whole second half. "
        "Youth and talent only go so far under the lights. "
        "{t2} keeps the pace slow and wins ugly.",

    ("Glass", "Pace-and-Space"):
        "{t1} is going to body {t2} inside and live at the free-throw line. "
        "If {t2} goes cold from three — and {t1} will make sure they do — this is over by halftime.",

    ("Pace-and-Space", "Glass"):
        "{t1} needs floor spacing and can't let this get into a paint battle. "
        "{t2} will pound every missed three into a second chance. "
        "Ball movement vs rim protection — who wins that fight decides this.",

    ("Cinderella", "Blue-Blood"):
        "This is the upset special. {t1} was built to deny {t2} transition, "
        "bleed the clock, and make every shot contested. "
        "Don't count them out — this is the exact matchup they prep for all year.",

    ("Blue-Blood", "Cinderella"):
        "{t1} is the class of the field — but {t2} is dangerous because their style "
        "cancels out talent advantages. {t1} has to be locked in from tip-off. "
        "Don't overlook them.",

    ("One-Man", "Grind-It-Out"):
        "{t2} was built to stop exactly one guy. They'll bracket {t1}'s star, "
        "pack the paint, and dare everyone else to beat them. "
        "Does {t1} have a plan B? The tournament will find out.",

    ("Résumé", "Grind-It-Out"):
        "{t1} hasn't seen a defense like {t2}'s all season. "
        "Soft-schedule fraud gets exposed on neutral courts. "
        "This could get uncomfortable fast.",

    ("Résumé", "Blue-Blood"):
        "That résumé doesn't mean anything now. "
        "{t1} is about to find out what tournament-caliber basketball actually looks like.",

    ("Grind-It-Out", "Glass"):
        "Defensive wall meets paint bruiser. "
        "Whoever controls the glass controls the game. "
        "Expect two teams trading 55-possession halves. Clutch FT shooting wins this.",

    ("Veteran", "Cinderella"):
        "{t1}'s experience edge matters here — but {t2} has the profile to pull this off. "
        "Veteran control vs Cinderella chaos. Watch the second half.",

    ("Veteran", "Résumé"):
        "{t1} will slow {t2} down, grind their offense to a halt, and make "
        "every bucket feel like a project. The résumé doesn't travel.",
}


def get_matchup_arc(row1, row2) -> str:
    """
    Return a plain-English matchup insight based on archetypal style clash.
    Falls back to advanced stat comparisons when no template exists.
    """
    arch1 = row1.get("archetype", "")
    arch2 = row2.get("archetype", "")
    t1    = row1.get("team", "Team A")
    t2    = row2.get("team", "Team B")

    # Keyword substring match (order-sensitive)
    for (kw1, kw2), tmpl in MATCHUP_MATRIX.items():
        if kw1.lower() in arch1.lower() and kw2.lower() in arch2.lower():
            return tmpl.format(t1=t1, t2=t2)

    # Fallback — use raw advanced stats where available, else sub-scores
    def v1(col, d=50.0): return _v(row1, col, d)
    def v2(col, d=50.0): return _v(row2, col, d)

    d1, d2    = v1("defense_score"),    v2("defense_score")
    g1, g2    = v1("guard_play_score"), v2("guard_play_score")
    r1, r2    = v1("rebounding_score"), v2("rebounding_score")
    cl1, cl2  = v1("clutch_score"),     v2("clutch_score")

    adj_net1 = v1("adj_offense", 0.0) - v1("adj_defense", 0.0)
    adj_net2 = v2("adj_offense", 0.0) - v2("adj_defense", 0.0)
    has_net  = adj_net1 != 0 and adj_net2 != 0

    if has_net and (adj_net1 - adj_net2) >= 8:
        return (f"{t1} has a significant efficiency edge (+{adj_net1:.1f} adj net vs "
                f"+{adj_net2:.1f}). That gap doesn't usually disappear on Selection Sunday.")
    if d1 >= d2 + 12:
        return (f"{t1}'s defense is the story — best on the floor by a wide margin. "
                f"They set the pace, dictate tempo, and hold {t2} to possessions, not runs.")
    if g1 >= g2 + 12:
        return (f"{t1} has a significant backcourt edge. Guard play decides March more than "
                f"anything else. {t2} needs to keep this a grinding half-court game.")
    if r1 >= r2 + 12:
        return (f"{t1} dominates the glass — second-chance points will be the margin here. "
                f"{t2} can't afford to miss and give them another look.")
    if cl1 >= cl2 + 10:
        return (f"If it comes down to the last three minutes — and March always does — "
                f"{t1} has the edge. They've been in this spot all year. {t2} hasn't.")

    return (f"Evenly matched styles — {arch1} vs {arch2}. "
            f"Execution and a single momentum run will decide this one.")


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level entry points
# ─────────────────────────────────────────────────────────────────────────────

def classify_all_teams(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Classifying archetypes for {len(df)} teams")
    df = df.copy()
    df["archetype"]     = df.apply(classify_archetype, axis=1)
    # Store tags as pipe-separated strings for CSV compatibility
    df["vuln_tags"]     = df.apply(
        lambda r: " | ".join(classify_vulnerabilities(r)), axis=1)
    df["strength_tags"] = df.apply(
        lambda r: " | ".join(classify_strengths(r)), axis=1)
    return df
