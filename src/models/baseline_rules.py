"""Weighted baseline scoring model. Produces contender_score and upset_risk_score."""
import pandas as pd, numpy as np
from src.utils.logging_utils import get_logger
log = get_logger(__name__)

# ── Group weights (must sum to 1.0) ──────────────────────────────────────────
CONTENDER_WEIGHTS = {
    "efficiency_score":   0.32,   # overall quality — adj_margin is the top March predictor
    "defense_score":      0.20,   # elite defense travels
    "clutch_score":       0.12,   # clutch: close games + Q1 wins (reduced — noisy, small sample)
    "guard_play_score":   0.13,   # backcourt execution
    "rebounding_score":   0.10,   # glass dominance
    "consistency_score":  0.10,   # margin variance + blowout rate (boosted — consistent teams survive)
    "region_bias_score":  0.03,   # geographic advantage (low weight)
}

# ── Penalties applied AFTER base contender_score is computed ─────────────────
FIRST_YEAR_COACH_PENALTY = 3.5   # points deducted for year-1 head coach
HIGH_FOUL_DEPENDENCE_THRESHOLD = 75.0  # foul_dependence percentile above which penalty kicks in
HIGH_FOUL_DEPENDENCE_PENALTY   = 2.5   # points deducted

# ── Availability adjustments (loaded from config) ────────────────────────────
def _load_availability():
    """Load all availability config safely — returns defaults if config missing."""
    try:
        from src.config.availability_2026 import (
            CONTENDER_SCORE_ADJ,
            UPSET_RISK_ADJ,
            PROVEN_COACH_EXEMPT,
        )
        return CONTENDER_SCORE_ADJ, UPSET_RISK_ADJ, PROVEN_COACH_EXEMPT
    except ImportError:
        return {}, {}, set()

AVAILABILITY_OVERRIDES, UPSET_RISK_OVERRIDES, PROVEN_COACH_EXEMPT = _load_availability()


def compute_contender_score(row):
    total = 0.0
    for feat, w in CONTENDER_WEIGHTS.items():
        val = row.get(feat, 50.0)
        if isinstance(val, float) and np.isnan(val):
            val = 50.0
        total += w * val
    return round(total, 1)


def compute_upset_risk_score(row):
    """Higher = more likely to lose early.

    Incorporates:
      - Low efficiency / clutch / defense (existing)
      - First-year coach penalty
      - High foul dependence (reliant on foul line = foul-trouble vulnerable)
      - Low Q1 win rate (hasn't beaten tournament-caliber teams)
    """
    risk = 100.0
    risk -= row.get("efficiency_score", 50) * 0.28
    risk -= row.get("clutch_score",     50) * 0.28
    risk -= row.get("defense_score",    50) * 0.20
    risk -= row.get("guard_play_score", 50) * 0.15
    risk -= row.get("consistency_score",50) * 0.09

    # First-year coach: extra volatility
    if row.get("first_year_coach_flag", 0):
        risk += 6.0

    # High foul dependence: vulnerable when star can't get to the line
    foul_dep = row.get("foul_dependence", 50.0)
    if not (isinstance(foul_dep, float) and np.isnan(foul_dep)):
        if foul_dep > HIGH_FOUL_DEPENDENCE_THRESHOLD:
            risk += HIGH_FOUL_DEPENDENCE_PENALTY

    # No Q1 wins: untested against tournament-caliber opponents
    q1 = row.get("q1_win_pct", np.nan)
    q1_games = row.get("q1_games", 0)
    if not (isinstance(q1, float) and np.isnan(q1)):
        if q1_games >= 3 and q1 < 0.25:   # 0 or 1 wins in Q1 with real opportunities
            risk += 5.0
        elif q1_games >= 3 and q1 < 0.50:
            risk += 2.5

    return round(max(0, min(100, risk)), 1)


def predict_expected_round(contender, seed=None):
    """Map contender score to expected round of exit."""
    if contender >= 80:
        return "Championship"
    elif contender >= 70:
        return "Final Four"
    elif contender >= 60:
        return "Elite 8"
    elif contender >= 50:
        return "Sweet 16"
    elif contender >= 40:
        return "R32"
    else:
        return "R64"


def predicted_seed_line(committee_alignment: float) -> int:
    """Map committee_alignment_score (0-100) to an estimated NCAA seed line (1-16).

    Thresholds derived from 10-year historical calibration of how committee
    alignment scores map to actual seedings.  Approximate — exact seedings
    depend on bracket construction and committee discretion.
    """
    if committee_alignment >= 95:   return 1
    elif committee_alignment >= 87: return 2
    elif committee_alignment >= 79: return 3
    elif committee_alignment >= 71: return 4
    elif committee_alignment >= 63: return 5
    elif committee_alignment >= 55: return 6
    elif committee_alignment >= 47: return 7
    elif committee_alignment >= 40: return 8
    elif committee_alignment >= 33: return 9
    elif committee_alignment >= 26: return 10
    elif committee_alignment >= 19: return 11
    elif committee_alignment >= 13: return 12
    elif committee_alignment >= 8:  return 13
    elif committee_alignment >= 5:  return 14
    elif committee_alignment >= 2:  return 15
    else:                           return 16


def score_all_teams(features_df, apply_availability: bool = True):
    """Score all teams with the baseline model.

    Args:
        features_df: DataFrame with derived features.
        apply_availability: If True (default), apply season-specific injury/
            availability overrides from config/availability_2026.py.
            Set to False when backtesting historical seasons so that 2026
            injury adjustments don't distort prior-year evaluations.
    """
    log.info(f"Scoring {len(features_df)} teams with baseline model")
    df = features_df.copy()

    # ── Ensure pbp_composite exists (future PBP enrichment) ─────────────────
    pbp_cols = ["avg_scoring_drought_secs","avg_run_allowed","avg_run_created",
                 "halftime_adjustment_margin","pressure_game_stability_score"]
    present = [c for c in pbp_cols if c in df.columns]
    if present:
        df["pbp_composite"] = df[present].mean(axis=1)
    elif "pbp_composite" not in df.columns:
        df["pbp_composite"] = 50.0

    # ── Base contender score ─────────────────────────────────────────────────
    df["contender_score"] = df.apply(compute_contender_score, axis=1)

    # ── Post-score adjustments ───────────────────────────────────────────────

    # 1. First-year coach penalty (reduces contender_score directly)
    #    Exemption: teams in PROVEN_COACH_EXEMPT skip the penalty because
    #    their coach has an elite résumé (nat'l title / 3+ Final Fours).
    if "first_year_coach_flag" in df.columns:
        base_mask = df["first_year_coach_flag"].astype(bool)
        if PROVEN_COACH_EXEMPT:
            exempt_mask = df["team"].isin(PROVEN_COACH_EXEMPT)
            # Clear the flag for exempt coaches so downstream logic also skips them
            df.loc[exempt_mask & base_mask, "first_year_coach_flag"] = 0
            exempted = df.loc[exempt_mask & base_mask, "team"].tolist()
            for t in exempted:
                log.info(f"Proven-coach exemption: {t} — first-year penalty waived")
        mask = df["first_year_coach_flag"].astype(bool)   # re-evaluate after exemptions
        df.loc[mask, "contender_score"] -= FIRST_YEAR_COACH_PENALTY
        n_penalized = mask.sum()
        if n_penalized > 0:
            log.info(f"First-year coach penalty (-{FIRST_YEAR_COACH_PENALTY}pt): "
                     f"{n_penalized} teams")

    # 2. Availability overrides (manual config for known absences/injuries)
    #    Skipped when apply_availability=False (e.g. historical backtests)
    if apply_availability and AVAILABILITY_OVERRIDES:
        for team, adj in AVAILABILITY_OVERRIDES.items():
            idx = df[df["team"] == team].index
            if len(idx) > 0:
                df.loc[idx, "contender_score"] += adj
                log.info(f"Availability override: {team} {adj:+.1f}pt")
            else:
                log.warning(f"Availability override: '{team}' not found in data")

    # 3. Clip to valid range
    df["contender_score"] = df["contender_score"].clip(0, 100).round(1)

    # ── Derived scores ───────────────────────────────────────────────────────
    df["upset_risk_score"] = df.apply(compute_upset_risk_score, axis=1)

    # 4. Upset risk overrides (manual config for known injury/availability risk)
    #    Skipped when apply_availability=False (e.g. historical backtests)
    if apply_availability and UPSET_RISK_OVERRIDES:
        for team, adj in UPSET_RISK_OVERRIDES.items():
            idx = df[df["team"] == team].index
            if len(idx) > 0:
                df.loc[idx, "upset_risk_score"] = (
                    df.loc[idx, "upset_risk_score"] + adj
                ).clip(0, 100).round(1)
                log.info(f"Upset risk override: {team} {adj:+.1f}pt")

    df["expected_round"]   = df["contender_score"].apply(predict_expected_round)

    # ── Predicted seed line from committee alignment ─────────────────────────
    if "committee_alignment_score" in df.columns:
        df["predicted_seed_line"] = df["committee_alignment_score"].apply(predicted_seed_line)
    else:
        df["predicted_seed_line"] = None

    # ── Tempo mismatch flag (when tempo data available) ──────────────────────
    if "tempo" in df.columns and df["tempo"].notna().mean() > 0.10:
        tempo_med = df["tempo"].median()
        df["slow_paced"]  = df["tempo"] < (tempo_med - 4)   # 4+ poss/game slower
        df["fast_paced"]  = df["tempo"] > (tempo_med + 4)   # 4+ poss/game faster
    else:
        df["slow_paced"] = False
        df["fast_paced"] = False

    # ── Flags ─────────────────────────────────────────────────────────────────
    net = df.get("net_rank", pd.Series(999, index=df.index)).fillna(999)
    df["title_darkhorse_flag"]    = (df["contender_score"] >= 65) & (net > 15)
    df["fraud_favorite_flag"]     = (df["contender_score"] < 50) & (net <= 10)
    df["dangerous_low_seed_flag"] = (df["contender_score"] >= 55) & (net > 20)

    # Gap-based flags
    if "model_vs_committee_gap" in df.columns:
        gap = df["model_vs_committee_gap"]
        df["underseeded_flag"] = gap >= 15
        df["overseeded_flag"]  = gap <= -18   # tightened from -15; -18 captures ~11 teams after weight rebalance
    else:
        df["underseeded_flag"] = False
        df["overseeded_flag"]  = False

    # Cinderella flag: must be a realistic tournament team AND underseeded
    # Requires NET rank <= 60 (tournament-caliber) + strong cinderella_score
    # + underseeded (model >> committee) + not a top-4 seed
    if "cinderella_score" in df.columns and "predicted_seed_line" in df.columns:
        ps  = df["predicted_seed_line"].fillna(16)
        has_net = "net_rank" in df.columns
        if has_net and df["net_rank"].notna().mean() > 0.10:
            realistic = df["net_rank"].fillna(999) <= 60
        else:
            realistic = df["contender_score"] >= 58  # fallback for historical years
        df["cinderella_flag"] = (
            (df["cinderella_score"] >= 68) &
            df["underseeded_flag"] &
            (ps >= 5) &         # 5-seed or lower
            (df["contender_score"] >= 60) &   # actual tournament-quality team
            realistic            # in the realistic tournament field
        )
    else:
        df["cinderella_flag"] = False

    # High foul-dependence flag
    if "foul_dependence" in df.columns:
        foul_p75 = df["foul_dependence"].quantile(0.75)
        df["high_foul_dependence_flag"] = df["foul_dependence"] > foul_p75
    else:
        df["high_foul_dependence_flag"] = False

    # ── OVERSEEDED haircut: small penalty for teams the committee overvalues ──
    # In tossup games, this nudges the model toward the underdog/DANGEROUS team.
    if "overseeded_flag" in df.columns:
        overseeded_mask = df["overseeded_flag"].astype(bool)
        n_os = overseeded_mask.sum()
        df.loc[overseeded_mask, "contender_score"] -= 2.0
        df["contender_score"] = df["contender_score"].clip(0, 100).round(1)
        log.info(f"OVERSEEDED haircut: -2.0pt applied to {n_os} teams")

    return df
