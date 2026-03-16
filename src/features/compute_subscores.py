"""Compute normalized 0–100 sub-scores from raw team features.

This is the bridge between build_team_features (raw/derived stats) and
baseline_rules (weighted scoring model).  Each sub-score uses percentile
ranking within the season so the model stays calibrated year-over-year.

Sub-scores produced:
  - efficiency_score       — overall quality (barthag + adj_margin + NET + SOS)
  - defense_score          — defensive dominance + rebounding
  - guard_play_score       — backcourt experience + ball security
  - clutch_score           — close-game record + FT + late TOs
  - rebounding_score       — offensive + defensive glass
  - region_bias_score      — player-origin regional advantage (low weight)
  - committee_alignment_score — how the NCAA committee views this team
                                (NET-heavy résumé metric; predicts seeding,
                                 NOT tournament success — intentionally distinct
                                 from efficiency_score)
"""

import numpy as np
import pandas as pd
from src.utils.logging_utils import get_logger

log = get_logger("compute_subscores")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _pctile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Return 0-100 percentile rank. ascending=True means higher raw → higher score."""
    return series.rank(pct=True, ascending=ascending, na_option="bottom") * 100


def _safe_col(df: pd.DataFrame, col: str, default: float = 50.0) -> pd.Series:
    """Return column if present, else constant series."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def _weighted_composite(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Weighted average of percentile-ranked columns.

    spec: dict of {col_name: (weight, ascending)}
          ascending=True  → higher raw value  = higher percentile (e.g. off-rebound %)
          ascending=False → lower raw value   = higher percentile (e.g. adj_defense)
    """
    total_weight = sum(w for w, _ in spec.values())
    composite = pd.Series(0.0, index=df.index)
    for col, (weight, ascending) in spec.items():
        raw = _safe_col(df, col)
        pct = _pctile(raw, ascending=ascending)
        composite += pct * (weight / total_weight)
    return composite.round(1)


# ─────────────────────────────────────────────
# Individual sub-scores
# ─────────────────────────────────────────────

def _defense_score(df: pd.DataFrame) -> pd.Series:
    """Elite defense travels — emphasises adj_defense, rebounding, opponent shooting."""
    return _weighted_composite(df, {
        "adj_defense":       (0.35, False),   # lower is better
        "def_rebound_pct":   (0.20, True),
        "opp_three_pt_pct":  (0.15, False),   # lower is better
        "turnover_pct":      (0.10, False),   # team TO — lower is better
        "opp_turnover_pct":  (0.10, True),    # forcing TOs — higher is better
        "road_win_pct":      (0.10, True),    # proxy for defense traveling
    })


def _efficiency_score(df: pd.DataFrame) -> pd.Series:
    """Overall team quality — the single strongest March predictor."""
    spec = {
        "barthag":              (0.45, True),   # Barttorvik power rating
        "adj_margin":           (0.30, True),   # net point differential
        "strength_of_schedule": (0.15, True),   # schedule quality
        "wab":                  (0.10, True),   # wins above bubble
    }
    # Only use NET ranking when real data is available (>10% of teams have it)
    if "net_rank" in df.columns and df["net_rank"].notna().mean() > 0.10:
        spec = {
            "barthag":              (0.35, True),
            "adj_margin":           (0.25, True),
            "net_rank":             (0.25, False),  # lower rank = better
            "strength_of_schedule": (0.10, True),
            "wab":                  (0.05, True),
        }
    return _weighted_composite(df, spec)


def _guard_play_score(df: pd.DataFrame) -> pd.Series:
    """Guard-dominated teams advance further — backcourt experience + ball security."""
    return _weighted_composite(df, {
        "backcourt_experience_score": (0.35, True),
        "turnover_pct":              (0.25, False),  # lower is better
        "ft_pct":                    (0.20, True),
        "three_pt_pct":              (0.10, True),
        "primary_guard_experience_score": (0.10, True),
    })


def _clutch_score(df: pd.DataFrame) -> pd.Series:
    """Ability to win tight games — close-game record, FT, late TOs.

    Extended with:
      - q1_win_pct:    actual Quad-1 (top-30 NET) win rate — proven can beat
                       tournament-caliber opponents under pressure
      - close_win_pct: from game log, win rate in ≤5-point games (more granular
                       than close_game_record which is a binary W/L)
    """
    spec = {
        "close_game_record": (0.28, True),
        "ft_pct":            (0.20, True),
        "turnover_pct":      (0.12, False),
        "last10_win_pct":    (0.15, True),
    }
    # Q1 win rate: proven beating top-30 NET teams
    has_q1 = "q1_win_pct" in df.columns and df["q1_win_pct"].notna().mean() > 0.10
    if has_q1:
        spec["q1_win_pct"] = (0.15, True)   # can beat tournament-level opponents

    # Granular close-game win rate from game log (when available)
    if "close_win_pct" in df.columns and df["close_win_pct"].notna().mean() > 0.10:
        spec["close_win_pct"] = (0.10, True)

    # PBP features (when available — future enrichment)
    if "final5_ft_pct" in df.columns:
        spec["final5_ft_pct"] = (0.08, True)
    if "comeback_win_pct" in df.columns:
        spec["comeback_win_pct"] = (0.08, True)

    return _weighted_composite(df, spec)


def _rebounding_score(df: pd.DataFrame) -> pd.Series:
    """Rebounding wins tournament games."""
    return _weighted_composite(df, {
        "off_rebound_pct":  (0.45, True),
        "def_rebound_pct":  (0.35, True),
        "adj_margin":       (0.20, True),   # proxy for overall physicality
    })


def _consistency_score(df: pd.DataFrame) -> pd.Series:
    """How consistent/predictable a team is — reduces variance in projections.

    Low margin_std = performs similarly every night (safe, reliable).
    High blowout_pct = wins big regularly (dominant).
    These teams are less likely to have a random bad night in the tournament.

    Only available for 2026 (requires game log data).  Falls back to adj_margin
    percentile for historical years.
    """
    has_game_log = ("margin_std" in df.columns and
                    df["margin_std"].notna().mean() > 0.10)

    if has_game_log:
        spec = {
            "margin_std":   (0.40, False),   # lower std = more consistent
            "blowout_pct":  (0.35, True),    # higher blowout rate = dominant
            "adj_margin":   (0.25, True),    # overall quality anchor
        }
    else:
        spec = {
            "adj_margin": (0.60, True),
            "win_pct":    (0.40, True),
        }
    return _weighted_composite(df, spec)


def _cinderella_score(df: pd.DataFrame) -> pd.Series:
    """Identify teams with the profile to pull tournament upsets.

    Cinderella profile (based on historical upsets 2015-2025):
      - Strong defense (travels to neutral courts)
      - Good clutch / FT performance (survives close games)
      - Lower committee alignment = underseeded relative to model quality
      - Moderate consistency (not completely volatile)

    This score is HIGHEST for mid-major darkhorse teams.
    """
    spec = {
        "defense_score": (0.35, True),
        "clutch_score":  (0.30, True),
        "ft_pct":        (0.20, True),
        "turnover_pct":  (0.15, False),   # low TOs = survive neutral courts
    }
    # Cinderella bonus: higher if model thinks more of them than committee
    # (underseeded = dangerous lower seed)
    if "model_vs_committee_gap" in df.columns:
        spec["model_vs_committee_gap"] = (0.10, True)
    return _weighted_composite(df, spec)


def _committee_alignment_score(df: pd.DataFrame) -> pd.Series:
    """How the NCAA selection committee views this team.

    Predicts the *seed line* the committee will assign — NOT how far the team
    will actually advance.  Heavy NET weight mirrors the committee's own stated
    methodology.  Intentionally ignores clutch/close-game metrics the committee
    does not formally consider.

    Key design difference from efficiency_score:
      - efficiency_score  →  "Can they win 6 straight elimination games?"
      - committee_alignment_score  →  "What seed line will the committee give?"

    When efficiency_score >> committee_alignment_score: team is underseeded
      (model loves them, committee is skeptical — darkhorse threat).
    When efficiency_score << committee_alignment_score: team is overseeded
      (committee loves them, model is skeptical — fraud favorite / upset risk).
    """
    has_net = "net_rank" in df.columns and df["net_rank"].notna().mean() > 0.10

    if has_net:
        # NET era (2019+): NET rank is the committee's #1 stated metric
        spec = {
            "net_rank":             (0.40, False),  # lower rank = better committee view
            "wins":                 (0.18, True),   # overall record still matters
            "strength_of_schedule": (0.15, True),   # NET SOS proxy
            "adj_margin":           (0.10, True),   # efficiency sanity check
            "last10_win_pct":       (0.10, True),   # hot-team / conference tourney credit
        }
        # Q1 win rate: committee explicitly evaluates Quad 1 record
        has_q1 = "q1_win_pct" in df.columns and df["q1_win_pct"].notna().mean() > 0.10
        if has_q1:
            spec["q1_win_pct"] = (0.07, True)  # positive Q1 résumé = committee credit
    else:
        # Pre-NET era: KenPom adj_margin + SOS + record (mirrors old RPI logic)
        spec = {
            "adj_margin":           (0.40, True),
            "wins":                 (0.25, True),
            "strength_of_schedule": (0.20, True),
            "last10_win_pct":       (0.15, True),
        }
    return _weighted_composite(df, spec)


def _region_bias_score(df: pd.DataFrame) -> pd.Series:
    """Mild regional player-origin advantage. Low weight in final model."""
    region_cols = [
        "chicago_guard_count", "nyc_guard_count", "indiana_guard_count",
        "texas_big_count", "southern_big_count", "west_coast_wing_count",
        "local_site_player_count", "host_state_player_count",
    ]
    present = [c for c in region_cols if c in df.columns]
    if not present:
        return pd.Series(50.0, index=df.index)
    raw = df[present].sum(axis=1)
    return _pctile(raw, ascending=True).round(1)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def compute_all_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """Add all sub-score columns to the features DataFrame.

    Expects output from build_team_features.build_features().
    Returns copy with new columns added.
    """
    out = df.copy()
    log.info(f"Computing sub-scores for {len(out)} teams")

    # Ensure road_win_pct exists (build_team_features may have created it)
    if "road_win_pct" not in out.columns:
        if "road_wins" in out.columns and "road_losses" in out.columns:
            total = out["road_wins"] + out["road_losses"]
            out["road_win_pct"] = np.where(total > 0, out["road_wins"] / total, 0.5)
        else:
            out["road_win_pct"] = 0.5

    out["defense_score"]              = _defense_score(out)
    out["efficiency_score"]           = _efficiency_score(out)
    out["guard_play_score"]           = _guard_play_score(out)
    out["clutch_score"]               = _clutch_score(out)
    out["rebounding_score"]           = _rebounding_score(out)
    out["region_bias_score"]          = _region_bias_score(out)
    out["consistency_score"]          = _consistency_score(out)

    # committee_alignment_score needs clutch already computed (for gap calc)
    out["committee_alignment_score"]  = _committee_alignment_score(out)

    # Derived gap: positive = model rates team HIGHER than committee (underseeded)
    #              negative = committee rates team HIGHER than model (overseeded)
    out["model_vs_committee_gap"] = (
        out["efficiency_score"] - out["committee_alignment_score"]
    ).round(1)

    # Cinderella score needs model_vs_committee_gap (computed above)
    out["cinderella_score"] = _cinderella_score(out)

    # First-year coach penalty: propagate to a numeric flag for use in baseline_rules
    if "first_year_coach" in out.columns:
        out["first_year_coach_flag"] = out["first_year_coach"].astype(bool).astype(int)
    else:
        out["first_year_coach_flag"] = 0

    # Foul dependence: already in the CSV if enrich_2026_features was run,
    # otherwise compute from ft_rate
    if "foul_dependence" not in out.columns:
        if "ft_rate" in out.columns:
            out["foul_dependence"] = out["ft_rate"].fillna(out["ft_rate"].median())
        else:
            out["foul_dependence"] = np.nan

    log.info("Sub-scores complete")
    return out
