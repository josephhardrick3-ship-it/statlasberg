"""Monte Carlo bracket simulation using pre-built team profiles.

Optimised for speed: win probabilities are computed inline from
pre-cached dicts rather than building pd.Series per game.
"""

import pandas as pd
import numpy as np
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _sg(d: dict, key: str, default: float) -> float:
    """NaN-safe dict getter: returns default when key is missing OR value is NaN."""
    v = d.get(key, default)
    try:
        if np.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _compute_win_prob(fa: dict, fb: dict) -> float:
    """Win probability for team_a over team_b.

    Uses computed sub-scores (0-100, always populated) as primary signal,
    with raw-feature edges as secondary.  NaN-safe throughout.
    """
    def _edge(va, vb, lower_better=False):
        if lower_better:
            diff = vb - va
        else:
            diff = va - vb
        scale = max(abs(va), abs(vb), 1)
        return diff / scale

    # ── Primary: computed sub-scores (reliable, always 0-100) ───────────────
    ca_cs = _sg(fa, "contender_score",  50.0)
    cb_cs = _sg(fb, "contender_score",  50.0)
    ca_df = _sg(fa, "defense_score",    50.0)
    cb_df = _sg(fb, "defense_score",    50.0)
    ca_cl = _sg(fa, "clutch_score",     50.0)
    cb_cl = _sg(fb, "clutch_score",     50.0)
    ca_rb = _sg(fa, "rebounding_score", 50.0)
    cb_rb = _sg(fb, "rebounding_score", 50.0)

    contender_edge = (ca_cs - cb_cs) / 100.0
    defense_edge   = (ca_df - cb_df) / 100.0
    clutch_edge    = (ca_cl - cb_cl) / 100.0
    rebound_edge   = (ca_rb - cb_rb) / 100.0

    # ── Secondary: raw features (only when populated) ───────────────────────
    to_a = _sg(fa, "turnover_pct",  16.0)
    to_b = _sg(fb, "turnover_pct",  16.0)
    # Scale normalisation: turnover_pct may be stored as pct (0.16) or rate (16)
    # Normalise to same scale so the edge is meaningful
    if to_a < 1.0:  # fraction form
        to_a *= 100
    if to_b < 1.0:
        to_b *= 100
    turnover_edge = _edge(to_a, to_b, lower_better=True)

    def3pt_a = _sg(fa, "three_pt_pct", 0.34)
    def3pt_b = _sg(fb, "three_pt_pct", 0.34)
    shooting_edge = _edge(def3pt_a, def3pt_b)

    # ── Consistency / variance bonus ─────────────────────────────────────────
    ca_con = _sg(fa, "consistency_score", 50.0)
    cb_con = _sg(fb, "consistency_score", 50.0)
    consistency_edge = (ca_con - cb_con) / 100.0

    # ── Tempo mismatch variance adjustment ───────────────────────────────────
    # When teams have very different tempos (≥8 possessions/game apart), the
    # stylistic clash adds randomness — slightly widens outcome distribution.
    tempo_a = _sg(fa, "tempo", 68.0)
    tempo_b = _sg(fb, "tempo", 68.0)
    tempo_diff = abs(tempo_a - tempo_b)
    # Slow-pace teams (Houston/Virginia style) disrupt fast teams more than reverse
    if tempo_diff >= 8:
        slower_team_cs = ca_cs if tempo_a < tempo_b else cb_cs
        faster_team_cs = cb_cs if tempo_a < tempo_b else ca_cs
        tempo_disruption = (slower_team_cs - faster_team_cs) / 100.0 * 0.2
    else:
        tempo_disruption = 0.0

    # ── Seed adjustment ──────────────────────────────────────────────────────
    seed_a = _sg(fa, "seed", 8.0)
    seed_b = _sg(fb, "seed", 8.0)
    seed_adj = (seed_b - seed_a) * 0.04

    # ── Logit combination ────────────────────────────────────────────────────
    logit = (contender_edge   * 2.5
             + defense_edge   * 0.8
             + clutch_edge    * 0.7
             + rebound_edge   * 0.4
             + consistency_edge * 0.3
             + turnover_edge  * 0.25
             + shooting_edge  * 0.15
             + tempo_disruption
             + seed_adj)

    prob = 1 / (1 + np.exp(-logit))
    return max(0.02, min(0.98, prob))


def simulate_bracket(bracket_df, features_df, n_sims=10000, seed=42):
    """Run full bracket simulation n_sims times.

    Returns a DataFrame with per-team probabilities for each round:
      sweet_16_pct, elite_8_pct, final_four_pct, championship_pct
    """
    log.info(f"Simulating bracket {n_sims} times")

    feat = {}
    for _, row in features_df.iterrows():
        feat[row["team"]] = row.to_dict()

    pairing_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    regions = bracket_df["region"].unique()
    region_teams = {}
    for region in regions:
        rdf = bracket_df[bracket_df["region"] == region].set_index("seed")
        ordered = []
        for s in pairing_order:
            if s in rdf.index:
                t = rdf.loc[s, "team"]
                ordered.append((t, s))
        region_teams[region] = ordered

    rng = np.random.default_rng(seed)

    # Round-by-round counters
    # Regional rounds: 16→8 (R32), 8→4 (S16), 4→2 (E8), 2→1 (FF)
    # Map bracket length BEFORE round plays → round name of survivors
    #   len=16 → R64 round → survivors made R32
    #   len=8  → R32 round → survivors made Sweet 16
    #   len=4  → S16 round → survivors made Elite 8
    #   len=2  → E8  round → survivor  made Final Four
    sweet_16   = {}   # advanced to Sweet 16 (won R32)
    elite_8    = {}   # advanced to Elite 8  (won Sweet 16)
    final_four = {}   # advanced to Final Four (won regional final)
    champions  = {}   # won the championship

    for _ in range(n_sims):
        ff_teams = []
        for region in regions:
            bracket = list(region_teams[region])
            while len(bracket) > 1:
                bracket_len = len(bracket)
                nxt = []
                for i in range(0, len(bracket), 2):
                    ta, sa = bracket[i]
                    tb, sb = bracket[i + 1]
                    fa_d = {**feat.get(ta, {}), "seed": sa}
                    fb_d = {**feat.get(tb, {}), "seed": sb}
                    prob_a = _compute_win_prob(fa_d, fb_d)
                    winner = ta if rng.random() < prob_a else tb
                    ws = sa if winner == ta else sb
                    nxt.append((winner, ws))

                    # Track round advancement
                    if bracket_len == 8:      # R32 round → winners make Sweet 16
                        sweet_16[winner] = sweet_16.get(winner, 0) + 1
                    elif bracket_len == 4:    # S16 round → winners make Elite 8
                        elite_8[winner] = elite_8.get(winner, 0) + 1
                    elif bracket_len == 2:    # E8 round  → winner makes Final Four
                        final_four[winner] = final_four.get(winner, 0) + 1

                bracket = nxt
            ff_teams.append(bracket[0])

        if len(ff_teams) >= 4:
            ta, sa = ff_teams[0]; tb, sb = ff_teams[1]
            p = _compute_win_prob({**feat.get(ta, {}), "seed": sa}, {**feat.get(tb, {}), "seed": sb})
            w1 = ta if rng.random() < p else tb
            s1 = sa if w1 == ta else sb

            ta, sa = ff_teams[2]; tb, sb = ff_teams[3]
            p = _compute_win_prob({**feat.get(ta, {}), "seed": sa}, {**feat.get(tb, {}), "seed": sb})
            w2 = ta if rng.random() < p else tb
            s2 = sa if w2 == ta else sb

            p = _compute_win_prob({**feat.get(w1, {}), "seed": s1}, {**feat.get(w2, {}), "seed": s2})
            champ = w1 if rng.random() < p else w2
            champions[champ] = champions.get(champ, 0) + 1

    # ── Build results DataFrame with all rounds ────────────────────────────
    all_teams = set(list(sweet_16) + list(elite_8) + list(final_four) + list(champions))
    rows = []
    for t in all_teams:
        rows.append({
            "team": t,
            "sweet_16": sweet_16.get(t, 0),
            "sweet_16_pct": round(sweet_16.get(t, 0) / n_sims * 100, 2),
            "elite_8": elite_8.get(t, 0),
            "elite_8_pct": round(elite_8.get(t, 0) / n_sims * 100, 2),
            "final_four": final_four.get(t, 0),
            "final_four_pct": round(final_four.get(t, 0) / n_sims * 100, 2),
            "championships": champions.get(t, 0),
            "championship_pct": round(champions.get(t, 0) / n_sims * 100, 2),
        })

    result_df = pd.DataFrame(rows).sort_values("championship_pct", ascending=False).reset_index(drop=True)

    if len(result_df) > 0:
        log.info(f"Simulation complete. Top: {result_df.iloc[0]['team']} ({result_df.iloc[0]['championship_pct']}%)")
    return result_df
