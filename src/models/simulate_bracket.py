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
    """Run full bracket simulation n_sims times."""
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
    champions = {}

    for _ in range(n_sims):
        final_four = []
        for region in regions:
            bracket = list(region_teams[region])
            while len(bracket) > 1:
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
                bracket = nxt
            final_four.append(bracket[0])

        if len(final_four) >= 4:
            ta, sa = final_four[0]; tb, sb = final_four[1]
            p = _compute_win_prob({**feat.get(ta, {}), "seed": sa}, {**feat.get(tb, {}), "seed": sb})
            w1 = ta if rng.random() < p else tb
            s1 = sa if w1 == ta else sb

            ta, sa = final_four[2]; tb, sb = final_four[3]
            p = _compute_win_prob({**feat.get(ta, {}), "seed": sa}, {**feat.get(tb, {}), "seed": sb})
            w2 = ta if rng.random() < p else tb
            s2 = sa if w2 == ta else sb

            p = _compute_win_prob({**feat.get(w1, {}), "seed": s1}, {**feat.get(w2, {}), "seed": s2})
            champ = w1 if rng.random() < p else w2
            champions[champ] = champions.get(champ, 0) + 1

    champ_df = pd.DataFrame([
        {"team": t, "championships": c, "championship_pct": round(c / n_sims * 100, 2)}
        for t, c in sorted(champions.items(), key=lambda x: -x[1])
    ])

    if len(champ_df) > 0:
        log.info(f"Simulation complete. Top: {champ_df.iloc[0]['team']} ({champ_df.iloc[0]['championship_pct']}%)")
    return champ_df
