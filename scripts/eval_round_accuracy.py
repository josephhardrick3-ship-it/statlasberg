#!/usr/bin/env python3
"""
eval_round_accuracy.py
──────────────────────
Evaluate how well the model identifies actual Sweet 16, Elite 8, and
Final Four teams for each historical season (2015–2025).

For each year, the model is run on real SR data and we check:
  - What model rank did each Final Four / Elite 8 / Sweet 16 team receive?
  - How many were in model top 4 / 8 / 16 / 20?
  - Average model rank per round

Answers the question: "How good is the model at picking deeper rounds?"
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.ingest.fetch_team_stats import fetch_from_sports_ref
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth: actual tournament results 2015–2025
# Sources: NCAA.com / Sports-Reference tournament history
# ─────────────────────────────────────────────────────────────────────────────

CHAMPIONS = {
    2015: "Duke",        2016: "Villanova",     2017: "North Carolina",
    2018: "Villanova",   2019: "Virginia",       2021: "Baylor",
    2022: "Kansas",      2023: "Connecticut",    2024: "Connecticut",
    2025: "Florida",
}

FINAL_FOUR = {
    2015: ["Duke", "Wisconsin", "Michigan State", "Kentucky"],
    2016: ["Villanova", "Oklahoma", "Syracuse", "North Carolina"],
    2017: ["North Carolina", "Oregon", "South Carolina", "Gonzaga"],
    2018: ["Villanova", "Kansas", "Michigan", "Loyola-Chicago"],
    2019: ["Virginia", "Auburn", "Michigan State", "Texas Tech"],
    2021: ["Baylor", "Houston", "UCLA", "Gonzaga"],
    2022: ["Kansas", "Villanova", "Duke", "North Carolina"],
    2023: ["Connecticut", "San Diego State", "Florida Atlantic", "Miami FL"],
    2024: ["Connecticut", "Purdue", "Alabama", "North Carolina State"],
    2025: ["Florida", "Houston", "Duke", "Auburn"],
}

ELITE_8 = {
    2015: ["Duke", "Wisconsin", "Michigan State", "Kentucky",
           "Gonzaga", "Louisville", "West Virginia", "Arizona"],
    2016: ["Villanova", "Oklahoma", "Syracuse", "North Carolina",
           "Kansas", "Oregon", "Notre Dame", "Iowa State"],
    2017: ["North Carolina", "Oregon", "South Carolina", "Gonzaga",
           "Kentucky", "Kansas", "Florida", "Xavier"],
    2018: ["Villanova", "Kansas", "Michigan", "Loyola-Chicago",
           "Texas Tech", "Duke", "Florida State", "Kansas State"],
    2019: ["Virginia", "Auburn", "Michigan State", "Texas Tech",
           "Oregon", "North Carolina", "LSU", "Gonzaga"],
    2021: ["Baylor", "Houston", "UCLA", "Gonzaga",
           "Arkansas", "Oregon State", "Michigan", "USC"],
    2022: ["Kansas", "Villanova", "Duke", "North Carolina",
           "Miami FL", "Houston", "Texas Tech", "Saint Peter's"],
    2023: ["Connecticut", "San Diego State", "Florida Atlantic", "Miami FL",
           "Arkansas", "Creighton", "Tennessee", "Texas"],
    2024: ["Connecticut", "Purdue", "Alabama", "North Carolina State",
           "Illinois", "Tennessee", "Clemson", "Duke"],
    2025: ["Florida", "Houston", "Duke", "Auburn",
           "Connecticut", "Tennessee", "Arizona", "Michigan State"],
}

SWEET_16 = {
    2015: ["Duke", "Wisconsin", "Michigan State", "Kentucky",
           "Gonzaga", "Louisville", "West Virginia", "Arizona",
           "North Carolina", "Notre Dame", "Wichita State", "Utah",
           "North Carolina State", "Georgetown", "Villanova", "Maryland"],
    2016: ["Villanova", "Oklahoma", "Syracuse", "North Carolina",
           "Kansas", "Oregon", "Notre Dame", "Iowa State",
           "Indiana", "Miami FL", "Texas A&M", "Gonzaga",
           "Virginia", "Utah", "Wisconsin", "Purdue"],
    2017: ["North Carolina", "Oregon", "South Carolina", "Gonzaga",
           "Kentucky", "Kansas", "Florida", "Xavier",
           "Butler", "Michigan State", "Michigan", "Louisville",
           "Baylor", "Purdue", "West Virginia", "Arizona"],
    2018: ["Villanova", "Kansas", "Michigan", "Loyola-Chicago",
           "Texas Tech", "Duke", "Florida State", "Kansas State",
           "Purdue", "Texas A&M", "Cincinnati", "Nevada",
           "Gonzaga", "Michigan State", "Clemson", "Rhode Island"],
    2019: ["Virginia", "Auburn", "Michigan State", "Texas Tech",
           "Oregon", "North Carolina", "LSU", "Gonzaga",
           "Duke", "Virginia Tech", "Michigan", "Florida State",
           "Houston", "Purdue", "Tennessee", "Iowa State"],
    2021: ["Baylor", "Houston", "UCLA", "Gonzaga",
           "Arkansas", "Oregon State", "Michigan", "USC",
           "Villanova", "Oral Roberts", "Florida State", "Syracuse",
           "Wisconsin", "Loyola-Chicago", "Creighton", "Alabama"],
    2022: ["Kansas", "Villanova", "Duke", "North Carolina",
           "Miami FL", "Houston", "Texas Tech", "Saint Peter's",
           "Iowa State", "Providence", "Michigan", "Wisconsin",
           "Arkansas", "Gonzaga", "UCLA", "Tennessee"],
    2023: ["Connecticut", "San Diego State", "Florida Atlantic", "Miami FL",
           "Arkansas", "Creighton", "Tennessee", "Texas",
           "Alabama", "Maryland", "Kansas State", "Michigan State",
           "Gonzaga", "Xavier", "Houston", "Indiana"],
    2024: ["Connecticut", "Purdue", "Alabama", "North Carolina State",
           "Illinois", "Tennessee", "Clemson", "Duke",
           "Iowa State", "North Carolina", "Marquette", "Arizona",
           "Gonzaga", "San Diego State", "Houston", "Duke"],
    2025: ["Florida", "Houston", "Duke", "Auburn",
           "Connecticut", "Tennessee", "Arizona", "Michigan State",
           "Maryland", "Texas A&M", "St. Johns", "Marquette",
           "Iowa State", "Alabama", "Texas Tech", "Kentucky"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization aliases  (tournament names → SR data names)
# ─────────────────────────────────────────────────────────────────────────────
ALIASES = {
    "UConn":                 "Connecticut",
    "NC State":              "North Carolina State",
    "N.C. State":            "North Carolina State",
    "Saint Peter's":         "Saint Peters",
    "Loyola-Chicago":        "Loyola-Chicago",
    "Loyola Chicago":        "Loyola-Chicago",
    "LSU":                   "Louisiana State",
    "Miami FL":              "Miami FL",
    "Miami (FL)":            "Miami FL",
    "Miami (Ohio)":          "Miami OH",
    "Florida Atlantic":      "Florida Atlantic",
    "FAU":                   "Florida Atlantic",
    "USC":                   "Southern California",
    "BYU":                   "Brigham Young",
    "SMU":                   "Southern Methodist",
    "TCU":                   "Texas Christian",
    "SDSU":                  "San Diego State",
    "St. Johns":             "St. John's",
    "St. John's":            "St. John's",
    "Iowa St.":              "Iowa State",
    "Oregon St.":            "Oregon State",
    "Kansas St.":            "Kansas State",
    "Kansas State":          "Kansas State",
    "Michigan St.":          "Michigan State",
    "Mississippi St.":       "Mississippi State",
}


def normalize(name: str) -> str:
    return ALIASES.get(name, name)


def find_rank(df: pd.DataFrame, team: str) -> int:
    """Return model rank for a team. Exact → first-word fallback → -1."""
    norm = normalize(team)
    exact = df[df["team"] == norm]
    if not exact.empty:
        return int(exact.iloc[0]["model_rank"])
    # Try original name too
    exact2 = df[df["team"] == team]
    if not exact2.empty:
        return int(exact2.iloc[0]["model_rank"])
    # First-word fuzzy fallback
    first = norm.split()[0]
    partial = df[df["team"].str.startswith(first, na=False)]
    if not partial.empty:
        return int(partial.sort_values("model_rank").iloc[0]["model_rank"])
    return -1


def run_model(season: int) -> pd.DataFrame:
    df = fetch_from_sports_ref(season)
    df = build_features(df)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df = df.sort_values("contender_score", ascending=False).reset_index(drop=True)
    df["model_rank"] = df.index + 1
    return df


def eval_round(df: pd.DataFrame, teams: list, label: str) -> dict:
    """Evaluate model accuracy for a given round's teams."""
    n = len(teams)
    ranks = []
    found = []
    not_found = []

    for t in teams:
        r = find_rank(df, t)
        if r != -1:
            ranks.append(r)
            found.append((t, r))
        else:
            not_found.append(t)

    if not ranks:
        return {"label": label, "n": n, "found": 0, "avg_rank": None,
                "median_rank": None, "in_top4": 0, "in_top8": 0,
                "in_top16": 0, "in_top20": 0, "not_found": not_found}

    return {
        "label":       label,
        "n":           n,
        "found":       len(ranks),
        "avg_rank":    round(np.mean(ranks), 1),
        "median_rank": round(np.median(ranks), 1),
        "in_top4":     sum(1 for r in ranks if r <= 4),
        "in_top8":     sum(1 for r in ranks if r <= 8),
        "in_top16":    sum(1 for r in ranks if r <= 16),
        "in_top20":    sum(1 for r in ranks if r <= 20),
        "not_found":   not_found,
        "_ranks":      sorted(ranks),
        "_teams":      sorted(found, key=lambda x: x[1]),
    }


W = 78


def print_season(season: int, df: pd.DataFrame):
    ff   = eval_round(df, FINAL_FOUR[season],   "Final Four")
    e8   = eval_round(df, ELITE_8[season],       "Elite 8")
    s16  = eval_round(df, SWEET_16.get(season, []), "Sweet 16")
    champ = CHAMPIONS[season]
    champ_rank = find_rank(df, champ)

    print(f"\n{'─'*W}")
    print(f"  {season}  —  Champion: {champ} (model #{champ_rank if champ_rank != -1 else 'NF'})")
    print(f"{'─'*W}")
    print(f"  {'Round':<12} {'Found':>6} {'Avg Rank':>9} {'Med':>6} "
          f"{'Top 4':>6} {'Top 8':>7} {'Top16':>7} {'Top20':>7}")
    print(f"  {'-'*70}")

    for ev in [s16, e8, ff]:
        if not ev["n"]:
            continue
        bar_top = ev["in_top8"] if ev["label"] != "Sweet 16" else ev["in_top16"]
        bar_n   = ev["n"]
        bar_str = "█" * bar_top + "░" * (bar_n - bar_top)
        print(f"  {ev['label']:<12} {ev['found']:>4}/{ev['n']:<2} "
              f"{str(ev['avg_rank']) if ev['avg_rank'] else 'N/A':>9} "
              f"{str(ev['median_rank']) if ev['median_rank'] else 'N/A':>6} "
              f"{ev['in_top4']:>4}/{ev['n']:<2} "
              f"{ev['in_top8']:>4}/{ev['n']:<2} "
              f"{ev['in_top16']:>4}/{ev['n']:<2} "
              f"{ev['in_top20']:>4}/{ev['n']:<2}")

    # Spotlight: where did model miss FF/E8 teams, and what was model top 5?
    model_top5 = df.head(5)["team"].tolist()
    ff_teams   = [normalize(t) for t in FINAL_FOUR[season]]
    hits = [t for t in model_top5 if t in ff_teams or any(t.startswith(n.split()[0]) for n in ff_teams)]

    print(f"\n  Model top 5:   {' | '.join(f'#{i+1} {t}' for i,t in enumerate(model_top5))}")
    if hits:
        print(f"  FF overlap:    {hits}")

    # Flag any FF/E8 teams ranked very poorly
    missed = [(t, r) for t, r in [
        (t, find_rank(df, t)) for t in ELITE_8[season]
    ] if r != -1 and r > 30]
    if missed:
        print(f"  ⚠️  E8 teams ranked > #30: {[(t, f'#{r}') for t,r in missed]}")

    return {"season": season, "ff": ff, "e8": e8, "s16": s16,
            "champ_rank": champ_rank}


def main():
    print("=" * W)
    print("  MODEL ACCURACY — SWEET 16 / ELITE 8 / FINAL FOUR  (2015–2025)")
    print("=" * W)

    seasons      = sorted(FINAL_FOUR.keys())
    all_results  = []

    for season in seasons:
        df = run_model(season)
        result = print_season(season, df)
        all_results.append(result)

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  10-YEAR AGGREGATE SUMMARY")
    print(f"{'='*W}")

    def agg(round_key, stat, fmt=".1f"):
        vals = [r[round_key][stat] for r in all_results
                if r[round_key]["n"] > 0 and r[round_key][stat] is not None]
        if not vals:
            return "N/A"
        return format(np.mean(vals), fmt)

    def pct(round_key, top_k, n_teams):
        hits  = sum(r[round_key][f"in_top{top_k}"] for r in all_results if r[round_key]["n"] > 0)
        total = sum(r[round_key]["n"] for r in all_results if r[round_key]["n"] > 0)
        return f"{hits}/{total} ({100*hits/total:.0f}%)" if total else "N/A"

    print(f"\n  {'Metric':<48} {'S16':>10} {'E8':>10} {'FF':>10}")
    print(f"  {'-'*78}")
    print(f"  {'Avg model rank of round teams':<48} "
          f"{agg('s16','avg_rank'):>10} {agg('e8','avg_rank'):>10} {agg('ff','avg_rank'):>10}")
    print(f"  {'Median model rank':<48} "
          f"{agg('s16','median_rank'):>10} {agg('e8','median_rank'):>10} {agg('ff','median_rank'):>10}")
    print(f"  {'% teams in model top 4':<48} "
          f"{'—':>10} {'—':>10} {pct('ff','4',4):>10}")
    print(f"  {'% teams in model top 8':<48} "
          f"{'—':>10} {pct('e8','8',8):>10} {pct('ff','8',4):>10}")
    print(f"  {'% teams in model top 16':<48} "
          f"{pct('s16','16',16):>10} {pct('e8','16',8):>10} {pct('ff','16',4):>10}")
    print(f"  {'% teams in model top 20':<48} "
          f"{pct('s16','20',16):>10} {pct('e8','20',8):>10} {pct('ff','20',4):>10}")

    # Champion summary
    champ_ranks = [r["champ_rank"] for r in all_results if r["champ_rank"] != -1]
    print(f"\n  Champion accuracy ({len(champ_ranks)}/10 found):")
    print(f"    Avg champion rank:      {np.mean(champ_ranks):.1f}")
    print(f"    Champions in top 3:     {sum(1 for r in champ_ranks if r <= 3)}/10")
    print(f"    Champions in top 5:     {sum(1 for r in champ_ranks if r <= 5)}/10")
    print(f"    Champions in top 10:    {sum(1 for r in champ_ranks if r <= 10)}/10")
    print(f"    Champions in top 20:    {sum(1 for r in champ_ranks if r <= 20)}/10")

    # Per-year champion table
    print(f"\n  Year-by-year champion ranks:")
    print(f"  {'Year':<6} {'Champion':<26} {'Rank':>5}  {'Bar'}")
    print(f"  {'-'*55}")
    for r in all_results:
        champ = CHAMPIONS[r["season"]]
        rank  = r["champ_rank"]
        if rank == -1:
            bar = "team not found"
        elif rank <= 4:
            bar = "▓" * rank + " ✅ TOP 4"
        elif rank <= 8:
            bar = "▓" * rank + " 🟡 TOP 8"
        elif rank <= 15:
            bar = "▓" * rank + " 🔶 TOP 15"
        else:
            bar = "▓" * min(rank, 30) + f" ❌ #{rank}"
        print(f"  {r['season']:<6} {champ:<26} {rank if rank != -1 else 'NF':>5}  {bar}")

    # Best/worst round predictions
    print(f"\n  Best years (most FF teams in model top 8):")
    ff_top8 = sorted(all_results, key=lambda x: -x["ff"]["in_top8"])
    for r in ff_top8[:3]:
        print(f"    {r['season']}: {r['ff']['in_top8']}/4 FF in top 8  "
              f"(avg FF rank {r['ff']['avg_rank']})")

    print(f"\n  Worst years (fewest FF teams in model top 8):")
    for r in ff_top8[-3:]:
        print(f"    {r['season']}: {r['ff']['in_top8']}/4 FF in top 8  "
              f"(avg FF rank {r['ff']['avg_rank']})")

    # Save CSV
    rows = []
    for r in all_results:
        rows.append({
            "season":          r["season"],
            "champion":        CHAMPIONS[r["season"]],
            "champ_rank":      r["champ_rank"],
            "ff_avg_rank":     r["ff"]["avg_rank"],
            "ff_in_top4":      r["ff"]["in_top4"],
            "ff_in_top8":      r["ff"]["in_top8"],
            "e8_avg_rank":     r["e8"]["avg_rank"],
            "e8_in_top8":      r["e8"]["in_top8"],
            "e8_in_top16":     r["e8"]["in_top16"],
            "s16_avg_rank":    r["s16"]["avg_rank"],
            "s16_in_top16":    r["s16"]["in_top16"],
            "s16_in_top20":    r["s16"]["in_top20"],
        })
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "outputs", "round_accuracy.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    main()
