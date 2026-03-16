"""Generate text explanations for team profiles. Uses templates (LLM integration optional)."""
import pandas as pd
from src.explain.prompt_templates import TEAM_SUMMARY_TEMPLATE
from src.utils.logging_utils import get_logger
log = get_logger(__name__)

def generate_team_explanation(row):
    """Generate a rule-based explanation (no LLM required)."""
    parts = []
    team = row.get("team", "Unknown")
    archetype = row.get("archetype", "Standard")
    contender = row.get("contender_score", 50)
    expected = row.get("expected_round", "R64")

    parts.append(f"{team} profiles as a {archetype} with a contender score of {contender}/100.")

    # Strengths
    strengths = []
    if row.get("defense_score", 50) >= 65:
        strengths.append("elite defense")
    if row.get("experience_score", 50) >= 65:
        strengths.append("deep experience")
    if row.get("guard_play_score", 50) >= 65:
        strengths.append("strong guard play")
    if row.get("clutch_score", 50) >= 65:
        strengths.append("clutch performance")
    if row.get("rebounding_score", 50) >= 65:
        strengths.append("dominant rebounding")

    if strengths:
        parts.append(f"Key strengths: {', '.join(strengths)}.")

    # Weaknesses
    weaknesses = []
    if row.get("experience_score", 50) < 40:
        weaknesses.append("lack of experience")
    if row.get("clutch_score", 50) < 40:
        weaknesses.append("poor clutch metrics")
    if row.get("defense_score", 50) < 40:
        weaknesses.append("defensive vulnerability")

    if weaknesses:
        parts.append(f"Watch out for: {', '.join(weaknesses)}.")

    parts.append(f"Model projects {expected} exit.")

    # Flags
    flags = []
    if row.get("title_darkhorse_flag"):
        flags.append("DARKHORSE ALERT")
    if row.get("fraud_favorite_flag"):
        flags.append("FRAUD FAVORITE WARNING")
    if row.get("dangerous_low_seed_flag"):
        flags.append("DANGEROUS LOW SEED")
    if flags:
        parts.append(f"[{' | '.join(flags)}]")

    return " ".join(parts)

def generate_all_explanations(scores_df):
    log.info(f"Generating explanations for {len(scores_df)} teams")
    df = scores_df.copy()
    df["explanation_summary"] = df.apply(generate_team_explanation, axis=1)
    return df
