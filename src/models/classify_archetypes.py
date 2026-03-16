"""Classify teams into tournament archetypes."""
from src.utils.logging_utils import get_logger
log = get_logger(__name__)

def classify_archetype(row):
    exp = row.get("experience_score", 50)
    defense = row.get("defense_score", 50)
    guard = row.get("guard_play_score", 50)
    clutch = row.get("clutch_score", 50)
    contender = row.get("contender_score", 50)
    reb = row.get("rebounding_score", 50)
    age = row.get("avg_age", 21)

    # Balanced Powerhouse: top-tier everything
    if contender >= 70 and defense >= 60 and exp >= 60 and guard >= 55:
        return "Balanced Powerhouse"

    # Fraud Favorite: high ranking but weak profile
    if contender < 50 and row.get("net_rank", 50) <= 10:
        return "Fraud Favorite"

    # Veteran Guard Control: experienced backcourt, high clutch
    if exp >= 65 and guard >= 60 and clutch >= 60:
        return "Veteran Guard Control"

    # Elite Defense Traveler: defense-first, can win ugly
    if defense >= 65 and clutch >= 55:
        return "Elite Defense Traveler"

    # High-Variance Freshman Team: young, talented but fragile
    if age < 20.5 and exp < 45:
        return "High-Variance Freshman Team"

    # Dangerous Low Seed: strong profile, low ranking
    if contender >= 55 and row.get("net_rank", 50) > 20:
        return "Dangerous Low Seed"

    return "Standard Tournament Team"

def classify_all_teams(df):
    log.info(f"Classifying archetypes for {len(df)} teams")
    df = df.copy()
    df["archetype"] = df.apply(classify_archetype, axis=1)
    return df
