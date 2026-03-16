"""Evaluation metrics for backtesting predictions."""

import pandas as pd
import numpy as np


def game_accuracy(predictions: pd.DataFrame, results: pd.DataFrame) -> float:
    """Calculate % of correctly predicted game winners."""
    merged = predictions.merge(results, on=["season", "round", "team", "opponent"])
    if len(merged) == 0:
        return 0.0
    correct = (merged["predicted_winner"] == merged["actual_winner"]).sum()
    return correct / len(merged)


def round_accuracy(predictions: pd.DataFrame, results: pd.DataFrame, round_name: str) -> dict:
    """Calculate accuracy for a specific round."""
    pred_round = predictions[predictions["round"] == round_name]
    res_round = results[results["round"] == round_name]
    merged = pred_round.merge(res_round, on=["season", "team"], suffixes=("_pred", "_actual"))
    if len(merged) == 0:
        return {"round": round_name, "correct": 0, "total": 0, "accuracy": 0.0}
    correct = (merged["advanced_pred"] == merged["advanced_actual"]).sum()
    return {
        "round": round_name,
        "correct": int(correct),
        "total": len(merged),
        "accuracy": correct / len(merged),
    }


def upset_detection_rate(predictions: pd.DataFrame, results: pd.DataFrame) -> dict:
    """How often the model correctly flagged upsets."""
    flagged = predictions[predictions["upset_flag"] == True]
    actual_upsets = results[results["upset_flag"] == True]
    if len(actual_upsets) == 0:
        return {"flagged": len(flagged), "actual_upsets": 0, "caught": 0, "rate": 0.0}
    caught = flagged.merge(actual_upsets, on=["season", "team"])
    return {
        "flagged": len(flagged),
        "actual_upsets": len(actual_upsets),
        "caught": len(caught),
        "rate": len(caught) / len(actual_upsets) if len(actual_upsets) > 0 else 0.0,
    }


def champion_rank(team_scores: pd.DataFrame, actual_champion: str) -> int:
    """Where the actual champion ranked in model's contender scores."""
    sorted_scores = team_scores.sort_values("contender_score", ascending=False).reset_index(drop=True)
    match = sorted_scores[sorted_scores["team"] == actual_champion]
    if len(match) == 0:
        return -1
    return int(match.index[0]) + 1


def percentile_rank(series: pd.Series, value: float) -> float:
    """Return percentile rank (0-100) of a value within a series."""
    return (series < value).sum() / len(series) * 100
