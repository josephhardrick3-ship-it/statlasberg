"""Tests for the March Madness Bot pipeline."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest.fetch_team_stats import generate_sample_data
from src.features.build_team_features import build_features
from src.features.compute_subscores import compute_all_subscores
from src.models.baseline_rules import score_all_teams, compute_contender_score
from src.models.classify_archetypes import classify_archetype, classify_all_teams
from src.explain.generate_explanations import generate_team_explanation
from src.clean.normalize_team_names import normalize


# ── Fixtures ──────────────────────────────────────

@pytest.fixture
def sample_data():
    return generate_sample_data(season=2024, n_teams=68)

@pytest.fixture
def scored_data(sample_data):
    df = build_features(sample_data)
    df = compute_all_subscores(df)
    df = score_all_teams(df)
    df = classify_all_teams(df)
    return df


# ── Data generation ───────────────────────────────

class TestSampleData:
    def test_generates_correct_count(self):
        df = generate_sample_data(n_teams=32)
        assert len(df) == 32

    def test_has_required_columns(self, sample_data):
        required = ["season", "team", "adj_offense", "adj_defense", "adj_margin",
                     "turnover_pct", "ft_pct", "avg_age", "wins", "losses"]
        for col in required:
            assert col in sample_data.columns, f"Missing column: {col}"

    def test_stats_in_plausible_ranges(self, sample_data):
        assert sample_data["adj_offense"].between(80, 140).all()
        assert sample_data["adj_defense"].between(70, 130).all()
        assert sample_data["turnover_pct"].between(5, 30).all()
        assert sample_data["ft_pct"].between(50, 95).all()

    def test_unique_teams(self, sample_data):
        assert sample_data["team"].nunique() == len(sample_data)


# ── Feature building ─────────────────────────────

class TestFeatures:
    def test_build_features_adds_columns(self, sample_data):
        df = build_features(sample_data)
        # Should add percentile columns
        assert any("_pctile" in c for c in df.columns)

    def test_percentiles_0_to_100(self, sample_data):
        df = build_features(sample_data)
        for col in df.columns:
            if "_pctile" in col:
                assert df[col].between(0, 100).all(), f"{col} out of range"


# ── Sub-scores ───────────────────────────────────

class TestSubscores:
    def test_all_subscores_added(self, sample_data):
        df = build_features(sample_data)
        df = compute_all_subscores(df)
        for col in ["defense_score", "experience_score", "guard_play_score",
                     "clutch_score", "rebounding_score", "region_bias_score"]:
            assert col in df.columns, f"Missing sub-score: {col}"

    def test_subscores_0_to_100(self, sample_data):
        df = build_features(sample_data)
        df = compute_all_subscores(df)
        for col in ["defense_score", "experience_score", "guard_play_score",
                     "clutch_score", "rebounding_score"]:
            assert df[col].between(0, 101).all(), f"{col} out of range"


# ── Scoring model ────────────────────────────────

class TestBaseline:
    def test_contender_score_range(self, scored_data):
        assert scored_data["contender_score"].between(0, 100).all()

    def test_upset_risk_range(self, scored_data):
        assert scored_data["upset_risk_score"].between(0, 100).all()

    def test_expected_round_valid(self, scored_data):
        valid_rounds = {"R64", "R32", "Sweet 16", "Elite 8", "Final Four", "Championship"}
        for r in scored_data["expected_round"].unique():
            assert r in valid_rounds, f"Unknown round: {r}"

    def test_higher_contender_means_deeper_expected(self, scored_data):
        top = scored_data.nlargest(5, "contender_score")
        bottom = scored_data.nsmallest(5, "contender_score")
        # Top teams shouldn't all be projected R64
        assert not (top["expected_round"] == "R64").all()


# ── Archetypes ───────────────────────────────────

class TestArchetypes:
    def test_archetype_assigned(self, scored_data):
        assert "archetype" in scored_data.columns
        assert scored_data["archetype"].notna().all()

    def test_known_archetypes(self, scored_data):
        known = {
            "Balanced Powerhouse", "Fraud Favorite", "Veteran Guard Control",
            "Elite Defense Traveler", "High-Variance Freshman Team",
            "Dangerous Low Seed", "Standard Tournament Team",
        }
        for a in scored_data["archetype"].unique():
            assert a in known, f"Unknown archetype: {a}"

    def test_fraud_favorite_classification(self):
        row = {
            "contender_score": 40, "net_rank": 5, "experience_score": 50,
            "defense_score": 50, "guard_play_score": 50, "clutch_score": 50,
            "rebounding_score": 50, "avg_age": 21,
        }
        assert classify_archetype(row) == "Fraud Favorite"


# ── Explanations ─────────────────────────────────

class TestExplanations:
    def test_explanation_generated(self, scored_data):
        row = scored_data.iloc[0]
        exp = generate_team_explanation(row)
        assert len(exp) > 50
        assert row["team"] in exp

    def test_flags_in_explanation(self):
        row = pd.Series({
            "team": "TestU", "archetype": "Fraud Favorite",
            "contender_score": 45, "expected_round": "R64",
            "defense_score": 30, "experience_score": 30,
            "guard_play_score": 30, "clutch_score": 30,
            "rebounding_score": 50,
            "title_darkhorse_flag": False, "fraud_favorite_flag": True,
            "dangerous_low_seed_flag": False,
        })
        exp = generate_team_explanation(row)
        assert "FRAUD FAVORITE" in exp


# ── Team name normalization ──────────────────────

class TestNormalization:
    def test_alias_resolution(self):
        assert normalize("UConn") == "Connecticut"
        assert normalize("UNC") == "North Carolina"
        assert normalize("LSU") == "Louisiana State"

    def test_seed_stripping(self):
        assert normalize("Duke (1)") == "Duke"

    def test_passthrough(self):
        assert normalize("Houston") == "Houston"


# ── Integration test ─────────────────────────────

class TestFullPipeline:
    def test_end_to_end(self):
        """Full pipeline should run without errors and produce valid output."""
        df = generate_sample_data(season=2024, n_teams=32)
        df = build_features(df)
        df = compute_all_subscores(df)
        df = score_all_teams(df)
        df = classify_all_teams(df)

        assert len(df) == 32
        assert "contender_score" in df.columns
        assert "archetype" in df.columns
        assert df["contender_score"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
