"""I/O utilities for loading configs, reading/writing CSVs."""

import os
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from config/ directory."""
    path = os.path.join(PROJECT_ROOT, "config", filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_settings() -> dict:
    return load_yaml("settings.yaml")


def load_region_map() -> dict:
    return load_yaml("region_map.yaml")


def load_feature_weights() -> dict:
    return load_yaml("feature_weights.yaml")


def read_csv(relative_path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV relative to project root."""
    path = os.path.join(PROJECT_ROOT, relative_path)
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, relative_path: str, index: bool = False):
    """Write a DataFrame to CSV relative to project root."""
    path = os.path.join(PROJECT_ROOT, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  Wrote {len(df)} rows to {relative_path}")


def ensure_dirs():
    """Ensure all data directories exist."""
    dirs = [
        "data/raw/teams", "data/raw/games", "data/raw/play_by_play",
        "data/raw/rosters", "data/raw/players", "data/raw/coaches",
        "data/raw/tournament_history",
        "data/interim", "data/features", "data/outputs", "data/brackets",
    ]
    for d in dirs:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
