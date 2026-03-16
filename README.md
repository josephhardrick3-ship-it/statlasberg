# March Madness Predictive Analysis Bot

A data-driven NCAA tournament prediction system that scores teams, classifies archetypes, and simulates brackets using Monte Carlo methods.

## How It Works

The bot evaluates tournament teams across six dimensions, each scored 0–100:

| Sub-Score | What It Measures |
|-----------|-----------------|
| **Defense** | Adj. defense, rebounding, opponent shooting, turnover forcing |
| **Experience** | Age, senior minutes, returning starters, coach tournament games |
| **Guard Play** | Backcourt experience, ball security, free throw shooting |
| **Clutch** | Close-game record, FT%, late-game turnovers |
| **Rebounding** | Offensive and defensive board rates |
| **Region Bias** | Player-origin advantage (Chicago guards, Texas bigs, etc.) |

These feed a weighted **Contender Score** (0–100) that predicts tournament ceiling, plus an **Upset Risk Score** that flags early-exit danger.

### Archetypes

Every team gets classified into one of six tournament archetypes:

- **Balanced Powerhouse** — elite offense + defense + experience
- **Veteran Guard Control** — experienced backcourt, high clutch rating
- **Elite Defense Traveler** — defense-first, wins ugly on the road
- **Fraud Favorite** — high seed / low profile (overrated by committee)
- **High-Variance Freshman Team** — talented but fragile
- **Dangerous Low Seed** — strong profile, overlooked by rankings

### 10 Historical Patterns Baked In

1. Guard-dominated teams advance further
2. Elite defense travels (most champions are top-20 defense)
3. High turnover rate predicts early exits
4. Older rosters outperform younger ones
5. Freshman-heavy guard rotations are volatile
6. 12-vs-5 upsets follow specific patterns
7. Teams that win ugly games survive longer
8. Balanced offense beats one-star teams
9. Rebounding wins tournament games
10. Travel/regional advantage matters at the margins

## Setup

```bash
pip install pandas numpy pyyaml rapidfuzz
pip install pytest  # for tests
pip install streamlit  # for dashboard (optional)
```

## Usage

### Score all teams (sample data)
```bash
python run_pipeline.py
```

### Score from your own data
```bash
python run_pipeline.py --csv data/raw/teams/my_team_stats.csv
```

Your CSV needs columns like `team`, `adj_offense`, `adj_defense`, `adj_margin`, `turnover_pct`, `ft_pct`, `avg_age`, etc. See `src/utils/constants.py` for the full schema.

### Score + simulate bracket
```bash
python run_pipeline.py --bracket data/brackets/bracket_2026.csv --simulate --sims 10000
```

Bracket CSV format:
```
season,region,seed,team
2026,East,1,Houston
2026,East,2,Duke
...
```

### Run backtest
```bash
python run_backtest.py
python run_backtest.py --seasons 2022 2023 2024
```

### Launch dashboard
```bash
streamlit run src/app/streamlit_app.py
```

### Run tests
```bash
pytest tests/ -v
```

## Project Structure

```
march-madness-bot/
├── run_pipeline.py          # Main entry point
├── run_backtest.py          # Historical validation
├── config/
│   ├── settings.yaml        # Seasons, paths, model config
│   ├── feature_weights.yaml # All sub-score and matchup weights
│   └── region_map.yaml      # Player-origin region definitions
├── data/
│   ├── raw/                 # Input CSVs (teams, rosters, games)
│   ├── features/            # Computed feature tables
│   ├── outputs/             # team_scores.csv, simulation_results.csv
│   └── brackets/            # Bracket files (post-Selection Sunday)
├── src/
│   ├── ingest/              # Data loading (CSV, stubs for scraping)
│   ├── clean/               # Name normalization, data cleaning
│   ├── features/            # Feature engineering + sub-scores
│   ├── models/              # Scoring, archetypes, bracket sim
│   ├── explain/             # Text explanation generator
│   ├── app/                 # Streamlit dashboard
│   └── utils/               # I/O, logging, constants, metrics
└── tests/
```

## Data Sources

The bot is designed to work with data from:
- **Bart Torvik** (barttorvik.com) — adjusted efficiency, tempo-free profiles
- **Sports Reference** (sports-reference.com/cbb) — season stats, historical
- **NCAA** (stats.ncaa.org) — NET rankings, official stats
- **ESPN** — game-level data, play-by-play

Currently ships with a sample data generator for testing. Plug in real CSVs by placing them in `data/raw/teams/` and running with `--csv`.

## Workflow

**Before Selection Sunday (March 15):**
1. Download team stats → `data/raw/teams/team_stats_2026.csv`
2. `python run_pipeline.py --csv data/raw/teams/team_stats_2026.csv`
3. Review `data/outputs/team_scores.csv` for pre-bracket rankings

**After Selection Sunday:**
4. Create bracket CSV → `data/brackets/bracket_2026.csv`
5. `python run_pipeline.py --csv data/raw/teams/team_stats_2026.csv --bracket data/brackets/bracket_2026.csv --simulate`
6. Review championship probabilities and matchup analysis
