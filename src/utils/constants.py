"""Constants used across the March Madness bot."""

TEAM_SEASON_COLS = [
    "season", "team", "conference", "coach", "coach_years_at_school",
    "coach_ncaa_games", "coach_sweet16s", "coach_finalfours",
    "wins", "losses", "win_pct", "road_wins", "road_losses",
    "neutral_wins", "neutral_losses",
    "net_rank", "strength_of_schedule", "quad1_wins", "quad2_wins",
    "adj_offense", "adj_defense", "adj_margin", "tempo",
    "turnover_pct", "opp_turnover_pct", "off_rebound_pct", "def_rebound_pct",
    "ft_rate", "ft_pct", "three_pa_rate", "three_pt_pct", "opp_three_pt_pct",
    "points_per_game", "points_allowed_per_game",
    "close_game_record", "last10_win_pct", "last10_adj_margin",
    "avg_age", "underclass_minutes_pct", "freshman_minutes_pct",
    "freshman_guard_minutes_pct", "sophomore_minutes_pct",
    "junior_minutes_pct", "senior_minutes_pct",
    "returning_starters", "returning_minutes_pct",
    "primary_guard_experience_score", "backcourt_experience_score",
    "frontcourt_experience_score", "bench_minutes_pct", "bench_points_pct",
    "injury_flag", "star_player_flag",
    "chicago_guard_count", "nyc_guard_count", "indiana_guard_count",
    "texas_big_count", "southern_big_count", "west_coast_wing_count",
    "local_site_player_count", "host_state_player_count",
    "archetype_label",
    "made_tournament", "won_round64", "won_round32",
    "made_sweet16", "made_elite8", "made_final4", "won_title",
]

PLAYER_ORIGIN_COLS = [
    "season", "team", "player", "position", "height", "class_year",
    "starter_flag", "minutes_pct", "usage_pct",
    "home_city", "home_state", "metro_region", "bias_region", "aau_program",
    "high_school", "is_guard", "is_wing", "is_big",
    "from_chicagoland", "from_nyc", "from_indiana", "from_texas",
    "from_south", "from_west_coast",
    "tournament_site_state", "is_local_to_site", "is_hometown_game",
    "player_rival_region_tag", "notes",
]

PBP_FEATURES_COLS = [
    "season", "team", "games_with_pbp",
    "avg_scoring_drought_secs", "max_scoring_drought_secs",
    "avg_run_allowed", "max_run_allowed", "avg_run_created", "max_run_created",
    "clutch_off_rating", "clutch_def_rating",
    "final5_turnover_pct", "final5_ft_pct", "final5_rebound_pct",
    "foul_rate_one_possession_games",
    "timeout_response_points_for", "timeout_response_points_against",
    "halftime_adjustment_margin",
    "comeback_win_pct", "blown_lead_pct",
    "star_usage_late", "bench_usage_close_games",
    "single_scorer_dependency_score", "pressure_game_stability_score",
]

SCORING_OUTPUT_COLS = [
    "season", "team", "seed",
    "contender_score", "upset_risk_score", "guard_play_score",
    "experience_score", "defense_travel_score", "rebounding_score",
    "clutch_score", "region_bias_score", "archetype",
    "expected_round", "title_darkhorse_flag", "fraud_favorite_flag",
    "dangerous_low_seed_flag", "explanation_summary",
]

BRACKET_COLS = ["season", "region", "seed", "team"]

ARCHETYPES = [
    "Veteran Guard Control",
    "Elite Defense Traveler",
    "Balanced Powerhouse",
    "Fraud Favorite",
    "High-Variance Freshman Team",
    "Dangerous Low Seed",
]

ROUNDS = {
    1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
    4: "Elite 8", 5: "Final Four", 6: "Championship", 7: "Champion",
}

TEAM_ALIASES = {
    "UConn": "Connecticut", "UCONN": "Connecticut",
    "UNC": "North Carolina", "Ole Miss": "Mississippi",
    "SMU": "Southern Methodist", "LSU": "Louisiana State",
    "USC": "Southern California", "UCF": "Central Florida",
    "UNLV": "UNLV", "VCU": "VCU",
    # Note: team data uses "BYU" as canonical — "Brigham Young" normalized in bracket loader
    "TCU": "TCU", "Pitt": "Pittsburgh",
    "Saint Mary's": "Saint Marys CA", "St. Mary's": "Saint Marys CA",
    "Miami (FL)": "Miami FL", "Miami": "Miami FL",
    "St. John's": "St Johns NY", "Saint John's": "St Johns NY",
    "NC State": "North Carolina State", "N.C. State": "North Carolina State",
}
