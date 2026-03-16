"""Templates for LLM-generated explanations. Claude explains, never invents."""

TEAM_SUMMARY_TEMPLATE = """You are a March Madness analyst. Based ONLY on the data below, write a 3-4 sentence
tournament profile for {team}. Do NOT invent any statistics. Only reference numbers from the data.

Team: {team}
Contender Score: {contender_score}/100
Archetype: {archetype}
Expected Round: {expected_round}
Experience Score: {experience_score}
Defense Score: {defense_score}
Guard Play Score: {guard_play_score}
Clutch Score: {clutch_score}
Rebounding Score: {rebounding_score}
NET Rank: {net_rank}
Record: {wins}-{losses}
Flags: darkhorse={title_darkhorse_flag}, fraud={fraud_favorite_flag}, dangerous={dangerous_low_seed_flag}
"""

MATCHUP_TEMPLATE = """Based ONLY on the data below, write a 2-3 sentence matchup preview.

{team_a} vs {team_b}
Matchup Score (>0.5 favors {team_a}): {matchup_score}
Turnover Edge: {turnover_edge}
Rebound Edge: {rebound_edge}
Guard Experience Edge: {guard_experience_edge}
Defense Edge: {defense_edge}
"""

BRACKET_TEMPLATE = """Based ONLY on the simulation data below, write a 2-3 sentence bracket prediction summary.

Top 5 championship probabilities:
{top5_champs}

Most common Final Four teams:
{final_four_teams}
"""
