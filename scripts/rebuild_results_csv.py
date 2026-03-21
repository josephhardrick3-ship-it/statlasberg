"""Rebuild game_results_2026.csv from ESPN using team_scores_2026.csv canonical names
and seed-based score fallback for teams not in the scores file."""
import math, pandas as pd, requests, os

def safe_f(v, d=0.0):
    try: return float(v)
    except: return d

def win_prob(a, b, k=0.18):
    try: return 1.0 / (1.0 + math.exp(-k * (a - b)))
    except: return 0.5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load both source files ───────────────────────────────────────────────────
scores_df  = pd.read_csv(os.path.join(ROOT, 'data/outputs/team_scores_2026.csv'))
bracket_df = pd.read_csv(os.path.join(ROOT, 'data/outputs/bracket_analysis_2026.csv'))

# Normalize bracket short names → team_scores canonical names
# (mirrors _BRACKET_NORM + _smart_norm in the app)
_BKTSHORT = {
    "NC State":      "North Carolina State",
    "BYU":           "Brigham Young",
    "St. Johns":     "St. John's",
    "Saint Marys CA":"Saint Mary's",
}
score_teams = set(scores_df['team'])

def norm_bracket_name(t):
    if t in score_teams:
        return t
    return _BKTSHORT.get(t, t)

bracket_df['team'] = bracket_df['team'].apply(norm_bracket_name)

# Merge bracket seeds into scores so in_bracket has canonical names + seed
scores_merged = scores_df.merge(
    bracket_df[['team','region','seed']].dropna(subset=['seed']),
    on='team', how='left'
)
in_bracket = scores_merged[scores_merged['seed'].notna()].copy()
in_bracket['seed'] = in_bracket['seed'].astype(int)

flag_lkp = {}
for _, r in in_bracket.iterrows():
    t = str(r['team'])
    flags = []
    if r.get('fraud_favorite_flag'): flags.append('Fraud Fav')
    if r.get('cinderella_flag'):     flags.append('Cinderella')
    if r.get('dangerous_low_seed_flag'): flags.append('Dangerous')
    if r.get('underseeded_flag'):    flags.append('Underseeded')
    flag_lkp[t] = ', '.join(flags)

# Seed → estimated contender score fallback for teams not in scores file at all
SEED_FALLBACK = {
    1: 79, 2: 74, 3: 71, 4: 68, 5: 65,
    6: 63, 7: 61, 8: 58, 9: 56, 10: 54,
    11: 52, 12: 49, 13: 46, 14: 43, 15: 38, 16: 30,
}

# Inject bracket teams not in team_scores (e.g., Miami OH — play-in winners without a scores entry).
# Mirrors the app's injection logic so model picks use seed-fallback for play-in challengers.
_in_bkt_set = set(in_bracket['team'])
_inject = []
for _, _br in bracket_df.iterrows():
    t = norm_bracket_name(str(_br['team']))
    if pd.notna(_br.get('seed')) and t not in _in_bkt_set:
        _inject.append({'team': t, 'region': _br.get('region', ''),
                        'seed': int(_br['seed']),
                        'contender_score': SEED_FALLBACK.get(int(_br['seed']), 50)})
if _inject:
    in_bracket = pd.concat([in_bracket, pd.DataFrame(_inject)], ignore_index=True)

# Score lookup: BRACKET TEAMS ONLY — mirrors app's in_bracket score_lkp so model picks
# use seed-fallback for play-in challengers not in the bracket (e.g., NC State vs Texas seed 11)
score_lkp = dict(zip(in_bracket['team'], in_bracket['contender_score'].apply(lambda x: safe_f(x, 50))))
seed_lkp  = dict(zip(in_bracket['team'], in_bracket['seed']))
bracket_teams = set(in_bracket['team'])
print(f"Loaded {len(in_bracket)} seeded teams | {len(score_lkp)} bracket score entries")

def score_for(team, espn_seed):
    if team in score_lkp:
        return score_lkp[team]
    return SEED_FALLBACK.get(int(espn_seed or 0), 50)

# ── ESPN display name → team_scores canonical name ───────────────────────────
NORM = {
    "TCU Horned Frogs":              "Texas Christian",
    "BYU Cougars":                   "Brigham Young",
    "VCU Rams":                      "Virginia Commonwealth",
    "SMU Mustangs":                  "Southern Methodist",
    "Saint Mary's Gaels":            "Saint Mary's",
    "Saint Mary's (CA)":             "Saint Mary's",
    "Duke Blue Devils":              "Duke",
    "Michigan Wolverines":           "Michigan",
    "Michigan State Spartans":       "Michigan State",
    "Ohio State Buckeyes":           "Ohio State",
    "Nebraska Cornhuskers":          "Nebraska",
    "Arkansas Razorbacks":           "Arkansas",
    "Wisconsin Badgers":             "Wisconsin",
    "Vanderbilt Commodores":         "Vanderbilt",
    "Louisiana Cardinals":           "Louisville",
    "Louisville Cardinals":          "Louisville",
    "North Carolina Tar Heels":      "North Carolina",
    "Texas Longhorns":               "Texas",
    "Texas A&M Aggies":              "Texas A&M",
    "Georgia Bulldogs":              "Georgia",
    "Iowa State Cyclones":           "Iowa State",
    "Iowa Hawkeyes":                 "Iowa",
    "Illinois Fighting Illini":      "Illinois",
    "Houston Cougars":               "Houston",
    "Florida Gators":                "Florida",
    "Arizona Wildcats":              "Arizona",
    "Purdue Boilermakers":           "Purdue",
    "Gonzaga Bulldogs":              "Gonzaga",
    "Kansas Jayhawks":               "Kansas",
    "Tennessee Volunteers":          "Tennessee",
    "Virginia Cavaliers":            "Virginia",
    "Alabama Crimson Tide":          "Alabama",
    "Kentucky Wildcats":             "Kentucky",
    "Texas Tech Red Raiders":        "Texas Tech",
    "Connecticut Huskies":           "Connecticut",
    "UConn Huskies":                 "Connecticut",
    "St. John's Red Storm":          "St. John's",
    "St John's Red Storm":           "St. John's",
    "UCLA Bruins":                   "UCLA",
    "Villanova Wildcats":            "Villanova",
    "Miami Hurricanes":              "Miami",
    "North Carolina State Wolfpack": "North Carolina State",
    "NC State Wolfpack":             "North Carolina State",
    "Siena Saints":                  "Siena",
    "Howard Bison":                  "Howard",
    "North Dakota State Bison":      "North Dakota State",
    "High Point Panthers":           "High Point",
    "McNeese Cowboys":               "McNeese State",
    "Troy Trojans":                  "Troy",
    "Pennsylvania Quakers":          "Pennsylvania",
    "Hawai'i Rainbow Warriors":      "Hawaii",
    "Hawaii Rainbow Warriors":       "Hawaii",
    "Idaho Vandals":                 "Idaho",
    "Kennesaw State Owls":           "Kennesaw State",
    "Saint Louis Billikens":         "Saint Louis",
    "Utah State Aggies":             "Utah State",
    "Santa Clara Broncos":           "Santa Clara",
    "Hofstra Pride":                 "Hofstra",
    "Akron Zips":                    "Akron",
    "Wright State Raiders":          "Wright State",
    "Tennessee State Tigers":        "Tennessee State",
    "Furman Paladins":               "Furman",
    "Long Island University Sharks": "Long Island University",
    "California Baptist Lancers":    "California Baptist",
    "Prairie View A&M Panthers":     "Prairie View",
    "Prairie View A&M":              "Prairie View",
    "Queens Royals":                 "Queens",
    # UCF → Central Florida (canonical name in team_scores)
    "UCF Knights":                   "Central Florida",
    "UCF":                           "Central Florida",
    # Anti-fuzzy-match: explicit so South Florida != Florida, etc.
    "South Florida Bulls":           "South Florida",
    "Central Florida Knights":       "Central Florida",
    "Florida State Seminoles":       "Florida State",
    "Florida Atlantic Owls":         "Florida Atlantic",
    "Miami (OH) RedHawks":           "Miami OH",
    "Miami Ohio RedHawks":           "Miami OH",
    "Miami Ohio":                    "Miami OH",
    "Miami (Ohio)":                  "Miami OH",
    "Miami Redhawks":                "Miami OH",
    "Miami OH Redhawks":             "Miami OH",
    "Northern Iowa Panthers":        "Northern Iowa",
}

def norm(raw):
    n = NORM.get(raw, raw)
    if n in score_lkp:
        return n
    # Fuzzy only for unmapped names
    if raw not in NORM:
        matches = [bt for bt in score_lkp
                   if bt.lower() in raw.lower() or raw.lower() in bt.lower()]
        if matches:
            return max(matches, key=len)
    return n

ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=50'
ESPN_PLAYBYPLAY = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={event_id}'

def fetch_box(eid):
    try:
        r = requests.get(ESPN_PLAYBYPLAY.format(event_id=eid), timeout=8,
                         headers={'User-Agent': 'statlasberg/1.0'})
        r.raise_for_status()
        d = r.json()
        result = {}
        for i, td in enumerate(d.get('boxscore', {}).get('teams', [])[:2]):
            pfx = f't{i+1}_'
            stats = {s.get('name',''): s.get('displayValue','') for s in td.get('statistics', [])}
            for ek, ok in [
                ('fieldGoalPct','fg_pct'),('threePointFieldGoalPct','fg3_pct'),
                ('freeThrowPct','ft_pct'),('totalRebounds','rebounds'),
                ('offensiveRebounds','off_reb'),('turnovers','turnovers'),
                ('assists','assists'),('steals','steals'),('blocks','blocks'),
            ]:
                result[f'{pfx}{ok}'] = stats.get(ek, '')
        comps = d.get('header', {}).get('competitions', [])
        if comps:
            for i, comp in enumerate(comps[0].get('competitors', [])[:2]):
                pfx = f't{i+1}_'
                ls = comp.get('linescores', [])
                result[f'{pfx}h1'] = ls[0].get('displayValue','') if ls else ''
                result[f'{pfx}h2'] = ls[1].get('displayValue','') if len(ls) > 1 else ''
        return result
    except Exception:
        return {}

TOURNEY_DATES = [
    '20260317', '20260318',
    '20260319', '20260320', '20260321', '20260322', '20260323',
    '20260324', '20260325', '20260327', '20260328', '20260329',
    '20260330', '20260404', '20260405', '20260407',
]

from datetime import datetime
today_str = datetime.now().strftime('%Y%m%d')

seen = {}
for d in TOURNEY_DATES:
    if d > today_str:
        break
    resp = requests.get(ESPN_SCOREBOARD + f'&dates={d}', timeout=6,
                        headers={'User-Agent': 'statlasberg/1.0'})
    for event in resp.json().get('events', []):
        eid = event.get('id', '')
        if eid in seen:
            continue
        comp = event.get('competitions', [{}])[0]
        comps = comp.get('competitors', [])
        if len(comps) < 2:
            continue
        completed = event.get('status', {}).get('type', {}).get('completed', False)
        if not completed:
            continue
        hl_raw = comp.get('notes', [{}])[0].get('headline', '')
        if 'NIT' in hl_raw or 'CBI' in hl_raw or 'CIT' in hl_raw:
            continue
        t1c = comps[0]; t2c = comps[1]
        t1n = norm(t1c.get('team', {}).get('displayName', ''))
        t2n = norm(t2c.get('team', {}).get('displayName', ''))
        sc1 = int(t1c.get('score', 0) or 0)
        sc2 = int(t2c.get('score', 0) or 0)
        t1_seed = t1c.get('curatedRank', {}).get('current', 0) or 0
        t2_seed = t2c.get('curatedRank', {}).get('current', 0) or 0
        winner = t1n if sc1 >= sc2 else t2n
        loser  = t2n if sc1 >= sc2 else t1n
        w_seed = t1_seed if sc1 >= sc2 else t2_seed
        l_seed = t2_seed if sc1 >= sc2 else t1_seed
        hl = comp.get('notes', [{}])[0].get('headline', '')
        box = fetch_box(eid)
        seen[eid] = {
            'event_id': eid, 'date': d, 'headline': hl,
            't1': t1n, 't2': t2n, 'winner': winner, 'loser': loser,
            't1_score': sc1, 't2_score': sc2,
            't1_espn_seed': t1_seed, 't2_espn_seed': t2_seed,
            'winner_espn_seed': w_seed, 'loser_espn_seed': l_seed,
            **box,
        }

rows = []
for eid, g in seen.items():
    t1, t2 = g['t1'], g['t2']
    winner, loser = g['winner'], g['loser']
    hl = g.get('headline', '').lower()
    if   'first four'   in hl: rnd = 'FF4'
    elif 'second round' in hl or 'round of 32' in hl: rnd = 'R32'
    elif 'sweet 16'     in hl or 'sweet sixteen' in hl: rnd = 'S16'
    elif 'elite 8'      in hl or 'elite eight'   in hl: rnd = 'E8'
    elif 'final four'   in hl: rnd = 'FF'
    elif 'national championship' in hl or 'championship game' in hl: rnd = 'Championship'
    elif g.get('date','') in ('20260321','20260322','20260323'): rnd = 'R32'
    elif g.get('date','') in ('20260324','20260325'): rnd = 'S16'
    elif g.get('date','') in ('20260327','20260328'): rnd = 'E8'
    elif g.get('date','') in ('20260404','20260405'): rnd = 'FF'
    elif g.get('date','') == '20260407': rnd = 'Championship'
    else: rnd = 'R64'

    s1 = score_for(t1, g.get('t1_espn_seed', 0))
    s2 = score_for(t2, g.get('t2_espn_seed', 0))
    mw = t1 if win_prob(s1, s2) >= 0.5 else t2
    ml = t2 if mw == t1 else t1
    mw_s = g.get('t1_espn_seed', 0) if mw == t1 else g.get('t2_espn_seed', 0)
    ml_s = g.get('t2_espn_seed', 0) if mw == t1 else g.get('t1_espn_seed', 0)
    mc = win_prob(score_for(mw, mw_s), score_for(ml, ml_s))
    correct = (winner == mw)

    ws = seed_lkp.get(winner, g.get('winner_espn_seed', 0))
    ls_ = seed_lkp.get(loser, g.get('loser_espn_seed', 0))
    upset = (ws > ls_ + 3) if ws and ls_ else False

    rows.append({
        'event_id': eid, 'date': g['date'], 'round': rnd, 'region': '',
        't1': t1, 't2': t2, 'winner': winner, 'loser': loser,
        't1_score': g['t1_score'], 't2_score': g['t2_score'],
        'model_pick': mw, 'model_conf': round(mc, 3),
        'correct': correct, 'upset': upset,
        'winner_seed': ws, 'loser_seed': ls_,
        'winner_flags': flag_lkp.get(winner, ''), 'loser_flags': flag_lkp.get(loser, ''),
        'winner_hot': '', 'loser_hot': '', 'narrative': '',
        **{k: g.get(k, '') for k in [
            't1_fg_pct','t1_fg3_pct','t1_ft_pct','t1_rebounds','t1_off_reb',
            't1_turnovers','t1_assists','t1_steals','t1_blocks','t1_h1','t1_h2',
            't2_fg_pct','t2_fg3_pct','t2_ft_pct','t2_rebounds','t2_off_reb',
            't2_turnovers','t2_assists','t2_steals','t2_blocks','t2_h1','t2_h2',
        ]}
    })

df = pd.DataFrame(rows)
correct_n = int(df['correct'].sum())
total_n = len(df)
print(f"\nRecord: {correct_n}-{total_n - correct_n} ({total_n} games)\n")
for _, r in df.iterrows():
    st = '✅' if r['correct'] else '❌'
    print(f"  {st} {r['winner']:30} ({score_for(r['winner'], r.get('winner_seed',0)):.1f}) def "
          f"{r['loser']:25} ({score_for(r['loser'], r.get('loser_seed',0)):.1f}) "
          f"| pick: {r['model_pick']:30} @ {r['model_conf']:.0%}")

out = os.path.join(ROOT, 'data/outputs/game_results_2026.csv')
df.to_csv(out, index=False)
print(f"\nSaved {len(df)} rows to {out}")
