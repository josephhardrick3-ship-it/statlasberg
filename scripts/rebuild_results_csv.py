"""Rebuild game_results_2026.csv from ESPN using correct team name normalization
and seed-based score fallback for teams not in bracket_analysis_2026.csv."""
import math, pandas as pd, requests, os

def safe_f(v, d=0.0):
    try: return float(v)
    except: return d

def win_prob(a, b, k=0.18):
    try: return 1.0 / (1.0 + math.exp(-k * (a - b)))
    except: return 0.5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bkt_df = pd.read_csv(os.path.join(ROOT, 'data/outputs/bracket_analysis_2026.csv'))
bracket_teams = set(bkt_df['team'].tolist())
score_lkp = {}; seed_lkp = {}; flag_lkp = {}
for _, r in bkt_df.iterrows():
    t = str(r['team'])
    score_lkp[t] = safe_f(r.get('contender_score', 50), 50)
    seed_lkp[t] = int(r['seed']) if pd.notna(r.get('seed')) else 0
    flags = []
    if r.get('fraud_favorite_flag'): flags.append('Fraud Fav')
    if r.get('cinderella_flag'): flags.append('Cinderella')
    if r.get('dangerous_low_seed_flag'): flags.append('Dangerous')
    if r.get('underseeded_flag'): flags.append('Underseeded')
    flag_lkp[t] = ', '.join(flags)

# Seed -> estimated contender score for teams NOT in bracket_analysis_2026.csv
SEED_FALLBACK = {
    1: 79, 2: 74, 3: 71, 4: 68, 5: 65,
    6: 63, 7: 61, 8: 58, 9: 56, 10: 54,
    11: 52, 12: 49, 13: 46, 14: 43, 15: 38, 16: 30,
}

def score_for(team, espn_seed):
    if team in score_lkp:
        return score_lkp[team]
    return SEED_FALLBACK.get(int(espn_seed or 0), 50)

# Maps ESPN display names -> exact bracket_analysis_2026.csv team names
NORM = {
    "Duke Blue Devils": "Duke",
    "Michigan Wolverines": "Michigan",
    "Houston Cougars": "Houston",
    "Michigan State Spartans": "Michigan State",
    "Illinois Fighting Illini": "Illinois",
    "Gonzaga Bulldogs": "Gonzaga",
    "Nebraska Cornhuskers": "Nebraska",
    "Arkansas Razorbacks": "Arkansas",
    "High Point Panthers": "High Point",
    "Vanderbilt Commodores": "Vanderbilt",
    "McNeese Cowboys": "McNeese State",
    "Louisville Cardinals": "Louisville",
    "VCU Rams": "VCU",
    "North Carolina Tar Heels": "North Carolina",
    "Texas Longhorns": "Texas",
    "Texas A&M Aggies": "Texas A&M",
    "TCU Horned Frogs": "TCU",
    "Ohio State Buckeyes": "Ohio State",
    "Georgia Bulldogs": "Georgia",
    "Saint Louis Billikens": "Saint Louis",
    "Arizona Wildcats": "Arizona",
    "Florida Gators": "Florida",
    "Iowa State Cyclones": "Iowa State",
    "Purdue Boilermakers": "Purdue",
    "Connecticut Huskies": "Connecticut",
    "UConn Huskies": "Connecticut",
    "Virginia Cavaliers": "Virginia",
    "Alabama Crimson Tide": "Alabama",
    "Kansas Jayhawks": "Kansas",
    "Texas Tech Red Raiders": "Texas Tech",
    "St. John's Red Storm": "St. Johns",
    "St John's Red Storm": "St. Johns",
    "Tennessee Volunteers": "Tennessee",
    "Kentucky Wildcats": "Kentucky",
    "UCLA Bruins": "UCLA",
    "Missouri Tigers": "Missouri",
    "Wisconsin Badgers": "Wisconsin",
    "Iowa Hawkeyes": "Iowa",
    "BYU Cougars": "BYU",
    "Villanova Wildcats": "Villanova",
    "Miami Hurricanes": "Miami",
    "Clemson Tigers": "Clemson",
    "Utah State Aggies": "Utah State",
    "Akron Zips": "Akron",
    "Saint Mary's Gaels": "Saint Marys CA",
    "Idaho Vandals": "Idaho",
    "North Dakota State Bison": "North Dakota State",
    "Pennsylvania Quakers": "Pennsylvania",
    "Kennesaw State Owls": "Kennesaw State",
    "Troy Trojans": "Troy",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "North Carolina State Wolfpack": "NC State",
    "Siena Saints": "Siena",
    "Howard Bison": "Howard",
    "Hofstra Pride": "Hofstra",
    "Wright State Raiders": "Wright State",
    "Furman Paladins": "Furman",
    "Long Island University Sharks": "Long Island University",
    "California Baptist Lancers": "California Baptist",
    "Prairie View A&M Panthers": "Prairie View A&M",
    "Queens Royals": "Queens",
    "Santa Clara Broncos": "Santa Clara",
    # Explicit anti-fuzzy-match mappings
    "South Florida Bulls": "South Florida",
    "Miami (OH) RedHawks": "Miami OH",
    "Northern Iowa Panthers": "Northern Iowa",
    "UCF Knights": "UCF",
    "Tennessee State Tigers": "Tennessee State",
}

def norm(raw):
    n = NORM.get(raw, raw)
    if n in bracket_teams:
        return n
    # Fuzzy fallback — but only if the raw name isn't already explicitly handled above
    if raw in NORM:
        return n  # already mapped; don't fuzzy further
    matches = [bt for bt in bracket_teams
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

seen = {}
for d in ['20260319', '20260320']:
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
        t1c = comps[0]; t2c = comps[1]
        t1n = norm(t1c.get('team', {}).get('displayName', ''))
        t2n = norm(t2c.get('team', {}).get('displayName', ''))
        sc1 = int(t1c.get('score', 0) or 0)
        sc2 = int(t2c.get('score', 0) or 0)
        t1_seed = t1c.get('curatedRank', {}).get('current', 0) or 0
        t2_seed = t2c.get('curatedRank', {}).get('current', 0) or 0
        winner = t1n if sc1 >= sc2 else t2n
        loser  = t2n if sc1 >= sc2 else t1n
        winner_seed = t1_seed if sc1 >= sc2 else t2_seed
        loser_seed  = t2_seed if sc1 >= sc2 else t1_seed
        hl = comp.get('notes', [{}])[0].get('headline', '')
        box = fetch_box(eid)
        seen[eid] = {
            'event_id': eid, 'date': d, 'headline': hl,
            't1': t1n, 't2': t2n, 'winner': winner, 'loser': loser,
            't1_score': sc1, 't2_score': sc2,
            't1_espn_seed': t1_seed, 't2_espn_seed': t2_seed,
            'winner_espn_seed': winner_seed, 'loser_espn_seed': loser_seed,
            **box,
        }

rows = []
for eid, g in seen.items():
    t1, t2 = g['t1'], g['t2']
    winner, loser = g['winner'], g['loser']
    hl = g.get('headline', '').lower()
    rnd = 'FF4' if 'first four' in hl else 'R64'

    s1 = score_for(t1, g.get('t1_espn_seed', 0))
    s2 = score_for(t2, g.get('t2_espn_seed', 0))
    mw = t1 if win_prob(s1, s2) >= 0.5 else t2
    ml = t2 if mw == t1 else t1
    mw_seed = g.get('t1_espn_seed', 0) if mw == t1 else g.get('t2_espn_seed', 0)
    ml_seed = g.get('t2_espn_seed', 0) if mw == t1 else g.get('t1_espn_seed', 0)
    mc = win_prob(score_for(mw, mw_seed), score_for(ml, ml_seed))
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
print(f"Record: {correct_n}-{total_n - correct_n} ({total_n} games)\n")
for _, r in df.iterrows():
    st = '✅' if r['correct'] else '❌'
    print(f"  {st} {r['winner']:30} def {r['loser']:30} | model: {r['model_pick']:30} @ {r['model_conf']:.0%}")

out = os.path.join(ROOT, 'data/outputs/game_results_2026.csv')
df.to_csv(out, index=False)
print(f"\nSaved {len(df)} rows to {out}")
