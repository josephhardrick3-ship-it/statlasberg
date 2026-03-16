#!/usr/bin/env python3
"""
One-shot update script for March 15, 2026 (day before Selection Sunday).
Applies:
  1. Fresh NET rankings (March 15, fetched from NCAA.com)
  2. Fresh ESPN team records (conference tournament games now included)
  3. Merges both into team_stats_2026.csv
"""

import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

BOT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMS_DIR = os.path.join(BOT_ROOT, "data", "raw", "teams")
sys.path.insert(0, BOT_ROOT)

# ─────────────────────────────────────────────────────────────
# NET RANKINGS — March 15, 2026 (from NCAA.com)
# ─────────────────────────────────────────────────────────────

NET_MARCH15 = """1,Duke
2,Michigan
3,Arizona
4,Florida
5,Houston
6,Gonzaga
7,Iowa St.
8,Illinois
9,Purdue
10,UConn
11,Michigan St.
12,Virginia
13,Vanderbilt
14,Nebraska
15,St. John's (NY)
16,Louisville
17,Arkansas
18,Alabama
19,Texas Tech
20,Tennessee
21,Kansas
22,Saint Mary's (CA)
23,BYU
24,North Carolina
25,Wisconsin
26,Utah St.
27,Iowa
28,Kentucky
29,Ohio St.
30,Saint Louis
31,UCLA
32,Miami (FL)
33,Georgia
34,Clemson
35,Villanova
36,NC State
37,SMU
38,Auburn
39,TCU
40,Santa Clara
41,Indiana
42,Texas
43,VCU
44,Texas A&M
45,South Fla.
46,New Mexico
47,San Diego St.
48,Oklahoma
49,Cincinnati
50,Baylor
51,UCF
52,Tulsa
53,Seton Hall
54,Akron
55,Virginia Tech
56,McNeese
57,Washington
58,Missouri
59,West Virginia
60,Boise St.
61,Stanford
62,Florida St.
63,Belmont
64,Miami (OH)
65,Yale
66,Northwestern
67,Wake Forest
68,California
69,Dayton
70,Nevada
71,Grand Canyon
72,UNI
73,Arizona St.
74,Wichita St.
75,Oklahoma St.
76,High Point
77,Colorado
78,LSU
79,Southern California
80,Minnesota
81,Providence
82,Ole Miss
83,Creighton
84,Butler
85,Utah Valley
86,Syracuse
87,Colorado St.
88,Hofstra
89,SFA
90,Georgetown
91,George Washington
92,George Mason
93,Marquette
94,Notre Dame
95,Illinois St.
96,UNCW
97,Xavier
98,California Baptist
99,Wyoming
100,Kansas St.
101,Hawaii
102,DePaul
103,St. Thomas (MN)
104,Pittsburgh
105,Murray St.
106,Liberty
107,Sam Houston
108,South Carolina
109,Oregon
110,UIC
111,UNLV
112,Mississippi St.
113,Bradley
114,North Dakota St.
115,Davidson
116,Pacific
117,UC Irvine
118,UAB
119,San Francisco
120,Southern Ill.
121,Saint Joseph's
122,Seattle U
123,Fla. Atlantic
124,UC San Diego
125,Troy
126,UTRGV
127,Wright St.
128,Montana St.
129,William & Mary
130,Toledo
131,Winthrop
132,Utah
133,Duquesne
134,Memphis
135,Rutgers
136,Navy
137,North Texas
138,Maryland
139,ETSU
140,Kent St.
141,Penn St.
142,UC Santa Barbara
143,St. Bonaventure
144,Idaho
145,Washington St.
146,Portland St.
147,Robert Morris
148,Penn
149,Cornell
150,Northern Colo.
151,Rhode Island
152,Harvard
153,Fresno St.
154,Arkansas St.
155,Kennesaw St.
156,Valparaiso
157,Bowling Green
158,Boston College
159,Towson
160,Western Ky.
161,UT Arlington
162,Col. of Charleston
163,Middle Tenn.
164,Austin Peay
165,Richmond
166,Central Ark.
167,Temple
168,Monmouth
169,Georgia Tech
170,Oregon St.
171,Marshall
172,Tennessee St.
173,Charlotte
174,Fordham
175,Oakland
176,Utah Tech
177,Cal St. Fullerton
178,UC Davis
179,LMU (CA)
180,Merrimack
181,Marist
182,Mercer
183,Siena
184,Northern Ky.
185,Columbia
186,Furman
187,South Alabama
188,CSUN
189,Queens (NC)
190,New Mexico St.
191,A&M-Corpus Christi
192,App State
193,Lipscomb
194,Montana
195,Buffalo
196,UMBC
197,FIU
198,LIU
199,Drake
200,Campbell
201,Massachusetts
202,Weber St.
203,Howard
204,Eastern Wash.
205,UT Martin
206,Missouri St.
207,Tulane
208,James Madison
209,Louisiana Tech
210,South Dakota St.
211,Quinnipiac
212,Drexel
213,Indiana St.
214,Southern Miss.
215,New Orleans
216,Jacksonville St.
217,Green Bay
218,Samford
219,Youngstown St.
220,Ga. Southern
221,Detroit Mercy
222,Portland
223,Elon
224,Southeast Mo. St.
225,Cal Poly
226,Colgate
227,Vermont
228,Western Caro.
229,Coastal Carolina
230,Tarleton St.
231,Ohio
232,Stony Brook
233,Saint Peter's
234,San Diego
235,Texas St.
236,Idaho St.
237,Iona
238,Eastern Mich.
239,UNC Asheville
240,Fairfield
241,Denver
242,Charleston So.
243,Rice
244,San Jose St.
245,Wofford
246,La Salle
247,FGCU
248,Abilene Christian
249,Princeton
250,Nicholls
251,Lamar University
252,Radford
253,Milwaukee
254,Purdue Fort Wayne
255,Old Dominion
256,American
257,Long Beach St.
258,Omaha
259,SIUE
260,Presbyterian
261,Bethune-Cookman
262,Lindenwood
263,Boston U.
264,Morehead St.
265,Dartmouth
266,Longwood
267,East Carolina
268,UTEP
269,Sacramento St.
270,Hampton
271,North Dakota
272,Southern Utah
273,UC Riverside
274,Pepperdine
275,Lehigh
276,N.C. A&T
277,Mercyhurst
278,Loyola Chicago
279,Southern U.
280,Central Mich.
281,Mount St. Mary's
282,UIW
283,Western Mich.
284,South Dakota
285,Bellarmine
286,Northeastern
287,Delaware
288,Chattanooga
289,Central Conn. St.
290,Le Moyne
291,Eastern Ky.
292,UNC Greensboro
293,Sacred Heart
294,Houston Christian
295,Southeastern La.
296,Brown
297,Jacksonville
298,West Ga.
299,Ball St.
300,Prairie View
301,Northwestern St.
302,Wagner
303,USC Upstate
304,East Texas A&M
305,Grambling
306,Evansville
307,Georgia St.
308,Tennessee Tech
309,Little Rock
310,Alabama A&M
311,Oral Roberts
312,Florida A&M
313,Louisiana
314,Stetson
315,Northern Ariz.
316,Ark.-Pine Bluff
317,Cleveland St.
318,Norfolk St.
319,IU Indy
320,Eastern Ill.
321,UMass Lowell
322,Texas Southern
323,Lafayette
324,Alabama St.
325,New Haven
326,UAlbany
327,NJIT
328,Loyola Maryland"""

# ─────────────────────────────────────────────────────────────
# Name normalization aliases  NCAA → Sports-Reference
# ─────────────────────────────────────────────────────────────
NCAA_TO_SR = {
    "Iowa St.":          "Iowa State",
    "UConn":             "Connecticut",
    "Michigan St.":      "Michigan State",
    "Saint Mary's (CA)": "Saint Mary's",
    "St. John's (NY)":   "St. John's",
    "Miami (FL)":        "Miami FL",
    "Miami (OH)":        "Miami OH",
    "Ohio St.":          "Ohio State",
    "Penn St.":          "Penn State",
    "Utah St.":          "Utah State",
    "Colorado St.":      "Colorado State",
    "Arizona St.":       "Arizona State",
    "BYU":               "Brigham Young",
    "SMU":               "Southern Methodist",
    "TCU":               "Texas Christian",
    "LSU":               "Louisiana State",
    "UCF":               "Central Florida",
    "South Fla.":        "South Florida",
    "VCU":               "Virginia Commonwealth",
    "UNI":               "Northern Iowa",
    "Ole Miss":          "Mississippi",
    "SE Missouri St.":   "Southeast Missouri State",
    "Southeast Mo. St.": "Southeast Missouri State",
    "ETSU":              "East Tennessee State",
    "Loyola Chicago":    "Loyola-Chicago",
    "McNeese":           "McNeese State",
    "San Diego St.":     "San Diego State",
    "Oklahoma St.":      "Oklahoma State",
    "Wichita St.":       "Wichita State",
    "Arkansas St.":      "Arkansas State",
    "Mississippi St.":   "Mississippi State",
    "Texas A&M":         "Texas A&M",
    "George Washington": "George Washington",
    "San Francisco":     "San Francisco",
    "Murray St.":        "Murray State",
    "Kansas St.":        "Kansas State",
    "Ball St.":          "Ball State",
    "Fresno St.":        "Fresno State",
    "New Mexico St.":    "New Mexico State",
    "Georgia St.":       "Georgia State",
    "Jacksonville St.":  "Jacksonville State",
    "Morehead St.":      "Morehead State",
    "Tennessee Tech":    "Tennessee Tech",
    "SFA":               "Stephen F. Austin",
    "UNLV":              "Nevada-Las Vegas",
    "Col. of Charleston":"College of Charleston",
    "SIU-Edwardsville":  "SIU Edwardsville",
    "SIUE":              "SIU Edwardsville",
    "Penn":              "Pennsylvania",
    "Boise St.":         "Boise State",
    "Oregon St.":        "Oregon State",
    "NC State":          "North Carolina State",
    "N.C. State":        "North Carolina State",
    "Fla. Atlantic":     "Florida Atlantic",
    "LMU (CA)":          "Loyola Marymount",
    "ETSU":              "East Tennessee State",
    "Sam Houston":       "Sam Houston",
    "St. Thomas (MN)":   "St. Thomas",
    "Northern Colo.":    "Northern Colorado",
    "Western Ky.":       "Western Kentucky",
    "UT Arlington":      "UT Arlington",
    "Ga. Southern":      "Georgia Southern",
    "Southeast Mo. St.": "Southeast Missouri State",
    "Eastern Wash.":     "Eastern Washington",
    "UT Martin":         "UT Martin",
    "South Dakota St.":  "South Dakota State",
    "North Dakota St.":  "North Dakota State",
    "Southern Ill.":     "Southern Illinois",
    "Southern Miss.":    "Southern Mississippi",
    "Central Conn. St.": "Central Connecticut State",
    "Western Caro.":     "Western Carolina",
    "Charleston So.":    "Charleston Southern",
    "Purdue Fort Wayne": "Purdue Fort Wayne",
    "A&M-Corpus Christi":"Texas A&M Corpus Christi",
    "App State":         "Appalachian State",
    "East Texas A&M":    "East Texas A&M",
    "N.C. A&T":          "North Carolina A&T",
    "Northern Ariz.":    "Northern Arizona",
    "Ark.-Pine Bluff":   "Arkansas Pine Bluff",
    "UMass Lowell":      "Massachusetts Lowell",
    "Southeastern La.":  "Southeastern Louisiana",
    "Houston Christian": "Houston Christian",
    "Northwest St.":     "Northwestern State",
    "Northwestern St.":  "Northwestern State",
    "Lamar University":  "Lamar",
    "Eastern Mich.":     "Eastern Michigan",
    "Central Mich.":     "Central Michigan",
    "Western Mich.":     "Western Michigan",
    "Mount St. Mary's":  "Mount St. Mary's",
    "Southern Utah":     "Southern Utah",
    "UC Santa Barbara":  "UC Santa Barbara",
    "UC San Diego":      "UC San Diego",
    "UC Riverside":      "UC Riverside",
    "Long Beach St.":    "Long Beach State",
    "Sacramento St.":    "Sacramento State",
    "Montana St.":       "Montana State",
    "UNCW":              "North Carolina Wilmington",
    "UNC Asheville":     "North Carolina Asheville",
    "UNC Greensboro":    "North Carolina Greensboro",
    "Queens (NC)":       "Queens",
    "UTRGV":             "Texas Rio Grande Valley",
    "UIW":               "Incarnate Word",
    "FGCU":              "Florida Gulf Coast",
    "CSUN":              "Cal State Northridge",
    "Cal Poly":          "Cal Poly",
    "Cal St. Fullerton": "Cal State Fullerton",
    "UC Davis":          "UC Davis",
    "UIC":               "Illinois Chicago",
    "Tennessee St.":     "Tennessee State",
    "Tennessee Tech":    "Tennessee Tech",
    "Kent St.":          "Kent State",
    "Idaho St.":         "Idaho State",
    "Weber St.":         "Weber State",
}


def parse_net(raw: str):
    rows = []
    for line in raw.strip().split('\n'):
        parts = line.split(',', 1)
        if len(parts) == 2:
            rank, name = int(parts[0].strip()), parts[1].strip()
            rows.append({"net_rank": rank, "ncaa_name": name,
                         "norm_name": NCAA_TO_SR.get(name, name)})
    return pd.DataFrame(rows)


def merge_net(sr: pd.DataFrame, net_df: pd.DataFrame) -> pd.DataFrame:
    sr = sr.copy()
    sr["net_rank"] = np.nan
    sr_names = sr["team"].tolist()
    exact = fuzzy_n = skip = 0

    for _, row in net_df.iterrows():
        norm = row["norm_name"]
        rank = row["net_rank"]

        if norm in sr_names:
            sr.loc[sr["team"] == norm, "net_rank"] = rank
            exact += 1
            continue

        hit = process.extractOne(norm, sr_names, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= 82:
            sr.loc[sr["team"] == hit[0], "net_rank"] = rank
            fuzzy_n += 1
        else:
            skip += 1

    # Manual overrides for tricky duplicates
    # Miami FL = ACC (conf 13), Miami OH = MAC (conf 18)
    miami_fl_net = net_df.loc[net_df["ncaa_name"] == "Miami (FL)", "net_rank"]
    miami_oh_net = net_df.loc[net_df["ncaa_name"] == "Miami (OH)", "net_rank"]
    if not miami_fl_net.empty:
        sr.loc[(sr["team"] == "Miami") & (sr["conference"] == 13), "net_rank"] = float(miami_fl_net.iloc[0])
    if not miami_oh_net.empty:
        sr.loc[(sr["team"] == "Miami") & (sr["conference"] == 18), "net_rank"] = float(miami_oh_net.iloc[0])

    print(f"  NET merge: exact={exact}  fuzzy={fuzzy_n}  skipped={skip}  "
          f"total_assigned={sr['net_rank'].notna().sum()}")
    return sr


# ─────────────────────────────────────────────────────────────
# ESPN records merge
# ─────────────────────────────────────────────────────────────

def merge_espn_records(sr: pd.DataFrame) -> pd.DataFrame:
    rec_path = os.path.join(TEAMS_DIR, "espn_team_records_2026.csv")
    if not os.path.exists(rec_path):
        print("  ESPN records file not found — skipping")
        return sr

    rec = pd.read_csv(rec_path)
    # Force numeric
    for col in ["close_game_record", "last10_win_pct", "last10_adj_margin"]:
        if col in rec.columns:
            rec[col] = pd.to_numeric(rec[col], errors="coerce")

    # Use fetch_espn_games normalizer to match ESPN names → SR names
    sys.path.insert(0, BOT_ROOT)
    from scripts.fetch_espn_games import normalize_espn, set_sr_teams
    set_sr_teams(sr["team"].tolist())

    rec["sr_name"] = rec["team"].apply(normalize_espn)

    merged = sr.copy()
    for col in ["close_game_record", "last10_win_pct", "last10_adj_margin"]:
        if col in rec.columns:
            merged[col] = np.nan

    matched = total = 0
    for _, r in rec.iterrows():
        sr_name = r["sr_name"]
        mask = merged["team"] == sr_name
        if mask.any():
            for col in ["close_game_record", "last10_win_pct", "last10_adj_margin"]:
                if col in r.index:
                    merged.loc[mask, col] = r[col]
            matched += 1
        total += 1

    print(f"  ESPN records merged: {matched}/{total} teams matched")
    return merged


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr = pd.read_csv(sr_path)
    print(f"Loaded team_stats_2026.csv: {len(sr)} teams")

    # 1. NET rankings
    print("\n[1] Merging NET rankings (March 15) ...")
    net_df = parse_net(NET_MARCH15)
    sr = merge_net(sr, net_df)

    # 2. ESPN records
    print("\n[2] Merging ESPN records (through March 15) ...")
    sr = merge_espn_records(sr)

    # Show key changes vs expected
    print("\n[3] Top-30 teams by NET rank:")
    top = sr[sr["net_rank"].notna()].sort_values("net_rank").head(30)
    print(f"  {'NET':<5} {'Team':<28} {'AdjMargin':>10} {'Barthag':>8} "
          f"{'Last10':>7} {'ClosePct':>9}")
    print(f"  {'-'*72}")
    for _, row in top.iterrows():
        l10  = f"{row.get('last10_win_pct', float('nan')):.0%}" if not pd.isna(row.get('last10_win_pct', float('nan'))) else " N/A"
        cpct = f"{row.get('close_game_record', float('nan')):.0%}" if not pd.isna(row.get('close_game_record', float('nan'))) else " N/A"
        print(f"  #{int(row['net_rank']):<4} {row['team']:<28} {row.get('adj_margin', float('nan')):>10.1f} "
              f"{row.get('barthag', float('nan')):>8.4f} {l10:>7} {cpct:>9}")

    sr.to_csv(sr_path, index=False)
    print(f"\nSaved → {sr_path}")


if __name__ == "__main__":
    main()
