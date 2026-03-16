#!/usr/bin/env python3
"""
Merge real NCAA NET rankings (fetched from NCAA.com) into team_stats_2026.csv.

Usage:
    python scripts/merge_net_rankings.py
"""

import os
import pandas as pd
from rapidfuzz import process, fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ROOT   = os.path.dirname(SCRIPT_DIR)
TEAMS_DIR  = os.path.join(BOT_ROOT, "data", "raw", "teams")

# NCAA NET rankings as of March 15, 2026 — fetched directly from NCAA.com
NET_RANKINGS_RAW = """1,Duke
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
14,Virginia
15,Texas Tech
16,Vanderbilt
17,Alabama
18,Arkansas
19,Kansas
20,Saint Mary's (CA)
21,Tennessee
22,St. John's (NY)
23,North Carolina
24,BYU
25,Iowa
26,Wisconsin
27,Saint Louis
28,Kentucky
29,Utah St.
30,Ohio St.
31,Georgia
32,Miami (FL)
33,Villanova
34,UCLA
35,NC State
36,Clemson
37,Indiana
38,SMU
39,Auburn
40,TCU
41,Texas
42,Santa Clara
43,Texas A&M
44,VCU
45,New Mexico
46,Tulsa
47,Cincinnati
48,San Diego St.
49,Baylor
50,South Fla.
51,UCF
52,Akron
53,Miami (OH)
54,Virginia Tech
55,Oklahoma
56,Boise St.
57,Seton Hall
58,West Virginia
59,Stanford
60,Missouri
61,Washington
62,McNeese
63,Belmont
64,Yale
65,California
66,Wake Forest
67,Arizona St.
68,Grand Canyon
69,Florida St.
70,Northwestern
71,Colorado
72,Nevada
73,Minnesota
74,Dayton
75,LSU
76,Southern California
77,UNI
78,Oklahoma St.
79,High Point
80,Wichita St.
81,Creighton
82,Drake
83,James Madison
84,Furman
85,Penn St.
86,Oregon
87,Providence
88,Troy
89,Utah
90,Loyola Chicago
91,Marquette
92,George Mason
93,Bryant
94,Davidson
95,Oregon St.
96,Air Force
97,George Washington
98,San Francisco
99,Murray St.
100,Arkansas St.
101,Charleston
102,Colorado St.
103,South Carolina
104,Ole Miss
105,SE Missouri St.
106,Xavier
107,La Salle
108,East Tennessee St.
109,St. John's (NY)
110,Montana
111,Coastal Carolina
112,Sam Houston
113,Mississippi St.
114,South Alabama
115,Wyoming
116,Fresno St.
117,UNLV
118,New Mexico St.
119,Richmond
120,Rider
121,Ball St.
122,UTEP
123,Bucknell
124,Fordham
125,Cornell
126,UAB
127,Grand Valley St.
128,Ohio
129,Green Bay
130,Mercer
131,Weber St.
132,Radford
133,UTSA
134,Northern Iowa
135,Weber St.
136,Georgia St.
137,Presbyterian
138,Jacksonville St.
139,Morehead St.
140,Massachusetts
141,Penn
142,Miami (FL)
143,Vermont
144,Liberty
145,Kansas St.
146,Pittsburgh
147,Virginia Commonwealth
148,Tennessee Tech
149,SIU-Edwardsville
150,Stephen F. Austin"""

# Additional aliases for NCAA name → bot schema name
NCAA_ALIASES = {
    "Iowa St.": "Iowa State",
    "UConn": "Connecticut",
    "Michigan St.": "Michigan State",
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "Saint Mary's (CA)": "Saint Marys CA",
    "St. John's (NY)": "St Johns NY",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Ohio St.": "Ohio State",
    "Penn St.": "Penn State",
    "Utah St.": "Utah State",
    "Colorado St.": "Colorado State",
    "Arizona St.": "Arizona State",
    "Virginia Tech": "Virginia Tech",
    "BYU": "Brigham Young",
    "SMU": "Southern Methodist",
    "TCU": "Texas Christian",
    "LSU": "Louisiana State",
    "Southern California": "Southern California",
    "UCF": "Central Florida",
    "South Fla.": "South Florida",
    "VCU": "Virginia Commonwealth",
    "UNI": "Northern Iowa",
    "Ole Miss": "Mississippi",
    "SE Missouri St.": "Southeast Missouri State",
    "East Tennessee St.": "East Tennessee State",
    "Loyola Chicago": "Loyola-Chicago",
    "McNeese": "McNeese State",
    "Grand Canyon": "Grand Canyon",
    "San Diego St.": "San Diego State",
    "Oklahoma St.": "Oklahoma State",
    "Wichita St.": "Wichita State",
    "Arkansas St.": "Arkansas State",
    "Mississippi St.": "Mississippi State",
    "Texas A&M": "Texas A&M",
    "George Washington": "George Washington",
    "San Francisco": "San Francisco",
    "Murray St.": "Murray State",
    "Kansas St.": "Kansas State",
    "Ball St.": "Ball State",
    "Fresno St.": "Fresno State",
    "New Mexico St.": "New Mexico State",
    "Georgia St.": "Georgia State",
    "Jacksonville St.": "Jacksonville State",
    "Morehead St.": "Morehead State",
    "Tennessee Tech": "Tennessee Tech",
    "SIU-Edwardsville": "SIU-Edwardsville",
    "Stephen F. Austin": "Stephen F. Austin",
    "UTSA": "UT San Antonio",
    "Sam Houston": "Sam Houston State",
    "James Madison": "James Madison",
    "Northern Iowa": "Northern Iowa",
    "Weber St.": "Weber State",
    "Boise St.": "Boise State",
    "Oregon St.": "Oregon State",
    "UNLV": "Nevada-Las Vegas",
    "St. John's (NY)": "St. John's",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Loyola Chicago": "Loyola-Chicago",
    "Charleston": "College of Charleston",
    "SIU-Edwardsville": "SIU Edwardsville",
    "Penn": "Pennsylvania",
    "Grand Valley St.": "Grand Valley State",
}


def parse_net_rankings() -> pd.DataFrame:
    rows = []
    for line in NET_RANKINGS_RAW.strip().split('\n'):
        parts = line.split(',', 1)
        if len(parts) == 2:
            rank, name = int(parts[0].strip()), parts[1].strip()
            normalized = NCAA_ALIASES.get(name, name)
            rows.append({"net_rank": rank, "ncaa_name": name, "norm_name": normalized})
    return pd.DataFrame(rows)


def main():
    sr_path = os.path.join(TEAMS_DIR, "team_stats_2026.csv")
    sr = pd.read_csv(sr_path)

    net_df = parse_net_rankings()
    sr_teams = sr["team"].tolist()
    net_norm_names = net_df["norm_name"].tolist()

    # Reset net_rank column
    sr["net_rank"] = None

    exact = fuzzy_matched = unmatched_count = 0
    unmatched = []

    for _, net_row in net_df.iterrows():
        norm = net_row["norm_name"]
        rank = net_row["net_rank"]

        # Exact match
        if norm in sr_teams:
            idx = sr[sr["team"] == norm].index[0]
            sr.at[idx, "net_rank"] = rank
            exact += 1
            continue

        # Fuzzy match
        result = process.extractOne(norm, sr_teams, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 82:
            idx = sr[sr["team"] == result[0]].index[0]
            sr.at[idx, "net_rank"] = rank
            fuzzy_matched += 1
        else:
            unmatched.append((rank, net_row["ncaa_name"], norm))
            unmatched_count += 1

    print(f"NET rankings merged: exact={exact}, fuzzy={fuzzy_matched}, unmatched={unmatched_count}")
    if unmatched:
        print("  Unmatched (rank, ncaa_name, normalized):")
        for r, n, nn in unmatched[:15]:
            print(f"    #{r}: '{n}' → '{nn}'")

    # Show top 20 teams with NET rank assigned
    top = sr[sr["net_rank"].notna()].sort_values("net_rank").head(20)
    print("\nTop 20 by NET rank:")
    print(top[["net_rank", "team", "adj_margin", "barthag"]].to_string(index=False))

    sr.to_csv(sr_path, index=False)
    print(f"\nSaved {sr_path}")


if __name__ == "__main__":
    main()
