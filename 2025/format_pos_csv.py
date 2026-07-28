import pandas as pd
from FootballNameMatcher.match import match_name

FILE_NAME = "FantasyRankings2025.csv"

df = pd.read_csv(FILE_NAME)

# drop rows that have nan values besides the pos_rk column
df = df.dropna(subset=df.columns.difference(["POS_RK"]), how="all")
# drop DEF, DEF_Notes, K, K_Notes
df = df.drop(columns=["DEF", "DEF_Notes", "K", "K_Notes"])

# Replace all newlines with ". " in any Notes column
for col in df.columns:
    if "Notes" in col:
        df[col] = df[col].astype(str).str.replace("\n([A-Z])", ". \\1", regex=True)
        df[col] = df[col].str.replace("\n", " ", regex=False)

# in the QB, RB, WR, and TE columns, replace the value with match_names(value) unless the value is nan
for col in ["QB", "RB", "WR", "TE"]:
    df[col] = df[col].apply(lambda x: match_name(x) if pd.notna(x) else x)

df.to_csv("FantasyPositionRankings2025.csv", index=False)
