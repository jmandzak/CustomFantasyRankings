"""Script to clean up the ranks csv"""

from FootballNameMatcher.match import match_name
import typing
import pandas as pd
from argparse import ArgumentParser, Namespace


def remove_team_from_name(name: str) -> str:
    """Removes the team name from a player's name, team name will be the last word"""
    return " ".join(name.split()[:-1]) if len(name.split()) > 1 else name


def clean_name(name: str) -> str:
    """Cleans a player's name by removing the team name and matching it"""
    name = remove_team_from_name(name)
    new_name = match_name(name)
    return new_name if new_name else name


def remove_duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate player names, keeping the player with higher TTL"""
    return df.sort_values("TTL", ascending=False).drop_duplicates(
        subset="PLAYER", keep="first"
    )


def remove_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes columns that are not needed from the dataframe"""
    # only keep PLAYER, GP, AVG, TTL
    return df[["PLAYER", "GP", "AVG", "TTL"]]


def get_df(filename: str) -> pd.DataFrame:
    """Reads the csv file and returns a cleaned dataframe"""
    df = pd.read_csv(filename)
    df = remove_useless_columns(df)
    df["PLAYER"] = df["PLAYER"].apply(clean_name)
    df = remove_duplicate_names(df)
    return df


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Clean up the ranks CSV file")
    parser.add_argument("input", type=str, default="", help="Input CSV file")
    parser.add_argument(
        "--output", type=str, default="cleaned_actual_ranks.csv", help="Output CSV file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = get_df(args.input)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
