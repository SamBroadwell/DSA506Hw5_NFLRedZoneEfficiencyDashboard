# data prep code
# data_prep.py
# Helper functions to load + transform your NFL 2024 play-by-play into red-zone "trips" and metrics.

import re
import numpy as np
import pandas as pd


# PlayTypes that are not meaningful for "red zone efficiency" by default
DEFAULT_EXCLUDE_PLAYTYPES = {
    "TIMEOUT", "CLOCK STOP", "NO PLAY", "KICK OFF"
}

# Plays you usually DON'T want counted as part of "red zone offense efficiency" (post-TD, admin, etc.)
DEFAULT_EXCLUDE_FOR_EFFICIENCY = DEFAULT_EXCLUDE_PLAYTYPES | {
    "EXTRA POINT", "TWO-POINT CONVERSION", "QB KNEEL"
}

# Core offense plays that represent attempts to score in the red zone
CORE_OFFENSE_PLAYTYPES = {"PASS", "RUSH", "SACK", "SCRAMBLE", "FIELD GOAL"}


def load_pbp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    blank_cols = [c for c in df.columns if c == "" or c.lower().startswith("unnamed")]
    if blank_cols:
        df = df.drop(columns=blank_cols)

    # Parse date
    if "GameDate" in df.columns:
        df["GameDate"] = pd.to_datetime(df["GameDate"], errors="coerce")

    # Coerce numeric columns (if present)
    for col in ["Quarter", "Minute", "Second", "Down", "ToGo", "YardLine", "YardLineFixed", "Yards"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce indicator columns (0/1 expected)
    flag_cols = [
        "IsRush","IsPass","IsIncomplete","IsTouchdown","IsSack","IsInterception","IsFumble",
        "IsPenalty","IsPenaltyAccepted","IsNoPlay"
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize string columns
    for col in ["OffenseTeam","DefenseTeam","PlayType","YardLineDirection","Description"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Standardize PlayType safely
    if "PlayType" in df.columns:
        df["PlayType"] = df["PlayType"].str.upper()
        df.loc[df["PlayType"].isna(), "PlayType"] = pd.NA

    return df


def add_time_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a sortable time index within each game so we can approximate possession breaks.
    """
    out = df.copy()
    # seconds elapsed from start of game (approx) using quarter and clock
    out["GameSecondsElapsed"] = (
        (out["Quarter"].fillna(0).astype(int) - 1) * 15 * 60
        + (15 * 60 - (out["Minute"].fillna(0) * 60 + out["Second"].fillna(0)))
    )
    return out


def add_red_zone_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the validated rule: opponent side + YardLineFixed >= 80.
    """
    out = df.copy()
    out["IsRedZone"] = (
        out["YardLineDirection"].str.upper().eq("OPP")
        & (pd.to_numeric(out["YardLineFixed"], errors="coerce") >= 80)
    ).fillna(False)
    return out


def add_field_goal_result_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives FG attempt + make/miss using Description patterns.
    """
    out = df.copy()
    desc = out.get("Description", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    ptype = out.get("PlayType", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()

    is_fg_play = ptype.eq("FIELD GOAL") | desc.str.contains("FIELD GOAL", regex=False)

    # Attempt = any FG play (including blocked)
    out["IsFGAttempt"] = is_fg_play.astype(int)

    # Good vs Missed/Blocked
    out["IsFGGood"] = (is_fg_play & desc.str.contains(" IS GOOD", regex=False)).astype(int)
    out["IsFGMiss"] = (
        is_fg_play & (
            desc.str.contains(" IS NO GOOD", regex=False)
            | desc.str.contains(" IS BLOCKED", regex=False)
            | desc.str.contains(" MISSED", regex=False)
        )
    ).astype(int)

    # Sometimes descriptions might not match; keep good/miss mutually exclusive where possible
    out.loc[out["IsFGGood"] == 1, "IsFGMiss"] = 0

    return out


def build_possession_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates possessions by detecting OffenseTeam changes within a game,
    after removing IsNoPlay==1 rows for sequencing.
    """
    out = df.copy()

    # Remove no-play from sequencing (still keep in df; we just don't want them to define possession breaks)
    seq = out.copy()
    if "IsNoPlay" in seq.columns:
        seq = seq[seq["IsNoPlay"] != 1].copy()

    seq = add_time_ordering(seq)
    seq = seq.sort_values(["GameId", "GameSecondsElapsed"], ascending=[True, True])

    seq["OffensePrev"] = seq.groupby("GameId")["OffenseTeam"].shift(1)
    seq["NewPoss"] = (seq["OffenseTeam"] != seq["OffensePrev"]).fillna(True).astype(int)
    seq["PossessionIndex"] = seq.groupby("GameId")["NewPoss"].cumsum()
    seq["PossessionId"] = seq["GameId"].astype(str) + "-" + seq["PossessionIndex"].astype(int).astype(str)

    out["PossessionId"] = np.nan
    out.loc[seq.index, "PossessionId"] = seq["PossessionId"]
    return out


def compute_trip_table(
    df: pd.DataFrame,
    exclude_playtypes_for_efficiency = None
) -> pd.DataFrame:
    """
    Defines a red-zone "trip" as a possession that contains at least one red-zone efficiency play.
    Then returns one row per trip with trip-level outcomes and summary features.

    Trip outcomes (per trip):
      - trip_TD: any touchdown during RZ core offense play
      - trip_FG_good: any made FG in the red zone
      - trip_FG_att: any FG attempt in the red zone
      - trip_INT: any interception in the red zone
      - trip_FUM_flag: any fumble flagged in the red zone (not necessarily lost)
      - trip_empty: not TD and not FG_good
    """
    if exclude_playtypes_for_efficiency is None:
        exclude_playtypes_for_efficiency = DEFAULT_EXCLUDE_FOR_EFFICIENCY

    work = df.copy()
    work["PlayType"] = work["PlayType"].astype(str).str.upper()

    # must have a possession id to do trips
    work = work.dropna(subset=["PossessionId"]).copy()

    # mark "efficiency eligible" plays (for entry + denominators)
    work["IsEfficiencyPlay"] = (
        ~work["PlayType"].isin(exclude_playtypes_for_efficiency)
        & work["PlayType"].isin(CORE_OFFENSE_PLAYTYPES | {"PASS", "RUSH", "SACK", "SCRAMBLE"})
    )

    # red zone eligible subset
    rz = work[work["IsRedZone"] & work["IsEfficiencyPlay"]].copy()
    if rz.empty:
        return pd.DataFrame(columns=[
            "PossessionId","OffenseTeam","DefenseTeam","GameId","GameDate",
            "trip_TD","trip_FG_good","trip_FG_att","trip_INT","trip_FUM_flag",
            "plays_in_trip","yards_in_trip","trip_empty"
        ])

    # Determine a single defense team for the trip (take most common within the trip)
    def_mode = lambda s: s.value_counts().index[0] if len(s.value_counts()) else np.nan

    agg = {
        "OffenseTeam": "first",
        "DefenseTeam": def_mode,
        "GameId": "first",
        "GameDate": "first",
        "IsTouchdown": "max",
        "IsFGGood": "max",
        "IsFGAttempt": "max",
        "IsInterception": "max",
        "IsFumble": "max",
        "Yards": "sum",
        "PlayType": "count",
    }

    trip = rz.groupby("PossessionId", as_index=False).agg(agg).rename(columns={
        "IsTouchdown": "trip_TD",
        "IsFGGood": "trip_FG_good",
        "IsFGAttempt": "trip_FG_att",
        "IsInterception": "trip_INT",
        "IsFumble": "trip_FUM_flag",
        "Yards": "yards_in_trip",
        "PlayType": "plays_in_trip",
    })

    trip["trip_empty"] = ((trip["trip_TD"] != 1) & (trip["trip_FG_good"] != 1)).astype(int)
    return trip


def compute_team_metrics(trip: pd.DataFrame, min_trips: int = 10) -> pd.DataFrame:
    """
    Returns offense-team aggregated trip metrics.
    """
    if trip.empty:
        return pd.DataFrame()

    g = trip.groupby("OffenseTeam").agg(
        trips=("PossessionId", "nunique"),
        TD_rate=("trip_TD", "mean"),
        FG_good_rate=("trip_FG_good", "mean"),
        FG_att_rate=("trip_FG_att", "mean"),
        INT_rate=("trip_INT", "mean"),
        FUM_flag_rate=("trip_FUM_flag", "mean"),
        empty_rate=("trip_empty", "mean"),
        plays_per_trip=("plays_in_trip", "mean"),
        yards_per_trip=("yards_in_trip", "mean"),
    ).reset_index()

    return g[g["trips"] >= min_trips].sort_values("TD_rate", ascending=False)


def compute_defense_metrics(trip: pd.DataFrame, min_trips: int = 10) -> pd.DataFrame:
    """
    Returns defense-team aggregated trip metrics (how often opponents score vs you).
    """
    if trip.empty:
        return pd.DataFrame()

    g = trip.groupby("DefenseTeam").agg(
        defended_trips=("PossessionId", "nunique"),
        opp_TD_rate=("trip_TD", "mean"),
        opp_FG_good_rate=("trip_FG_good", "mean"),
        opp_FG_att_rate=("trip_FG_att", "mean"),
        opp_INT_rate=("trip_INT", "mean"),
        opp_FUM_flag_rate=("trip_FUM_flag", "mean"),
        opp_empty_rate=("trip_empty", "mean"),
        opp_plays_per_trip=("plays_in_trip", "mean"),
        opp_yards_per_trip=("yards_in_trip", "mean"),
    ).reset_index()

    return g[g["defended_trips"] >= min_trips].sort_values("opp_TD_rate", ascending=True)


def add_week_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an NFL-style sequential week number (W1, W2, ...) based on GameDate.
    Uses Monday week-start buckets and renumbers from the first week in the dataset.
    """
    out = df.copy()
    if "GameDate" in out.columns:
        week_start = out["GameDate"] - pd.to_timedelta(out["GameDate"].dt.weekday, unit="D")
        week_start = week_start.dt.normalize()

        # sequential numbering from first week in the data
        out["WeekNum"] = ((week_start - week_start.min()).dt.days // 7 + 1).astype(int)
        out["Week"] = out["WeekNum"].map(lambda x: f"W{x}")
    else:
        out["WeekNum"] = np.nan
        out["Week"] = "Unknown"
    return out


def down_yardline_band_heatmap_data(df_rz_eff: pd.DataFrame, value_col: str = "IsTouchdown") -> pd.DataFrame:
    """
    Builds a Down x Yardline-band pivot for heatmap.
    Bands based on YardLineFixed: 80-84 (20-16), 85-89 (15-11), 90-94 (10-6), 95-99 (5-1), 100+ (GL)
    """
    x = df_rz_eff.copy()
    x["RZ_Band"] = pd.cut(
        pd.to_numeric(x["YardLineFixed"], errors="coerce"),
        bins=[79, 84, 89, 94, 99, 101],
        labels=["20-16", "15-11", "10-6", "5-1", "GL"],
        include_lowest=True
    )
    pivot = x.pivot_table(index="Down", columns="RZ_Band", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=[1, 2, 3, 4])
    return pivot
