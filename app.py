# dashboard code
# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_prep import (
    load_pbp_csv,
    add_red_zone_flag,
    add_field_goal_result_flags,
    build_possession_ids,
    compute_trip_table,
    compute_team_metrics,
    compute_defense_metrics,
    add_week_bucket,
    down_yardline_band_heatmap_data,
    DEFAULT_EXCLUDE_FOR_EFFICIENCY,
)

st.set_page_config(page_title="NFL Play-By-Play Red Zone Efficiency Dashboard", layout="wide")

st.title("NFL Play-By-Play Red Zone Efficiency Dashboard")

st.markdown(
    """
**Analytical Objective:** For this dashboard, I wanted to visualize and compare how efficiently NFL teams convert **red-zone trips**
(which are considered plays inside the opponent 20 yard line) into **touchdowns, field goals, and other metrics**, and how that efficiency varies
by down, yardline, and over time. This dashboard provides a league overview in the first tab, where the user can compare team performances,
and a single team analysis in the second tab, where the user can dive deeper into a single teams efficiency. Each tab provides several charts portraying different metrics, 
and there are a handful of different toggles, filters, and options the user has for specific charts or for the entire dahsboard (via the sidebar controls)

Data Source: Publicly available NFL play-by-play data (csv file in repo) found at the following link: https://nflsavant.com/about.php. 
Note that the sidebar controls provide an option to select different csv files that are uploaded in the Github repository
"""
)

# -------------------------
# Load + preprocess (cached)
# -------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare(data_path: str) -> pd.DataFrame:
    df = load_pbp_csv(data_path)
    df = add_red_zone_flag(df)
    df = add_field_goal_result_flags(df)
    df = build_possession_ids(df)
    df = add_week_bucket(df)
    return df

# -------------------------
# Sidebar controls (recommended)
# -------------------------
st.sidebar.header("Controls")

# (Optional) while debugging: easy cache clear
if st.sidebar.button("Clear cache / rerun"):
    st.cache_data.clear()
    st.rerun()

# -------------------------
# Data source
# -------------------------
st.sidebar.subheader("Data")

# Option A: choose from known files in the repo (safer than free text)
KNOWN_FILES = ["pbp-2024.csv"]  # add more if you have them in your repo
data_mode = st.sidebar.radio(
    "Data source",
    options=["Use repo file", "Upload CSV"],
    index=0,
    horizontal=False,
    help="Use a known file in the repo (recommended) or upload a CSV for ad-hoc analysis."
)

uploaded = None
if data_mode == "Use repo file":
    DATA_PATH = st.sidebar.selectbox(
        "Repo CSV",
        options=KNOWN_FILES,
        index=0,
        help="Select a CSV that exists in your repo."
    )
else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a play-by-play CSV to analyze. Useful when sharing the dashboard."
    )
    # If nothing uploaded, fall back to default
    DATA_PATH = uploaded if uploaded is not None else "pbp-2024.csv"

# Load data (same cached function you already have)
df = load_and_prepare(DATA_PATH)

# -------------------------
# Filters
# -------------------------
st.sidebar.subheader("Filters")

# Date range
if "GameDate" in df.columns and df["GameDate"].notna().any():
    min_date = df["GameDate"].min().date()
    max_date = df["GameDate"].max().date()
    date_range = st.sidebar.date_input(
        "Game date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filters all charts/tables to games within this date range."
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = None, None
    st.sidebar.info("No GameDate column found; date filtering disabled.")

# -------------------------
# Ranking / stability controls
# -------------------------
st.sidebar.subheader("Ranking & stability")

# We'll compute trip later, but we can still set a sensible max for the slider.
# Use a conservative bound; we'll also clamp later once trip exists.
min_trips = st.sidebar.slider(
    "Minimum red-zone trips (sample size filter)",
    min_value=5,
    max_value=100,
    value=25,
    step=5,
    help="Excludes teams with fewer trips to avoid noisy small-sample rankings."
)

metric_labels = {
    "TD_rate": "Touchdown rate (per trip)",
    "FG_good_rate": "Field goal make rate (per trip)",
    "FG_att_rate": "Field goal attempt rate (per trip)",
    "empty_rate": "Empty trip rate (no TD/FG)",
    "INT_rate": "Interception rate (per trip)",
}
metric_choice = st.sidebar.selectbox(
    "Primary metric",
    options=list(metric_labels.keys()),
    format_func=lambda k: metric_labels[k],
    index=0,
    help="Controls team ranking and the league trend chart."
)

# -------------------------
# Advanced options (hide clutter)
# -------------------------
with st.sidebar.expander("Advanced", expanded=False):
    include_special = st.checkbox(
        "Include non-offense/admin plays in denominators (not recommended)",
        value=False,
        help=(
            "By default, plays like EXTRA POINT / TWO-POINT / TIMEOUT / NO PLAY / KICK OFF / QB KNEEL "
            "are excluded from efficiency calculations."
        )
    )

# -------------------------
# Team selections
# -------------------------
st.sidebar.subheader("Teams")

teams = sorted(df["OffenseTeam"].dropna().unique().tolist())
focus_team = st.sidebar.selectbox(
    "Focus team (Deep Dive)",
    options=teams,
    index=teams.index("BUF") if "BUF" in teams else 0,
    help="Controls the Team Deep Dive tab."
)

# Cap compare teams for readability
MAX_COMPARE = 6
default_compare = [t for t in ["BUF", "BAL", "DET", "KC"] if t in teams][:3]
compare_teams = st.sidebar.multiselect(
    "Label teams on scatter (optional)",
    options=teams,
    default=default_compare,
    help=f"Select up to {MAX_COMPARE} teams to label on the scatter plot."
)

if len(compare_teams) > MAX_COMPARE:
    st.sidebar.warning(f"Please select at most {MAX_COMPARE} teams for readability.")
    compare_teams = compare_teams[:MAX_COMPARE]

# -------------------------
# Filter data by date range
# -------------------------
df_f = df.copy()
if start_date is not None:
    df_f = df_f[(df_f["GameDate"].dt.date >= start_date) & (df_f["GameDate"].dt.date <= end_date)].copy()

exclude_set = set() if include_special else DEFAULT_EXCLUDE_FOR_EFFICIENCY

# -------------------------
# Build trips + team tables
# -------------------------
trip = compute_trip_table(df_f, exclude_playtypes_for_efficiency=exclude_set)
# Ensure trip table has Week so the league trend chart can render
# Attach Week/WeekNum to trip table for trend charts
if not trip.empty and "WeekNum" in df_f.columns:
    week_map = (
        df_f[["PossessionId", "Week", "WeekNum"]]
        .dropna(subset=["PossessionId", "WeekNum"])
        .drop_duplicates("PossessionId")
    )
    trip = trip.merge(week_map, on="PossessionId", how="left")
off_tbl = compute_team_metrics(trip, min_trips=min_trips)
def_tbl = compute_defense_metrics(trip, min_trips=min_trips)

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_team = st.tabs(["League Overview", "Team Deep Dive"])

# -------------------------
# TAB 1: League Overview
# -------------------------
with tab_overview:
    st.subheader("League Overview")

#--------------------------
# ANALYSIS
#--------------------------
    st.markdown("""
    #### Analysis
    In this first tab providing metrics for the entire league, the first two charts (bar chart and scatter plot) clearly show which teams performed the best
    in the season based on the metric that is selected. For example, when using 'Touchdown rate' for this analysis, these charts show the teams that had the highest rate of successfully
    scoring a touchdown within the red zone that season. In the case of the 2024 play-by-play data this would be the Buffalo Bills, who had a nearly 70% touchdown 
    rate when in the red zone. In the scatter plot comparing the touchdown rate and the field goal rate, there are teams who had much higher touchdown rates which 
    resulted in a lower field goal rate, some teams with nearly equal rates, and some teams that had much higher field goal rates than touchdown rates. 
    The teams with the much higher field goal rates should focus on maximizing the opportunity of being in the red zone and scoring a touchdown instead of a field goal,
    for example trying new plays once in the red zone. 

    Then there are league-wide metrics provided with the time series plot displaying the average touchdown rate as a rolling mean over the configured number of weeks, 
    and the heatmap portraying the touchdown rate from various positions within the red zone. The time series plot shows clear trends in how all teams performed that week 
    within the red zone. For example, with a four week rolling window the touchdown rate dropped down to its lowest of the season around 40%, meaning this is when most teams
    around the league did not convert a touchdown when in the red zone. But in the succeeding weeks the rate jumped back up to values around 50%. 
    Lastly, the heatmap provides values that would be expected for touchdown rates within the red zone. Where the highest touchdown rates are seen between the 1-5 yard line.
    But somewhat surprisingly, the highest rate of all is on fourth down, indicating that teams who take the risk on fourth down generally succeed on scoring a touchdown.

    Overall, this first tab provides an overview of league-wide performance by comparing each team based on a selected metric (e.g., touchdown rate, as used for this 
    analysis), and total league performance to see how the metric trended throughout the season. These insights from this tab can be analyzed further on a per-team basis 
    in the second tab of the dashboard.
    """)

    # Metric cards (dashboard elements)
    c1, c2, c3, c4 = st.columns(4)
    total_trips = int(trip["PossessionId"].nunique()) if not trip.empty else 0
    overall_td = float(trip["trip_TD"].mean()) if total_trips else 0.0
    overall_fg = float(trip["trip_FG_good"].mean()) if total_trips else 0.0
    overall_empty = float(trip["trip_empty"].mean()) if total_trips else 0.0

    c1.metric("Red-zone trips (approx)", f"{total_trips:,}")
    c2.metric("TD rate (per trip)", f"{overall_td:.3f}")
    c3.metric("FG good rate (per trip)", f"{overall_fg:.3f}")
    c4.metric("Empty trip rate", f"{overall_empty:.3f}")

    st.caption(
        "Note: Trips are approximated from possession changes (OffenseTeam switches) using play ordering. "
        "This is sufficient for a dashboard comparison, but not an official NFL drive table."
    )

    # -------------------------
    # BAR CHART: Top/bottom teams by selected metric
    # -------------------------
    st.markdown("### Team rankings (bar chart)")
    if off_tbl.empty:
        st.info("No trip data available for the current filters.")
    else:
        show_n = st.slider("Number of teams to show", 1, 32, 10, 1)
        plot_df = off_tbl.sort_values(metric_choice, ascending=False).head(show_n)

        fig, ax = plt.subplots()
        ax.bar(plot_df["OffenseTeam"], plot_df[metric_choice])
        ax.set_title(f"Top {show_n} offenses by {metric_choice} (min {min_trips} trips)")
        ax.set_xlabel("Team")
        ax.set_ylabel(metric_choice)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------
    # SCATTER: TD rate vs FG good rate (with optional labels)
    # -------------------------
    st.markdown("### TD vs FG profile (scatter plot)")
    if not off_tbl.empty:
        fig, ax = plt.subplots()
        ax.scatter(off_tbl["FG_good_rate"], off_tbl["TD_rate"], alpha=0.8)

        for _, row in off_tbl.iterrows():
            ax.text(
                row["FG_good_rate"],
                row["TD_rate"],
                row["OffenseTeam"],
                fontsize=8,
                ha="center",
                va="bottom"
            )

        ax.set_title("Red-zone outcomes: FG-good rate vs TD rate (per trip)")
        ax.set_xlabel("FG-good rate")
        ax.set_ylabel("TD rate")
        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------
    # LINE: Weekly trend of league-wide metric
    # -------------------------
    st.markdown("### League trend (line chart)")
    if trip.empty:
        st.info("No trip data available for the current filters.")
    elif "Week" not in trip.columns:
        st.info("Trip table has no 'Week' column. Attach Week to trips (see merge fix).")
    else:
        metric_map = {
            "TD_rate": "trip_TD",
            "FG_good_rate": "trip_FG_good",
            "FG_att_rate": "trip_FG_att",
            "empty_rate": "trip_empty",
            "INT_rate": "trip_INT",
        }
        col = metric_map[metric_choice]
        weekly = (
            trip.dropna(subset=["WeekNum"])
                .groupby("WeekNum")[col].mean()
                .reset_index()
                .sort_values("WeekNum")
        )
        weekly["WeekLabel"] = weekly["WeekNum"].map(lambda x: f"W{int(x)}")

        rolling_k = st.slider("Rolling window (weeks)", 1, 8, 4, 1)
        weekly["rolling"] = weekly[col].rolling(rolling_k, min_periods=1).mean()

        fig, ax = plt.subplots()
        ax.plot(weekly["WeekNum"], weekly["rolling"])
        ax.set_xticks(weekly["WeekNum"])
        ax.set_xticklabels(weekly["WeekLabel"], rotation=90)
        ax.set_title(f"League-wide {metric_choice} (rolling {rolling_k}-week mean)")
        ax.set_xlabel("Week")
        ax.set_ylabel(metric_choice)
        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------
    # HEATMAP: Down x Yardline band (league-wide)
    # -------------------------
    st.markdown("### Where do TDs happen? (heatmap)")
    # Build a red-zone, efficiency-play subset for heatmap
    heat_df = df_f.copy()
    heat_df["PlayType"] = heat_df["PlayType"].astype(str).str.upper()
    heat_df["IsEfficiencyPlay"] = (~heat_df["PlayType"].isin(exclude_set)) & heat_df["PlayType"].isin(
        {"PASS","RUSH","SACK","SCRAMBLE"}
    )
    heat_df = heat_df[heat_df["IsRedZone"] & heat_df["IsEfficiencyPlay"]].copy()

    if heat_df.empty:
        st.info("No red-zone efficiency plays available for the current filters.")
    else:
        pivot = down_yardline_band_heatmap_data(heat_df, value_col="IsTouchdown")
        fig, ax = plt.subplots()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title("TD rate by Down × Yardline band (league-wide)")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(str))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(str))
        ax.set_xlabel("Yardline band (inside opp 20)")
        ax.set_ylabel("Down")
        fig.colorbar(im, ax=ax, label="TD rate")
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------
# TAB 2: Team Deep Dive
# -------------------------
with tab_team:
    st.subheader(f"Team Deep Dive: {focus_team}")

    team_trip = trip[trip["OffenseTeam"] == focus_team].copy() if not trip.empty else pd.DataFrame()
    team_pbp = df_f[df_f["OffenseTeam"] == focus_team].copy()

    st.markdown("""
    #### Analysis
    In this second tab providing metrics for the selected team, similar types of charts are provided but adjusted to be relevant for the single team. For this analysis,
    I focused on PIT, Pittsburgh Steelers (since they are my favorite team). 
    
    The first bar chart displays the distribution of plays within the red zone. In the case of 
    PIT, their most common plays in the red zone were rush, pass, and timeout. This distribution could tell the team what is most common and possibly guide decisions around
    the types of plays to run within the red zone so they are not predictable.
    
    The second time series chart displays their touchdown rate throughout the season as a rolling average. With a rolling average setting of four weeks, their touchdown rate 
    was at various different values throughout the season. The first few weeks of the season, their touchdown rate was pretty low, but by week seven they hit their highest 
    touchdown rate just above 70%, which then dropped down and fluctuated through the remainder of the season. This could tell the team which times of the season
    contained the best and worst touchdown rates, where they could then identify the methods that were used in each for future reference. 
                
    With the heatmap displaying the teams touchdown rate by down and yard
    line, it is clear to see where PIT was most successful within the red zone. for the most part, the touchdown rate is pretty low except for the 1-5 yard line on fourth down,
    which follows the trend that was seen in the league overview. This means that PIT really did not perform well within the red zone, they were not converting touchdowns. 
    This tells the team that they really need to focus on red zone efficiency so they can begin to capitalize on touchdowns instead of settling for field goals.
    
    Lastly, for the defense-related bar chart, this compares how well the separate teams performed on defense against the offense when in the red zone. In other words, teams
    with a lower opponent touchdown rate had better red zone defense. Although this is not focusing specifically on PIT, it is a team-related metric that provides a different 
    point of view from the other offensive red zone metrics that are seen in the dashboard. This chart clearly shows which teams have the best defense, where the teams with the
    highest opponent touchdown rate should focus on their teams red zone defense to prevent teams from scoring in those moments. 

    Overall, this tab provides great insight into a specific teams red zone efficiency during the season. In this case with PIT, it is clear that they did not perform well
    in the red zone and should focus on that in succeeding seasons. The dashboard user can go team by team to do some more in depth comparisons and check these metrics.             
    """)

    # Metrics row for the focus team
    if team_trip.empty:
        st.warning("No trips for this team under current filters.")
    else:
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        cc1.metric("Trips", f"{team_trip['PossessionId'].nunique():,}")
        cc2.metric("TD / trip", f"{team_trip['trip_TD'].mean():.3f}")
        cc3.metric("FG good / trip", f"{team_trip['trip_FG_good'].mean():.3f}")
        cc4.metric("Empty / trip", f"{team_trip['trip_empty'].mean():.3f}")
        cc5.metric("INT / trip", f"{team_trip['trip_INT'].mean():.3f}")

    # Bar: red-zone play mix for the team (counts)
    st.markdown("### Red-zone play mix (bar chart)")
    rz_team = team_pbp[team_pbp["IsRedZone"]].copy()
    rz_team = rz_team[rz_team["PlayType"].notna()].copy()

    mix = rz_team["PlayType"].value_counts().head(12)
    fig, ax = plt.subplots()
    ax.bar(mix.index.astype(str), mix.values)
    ax.set_title("Top PlayTypes in red zone (counts)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Plays")
    plt.tight_layout()
    st.pyplot(fig)

    # Line: weekly trend for the focus team (TD rate per trip)
    st.markdown("### Team trend over time (line chart)")
    if not team_trip.empty and "Week" in team_trip.columns:
        weekly = (
            team_trip.dropna(subset=["WeekNum"])
                    .groupby("WeekNum")["trip_TD"].mean()
                    .reset_index()
                    .sort_values("WeekNum")
        )
        weekly["WeekLabel"] = weekly["WeekNum"].map(lambda x: f"W{int(x)}")
        rolling_w = st.slider("Rolling window (weeks) — team trend", 1, 8, 4, 1)
        weekly["rolling"] = weekly["trip_TD"].rolling(rolling_w, min_periods=1).mean()

        fig, ax = plt.subplots()
        ax.plot(weekly["WeekNum"], weekly["rolling"])
        ax.set_xticks(weekly["WeekNum"])
        ax.set_xticklabels(weekly["WeekLabel"], rotation=45)
        ax.set_title(f"{focus_team} TD rate per trip (rolling {rolling_w}-week mean)")
        ax.set_xlabel("Week")
        ax.set_ylabel("TD rate")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Heatmap: Down x Yardline band for the team (TD rate on plays)
    st.markdown("### Situational efficiency (heatmap)")
    heat_team = team_pbp.copy()
    heat_team["PlayType"] = heat_team["PlayType"].astype(str).str.upper()
    heat_team["IsEfficiencyPlay"] = (~heat_team["PlayType"].isin(exclude_set)) & heat_team["PlayType"].isin(
        {"PASS","RUSH","SACK","SCRAMBLE"}
    )
    heat_team = heat_team[heat_team["IsRedZone"] & heat_team["IsEfficiencyPlay"]].copy()

    if heat_team.empty:
        st.info("No red-zone efficiency plays for this team under current filters.")
    else:
        pivot = down_yardline_band_heatmap_data(heat_team, value_col="IsTouchdown")
        fig, ax = plt.subplots()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"{focus_team} TD rate by Down × Yardline band")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(str))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(str))
        ax.set_xlabel("Yardline band (inside opp 20)")
        ax.set_ylabel("Down")
        fig.colorbar(im, ax=ax, label="TD rate")
        plt.tight_layout()
        st.pyplot(fig)

    # Defense comparison bar chart
    st.markdown("### Defenses that hold opponents to fewer TDs (bar chart)")
    if def_tbl.empty:
        st.info("No defense table available for the current filters.")
    else:
        show_n_def = st.slider("Number of defenses to show", 5, 32, 10, 1, key="defn")
        top_def = def_tbl.head(show_n_def)

        fig, ax = plt.subplots()
        ax.bar(top_def["DefenseTeam"], top_def["opp_TD_rate"])
        ax.set_title(f"Top {show_n_def} defenses by lowest opponent TD rate (min {min_trips} defended trips)")
        ax.set_xlabel("Defense")
        ax.set_ylabel("Opponent TD rate (per trip)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
