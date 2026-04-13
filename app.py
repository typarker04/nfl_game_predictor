import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import joblib
import nflreadpy as nfl

st.set_page_config(
    page_title="NFL Game Predictions",
    layout="wide",
    page_icon=":material/home:"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def moneyline_to_prob(ml):
    """Convert American moneyline to raw implied probability (before vig removal)."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def add_vegas_implied(df):
    """Add vig-normalized Vegas implied probability columns to a dataframe."""
    df = df.copy()
    df["away_implied_raw"] = df["away_moneyline"].apply(moneyline_to_prob)
    df["home_implied_raw"] = df["home_moneyline"].apply(moneyline_to_prob)
    total = df["away_implied_raw"] + df["home_implied_raw"]
    df["away_implied"] = df["away_implied_raw"] / total
    df["home_implied"] = df["home_implied_raw"] / total
    df.drop(columns=["away_implied_raw", "home_implied_raw"], inplace=True)
    return df

def get_last_updated():
    path = Path("data/latest_predictions.csv")
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y %I:%M %p")

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions():
    df = pd.read_csv("data/latest_predictions.csv")
    return add_vegas_implied(df)

@st.cache_data
def load_season_performance():
    """Apply the saved model to all 2025 regular-season games and return results."""
    features = joblib.load("models/feature_list.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    model    = joblib.load("models/finalized_model.pkl")

    df = pd.read_csv("data/games_with_stats.csv")
    season = df[(df["season"] == 2025) & (df["game_type"] == "REG")].copy()
    season = season.dropna(subset=features + ["home_win", "away_moneyline", "home_moneyline"])

    X = scaler.transform(season[features])
    season["home_win_prob"] = model.predict_proba(X)[:, 1]
    season["away_win_prob"] = 1 - season["home_win_prob"]
    season["pred_home_win"] = (season["home_win_prob"] > 0.5).astype(int)
    season["model_correct"] = (season["pred_home_win"] == season["home_win"]).astype(int)

    season = add_vegas_implied(season)
    season["vegas_pred_home_win"] = (season["home_implied"] > 0.5).astype(int)
    season["vegas_correct"] = (season["vegas_pred_home_win"] == season["home_win"]).astype(int)

    season["matchup"] = season["away_team"] + " @ " + season["home_team"]
    season["predicted_winner"] = season.apply(
        lambda r: r["home_team"] if r["pred_home_win"] == 1 else r["away_team"], axis=1
    )
    season["actual_winner"] = season.apply(
        lambda r: r["home_team"] if r["home_win"] == 1 else r["away_team"], axis=1
    )

    return season.sort_values(["week", "gameday"]).reset_index(drop=True)

predictions_df   = load_predictions()
season_df        = load_season_performance()
current_week     = nfl.get_current_week()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.info(f"Last updated: {get_last_updated()}")
    st.markdown("---")
    st.header("Filters")

    all_teams = sorted(
        set(predictions_df["home_team"].tolist() + predictions_df["away_team"].tolist())
    )
    selected_teams = st.multiselect(
        label="Filter by team (This Week)",
        options=all_teams,
        placeholder="All teams",
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_week, tab_season = st.tabs([f"Week {current_week} Predictions", "2025 Season Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THIS WEEK
# ══════════════════════════════════════════════════════════════════════════════

with tab_week:
    if selected_teams:
        filtered_df = predictions_df[
            predictions_df["home_team"].isin(selected_teams)
            | predictions_df["away_team"].isin(selected_teams)
        ]
    else:
        filtered_df = predictions_df

    st.title(f"Week {current_week} NFL Game Predictions")
    st.markdown("Win probability predictions compared to Vegas moneylines")
    st.markdown("---")

    total_games = len(filtered_df)
    avg_confidence = filtered_df["confidence"].mean() if total_games else 0
    home_favorites = int((filtered_df["home_win_prob"] > 0.5).sum())
    model_vegas_agree = int(
        ((filtered_df["home_win_prob"] > 0.5) == (filtered_df["home_implied"] > 0.5)).sum()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games This Week", total_games)
    col2.metric("Avg Model Confidence", f"{avg_confidence:.1%}")
    col3.metric("Home Favorites (Model)", f"{home_favorites} / {total_games}")
    col4.metric("Agrees w/ Vegas", f"{model_vegas_agree} / {total_games}")

    st.markdown("---")
    st.subheader("Home Team Win Probability vs Vegas")

    if filtered_df.empty:
        st.warning("No games match the selected filter.")
    else:
        bar_colors = ["#388e3c" if p > 0.5 else "#d32f2f" for p in filtered_df["home_win_prob"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Model",
            x=filtered_df["home_win_prob"],
            y=filtered_df["matchup"],
            orientation="h",
            marker=dict(color=bar_colors, opacity=0.8),
            text=filtered_df["home_win_prob"].apply(lambda x: f"{x:.0%}"),
            textposition="auto",
        ))
        fig.add_trace(go.Scatter(
            name="Vegas Implied",
            x=filtered_df["home_implied"],
            y=filtered_df["matchup"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="white",
                        line=dict(color="#333333", width=2)),
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.4)
        fig.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, 1], title="Win Probability (home team)"),
            height=max(420, len(filtered_df) * 60),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Game Details")
    if not filtered_df.empty:
        display_df = filtered_df[[
            "matchup", "game_date", "predicted_winner",
            "home_win_prob", "away_win_prob",
            "home_implied", "away_implied",
            "home_moneyline", "away_moneyline",
        ]].copy()
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "matchup":          st.column_config.TextColumn("Matchup"),
                "game_date":        st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                "predicted_winner": st.column_config.TextColumn("Predicted Winner"),
                "home_win_prob":    st.column_config.ProgressColumn("Home Win %", format="%.0f%%", min_value=0, max_value=1),
                "away_win_prob":    st.column_config.ProgressColumn("Away Win %", format="%.0f%%", min_value=0, max_value=1),
                "home_implied":     st.column_config.ProgressColumn("Vegas Home %", format="%.0f%%", min_value=0, max_value=1),
                "away_implied":     st.column_config.ProgressColumn("Vegas Away %", format="%.0f%%", min_value=0, max_value=1),
                "home_moneyline":   st.column_config.NumberColumn("Home ML", format="%+d"),
                "away_moneyline":   st.column_config.NumberColumn("Away ML", format="%+d"),
            },
        )

    st.markdown("---")
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name=f"nfl_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 2025 SEASON PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_season:
    st.title("2025 Season Performance")
    st.markdown("How the model performed on completed regular-season games vs Vegas moneylines")
    st.markdown("---")

    # ── Overall metrics ───────────────────────────────────────────────────────

    total       = len(season_df)
    model_acc   = season_df["model_correct"].mean()
    vegas_acc   = season_df["vegas_correct"].mean()
    model_only  = int(((season_df["model_correct"] == 1) & (season_df["vegas_correct"] == 0)).sum())
    vegas_only  = int(((season_df["model_correct"] == 0) & (season_df["vegas_correct"] == 1)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Evaluated", total)
    c2.metric("Model Accuracy", f"{model_acc:.1%}")
    c3.metric("Vegas Accuracy", f"{vegas_acc:.1%}")
    c4.metric("Model Only Correct", model_only, help="Games model got right that Vegas missed")
    c5.metric("Vegas Only Correct", vegas_only, help="Games Vegas got right that model missed")

    st.markdown("---")

    # ── Weekly accuracy chart ─────────────────────────────────────────────────

    weekly = (
        season_df.groupby("week")
        .agg(
            games          = ("model_correct", "count"),
            model_correct  = ("model_correct", "sum"),
            vegas_correct  = ("vegas_correct", "sum"),
        )
        .reset_index()
    )
    weekly["Model"]  = weekly["model_correct"] / weekly["games"]
    weekly["Vegas"]  = weekly["vegas_correct"]  / weekly["games"]

    st.subheader("Weekly Accuracy: Model vs Vegas")

    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Bar(
        name="Model",
        x=weekly["week"],
        y=weekly["Model"],
        marker_color="#1565c0",
        opacity=0.85,
        text=weekly["Model"].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
    ))
    fig_weekly.add_trace(go.Bar(
        name="Vegas",
        x=weekly["week"],
        y=weekly["Vegas"],
        marker_color="#ef6c00",
        opacity=0.85,
        text=weekly["Vegas"].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
    ))
    fig_weekly.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4)
    fig_weekly.update_layout(
        barmode="group",
        xaxis=dict(title="Week", tickmode="linear", dtick=1),
        yaxis=dict(tickformat=".0%", range=[0, 1.15], title="Accuracy"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_weekly, width="stretch")

    # ── Cumulative accuracy chart ─────────────────────────────────────────────

    st.subheader("Cumulative Accuracy Over the Season")

    season_df["model_cum"]  = season_df["model_correct"].expanding().mean()
    season_df["vegas_cum"]  = season_df["vegas_correct"].expanding().mean()
    season_df["game_num"]   = range(1, len(season_df) + 1)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        name="Model",
        x=season_df["game_num"],
        y=season_df["model_cum"],
        mode="lines",
        line=dict(color="#1565c0", width=2),
    ))
    fig_cum.add_trace(go.Scatter(
        name="Vegas",
        x=season_df["game_num"],
        y=season_df["vegas_cum"],
        mode="lines",
        line=dict(color="#ef6c00", width=2),
    ))
    fig_cum.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4,
                      annotation_text="50%", annotation_position="right")
    fig_cum.update_layout(
        xaxis=dict(title="Game #"),
        yaxis=dict(tickformat=".0%", range=[0.4, 1.0], title="Cumulative Accuracy"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cum, width="stretch")

    # ── Game-by-game results table ────────────────────────────────────────────

    st.subheader("Game-by-Game Results")

    week_options = ["All weeks"] + [f"Week {w}" for w in sorted(season_df["week"].unique())]
    sel_week = st.selectbox("Filter by week", week_options)

    if sel_week == "All weeks":
        table_df = season_df
    else:
        w = int(sel_week.split()[-1])
        table_df = season_df[season_df["week"] == w]

    results_display = table_df[[
        "week", "matchup", "actual_winner", "predicted_winner",
        "home_win_prob", "home_implied", "model_correct",
    ]].copy()

    st.dataframe(
        results_display,
        width="stretch",
        hide_index=True,
        column_config={
            "week":              st.column_config.NumberColumn("Week"),
            "matchup":           st.column_config.TextColumn("Matchup"),
            "actual_winner":     st.column_config.TextColumn("Actual Winner"),
            "predicted_winner":  st.column_config.TextColumn("Model Pick"),
            "home_win_prob":     st.column_config.ProgressColumn("Home Win %", format="%.0f%%", min_value=0, max_value=1),
            "home_implied":      st.column_config.ProgressColumn("Vegas Home %", format="%.0f%%", min_value=0, max_value=1),
            "model_correct":     st.column_config.CheckboxColumn("Model Correct"),
        },
    )
