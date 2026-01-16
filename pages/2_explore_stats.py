import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Team Stats Explorer",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_team_stats():
    """Load processed team statistics."""
    return pd.read_csv('data/df_clean.csv')

# Title
st.title("Team Statistics Explorer")
st.markdown("Explore individual team performance trends throughout the season using EWMA (Exponentially Weighted Moving Average)")

# Load data
try:
    df = load_team_stats()
    
    # Sidebar controls
    st.sidebar.header("Filters")
    
    # Season selection
    available_seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.sidebar.selectbox(
        "Select Season",
        available_seasons,
        index=0  # Default to most recent
    )
    
    # Filter data by season
    season_df = df[df['season'] == selected_season]
    
    # Team selection
    available_teams = sorted(season_df['team'].unique())
    selected_team = st.sidebar.selectbox(
        "Select Team",
        available_teams
    )
    
    # Get team data
    team_df = season_df[season_df['team'] == selected_team].sort_values('week')
    
    # Statistic selection
    stat_columns = [col for col in df.columns if col.endswith('_ewma')]
    stat_names = [col.replace('_ewma', '').replace('_', ' ').title() for col in stat_columns]
    
    stat_display_to_col = dict(zip(stat_names, stat_columns))
    
    selected_stat_display = st.sidebar.selectbox(
        "Select Statistic",
        stat_names
    )
    
    selected_stat = stat_display_to_col[selected_stat_display]
    
    # Comparison team (optional)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Compare Teams")
    compare_enabled = st.sidebar.checkbox("Add comparison team")
    
    comparison_team = None
    if compare_enabled:
        other_teams = [t for t in available_teams if t != selected_team]
        comparison_team = st.sidebar.selectbox(
            "Compare with",
            other_teams
        )
        comparison_df = season_df[season_df['team'] == comparison_team].sort_values('week')
    
    # Main content
    st.markdown("---")
    
    # Display team info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Team", selected_team)
    with col2:
        st.metric("Season", selected_season)
    with col3:
        current_value = team_df[selected_stat].iloc[-1] if len(team_df) > 0 else 0
        st.metric("Current Value", f"{current_value:.2f}")
    with col4:
        if len(team_df) > 1:
            prev_value = team_df[selected_stat].iloc[-2]
            delta = current_value - prev_value
            st.metric("Change", f"{delta:+.2f}", delta=f"{delta:+.2f}")
        else:
            st.metric("Change", "N/A")
    
    st.markdown("---")
    
    # Create plot
    fig = go.Figure()
    
    # Add main team line
    fig.add_trace(go.Scatter(
        x=team_df['week'],
        y=team_df[selected_stat],
        mode='lines+markers',
        name=selected_team,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Week %{x}</b><br>' +
                      f'{selected_stat_display}: ' + '%{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add comparison team if enabled
    if compare_enabled and comparison_team:
        fig.add_trace(go.Scatter(
            x=comparison_df['week'],
            y=comparison_df[selected_stat],
            mode='lines+markers',
            name=comparison_team,
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>Week %{x}</b><br>' +
                          f'{selected_stat_display}: ' + '%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{selected_stat_display} Trend - {selected_team} ({selected_season})",
        xaxis_title="Week",
        yaxis_title=selected_stat_display,
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            dtick=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        )
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Statistics summary
    st.markdown("---")
    st.subheader("Season Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Statistics")
        summary_stats = pd.DataFrame({
            'Metric': ['Season Average', 'Maximum', 'Minimum', 'Std Deviation', 'Current (Latest Week)'],
            'Value': [
                f"{team_df[selected_stat].mean():.2f}",
                f"{team_df[selected_stat].max():.2f}",
                f"{team_df[selected_stat].min():.2f}",
                f"{team_df[selected_stat].std():.2f}",
                f"{team_df[selected_stat].iloc[-1]:.2f}" if len(team_df) > 0 else "N/A"
            ]
        })
        st.dataframe(summary_stats, width='content', hide_index=True)
    
    with col2:
        if compare_enabled and comparison_team:
            st.markdown("### Comparison")
            comparison_stats = pd.DataFrame({
                'Metric': ['Season Average', 'Maximum', 'Minimum', 'Std Deviation', 'Current (Latest Week)'],
                selected_team: [
                    f"{team_df[selected_stat].mean():.2f}",
                    f"{team_df[selected_stat].max():.2f}",
                    f"{team_df[selected_stat].min():.2f}",
                    f"{team_df[selected_stat].std():.2f}",
                    f"{team_df[selected_stat].iloc[-1]:.2f}" if len(team_df) > 0 else "N/A"
                ],
                comparison_team: [
                    f"{comparison_df[selected_stat].mean():.2f}",
                    f"{comparison_df[selected_stat].max():.2f}",
                    f"{comparison_df[selected_stat].min():.2f}",
                    f"{comparison_df[selected_stat].std():.2f}",
                    f"{comparison_df[selected_stat].iloc[-1]:.2f}" if len(comparison_df) > 0 else "N/A"
                ]
            })
            st.dataframe(comparison_stats, width='content', hide_index=True)
    
    # Detailed data table
    st.markdown("---")
    st.subheader("Week-by-Week Data")
    
    # Prepare display dataframe
    display_df = team_df[['week', 'opponent_team', selected_stat]].copy()
    display_df.columns = ['Week', 'Opponent', selected_stat_display]
    display_df[selected_stat_display] = display_df[selected_stat_display].round(2)
    
    st.dataframe(
        display_df.sort_values('Week', ascending=False),
        width='content',
        hide_index=True
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label=f"Download {selected_team} {selected_stat_display} Data",
        data=csv,
        file_name=f"{selected_team}_{selected_stat_display}_{selected_season}.csv",
        mime="text/csv"
    )
    
except FileNotFoundError:
    st.error("❌ Data file not found. Please make sure 'data/df_clean.csv' exists.")
    st.info("Run your data processing script first to generate the required data files.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    with st.expander("See full error details"):
        st.exception(e)

# Footer
st.markdown("---")
st.caption("📊 Data source: nflreadrpy | EWMA alpha = 0.4")