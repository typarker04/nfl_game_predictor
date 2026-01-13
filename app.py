import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import nflreadpy as nfl

sys.path.append(str(Path(__file__).parent / 'src'))

from nfl_predictor import get_nfl_diffs

current_week = nfl.get_current_week()
st.set_page_config(
    page_title="NFL Game Predictions",
    page_icon="🏈",
    layout="wide"
)

st.title(f"Week {current_week} NFL Game Predictions")
st.markdown("### Win probability predictions for upcoming games")

with st.sidebar:
    st.header("Settings")
    
    st.markdown("---")
    st.markdown("**Model Info**")
    st.info("Last updated: [Add timestamp]")
    
    # Optional filters
    st.markdown("---")
    st.header("Filters")
    show_favorites = st.checkbox("Highlight favorites", value=True)

predictions_df = get_nfl_diffs()
df = st.dataframe(predictions_df)
#with df:


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Games", len(predictions_df))
with col2:
    avg_confidence = predictions_df['win_prob'].apply(lambda x: max(x, 1-x)).mean()
    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
with col3:
    favorites = (predictions_df['win_prob'] > 0.5).sum()
    st.metric("Home Favorites", favorites)

st.subheader("Win Probability by Game")

# Create the chart
fig = go.Figure()

# Add bars
colors = ['#d32f2f' if p < 0.5 else '#388e3c' for p in predictions_df['win_prob']]

fig.add_trace(go.Bar(
    x=predictions_df['win_prob'],
    y=predictions_df['game_id'],
    orientation='h',
    marker=dict(color=colors),
    text=predictions_df['win_prob'].apply(lambda x: f"{x:.1%}"),
    textposition='auto',
))

    # Add vertical line at 50%
fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

fig.update_layout(
    title="Home Team Win Probability",
    xaxis_title="Win Probability",
    yaxis_title="Matchup",
    height=max(400, len(predictions_df) * 40),
    showlegend=False,
    xaxis=dict(tickformat=".0%", range=[0, 1])
)

st.plotly_chart(fig, width='stretch')

csv = predictions_df.to_csv(index=False)
st.download_button(
    label="Download predictions as CSV",
    data=csv,
    file_name=f"nfl_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)