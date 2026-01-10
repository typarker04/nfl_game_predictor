import nflreadpy as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def get_nfl_diffs():
    # 1. Load your cleaned stats and model
    most_recent_stats = pd.read_csv('data/most_recent_stats.csv')
    schedule = nfl.load_schedules(2025).to_pandas()
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()

    # 2. Identify current week/season

    week_games = schedule[(schedule['week'] == current_week) & 
                          (schedule['season'] == current_season)].copy()

    # 3. Define the stats we want to compare (based on your columns)
    keep_cols_old = [
        'team', 
        'passing_yards_ewma',
        'passing_tds_ewma',
        'rushing_yards_ewma',
        'sacks_suffered_ewma',
        'rushing_tds_ewma',
        'completion_pct_ewma',
        'turnovers_offense_ewma',
        'turnovers_defense_ewma',
        'turnover_margin_ewma',
        'def_tackles_for_loss_ewma'
    ]
    keep_cols = [
        'team',
        'completion_pct_ewma',
        'passing_tds_ewma',
        'rushing_tds_ewma',
        'turnover_margin_ewma',
        'turnovers_offense_ewma',
        'rushing_yards_ewma',
        'sacks_suffered_ewma',
        'turnovers_defense_ewma',
        'passing_yards_ewma',
        'def_tackles_for_loss_ewma']

    # 4. Merge Home Team Stats
    df_matchups = pd.merge(
        week_games[['game_id', 'home_team', 'away_team']],
        most_recent_stats[keep_cols],
        left_on='home_team',
        right_on='team',
        how='left'
    ).rename(columns={col: f"{col}_home" for col in keep_cols if col != 'team'})

    # 5. Merge Away Team Stats
    df_matchups = pd.merge(
        df_matchups,
        most_recent_stats[keep_cols],
        left_on='away_team',
        right_on='team',
        how='left'
    ).rename(columns={col: f"{col}_away" for col in keep_cols if col != 'team'})

    # 6. Calculate Differences (Home - Away)
    stat_names = [col.replace('_home', '') for col in df_matchups.columns if col.endswith('_home')]
    
    for stat in stat_names:
        df_matchups[f"{stat}_diff"] = df_matchups[f"{stat}_home"] - df_matchups[f"{stat}_away"]

    # 7. Final Cleanup: Keep only IDs and Diffs
    diff_cols = [c for c in df_matchups.columns if c.endswith('_diff')]
    final_df = df_matchups[['game_id', 'home_team', 'away_team'] + diff_cols]

    #8. Run each game through the model.

    model = joblib.load('models/finalized_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    model_cols = [c for c in final_df.columns if c.endswith("_diff")]
    X_scaled = scaler.transform(final_df[model_cols])

    final_df['win_prob'] = model.predict_proba(X_scaled)[:,1]
    final_df = final_df.sort_values(by='win_prob', ascending=False)

    #9. Visualize results.
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(final_df)), final_df['win_prob'])
    plt.yticks(range(len(final_df)), final_df['game_id'].to_list())
    plt.xlabel("Home Win Prob")
    plt.ylabel("This week's games")
    plt.savefig('outputs/wild_card_probs', dpi=300, bbox_inches='tight')
    plt.show()
    
    return final_df

if __name__ == "__main__":
    result = get_nfl_diffs()
    print("--- Upcoming NFL Game Stat Differences ---")
    print(result)#.to_string(index=False))
    result.to_csv('data/upcoming_diffs.csv')

