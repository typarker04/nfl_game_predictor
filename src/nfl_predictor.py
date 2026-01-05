import nflreadpy as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
schedule = nfl.load_schedules()
current_season = nfl.get_current_season()
current_week = nfl.get_current_week(False, kwargs=18)

week_games = schedule[(schedule['week'] == current_week) & (schedule['season'] == current_season)]


model_path = 'models/finalized_model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

#Create a dataframe with rows corresponding to the ewma stats for each team. Use this to index
# individual teams to plug into model
df = pd.read_csv('data/games_with_stats1.csv')
schedule = nfl.load_schedules(2025).to_pandas()


#loaded_model.predict(diff_df)

def load_matchup_df(team, week = 17):
    team_df = df[(df['season'] == 2025) & (df['week'] == week) & (df['team'] == team)]

    if len(team_df) == 0:
        print(f"No data available for {team} in week 17, 2025")
    else:
        opponent = team_df['opponent_team'].iloc[0]
        print(f"\nComparing {team} vs {opponent}")
        
    opponent_df = df[(df['season'] == 2025) & (df['week'] == 17) & (df['team'] == opponent)]
    if len(opponent_df) == 0:
        print(f"No data available for {opponent} in week 17")
    else:
        numeric_cols = team_df.select_dtypes(include=['float']).columns.tolist()
        diff_data = {}
        diff_data['team'] = team
        diff_data['opponent'] = opponent
        diff_data['season'] = 2025
        diff_data['week'] = 17

        for col in numeric_cols:
            team_val = team_df[col].iloc[0]
            opp_val = opponent_df[col].iloc[0]
            diff = team_val-opp_val
            diff_data[f'diff_{col}'] = diff

    diff_df = pd.DataFrame([diff_data])
    print("Showing game stat differences... \n")
    print(diff_df)

load_matchup_df('ATL', 17)

def load_next_matchup(team, current_date=None):
    if current_date is None:
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
    elif isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)
        formatted_date = current_date.strftime("%Y-%m-%d")
    
    df['game']