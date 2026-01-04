import nflreadpy as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime


model_path = 'models/finalized_model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

#Create a dataframe with rows corresponding to the ewma stats for each team. Use this to index
# individual teams to plug into model
df = pd.read_csv('data/df_clean.csv')
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