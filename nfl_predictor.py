import nflreadpy as nfl
import pandas as pd
import matplotlib as plt

team_stats = nfl.load_team_stats().to_pandas()

team_stats['turnovers_offense'] = (team_stats['passing_interceptions']+
                                 team_stats['sack_fumbles_lost']+
                                 team_stats['rushing_fumbles_lost']+
                                 team_stats['receiving_fumbles_lost']
                                 )
team_stats['turnovers_defense'] = (team_stats['def_interceptions']+
                                 team_stats['def_fumbles'])
team_stats['turnover_margin'] = (team_stats['turnovers_defense']-
                               team_stats['turnovers_offense'])
