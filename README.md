# NFL Game Prediction Project

This repository contains a data science / machine learning project focused on predicting outcomes of upcoming NFL games using historical game data. 
The core idea is to engineer informative team-level features from past games, with a weighting scheme that emphasizes more recent performance.

## Project Overview

NFL teams change significantly over the course of a season due to imjuries, roster changes, coaching decisions, and form. This project aims to capture those
disparities and create an up to date game predictor based on many of these changes. This project achieves this by: 
- Aggregating historical NFL data at the team level
- Engineer features based on performance (offense, defense, efficiency, etc)
- Applies recency weighted averaging to more accurately capture changes throughout a season
- Use these features to train a logistic regression model to predict upcoming games

The goal is to generate data-driven predictions that change as the season progresses.

## Data

This project utilizes the publicly available python package _nflreadpy_. Game data is transformed into team-centric features, allowing each upcoming 
matchup to be represented as a comparison between two teams’ recent performance profiles.


