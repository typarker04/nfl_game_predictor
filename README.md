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

## How To Use

This project gets updated weekly with the most recent NFL stats, ensuring the most up-to-date estimations. To see upcoming games, you can either run the nfl_predictor.py file, or inspect the csv file _upcoming_diffs.py_.

## Repository Structure

├── data/ # Raw and/or processed game data

├── notebooks/ # Exploratory analysis and prototyping

├── src/ # Feature engineering and modeling code

├── models/ # Saved models and artifacts

├── results/ # Evaluation outputs and predictions

└── README.md # Project documentation

