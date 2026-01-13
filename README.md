# NFL Game Predictor

A machine learning application that predicts NFL game outcomes using historical performance data and exponentially weighted moving averages (EWMA).

## Project Structure

```
nfl-predictions/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── __init__.py
│   └── nfl_predictor.py          # Prediction functions
│
├── data/                       # Data files
│   ├── df_clean.csv            # Processed team statistics
│   ├── games_with_stats.csv    # Games with merged statistics
│   └── most_recent_stats.csv   # Latest team statistics
│
├── models/                     # Saved models
│   ├── finalized_model.pkl     # Trained model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_list.pkl        # Selected features
│
└── outputs/                    # Generated outputs
|    ├── feature_importance.png
|    ├── predictions.png
|    └── latest_predictions.csv
|
└── notebooks/
    ├── nfl_predictor_notebook.ipynb # Where I messed around with the models/data
    └── nfl_predictor_organized.ipynb # Organized markdown notebook where model can get updates.
                                      # Integrates with Streamlit app
```

## Features

### Model Features
The model uses the following statistics (calculated as EWMA):
- **Passing**: Completions, yards, touchdowns, completion percentage
- **Rushing**: Yards, touchdowns
- **Defense**: Tackles for loss, turnovers forced
- **Turnovers**: Offensive turnovers, defensive takeaways, turnover margin
- **Special Teams**: Field goal percentage, PAT percentage
- **Other**: Sacks suffered, penalty yards

### Key Components
1. **Feature Engineering**: Creates difference features (Home - Away) for each statistic
2. **EWMA Calculation**: Uses exponentially weighted moving average (α=0.4) to emphasize recent performance
3. **Feature Selection**: Random Forest identifies top 10 most important features
4. **Prediction**: Logistic Regression trained on selected features

## Installation

### 1. Clone the repository
```bash
git clone <typarker04/nfl_game_predictor>
cd nfl-predictions
```

### 2. Create virtual environment
```bash
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the model on historical data:

```bash
predictor_organized.ipynb
```

This will:
- Load NFL data from 2021-2025 seasons
- Calculate EWMA features
- Train a logistic regression model
- Save the model, scaler, and features to `models/`
- Generate feature importance chart in `outputs/`

### 2. Make Predictions

#### Option A: Web App (Recommended)

Run the Streamlit app:

```bash
streamlit run app.py
```

Features:
- Interactive visualization of win probabilities
- Confidence filtering
- Detailed prediction table
- CSV download

#### Option B: Command Line

Run predictions from the Notebook:


This will:
- Load the trained model
- Get current week's games
- Make predictions
- Save results to `outputs/latest_predictions.csv`
- Generate visualization chart

## Model Performance

Typical performance metrics:
- **Training Accuracy**: ~80-82%
- **Testing Accuracy**: ~80-82%

The model achieves consistent performance across training and test sets, indicating good generalization.

## Top Features (by importance)

1. Completion Percentage Differential
2. Passing TDs Differential
3. Rushing TDs Differential
4. Turnover Margin Differential
5. Turnovers Offense Differential
6. Rushing Yards Differential
7. Sacks Suffered Differential
8. Turnovers Defense Differential
9. Passing Yards Differential
10. Defensive Tackles for Loss Differential

## Data Sources

- **nflreadrpy**: Python library for accessing NFL play-by-play data
- Seasons: 2021-2025 (regular season only)

## Customization

### Adjust EWMA Alpha

In `notebooks/predictor_organized.ipynb - Train Final Model`

```python
x.ewm(alpha=0.4, adjust=False).mean()  # Change 0.4 to desired value
```

### Change Features

In `notebooks/predictor_organized.ipynb - EWMA Features`, modify `independent_variables`:

```python
independent_variables = [
    'your_feature_1',
    'your_feature_2',
    # ...
]
```

### Adjust Model Parameters

In `notebooks/predictor_organized.ipynb - Train Final Model`, modify the model initialization:

```python
model = LogisticRegression(
    random_state=41,
    max_iter=1000,
    C=1.0,  # Add regularization parameter
    # ...
)
```

## Troubleshooting

### "No module named 'nflreadpy'"
```bash
pip install nflreadpy
```

### "FileNotFoundError: models/finalized_model.pkl"
Run `python train_model.py` first to train and save the model.

### "No games found for current week"
The model looks for unplayed games in the current NFL week. If all games are complete, it will show this message.

### Scaler is a list error
Make sure you're using the correct loading method in your code. The updated `model_training.py` saves/loads the scaler correctly.

## Future Improvements

- [ ] Add more features (weather, injuries, home field advantage)
- [ ] Implement ensemble methods
- [ ] Add historical accuracy tracking
- [ ] Include betting line comparisons
- [ ] Add player-level statistics

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Contact

[Tyler Parker](trparker@wisc.edu.com)


Project Link: [https://github.com/typarker04/nfl-predictions](https://github.com/typarker04/nfl-predictions)

