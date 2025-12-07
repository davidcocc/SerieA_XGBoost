import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import seaborn as sns
import joblib

data = pd.read_csv('matches_seriea.csv')
df = pd.DataFrame(data)
df.drop(columns = ['Unnamed: 0', 'result','time', 'comp', 'round', 'day', 'referee', 'attendance', 'match report', 'notes', 'season', 'captain', 'poss', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt'], inplace = True)

df['date'] = pd.to_datetime(df['date'])

df.sort_values(['team', 'date'], inplace=True)

for col in ['gf', 'ga', 'xg', 'xga']:
    df[f'{col}_ewm_team'] = (
        df.groupby('team')[col]
        .shift()  # esclude la partita corrente
        .ewm(span=5, min_periods=1, adjust=False)
        .mean()
    )

df.sort_values(['team', 'opponent', 'date'], inplace=True)

for col in ['gf', 'ga', 'xg', 'xga']:
    df[f'{col}_ewm_matchup'] = (
        df.groupby(['team', 'opponent'])[col]
        .shift()
        .ewm(alpha=0.6, min_periods=1, adjust=False)
        .mean()
    )

# managing NaN values through mean imputation or filling with 0 if no data is available

ewm_cols = [c for c in df.columns if c.endswith('_ewm_team') or c.endswith('_ewm_matchup')]
df = df.sort_values(['team', 'date'])

for col in ewm_cols:
    team_mean = df.groupby('team')[col].transform('mean')
    df[col] = df[col].fillna(team_mean)

for col in ewm_cols:
    df[col] = df[col].fillna(df[col].mean())

df[ewm_cols] = df[ewm_cols].fillna(0.0)

df.rename(columns = {'gf_ewm_team': 'gm_form', 'ga_ewm_team': 'ga_form', 'xg_ewm_team': 'xg_form', 'xga_ewm_team': 'xga_form', 'gf_ewm_matchup': 'gf_vs_opp', 'ga_ewm_matchup': 'ga_vs_opp', 'xg_ewm_matchup': 'xg_vs_opp', 'xga_ewm_matchup': 'xga_vs_opp'}, inplace = True)
print(df.head())

feature_cols = [
    'venue', 'team', 'opponent', 'formation', 'opp formation',
    'gm_form', 'ga_form', 'xg_form', 'xga_form',
    'gf_vs_opp', 'ga_vs_opp', 'xg_vs_opp', 'xga_vs_opp'
]

# features and target variables
X = df[feature_cols]
y_gf = df['gf']
y_ga = df['ga']

X_train, X_test, y_gf_train, y_gf_test, y_ga_train, y_ga_test = train_test_split(
    X,
    y_gf,
    y_ga,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)
categorical_cols = ['venue', 'team', 'opponent', 'formation', 'opp formation']
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

preprocessor = ColumnTransformer(
    [
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('numerical', 'passthrough', numerical_cols),
    ])

params = dict(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)

# Fit the preprocessor on training data and transform both train/test sets
preprocessor.fit(X_train)
X_train_enc = preprocessor.transform(X_train)
X_test_enc = preprocessor.transform(X_test)

model_gf = XGBRegressor(**params)
model_ga = XGBRegressor(**params)

model_gf.fit(X_train_enc, y_gf_train, verbose=True)
model_ga.fit(X_train_enc, y_ga_train, verbose=True)

preds_gf = model_gf.predict(X_test_enc)
preds_ga = model_ga.predict(X_test_enc)

mae_gf = mean_absolute_error(y_gf_test, preds_gf)
rmse_gf = np.sqrt(mean_squared_error(y_gf_test, preds_gf))
mae_ga = mean_absolute_error(y_ga_test, preds_ga)
rmse_ga = np.sqrt(mean_squared_error(y_ga_test, preds_ga))

print(f"GF MAE: {mae_gf:.2f}, RMSE: {rmse_gf:.2f}")
print(f"GA MAE: {mae_ga:.2f}, RMSE: {rmse_ga:.2f}")

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model_gf, "model_gf.pkl")
joblib.dump(model_ga, "model_ga.pkl")

team_form_cols = ['gm_form', 'ga_form', 'xg_form', 'xga_form']
matchup_form_cols = ['gf_vs_opp', 'ga_vs_opp', 'xg_vs_opp', 'xga_vs_opp']
default_forms = (
    df[team_form_cols + matchup_form_cols]
    .mean()
    .fillna(0.0)
    .to_dict()
)

def _get_team_form(team: str) -> dict:
    history = df[df['team'] == team].sort_values('date')
    if history.empty:
        return {col: default_forms.get(col, 0.0) for col in team_form_cols}
    latest = history.iloc[-1]
    return {
        col: float(latest.get(col, default_forms.get(col, 0.0)))
        for col in team_form_cols
    }

def _get_matchup_form(team: str, opponent: str) -> dict:
    history = (
        df[(df['team'] == team) & (df['opponent'] == opponent)]
        .sort_values('date')
    )
    if history.empty:
        team_form = _get_team_form(team)
        return {
            'gf_vs_opp': team_form['gm_form'],
            'ga_vs_opp': team_form['ga_form'],
            'xg_vs_opp': team_form['xg_form'],
            'xga_vs_opp': team_form['xga_form'],
        }
    latest = history.iloc[-1]
    return {
        col: float(latest.get(col, default_forms.get(col, 0.0)))
        for col in matchup_form_cols
    }

def _build_feature_row(team: str, opponent: str, venue: str,
                    formation: str, opp_formation: str) -> pd.DataFrame:
    team_form = _get_team_form(team)
    matchup_form = _get_matchup_form(team, opponent)

    row = {
        'venue': venue,
        'team': team,
        'opponent': opponent,
        'formation': formation,
        'opp formation': opp_formation,
        **team_form,
        **matchup_form,
    }
    return pd.DataFrame([row], columns=feature_cols)


def predict_match(team: str, opponent: str, venue: str,
                formation: str, opp_formation: str) -> dict:
    feature_row = _build_feature_row(team, opponent, venue, formation, opp_formation)
    feature_enc = preprocessor.transform(feature_row)

    pred_gf = float(model_gf.predict(feature_enc)[0])
    pred_ga = float(model_ga.predict(feature_enc)[0])

    pred_gf_round = int(np.clip(np.round(pred_gf), 0, 15))
    pred_ga_round = int(np.clip(np.round(pred_ga), 0, 15))

    if pred_gf_round > pred_ga_round:
        result = 'W'
    elif pred_gf_round < pred_ga_round:
        result = 'L'
    else:
        result = 'D'

    return {
        'team': team,
        'opponent': opponent,
        'venue': venue,
        'formation': formation,
        'opp_formation': opp_formation,
        'pred_goals_for': pred_gf,
        'pred_goals_against': pred_ga,
        'pred_goals_for_round': pred_gf_round,
        'pred_goals_against_round': pred_ga_round,
        'result': result,
    }


sample_matches = [
    {
        'team': 'Inter',
        'opponent': 'Benevento',
        'venue': 'Home',
        'formation': '3-5-2',
        'opp_formation': '4-1-4-1',
    },
    {
        'team': 'Inter',
        'opponent': 'Parma',
        'venue': 'Away',
        'formation': '3-5-2',
        'opp_formation': '3-4-3',
    },
    {
        'team': 'Salernitana',
        'opponent': 'Napoli',
        'venue': 'Away',
        'formation': '4-3-3',
        'opp_formation': '3-4-1-2',
    },
]

simulated_predictions = [predict_match(**match) for match in sample_matches]
simulation_df = pd.DataFrame(simulated_predictions)
print("\nSimulazioni partite fittizie:")
print(simulation_df[['team', 'opponent', 'venue',
                    'pred_goals_for_round', 'pred_goals_against_round', 'result']])