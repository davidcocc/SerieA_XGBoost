# Serie A XGBoost Match Predictor

> ⚠️ **Learning project only, to get familiar with XGBoost and FastAPI. DON'T use the predictions produced here for betting decisions.**

This repository explores end‑to‑end match forecasting using Serie A data, XGBoost regressors and a FastAPI-powered web interface. A lightweight model predicts goals scored and conceded by a selected team in an upcoming match; the UI provides a “matchboard” for quick simulations and an “analytics” tab with league-wide metrics.

---

## 🏗️ Project Structure

```
.
├─ matches_seriea.csv          # Source dataset (Serie A fixtures with stats)
├─ serieA.py                   # Training script (builds features + fits/saves models)
├─ app/
│  ├─ predictor.py             # PredictorService: feature prep + inference helpers
│  └─ main.py                  # FastAPI app serving UI + JSON endpoints
└─ badges/                     # Club crest PNGs used by the frontend
```

## 📦 Requirements

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows PowerShell
pip install -U pip
pip install pandas numpy scikit-learn xgboost fastapi "uvicorn[standard]" joblib matplotlib seaborn
```

---

## 🔁 Training the Models

1. Ensure `matches_seriea.csv` is present at the project root.
2. Run the training script (this also saves the encoders/models for inference):

```bash
python serieA.py
```

Outputs produced in the same directory:

- `preprocessor.pkl` – fitted `ColumnTransformer` (categoricals + numerics)
- `model_gf.pkl` – XGBoost regressor for goals scored
- `model_ga.pkl` – XGBoost regressor for goals conceded

You should re-run this script whenever you update the dataset or tweak feature engineering.

---

## 🚀 Running the Web App

With the models generated, start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` in your browser to explore the UI:

- **Matchboard**
  - Select two clubs, formations and venue.
  - View current form dots (last five fixtures) and head‑to‑head bar (last five meetings).
  - Press **Kick Off** to obtain projected scoreline and outcome.
- **Analytics**
  - Scatter plot of average xG created vs goals scored (hover tooltips include badges).
  - Full league table ordered by goal difference with win percentages.

REST endpoints available for automation/testing:

| Method | Path                 | Description                               |
| ------ | -------------------- | ----------------------------------------- |
| GET    | `/stats`             | League summary (per-team aggregates).     |
| GET    | `/recent/{team}`     | Last 5 results for the specified club.    |
| GET    | `/head-to-head`      | Query params `team`, `opponent`; returns recent meetings. |
| POST   | `/predict`           | JSON body `{team, opponent, venue, formation, oppFormation}`. |

All responses are JSON encoded.

---

## 🧠 Feature Engineering Highlights

- Exponentially weighted rolling means per team (`gf`, `ga`, `xg`, `xga`).
- Matchup-specific form (same metrics but scoped to the opponent).
- Fallback averages when historical data is missing.
- Two separate regressors (`gf` / `ga`) combined to derive W/D/L.
- Pydantic models ensure input validation on the API.

---

## 🛡️ Disclaimer

- The dataset and models are used purely for educational experimentation.
- Predictions are not guaranteed to be accurate and **must never** be used for gambling or financial decisions.
- Always cite the data source (Serie A match dataset) if you redistribute results.

---

## 🖋️ Credits

- Dataset: https://www.kaggle.com/datasets/marcelbiezunski/serie-a-matches-dataset-2020-2025/data
- Crests: https://github.com/luukhopman/football-logos

