from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "matches_seriea.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "preprocessor.pkl"
MODEL_GF_PATH = PROJECT_ROOT / "model_gf.pkl"
MODEL_GA_PATH = PROJECT_ROOT / "model_ga.pkl"


FEATURE_COLS = [
    "venue",
    "team",
    "opponent",
    "formation",
    "opp formation",
    "gm_form",
    "ga_form",
    "xg_form",
    "xga_form",
    "gf_vs_opp",
    "ga_vs_opp",
    "xg_vs_opp",
    "xga_vs_opp",
]

TEAM_FORM_COLS = ["gm_form", "ga_form", "xg_form", "xga_form"]
MATCHUP_FORM_COLS = ["gf_vs_opp", "ga_vs_opp", "xg_vs_opp", "xga_vs_opp"]


class PredictorService:
    """Encapsulates data preparation and inference utilities for match prediction."""

    def __init__(self) -> None:
        self.df = self._prepare_dataset()
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.model_gf = joblib.load(MODEL_GF_PATH)
        self.model_ga = joblib.load(MODEL_GA_PATH)

        self.default_forms = (
            self.df[TEAM_FORM_COLS + MATCHUP_FORM_COLS]
            .mean()
            .fillna(0.0)
            .to_dict()
        )

    def _prepare_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_PATH)
        df = df.drop(
            columns=[
                "Unnamed: 0",
                "result",
                "time",
                "comp",
                "round",
                "day",
                "referee",
                "attendance",
                "match report",
                "notes",
                "season",
                "captain",
                "poss",
                "sh",
                "sot",
                "dist",
                "fk",
                "pk",
                "pkatt",
            ],
            errors="ignore",
        )

        df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values(["team", "date"])
        for col in ["gf", "ga", "xg", "xga"]:
            df[f"{col}_ewm_team"] = (
                df.groupby("team")[col]
                .shift()
                .ewm(span=5, min_periods=1, adjust=False)
                .mean()
            )

        df = df.sort_values(["team", "opponent", "date"])
        for col in ["gf", "ga", "xg", "xga"]:
            df[f"{col}_ewm_matchup"] = (
                df.groupby(["team", "opponent"])[col]
                .shift()
                .ewm(alpha=0.6, min_periods=1, adjust=False)
                .mean()
            )

        ewm_cols = [
            c
            for c in df.columns
            if c.endswith("_ewm_team") or c.endswith("_ewm_matchup")
        ]

        df = df.sort_values(["team", "date"])
        for col in ewm_cols:
            team_mean = df.groupby("team")[col].transform("mean")
            df[col] = df[col].fillna(team_mean)
            df[col] = df[col].fillna(df[col].mean())
        df[ewm_cols] = df[ewm_cols].fillna(0.0)

        df = df.rename(
            columns={
                "gf_ewm_team": "gm_form",
                "ga_ewm_team": "ga_form",
                "xg_ewm_team": "xg_form",
                "xga_ewm_team": "xga_form",
                "gf_ewm_matchup": "gf_vs_opp",
                "ga_ewm_matchup": "ga_vs_opp",
                "xg_ewm_matchup": "xg_vs_opp",
                "xga_ewm_matchup": "xga_vs_opp",
            }
        )
        return df

    def _get_team_form(self, team: str) -> Dict[str, float]:
        history = self.df[self.df["team"] == team].sort_values("date")
        if history.empty:
            return {col: self.default_forms.get(col, 0.0) for col in TEAM_FORM_COLS}
        latest = history.iloc[-1]
        return {
            col: float(latest.get(col, self.default_forms.get(col, 0.0)))
            for col in TEAM_FORM_COLS
        }

    def _get_matchup_form(self, team: str, opponent: str) -> Dict[str, float]:
        history = (
            self.df[(self.df["team"] == team) & (self.df["opponent"] == opponent)]
            .sort_values("date")
        )
        if history.empty:
            team_form = self._get_team_form(team)
            return {
                "gf_vs_opp": team_form["gm_form"],
                "ga_vs_opp": team_form["ga_form"],
                "xg_vs_opp": team_form["xg_form"],
                "xga_vs_opp": team_form["xga_form"],
            }
        latest = history.iloc[-1]
        return {
            col: float(latest.get(col, self.default_forms.get(col, 0.0)))
            for col in MATCHUP_FORM_COLS
        }

    def _build_feature_row(
        self,
        team: str,
        opponent: str,
        venue: str,
        formation: str,
        opp_formation: str,
    ) -> pd.DataFrame:
        team_form = self._get_team_form(team)
        matchup_form = self._get_matchup_form(team, opponent)

        row = {
            "venue": venue,
            "team": team,
            "opponent": opponent,
            "formation": formation,
            "opp formation": opp_formation,
            **team_form,
            **matchup_form,
        }
        return pd.DataFrame([row], columns=FEATURE_COLS)

    def predict(
        self,
        team: str,
        opponent: str,
        venue: str,
        formation: str,
        opp_formation: str,
    ) -> Dict[str, float | str]:
        feature_row = self._build_feature_row(team, opponent, venue, formation, opp_formation)
        feature_enc = self.preprocessor.transform(feature_row)

        pred_gf = float(self.model_gf.predict(feature_enc)[0])
        pred_ga = float(self.model_ga.predict(feature_enc)[0])

        pred_gf_round = int(np.clip(np.round(pred_gf), 0, 15))
        pred_ga_round = int(np.clip(np.round(pred_ga), 0, 15))

        if pred_gf_round > pred_ga_round:
            result = "W"
        elif pred_gf_round < pred_ga_round:
            result = "L"
        else:
            result = "D"

        return {
            "team": team,
            "opponent": opponent,
            "venue": venue,
            "formation": formation,
            "opp_formation": opp_formation,
            "pred_goals_for": pred_gf,
            "pred_goals_against": pred_ga,
            "pred_goals_for_round": pred_gf_round,
            "pred_goals_against_round": pred_ga_round,
            "result": result,
        }

    def list_teams(self) -> List[str]:
        return sorted(self.df["team"].unique())

    def list_formations(self) -> List[str]:
        team_forms = set(self.df["formation"].dropna().unique())
        opp_forms = set(self.df["opp formation"].dropna().unique())
        return sorted(team_forms.union(opp_forms))

    def team_summary(self) -> List[Dict[str, float | int | str]]:
        summary_df = self.df.copy()
        summary_df["win"] = (summary_df["gf"] > summary_df["ga"]).astype(int)
        summary_df["draw"] = (summary_df["gf"] == summary_df["ga"]).astype(int)
        grouped = (
            summary_df.groupby("team")
            .agg(
                matches=("gf", "count"),
                avg_gf=("gf", "mean"),
                avg_ga=("ga", "mean"),
                avg_xg=("xg", "mean"),
                total_gf=("gf", "sum"),
                total_ga=("ga", "sum"),
                win_rate=("win", "mean"),
                draw_rate=("draw", "mean"),
            )
            .reset_index()
        )
        grouped["win_rate"] = grouped["win_rate"] * 100.0
        grouped["draw_rate"] = grouped["draw_rate"] * 100.0
        grouped["goal_diff"] = grouped["total_gf"] - grouped["total_ga"]

        return grouped.to_dict(orient="records")

    def recent_results(self, team: str, limit: int = 5) -> List[str]:
        history = (
            self.df[self.df["team"] == team]
            .sort_values("date", ascending=False)
            .head(limit)
        )
        results: List[str] = []
        for _, row in history.iterrows():
            if row["gf"] > row["ga"]:
                results.append("W")
            elif row["gf"] < row["ga"]:
                results.append("L")
            else:
                results.append("D")
        return results

    def head_to_head(
        self, team: str, opponent: str, limit: int = 5
    ) -> List[Dict[str, float | str]]:
        mask = (
            (self.df["team"] == team) & (self.df["opponent"] == opponent)
        ) | ((self.df["team"] == opponent) & (self.df["opponent"] == team))
        history = (
            self.df[mask]
            .sort_values("date", ascending=False)
            .head(limit)
        )

        records: List[Dict[str, float | str]] = []
        for _, row in history.iterrows():
            primary_team = row["team"]
            gf = row["gf"]
            ga = row["ga"]
            if primary_team == team:
                if gf > ga:
                    result = "W"
                elif gf < ga:
                    result = "L"
                else:
                    result = "D"
            else:
                if gf < ga:
                    result = "W"
                elif gf > ga:
                    result = "L"
                else:
                    result = "D"
            records.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "team": row["team"],
                    "opponent": row["opponent"],
                    "gf": float(gf),
                    "ga": float(ga),
                    "result": result,
                }
            )
        return records


