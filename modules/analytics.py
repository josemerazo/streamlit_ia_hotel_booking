from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isna().sum().values,
        "porcentaje": (df.isna().mean().values * 100).round(2),
    })
    return report.sort_values(["nulos", "variable"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T.reset_index().rename(columns={"index": "variable"})
    return out.round(3)


def target_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    return (
        corr.rename("correlacion")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def train_regression_demo(df: pd.DataFrame, target: str) -> dict:
    numeric = df.select_dtypes(include="number").dropna().copy()
    if target not in numeric.columns:
        raise ValueError("La variable objetivo debe ser numérica.")

    X = numeric.drop(columns=[target])
    y = numeric[target]
    if X.shape[1] == 0 or len(numeric) < 20:
        raise ValueError("No hay suficientes variables o registros para entrenar el modelo.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    coefficients = pd.DataFrame({
        "variable": X.columns,
        "coeficiente": model.coef_,
    }).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)

    predictions = pd.DataFrame({
        "real": y_test.to_numpy(),
        "predicho": pred,
    }).reset_index(drop=True)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
        "features": list(X.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def build_agent_context(df: pd.DataFrame, target: str, model_result: dict | None = None) -> dict:
    numeric = df.select_dtypes(include="number")
    context = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "target": target,
        },
        "target_summary": {},
        "top_correlations": [],
    }

    if target in numeric.columns:
        s = numeric[target].dropna()
        context["target_summary"] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        }
        context["top_correlations"] = target_correlations(df, target).head(5).round(3).to_dict("records")

    if model_result:
        context["model"] = {
            "type": "LinearRegression",
            "r2": round(model_result["metrics"]["r2"], 3),
            "mae": round(model_result["metrics"]["mae"], 3),
            "rmse": round(model_result["metrics"]["rmse"], 3),
            "n_train": model_result["n_train"],
            "n_test": model_result["n_test"],
        }
    return context
