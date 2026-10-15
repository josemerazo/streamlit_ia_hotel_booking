from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_canceled"

# reservation_status y su fecha revelan el resultado: se excluyen para evitar fuga de datos.
MODEL_NUMERIC = [
    "lead_time", "adults", "children", "babies", "total_nights",
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
]
MODEL_CATEGORICAL = ["hotel", "deposit_type", "customer_type", "market_segment"]


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
    return numeric.describe().T.reset_index().rename(columns={"index": "variable"}).round(3)


def cancellation_by(df: pd.DataFrame, column: str, min_count: int = 30) -> pd.DataFrame:
    """Tasa de cancelación (%) por categoría, ignorando categorías con pocas reservas."""
    grouped = (
        df.groupby(column, observed=True)[TARGET]
        .agg(reservas="count", cancelaciones="sum", tasa_cancelacion="mean")
        .reset_index()
    )
    grouped["tasa_cancelacion"] = (grouped["tasa_cancelacion"] * 100).round(2)
    grouped = grouped[grouped["reservas"] >= min_count]
    return grouped.sort_values("tasa_cancelacion", ascending=False).reset_index(drop=True)


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Reservas, tasa de cancelación y ADR promedio por mes de llegada."""
    out = (
        df.groupby("arrival_date_month", observed=True)
        .agg(reservas=(TARGET, "count"), tasa_cancelacion=(TARGET, "mean"), adr_promedio=("adr", "mean"))
        .reset_index()
    )
    out["tasa_cancelacion"] = (out["tasa_cancelacion"] * 100).round(2)
    out["adr_promedio"] = out["adr_promedio"].round(2)
    return out


def lead_time_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Segmenta lead_time en rangos y calcula la tasa de cancelación de cada uno."""
    bins = [-1, 7, 30, 90, 180, 365, np.inf]
    labels = ["0-7 días", "8-30", "31-90", "91-180", "181-365", ">365"]
    banded = df.assign(rango_anticipacion=pd.cut(df["lead_time"], bins=bins, labels=labels))
    return cancellation_by(banded, "rango_anticipacion", min_count=1).sort_values("rango_anticipacion").reset_index(drop=True)


def target_correlations(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])
    corr = numeric.corr()[target].drop(labels=[target], errors="ignore").dropna()
    return (
        corr.rename("correlacion").round(3)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def train_cancellation_model(df: pd.DataFrame) -> dict:
    """Regresión logística para predecir cancelación; interpretable y rápida."""
    cols = MODEL_NUMERIC + MODEL_CATEGORICAL
    data = df[cols + [TARGET]].dropna()
    if len(data) < 100:
        raise ValueError("No hay suficientes registros para entrenar el modelo.")

    X, y = data[cols], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pre = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefficients = (
        pd.DataFrame({"variable": names, "coeficiente": pipe.named_steps["clf"].coef_[0]})
        .assign(odds_ratio=lambda d: np.exp(d["coeficiente"]).round(3))
        .sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    coefficients["coeficiente"] = coefficients["coeficiente"].round(3)
    confusion = pd.crosstab(
        y_test.rename("real"), pd.Series(pred, index=y_test.index, name="predicho")
    )
    return {
        "model": pipe,
        "metrics": metrics,
        "coefficients": coefficients,
        "confusion": confusion,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def build_agent_context(df: pd.DataFrame, model_result: dict | None = None) -> dict:
    """Contexto base calculado en Python que recibe el agente."""
    context = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "target": TARGET,
            "cancel_rate_pct": round(float(df[TARGET].mean() * 100), 2),
            "adr_mean": round(float(df["adr"].mean()), 2),
            "lead_time_mean": round(float(df["lead_time"].mean()), 1),
        },
        "top_correlations": target_correlations(df).head(5).to_dict("records"),
    }
    if model_result:
        context["model"] = {
            "type": "LogisticRegression",
            **{k: round(v, 3) for k, v in model_result["metrics"].items()},
            "n_train": model_result["n_train"],
            "n_test": model_result["n_test"],
        }
    return context
