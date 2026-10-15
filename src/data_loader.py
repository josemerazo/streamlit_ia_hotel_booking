from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "hotel_bookings.csv"
LEGACY_CSV = BASE_DIR / "hotel_bookings.csv"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
REQUIRED_COLUMNS = {"hotel", "is_canceled", "lead_time", "adr", "arrival_date_month"}


def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nulos, tipos y agrega variables derivadas."""
    out = df.copy()
    out = out.replace("NULL", np.nan)

    for col in ("children", "agent", "company"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["children"] = out["children"].fillna(0)
    out["country"] = out["country"].fillna("Desconocido")

    out["arrival_date_month"] = pd.Categorical(
        out["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["total_guests"] = out["adults"] + out["children"] + out["babies"]
    out["estimated_revenue"] = out["adr"] * out["total_nights"]
    return out


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no tiene las columnas requeridas: {sorted(missing)}")


def load_default_data() -> pd.DataFrame:
    """Carga el CSV de Kaggle desde data/ (o la raíz del proyecto como respaldo)."""
    path = DEFAULT_CSV if DEFAULT_CSV.exists() else LEGACY_CSV
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró hotel_bookings.csv. Descárgalo de Kaggle y colócalo en data/."
        )
    df = pd.read_csv(path)
    validate_columns(df)
    return clean_bookings(df)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Carga un CSV con el esquema de Hotel Booking Demand enviado desde Streamlit."""
    df = pd.read_csv(uploaded_file)
    validate_columns(df)
    return clean_bookings(df)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Resume dimensiones, nulos, duplicados y tasa de cancelación."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "cancel_rate": round(float(df["is_canceled"].mean() * 100), 2),
    }
