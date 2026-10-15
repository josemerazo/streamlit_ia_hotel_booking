from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "diabetes_sklearn.csv"


def load_default_data() -> pd.DataFrame:
    """Carga el CSV incluido en el proyecto."""
    return pd.read_csv(DEFAULT_CSV)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Carga un CSV enviado desde Streamlit."""
    return pd.read_csv(uploaded_file)


def dataset_profile(df: pd.DataFrame) -> dict:
    """Resume dimensiones, duplicados, nulos y variables numéricas."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
    }
