from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram(df: pd.DataFrame, variable: str, color: str | None = None):
    """Distribución de una variable numérica con boxplot marginal."""
    return px.histogram(
        df, x=variable, color=color, nbins=40, marginal="box",
        title=f"Distribución de {variable}",
    )


def cancellation_bar(summary: pd.DataFrame, column: str):
    """Barras de tasa de cancelación para una variable categórica."""
    return px.bar(
        summary, x=column, y="tasa_cancelacion", text="tasa_cancelacion",
        hover_data=["reservas"], title=f"Tasa de cancelación (%) por {column}",
    )


def monthly_chart(monthly: pd.DataFrame):
    """Reservas (barras) y tasa de cancelación (línea) por mes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["arrival_date_month"], y=monthly["reservas"], name="Reservas"))
    fig.add_trace(
        go.Scatter(x=monthly["arrival_date_month"], y=monthly["tasa_cancelacion"],
                   name="Tasa de cancelación (%)", mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(title="Estacionalidad: reservas y cancelación por mes")
    return fig


def adr_box(df: pd.DataFrame):
    """Dispersión de ADR por tipo de hotel, recortando el outlier extremo."""
    sub = df[df["adr"].between(0, df["adr"].quantile(0.995))]
    return px.box(sub, x="hotel", y="adr", color="is_canceled", title="ADR por hotel y estado de cancelación")


def scatter(df: pd.DataFrame, x: str, y: str, sample: int = 5000):
    """Dispersión sobre una muestra para mantener la interfaz fluida."""
    data = df.sample(min(sample, len(df)), random_state=42)
    return px.scatter(data, x=x, y=y, color="is_canceled", opacity=0.5, title=f"{y} vs. {x} (muestra)")


def correlation_heatmap(df: pd.DataFrame):
    corr = df.select_dtypes(include="number").corr().round(2)
    return px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación")


def coefficients_bar(coefficients: pd.DataFrame, top: int = 15):
    sub = coefficients.head(top).sort_values("coeficiente")
    return px.bar(sub, x="coeficiente", y="variable", orientation="h", title=f"Top {top} coeficientes del modelo")


def confusion_heatmap(confusion: pd.DataFrame):
    labels = ["No cancelada", "Cancelada"]
    fig = px.imshow(
        confusion.to_numpy(), x=labels, y=labels, text_auto=",d",
        labels={"x": "Predicho", "y": "Real"}, title="Matriz de confusión (real vs. predicho)",
    )
    return fig
