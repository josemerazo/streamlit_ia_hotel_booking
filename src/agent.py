from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from src import eda
from src.llm_client import OllamaClient

SYSTEM_PROMPT = (
    "Eres un asistente académico de análisis de datos de reservas hoteleras (Hotel Booking Demand). "
    "Responde únicamente con base en el contexto y los resultados de herramientas entregados; "
    "todos los cálculos ya fueron hechos por Python. Distingue hechos observados de hipótesis, "
    "no inventes cifras ni variables, y aclara que correlación no implica causalidad. "
    "Responde en español, breve y técnico."
)


class HotelAgent:
    """Agente que elige herramientas autorizadas, las ejecuta en Python y pide a Ollama interpretarlas."""

    def __init__(self, df: pd.DataFrame, client: OllamaClient, model_result: dict | None = None):
        self.df = df
        self.client = client
        self.model_result = model_result
        # Herramientas autorizadas: nombre -> (palabras clave, función)
        self.tools: dict[str, tuple[tuple[str, ...], Callable[[], pd.DataFrame]]] = {
            "cancelacion_por_deposito": (("depósito", "deposito", "deposit"), lambda: eda.cancellation_by(self.df, "deposit_type")),
            "cancelacion_por_segmento": (("segmento", "canal", "market"), lambda: eda.cancellation_by(self.df, "market_segment")),
            "cancelacion_por_hotel": (("hotel", "resort", "city"), lambda: eda.cancellation_by(self.df, "hotel")),
            "cancelacion_por_pais": (("país", "pais", "country"), lambda: eda.cancellation_by(self.df, "country", 500).head(10)),
            "cancelacion_por_anticipacion": (("anticipación", "anticipacion", "lead", "antelación"), lambda: eda.lead_time_bands(self.df)),
            "estacionalidad_mensual": (("mes", "temporada", "estacional", "mensual"), lambda: eda.monthly_stats(self.df)),
            "correlaciones": (("correlaci", "relación", "relacion", "variables"), lambda: eda.target_correlations(self.df).head(8)),
        }
        if model_result is not None:
            self.tools["coeficientes_modelo"] = (
                ("modelo", "métrica", "metrica", "coeficiente", "predic"),
                lambda: model_result["coefficients"].head(10),
            )

    def select_tools(self, question: str) -> list[str]:
        q = question.lower()
        chosen = [name for name, (keys, _) in self.tools.items() if any(k in q for k in keys)]
        return chosen or ["correlaciones", "cancelacion_por_deposito", "estacionalidad_mensual"]

    def run_tools(self, names: list[str]) -> dict:
        return {n: self.tools[n][1]().to_dict("records") for n in names if n in self.tools}

    def ask(self, question: str, context: dict) -> tuple[str, list[str], dict]:
        """Devuelve (respuesta, herramientas usadas, resultados calculados)."""
        used = self.select_tools(question)
        results = self.run_tools(used)
        user = (
            f"CONTEXTO GLOBAL:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            f"RESULTADOS DE HERRAMIENTAS ({', '.join(used)}):\n"
            f"{json.dumps(results, ensure_ascii=False, default=str)}\n\nPREGUNTA:\n{question}"
        )
        return self.client.chat(SYSTEM_PROMPT, user), used, results


def offline_summary(context: dict) -> str:
    d = context.get("dataset", {})
    m = context.get("model", {})
    text = (
        f"El dataset contiene {d.get('rows', 0)} reservas y {d.get('columns', 0)} variables; "
        f"la tasa de cancelación es {d.get('cancel_rate_pct')}%, con ADR medio de {d.get('adr_mean')}."
    )
    corr = context.get("top_correlations", [])
    if corr:
        text += f" La variable numérica más asociada a la cancelación es {corr[0]['variable']} (r={corr[0]['correlacion']})."
    if m:
        text += f" El modelo logístico obtuvo accuracy={m.get('accuracy')} y ROC-AUC={m.get('roc_auc')}."
    return text + " Salida descriptiva generada sin LLM."
