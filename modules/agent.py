from __future__ import annotations
import json
import requests

OLLAMA_URL = "http://localhost:11434"


class LocalDataAgent:
    """Agente simple que envía un contexto analítico resumido a Ollama."""

    def __init__(self, model: str = "llama3.2:3b", base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def status(self) -> tuple[bool, list[str], str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            return True, models, "OK"
        except Exception as exc:
            return False, [], str(exc)

    def ask(self, question: str, context: dict) -> str:
        system = (
            "Eres un asistente académico de análisis de datos. "
            "Responde únicamente con base en el contexto estadístico entregado. "
            "Distingue hechos observados de hipótesis. No inventes variables ni resultados. "
            "El dataset de demostración Diabetes de Scikit-learn se usa solo con fines educativos; "
            "no emitas diagnósticos ni recomendaciones clínicas. Responde en español, breve y técnico."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"CONTEXTO:\n{json.dumps(context, ensure_ascii=False)}\n\nPREGUNTA:\n{question}",
                },
            ],
            "options": {"temperature": 0.2, "num_predict": 350},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]


def offline_summary(context: dict) -> str:
    d = context.get("dataset", {})
    t = context.get("target_summary", {})
    m = context.get("model", {})
    text = (
        f"El dataset contiene {d.get('rows', 0)} registros y {d.get('columns', 0)} variables, "
        f"con {d.get('missing_total', 0)} valores nulos."
    )
    if t:
        text += f" La variable objetivo presenta una media de {t.get('mean')} y una mediana de {t.get('median')}."
    if m:
        text += (
            f" El modelo lineal de demostración obtuvo R²={m.get('r2')}, "
            f"MAE={m.get('mae')} y RMSE={m.get('rmse')}."
        )
    return text + " Esta salida es descriptiva y académica."
