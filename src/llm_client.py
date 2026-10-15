from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    """Cliente mínimo para la API HTTP de Ollama."""

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

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
