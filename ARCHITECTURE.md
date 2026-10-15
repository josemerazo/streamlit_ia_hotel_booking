# Arquitectura del demo

```text
CSV de Scikit-learn / archivo cargado
              ↓
      modules/data.py
              ↓
    modules/analytics.py
       ↙              ↘
visualizations.py   contexto analítico
       ↓              ↓
     Streamlit ← modules/agent.py → Ollama local
```

## Propósito didáctico

El demo separa la interfaz, la carga de datos, la lógica analítica, las visualizaciones y la conexión con el modelo local. Esto permite mostrar modularidad, funciones reutilizables, programación imperativa y una clase simple para el agente.
