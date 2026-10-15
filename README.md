# Análisis de cancelaciones hoteleras · Streamlit + Ollama

Proyecto Final Integrador — Maestría en Inteligencia Artificial y Ciencia de Datos, *Paradigmas de Programación para IA y Análisis de Datos*.

Aplicación modular en Streamlit que explora el dataset **Hotel Booking Demand**, entrena una regresión logística para predecir cancelaciones y ofrece un agente de IA local (Ollama) que interpreta resultados calculados por Python.

## Dataset

- Fuente: Kaggle, [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Antonio, Almeida & Nunes, 2019).
- 119 390 reservas, 32 variables (hotel, `lead_time`, `deposit_type`, `adr`, `market_segment`, `is_canceled`, etc.).
- Problema: identificar factores asociados a la cancelación de reservas.
- El CSV (~16 MB) se incluye en `data/hotel_bookings.csv`. Si falta, descárgalo desde Kaggle y colócalo en esa ruta.

## Estructura

```text
├── app.py                # Interfaz Streamlit
├── requirements.txt
├── README.md
├── .gitignore
├── data/hotel_bookings.csv
├── assets/               # Capturas para el informe
├── docs/informe.md       # Borrador del informe final
└── src/
    ├── __init__.py
    ├── data_loader.py    # Carga, limpieza, variables derivadas
    ├── eda.py            # Perfilado, segmentación, correlaciones, modelo
    ├── visualizations.py # Gráficos Plotly
    ├── llm_client.py     # Clase OllamaClient
    └── agent.py          # Clase HotelAgent con herramientas autorizadas
```

## 1. Entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Ollama y modelo

1. Instalar Ollama desde https://ollama.com/download.
2. Descargar el modelo (versión usada: `llama3.2:3b`; con pocos recursos, `llama3.2:1b`):

```bash
ollama pull llama3.2:3b
```

El modelo no se almacena en el repositorio. Sin Ollama la app funciona y muestra un resumen de respaldo.

## 4. Ejecución

```bash
streamlit run app.py
```

## Agente de IA

`HotelAgent` recibe una pregunta, elige herramientas autorizadas por palabras clave (depósito, segmento, mes, anticipación, país, correlaciones, modelo), las ejecuta en Python y envía los resultados a `OllamaClient`. El LLM solo interpreta; no calcula.
"# streamlit_ia_hotel_booking" 
