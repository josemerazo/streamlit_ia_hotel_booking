from __future__ import annotations

import pandas as pd
import streamlit as st

from src import eda, visualizations as viz
from src.agent import HotelAgent, offline_summary
from src.data_loader import dataset_profile, load_default_data, load_uploaded_data
from src.llm_client import OllamaClient

st.set_page_config(page_title="Hotel Bookings · IA + Datos", page_icon="🏨", layout="wide")

st.title("Análisis de cancelaciones hoteleras con agente de IA local")
st.caption("Streamlit + Plotly + Scikit-learn + Ollama · Proyecto Final Integrador")


@st.cache_data(show_spinner="Cargando dataset...")
def get_default_data() -> pd.DataFrame:
    return load_default_data()


@st.cache_resource(show_spinner="Entrenando modelo...")
def get_model(_df: pd.DataFrame, key: str):
    return eda.train_cancellation_model(_df)


with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("CSV opcional (mismo esquema)", type=["csv"])
    model_name = st.text_input("Modelo Ollama", value="llama3.2:3b")

try:
    df = load_uploaded_data(uploaded) if uploaded else get_default_data()
except Exception as exc:
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()

hotels = sorted(df["hotel"].dropna().unique())
with st.sidebar:
    selected_hotels = st.multiselect("Tipo de hotel", hotels, default=hotels)
if selected_hotels:
    df = df[df["hotel"].isin(selected_hotels)]
if df.empty:
    st.warning("Selecciona al menos un tipo de hotel.")
    st.stop()

profile = dataset_profile(df)
data_key = f"{uploaded.name if uploaded else 'default'}-{','.join(selected_hotels)}-{len(df)}"
try:
    model_result = get_model(df, data_key)
except Exception:
    model_result = None

client = OllamaClient(model=model_name)
ollama_ok, installed_models, _ = client.status()
with st.sidebar:
    st.markdown("---")
    if ollama_ok:
        st.success("Ollama disponible")
        if installed_models:
            st.caption("Modelos: " + ", ".join(installed_models[:5]))
    else:
        st.warning("Ollama no detectado")
        st.caption("La aplicación continúa funcionando sin el agente generativo.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reservas", f"{profile['rows']:,}")
c2.metric("Variables", profile["columns"])
c3.metric("Valores nulos", f"{profile['missing']:,}")
c4.metric("Duplicados", f"{profile['duplicates']:,}")
c5.metric("Tasa de cancelación", f"{profile['cancel_rate']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Datos", "📊 Exploración", "📈 Modelo", "🤖 Agente", "🧩 Arquitectura"])

with tab1:
    st.subheader("Dataset: Hotel Booking Demand")
    st.info(
        "Fuente: Kaggle, *Hotel booking demand* (Antonio, Almeida & Nunes, 2019). "
        "Problema: identificar factores asociados a la cancelación de reservas (`is_canceled`)."
    )
    st.dataframe(df.head(25), width="stretch", hide_index=True)
    st.markdown("#### Calidad de datos")
    st.dataframe(eda.missing_report(df), width="stretch", hide_index=True)
    st.markdown("#### Resumen numérico")
    st.dataframe(eda.numeric_summary(df), width="stretch", hide_index=True)

with tab2:
    st.subheader("Exploración visual")
    st.plotly_chart(viz.monthly_chart(eda.monthly_stats(df)), width="stretch")

    cat_options = {
        "Tipo de depósito": "deposit_type",
        "Segmento de mercado": "market_segment",
        "Tipo de cliente": "customer_type",
        "Hotel": "hotel",
        "Tipo de habitación reservada": "reserved_room_type",
    }
    label = st.selectbox("Segmentar cancelación por", list(cat_options))
    col = cat_options[label]
    st.plotly_chart(viz.cancellation_bar(eda.cancellation_by(df, col), col), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            viz.cancellation_bar(eda.lead_time_bands(df), "rango_anticipacion"), width="stretch"
        )
    with right:
        st.plotly_chart(viz.adr_box(df), width="stretch")

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "is_canceled"]
    xcol = st.selectbox("Variable numérica para distribución", numeric_cols, index=numeric_cols.index("lead_time"))
    st.plotly_chart(viz.histogram(df, xcol, color="hotel"), width="stretch")

    st.markdown("#### Correlaciones con la cancelación")
    st.dataframe(eda.target_correlations(df).head(10), width="stretch", hide_index=True)
    st.plotly_chart(viz.correlation_heatmap(df), width="stretch")

with tab3:
    st.subheader("Regresión logística: predicción de cancelación")
    if model_result is None:
        st.warning("No fue posible entrenar el modelo con la configuración actual.")
    else:
        m = model_result["metrics"]
        cols = st.columns(5)
        for c, (k, label_) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]):
            c.metric(label_, f"{m[k]:.3f}")
        st.caption(
            f"Entrenamiento: {model_result['n_train']:,} · Prueba: {model_result['n_test']:,}. "
            "Se excluyen reservation_status y su fecha para evitar fuga de datos."
        )
        a, b = st.columns(2)
        a.plotly_chart(viz.confusion_heatmap(model_result["confusion"]), width="stretch")
        b.plotly_chart(viz.coefficients_bar(model_result["coefficients"]), width="stretch")
        st.dataframe(model_result["coefficients"], width="stretch", hide_index=True)

with tab4:
    st.subheader("Agente local para interpretación")
    context = eda.build_agent_context(df, model_result)
    agent = HotelAgent(df, client, model_result)
    st.caption(
        "El agente selecciona herramientas autorizadas, Python calcula los resultados y Ollama solo los interpreta."
    )

    presets = [
        "Resume los principales hallazgos del dataset.",
        "¿Cómo influye el tipo de depósito en la cancelación?",
        "¿Qué meses presentan mayor cancelación?",
        "¿Cómo se relaciona la anticipación (lead time) con cancelar?",
        "Interpreta las métricas y coeficientes del modelo.",
        "¿Qué segmento de mercado cancela más?",
    ]
    choice = st.selectbox("Pregunta sugerida", ["Escribir otra pregunta..."] + presets)
    question = st.text_area("Pregunta", value="" if choice == "Escribir otra pregunta..." else choice, height=100)

    if st.button("Analizar con el agente", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        else:
            st.caption("Herramientas ejecutadas: " + ", ".join(agent.select_tools(question)))
            if ollama_ok:
                with st.spinner("Consultando el modelo local..."):
                    try:
                        answer, used, results = agent.ask(question.strip(), context)
                        st.markdown(answer)
                        with st.expander("Resultados calculados por Python (evidencia)"):
                            for name, rows in results.items():
                                st.markdown(f"**{name}**")
                                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    except Exception as exc:
                        st.error(f"Ollama respondió con un error: {exc}")
                        st.markdown(offline_summary(context))
            else:
                st.markdown(offline_summary(context))
                st.caption("Respuesta de respaldo: Ollama no está disponible en este equipo.")

with tab5:
    st.subheader("Arquitectura modular")
    st.code(
        """hotel_bookings.csv
        ↓
src/data_loader.py  (carga, limpieza, variables derivadas)
        ↓
src/eda.py  (perfilado, segmentación, correlaciones, modelo)
   ↙                    ↘
src/visualizations.py    src/agent.py (HotelAgent + herramientas)
   ↓                        ↓
app.py (Streamlit) ←  src/llm_client.py (OllamaClient) → Ollama local""",
        language="text",
    )
    st.markdown(
        """
- **Imperativo:** flujo de carga, validaciones y filtros en `app.py`.
- **Funcional:** funciones puras de perfilado, segmentación y gráficos.
- **Orientado a objetos:** `OllamaClient` y `HotelAgent` encapsulan el LLM y las herramientas autorizadas.
- **Modularidad:** interfaz, datos, análisis, visualización e IA en módulos separados.
        """
    )

st.markdown("---")
st.caption("Proyecto académico. Las asociaciones observadas no implican causalidad.")
