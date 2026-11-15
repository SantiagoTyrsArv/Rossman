"""Página de inicio: objetivo, metodología CRISP-DM, datos y limitaciones."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.utils.loaders import data_status
from app.utils.settings import (
    FUTURE_END,
    FUTURE_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)

st.set_page_config(page_title="Rossmann · Predicción de demanda", page_icon="📈", layout="wide")

st.title("Predicción de demanda Rossmann")
st.caption(
    "Proyecto final de Inteligencia de Minería de Datos · SARIMAX y GRU con variables externas "
    "para planificar el marketing"
)

status = data_status()
if not all(status.values()):
    st.info(
        "La app está corriendo con **datos de prueba** en: "
        + ", ".join(k for k, v in status.items() if not v)
        + ". Se reemplazan solos cuando lleguen los artifacts reales.",
        icon="ℹ️",
    )

st.subheader("Objetivo")
st.markdown(
    "En 4 semanas, **pronosticar las ventas diarias de los próximos 42 días** a nivel total, por "
    "tipo de tienda (a/b/c/d) y para 5 tiendas seleccionadas, **superando al baseline ingenuo "
    "estacional y con MAPE < 15 %** en el periodo de prueba, y publicarlo en esta app con "
    "simulación de escenarios de promoción."
)

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**¿Cuánto venderemos?**\n\nDashboard de predicción a 42 días.")
c2.markdown("**¿Cuánto suben con promos?**\n\nSimulación de escenarios con SARIMAX.")
c3.markdown("**¿Quién responde mejor?**\n\nComparación por tipo de tienda.")
c4.markdown("**¿Cuándo invertir?**\n\nEstacionalidad semanal y anual.")

st.subheader("Metodología CRISP-DM")
# TODO(B7): reemplazar con los textos de metodología que entregue Persona A (tarea A8)
phases = {
    "1. Comprensión del negocio": "Planificar promociones y presupuesto de marketing con un "
    "pronóstico confiable de la demanda diaria.",
    "2. Comprensión de los datos": "EDA de `train.csv` y `store.csv` (2013–2015, 1 115 tiendas) y "
    "checklist de calidad ISO/IEC 25012.",
    "3. Preparación": "Limpieza, unión con `store.csv`, series agregadas por nivel y variables "
    "exógenas conocidas de antemano (promo, festivos, apertura). `Customers` se excluye para "
    "evitar fuga de datos.",
    "4. Modelado": "Baseline ingenuo estacional, SARIMA, SARIMAX con exógenas y términos de "
    "Fourier, y una red GRU con ventanas de 56 → 42 días.",
    "5. Evaluación": "RMSE, MAE y MAPE (solo días con venta > 0) en validación y una sola vez "
    "en prueba; prueba de Wilcoxon entre modelos.",
    "6. Despliegue": "Esta app en Streamlit Community Cloud con artifacts precalculados.",
}
for phase, text in phases.items():
    with st.expander(phase):
        st.markdown(text)

st.subheader("Datos y partición temporal")
st.markdown(
    "Dataset [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) "
    "(Kaggle). La partición es **temporal**, nunca aleatoria:"
)
fmt = "%d/%m/%Y"
st.table(
    {
        "Conjunto": ["Entrenamiento", "Validación", "Prueba", "Futuro (app)"],
        "Desde": [d.strftime(fmt) for d in (TRAIN_START, VAL_START, TEST_START, FUTURE_START)],
        "Hasta": [d.strftime(fmt) for d in (TRAIN_END, VAL_END, TEST_END, FUTURE_END)],
        "Uso": [
            "Ajuste de modelos y escaladores",
            "Elegir órdenes e hiperparámetros",
            "Evaluación final (una sola vez)",
            "Pronóstico y simulación",
        ],
    }
)

st.subheader("Limitaciones")
st.markdown(
    '- Los datos terminan en julio de 2015; el "futuro" es agosto–septiembre de 2015.\n'
    "- No incluye clima, precios ni acciones de la competencia en el tiempo.\n"
    "- El simulador usa SARIMAX, que es lineal en log: solo vale para las variables que incluye y "
    "dentro de los rangos históricos. La GRU no participa del simulador.\n"
    "- Los intervalos al 95 % suponen que el futuro se parece al pasado."
)

st.caption("Navega con el menú de la izquierda: Dashboard · Escenarios · Series.")
