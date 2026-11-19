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

st.subheader("Metodología y resultados")
# Texto de Persona A: docs/INFORME_PERSONA_A.md, sección 8 (tarea A8)
st.markdown(
    "**Metodología.** Seguimos CRISP-DM con datos de Rossmann, de enero de 2013 a julio de 2015. "
    "Limpiamos el hueco de 180 tiendas en 2014 sin usar información futura y separamos los datos "
    "por tiempo: entrenamiento, validación y prueba, estos dos últimos de 42 días cada uno.\n\n"
    "Comparamos un baseline que repite la última semana, SARIMA, SARIMAX y una red GRU. SARIMAX "
    "usa la estacionalidad semanal, términos anuales de Fourier y variables conocidas de antemano: "
    "promociones, aperturas y festivos. Nunca usa el número de clientes, porque no se conoce al "
    "pronosticar. Los órdenes se eligieron solo con validación y la prueba se usó una vez.\n\n"
    "**Resultados.** SARIMAX supera al baseline y logra un MAPE menor a 15 % en los 10 niveles "
    "evaluados. En la venta total, el MAPE es 6,8 % frente a 17,4 % del baseline. La GRU logra "
    "5,5 % en el total, pero su diferencia con SARIMAX no es estadísticamente significativa "
    "(Wilcoxon, p = 0,29). El periodo de prueba no incluye festivos, y en validación los tipos a "
    "y d quedaron en 19,0 % y 17,5 %.\n\n"
    "**Cómo leer el simulador.** La promoción se asocia con cerca de 39 % más venta en el total, y "
    "el efecto varía por tipo y tienda. Es una estimación histórica: no prueba causalidad ni "
    "considera costos o margen. Los datos terminan el 31/07/2015."
)

st.subheader("Fases CRISP-DM")
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
