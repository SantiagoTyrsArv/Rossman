"""Dashboard de predicción: pronóstico por nivel, comparación de modelos y descarga."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.utils.charts import forecast_figure
from app.utils.loaders import load_comparison, load_forecasts, load_history, load_metrics
from app.utils.settings import DOW_ES, MODEL_LABELS, SPLIT_START, level_label
from app.utils.sidebar import forecast_sidebar, mock_banner

st.set_page_config(page_title="Dashboard · Rossmann", page_icon="📈", layout="wide")
st.title("Dashboard de predicción")
mock_banner("history", "forecasts", "metrics")

fc_all = load_forecasts()
hist_all = load_history()
met_all = load_metrics()

sel = forecast_sidebar(sorted(fc_all["model"].unique()))
if not sel.models:
    st.info("Elige al menos un modelo en la barra lateral.")
    st.stop()

start = SPLIT_START[sel.split]
dates = pd.date_range(start, periods=sel.horizon, freq="D")
fc = fc_all[
    (fc_all["unique_id"] == sel.unique_id)
    & (fc_all["split"] == sel.split)
    & (fc_all["model"].isin(sel.models))
    & (fc_all["ds"].isin(dates))
]
hist = hist_all[hist_all["unique_id"] == sel.unique_id].set_index("ds")["y"]

if fc.empty:
    st.warning("No hay pronósticos para esta combinación de nivel, modelo y periodo.")
    st.stop()

# --- Tarjetas -------------------------------------------------------------------------------
main_model = sel.models[0]
main = fc[fc["model"] == main_model].sort_values("ds")
total = main["y_hat"].sum()
last_year = hist.reindex(main["ds"] - pd.Timedelta(days=364)).sum()  # mismos días de la semana
peak = main.loc[main["y_hat"].idxmax()]
mape_row = met_all[
    (met_all["model"] == main_model)
    & (met_all["unique_id"] == sel.unique_id)
    & (met_all["split"] == "test")
]

st.markdown(
    f"**{level_label(sel.unique_id)}** · modelo principal: **{MODEL_LABELS.get(main_model)}** "
    f"(el primero elegido) · {sel.horizon} días desde {start:%d/%m/%Y}"
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas previstas del horizonte", f"{total:,.0f} €")
if last_year > 0:
    c2.metric(
        "Vs. mismos días del año anterior",
        f"{last_year:,.0f} €",
        delta=f"{(total / last_year - 1) * 100:+.1f} %",
    )
else:
    c2.metric("Vs. mismos días del año anterior", "sin datos")
peak_day = f"{DOW_ES[peak['ds'].dayofweek][:3]} {peak['ds']:%d/%m}"
c3.metric("Día pico previsto", peak_day, delta=f"{peak['y_hat']:,.0f} €", delta_color="off")
c4.metric(
    "MAPE en prueba",
    f"{mape_row['mape'].iloc[0]:.1f} %" if not mape_row.empty else "—",
    help="Calculado solo sobre días con venta real > 0.",
)

tab_fc, tab_cmp = st.tabs(["Pronóstico", "Comparar modelos"])

with tab_fc:
    recent = hist.loc[: start - pd.Timedelta(days=1)].iloc[-90:].reset_index()
    st.plotly_chart(
        forecast_figure(recent, fc, sel.models, "Últimos 90 días reales y pronóstico"),
        width="stretch",
    )
    table = fc.pivot_table(index="ds", columns="model", values="y_hat").rename(columns=MODEL_LABELS)
    if sel.split == "test":
        table.insert(0, "Real", fc.drop_duplicates("ds").set_index("ds")["y"])
    st.download_button(
        "Descargar pronóstico (CSV)",
        fc.to_csv(index=False).encode("utf-8"),
        file_name=f"pronostico_{sel.unique_id}_{sel.split}.csv",
        mime="text/csv",
    )
    with st.expander("Ver tabla diaria"):
        st.dataframe(table.style.format("{:,.0f}"), width="stretch")

with tab_cmp:
    st.caption("Todas las líneas sobre el periodo de prueba (20/06–31/07/2015).")
    test = fc_all[(fc_all["unique_id"] == sel.unique_id) & (fc_all["split"] == "test")]
    models_all = [m for m in MODEL_LABELS if m in test["model"].unique()]
    recent = hist.loc[: SPLIT_START["test"] - pd.Timedelta(days=1)].iloc[-28:].reset_index()
    st.plotly_chart(
        forecast_figure(recent, test, models_all, "Modelos vs. real en prueba"),
        width="stretch",
    )

    met = met_all[met_all["unique_id"] == sel.unique_id]
    split_m = st.radio(
        "Métricas de",
        ["test", "val"],
        horizontal=True,
        format_func=lambda s: {"test": "Prueba", "val": "Validación"}[s],
    )
    met = met[met["split"] == split_m].set_index("model")[["rmse", "mae", "mape"]]
    met.index = [MODEL_LABELS.get(m, m) for m in met.index]
    met.columns = ["RMSE (€)", "MAE (€)", "MAPE (%)"]
    if met.empty:
        st.info("Todavía no hay métricas para este nivel.")
    else:
        st.dataframe(
            met.style.format(
                {"RMSE (€)": "{:,.0f}", "MAE (€)": "{:,.0f}", "MAPE (%)": "{:.2f}"}
            ).highlight_min(axis=0, props="font-weight:700; background-color:rgba(31,158,118,.18)"),
            width="stretch",
        )
        st.caption("En verde, el mejor modelo por métrica. Menor es mejor.")

    comp = load_comparison()
    if comp:
        st.markdown("**Prueba de Wilcoxon** (errores absolutos diarios, nivel total)")
        st.json(comp, expanded=False)
