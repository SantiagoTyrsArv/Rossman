"""Visualización temporal: historia, STL, calendario, efecto de promo y correlogramas."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from app.utils.charts import correlogram, heatmap_figure, history_figure, stl_figure
from app.utils.loaders import load_history
from app.utils.settings import DOW_ES, LEVELS_TYPES, MONTH_ES, SEASON, TRAIN_END, level_label
from app.utils.sidebar import level_selector, mock_banner

st.set_page_config(page_title="Series · Rossmann", page_icon="📈", layout="wide")
st.title("Visualización temporal")
mock_banner("history")

hist = load_history()
uid = level_selector()
s = hist[hist["unique_id"] == uid].set_index("ds")["y"].asfreq("D")

# --- Serie histórica ------------------------------------------------------------------------
st.subheader(f"Ventas históricas · {level_label(uid)}")
agg = st.radio("Agregación", ["Diaria", "Semanal", "Mensual"], horizontal=True)
rule = {"Diaria": None, "Semanal": "W-SUN", "Mensual": "MS"}[agg]
view = s if rule is None else s.resample(rule).sum()
st.plotly_chart(history_figure(view.reset_index()), width="stretch")


# --- STL ------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando STL…")
def stl_parts(uid: str, train: pd.Series) -> pd.DataFrame:
    """STL robusta (period = 7) sobre log1p(y) del entrenamiento."""
    y = np.log1p(train)
    res = STL(y, period=SEASON, robust=True).fit()
    return pd.DataFrame(
        {"observed": y, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}
    )


train = s.loc[:TRAIN_END].ffill().fillna(0)
st.subheader("Descomposición STL (entrenamiento, log1p de las ventas)")
parts = stl_parts(uid, train)
st.plotly_chart(
    stl_figure(parts.index, parts["observed"], parts["trend"], parts["seasonal"], parts["resid"]),
    width="stretch",
)
var_resid = parts["resid"].var()
f_season = max(0.0, 1 - var_resid / (parts["seasonal"] + parts["resid"]).var())
f_trend = max(0.0, 1 - var_resid / (parts["trend"] + parts["resid"]).var())
c1, c2 = st.columns(2)
c1.metric(
    "Fuerza de la estacionalidad semanal",
    f"{f_season:.2f}",
    help="1 − Var(R) / Var(S + R). Cerca de 1 = estacionalidad fuerte.",
)
c2.metric("Fuerza de la tendencia", f"{f_trend:.2f}", help="1 − Var(R) / Var(T + R).")

# --- Calendario -----------------------------------------------------------------------------
st.subheader("Patrones de calendario")
col_h, col_p = st.columns(2)
with col_h:
    open_days = s[s > 0]
    pivot = (
        open_days.to_frame("y")
        .assign(dow=open_days.index.dayofweek, month=open_days.index.month)
        .pivot_table(index="dow", columns="month", values="y", aggfunc="mean")
        .reindex(index=range(7), columns=range(1, 13))
    )
    pivot.index = DOW_ES
    pivot.columns = MONTH_ES
    st.plotly_chart(
        heatmap_figure(pivot, "Venta promedio por día de la semana y mes (días abiertos)"),
        width="stretch",
    )
with col_p:
    rows = []
    for t in LEVELS_TYPES:
        g = hist[(hist["unique_id"] == t) & (hist["y"] > 0)]
        promo = g["promo_share"] >= 0.5
        rows.append(
            {
                "tipo": t[-1].upper(),
                "Con promo": g.loc[promo, "y"].mean(),
                "Sin promo": g.loc[~promo, "y"].mean(),
            }
        )
    pr = pd.DataFrame(rows)
    pr["uplift"] = (pr["Con promo"] / pr["Sin promo"] - 1) * 100
    fig = go.Figure()
    fig.add_bar(x=pr["tipo"], y=pr["Sin promo"], name="Sin promo", marker_color="#8c8c8c")
    fig.add_bar(
        x=pr["tipo"],
        y=pr["Con promo"],
        name="Con promo",
        marker_color="#d1495b",
        text=[f"{u:+.0f} %" for u in pr["uplift"]],
        textposition="outside",
    )
    fig.update_layout(
        title="Venta diaria promedio con y sin promo por tipo de tienda",
        template="plotly_white",
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(tickformat=",.0f", title="€"),
        xaxis_title="Tipo de tienda",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Comparación descriptiva (no causal): los días con promo también difieren en día de la "
        "semana y época del año. El efecto aislado lo estima el SARIMAX."
    )

# --- ACF / PACF -----------------------------------------------------------------------------
st.subheader("Autocorrelación (entrenamiento, log1p)")
transform = st.radio(
    "Transformación",
    ["y", "Δy (diferencia simple)", "Δ₇y (diferencia estacional)"],
    horizontal=True,
)
z = np.log1p(train)
if transform.startswith("Δy"):
    z = z.diff()
elif transform.startswith("Δ₇"):
    z = z.diff(SEASON)
z = z.dropna()
nlags = 60
conf = 1.96 / np.sqrt(len(z))
col_a, col_b = st.columns(2)
col_a.plotly_chart(correlogram(acf(z, nlags=nlags), conf, "ACF"), width="stretch")
col_b.plotly_chart(correlogram(pacf(z, nlags=nlags, method="ywm"), conf, "PACF"), width="stretch")
st.caption(
    "Líneas punteadas: banda de significancia aproximada ±1,96/√n. Picos en 7, 14, 21… "
    "indican estacionalidad semanal."
)
