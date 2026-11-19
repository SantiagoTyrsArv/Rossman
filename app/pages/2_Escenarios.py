"""Simulación de escenarios de promoción y festivos con SARIMAX (periodo futuro)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.utils.charts import scenario_figure
from app.utils.loaders import load_history
from app.utils.settings import DOW_ES, FUTURE_START, HORIZON, level_label
from app.utils.sidebar import level_selector
from app.utils.simulator import simulate

st.set_page_config(page_title="Escenarios · Rossmann", page_icon="📈", layout="wide")
st.title("Simulación de escenarios")
st.caption(
    "¿Cuánto cambian las ventas de agosto–septiembre de 2015 si cambiamos el calendario de "
    "promociones o hay festivos? Base = patrón histórico de promociones."
)

st.sidebar.header("Escenario")
uid = level_selector()
horizon = st.sidebar.slider("Horizonte (días)", 7, HORIZON, HORIZON, step=7)
dates = pd.date_range(FUTURE_START, periods=horizon, freq="D")
period_end = dates[-1].date()

PRESETS = [
    "Patrón histórico",
    "Sin promociones",
    "Promo intensiva (lun–vie todas las semanas)",
    "Personalizado",
]
preset = st.sidebar.radio(
    "Promociones",
    PRESETS,
    help=(
        "Los días elegidos **reemplazan** el calendario de promos de la base, no se suman a él. "
        "La base ya trae el patrón histórico (unos 15 días con promo en 42 días): un escenario "
        "con menos días de promo que ese puede vender menos que la base."
    ),
)

promo_days: set[pd.Timestamp] | None
if preset == "Patrón histórico":
    promo_days = None
elif preset == "Sin promociones":
    promo_days = set()
elif preset.startswith("Promo intensiva"):
    promo_days = {d for d in dates if d.dayofweek < 5}
else:
    rng = st.sidebar.date_input(
        "Rango con promoción",
        value=(dates[0].date(), dates[13].date()),
        min_value=dates[0].date(),
        max_value=period_end,
    )
    dows = st.sidebar.multiselect("Días de la semana", DOW_ES, default=DOW_ES[:5])
    start, end = rng if isinstance(rng, tuple) and len(rng) == 2 else (rng, rng)
    idx = {DOW_ES.index(d) for d in dows}
    promo_days = {d for d in dates if start <= d.date() <= end and d.dayofweek in idx}

if promo_days is None:
    st.sidebar.caption("Escenario = base: mismo calendario de promos que el histórico.")
else:
    st.sidebar.caption(
        f"Escenario con **{len(promo_days)} días de promo** en {horizon} días. Estos días "
        "reemplazan el calendario histórico de la base; el resto queda sin promo."
    )

with st.sidebar.expander("Festivos"):
    use_school = st.checkbox("Vacaciones escolares", value=False)
    school_days: set[pd.Timestamp] | None = None
    if use_school:
        srng = st.date_input(
            "Rango de vacaciones escolares",
            value=(dates[0].date(), dates[min(20, horizon - 1)].date()),
            min_value=dates[0].date(),
            max_value=period_end,
        )
        s0, s1 = srng if isinstance(srng, tuple) and len(srng) == 2 else (srng, srng)
        school_days = {d for d in dates if s0 <= d.date() <= s1}
    holi = st.multiselect(
        "Festivos estatales",
        list(dates),
        format_func=lambda d: f"{DOW_ES[d.dayofweek][:3]} {d:%d/%m/%Y}",
        help="Ej.: 15/08 (Asunción) en algunos estados.",
    )
    holi_type = st.selectbox(
        "Tipo de festivo",
        ["a", "b", "c"],
        format_func=lambda t: {"a": "a · festivo público", "b": "b · Pascua", "c": "c · Navidad"}[
            t
        ],
    )
    state_holidays = {d: holi_type for d in holi} or None

hist = load_history()
h_uid = hist[hist["unique_id"] == uid]
p2_hist = float(h_uid["promo2_share"].iloc[-56:].mean()) if not h_uid.empty else 0.5
promo2 = st.sidebar.slider(
    "Proporción de tiendas en Promo2",
    0.0,
    1.0,
    round(p2_hist, 2),
    0.01,
    help=f"Por defecto, el promedio de las últimas 8 semanas ({p2_hist:.2f}).",
)
promo2_share = None if abs(promo2 - round(p2_hist, 2)) < 1e-9 else promo2

sim, is_real = simulate(uid, horizon, promo_days, school_days, state_holidays, promo2_share)

if not is_real:
    st.warning(
        "**DATOS DE PRUEBA** · Todavía no están el modelo SARIMAX y `build_future_exog` de "
        "Persona A para este nivel. El simulador usa un efecto de promoción sintético.",
        icon="⚠️",
    )
st.info(
    "El simulador usa **SARIMAX** y solo vale para las variables que el modelo incluye y dentro "
    "de los rangos históricos. La GRU no participa (sus predicciones son fijas).",
    icon="ℹ️",
)

base_total, scen_total = sim["base"].sum(), sim["scen"].sum()
delta = scen_total - base_total
pct = delta / base_total * 100 if base_total else 0.0

st.markdown(f"**{level_label(uid)}** · {horizon} días desde {FUTURE_START:%d/%m/%Y}")
c1, c2, c3 = st.columns(3)
c1.metric("Ventas base", f"{base_total:,.0f} €")
c2.metric("Ventas del escenario", f"{scen_total:,.0f} €")
c3.metric("Ventas incrementales", f"{delta:+,.0f} €", delta=f"{pct:+.1f} %")

st.plotly_chart(scenario_figure(sim), use_container_width=True)

sim = sim.assign(
    semana=sim["ds"].dt.to_period("W-SUN").dt.start_time,
    dia=sim["ds"].dt.dayofweek,
    incremento=sim["scen"] - sim["base"],
)
col_w, col_d = st.columns(2)
with col_w:
    st.markdown("**Por semana**")
    w = sim.groupby("semana")[["base", "scen", "incremento"]].sum()
    w["incremento_%"] = w["incremento"] / w["base"] * 100
    w.index = [f"Semana del {d:%d/%m}" for d in w.index]
    w.columns = ["Base (€)", "Escenario (€)", "Incremento (€)", "Incremento (%)"]
    st.dataframe(
        w.style.format("{:,.0f}", subset=w.columns[:3]).format(
            "{:+.1f}", subset=["Incremento (%)"]
        ),
        use_container_width=True,
    )
with col_d:
    st.markdown("**Por día de la semana**")
    d = sim.groupby("dia")[["base", "scen", "incremento"]].sum()
    d["incremento_%"] = d["incremento"] / d["base"].where(d["base"] > 0) * 100
    d.index = [DOW_ES[i] for i in d.index]
    d.columns = ["Base (€)", "Escenario (€)", "Incremento (€)", "Incremento (%)"]
    st.dataframe(
        d.style.format("{:,.0f}", subset=d.columns[:3]).format(
            "{:+.1f}", subset=["Incremento (%)"], na_rep="—"
        ),
        use_container_width=True,
    )

st.download_button(
    "Descargar escenario (CSV)",
    sim.drop(columns=["semana", "dia"]).to_csv(index=False).encode("utf-8"),
    file_name=f"escenario_{uid}.csv",
    mime="text/csv",
)
