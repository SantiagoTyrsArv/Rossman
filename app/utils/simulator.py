"""Simulación de escenarios: la única inferencia en vivo de la app (SARIMAX de Persona A).

Si existen `artifacts/models/sarimax_<uid>.pkl`, su `_meta.json` y `build_future_exog` de A,
se usa el modelo real. Si no, `mock_data.simulate` (marcado como DATOS DE PRUEBA).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from app import mock_data
from app.utils.settings import ARTIFACTS, FUTURE_START

SIM_COLS = ["ds", "base", "base_lo", "base_hi", "scen", "scen_lo", "scen_hi"]


def sarimax_available(uid: str) -> bool:
    """True si están el pkl, el meta y `build_future_exog` para este nivel."""
    models = ARTIFACTS / "models"
    if not (models / f"sarimax_{uid}.pkl").exists():
        return False
    if not (models / f"sarimax_{uid}_meta.json").exists():
        return False
    try:
        from src.features.exog import build_future_exog  # noqa: F401
    except ImportError:
        return False
    return True


@st.cache_resource(show_spinner="Cargando modelo SARIMAX…")
def load_sarimax(uid: str):
    """Carga el resultado SARIMAX reajustado y su meta (sección 5.4)."""
    from statsmodels.iolib.smpickle import load_pickle

    meta = json.loads((ARTIFACTS / "models" / f"sarimax_{uid}_meta.json").read_text("utf-8"))
    return load_pickle(ARTIFACTS / "models" / f"sarimax_{uid}.pkl"), meta


def _forecast(res, meta: dict, exog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc = res.get_forecast(steps=len(exog), exog=exog[meta["exog_cols"]])
    mean = np.asarray(fc.predicted_mean, float)
    ci = np.asarray(fc.conf_int(alpha=0.05), float)
    lo, hi = ci[:, 0], ci[:, 1]
    if meta.get("log_target", False):
        mean, lo, hi = np.expm1(mean), np.expm1(lo), np.expm1(hi)
    return mean, lo, hi


def simulate(
    uid: str,
    horizon: int,
    promo_days: set[pd.Timestamp] | None = None,
    school_holiday_days: set[pd.Timestamp] | None = None,
    state_holidays: dict[pd.Timestamp, str] | None = None,
    promo2_share: float | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Base (patrón histórico) vs escenario. Devuelve (tabla, es_real)."""
    if not sarimax_available(uid):
        sim = mock_data.simulate(
            uid, horizon, promo_days, school_holiday_days, state_holidays, promo2_share
        )
        return sim[SIM_COLS], False

    from src.features.exog import build_future_exog

    res, meta = load_sarimax(uid)
    dates = pd.date_range(FUTURE_START, periods=horizon, freq="D")
    x_base = build_future_exog(uid, dates)
    x_scen = build_future_exog(
        uid,
        dates,
        promo_days=promo_days,
        school_holiday_days=school_holiday_days,
        state_holidays=state_holidays,
        promo2_share=promo2_share,
    )
    b, b_lo, b_hi = _forecast(res, meta, x_base)
    s, s_lo, s_hi = _forecast(res, meta, x_scen)
    sim = pd.DataFrame(
        {
            "ds": dates,
            "base": b,
            "base_lo": b_lo,
            "base_hi": b_hi,
            "scen": s,
            "scen_lo": s_lo,
            "scen_hi": s_hi,
        }
    )
    return sim, True
