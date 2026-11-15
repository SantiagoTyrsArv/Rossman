"""Barra lateral común y avisos de datos de prueba."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.utils.loaders import data_status
from app.utils.settings import HORIZON, MODEL_LABELS, MODELS, all_levels, level_label


@dataclass
class Selection:
    """Lo que el usuario eligió en la barra lateral."""

    unique_id: str
    models: list[str]
    horizon: int
    split: str


def level_selector(key: str = "level") -> str:
    """Selector de nivel (`unique_id`) compartido entre páginas."""
    levels = all_levels()
    return st.sidebar.selectbox("Nivel", levels, format_func=level_label, key=key)


def forecast_sidebar(available_models: list[str]) -> Selection:
    """Selector común: nivel, modelos, horizonte y periodo."""
    st.sidebar.header("Filtros")
    uid = level_selector()
    ordered = [m for m in MODELS if m in available_models] or available_models
    default = [m for m in ordered if m in ("sarimax", "gru")] or ordered[:1]
    models = st.sidebar.multiselect(
        "Modelos", ordered, default=default, format_func=lambda m: MODEL_LABELS.get(m, m)
    )
    horizon = st.sidebar.slider("Horizonte (días)", 1, HORIZON, HORIZON)
    split = st.sidebar.radio(
        "Periodo",
        ["test", "future"],
        format_func=lambda s: {
            "test": "Prueba (20/06–31/07/2015)",
            "future": "Futuro (01/08–11/09/2015)",
        }[s],
    )
    return Selection(uid, models, horizon, split)


def mock_banner(*sources: str) -> None:
    """Aviso visible cuando alguna fuente usa datos de prueba."""
    status = data_status()
    fake = [s for s in sources if not status.get(s, True)]
    if fake:
        st.warning(
            "**DATOS DE PRUEBA** · Aún no hay artifacts reales para: "
            + ", ".join(fake)
            + ". Los gráficos usan datos sintéticos con el mismo esquema.",
            icon="⚠️",
        )
