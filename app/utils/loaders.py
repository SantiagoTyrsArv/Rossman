"""Lectura y validación de artifacts según el contrato de datos (sección 5).

Cada `read_*` es una función pura (fácil de testear) y cada `load_*` es su versión cacheada
para Streamlit. Si todavía no existen artifacts reales, se usan los datos de prueba de
`app/mock_data.py`, que cumplen el mismo esquema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import mock_data
from app.utils.settings import ARTIFACTS, SPLITS

FORECAST_COLS = ["unique_id", "ds", "y", "y_hat", "lo_95", "hi_95", "model", "split"]
HISTORY_COLS = [
    "unique_id",
    "ds",
    "y",
    "promo_share",
    "open_share",
    "school_holiday_share",
    "state_holiday",
    "promo2_share",
]
METRIC_KEYS = {"rmse", "mae", "mape"}


class ContractError(ValueError):
    """Un artifact no cumple el contrato de datos."""


def _check_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ContractError(f"Contrato roto en {source}, faltan columnas: {sorted(missing)}")


def validate_forecasts(df: pd.DataFrame, source: str = "forecasts") -> pd.DataFrame:
    """Valida el esquema 5.1 y normaliza tipos."""
    _check_columns(df, FORECAST_COLS, source)
    bad_split = set(df["split"].unique()) - set(SPLITS)
    if bad_split:
        raise ContractError(f"{source}: valores de split no válidos {sorted(bad_split)}")
    if df["y_hat"].isna().any():
        raise ContractError(f"{source}: y_hat tiene NaN")
    out = df[FORECAST_COLS].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["unique_id"] = out["unique_id"].astype(str)
    return out


def validate_history(df: pd.DataFrame, source: str = "history") -> pd.DataFrame:
    """Valida el esquema 5.3 y normaliza tipos."""
    _check_columns(df, HISTORY_COLS, source)
    out = df[HISTORY_COLS].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["state_holiday"] = out["state_holiday"].astype(str)
    return out.sort_values(["unique_id", "ds"], ignore_index=True)


def metrics_to_frame(payloads: list[dict]) -> pd.DataFrame:
    """Convierte JSON de métricas (sección 5.2) a formato largo:
    model, unique_id, split, rmse, mae, mape."""
    rows = []
    for p in payloads:
        if "model" not in p or "levels" not in p:
            raise ContractError("JSON de métricas sin 'model' o 'levels'")
        for uid, splits in p["levels"].items():
            for split, vals in splits.items():
                if not METRIC_KEYS <= set(vals):
                    raise ContractError(f"Métricas incompletas en {p['model']}/{uid}/{split}")
                rows.append({"model": p["model"], "unique_id": uid, "split": split, **vals})
    cols = ["model", "unique_id", "split", "rmse", "mae", "mape"]
    return pd.DataFrame(rows, columns=cols)


def read_forecasts(directory: Path = ARTIFACTS / "forecasts") -> pd.DataFrame | None:
    """Une todos los `<modelo>.parquet`. Devuelve None si no hay ninguno."""
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        return None
    parts = [validate_forecasts(pd.read_parquet(f), f.name) for f in files]
    return pd.concat(parts, ignore_index=True)


def read_metrics(directory: Path = ARTIFACTS / "metrics") -> pd.DataFrame | None:
    """Une todos los `<modelo>.json` (excepto `comparison.json`). None si no hay."""
    files = sorted(f for f in Path(directory).glob("*.json") if f.stem != "comparison")
    if not files:
        return None
    return metrics_to_frame([json.loads(f.read_text(encoding="utf-8")) for f in files])


def read_history(path: Path = ARTIFACTS / "series" / "history.parquet") -> pd.DataFrame | None:
    """Lee `history.parquet`. None si no existe."""
    path = Path(path)
    return validate_history(pd.read_parquet(path), path.name) if path.exists() else None


def read_comparison(path: Path = ARTIFACTS / "metrics" / "comparison.json") -> dict | None:
    """Resultado de Wilcoxon de A, si existe."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def data_status() -> dict[str, bool]:
    """True si la fuente es real, False si se usan datos de prueba."""
    return {
        "history": (ARTIFACTS / "series" / "history.parquet").exists(),
        "forecasts": any((ARTIFACTS / "forecasts").glob("*.parquet")),
        "metrics": any(f.stem != "comparison" for f in (ARTIFACTS / "metrics").glob("*.json")),
    }


@st.cache_data(show_spinner="Cargando historia…")
def load_history() -> pd.DataFrame:
    """Serie histórica por nivel (real o de prueba)."""
    df = read_history()
    return df if df is not None else validate_history(mock_data.history(), "mock")


@st.cache_data(show_spinner="Cargando pronósticos…")
def load_forecasts() -> pd.DataFrame:
    """Pronósticos de todos los modelos (reales o de prueba)."""
    df = read_forecasts()
    if df is not None:
        return df
    return validate_forecasts(mock_data.forecasts(mock_data.history()), "mock")


@st.cache_data(show_spinner="Cargando métricas…")
def load_metrics() -> pd.DataFrame:
    """Métricas en formato largo (reales o de prueba)."""
    df = read_metrics()
    if df is not None:
        return df
    return metrics_to_frame(mock_data.metrics(mock_data.forecasts(mock_data.history())))


@st.cache_data
def load_comparison() -> dict | None:
    """Resultado de Wilcoxon (`comparison.json` de A), si existe."""
    return read_comparison()
