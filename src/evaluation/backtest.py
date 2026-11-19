"""Particiones temporales y escritura validada de pronósticos y métricas (contrato 5.1 y 5.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS,
    FUTURE_END,
    FUTURE_START,
    HORIZON,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from src.evaluation.metrics import compute_all

FORECAST_COLS = ["unique_id", "ds", "y", "y_hat", "lo_95", "hi_95", "model", "split"]
METRIC_KEYS = {"rmse", "mae", "mape"}
OWN_MODELS = {"naive", "sarima", "sarimax"}  # gru.* pertenece a Persona B
WINDOWS = {
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
    "future": (FUTURE_START, FUTURE_END),
}
ORIGINS = {"val": TRAIN_END, "test": VAL_END, "future": TEST_END}


def split(series: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parte una serie con columna `ds` en train, val y test según `src/config.py`."""
    ds = pd.to_datetime(series["ds"])
    return {
        "train": series[ds <= TRAIN_END],
        "val": series[ds.between(VAL_START, VAL_END)],
        "test": series[ds.between(TEST_START, TEST_END)],
    }


def split_dates(name: str) -> pd.DatetimeIndex:
    """Fechas diarias de `val`, `test` o `future`."""
    start, end = WINDOWS[name]
    return pd.date_range(start, end, freq="D")


def _check_model(model: str) -> None:
    if model not in OWN_MODELS:
        raise ValueError(f"Persona A solo escribe {sorted(OWN_MODELS)}, no '{model}'")


def validate_forecast(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Comprueba el esquema 5.1 y devuelve las columnas en el orden del contrato."""
    _check_model(model)
    missing = set(FORECAST_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    out = df[FORECAST_COLS].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    for col in ("y", "y_hat", "lo_95", "hi_95"):
        out[col] = out[col].astype(float)
    if set(out["model"]) != {model}:
        raise ValueError(f"La columna model debe valer solo '{model}'")
    bad_splits = set(out["split"]) - set(WINDOWS)
    if bad_splits:
        raise ValueError(f"Valores de split no válidos: {sorted(bad_splits)}")
    if not np.isfinite(out["y_hat"]).all() or (out["y_hat"] < 0).any():
        raise ValueError("y_hat debe ser finito y no negativo")
    both = out["lo_95"].notna() & out["hi_95"].notna()
    if out["lo_95"].notna().ne(out["hi_95"].notna()).any():
        raise ValueError("lo_95 y hi_95 deben ser ambos NaN o ambos numéricos")
    if (out.loc[both, "lo_95"] > out.loc[both, "hi_95"]).any():
        raise ValueError("lo_95 no puede superar a hi_95")
    for (uid, name), g in out.groupby(["unique_id", "split"]):
        if len(g) != HORIZON or list(g["ds"].sort_values()) != list(split_dates(name)):
            raise ValueError(f"{uid}/{name}: las fechas no son las {HORIZON} del periodo")
        if name == "future" and g["y"].notna().any():
            raise ValueError(f"{uid}: y debe ser NaN en future")
        if name != "future" and not np.isfinite(g["y"]).all():
            raise ValueError(f"{uid}/{name}: falta la venta real")
    return out.sort_values(["unique_id", "split", "ds"], ignore_index=True)


def write_forecast(df: pd.DataFrame, model: str, root: Path = ARTIFACTS) -> Path:
    """Valida y escribe `forecasts/<modelo>.parquet`."""
    out = validate_forecast(df, model)
    path = Path(root) / "forecasts" / f"{model}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def metrics_payload(df: pd.DataFrame, model: str) -> dict:
    """Métricas por nivel en val y test, siempre con `compute_all`."""
    levels: dict[str, dict] = {}
    scored = df[df["split"].isin(["val", "test"])]
    for (uid, name), g in scored.groupby(["unique_id", "split"]):
        levels.setdefault(str(uid), {})[str(name)] = compute_all(g["y"], g["y_hat"])
    return {"model": model, "levels": levels}


def write_metrics(payload: dict, model: str, root: Path = ARTIFACTS) -> Path:
    """Valida el esquema 5.2 y escribe `metrics/<modelo>.json`."""
    _check_model(model)
    if payload.get("model") != model or not payload.get("levels"):
        raise ValueError("El JSON debe traer 'model' coincidente y 'levels' no vacío")
    for uid, splits in payload["levels"].items():
        for name, vals in splits.items():
            if name not in {"val", "test"} or set(vals) != METRIC_KEYS:
                raise ValueError(f"Métricas no válidas en {uid}/{name}")
    path = Path(root) / "metrics" / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
