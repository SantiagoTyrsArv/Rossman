"""GRU multihorizonte para Rossmann (Persona B).

Arquitectura: 56 días pasados (ventas escaladas + exógenas) y 42 días de exógenas futuras
conocidas (promo, festivos, apertura, calendario) → 42 días de ventas.

Reglas del contrato que se respetan aquí:
- sin `Customers` (fuga de datos);
- el escalador se ajusta solo con filas <= TRAIN_END;
- las ventanas de entrenamiento no tienen objetivo después de su fecha de corte;
- TensorFlow se importa solo dentro de las funciones que lo necesitan (la app no lo instala).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

try:
    from src.config import (  # type: ignore[import-not-found]
        ARTIFACTS,
        HORIZON,
        ROOT,
        SEED,
        TEST_END,
        TRAIN_END,
        VAL_END,
    )
except ImportError:  # TODO quitar cuando exista src/config.py (P-01)
    ROOT = Path(__file__).resolve().parents[2]
    ARTIFACTS = ROOT / "artifacts"
    TRAIN_END = pd.Timestamp("2015-05-08")
    VAL_END = pd.Timestamp("2015-06-19")
    TEST_END = pd.Timestamp("2015-07-31")
    HORIZON = 42
    SEED = 42

try:
    from src.evaluation.metrics import compute_all  # type: ignore[import-not-found]
except ImportError:  # TODO quitar: stub local hasta que A entregue metrics.py (P-02)

    def compute_all(y, y_hat) -> dict[str, float]:
        """Stub con la misma fórmula del contrato (MAPE solo con y > 0)."""
        y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
        err = y - y_hat
        m = y > 0
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "mape": float(np.mean(np.abs(err[m] / y[m])) * 100),
        }


LOOKBACK = 56
MODEL_NAME = "gru"
EXOG_COLS = ["promo_share", "open_share", "school_holiday_share", "sh_a", "sh_b", "sh_c"]
CALENDAR_COLS = [f"dow_{i}" for i in range(7)] + ["month_sin", "month_cos"]
FUT_COLS = EXOG_COLS + CALENDAR_COLS  # exógenas futuras conocidas (entrada `future_exog`)
PAST_COLS = ["y_scaled", *FUT_COLS]  # entrada `past`
FORECAST_COLS = ["unique_id", "ds", "y", "y_hat", "lo_95", "hi_95", "model", "split"]


# --- Variables de entrada -------------------------------------------------------------------
def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Día de la semana one-hot y mes en seno/coseno."""
    dates = pd.DatetimeIndex(dates)
    dow = np.eye(7)[dates.dayofweek.to_numpy()]
    angle = 2 * np.pi * (dates.month.to_numpy() - 1) / 12
    out = pd.DataFrame(dow, index=dates, columns=CALENDAR_COLS[:7])
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    return out


def exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exógenas conocidas de antemano en el orden `FUT_COLS`, indexadas por fecha.

    Acepta `history.parquet` (con `state_holiday`) o la salida de `build_future_exog` de A
    (con `sh_a`, `sh_b`, `sh_c` e índice de fechas).
    """
    df = df.copy()
    if "ds" in df.columns:
        df = df.set_index("ds")
    df.index = pd.DatetimeIndex(df.index)
    if "state_holiday" in df.columns:
        sh = df["state_holiday"].astype(str)
        for k in ("a", "b", "c"):
            df[f"sh_{k}"] = (sh == k).astype(float)
    missing = set(EXOG_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan exógenas para la GRU: {sorted(missing)}")
    out = df[EXOG_COLS].astype(float).join(calendar_features(df.index))
    if out.isna().any().any():
        raise ValueError("Las exógenas de la GRU tienen NaN")
    return out[FUT_COLS]


def fit_scaler(y: pd.Series, fit_end: pd.Timestamp = TRAIN_END) -> MinMaxScaler:
    """MinMaxScaler sobre log1p(y) ajustado SOLO con fechas <= fit_end."""
    train = np.log1p(y.loc[:fit_end].to_numpy(float)).reshape(-1, 1)
    if len(train) == 0:
        raise ValueError("No hay datos de entrenamiento para ajustar el escalador")
    return MinMaxScaler().fit(train)


def scale_target(y: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """log1p + escalado."""
    y = np.asarray(y, float)
    return scaler.transform(np.log1p(y).reshape(-1, 1)).reshape(y.shape)


def inverse_target(y_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Revierte el escalado y log1p; recorta en 0 (no hay ventas negativas)."""
    y_scaled = np.asarray(y_scaled, float)
    y_log = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)
    return np.clip(np.expm1(y_log), 0, None)


@dataclass
class SeriesData:
    """Una serie lista para la GRU."""

    unique_id: str
    dates: pd.DatetimeIndex
    y: np.ndarray  # ventas en euros
    y_scaled: np.ndarray  # log1p + MinMax
    past: np.ndarray  # (n, len(PAST_COLS))
    fut: np.ndarray  # (n, len(FUT_COLS))
    scaler: MinMaxScaler

    def idx(self, ts: pd.Timestamp) -> int:
        """Posición de una fecha en la serie."""
        return int(self.dates.get_loc(pd.Timestamp(ts)))


def prepare_series(
    hist_uid: pd.DataFrame,
    fit_end: pd.Timestamp = TRAIN_END,
    scaler: MinMaxScaler | None = None,
) -> SeriesData:
    """Construye las matrices de entrada de un `unique_id` a partir de `history.parquet`."""
    h = hist_uid.sort_values("ds").set_index("ds")
    h.index = pd.DatetimeIndex(h.index)
    if h.index.has_duplicates:
        raise ValueError("Fechas duplicadas en la serie")
    if len(h) != len(pd.date_range(h.index[0], h.index[-1], freq="D")):
        raise ValueError("La serie no tiene frecuencia diaria completa")
    uid = str(h["unique_id"].iloc[0]) if "unique_id" in h.columns else "serie"
    y = h["y"].astype(float)
    scaler = scaler or fit_scaler(y, fit_end)
    y_scaled = scale_target(y.to_numpy(), scaler)
    fut = exog_features(h).to_numpy()
    past = np.column_stack([y_scaled, fut])
    return SeriesData(uid, h.index, y.to_numpy(), y_scaled, past, fut, scaler)


# --- Ventanas -------------------------------------------------------------------------------
def make_windows(arr_past, arr_fut_exog, y, lookback=LOOKBACK, horizon=HORIZON, end_idx=None):
    """Devuelve X_past (n,lookback,f), X_fut (n,horizon,k), Y (n,horizon).

    La ventana que empieza a predecir en la posición t usa `arr_past[t-lookback:t]`,
    `arr_fut_exog[t:t+horizon]` y tiene objetivo `y[t:t+horizon]`.
    Solo crea ventanas cuyo objetivo termina en o antes de end_idx (evita fuga).
    """
    arr_past, arr_fut_exog, y = np.asarray(arr_past), np.asarray(arr_fut_exog), np.asarray(y)
    n = len(y)
    end_idx = n - 1 if end_idx is None else min(int(end_idx), n - 1)
    last_t = end_idx - horizon + 1  # último inicio de objetivo permitido
    if last_t < lookback:
        f, k = arr_past.shape[1], arr_fut_exog.shape[1]
        return np.empty((0, lookback, f)), np.empty((0, horizon, k)), np.empty((0, horizon))
    starts = np.arange(lookback, last_t + 1)
    # sliding_window_view devuelve (n_ventanas, f, largo); se reordena a (n_ventanas, largo, f)
    past_w = sliding_window_view(arr_past, lookback, axis=0).transpose(0, 2, 1)
    fut_w = sliding_window_view(arr_fut_exog, horizon, axis=0).transpose(0, 2, 1)
    y_w = sliding_window_view(y, horizon)
    return (
        past_w[starts - lookback].copy(),
        fut_w[starts].copy(),
        y_w[starts].copy(),
    )


def train_val_windows(sd: SeriesData, lookback: int = LOOKBACK, horizon: int = HORIZON):
    """Ventanas de entrenamiento (objetivo <= TRAIN_END) y de validación (objetivo que termina
    entre TRAIN_END y VAL_END). Las de validación sirven para EarlyStopping."""
    i_train, i_val = sd.idx(TRAIN_END), sd.idx(VAL_END)
    tr = make_windows(sd.past, sd.fut, sd.y_scaled, lookback, horizon, end_idx=i_train)
    allw = make_windows(sd.past, sd.fut, sd.y_scaled, lookback, horizon, end_idx=i_val)
    n_tr = len(tr[2])
    va = tuple(a[n_tr:] for a in allw)
    return tr, va


def origin_input(
    arr_past: np.ndarray,
    arr_fut_exog: np.ndarray,
    origin_idx: int,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Entrada para pronosticar con origen fijo: los `lookback` días hasta `origin_idx`
    (incluido) y las exógenas de los `horizon` días siguientes. Forma de lote (1, …)."""
    if origin_idx + 1 < lookback:
        raise ValueError("No hay suficiente historia antes del origen")
    if origin_idx + 1 + horizon > len(arr_fut_exog):
        raise ValueError("Faltan exógenas futuras para el horizonte pedido")
    past = arr_past[origin_idx + 1 - lookback : origin_idx + 1]
    fut = arr_fut_exog[origin_idx + 1 : origin_idx + 1 + horizon]
    return past[None, ...], fut[None, ...]


def future_exog_matrix(uid: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Exógenas del periodo futuro con `build_future_exog` de A (solo se importa)."""
    from src.features.exog import build_future_exog  # type: ignore[import-not-found]

    return exog_features(build_future_exog(uid, dates)).to_numpy()


# --- Modelo (requiere TensorFlow) -----------------------------------------------------------
def build_model(
    n_feat: int = len(PAST_COLS),
    n_exog: int = len(FUT_COLS),
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    units: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
):
    """GRU(units) → Dropout → GRU(units/2), concatenada con las exógenas futuras aplanadas."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(SEED)
    past = keras.Input(shape=(lookback, n_feat), name="past")
    fut = keras.Input(shape=(horizon, n_exog), name="future_exog")
    x = layers.GRU(units, return_sequences=True)(past)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(max(units // 2, 8))(x)
    f = layers.Flatten()(fut)
    h = layers.Concatenate()([x, f])
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(horizon, name="y")(h)
    model = keras.Model([past, fut], out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.Huber())
    return model


def default_callbacks(patience: int = 15):
    """EarlyStopping (restaura los mejores pesos) + ReduceLROnPlateau."""
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=max(patience // 2, 3), factor=0.5),
    ]


def predict_mc(model, inputs, n_samples: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media e intervalo al 95 % por MC Dropout (dropout activo en inferencia).
    Devuelve (media, p2.5, p97.5) en la escala del modelo."""
    draws = np.stack([np.asarray(model(inputs, training=True)) for _ in range(n_samples)])
    return draws.mean(0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


# --- Salidas según el contrato --------------------------------------------------------------
def forecast_frame(
    unique_id: str,
    dates: pd.DatetimeIndex,
    y: np.ndarray | None,
    y_hat: np.ndarray,
    split: str,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> pd.DataFrame:
    """Filas de `gru.parquet` (sección 5.1) para un nivel y un split."""
    n = len(dates)
    nan = np.full(n, np.nan)
    return pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": pd.DatetimeIndex(dates),
            "y": nan if y is None else np.asarray(y, float),
            "y_hat": np.asarray(y_hat, float).reshape(n),
            "lo_95": nan if lo is None else np.asarray(lo, float).reshape(n),
            "hi_95": nan if hi is None else np.asarray(hi, float).reshape(n),
            "model": MODEL_NAME,
            "split": split,
        }
    )[FORECAST_COLS]


def metrics_payload(fc: pd.DataFrame) -> dict:
    """`gru.json` (sección 5.2) con `compute_all` sobre val y test."""
    levels: dict[str, dict] = {}
    for (uid, split), g in fc[fc["split"].isin(["val", "test"])].groupby(["unique_id", "split"]):
        levels.setdefault(str(uid), {})[split] = compute_all(g["y"], g["y_hat"])
    return {"model": MODEL_NAME, "levels": levels}


def validate_forecast(fc: pd.DataFrame) -> None:
    """Valida el esquema antes de escribir (lo mismo que comprueba test_contract_b)."""
    missing = set(FORECAST_COLS) - set(fc.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if fc["y_hat"].isna().any():
        raise ValueError("y_hat tiene NaN")
    if set(fc["model"]) != {MODEL_NAME}:
        raise ValueError("gru.parquet solo puede tener model == 'gru'")
    if not set(fc["split"]) <= {"val", "test", "future"}:
        raise ValueError("split no válido")
    sizes = fc.groupby(["unique_id", "split"]).size()
    if not (sizes == HORIZON).all():
        raise ValueError(f"Cada unique_id y split debe tener {HORIZON} filas")


def write_artifacts(fc: pd.DataFrame, root: Path = ARTIFACTS) -> dict:
    """Escribe `forecasts/gru.parquet` y `metrics/gru.json`. Devuelve las métricas."""
    validate_forecast(fc)
    metrics = metrics_payload(fc)
    (root / "forecasts").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    fc[FORECAST_COLS].to_parquet(root / "forecasts" / f"{MODEL_NAME}.parquet", index=False)
    (root / "metrics" / f"{MODEL_NAME}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def save_scaler(scaler: MinMaxScaler, path: Path = ARTIFACTS / "models" / "gru_scaler.joblib"):
    """Guarda el escalador (archivo de Persona B)."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def log_experiment(
    params: dict,
    val_metrics: dict[str, float],
    path: Path = ROOT / "experiments" / "experiments.csv",
    model: str = MODEL_NAME,
) -> None:
    """Agrega UNA fila a `experiments/experiments.csv` (compartido: solo agregar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["fecha", "modelo", "parametros", "mape_val", "rmse_val", "mae_val"])
        w.writerow(
            [
                date.today().isoformat(),
                model,
                json.dumps(params, sort_keys=True),
                f"{val_metrics['mape']:.4f}",
                f"{val_metrics['rmse']:.2f}",
                f"{val_metrics.get('mae', float('nan')):.2f}",
            ]
        )


def split_dates(origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DatetimeIndex:
    """Las `horizon` fechas siguientes a un origen."""
    return pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=horizon, freq="D")


ORIGINS = {"val": TRAIN_END, "test": VAL_END, "future": TEST_END}
