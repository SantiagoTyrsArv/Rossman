"""Baseline ingenuo estacional: y_hat(t) = y(t − 7), recursivo sobre el horizonte.

Uso: `python -m src.models.baseline` → `artifacts/forecasts/naive.parquet` y
`artifacts/metrics/naive.json`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON, SEASON
from src.evaluation.backtest import (
    FORECAST_COLS,
    ORIGINS,
    metrics_payload,
    write_forecast,
    write_metrics,
)
from src.features.exog import load_history

MODEL = "naive"


def seasonal_naive(
    y: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON, season: int = SEASON
) -> pd.Series:
    """Repite la última semana conocida en `origin`; no usa nada posterior."""
    past = y.loc[:origin]
    if len(past) < season or past.index[-1] != origin:
        raise ValueError(f"La serie debe llegar a {origin.date()} con al menos {season} días")
    idx = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(np.resize(past.to_numpy(float)[-season:], horizon), index=idx)


def make_forecasts(history: pd.DataFrame) -> pd.DataFrame:
    """Pronósticos de val, test y future para todos los niveles de `history`."""
    frames = []
    for uid, g in history.groupby("unique_id"):
        y = g.set_index("ds")["y"].sort_index().astype(float)
        for name, origin in ORIGINS.items():
            y_hat = seasonal_naive(y, origin)
            frames.append(
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": y_hat.index,
                        "y": y.reindex(y_hat.index).to_numpy(),
                        "y_hat": y_hat.to_numpy(),
                        "lo_95": np.nan,
                        "hi_95": np.nan,
                        "model": MODEL,
                        "split": name,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[FORECAST_COLS]


def main() -> None:
    fc = make_forecasts(load_history())
    print(f"OK → {write_forecast(fc, MODEL)}")
    print(f"OK → {write_metrics(metrics_payload(fc, MODEL), MODEL)}")


if __name__ == "__main__":
    main()
