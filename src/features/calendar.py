"""Variables de calendario."""

from __future__ import annotations

import numpy as np
import pandas as pd

FOURIER_K = 3


def fourier_terms(
    dates: pd.DatetimeIndex, K: int = FOURIER_K, period: float = 365.25
) -> pd.DataFrame:
    """Pares seno/coseno de la estacionalidad anual (`f_sin1`, `f_cos1`, …)."""
    dates = pd.DatetimeIndex(dates)
    t = dates.dayofyear.to_numpy()
    cols = {}
    for k in range(1, K + 1):
        cols[f"f_sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"f_cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=dates)


def fourier_cols(K: int = FOURIER_K) -> list[str]:
    """Nombres de las columnas de Fourier en orden."""
    return [c for k in range(1, K + 1) for c in (f"f_sin{k}", f"f_cos{k}")]
