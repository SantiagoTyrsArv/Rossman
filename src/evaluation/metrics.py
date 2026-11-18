"""Métricas de evaluación. Las usan todos los modelos (contrato, sección 5.2)."""

from __future__ import annotations

import numpy as np


def _arrays(y, y_hat) -> tuple[np.ndarray, np.ndarray]:
    """Convierte a arreglos float y exige misma forma, sin vacíos ni valores no finitos."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    if y.shape != y_hat.shape:
        raise ValueError(f"Formas distintas: {y.shape} y {y_hat.shape}")
    if y.size == 0:
        raise ValueError("No hay observaciones para evaluar")
    if not (np.isfinite(y).all() and np.isfinite(y_hat).all()):
        raise ValueError("Las métricas no admiten NaN ni infinitos")
    return y, y_hat


def rmse(y, y_hat) -> float:
    """Raíz del error cuadrático medio."""
    y, y_hat = _arrays(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y, y_hat) -> float:
    """Error absoluto medio."""
    y, y_hat = _arrays(y, y_hat)
    return float(np.mean(np.abs(y - y_hat)))


def mape(y, y_hat) -> float:
    """MAPE en %, solo sobre posiciones con y > 0. NaN si ninguna cumple."""
    y, y_hat = _arrays(y, y_hat)
    m = y > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - y_hat[m]) / y[m])) * 100)


def compute_all(y, y_hat) -> dict[str, float]:
    """RMSE, MAE y MAPE con las claves del contrato."""
    return {"rmse": rmse(y, y_hat), "mae": mae(y, y_hat), "mape": mape(y, y_hat)}
