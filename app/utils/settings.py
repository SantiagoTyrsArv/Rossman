"""Constantes del proyecto para la app.

Usa `src/config.py` (compartido, lo crea Persona A). Mientras no exista, usa una copia local de
la sección 5.6 del contrato (ver P-01 en docs/PROPUESTAS.md). TODO quitar el respaldo cuando
`src/config.py` esté en `dev`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.config import (  # type: ignore[import-not-found]
        ARTIFACTS,
        FUTURE_END,
        FUTURE_START,
        HORIZON,
        LEVELS_TYPES,
        MODELS,
        ROOT,
        SEASON,
        SELECTED_STORES,
        TEST_END,
        TEST_START,
        TRAIN_END,
        TRAIN_START,
        VAL_END,
        VAL_START,
    )

    CONFIG_FROM_SRC = True
except ImportError:  # respaldo local (P-01)
    ROOT = Path(__file__).resolve().parents[2]
    ARTIFACTS = ROOT / "artifacts"
    TRAIN_START = pd.Timestamp("2013-01-01")
    TRAIN_END = pd.Timestamp("2015-05-08")
    VAL_START, VAL_END = pd.Timestamp("2015-05-09"), pd.Timestamp("2015-06-19")
    TEST_START, TEST_END = pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31")
    FUTURE_START, FUTURE_END = pd.Timestamp("2015-08-01"), pd.Timestamp("2015-09-11")
    HORIZON = 42
    SEASON = 7
    LEVELS_TYPES = ["type_a", "type_b", "type_c", "type_d"]
    SELECTED_STORES: list[int] = []
    MODELS = ["naive", "sarima", "sarimax", "gru"]
    CONFIG_FROM_SRC = False

# Tiendas de relleno para los datos de prueba mientras A no defina SELECTED_STORES
MOCK_STORES = [1, 2, 3, 4, 5]

SPLITS = ["val", "test", "future"]
SPLIT_START = {"val": VAL_START, "test": TEST_START, "future": FUTURE_START}

MODEL_LABELS = {
    "naive": "Ingenuo estacional",
    "sarima": "SARIMA",
    "sarimax": "SARIMAX",
    "gru": "GRU",
}


DOW_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MONTH_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def store_ids() -> list[int]:
    """Tiendas que muestra la app: las de config o, si aún no hay, las de prueba."""
    return list(SELECTED_STORES) or MOCK_STORES


def all_levels() -> list[str]:
    """Todos los `unique_id` del contrato en orden de presentación."""
    return ["total", *LEVELS_TYPES, *[f"store_{n}" for n in store_ids()]]


def level_label(uid: str) -> str:
    """Nombre legible de un `unique_id`."""
    if uid == "total":
        return "Total cadena"
    if uid.startswith("type_"):
        return f"Tipo de tienda {uid[-1].upper()}"
    if uid.startswith("store_"):
        return f"Tienda {uid.split('_', 1)[1]}"
    return uid
