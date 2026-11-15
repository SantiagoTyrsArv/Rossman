"""Constantes compartidas del proyecto (contrato, sección 5.6).

Archivo COMPARTIDO: cualquier cambio requiere el visto bueno de Persona A y Persona B.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2015-05-08")
VAL_START, VAL_END = pd.Timestamp("2015-05-09"), pd.Timestamp("2015-06-19")
TEST_START, TEST_END = pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31")
FUTURE_START, FUTURE_END = pd.Timestamp("2015-08-01"), pd.Timestamp("2015-09-11")
HORIZON = 42
SEASON = 7
SEED = 42

LEVELS_TYPES = ["type_a", "type_b", "type_c", "type_d"]
SELECTED_STORES: list[int] = []  # los define Persona A tras el EDA (tarea A2)
MODELS = ["naive", "sarima", "sarimax", "gru"]
