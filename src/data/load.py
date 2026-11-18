"""Lectura de los CSV crudos de Kaggle (Rossmann Store Sales)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_RAW

RAW_FILES = ("train.csv", "store.csv", "test.csv")


def load_raw(raw_dir: Path = DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, store). `StateHoliday` se lee como texto (mezcla 0 y '0')."""
    train = pd.read_csv(
        Path(raw_dir) / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    store = pd.read_csv(Path(raw_dir) / "store.csv")
    return train, store


def load_test(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """`test.csv`: solo como referencia de calendario futuro (no tiene Sales)."""
    return pd.read_csv(
        Path(raw_dir) / "test.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
