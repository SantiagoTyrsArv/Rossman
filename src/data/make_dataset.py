"""Regenera `data/processed/rossmann.parquet` desde cero.

Uso: `python -m src.data.make_dataset`  (descarga de Kaggle si faltan los CSV).
Requisitos: `~/.kaggle/kaggle.json` (o variables KAGGLE_USERNAME/KAGGLE_KEY) y haber aceptado
las reglas de la competencia en https://www.kaggle.com/competitions/rossmann-store-sales/rules
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_RAW
from src.data.clean import clean, impute_missing_store_days
from src.data.load import RAW_FILES, load_raw

COMPETITION = "rossmann-store-sales"
PROCESSED_FILE = DATA_PROCESSED / "rossmann.parquet"


def download(raw_dir: Path = DATA_RAW) -> None:
    """Descarga y descomprime los CSV con la API de Kaggle si no están."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f).exists() for f in RAW_FILES):
        return
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(raw_dir),
    ]
    subprocess.run(cmd, check=True)
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as fh:
            fh.extractall(raw_dir)


def build(raw_dir: Path = DATA_RAW, out: Path = PROCESSED_FILE) -> Path:
    """Carga, limpia, rellena huecos y guarda el parquet procesado."""
    train, store = load_raw(raw_dir)
    df = impute_missing_store_days(clean(train, store))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main() -> None:
    download()
    out = build()
    print(f"OK → {out}")


if __name__ == "__main__":
    main()
