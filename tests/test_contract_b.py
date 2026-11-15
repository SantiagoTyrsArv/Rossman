"""Contrato de los artifacts de Persona B (`gru.parquet` y `gru.json`).

Se saltan mientras la GRU no haya generado sus artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.gru import ARTIFACTS as ART
from src.models.gru import FORECAST_COLS, HORIZON

FC = ART / "forecasts" / "gru.parquet"
MET = ART / "metrics" / "gru.json"

pytestmark = pytest.mark.skipif(not FC.exists(), reason="gru.parquet todavía no existe")


@pytest.fixture(scope="module")
def fc() -> pd.DataFrame:
    return pd.read_parquet(FC)


def test_schema(fc):
    assert list(fc.columns) == FORECAST_COLS
    assert pd.api.types.is_datetime64_any_dtype(fc["ds"])
    assert set(fc["model"]) == {"gru"}
    assert set(fc["split"]) <= {"val", "test", "future"}


def test_42_rows_per_level_and_split(fc):
    assert (fc.groupby(["unique_id", "split"]).size() == HORIZON).all()


def test_no_nan_in_y_hat(fc):
    assert fc["y_hat"].notna().all()
    assert np.isfinite(fc["y_hat"]).all()
    assert (fc["y_hat"] >= 0).all()


def test_future_has_no_actuals(fc):
    assert fc.loc[fc["split"] == "future", "y"].isna().all()


def test_metrics_json_matches_levels(fc):
    met = json.loads(MET.read_text(encoding="utf-8"))
    assert met["model"] == "gru"
    evaluated = set(fc.loc[fc["split"] != "future", "unique_id"])
    assert set(met["levels"]) == evaluated
    for splits in met["levels"].values():
        for vals in splits.values():
            assert {"rmse", "mae", "mape"} <= set(vals)
