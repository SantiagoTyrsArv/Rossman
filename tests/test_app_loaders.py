"""Tests de los loaders de la app y de los datos de prueba (Persona B)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app import mock_data
from app.utils import loaders
from app.utils.settings import HORIZON, MODELS, all_levels


@pytest.fixture(scope="module")
def mock_hist() -> pd.DataFrame:
    return mock_data.history()


@pytest.fixture(scope="module")
def mock_fc(mock_hist) -> pd.DataFrame:
    return mock_data.forecasts(mock_hist)


def test_mock_history_schema(mock_hist):
    df = loaders.validate_history(mock_hist)
    assert list(df.columns) == loaders.HISTORY_COLS
    assert set(df["unique_id"]) == set(all_levels())
    assert df["ds"].min() == pd.Timestamp("2013-01-01")
    assert df["ds"].max() == pd.Timestamp("2015-07-31")
    assert set(df["state_holiday"]) <= {"0", "a", "b", "c"}
    assert not df[loaders.HISTORY_COLS].drop(columns="y").isna().any().any()


def test_mock_total_is_sum_of_types(mock_hist):
    piv = mock_hist.pivot_table(index="ds", columns="unique_id", values="y")
    types = piv[["type_a", "type_b", "type_c", "type_d"]].sum(axis=1)
    np.testing.assert_allclose(piv["total"], types, rtol=1e-9)


def test_mock_forecasts_schema(mock_fc):
    df = loaders.validate_forecasts(mock_fc)
    assert list(df.columns) == loaders.FORECAST_COLS
    assert set(df["model"]) == set(MODELS)
    counts = df.groupby(["unique_id", "model", "split"]).size()
    assert (counts == HORIZON).all()
    assert df.loc[df["split"] == "future", "y"].isna().all()
    assert df.loc[df["split"] != "future", "y"].notna().all()


def test_mock_metrics_roundtrip(mock_fc):
    met = loaders.metrics_to_frame(mock_data.metrics(mock_fc))
    assert set(met.columns) == {"model", "unique_id", "split", "rmse", "mae", "mape"}
    assert set(met["split"]) == {"val", "test"}
    assert (met[["rmse", "mae", "mape"]] >= 0).all().all()


def test_read_forecasts_concatenates_files(tmp_path, mock_fc):
    for m in ["naive", "gru"]:
        mock_fc[mock_fc["model"] == m].to_parquet(tmp_path / f"{m}.parquet")
    df = loaders.read_forecasts(tmp_path)
    assert set(df["model"]) == {"naive", "gru"}


def test_read_forecasts_empty_dir_returns_none(tmp_path):
    assert loaders.read_forecasts(tmp_path) is None
    assert loaders.read_metrics(tmp_path) is None
    assert loaders.read_history(tmp_path / "history.parquet") is None


def test_parquet_missing_columns_raises(tmp_path, mock_fc):
    mock_fc.drop(columns=["lo_95", "split"]).to_parquet(tmp_path / "roto.parquet")
    with pytest.raises(loaders.ContractError, match="faltan columnas"):
        loaders.read_forecasts(tmp_path)


def test_invalid_split_raises(mock_fc):
    bad = mock_fc.head(5).assign(split="train")
    with pytest.raises(loaders.ContractError, match="split"):
        loaders.validate_forecasts(bad)


def test_nan_y_hat_raises(mock_fc):
    bad = mock_fc.head(5).copy()
    bad.loc[bad.index[0], "y_hat"] = np.nan
    with pytest.raises(loaders.ContractError, match="NaN"):
        loaders.validate_forecasts(bad)


def test_read_metrics_skips_comparison(tmp_path):
    payload = {"model": "naive", "levels": {"total": {"val": {"rmse": 1, "mae": 1, "mape": 1}}}}
    (tmp_path / "naive.json").write_text(json.dumps(payload))
    (tmp_path / "comparison.json").write_text(json.dumps({"p_value": 0.01}))
    met = loaders.read_metrics(tmp_path)
    assert list(met["model"]) == ["naive"]


def test_incomplete_metrics_raise():
    payload = {"model": "gru", "levels": {"total": {"test": {"rmse": 1.0}}}}
    with pytest.raises(loaders.ContractError):
        loaders.metrics_to_frame([payload])


def test_mock_simulate_promo_increases_sales():
    dates = pd.date_range("2015-08-01", periods=HORIZON, freq="D")
    none = mock_data.simulate("type_a", HORIZON, promo_days=set())
    full = mock_data.simulate("type_a", HORIZON, promo_days={d for d in dates if d.dayofweek < 5})
    assert full["scen"].sum() > none["scen"].sum()
    hist_pattern = mock_data.simulate("total", HORIZON)
    np.testing.assert_allclose(hist_pattern["base"], hist_pattern["scen"])
