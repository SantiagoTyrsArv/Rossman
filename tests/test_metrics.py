"""Tests de métricas, escritura de artifacts y baseline (tarea A5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import FUTURE_START, HORIZON, TEST_START, VAL_END, VAL_START
from src.evaluation import backtest
from src.evaluation.metrics import compute_all, mae, mape, rmse
from src.models.baseline import make_forecasts, seasonal_naive


def _forecast(split: str = "val", start: pd.Timestamp = VAL_START) -> pd.DataFrame:
    ds = pd.date_range(start, periods=HORIZON, freq="D")
    return pd.DataFrame(
        {
            "unique_id": "total",
            "ds": ds,
            "y": np.nan if split == "future" else 100.0,
            "y_hat": 90.0,
            "lo_95": 80.0,
            "hi_95": 110.0,
            "model": "naive",
            "split": split,
        }
    )


def test_metrics_hand_computed():
    y, y_hat = [100, 200], [110, 170]
    assert rmse(y, y_hat) == pytest.approx(np.sqrt((100 + 900) / 2))
    assert mae(y, y_hat) == pytest.approx(20)
    assert mape(y, y_hat) == pytest.approx((10 + 15) / 2)
    assert set(compute_all(y, y_hat)) == {"rmse", "mae", "mape"}


def test_mape_ignores_zero_sales():
    assert mape([0, 100], [50, 110]) == pytest.approx(10)
    assert np.isnan(mape([0, 0], [1, 2]))


@pytest.mark.parametrize(("y", "y_hat"), [([1, 2], [1]), ([], []), ([1, np.nan], [1, 2])])
def test_metrics_reject_bad_input(y, y_hat):
    with pytest.raises(ValueError):
        rmse(y, y_hat)


def test_split_uses_config_dates():
    ds = pd.date_range("2015-05-01", "2015-07-31", freq="D")
    parts = backtest.split(pd.DataFrame({"ds": ds, "y": 1.0}))
    assert parts["train"]["ds"].max() == VAL_START - pd.Timedelta(days=1)
    assert parts["val"]["ds"].max() == VAL_END and len(parts["val"]) == HORIZON
    assert parts["test"]["ds"].min() == TEST_START and len(parts["test"]) == HORIZON


def test_write_forecast_roundtrip(tmp_path):
    df = pd.concat([_forecast(), _forecast("future", FUTURE_START)], ignore_index=True)
    path = backtest.write_forecast(df, "naive", tmp_path)
    back = pd.read_parquet(path)
    assert list(back.columns) == backtest.FORECAST_COLS and len(back) == 2 * HORIZON


def test_write_forecast_rejects_missing_column(tmp_path):
    with pytest.raises(ValueError, match="Faltan columnas"):
        backtest.write_forecast(_forecast().drop(columns="y_hat"), "naive", tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"y_hat": -1.0},
        {"y_hat": np.nan},
        {"split": "train"},
        {"model": "sarima"},
        {"lo_95": 500.0},
        {"lo_95": np.nan},
        {"y": np.nan},
    ],
)
def test_write_forecast_rejects_bad_values(tmp_path, change):
    with pytest.raises(ValueError):
        backtest.write_forecast(_forecast().assign(**change), "naive", tmp_path)


def test_write_forecast_rejects_wrong_dates(tmp_path):
    with pytest.raises(ValueError, match="fechas"):
        backtest.write_forecast(_forecast().iloc[:-1], "naive", tmp_path)


def test_future_must_not_have_actuals(tmp_path):
    bad = _forecast("future", FUTURE_START).assign(y=1.0)
    with pytest.raises(ValueError, match="NaN en future"):
        backtest.write_forecast(bad, "naive", tmp_path)


def test_cannot_write_artifacts_of_persona_b(tmp_path):
    with pytest.raises(ValueError, match="Persona A"):
        backtest.write_forecast(_forecast().assign(model="gru"), "gru", tmp_path)
    with pytest.raises(ValueError, match="Persona A"):
        backtest.write_metrics({"model": "gru", "levels": {"total": {}}}, "gru", tmp_path)


def test_write_metrics_roundtrip_and_schema(tmp_path):
    payload = backtest.metrics_payload(_forecast(), "naive")
    path = backtest.write_metrics(payload, "naive", tmp_path)
    assert json.loads(path.read_text("utf-8"))["levels"]["total"]["val"]["mae"] == 10
    payload["levels"]["total"]["val"].pop("mae")
    with pytest.raises(ValueError):
        backtest.write_metrics(payload, "naive", tmp_path)


def test_seasonal_naive_is_recursive():
    idx = pd.date_range("2015-01-01", periods=14, freq="D")
    y = pd.Series(np.arange(14.0), index=idx)
    fc = seasonal_naive(y, idx[-1], horizon=15)
    assert list(fc.iloc[:7]) == list(range(7, 14))
    assert list(fc.iloc[7:14]) == list(range(7, 14)) and fc.iloc[14] == 7
    assert fc.index[0] == idx[-1] + pd.Timedelta(days=1)


def test_seasonal_naive_ignores_data_after_origin():
    idx = pd.date_range("2015-01-01", periods=30, freq="D")
    y = pd.Series(np.arange(30.0), index=idx)
    changed = y.copy()
    changed.iloc[15:] = 1e9
    pd.testing.assert_series_equal(seasonal_naive(y, idx[14]), seasonal_naive(changed, idx[14]))


def test_seasonal_naive_requires_origin_in_series():
    idx = pd.date_range("2015-01-01", periods=10, freq="D")
    with pytest.raises(ValueError):
        seasonal_naive(pd.Series(1.0, index=idx), idx[-1] + pd.Timedelta(days=1))


def test_make_forecasts_meets_contract(tmp_path):
    ds = pd.date_range("2015-01-01", "2015-07-31", freq="D")
    hist = pd.DataFrame({"unique_id": "total", "ds": ds, "y": 100.0 + ds.dayofweek})
    fc = make_forecasts(hist)
    assert backtest.write_forecast(fc, "naive", tmp_path).exists()
    assert set(fc["split"]) == {"val", "test", "future"}
    assert (fc["y_hat"] == 100.0 + fc["ds"].dt.dayofweek).all()
