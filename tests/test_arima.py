"""Tests del flujo SARIMA/SARIMAX con una historia sintética (tareas A6 y A7)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from statsmodels.iolib.smpickle import load_pickle

from src.config import FUTURE_END, FUTURE_START, HORIZON, TEST_END, VAL_END
from src.evaluation import backtest
from src.features import exog as exog_module
from src.features.exog import EXOG_COLS, HISTORY_COLS
from src.models import arima

SPEC = arima.Spec((0, 0, 1), (0, 1, 1, 7))


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ds = pd.date_range("2014-06-01", TEST_END, freq="D")
    week = ds.isocalendar().week.to_numpy(int)
    promo = ((ds.dayofweek < 5) & (week % 2 == 0)).astype(float)
    open_share = np.where(ds.dayofweek == 6, 0.1, 1.0)
    y = 1000 * open_share * (1 + 0.3 * promo) * rng.lognormal(0, 0.03, len(ds))
    return pd.DataFrame(
        {
            "unique_id": "total",
            "ds": ds,
            "y": y,
            "promo_share": promo,
            "open_share": open_share,
            "school_holiday_share": (ds.month == 7).astype(float),
            "state_holiday": "0",
            "promo2_share": 0.2,
        }
    )[HISTORY_COLS]


@pytest.fixture()
def patched_history(history, monkeypatch):
    monkeypatch.setattr(exog_module, "load_history", lambda: history)
    return history


def test_candidate_grid_matches_guide():
    specs = arima.candidate_specs(d=0)
    assert len(specs) == 3 * 3 * 2 * 2 and len(set(specs)) == len(specs)
    assert {s.order[1] for s in specs} == {0} and {s.seasonal_order[1] for s in specs} == {1}
    assert {s.seasonal_order[3] for s in specs} == {7}


def test_choose_d_detects_random_walk():
    # La regla marca d = 1 en ~6 % del ruido blanco y ~98 % de las caminatas aleatorias
    idx = pd.date_range("2014-01-01", periods=500, freq="D")
    noise = np.random.default_rng(0).normal(size=500)
    assert arima.choose_d(pd.Series(noise, index=idx)) == 0
    assert arima.choose_d(pd.Series(noise.cumsum(), index=idx)) == 1


def test_pick_best_uses_validation_among_aic_finalists():
    rows = [
        {"source": "grid", "valid": True, "aic": 1.0, "rmse": 50.0},
        {"source": "grid", "valid": True, "aic": 2.0, "rmse": 40.0},
        {"source": "grid", "valid": True, "aic": 3.0, "rmse": 1.0},  # fuera del top 2 por AIC
        {"source": "grid", "valid": False, "aic": 0.0, "rmse": 0.0},
        {"source": "auto_arima", "valid": True, "aic": 9.0, "rmse": 45.0},
    ]
    assert arima.pick_best(rows, top=2)["rmse"] == 40.0
    with pytest.raises(RuntimeError):
        arima.pick_best([rows[3]])


def test_training_target_masks_full_closures_only_with_partial_opening():
    idx = pd.date_range("2015-01-01", periods=20, freq="D")
    y = pd.Series(100.0, index=idx)
    y.iloc[0] = 0.0
    share = pd.Series(1.0, index=idx)
    share.iloc[0] = 0.0
    assert arima.training_target(y, pd.DataFrame({"open_share": share})).equals(y)  # tienda
    share.iloc[1:4] = 0.02
    masked = arima.training_target(y, pd.DataFrame({"open_share": share}))
    assert np.isnan(masked.iloc[0]) and masked.iloc[1:].eq(100.0).all()


def test_forecast_is_zero_when_everything_is_closed(history):
    y, x = arima.level_data(history, "total")
    res = arima.fit_model(y.loc[:VAL_END], x.loc[:VAL_END], SPEC)
    future = x.iloc[-HORIZON:].copy()
    future.iloc[0, future.columns.get_loc("open_share")] = 0.0
    fc = arima.forecast_euros(res, HORIZON, future)
    assert fc.iloc[0].eq(0).all() and fc["y_hat"].iloc[1:].gt(0).all()


def test_compact_model_keeps_forecast_and_is_small(history, tmp_path):
    y, x = arima.level_data(history, "total")
    res = arima.fit_model(y, x, SPEC)
    future = x.iloc[-HORIZON:].set_axis(pd.date_range(FUTURE_START, FUTURE_END))
    full = res.get_forecast(HORIZON, exog=future)
    path = arima.save_simulator_model(res, "total", SPEC, tmp_path)
    small = load_pickle(path).get_forecast(HORIZON, exog=future)
    np.testing.assert_allclose(small.predicted_mean, full.predicted_mean)
    np.testing.assert_allclose(small.conf_int(), full.conf_int())
    assert path.stat().st_size < 2 * 1024**2


def test_evaluate_candidate_ignores_data_after_validation(history):
    y, x = arima.level_data(history, "total")
    changed = y.copy()
    changed.loc[y.index > VAL_END] = 1e12
    base = arima.evaluate_candidate("total", "sarimax", SPEC, y, x)
    other = arima.evaluate_candidate("total", "sarimax", SPEC, changed, x)
    assert base["valid"] and base["rmse"] == pytest.approx(other["rmse"])


def test_select_does_not_receive_test_period(history, tmp_path, monkeypatch):
    seen = []

    def fake_search(uid, model, y, exog):
        seen.append(y.index.max())
        return [arima.evaluate_candidate(uid, model, SPEC, y, exog)]

    monkeypatch.setattr(arima, "_search_level", fake_search)
    selection = arima.select(
        history,
        n_jobs=1,
        selection_file=tmp_path / "sel.json",
        grid_file=tmp_path / "grid.csv",
        experiments_file=tmp_path / "exp.csv",
    )
    assert max(seen) == VAL_END
    assert set(selection["levels"]["total"]) == {"sarima", "sarimax"}
    assert len(pd.read_csv(tmp_path / "exp.csv")) == 2


def test_final_meets_contract_and_simulator_usage(patched_history, tmp_path):
    cfg = {"order": list(SPEC.order), "seasonal_order": list(SPEC.seasonal_order)}
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"levels": {"total": {"sarima": cfg, "sarimax": cfg}}}))
    out = arima.final(patched_history, n_jobs=1, selection_file=sel, root=tmp_path)

    assert set(out["sarima"]["split"]) == {"val", "test"}
    assert set(out["sarimax"]["split"]) == {"val", "test", "future"}
    for model in ("sarima", "sarimax"):
        back = pd.read_parquet(tmp_path / "forecasts" / f"{model}.parquet")
        backtest.validate_forecast(back, model)
        assert json.loads((tmp_path / "metrics" / f"{model}.json").read_text())["levels"]["total"]

    meta = json.loads((tmp_path / "models" / "sarimax_total_meta.json").read_text())
    assert meta["exog_cols"] == EXOG_COLS and meta["train_end"] == "2015-07-31"
    assert meta["log_target"] is True
    res = load_pickle(tmp_path / "models" / "sarimax_total.pkl")
    assert res.model.data.dates[-1] == TEST_END
    dates = pd.date_range(FUTURE_START, FUTURE_END)
    x = exog_module.build_future_exog("total", dates)
    fc = res.get_forecast(steps=len(x), exog=x[meta["exog_cols"]])
    assert len(fc.predicted_mean) == HORIZON and np.isfinite(fc.predicted_mean).all()


def test_promo_scenario_raises_sarimax_forecast(patched_history, tmp_path):
    y, x = arima.level_data(patched_history, "total")
    res = arima.fit_model(y, x, SPEC)
    dates = pd.date_range(FUTURE_START, FUTURE_END)
    weekdays = {d for d in dates if d.dayofweek < 5}
    base = exog_module.build_future_exog("total", dates, promo_days=set())
    promo = exog_module.build_future_exog("total", dates, promo_days=weekdays)
    no = arima.forecast_euros(res, len(dates), base)["y_hat"].sum()
    yes = arima.forecast_euros(res, len(dates), promo)["y_hat"].sum()
    assert yes > no * 1.1
