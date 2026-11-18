"""Contrato de los artifacts de Persona A: pronósticos, métricas y modelos del simulador.

Carga cada modelo como lo hace `app/utils/simulator.py`. Se salta si aún no existen.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from statsmodels.iolib.smpickle import load_pickle

from src.config import ARTIFACTS, FUTURE_END, FUTURE_START, HORIZON, TEST_END
from src.evaluation.backtest import METRIC_KEYS, validate_forecast
from src.features.exog import EXOG_COLS, all_levels, build_future_exog

MODELS_DIR = ARTIFACTS / "models"
MAX_PKL_MB = 25
LEVELS = all_levels()
FUTURE = pd.date_range(FUTURE_START, FUTURE_END, freq="D")

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "sarimax_total.pkl").exists(), reason="sarimax_*.pkl todavía no existen"
)


@pytest.mark.parametrize("model", ["naive", "sarima", "sarimax"])
def test_forecast_and_metrics_follow_contract(model):
    fc = validate_forecast(pd.read_parquet(ARTIFACTS / "forecasts" / f"{model}.parquet"), model)
    assert set(fc["unique_id"]) == set(LEVELS)
    expected = {"val", "test"} if model == "sarima" else {"val", "test", "future"}
    assert set(fc["split"]) == expected
    payload = json.loads((ARTIFACTS / "metrics" / f"{model}.json").read_text("utf-8"))
    assert payload["model"] == model and set(payload["levels"]) == set(LEVELS)
    for splits in payload["levels"].values():
        assert set(splits) == {"val", "test"}
        assert all(
            set(v) == METRIC_KEYS and np.isfinite(list(v.values())).all() for v in splits.values()
        )


@pytest.mark.parametrize("uid", LEVELS)
def test_simulator_model_forecasts_like_the_app(uid):
    path = MODELS_DIR / f"sarimax_{uid}.pkl"
    assert path.stat().st_size < MAX_PKL_MB * 1024**2
    meta = json.loads((MODELS_DIR / f"sarimax_{uid}_meta.json").read_text("utf-8"))
    assert meta["unique_id"] == uid and meta["exog_cols"] == EXOG_COLS
    assert meta["log_target"] is True and meta["train_end"] == TEST_END.strftime("%Y-%m-%d")

    res = load_pickle(path)
    x = build_future_exog(uid, FUTURE)
    assert list(x[meta["exog_cols"]].columns) == EXOG_COLS
    fc = res.get_forecast(steps=len(x), exog=x[meta["exog_cols"]])
    y_hat = np.expm1(np.asarray(fc.predicted_mean, float))
    assert len(y_hat) == HORIZON and np.isfinite(y_hat).all()
    # Con la tienda cerrada la venta esperada es 0: ahí se admite un residuo de centavos
    is_open = x["open_share"].to_numpy() > 0
    assert (y_hat[is_open] > 0).all()
    assert (y_hat[~is_open] > -1).all()


@pytest.mark.parametrize("uid", LEVELS)
def test_promo_increases_simulated_sales(uid):
    meta = json.loads((MODELS_DIR / f"sarimax_{uid}_meta.json").read_text("utf-8"))
    res = load_pickle(MODELS_DIR / f"sarimax_{uid}.pkl")
    weekdays = {d for d in FUTURE if d.dayofweek < 5}
    base = build_future_exog(uid, FUTURE, promo_days=set())[meta["exog_cols"]]
    promo = build_future_exog(uid, FUTURE, promo_days=weekdays)[meta["exog_cols"]]
    without = np.expm1(res.get_forecast(steps=HORIZON, exog=base).predicted_mean).sum()
    with_promo = np.expm1(res.get_forecast(steps=HORIZON, exog=promo).predicted_mean).sum()
    assert with_promo > without
