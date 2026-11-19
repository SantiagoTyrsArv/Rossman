"""Tests de la evaluación final (tarea A8)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON, TEST_START
from src.evaluation import compare


def _fc(model: str, error: float) -> pd.DataFrame:
    ds = pd.date_range(TEST_START, periods=HORIZON, freq="D")
    wobble = np.resize([0.0, 1.0, 2.0], HORIZON)
    y = 1000.0 + np.arange(HORIZON)
    return pd.DataFrame(
        {
            "unique_id": "total",
            "ds": ds,
            "y": y,
            "y_hat": y + error + wobble,
            "lo_95": np.nan,
            "hi_95": np.nan,
            "model": model,
            "split": "test",
        }
    )


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "forecasts").mkdir()
    (tmp_path / "metrics").mkdir()
    for model, error, rmse in (("naive", 100.0, 100.0), ("sarimax", 5.0, 6.0)):
        _fc(model, error).to_parquet(tmp_path / "forecasts" / f"{model}.parquet")
        vals = {"rmse": rmse, "mae": error, "mape": error / 10}
        payload = {"model": model, "levels": {"total": {"test": vals}}}
        (tmp_path / "metrics" / f"{model}.json").write_text(json.dumps(payload))
    return tmp_path


@pytest.fixture()
def history():
    ds = pd.date_range(TEST_START, periods=HORIZON, freq="D")
    return pd.DataFrame(
        {
            "unique_id": "total",
            "ds": ds,
            "promo_share": (ds.dayofweek < 2).astype(float),
            "open_share": 1.0,
            "state_holiday": "0",
        }
    )


def test_final_table_flags_baseline_and_goal(root):
    table = compare.final_table(compare.load_metrics(root)).set_index("model")
    assert bool(table.at["sarimax", "beats_baseline"]) is True
    assert bool(table.at["naive", "beats_baseline"]) is False
    assert bool(table.at["sarimax", "meets_mape_goal"]) is True


def test_wilcoxon_detects_better_model(root):
    result = compare.wilcoxon_test(compare.load_forecasts(root), "sarimax", "naive")
    assert result["status"] == "ok" and result["n"] == HORIZON
    assert result["significant_5pct"] and result["better_model"] == "sarimax"


def test_wilcoxon_reports_missing_gru(root):
    result = compare.wilcoxon_test(compare.load_forecasts(root), "sarimax", "gru")
    assert result["status"] == "pendiente"


def test_errors_by_day_type_omits_empty_groups(root, history):
    out = compare.errors_by_day_type(compare.load_forecasts(root), history)
    assert "festivo" not in set(out["day_type"])
    counts = out[out["model"] == "naive"].set_index("day_type")["n"]
    assert counts["promo"] + counts["sin_promo"] == HORIZON


def test_comparison_is_strict_json(root, history):
    text = json.dumps(compare.build_comparison(root, history), allow_nan=False)
    assert json.loads(text)["wilcoxon"]["sarimax_vs_gru"]["status"] == "pendiente"
