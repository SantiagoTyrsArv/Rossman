"""Tests de la preparación de la GRU (Persona B). Los que importan TensorFlow son `slow`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import gru


@pytest.fixture(scope="module")
def hist() -> pd.DataFrame:
    """Serie sintética diaria 2013-01-01 a 2015-07-31 con el esquema de history.parquet."""
    ds = pd.date_range("2013-01-01", "2015-07-31", freq="D")
    rng = np.random.default_rng(0)
    promo = ((ds.dayofweek < 5) & (ds.isocalendar().week.to_numpy() % 2 == 0)).astype(float)
    open_share = np.where(ds.dayofweek == 6, 0.0, 1.0)
    y = 1e6 * (1 + 0.3 * promo) * open_share * rng.lognormal(0, 0.05, len(ds))
    # un pico fuera de entrenamiento para comprobar que el escalador no lo ve
    y[ds.get_loc(pd.Timestamp("2015-07-01"))] = 9e7
    sh = np.where((ds.month == 12) & (ds.day == 25), "c", "0")
    return pd.DataFrame(
        {
            "unique_id": "total",
            "ds": ds,
            "y": y,
            "promo_share": promo,
            "open_share": open_share,
            "school_holiday_share": 0.1,
            "state_holiday": sh,
            "promo2_share": 0.5,
        }
    )


@pytest.fixture(scope="module")
def sd(hist) -> gru.SeriesData:
    return gru.prepare_series(hist)


def test_feature_shapes(sd, hist):
    n = len(hist)
    assert sd.past.shape == (n, len(gru.PAST_COLS))
    assert sd.fut.shape == (n, len(gru.FUT_COLS))
    assert not np.isnan(sd.past).any()
    # día de la semana one-hot: exactamente un 1 por fila
    dow = sd.fut[:, len(gru.EXOG_COLS) : len(gru.EXOG_COLS) + 7]
    assert (dow.sum(axis=1) == 1).all()


def test_scaler_fit_only_on_train(sd, hist):
    train = hist[hist["ds"] <= gru.TRAIN_END]
    assert sd.scaler.data_max_[0] == pytest.approx(np.log1p(train["y"].max()))
    i_train = sd.idx(gru.TRAIN_END)
    assert sd.y_scaled[: i_train + 1].max() == pytest.approx(1.0)
    assert sd.y_scaled.max() > 1.0  # el pico posterior queda fuera de [0, 1]


def test_inverse_roundtrip(sd):
    back = gru.inverse_target(sd.y_scaled, sd.scaler)
    np.testing.assert_allclose(back, sd.y, rtol=1e-6, atol=1e-3)


def test_make_windows_shapes_and_alignment(sd):
    lb, hz = 56, 42
    end = sd.idx(gru.TRAIN_END)
    xp, xf, yy = gru.make_windows(sd.past, sd.fut, sd.y_scaled, lb, hz, end_idx=end)
    n_expected = end - hz + 1 - lb + 1
    assert xp.shape == (n_expected, lb, len(gru.PAST_COLS))
    assert xf.shape == (n_expected, hz, len(gru.FUT_COLS))
    assert yy.shape == (n_expected, hz)
    # primera ventana: pasado = filas 0..55, objetivo = filas 56..97
    np.testing.assert_array_equal(xp[0], sd.past[:lb])
    np.testing.assert_array_equal(xf[0], sd.fut[lb : lb + hz])
    np.testing.assert_array_equal(yy[0], sd.y_scaled[lb : lb + hz])
    # el pasado de cada ventana termina justo antes de su objetivo
    np.testing.assert_array_equal(xp[:, -1, 0], sd.y_scaled[lb - 1 : lb - 1 + n_expected])


def test_no_training_window_after_train_end(sd):
    end = sd.idx(gru.TRAIN_END)
    _, _, yy = gru.make_windows(sd.past, sd.fut, sd.y_scaled, end_idx=end)
    # la última ventana termina exactamente en TRAIN_END
    np.testing.assert_array_equal(yy[-1], sd.y_scaled[end - 41 : end + 1])
    # ninguna ventana contiene valores posteriores (el pico de julio no aparece)
    assert yy.max() <= 1.0 + 1e-9


def test_train_val_windows_disjoint_targets(sd):
    (xp_tr, _, y_tr), (xp_va, _, y_va) = gru.train_val_windows(sd)
    assert len(y_tr) > 0 and len(y_va) == 42
    np.testing.assert_array_equal(y_va[-1], sd.y_scaled[sd.idx(gru.VAL_END) - 41 :][:42])


def test_make_windows_too_short_returns_empty():
    xp, xf, yy = gru.make_windows(np.zeros((50, 3)), np.zeros((50, 2)), np.zeros(50))
    assert xp.shape == (0, 56, 3) and xf.shape == (0, 42, 2) and yy.shape == (0, 42)


def test_origin_input(sd):
    origin = sd.idx(gru.TRAIN_END)
    xp, xf = gru.origin_input(sd.past, sd.fut, origin)
    assert xp.shape == (1, 56, len(gru.PAST_COLS))
    assert xf.shape == (1, 42, len(gru.FUT_COLS))
    np.testing.assert_array_equal(xp[0, -1], sd.past[origin])
    np.testing.assert_array_equal(xf[0, 0], sd.fut[origin + 1])
    with pytest.raises(ValueError):
        gru.origin_input(sd.past, sd.fut, len(sd.y) - 1)  # faltan exógenas futuras


def test_exog_features_from_future_exog_format():
    dates = pd.date_range("2015-08-01", periods=42, freq="D")
    x = pd.DataFrame(
        {
            c: 0.0
            for c in [
                "promo_share",
                "open_share",
                "school_holiday_share",
                "sh_a",
                "sh_b",
                "sh_c",
                "promo2_share",
                "f_sin1",
            ]
        },
        index=dates,
    )
    out = gru.exog_features(x)
    assert list(out.columns) == gru.FUT_COLS and len(out) == 42


def test_forecast_frame_and_metrics(tmp_path):
    rows = []
    for split, origin in gru.ORIGINS.items():
        d = gru.split_dates(origin)
        y = None if split == "future" else np.full(42, 100.0)
        rows.append(gru.forecast_frame("total", d, y, np.full(42, 90.0), split))
    fc = pd.concat(rows, ignore_index=True)
    assert list(fc.columns) == gru.FORECAST_COLS
    met = gru.write_artifacts(fc, root=tmp_path)
    assert met["levels"]["total"]["test"]["mape"] == pytest.approx(10.0)
    assert (tmp_path / "forecasts" / "gru.parquet").exists()
    assert (tmp_path / "metrics" / "gru.json").exists()


def test_validate_forecast_rejects_wrong_length():
    fc = gru.forecast_frame("total", gru.split_dates(gru.TRAIN_END, 10), None, np.ones(10), "val")
    with pytest.raises(ValueError, match="filas"):
        gru.validate_forecast(fc)


def test_log_experiment_appends(tmp_path):
    path = tmp_path / "experiments.csv"
    gru.log_experiment({"units": 64}, {"mape": 9.5, "rmse": 1.0, "mae": 1.0}, path)
    gru.log_experiment({"units": 32}, {"mape": 9.9, "rmse": 1.0, "mae": 1.0}, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("fecha")


@pytest.mark.slow
def test_build_model_output_shape(sd):
    pytest.importorskip("tensorflow")
    model = gru.build_model(units=16)
    xp, xf = gru.origin_input(sd.past, sd.fut, sd.idx(gru.TRAIN_END))
    assert model.predict([xp, xf], verbose=0).shape == (1, 42)
