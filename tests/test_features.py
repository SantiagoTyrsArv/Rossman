"""Tests de variables exógenas y series agregadas (tarea A3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean, impute_missing_store_days
from src.features import exog
from src.features.calendar import fourier_terms


@pytest.fixture(scope="module")
def df(raw):
    return impute_missing_store_days(clean(*raw))


@pytest.fixture(scope="module")
def history(df):
    return exog.make_history(df, ["total", "type_a", "type_b", "type_c", "type_d", "store_2"])


@pytest.fixture()
def use_history(history, monkeypatch):
    monkeypatch.setattr(exog, "load_history", lambda: history)
    return history


def _row(store_row: dict, d: str) -> pd.DataFrame:
    return pd.DataFrame([{**store_row, "Date": pd.Timestamp(d)}])


def test_promo2_active_known_case():
    base = {"Store": 1, "Promo2": 1, "Promo2SinceWeek": 13, "Promo2SinceYear": 2013,
            "PromoInterval": "Jan,Apr,Jul,Oct"}  # fmt: skip
    # semana ISO 13 de 2013 empieza el lunes 25/03/2013
    assert exog.promo2_active(_row(base, "2013-04-10")).iat[0] == 1  # abril, después del inicio
    assert exog.promo2_active(_row(base, "2013-05-10")).iat[0] == 0  # mayo no está en el ciclo
    assert exog.promo2_active(_row(base, "2013-01-10")).iat[0] == 0  # enero, antes del inicio
    assert exog.promo2_active(_row({**base, "Promo2": 0}, "2013-04-10")).iat[0] == 0


def test_promo2_active_sept_abbreviation():
    base = {"Store": 1, "Promo2": 1, "Promo2SinceWeek": 1, "Promo2SinceYear": 2013,
            "PromoInterval": "Mar,Jun,Sept,Dec"}  # fmt: skip
    assert exog.promo2_active(_row(base, "2013-09-05")).iat[0] == 1


def test_competition_active():
    d = pd.DataFrame({"Date": pd.to_datetime(["2013-02-01", "2013-04-01"]),
                      "CompetitionOpenSinceYear": [2013, 2013],
                      "CompetitionOpenSinceMonth": [3, 3]})  # fmt: skip
    assert list(exog.competition_active(d)) == [0, 1]


def test_history_schema(history):
    assert list(history.columns) == exog.HISTORY_COLS
    for _, g in history.groupby("unique_id"):
        assert g["ds"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert not history.drop(columns="unique_id").isna().any().any()
    assert set(history["state_holiday"]) <= {"0", "a", "b", "c"}
    assert (
        history[["promo_share", "open_share", "school_holiday_share", "promo2_share"]]
        .stack()
        .between(0, 1)
        .all()
    )


def test_total_is_sum_of_types(history):
    piv = history.pivot_table(index="ds", columns="unique_id", values="y")
    np.testing.assert_allclose(piv["total"], piv[["type_a", "type_b", "type_c", "type_d"]].sum(1))


def test_aggregate_does_not_use_future(df):
    cut = pd.Timestamp("2013-08-01")
    full = exog.aggregate_series(df, "total").set_index("ds").loc[:cut]
    past = exog.aggregate_series(df[df["Date"] <= cut], "total").set_index("ds")
    pd.testing.assert_frame_equal(full, past)


def test_build_exog_columns(history):
    x = exog.build_exog(history[history["unique_id"] == "total"])
    assert list(x.columns) == exog.EXOG_COLS
    assert not x.isna().any().any()


def test_build_future_exog_contract(use_history):
    dates = pd.date_range(use_history["ds"].max() + pd.Timedelta(days=1), periods=42)
    x = exog.build_future_exog("total", dates)
    assert list(x.columns) == exog.EXOG_COLS
    assert x.index.equals(dates)
    assert not x.isna().any().any()
    pd.testing.assert_frame_equal(x[[c for c in x if c.startswith("f_")]], fourier_terms(dates))


def test_build_future_exog_historical_promo_pattern(use_history):
    h = use_history[use_history["unique_id"] == "total"].set_index("ds")
    last = h.index.max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=28)
    x = exog.build_future_exog("total", dates)
    for d in dates:
        src = d - pd.Timedelta(days=14)
        src = src if src <= last else src - pd.Timedelta(days=14)
        expected = h.at[src, "promo_share"] if x.at[d, "open_share"] >= 0.05 else 0.0
        assert x.at[d, "promo_share"] == pytest.approx(expected)


def test_build_future_exog_scenarios(use_history):
    dates = pd.date_range(use_history["ds"].max() + pd.Timedelta(days=1), periods=42)
    weekdays = {d for d in dates if d.dayofweek < 5}
    holiday = dates[10]
    x = exog.build_future_exog(
        "total",
        dates,
        promo_days=weekdays,
        school_holiday_days={dates[0], dates[1]},
        state_holidays={holiday.strftime("%Y-%m-%d"): "a"},
        promo2_share=0.3,
    )
    assert x.loc[dates[0:2], "school_holiday_share"].eq(1).all()
    assert x["school_holiday_share"].sum() == 2
    assert x.at[holiday, "sh_a"] == 1 and x["sh_a"].sum() == 1
    assert x.at[holiday, "open_share"] < 0.5  # en festivo abren pocas tiendas
    assert x["promo2_share"].eq(0.3).all()
    sundays = [d for d in dates if d.dayofweek == 6]
    assert x.loc[sundays, "promo_share"].eq(0).all()


def test_build_future_exog_open_share_series(use_history):
    dates = pd.date_range(use_history["ds"].max() + pd.Timedelta(days=1), periods=7)
    x = exog.build_future_exog("total", dates, open_share=pd.Series(0.8, index=dates))
    assert x["open_share"].eq(0.8).all()
    with pytest.raises(ValueError):
        exog.build_future_exog("total", dates, open_share=pd.Series(0.8, index=dates[:3]))


def test_build_future_exog_bad_holiday(use_history):
    dates = pd.date_range(use_history["ds"].max() + pd.Timedelta(days=1), periods=7)
    with pytest.raises(ValueError):
        exog.build_future_exog("total", dates, state_holidays={dates[0]: "z"})


def test_export_history_roundtrip(tmp_path, history):
    path = exog.export_history(history, tmp_path / "history.parquet")
    back = pd.read_parquet(path)
    assert list(back.columns) == exog.HISTORY_COLS and len(back) == len(history)
