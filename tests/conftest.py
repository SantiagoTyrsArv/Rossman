"""Datos crudos sintéticos con la forma de train.csv y store.csv (para tests sin Kaggle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_raw(n_days: int = 400, gap: tuple[int, int] = (200, 230), seed: int = 0):
    """4 tiendas (una por tipo). La tienda 4 no tiene registros en los días `gap`."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    rows = []
    for store in (1, 2, 3, 4):
        for i, d in enumerate(dates):
            if store == 4 and gap[0] <= i < gap[1]:
                continue
            open_ = 0 if d.dayofweek == 6 else 1
            promo = int(d.dayofweek < 5 and d.isocalendar().week % 2 == 0) * open_
            sh = "a" if (d.month, d.day) in {(1, 1), (5, 1)} else (0 if i % 2 else "0")
            if sh == "a":
                open_ = 0
                promo = 0
            sales = int(5000 * store * (1 + 0.3 * promo) * rng.lognormal(0, 0.1)) * open_
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": d.dayofweek + 1,
                    "Date": d,
                    "Sales": sales,
                    "Customers": sales // 10,
                    "Open": open_,
                    "Promo": promo,
                    "StateHoliday": sh,
                    "SchoolHoliday": int(d.month in (7, 8)),
                }
            )
    train = pd.DataFrame(rows)
    train.loc[(train["Store"] == 1) & (train.index % 97 == 5) & (train["Open"] == 1), "Sales"] = 0
    store = pd.DataFrame(
        {
            "Store": [1, 2, 3, 4],
            "StoreType": ["a", "b", "c", "d"],
            "Assortment": ["a", "b", "c", "a"],
            "CompetitionDistance": [500.0, np.nan, 1200.0, 300.0],
            "CompetitionOpenSinceMonth": [9, np.nan, 3, 11],
            "CompetitionOpenSinceYear": [2008, np.nan, 2013, 2014],
            "Promo2": [0, 1, 1, 0],
            "Promo2SinceWeek": [np.nan, 13, 40, np.nan],
            "Promo2SinceYear": [np.nan, 2013, 2013, np.nan],
            "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct", "Mar,Jun,Sept,Dec", np.nan],
        }
    )
    return train, store


@pytest.fixture(scope="session")
def raw():
    return make_raw()
