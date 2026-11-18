"""Tests de carga y limpieza (tarea A1)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import clean, impute_missing_store_days, normalize_state_holiday
from src.data.load import load_raw


@pytest.fixture(scope="module")
def cleaned(raw):
    train, store = raw
    return clean(train, store)


def test_no_duplicates(cleaned):
    assert not cleaned.duplicated(["Store", "Date"]).any()


def test_state_holiday_single_type(cleaned):
    assert cleaned["StateHoliday"].map(type).nunique() == 1
    assert set(cleaned["StateHoliday"]) <= {"0", "a", "b", "c"}


def test_closed_implies_zero_sales(cleaned):
    assert (cleaned.loc[cleaned["Open"] == 0, "Sales"] == 0).all()


def test_rows_preserved_after_merge(raw, cleaned):
    assert len(cleaned) == len(raw[0])
    assert cleaned["StoreType"].notna().all()


def test_sorted_by_store_date(cleaned):
    assert cleaned[["Store", "Date"]].equals(
        cleaned[["Store", "Date"]].sort_values(["Store", "Date"], ignore_index=True)
    )


def test_open_zero_flagged_not_removed(raw, cleaned):
    train = raw[0]
    n = ((train["Open"] == 1) & (train["Sales"] == 0)).sum()
    assert n > 0 and cleaned["anomaly_open_zero"].sum() == n


def test_store_cleaning(cleaned):
    s2 = cleaned[cleaned["Store"] == 2].iloc[0]
    assert s2["comp_dist_missing"] == 1 and s2["CompetitionDistance"] > 0
    assert s2["comp_open_assumed"] == 1 and s2["CompetitionOpenSinceYear"] == 1900
    assert cleaned["CompetitionDistance"].notna().all()


def test_unknown_state_holiday_raises():
    with pytest.raises(ValueError):
        normalize_state_holiday(pd.Series(["0", "x"]))


def test_impute_fills_gap_without_future(cleaned):
    full = impute_missing_store_days(cleaned)
    n_dates = cleaned["Date"].nunique()
    assert (full.groupby("Store").size() == n_dates).all()
    imp = full[full["imputed"] == 1]
    assert len(imp) == 30 and set(imp["Store"]) == {4}
    first = imp["Date"].min()
    past = cleaned[(cleaned["Store"] == 4) & (cleaned["Date"] < first) & (cleaned["Open"] == 1)]
    # lo imputado está en el rango de lo observado ANTES del hueco
    opened = imp[imp["Open"] == 1]["Sales"]
    assert opened.between(past["Sales"].min(), past["Sales"].max()).all()
    assert (imp.loc[imp["Open"] == 0, "Sales"] == 0).all()
    assert imp["Customers"].isna().all()


def test_load_raw_reads_state_holiday_as_text(tmp_path, raw):
    train, store = raw
    train.to_csv(tmp_path / "train.csv", index=False)
    store.to_csv(tmp_path / "store.csv", index=False)
    tr, st = load_raw(tmp_path)
    assert tr["StateHoliday"].map(type).eq(str).all()
    assert pd.api.types.is_datetime64_any_dtype(tr["Date"])
    assert len(st) == 4
