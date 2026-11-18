"""Limpieza y unión de train + store. Cada decisión está documentada en docs/DECISIONES.md
(D-06 a D-11)."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_HOLIDAY_VALUES = {"0", "a", "b", "c"}


def normalize_state_holiday(s: pd.Series) -> pd.Series:
    """D-06: `StateHoliday` mezcla 0 numérico y '0' texto → un solo tipo str en {0, a, b, c}."""
    out = s.astype(str).str.strip().replace({"0.0": "0", "nan": "0", "": "0"})
    bad = set(out.unique()) - STATE_HOLIDAY_VALUES
    if bad:
        raise ValueError(f"Valores inesperados en StateHoliday: {sorted(bad)}")
    return out


def flag_open_zero(train: pd.DataFrame) -> pd.Series:
    """D-07: días abiertos con ventas 0 (anomalía). Se marcan, no se borran."""
    return ((train["Open"] == 1) & (train["Sales"] == 0)).astype("int8")


def clean_store(store: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de `store.csv`.

    - D-08: `CompetitionDistance` nulo → mediana + bandera `comp_dist_missing`.
    - D-09: competencia sin fecha de apertura → se asume que ya estaba abierta
      (año 1900, mes 1) + bandera `comp_open_assumed`.
    - `PromoInterval` nulo (tiendas sin Promo2) → cadena vacía.
    """
    st = store.copy()
    st["comp_dist_missing"] = st["CompetitionDistance"].isna().astype("int8")
    st["CompetitionDistance"] = st["CompetitionDistance"].fillna(st["CompetitionDistance"].median())
    no_date = st["CompetitionOpenSinceYear"].isna() | st["CompetitionOpenSinceMonth"].isna()
    st["comp_open_assumed"] = no_date.astype("int8")
    st["CompetitionOpenSinceYear"] = st["CompetitionOpenSinceYear"].fillna(1900).astype(int)
    st["CompetitionOpenSinceMonth"] = st["CompetitionOpenSinceMonth"].fillna(1).astype(int)
    st["PromoInterval"] = st["PromoInterval"].fillna("").astype(str)
    st["StoreType"] = st["StoreType"].astype(str)
    st["Assortment"] = st["Assortment"].astype(str)
    return st


def clean(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Une train y store (left join por Store), normaliza tipos y ordena por Store, Date."""
    tr = train.copy()
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["StateHoliday"] = normalize_state_holiday(tr["StateHoliday"])
    tr["anomaly_open_zero"] = flag_open_zero(tr)
    for col in ["Open", "Promo", "SchoolHoliday", "DayOfWeek"]:
        tr[col] = tr[col].astype("int8")

    n = len(tr)
    df = tr.merge(clean_store(store), on="Store", how="left", validate="many_to_one")
    if len(df) != n:
        raise AssertionError(f"El merge cambió el número de filas: {n} → {len(df)}")
    if df["StoreType"].isna().any():
        raise AssertionError("Hay tiendas de train sin datos en store.csv")
    return df.sort_values(["Store", "Date"], ignore_index=True)


def impute_missing_store_days(df: pd.DataFrame) -> pd.DataFrame:
    """D-10: rellena los días sin registro de cada tienda (hueco de 2014) SIN usar el futuro.

    Para cada (tienda, fecha) faltante dentro del rango global de fechas:
    - Promo, SchoolHoliday y StateHoliday: el valor más frecuente de ese día en las demás
      tiendas (se conoce el mismo día, no es información futura).
    - Open: 1 si la mayoría de tiendas del mismo StoreType abrió ese día.
    - Sales: promedio de la propia tienda en los 56 días ANTERIORES al primer día faltante,
      con el mismo día de la semana y el mismo estado de Promo (solo días abiertos).
    Las filas añadidas llevan `imputed = 1`.
    """
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full = pd.MultiIndex.from_product([df["Store"].unique(), dates], names=["Store", "Date"])
    have = pd.MultiIndex.from_frame(df[["Store", "Date"]])
    missing = full.difference(have)
    out = df.assign(imputed=np.int8(0))
    if len(missing) == 0:
        return out

    miss = missing.to_frame(index=False)
    day = df.groupby("Date").agg(
        Promo=("Promo", lambda s: s.mode().iat[0]),
        SchoolHoliday=("SchoolHoliday", lambda s: s.mode().iat[0]),
        StateHoliday=("StateHoliday", lambda s: s.mode().iat[0]),
    )
    store_cols = [c for c in df.columns if c not in train_columns() and c != "anomaly_open_zero"]
    static = df.drop_duplicates("Store").set_index("Store")[store_cols]
    miss = miss.join(day, on="Date").join(static, on="Store")
    open_by_type = df.groupby(["StoreType", "Date"])["Open"].mean().ge(0.5).astype("int8")
    miss["Open"] = (
        open_by_type.reindex(pd.MultiIndex.from_frame(miss[["StoreType", "Date"]]))
        .fillna(1)
        .to_numpy()
        .astype("int8")
    )
    miss["DayOfWeek"] = (miss["Date"].dt.dayofweek + 1).astype("int8")

    sales = np.zeros(len(miss))
    for store, idx in miss.groupby("Store").groups.items():
        first = miss.loc[idx, "Date"].min()
        past = df[
            (df["Store"] == store)
            & (df["Date"] < first)
            & (df["Date"] >= first - pd.Timedelta(days=56))
            & (df["Open"] == 1)
        ]
        ref = past.groupby(["DayOfWeek", "Promo"])["Sales"].mean()
        ref_dow = past.groupby("DayOfWeek")["Sales"].mean()
        fallback = past["Sales"].mean() if len(past) else 0.0
        for i in idx:
            if miss.at[i, "Open"] == 0:
                continue
            key = (miss.at[i, "DayOfWeek"], miss.at[i, "Promo"])
            val = ref.get(key, ref_dow.get(miss.at[i, "DayOfWeek"], fallback))
            sales[miss.index.get_loc(i)] = 0.0 if pd.isna(val) else float(val)
    miss["Sales"] = np.round(sales).astype(df["Sales"].dtype)
    miss["Customers"] = np.nan  # no se imputa: Customers nunca es exógena
    miss["anomaly_open_zero"] = np.int8(0)
    miss["imputed"] = np.int8(1)
    for col in ["Promo", "SchoolHoliday"]:
        miss[col] = miss[col].astype("int8")

    out = pd.concat([out, miss[out.columns]], ignore_index=True)
    return out.sort_values(["Store", "Date"], ignore_index=True)


def train_columns() -> list[str]:
    """Columnas originales de `train.csv`."""
    return [
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    ]
