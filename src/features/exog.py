"""Variables exógenas, series agregadas por nivel y exógenas futuras para el simulador.

Uso: `python -m src.features.exog` → genera `artifacts/series/history.parquet` a partir de
`data/processed/rossmann.parquet`.

Este módulo lo importa la app desplegada (`build_future_exog`), así que solo depende de
pandas y numpy.
"""

from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_PROCESSED, LEVELS_TYPES, SELECTED_STORES
from src.features.calendar import fourier_cols, fourier_terms

HISTORY_FILE = ARTIFACTS / "series" / "history.parquet"
HISTORY_COLS = [
    "unique_id",
    "ds",
    "y",
    "promo_share",
    "open_share",
    "school_holiday_share",
    "state_holiday",
    "promo2_share",
]
EXOG_COLS = [
    "promo_share",
    "open_share",
    "school_holiday_share",
    "sh_a",
    "sh_b",
    "sh_c",
    "promo2_share",
    *fourier_cols(),
]
# Abreviaturas de PromoInterval (septiembre aparece como "Sept")
MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}  # fmt: skip
STATE_HOLIDAY_TYPES = ("a", "b", "c")
MIN_OPEN_FOR_PROMO = 0.05  # con menos tiendas abiertas no tiene sentido una promo


# --- Variables a nivel tienda-día -----------------------------------------------------------
def promo2_active(df: pd.DataFrame) -> pd.Series:
    """1 si la tienda participa en Promo2, la fecha es posterior al inicio (lunes de la
    semana ISO `Promo2SinceWeek` de `Promo2SinceYear`) y el mes está en `PromoInterval`."""
    st = df.drop_duplicates("Store").set_index("Store")
    start = {}
    months = {}
    for store, row in st.iterrows():
        if row["Promo2"] == 1 and pd.notna(row["Promo2SinceYear"]):
            start[store] = pd.Timestamp(
                date.fromisocalendar(int(row["Promo2SinceYear"]), int(row["Promo2SinceWeek"]), 1)
            )
            names = {m.strip() for m in str(row["PromoInterval"]).split(",") if m.strip()}
            months[store] = {k for k, v in MONTH_ABBR.items() if v in names}
    start_s = pd.to_datetime(df["Store"].map(start))
    month_ok = np.array(
        [m in months.get(s, ()) for s, m in zip(df["Store"], df["Date"].dt.month, strict=True)]
    )
    active = (df["Promo2"] == 1) & start_s.notna() & (df["Date"] >= start_s) & month_ok
    return active.astype("int8")


def competition_active(df: pd.DataFrame) -> pd.Series:
    """1 si la fecha es igual o posterior a la apertura del competidor (D-09: sin fecha =
    abierta desde siempre)."""
    opened = pd.to_datetime(
        {
            "year": df["CompetitionOpenSinceYear"].astype(int),
            "month": df["CompetitionOpenSinceMonth"].astype(int),
            "day": 1,
        }
    )
    return (df["Date"] >= opened).astype("int8")


# --- Series agregadas -----------------------------------------------------------------------
def all_levels() -> list[str]:
    """`unique_id` del contrato: total, tipos y tiendas seleccionadas."""
    return ["total", *LEVELS_TYPES, *[f"store_{n}" for n in SELECTED_STORES]]


def _level_rows(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "total":
        return df
    if level.startswith("type_"):
        return df[df["StoreType"] == level.split("_", 1)[1]]
    if level.startswith("store_"):
        return df[df["Store"] == int(level.split("_", 1)[1])]
    raise ValueError(f"Nivel desconocido: {level}")


def _fill_gaps(out: pd.DataFrame) -> pd.DataFrame:
    """Frecuencia diaria completa. Huecos: exógenas con el último valor conocido y `y` con la
    venta de 7 días antes (solo pasado)."""
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    gaps = full.difference(out.index)
    out = out.reindex(full)
    if len(gaps):
        share_cols = ["promo_share", "open_share", "school_holiday_share", "promo2_share"]
        out[share_cols] = out[share_cols].ffill()
        out["state_holiday"] = out["state_holiday"].fillna("0")
        for d in gaps:
            out.loc[d, "y"] = out["y"].get(d - pd.Timedelta(days=7), np.nan)
        out["y"] = out["y"].ffill()
    return out


def aggregate_series(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Serie diaria de un nivel con las columnas de `history.parquet` (contrato 5.3)."""
    d = _level_rows(df, level)
    if d.empty:
        raise ValueError(f"No hay filas para el nivel {level}")
    if "promo2_active" not in d.columns:
        d = d.assign(promo2_active=promo2_active(d))
    d = d.assign(promo_open=d["Promo"] * d["Open"])
    g = d.groupby("Date")
    open_n = g["Open"].sum()
    out = pd.DataFrame(
        {
            "y": g["Sales"].sum().astype(float),
            "promo_share": (g["promo_open"].sum() / open_n.where(open_n > 0)).fillna(0.0),
            "open_share": g["Open"].mean(),
            "school_holiday_share": g["SchoolHoliday"].mean(),
            "promo2_share": g["promo2_active"].mean(),
        }
    )
    counts = pd.crosstab(d["Date"], d["StateHoliday"].astype(str))
    out["state_holiday"] = counts.idxmax(axis=1).reindex(out.index).astype(str)
    out = _fill_gaps(out)
    out.index.name = "ds"
    out = out.reset_index().assign(unique_id=level)
    return out[HISTORY_COLS]


def make_history(df: pd.DataFrame, levels: list[str] | None = None) -> pd.DataFrame:
    """Concatena todas las series agregadas (calcula `promo2_active` una sola vez)."""
    df = df.assign(promo2_active=promo2_active(df))
    levels = levels or all_levels()
    return pd.concat([aggregate_series(df, lv) for lv in levels], ignore_index=True)


def export_history(history: pd.DataFrame, path: Path = HISTORY_FILE) -> Path:
    """Escribe `artifacts/series/history.parquet` validando el esquema."""
    missing = set(HISTORY_COLS) - set(history.columns)
    if missing:
        raise ValueError(f"Faltan columnas en history: {sorted(missing)}")
    if history[HISTORY_COLS].drop(columns="unique_id").isna().any().any():
        raise ValueError("history tiene NaN")
    path.parent.mkdir(parents=True, exist_ok=True)
    history[HISTORY_COLS].to_parquet(path, index=False)
    return path


# --- Exógenas para los modelos --------------------------------------------------------------
def _sh_dummies(state_holiday: pd.Series) -> pd.DataFrame:
    sh = state_holiday.astype(str)
    return pd.DataFrame({f"sh_{k}": (sh == k).astype(float) for k in STATE_HOLIDAY_TYPES})


def build_exog(series_df: pd.DataFrame) -> pd.DataFrame:
    """Exógenas históricas de una serie (formato history) en el orden de `EXOG_COLS`."""
    s = series_df.sort_values("ds").set_index("ds")
    s.index = pd.DatetimeIndex(s.index)
    out = s[["promo_share", "open_share", "school_holiday_share", "promo2_share"]].astype(float)
    out = out.join(_sh_dummies(s["state_holiday"]).set_index(s.index))
    out = out.join(fourier_terms(s.index))
    return out[EXOG_COLS]


@lru_cache(maxsize=4)
def _read_history(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["state_holiday"] = df["state_holiday"].astype(str)
    return df


def load_history(path: Path = HISTORY_FILE) -> pd.DataFrame:
    """Lee `history.parquet` (cacheado mientras el archivo no cambie)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre `python -m src.features.exog`.")
    return _read_history(str(path), path.stat().st_mtime)


def _historical_promo(h: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Repite las últimas semanas con la misma alternancia quincenal: el valor del día
    `d − 14·k`, con el menor k que cae dentro de la historia."""
    last = h.index.max()
    vals = []
    for d in dates:
        k = max(1, math.ceil((d - last).days / 14))
        vals.append(float(h["promo_share"].get(d - pd.Timedelta(days=14 * k), 0.0)))
    return np.array(vals)


def _historical_open(h: pd.DataFrame, dates: pd.DatetimeIndex, sh: pd.Series) -> np.ndarray:
    """Promedio por día de la semana del último año (sin festivos); en festivos, el promedio
    histórico de los días con ese tipo de festivo."""
    last_year = h[h.index > h.index.max() - pd.Timedelta(days=364)]
    normal = last_year[last_year["state_holiday"] == "0"]
    by_dow = normal.groupby(normal.index.dayofweek)["open_share"].mean()
    holi = h[h["state_holiday"] != "0"]
    by_type = holi.groupby("state_holiday")["open_share"].mean()
    any_holiday = float(holi["open_share"].mean()) if len(holi) else MIN_OPEN_FOR_PROMO
    out = []
    for d in dates:
        t = sh[d]
        if t != "0":
            out.append(float(by_type.get(t, any_holiday)))
        else:
            out.append(float(by_dow.get(d.dayofweek, h["open_share"].mean())))
    return np.array(out)


def _historical_promo2(h: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Promedio del mismo mes en el año más reciente disponible (Promo2 depende del mes)."""
    fallback = float(h["promo2_share"].iloc[-56:].mean())
    out = []
    for d in dates:
        same = h[(h.index.month == d.month) & (h.index < d)]
        if same.empty:
            out.append(fallback)
            continue
        year = same.index.year.max()
        out.append(float(same[same.index.year == year]["promo2_share"].mean()))
    return np.array(out)


def build_future_exog(
    unique_id: str,
    dates: pd.DatetimeIndex,
    promo_days: set[pd.Timestamp] | None = None,  # None = patrón histórico
    school_holiday_days: set[pd.Timestamp] | None = None,
    state_holidays: dict[pd.Timestamp, str] | None = None,  # {"2015-08-15": "a"}
    open_share: float | pd.Series | None = None,  # None = promedio histórico por día de semana
    promo2_share: float | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por `dates` con EXACTAMENTE las columnas
    de meta['exog_cols'] y en ese orden. Sin NaN."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    hist = load_history()
    h = hist[hist["unique_id"] == unique_id].set_index("ds").sort_index()
    if h.empty:
        raise ValueError(f"No hay historia para {unique_id}")

    sh = pd.Series("0", index=dates, dtype=object)
    for k, v in (state_holidays or {}).items():
        if str(v) not in STATE_HOLIDAY_TYPES:
            raise ValueError(f"Tipo de festivo no válido: {v}")
        ts = pd.Timestamp(k).normalize()
        if ts in sh.index:
            sh[ts] = str(v)

    if open_share is None:
        open_v = _historical_open(h, dates, sh)
    elif isinstance(open_share, pd.Series):
        s = open_share.copy()
        s.index = pd.DatetimeIndex(s.index).normalize()
        s = s.reindex(dates)
        if s.isna().any():
            raise ValueError("open_share no cubre todas las fechas")
        open_v = s.to_numpy(float)
    else:
        open_v = np.full(len(dates), float(open_share))

    if promo_days is None:
        promo = _historical_promo(h, dates)
    else:
        chosen = {pd.Timestamp(d).normalize() for d in promo_days}
        promo = np.array([1.0 if d in chosen else 0.0 for d in dates])
    promo = np.where(open_v < MIN_OPEN_FOR_PROMO, 0.0, promo)

    school_set = {pd.Timestamp(d).normalize() for d in (school_holiday_days or set())}
    school = np.array([1.0 if d in school_set else 0.0 for d in dates])

    if promo2_share is None:
        p2 = _historical_promo2(h, dates)
    else:
        p2 = np.full(len(dates), float(promo2_share))

    out = pd.DataFrame(
        {
            "promo_share": promo,
            "open_share": open_v,
            "school_holiday_share": school,
            "promo2_share": p2,
        },
        index=dates,
    )
    out = out.join(_sh_dummies(sh).set_index(dates)).join(fourier_terms(dates))
    out = out[EXOG_COLS]
    if out.isna().any().any():
        raise ValueError("build_future_exog produjo NaN")
    return out


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "rossmann.parquet")
    if not SELECTED_STORES:
        print("Aviso: SELECTED_STORES vacío; solo se exportan total y tipos.")
    path = export_history(make_history(df))
    print(f"OK → {path}")


if __name__ == "__main__":
    main()
