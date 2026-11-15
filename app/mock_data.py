"""Datos de prueba con el mismo esquema del contrato (sección 5).

Sirven para construir y probar la app antes de que existan los artifacts reales de A y B.
Todo es determinista (semilla fija por `unique_id`). Nada de esto se usa para el informe.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from app.utils.settings import (
    FUTURE_START,
    HORIZON,
    LEVELS_TYPES,
    SPLIT_START,
    TEST_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    store_ids,
)

# Feriados estatales aproximados de Alemania en el periodo (a = público, b = Pascua, c = Navidad)
_STATE_HOLIDAYS: dict[str, str] = {
    **{f"{y}-01-01": "a" for y in (2013, 2014, 2015)},
    **{f"{y}-05-01": "a" for y in (2013, 2014, 2015)},
    **{f"{y}-10-03": "a" for y in (2013, 2014, 2015)},
    **{f"{y}-12-25": "c" for y in (2013, 2014, 2015)},
    **{f"{y}-12-26": "c" for y in (2013, 2014, 2015)},
    "2013-03-29": "b",
    "2013-04-01": "b",
    "2014-04-18": "b",
    "2014-04-21": "b",
    "2015-04-03": "b",
    "2015-04-06": "b",
}

# Parámetros por tipo: nivel diario (€), respuesta a la promo (log), apertura en domingo
_TYPE_PARAMS = {
    "a": {"level": 4.2e6, "promo": 0.40, "sunday_open": 0.01, "promo2": 0.45},
    "b": {"level": 1.7e5, "promo": 0.20, "sunday_open": 0.60, "promo2": 0.30},
    "c": {"level": 1.0e6, "promo": 0.35, "sunday_open": 0.00, "promo2": 0.50},
    "d": {"level": 2.4e6, "promo": 0.30, "sunday_open": 0.00, "promo2": 0.55},
}
_STORE_TYPE_CYCLE = ["a", "b", "c", "d", "a"]
_WEEKDAY = np.array([1.25, 1.05, 1.0, 1.0, 1.05, 0.85, 1.0])  # lunes..domingo

_MODEL_NOISE = {"naive": None, "sarima": 0.09, "sarimax": 0.06, "gru": 0.07}
_MODEL_HAS_PI = {"naive": False, "sarima": True, "sarimax": True, "gru": False}


def _rng(*keys: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(keys).encode()))


def _store_type(uid: str) -> str:
    n = int(uid.split("_", 1)[1])
    ids = store_ids()
    idx = ids.index(n) if n in ids else n
    return _STORE_TYPE_CYCLE[idx % len(_STORE_TYPE_CYCLE)]


def _params(uid: str) -> dict[str, float]:
    if uid.startswith("type_"):
        return _TYPE_PARAMS[uid[-1]]
    if uid.startswith("store_"):
        base = dict(_TYPE_PARAMS[_store_type(uid)])
        r = _rng(uid)
        base["level"] = float(r.uniform(4500, 9500))
        base["promo"] = float(r.uniform(0.25, 0.55))
        base["sunday_open"] = 0.0 if base["sunday_open"] < 0.5 else 1.0
        return base
    raise ValueError(uid)


def _historical_promo(dates: pd.DatetimeIndex) -> np.ndarray:
    """Patrón histórico: promo de lunes a viernes en semanas ISO pares."""
    week = dates.isocalendar().week.to_numpy()
    return ((dates.dayofweek < 5) & (week % 2 == 0)).astype(float)


def calendar(
    uid: str,
    dates: pd.DatetimeIndex,
    promo_days: set[pd.Timestamp] | None = None,
    school_holiday_days: set[pd.Timestamp] | None = None,
    state_holidays: dict[pd.Timestamp, str] | None = None,
    promo2_share: float | None = None,
) -> pd.DataFrame:
    """Exógenas diarias de prueba con las columnas de `history.parquet` (sección 5.3)."""
    if uid == "total":
        parts = [
            calendar(t, dates, promo_days, school_holiday_days, state_holidays, promo2_share)
            for t in LEVELS_TYPES
        ]
        w = np.array([_TYPE_PARAMS[t[-1]]["level"] for t in LEVELS_TYPES])
        out = parts[0].copy()
        out["unique_id"] = "total"
        for col in ["promo_share", "open_share", "school_holiday_share", "promo2_share"]:
            out[col] = sum(p[col] * wi for p, wi in zip(parts, w, strict=True)) / w.sum()
        return out

    p = _params(uid)
    dates = pd.DatetimeIndex(dates)
    dow = dates.dayofweek.to_numpy()

    if state_holidays is None:
        sh = [_STATE_HOLIDAYS.get(d.strftime("%Y-%m-%d"), "0") for d in dates]
    else:
        keys = {pd.Timestamp(k).normalize(): v for k, v in state_holidays.items()}
        sh = [keys.get(d.normalize(), "0") for d in dates]
    sh = np.array(sh, dtype=object)

    open_share = np.where(dow == 6, p["sunday_open"], 0.99)
    open_share = np.where(sh != "0", 0.03, open_share)

    if promo_days is None:
        promo = _historical_promo(dates)
    else:
        norm = {pd.Timestamp(d).normalize() for d in promo_days}
        promo = np.array([float(d.normalize() in norm) for d in dates])
    promo = np.where(open_share < 0.5, 0.0, promo)

    if school_holiday_days is None:
        md = dates.month * 100 + dates.day
        summer = (md >= 715) & (md <= 831)
        xmas = (md >= 1222) | (md <= 105)
        school = np.where(summer, 0.6, np.where(xmas, 0.8, 0.05))
    else:
        norm = {pd.Timestamp(d).normalize() for d in school_holiday_days}
        school = np.array([0.9 if d.normalize() in norm else 0.05 for d in dates])

    p2 = p["promo2"] if promo2_share is None else promo2_share
    p2 = np.where(np.isin(dates.month, [1, 4, 7, 10]), min(p2 * 1.2, 1.0), p2)

    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": dates,
            "promo_share": promo,
            "open_share": open_share,
            "school_holiday_share": school,
            "state_holiday": sh.astype(str),
            "promo2_share": p2.astype(float),
        }
    )


def expected(uid: str, dates: pd.DatetimeIndex, **scenario) -> np.ndarray:
    """Venta esperada (sin ruido) de un nivel; `total` es la suma de los tipos."""
    if uid == "total":
        return sum(expected(t, dates, **scenario) for t in LEVELS_TYPES)
    return expected_sales(uid, calendar(uid, dates, **scenario))


def expected_sales(uid: str, cal: pd.DataFrame) -> np.ndarray:
    """Venta esperada (sin ruido) de un tipo o tienda dado su calendario de exógenas."""
    p = _params(uid)
    ds = pd.DatetimeIndex(cal["ds"])
    doy = ds.dayofyear.to_numpy()
    years = (ds - TRAIN_START).days.to_numpy() / 365.25
    annual = 1 + 0.06 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    dec = np.where((ds.month == 12) & (ds.day <= 23), 1 + 0.45 * ds.day / 23, 1.0)
    trend = 1 + 0.02 * years
    effect = np.exp(p["promo"] * cal["promo_share"].to_numpy())
    effect *= 1 + 0.03 * cal["school_holiday_share"].to_numpy()
    effect *= 1 + 0.05 * cal["promo2_share"].to_numpy()
    return (
        p["level"]
        * _WEEKDAY[ds.dayofweek.to_numpy()]
        * annual
        * dec
        * trend
        * effect
        * cal["open_share"].to_numpy()
    )


def _noise_sigma(uid: str) -> float:
    return 0.12 if uid.startswith("store_") else 0.05


def history() -> pd.DataFrame:
    """`history.parquet` de prueba: 2013-01-01 a 2015-07-31 para todos los niveles."""
    dates = pd.date_range(TRAIN_START, TEST_END, freq="D")
    frames = []
    for uid in [*LEVELS_TYPES, *[f"store_{n}" for n in store_ids()]]:
        cal = calendar(uid, dates)
        noise = _rng("hist", uid).lognormal(0, _noise_sigma(uid), len(dates))
        cal["y"] = expected_sales(uid, cal) * noise
        frames.append(cal)
    types = pd.concat(frames[: len(LEVELS_TYPES)])
    total = calendar("total", dates)
    total["y"] = types.groupby("ds")["y"].sum().to_numpy()
    frames.insert(0, total)
    df = pd.concat(frames, ignore_index=True)
    df.loc[df["open_share"] < 0.005, "y"] = 0.0
    cols = [
        "unique_id",
        "ds",
        "y",
        "promo_share",
        "open_share",
        "school_holiday_share",
        "state_holiday",
        "promo2_share",
    ]
    return df[cols]


def _split_dates(split: str) -> pd.DatetimeIndex:
    return pd.date_range(SPLIT_START[split], periods=HORIZON, freq="D")


def forecasts(hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pronósticos de prueba de los 4 modelos para val, test y future (sección 5.1)."""
    hist = history() if hist is None else hist
    origins = {"val": VAL_START - pd.Timedelta(days=1), "test": VAL_END, "future": TEST_END}
    rows = []
    for uid, h in hist.groupby("unique_id", sort=False):
        h = h.set_index("ds")
        for split in ["val", "test", "future"]:
            dates = _split_dates(split)
            y = h["y"].reindex(dates).to_numpy() if split != "future" else np.full(HORIZON, np.nan)
            exp = expected(uid, dates)
            last_week = h["y"].loc[: origins[split]].iloc[-7:].to_numpy()
            for model, sigma in _MODEL_NOISE.items():
                if sigma is None:  # ingenuo estacional: repite la última semana observada
                    y_hat = np.resize(last_week, HORIZON)
                else:
                    s = sigma * (2 if uid.startswith("store_") else 1)
                    y_hat = exp * _rng("fc", uid, split, model).lognormal(0, s, HORIZON)
                lo = hi = np.full(HORIZON, np.nan)
                if _MODEL_HAS_PI[model]:
                    width = 1.96 * s * np.sqrt(1 + np.arange(HORIZON) / 14)
                    lo, hi = y_hat * np.exp(-width), y_hat * np.exp(width)
                rows.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": dates,
                            "y": y,
                            "y_hat": y_hat,
                            "lo_95": lo,
                            "hi_95": hi,
                            "model": model,
                            "split": split,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Solo para datos de prueba. Las métricas reales salen de `compute_all` de A."""
    err = y - y_hat
    m = y > 0
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err[m] / y[m])) * 100),
    }


def metrics(fc: pd.DataFrame | None = None) -> list[dict]:
    """Lista de JSON de métricas (uno por modelo) con el formato de la sección 5.2."""
    fc = forecasts() if fc is None else fc
    out = []
    for model, g in fc[fc["split"] != "future"].groupby("model", sort=False):
        levels: dict[str, dict] = {}
        for (uid, split), gg in g.groupby(["unique_id", "split"], sort=False):
            levels.setdefault(uid, {})[split] = _metrics(gg["y"].to_numpy(), gg["y_hat"].to_numpy())
        out.append({"model": model, "levels": levels})
    return out


def simulate(
    uid: str,
    horizon: int = HORIZON,
    promo_days: set[pd.Timestamp] | None = None,
    school_holiday_days: set[pd.Timestamp] | None = None,
    state_holidays: dict[pd.Timestamp, str] | None = None,
    promo2_share: float | None = None,
) -> pd.DataFrame:
    """Simulador de prueba: base (patrón histórico) vs escenario, con intervalos aproximados."""
    dates = pd.date_range(FUTURE_START, periods=horizon, freq="D")
    base = expected(uid, dates)
    scen = expected(
        uid,
        dates,
        promo_days=promo_days,
        school_holiday_days=school_holiday_days,
        state_holidays=state_holidays,
        promo2_share=promo2_share,
    )
    width = 1.96 * 0.08 * np.sqrt(1 + np.arange(horizon) / 14)
    return pd.DataFrame(
        {
            "ds": dates,
            "base": base,
            "base_lo": base * np.exp(-width),
            "base_hi": base * np.exp(width),
            "scen": scen,
            "scen_lo": scen * np.exp(-width),
            "scen_hi": scen * np.exp(width),
        }
    )
