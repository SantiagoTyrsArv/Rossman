"""SARIMA y SARIMAX sobre log1p(y): selección, pronósticos y modelos para el simulador.

El flujo tiene dos pasos separados para que la prueba se use una sola vez:

    python -m src.models.arima select   # solo train y val; congela los órdenes
    python -m src.models.arima final    # lee los órdenes y genera val, test y future

`statsforecast` se importa solo donde se usa: no está instalado en la CI ni en la app.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import (
    ARTIFACTS,
    HORIZON,
    ROOT,
    SEASON,
    TEST_END,
    TRAIN_END,
    VAL_END,
)
from src.evaluation.backtest import (
    FORECAST_COLS,
    metrics_payload,
    split_dates,
    write_forecast,
    write_metrics,
)
from src.evaluation.metrics import compute_all
from src.features.exog import EXOG_COLS, build_exog, build_future_exog, load_history

USES_EXOG = {"sarima": False, "sarimax": True}
SEASONAL_D = 1  # justificado en notebooks/02_estacionalidad.ipynb
TOP_BY_AIC = 5  # finalistas por AIC que pasan a compararse por RMSE de validación
MAX_FORECAST_RATIO = 3.0  # un pronóstico > 3 veces el máximo histórico se considera inestable
ALPHA = 0.05
SELECTION_FILE = ARTIFACTS / "models" / "arima_selection.json"
GRID_FILE = ROOT / "experiments" / "arima_grid.csv"
EXPERIMENTS_FILE = ROOT / "experiments" / "experiments.csv"
ROLLING_FILE = ROOT / "experiments" / "arima_rolling_cv.csv"


@dataclass(frozen=True)
class Spec:
    """Órdenes de un modelo SARIMA(p,d,q)(P,D,Q)s."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]

    @property
    def label(self) -> str:
        return f"{self.order}{self.seasonal_order}".replace(" ", "")


def level_data(history: pd.DataFrame, uid: str) -> tuple[pd.Series, pd.DataFrame]:
    """Venta diaria y exógenas históricas de un nivel."""
    s = history[history["unique_id"] == uid].sort_values("ds")
    if s.empty:
        raise ValueError(f"No hay historia para {uid}")
    y = s.set_index("ds")["y"].astype(float).asfreq("D")
    return y, build_exog(s).asfreq("D")


def choose_d(y_log: pd.Series, alpha: float = ALPHA) -> int:
    """d = 1 solo si ADF no rechaza raíz unitaria o KPSS rechaza estacionariedad."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(y_log, autolag="AIC")[1]
        kpss_p = kpss(y_log, regression="c", nlags="auto")[1]
    return int(adf_p >= alpha or kpss_p < alpha)


def candidate_specs(d: int, seasonal_d: int = SEASONAL_D, season: int = SEASON) -> list[Spec]:
    """Rejilla de la guía: p, q ∈ {0, 1, 2} y P, Q ∈ {0, 1}."""
    return [
        Spec((p, d, q), (sp, seasonal_d, sq, season))
        for p, q, sp, sq in product(range(3), range(3), range(2), range(2))
    ]


def fit_model(y: pd.Series, exog: pd.DataFrame | None, spec: Spec, maxiter: int = 200):
    """Ajusta SARIMAX sobre log1p(y). Reintenta con más iteraciones si no converge."""
    model = SARIMAX(
        np.log1p(y),
        exog=exog,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False, maxiter=maxiter)
        if not res.mle_retvals.get("converged", False):
            res = model.fit(disp=False, maxiter=maxiter * 5, start_params=res.params)
    return res


def forecast_euros(res, steps: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pronóstico en euros con intervalo al 95 %, sin valores negativos."""
    fc = res.get_forecast(steps=steps, exog=exog)
    ci = np.asarray(fc.conf_int(alpha=ALPHA), float)
    values = np.column_stack([np.asarray(fc.predicted_mean, float), ci[:, 0], ci[:, 1]])
    with np.errstate(over="ignore"):
        values = np.clip(np.expm1(values), 0, None)
    return pd.DataFrame(values, columns=["y_hat", "lo_95", "hi_95"])


def _is_stable(y_hat: np.ndarray, y_train: pd.Series) -> bool:
    limit = MAX_FORECAST_RATIO * float(y_train.max())
    return bool(np.isfinite(y_hat).all() and y_hat.max() <= limit)


def evaluate_candidate(
    uid: str, model: str, spec: Spec, y: pd.Series, exog: pd.DataFrame, source: str = "grid"
) -> dict:
    """Ajusta con train y mide en val. Nunca recibe datos posteriores a VAL_END."""
    y_tr, y_val = y.loc[:TRAIN_END], y.loc[split_dates("val")]
    use_exog = USES_EXOG[model]
    row = {
        "unique_id": uid,
        "model": model,
        "order": list(spec.order),
        "seasonal_order": list(spec.seasonal_order),
        "source": source,
    }
    try:
        res = fit_model(y_tr, exog.loc[:TRAIN_END] if use_exog else None, spec)
        x_val = exog.loc[y_val.index] if use_exog else None
        y_hat = forecast_euros(res, HORIZON, x_val)["y_hat"].to_numpy()
        ok = bool(res.mle_retvals.get("converged", False)) and _is_stable(y_hat, y_tr)
        return {
            **row,
            "aic": float(res.aic),
            "valid": ok and bool(np.isfinite(res.aic)),
            **compute_all(y_val, np.nan_to_num(y_hat, nan=0.0, posinf=0.0)),
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**row, "aic": np.nan, "valid": False, "error": str(exc)[:120]}


def auto_arima_spec(y: pd.Series, exog: pd.DataFrame | None) -> Spec:
    """Órdenes que elige AutoARIMA de statsforecast con los datos recibidos."""
    from statsforecast.models import AutoARIMA

    x = None
    if exog is not None:
        x = exog.loc[:, exog.nunique() > 1].to_numpy(float)
    model = AutoARIMA(season_length=SEASON)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.log1p(y.to_numpy(float)), x)
    p, q, sp, sq, season, d, sd = (int(v) for v in model.model_["arma"])
    return Spec((p, d, q), (sp, sd, sq, season or SEASON))


def pick_best(rows: list[dict], top: int = TOP_BY_AIC) -> dict:
    """Finalistas = mejores `top` por AIC + AutoARIMA; gana el menor RMSE de validación."""
    valid = [r for r in rows if r.get("valid")]
    if not valid:
        raise RuntimeError("Ningún candidato convergió con un pronóstico estable")
    grid = sorted((r for r in valid if r["source"] == "grid"), key=lambda r: r["aic"])[:top]
    finalists = grid + [r for r in valid if r["source"] == "auto_arima"]
    return min(finalists, key=lambda r: r["rmse"])


def _search_level(uid: str, model: str, y: pd.Series, exog: pd.DataFrame) -> list[dict]:
    y_tr = y.loc[:TRAIN_END]
    x_tr = exog.loc[:TRAIN_END] if USES_EXOG[model] else None
    rows = [
        evaluate_candidate(uid, model, spec, y, exog)
        for spec in candidate_specs(choose_d(np.log1p(y_tr)))
    ]
    try:
        rows.append(
            evaluate_candidate(uid, model, auto_arima_spec(y_tr, x_tr), y, exog, "auto_arima")
        )
    except ImportError:
        pass
    return rows


def _append_experiments(chosen: list[dict], path: Path) -> None:
    """Agrega una fila por modelo elegido, con el esquema compartido de experimentos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["fecha", "modelo", "parametros", "mape_val", "rmse_val", "mae_val"])
        for r in chosen:
            params = {k: r[k] for k in ("unique_id", "order", "seasonal_order", "source", "aic")}
            writer.writerow(
                [
                    date.today().isoformat(),
                    r["model"],
                    json.dumps(params, sort_keys=True),
                    f"{r['mape']:.4f}",
                    f"{r['rmse']:.2f}",
                    f"{r['mae']:.2f}",
                ]
            )


def select(
    history: pd.DataFrame,
    levels: list[str] | None = None,
    n_jobs: int = -2,
    selection_file: Path = SELECTION_FILE,
    grid_file: Path = GRID_FILE,
    experiments_file: Path = EXPERIMENTS_FILE,
) -> dict:
    """Elige los órdenes de cada nivel con train y val, y los deja congelados en JSON."""
    history = history[history["ds"] <= VAL_END]
    levels = levels or sorted(history["unique_id"].unique())
    data = {uid: level_data(history, uid) for uid in levels}
    results = Parallel(n_jobs=n_jobs)(
        delayed(_search_level)(uid, model, *data[uid]) for uid in levels for model in USES_EXOG
    )
    rows = [r for group in results for r in group]
    chosen = [pick_best(group) for group in results]
    selection: dict = {"created": date.today().isoformat(), "levels": {}}
    for r in chosen:
        selection["levels"].setdefault(r["unique_id"], {})[r["model"]] = {
            k: r[k] for k in ("order", "seasonal_order", "source", "aic", "rmse", "mae", "mape")
        }
    selection_file = Path(selection_file)
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    grid_file = Path(grid_file)
    grid_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(grid_file, index=False)
    _append_experiments(chosen, Path(experiments_file))
    return selection


def _frame(uid: str, model: str, name: str, ds, y, fc: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": ds,
            "y": y,
            "y_hat": fc["y_hat"].to_numpy(),
            "lo_95": fc["lo_95"].to_numpy(),
            "hi_95": fc["hi_95"].to_numpy(),
            "model": model,
            "split": name,
        }
    )[FORECAST_COLS]


def save_simulator_model(res, uid: str, spec: Spec, root: Path = ARTIFACTS) -> Path:
    """Guarda el modelo reajustado y su meta (contrato 5.4)."""
    folder = Path(root) / "models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sarimax_{uid}.pkl"
    res.save(path)
    meta = {
        "unique_id": uid,
        "order": list(spec.order),
        "seasonal_order": list(spec.seasonal_order),
        "exog_cols": EXOG_COLS,
        "log_target": True,
        "train_end": TEST_END.strftime("%Y-%m-%d"),
    }
    (folder / f"sarimax_{uid}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def final_level(
    uid: str, model: str, spec: Spec, y: pd.Series, exog: pd.DataFrame, root: Path = ARTIFACTS
) -> pd.DataFrame:
    """Val: ajusta con train. Test: con train + val. Future (SARIMAX): con todo."""
    use_exog = USES_EXOG[model]
    frames = []
    for name, cutoff in (("val", TRAIN_END), ("test", VAL_END)):
        ds = split_dates(name)
        res = fit_model(y.loc[:cutoff], exog.loc[:cutoff] if use_exog else None, spec)
        fc = forecast_euros(res, HORIZON, exog.loc[ds] if use_exog else None)
        frames.append(_frame(uid, model, name, ds, y.loc[ds].to_numpy(), fc))
    if use_exog:
        ds = split_dates("future")
        res = fit_model(y.loc[:TEST_END], exog.loc[:TEST_END], spec)
        save_simulator_model(res, uid, spec, root)
        fc = forecast_euros(res, HORIZON, build_future_exog(uid, ds)[EXOG_COLS])
        frames.append(_frame(uid, model, "future", ds, np.nan, fc))
    return pd.concat(frames, ignore_index=True)


def final(
    history: pd.DataFrame,
    n_jobs: int = -2,
    selection_file: Path = SELECTION_FILE,
    root: Path = ARTIFACTS,
) -> dict[str, pd.DataFrame]:
    """Genera pronósticos, métricas y modelos con los órdenes ya congelados."""
    selection = json.loads(Path(selection_file).read_text(encoding="utf-8"))["levels"]
    jobs = []
    for uid, models in selection.items():
        y, exog = level_data(history, uid)
        for model, cfg in models.items():
            spec = Spec(tuple(cfg["order"]), tuple(cfg["seasonal_order"]))
            jobs.append(delayed(final_level)(uid, model, spec, y, exog, root))
    frames = pd.concat(Parallel(n_jobs=n_jobs)(jobs), ignore_index=True)
    out = {}
    for model in USES_EXOG:
        fc = frames[frames["model"] == model]
        write_forecast(fc, model, root)
        write_metrics(metrics_payload(fc, model), model, root)
        out[model] = fc
    return out


def rolling_origin_cv(
    history: pd.DataFrame,
    levels: list[str] | None = None,
    n_windows: int = 4,
    path: Path = ROLLING_FILE,
) -> pd.DataFrame:
    """Validación con origen móvil de AutoARIMA con exógenas; termina en VAL_END."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    history = history[history["ds"] <= VAL_END]
    levels = levels or sorted(history["unique_id"].unique())
    rows = []
    for uid in levels:
        y, exog = level_data(history, uid)
        x = exog.loc[:, exog.nunique() > 1]
        df = x.assign(y=np.log1p(y), unique_id=uid).rename_axis("ds").reset_index()
        sf = StatsForecast(models=[AutoARIMA(season_length=SEASON)], freq="D", n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sf.cross_validation(df=df, h=HORIZON, n_windows=n_windows, step_size=HORIZON)
        for cutoff, g in cv.groupby("cutoff"):
            y_hat = np.clip(np.expm1(g["AutoARIMA"].to_numpy(float)), 0, None)
            scores = compute_all(np.expm1(g["y"].to_numpy(float)), y_hat)
            rows.append({"unique_id": uid, "cutoff": cutoff, **scores})
    out = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    step = (argv or sys.argv[1:] or [""])[0]
    history = load_history()
    if step == "select":
        select(history)
        print(f"OK → {SELECTION_FILE}")
    elif step == "cv":
        print(f"OK → {len(rolling_origin_cv(history))} ventanas en {ROLLING_FILE}")
    elif step == "final":
        final(history)
        print("OK → forecasts, metrics y models de sarima y sarimax")
    else:
        raise SystemExit("Uso: python -m src.models.arima select|cv|final")


if __name__ == "__main__":
    main()
