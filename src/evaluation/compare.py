"""Evaluación final en prueba: tabla de modelos, Wilcoxon y errores por tipo de día (A8).

Uso: `python -m src.evaluation.compare` → `artifacts/metrics/comparison.json`.
Solo lee pronósticos ya generados; `gru.*` es de Persona B y se usa si existe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import ARTIFACTS, MODELS
from src.evaluation.metrics import mae, mape
from src.features.exog import load_history

BASELINE = "naive"
MAPE_GOAL = 15.0
PAIRS = (("sarimax", "naive"), ("sarimax", "gru"))
MONTH_START_DAYS = 5


def load_forecasts(root: Path = ARTIFACTS) -> pd.DataFrame:
    """Une los pronósticos disponibles, en el orden de `MODELS`."""
    files = [Path(root) / "forecasts" / f"{m}.parquet" for m in MODELS]
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError("No hay pronósticos en artifacts/forecasts")
    return pd.concat(frames, ignore_index=True)


def load_metrics(root: Path = ARTIFACTS) -> dict[str, dict]:
    """Métricas por modelo, solo de los archivos que existen."""
    out = {}
    for m in MODELS:
        path = Path(root) / "metrics" / f"{m}.json"
        if path.exists():
            out[m] = json.loads(path.read_text(encoding="utf-8"))["levels"]
    return out


def final_table(metrics: dict[str, dict], split: str = "test") -> pd.DataFrame:
    """Modelo × nivel con RMSE, MAE, MAPE y si supera al baseline y a la meta de MAPE."""
    rows = []
    for model, levels in metrics.items():
        for uid, splits in levels.items():
            if split in splits:
                rows.append({"model": model, "unique_id": uid, **splits[split]})
    table = pd.DataFrame(rows)
    base = table[table["model"] == BASELINE].set_index("unique_id")["rmse"]
    table["beats_baseline"] = table["rmse"] < table["unique_id"].map(base)
    table.loc[table["model"] == BASELINE, "beats_baseline"] = False
    table["meets_mape_goal"] = table["mape"] < MAPE_GOAL
    return table.sort_values(["unique_id", "rmse"], ignore_index=True)


def _absolute_errors(fc: pd.DataFrame, model: str, uid: str, split: str) -> pd.Series:
    g = fc[(fc["model"] == model) & (fc["unique_id"] == uid) & (fc["split"] == split)]
    return (g["y"] - g["y_hat"]).abs().set_axis(pd.DatetimeIndex(g["ds"])).sort_index()


def wilcoxon_test(
    fc: pd.DataFrame, a: str, b: str, uid: str = "total", split: str = "test"
) -> dict:
    """Wilcoxon pareado (dos colas) entre errores absolutos diarios de dos modelos."""
    if not {a, b} <= set(fc["model"]):
        return {"status": "pendiente", "reason": f"faltan pronósticos de {a} o {b}"}
    ea, eb = _absolute_errors(fc, a, uid, split), _absolute_errors(fc, b, uid, split)
    ea, eb = ea.align(eb, join="inner")
    stat, p = wilcoxon(ea, eb, alternative="two-sided")
    diff = float(np.median(ea - eb))
    return {
        "status": "ok",
        "n": int(len(ea)),
        "statistic": float(stat),
        "p_value": float(p),
        f"mae_{a}": float(ea.mean()),
        f"mae_{b}": float(eb.mean()),
        "median_error_difference": diff,
        "significant_5pct": bool(p < 0.05),
        "better_model": a if diff < 0 else b,
    }


def day_masks(days: pd.DataFrame) -> dict[str, pd.Series]:
    """Máscaras por tipo de día, a partir de las exógenas del nivel."""
    idx = pd.DatetimeIndex(days.index)
    open_day = days["open_share"] >= 0.5
    return {
        "promo": (days["promo_share"] >= 0.5) & open_day,
        "sin_promo": (days["promo_share"] < 0.5) & open_day,
        "festivo": days["state_holiday"].astype(str) != "0",
        "inicio_de_mes": pd.Series(idx.day <= MONTH_START_DAYS, index=days.index),
        "domingo": pd.Series(idx.dayofweek == 6, index=days.index),
    }


def errors_by_day_type(
    fc: pd.DataFrame, history: pd.DataFrame, uid: str = "total", split: str = "test"
) -> pd.DataFrame:
    """MAE y MAPE de cada modelo por tipo de día. Omite los tipos sin observaciones."""
    days = history[history["unique_id"] == uid].set_index("ds").sort_index()
    rows = []
    for model, g in fc[(fc["unique_id"] == uid) & (fc["split"] == split)].groupby("model"):
        g = g.set_index("ds").sort_index()
        masks = day_masks(days.reindex(g.index))
        for name, mask in masks.items():
            part = g[mask.to_numpy()]
            if part.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "day_type": name,
                    "n": int(len(part)),
                    "mae": mae(part["y"], part["y_hat"]),
                    "mape": mape(part["y"], part["y_hat"]),
                }
            )
    return pd.DataFrame(rows)


def _records(df: pd.DataFrame) -> list[dict]:
    """Registros JSON válidos: NaN pasa a null."""
    return json.loads(df.to_json(orient="records"))


def build_comparison(root: Path = ARTIFACTS, history: pd.DataFrame | None = None) -> dict:
    """Arma el contenido de `comparison.json`."""
    history = load_history() if history is None else history
    fc = load_forecasts(root)
    return {
        "level": "total",
        "split": "test",
        "wilcoxon": {f"{a}_vs_{b}": wilcoxon_test(fc, a, b) for a, b in PAIRS},
        "table": _records(final_table(load_metrics(root))),
        "errors_by_day_type": _records(errors_by_day_type(fc, history)),
    }


def main() -> None:
    path = ARTIFACTS / "metrics" / "comparison.json"
    path.write_text(json.dumps(build_comparison(), indent=2), encoding="utf-8")
    print(f"OK → {path}")


if __name__ == "__main__":
    main()
