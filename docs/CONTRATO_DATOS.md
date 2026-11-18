# Contrato de datos (interfaz entre Persona A y Persona B)

> Copia de la sección 5 de las guías. Archivo COMPARTIDO: los cambios se proponen en `docs/PROPUESTAS.md`, los aprueban los dos y se actualizan aquí y en los tests de contrato.

> Mejora D-00 frente al documento de planificación: en vez de un único `forecasts.parquet` donde escriben los dos (y que generaría conflictos de Git), **cada modelo escribe su propio archivo** y la app los concatena. El esquema de columnas no cambia.

### 5.1 `artifacts/forecasts/<modelo>.parquet`

| Columna | Tipo | Regla |
| --- | --- | --- |
| `unique_id` | str | `total`, `type_a`…`type_d`, `store_<n>` |
| `ds` | datetime64[ns] | fecha diaria |
| `y` | float | venta real en euros (NaN cuando `split == "future"`) |
| `y_hat` | float | predicción en euros (ya revertida de log1p) |
| `lo_95`, `hi_95` | float | intervalo al 95 % (NaN si el modelo no lo da) |
| `model` | str | `naive`, `sarima`, `sarimax`, `gru` |
| `split` | str | `val`, `test`, `future` |

### 5.2 `artifacts/metrics/<modelo>.json`

```json
{"model": "sarimax",
 "levels": {"total": {"val":  {"rmse": 0.0, "mae": 0.0, "mape": 0.0},
                      "test": {"rmse": 0.0, "mae": 0.0, "mape": 0.0}}}}
```
Las métricas se calculan **siempre** con `src/evaluation/metrics.py::compute_all`.

### 5.3 `artifacts/series/history.parquet` (lo produce A, lo lee B)

Columnas: `unique_id, ds, y, promo_share, open_share, school_holiday_share, state_holiday` (str: `0`, `a`, `b`, `c`), `promo2_share`. Cubre 2013-01-01 a 2015-07-31 para todos los `unique_id`.

### 5.4 Modelos SARIMAX para el simulador (los produce A, los lee B)

- `artifacts/models/sarimax_<unique_id>.pkl`: modelo **reajustado con todos los datos** (hasta 2015-07-31), guardado con `results.save(path)`.
- `artifacts/models/sarimax_<unique_id>_meta.json`:
```json
{"unique_id": "total", "order": [1,0,1], "seasonal_order": [1,1,1,7],
 "exog_cols": ["promo_share","open_share","school_holiday_share","sh_a","sh_b","sh_c","promo2_share","f_sin1","f_cos1","f_sin2","f_cos2","f_sin3","f_cos3"],
 "log_target": true, "train_end": "2015-07-31"}
```

### 5.5 Función pública para el simulador (la escribe A, la llama B, su firma no cambia sin acuerdo)

```python
# src/features/exog.py
def build_future_exog(
    unique_id: str,
    dates: pd.DatetimeIndex,
    promo_days: set[pd.Timestamp] | None = None,       # None = patrón histórico
    school_holiday_days: set[pd.Timestamp] | None = None,
    state_holidays: dict[pd.Timestamp, str] | None = None,  # {"2015-08-15": "a"}
    open_share: float | pd.Series | None = None,       # None = promedio histórico por día de la semana
    promo2_share: float | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por `dates` con EXACTAMENTE las columnas
    de meta['exog_cols'] y en ese orden. Sin NaN."""
```

### 5.6 Constantes compartidas (`src/config.py`)

```python
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END   = pd.Timestamp("2015-05-08")
VAL_START,  VAL_END  = pd.Timestamp("2015-05-09"), pd.Timestamp("2015-06-19")
TEST_START, TEST_END = pd.Timestamp("2015-06-20"), pd.Timestamp("2015-07-31")
FUTURE_START, FUTURE_END = pd.Timestamp("2015-08-01"), pd.Timestamp("2015-09-11")
HORIZON = 42
SEASON = 7
SEED = 42

LEVELS_TYPES = ["type_a", "type_b", "type_c", "type_d"]
SELECTED_STORES: list[int] = []   # los define Persona A tras el EDA
MODELS = ["naive", "sarima", "sarimax", "gru"]
```

Cualquier cambio en las secciones 5.1–5.6 es un **cambio de contrato**: se propone en `docs/PROPUESTAS.md`, lo aprueban los dos y se actualiza `docs/CONTRATO_DATOS.md` y los tests de contrato.
