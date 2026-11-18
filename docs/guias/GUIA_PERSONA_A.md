# Guía Persona A: datos, estacionalidad y modelos ARIMA

> Proyecto final de Inteligencia de Minería de Datos · Predicción de demanda Rossmann con ARIMA/SARIMAX y GRU + app web.
> **Este archivo es para Persona A.** La guía de Persona B está en `docs/guias/GUIA_PERSONA_B.md`.
> **Regla de oro para Claude Code:** no modificar archivos de Persona B ni archivos compartidos sin confirmación explícita, y comparar toda mejora contra este contexto antes de aplicarla (sección 6).

## 0. Cómo usar este archivo con Claude Code

1. Clonen el repositorio y copien las dos guías en `docs/guias/` (`GUIA_PERSONA_A.md` y `GUIA_PERSONA_B.md`). Las dos se versionan para que cada uno sepa qué hace el otro.
2. Al abrir Claude Code en el repo, el primer mensaje de cada sesión debe ser:
   > Trabajo como **Persona A**. Lee `docs/guias/GUIA_PERSONA_A.md` completo antes de hacer cualquier cosa y respeta sus restricciones.
3. Opcional: cada persona puede crear en su copia local un `CLAUDE.md` en la raíz con solo esa línea, y excluirlo de Git con `echo CLAUDE.md >> .git/info/exclude`. Así no choca con el `CLAUDE.md` del compañero.
4. Este archivo es **autosuficiente**: contiene el contexto del proyecto, la arquitectura, el contrato de datos, las reglas y las tareas.

---

## 1. Contexto del proyecto

**Materia:** Inteligencia de Minería de Datos (proyecto final, equipo de 2 personas).

**Enunciado:** desarrollar un modelo predictivo de series de tiempo que estime la demanda futura integrando variables externas para optimizar la planificación de marketing. Dataset: Rossmann Store Sales (https://www.kaggle.com/competitions/rossmann-store-sales).

**Requerimientos técnicos:** modelos ARIMA, modelo GRU (opcional pero lo vamos a hacer), análisis de estacionalidad, métricas RMSE, MAE y MAPE.

**Requerimientos funcionales:** aplicación web desplegada con dashboard de predicción, simulación de escenarios y visualización temporal.

**Metodología:** CRISP-DM (comprensión del negocio → comprensión de los datos → preparación → modelado → evaluación → despliegue).

**Objetivo SMART:** en 4 semanas, pronosticar las ventas diarias de los próximos 42 días a nivel total, por tipo de tienda (StoreType a/b/c/d) y para 5 tiendas seleccionadas, superando al baseline ingenuo estacional y con MAPE < 15 % en el periodo de prueba. Desplegarlo en una app web pública con simulación de escenarios de promoción.

**Preguntas de negocio que la app responde:**
- ¿Cuánto venderemos en las próximas semanas?
- ¿Cuánto suben las ventas si activamos promociones en ciertos días?
- ¿Qué tipos de tienda responden mejor a las promociones?
- ¿En qué épocas conviene concentrar el presupuesto de marketing?

### 1.1 Datos

| Archivo | Columnas | Uso |
| --- | --- | --- |
| `train.csv` | Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday (2013-01-01 a 2015-07-31, ~1 millón de filas) | Serie objetivo y variables exógenas diarias |
| `store.csv` | Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval | Exógenas estáticas y agrupación |
| `test.csv` | Igual a train pero sin Sales ni Customers | **No sirve para evaluar.** Solo como referencia de calendario futuro |

Problemas conocidos (a verificar en el EDA):
- `StateHoliday` mezcla `0` numérico y `'0'` texto → leer con `dtype={"StateHoliday": str}`.
- Días cerrados (`Open = 0`) tienen `Sales = 0`; hay algunos días abiertos con ventas 0 (anomalía).
- Un grupo de tiendas no tiene registros en parte de 2014 (huecos).
- `PromoInterval` usa abreviaturas en inglés con "Sept" (no "Sep").
- Nulos en `CompetitionDistance` y en las columnas `*Since*`.

### 1.2 Reglas de minería de datos que NO se negocian

- **No usar `Customers` como variable exógena.** El número de clientes del futuro no se conoce cuando se pronostica: usarlo es fuga de datos.
- **Partición temporal, nunca aleatoria.** Fechas fijas (están en `src/config.py`):
  - Entrenamiento: 2013-01-01 a 2015-05-08
  - Validación: 2015-05-09 a 2015-06-19 (42 días)
  - Prueba: 2015-06-20 a 2015-07-31 (42 días)
  - Futuro (para la app): 2015-08-01 a 2015-09-11 (42 días)
- **El conjunto de prueba se usa una sola vez**, al final. Las decisiones (órdenes ARIMA, hiperparámetros de la GRU) se toman con validación.
- **Escaladores y transformaciones se ajustan solo con entrenamiento.**
- **Todo modelo se compara contra el baseline ingenuo estacional** (venta del mismo día de la semana anterior).
- **MAPE solo sobre días con venta real > 0** (los días cerrados con Sales = 0 hacen explotar el MAPE).
- **GIGO:** la calidad del modelo depende de la de los datos. Se revisan exactitud, completitud, consistencia, credibilidad y actualidad (ISO/IEC 25012).

---

## 2. Stack tecnológico (todo gratis)

| Capa | Herramienta |
| --- | --- |
| Lenguaje | Python 3.11 |
| Datos | pandas, numpy, pyarrow (parquet) |
| Series de tiempo | statsmodels (SARIMAX, STL, ADF, KPSS, ACF/PACF, Ljung-Box), statsforecast (AutoARIMA, cross_validation) |
| Deep learning | TensorFlow / Keras, entrenado en Google Colab o Kaggle Notebooks con GPU |
| Utilidades ML | scikit-learn (MinMaxScaler, métricas), scipy (Wilcoxon) |
| Visualización | Plotly (app), matplotlib/seaborn (notebooks) |
| App web | Streamlit (multipágina) |
| Despliegue | Streamlit Community Cloud (plan B: Render) |
| Versionado y CI | Git + GitHub + GitHub Actions + GitHub Projects |
| Calidad | pytest, ruff, black, pre-commit |
| Experimentos | `experiments/experiments.csv` (o MLflow local) |

No se agregan librerías nuevas sin pasar por el **protocolo de mejoras** (sección 6).

---

## 3. Arquitectura

**Monolito modular con capa de modelos precalculada:** el entrenamiento ocurre fuera de la app (notebooks y Colab), y la app solo lee archivos de `artifacts/`. La única inferencia en vivo es el SARIMAX del simulador, porque es liviano. TensorFlow **no** se instala en la app desplegada.

```
[Kaggle API] → data/raw/*.csv                      (no se versiona)
      ↓  src/data + src/features                    (Persona A)
data/processed/rossmann.parquet                    (no se versiona)
      ↓
 ┌───────────── entrenamiento offline ─────────────┐
 │ Persona A: baseline, SARIMA, SARIMAX            │
 │ Persona B: GRU (Colab/Kaggle con GPU)           │
 └─────────────────────────────────────────────────┘
      ↓  escriben según el CONTRATO DE DATOS
artifacts/                                          (SÍ se versiona, debe ser liviano)
  ├─ series/history.parquet
  ├─ forecasts/<modelo>.parquet
  ├─ metrics/<modelo>.json
  └─ models/sarimax_<unique_id>.pkl + sarimax_<unique_id>_meta.json
      ↓
app/ (Streamlit, Persona B) → Streamlit Community Cloud (rama main)
```

**Niveles de agregación (`unique_id`):**
- `total`: suma de todas las tiendas por día
- `type_a`, `type_b`, `type_c`, `type_d`: suma por StoreType
- `store_<n>`: 5 tiendas seleccionadas por Persona A en el EDA (una por tipo + una con mucha promo) y anotadas en `src/config.py`

**Decisión D-01 (por defecto, a validar en el EDA):** serie diaria que conserva los domingos, con estacionalidad s = 7 y la exógena `open_share` (proporción de tiendas abiertas, que se conoce de antemano). El MAPE se calcula solo sobre días con y > 0.

---

## 4. Estructura del repositorio y dueños

```
rossmann-forecast/
├─ data/                         # .gitignore
│  ├─ raw/  processed/
├─ notebooks/
│  ├─ 01_eda.ipynb                     [A]
│  ├─ 02_estacionalidad.ipynb          [A]
│  ├─ 03_baseline_sarima.ipynb         [A]
│  ├─ 04_sarimax_exogenas.ipynb        [A]
│  └─ 05_gru.ipynb                     [B]
├─ src/
│  ├─ __init__.py                      [COMPARTIDO]
│  ├─ config.py                        [COMPARTIDO]
│  ├─ data/        load.py, clean.py, make_dataset.py        [A]
│  ├─ features/    calendar.py, exog.py                      [A]
│  ├─ models/      baseline.py, arima.py                     [A]
│  │               gru.py                                    [B]
│  └─ evaluation/  metrics.py, backtest.py, compare.py       [A]
├─ artifacts/
│  ├─ series/history.parquet           [A]
│  ├─ forecasts/naive|sarima|sarimax.parquet   [A]
│  ├─ forecasts/gru.parquet            [B]
│  ├─ metrics/naive|sarima|sarimax.json        [A]
│  ├─ metrics/gru.json                 [B]
│  └─ models/sarimax_*.pkl, *_meta.json        [A]
├─ app/
│  ├─ Home.py, pages/, utils/, mock_data.py, requirements.txt   [B]
├─ tests/
│  ├─ test_data.py, test_features.py, test_metrics.py, test_contract_a.py   [A]
│  └─ test_gru.py, test_app_loaders.py, test_contract_b.py                 [B]
├─ experiments/experiments.csv         [COMPARTIDO: solo agregar filas]
├─ docs/
│  ├─ guias/GUIA_PERSONA_A.md, GUIA_PERSONA_B.md   [COMPARTIDO]
│  ├─ CONTRATO_DATOS.md                [COMPARTIDO]
│  ├─ DECISIONES.md                    [COMPARTIDO: solo agregar]
│  └─ PROPUESTAS.md                    [COMPARTIDO: solo agregar]
├─ .github/workflows/ci.yml            [B]
├─ .pre-commit-config.yaml, pyproject.toml   [B]
├─ requirements.txt                    [COMPARTIDO: entorno de entrenamiento]
├─ .gitignore                          [COMPARTIDO]
└─ README.md                           [B, con secciones que escribe A]
```

- `[A]` = Persona A (datos, estacionalidad, ARIMA/SARIMAX, evaluación)
- `[B]` = Persona B (GRU, app, despliegue, infraestructura)
- `[COMPARTIDO]` = cualquier cambio requiere el visto bueno de las dos personas
- Los archivos marcados "solo agregar" permiten añadir entradas propias, nunca editar ni borrar las del otro

---

## 5. Contrato de datos (interfaz entre A y B)

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

---

## 6. RESTRICCIONES PARA CLAUDE CODE (obligatorias)

### 6.1 Límites de propiedad

1. **Antes de crear, modificar, mover o borrar cualquier archivo**, revisa la tabla de dueños de la sección 4.
2. **Archivos de la otra persona (`[B]`): NO los toques.** Si crees que necesitan un cambio:
   - detente y explícale al usuario qué cambiarías, por qué y qué impacto tiene;
   - pide confirmación explícita con este formato: *"¿Confirmas que modifique `<ruta>` de Persona B? (sí/no)"*;
   - solo un **"sí" explícito** en el chat autoriza el cambio. "Ok", silencio o ambigüedad = no;
   - aun con confirmación, haz el cambio mínimo, en un commit separado con el prefijo `cross:` y mencionándolo en el PR para que el dueño lo revise.
3. **Archivos `[COMPARTIDO]`:** mismo procedimiento. Además, recuerda al usuario que debe avisar al compañero antes de hacer merge.
4. **Leer los archivos del otro siempre está permitido** (importar sus funciones, leer sus artifacts, revisar su código).
5. **Si necesitas algo del otro que todavía no existe**, no lo implementes en su carpeta: crea un *stub* o datos falsos **dentro de tu propia área** que cumplan el contrato, y anota la dependencia en `docs/PROPUESTAS.md`.
6. **Nunca** hagas `git push` a `main`, `git push --force`, ni reescribas historia de ramas que no sean tuyas.
7. Nunca borres ni sobrescribas artifacts del otro (`artifacts/**/gru.*` o `artifacts/**/naive|sarima|sarimax.*` según corresponda).

### 6.2 Protocolo de mejoras (siempre comparar antes de aplicar)

Cuando durante el desarrollo detectes una posible mejora (otra librería, otra técnica, un refactor, otra arquitectura, un hiperparámetro, un cambio en el contrato), **no la apliques directamente**. Haz esto:

1. **Describe** la situación actual y la propuesta en 2–3 líneas.
2. **Compárala contra el contexto del proyecto** con esta tabla:

| Criterio | Actual | Propuesta |
| --- | --- | --- |
| ¿Cumple los requerimientos (ARIMA, GRU, estacionalidad, RMSE/MAE/MAPE, dashboard, escenarios, visualización)? | | |
| ¿Sigue siendo gratis y desplegable en Streamlit Community Cloud (~1 GB)? | | |
| ¿Cabe en el cronograma restante? (horas estimadas) | | |
| ¿Respeta las reglas de la sección 1.2 (sin fuga, partición temporal)? | | |
| ¿Cambia el contrato de datos (sección 5)? | | |
| ¿Afecta el área de la otra persona? | | |
| ¿Mejora una métrica o la explicación para el informe? (evidencia, no intuición) | | |
| ¿Se puede explicar con los conceptos del curso (CRISP-DM, evaluación, métricas)? | | |
| Riesgo de romper algo que ya funciona | | |

3. **Da una recomendación:** aplicar ahora / posponer / descartar, con una línea de justificación.
4. **Regístrala** en `docs/PROPUESTAS.md` con esta plantilla:
```markdown
### P-<nn>: <título> (<fecha>, detectada por Persona A)
- Situación actual:
- Propuesta:
- Comparación: <tabla resumida>
- Recomendación de Claude:
- Afecta a: solo mi área | área del otro | contrato
- Decisión del equipo: pendiente
```
5. **Aplica** solo si (a) el usuario lo aprueba explícitamente y (b) queda dentro de tu área. Si toca el área del otro o el contrato, se aplica la regla 6.1.
6. Cuando se aplique, agrega la entrada a `docs/DECISIONES.md` (`D-<nn>: qué, por qué, alternativa descartada, impacto medido`).
7. Una mejora con evidencia (por ejemplo, baja el MAPE de validación) siempre se mide **en validación**, nunca en prueba.

### 6.3 Forma de trabajar

- Rama por tarea: `feature/a-<tema>` desde `dev`. PR hacia `dev`; `dev → main` solo cuando la app funcione.
- Commits convencionales: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `cross:` (cambio autorizado en el área del otro).
- Antes de cada commit corre `ruff check . && black --check . && pytest -q`.
- Código definitivo en `src/` o `app/`; los notebooks solo importan desde `src/` y no definen lógica que otra parte necesite.
- Funciones con docstring y tipos; nada de rutas absolutas (usa `src/config.py`).
- Al terminar una tarea, marca su casilla en esta guía (sección de tareas) y resume en el PR: qué se hizo, cómo se probó y qué métricas cambiaron.
- Si una instrucción del usuario contradice esta guía, señala la contradicción y pide confirmación antes de seguir.

---

## 7. Tu rol: Persona A, datos, estacionalidad y modelos ARIMA

Eres dueña o dueño de las fases **comprensión de los datos, preparación, análisis de estacionalidad, modelos estadísticos y evaluación** de CRISP-DM. Tu trabajo alimenta todo lo demás, así que lo más urgente es entregar `data/processed/rossmann.parquet` y `artifacts/series/history.parquet` a tiempo.

**Lo que produces para Persona B (con fecha límite):**

| Entregable | Para qué lo usa B | Límite |
| --- | --- | --- |
| `src/config.py` acordado | Fechas y rutas | Día 1 |
| `data/processed/rossmann.parquet` + `src/features/*` | Entradas de la GRU | Fin de semana 1 |
| `artifacts/series/history.parquet` | Visualización temporal | Fin de semana 1 |
| `src/evaluation/metrics.py` con tests | Métricas de la GRU | Mitad de semana 2 |
| `artifacts/forecasts/naive.parquet` + métricas | Primer dashboard real | Fin de semana 2 |
| `artifacts/models/sarimax_*.pkl` + meta + `build_future_exog` | Simulador de escenarios | Mitad de semana 3 |
| `sarima`/`sarimax` forecasts + métricas | Comparación final | Fin de semana 3 |

---

### Tarea A0: arranque (día 1, junto con B)
- [ ] Acordar con B el objetivo SMART, el contrato (sección 5) y las fechas de partición.
- [ ] Crear `src/config.py` (contenido de la sección 5.6) y `docs/CONTRATO_DATOS.md` (copia de la sección 5). Abrir PR y pedir revisión a B (es compartido).
- [ ] Configurar el entorno: `python -m venv .venv`, `pip install -r requirements.txt`. Si te falta una librería en `requirements.txt` (compartido), propónla en el PR.
- [ ] Configurar la Kaggle API (`~/.kaggle/kaggle.json`) y aceptar las reglas de la competencia en la web de Kaggle.

**Hecho cuando:** `python -c "from src.config import *"` funciona y B aprobó el PR.

---

### Tarea A1: carga y limpieza (`src/data/`)

`src/data/make_dataset.py`:
1. Descarga: `kaggle competitions download -c rossmann-store-sales -p data/raw` y descomprimir.
2. `load.py`:
```python
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_RAW / "train.csv", parse_dates=["Date"],
                        dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(DATA_RAW / "store.csv")
    return train, store
```
3. `clean.py`, que **documenta cada decisión en `docs/DECISIONES.md`**:
   - normalizar `StateHoliday` a `{"0","a","b","c"}`;
   - marcar (no borrar todavía) días `Open == 1` con `Sales == 0` → columna `anomaly_open_zero`; luego decidir si excluirlos del entrenamiento;
   - unir con `store.csv` por `Store` (left join) y validar que no se pierdan filas;
   - `CompetitionDistance` nulo → mediana + bandera `comp_dist_missing`;
   - competencia sin fecha de apertura → asumir que ya estaba abierta (documentarlo);
   - ordenar por `Store, Date`.
4. Guardar `data/processed/rossmann.parquet`.

Tests (`tests/test_data.py`): no hay duplicados `(Store, Date)`, `StateHoliday` tiene un solo tipo, `Open==0 ⇒ Sales==0`, el número de filas se conserva después del merge.

**Hecho cuando:** `python -m src.data.make_dataset` regenera el parquet desde cero y los tests pasan.

---

### Tarea A2: EDA y calidad de datos (`notebooks/01_eda.ipynb`)

Sigue los pasos de EDA del curso y deja **una conclusión escrita debajo de cada gráfico**:
- [ ] Revisión inicial: filas, columnas, tipos, rango de fechas, tiendas por día (detecta los huecos de 2014).
- [ ] Resumen estadístico de Sales por StoreType, Assortment, DayOfWeek, Promo, StateHoliday.
- [ ] Outliers: boxplots por StoreType, top 10 días más altos y más bajos.
- [ ] Visual: serie total diaria, histograma (normal vs log1p), heatmap día de la semana × mes, Sales con y sin Promo por StoreType, Sales vs Customers (solo descriptivo; recuerda que no es exógena).
- [ ] Checklist ISO/IEC 25012 (exactitud, completitud, consistencia, credibilidad, actualidad) con lo encontrado.
- [ ] Elegir las **5 tiendas** (`SELECTED_STORES`): una por StoreType con historia completa, más una con alta respuesta a la promo. Proponer el cambio de `src/config.py` a B (compartido).
- [ ] Validar o reemplazar la decisión D-01 (domingos y `open_share`). Si cambias D-01, aplica el protocolo de la sección 6.2 porque afecta al contrato.

---

### Tarea A3: variables exógenas y series agregadas (`src/features/`)

`calendar.py`:
```python
def fourier_terms(dates: pd.DatetimeIndex, K: int = 3, period: float = 365.25) -> pd.DataFrame:
    t = dates.dayofyear.to_numpy()
    cols = {}
    for k in range(1, K + 1):
        cols[f"f_sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"f_cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=dates)
```

`exog.py`:
- `promo2_active(df)`: 1 si `Promo2 == 1`, la fecha es posterior a `Promo2Since` (usa `datetime.date.fromisocalendar(year, week, 1)`) y el mes está en `PromoInterval` (mapa de meses con `"Sept"`).
- `competition_active(df)`: 1 si la fecha ≥ apertura del competidor.
- `aggregate_series(df, level) -> pd.DataFrame` para `total`, `type_*` y `store_<n>`, con las columnas de la sección 5.3:
  - `y = sum(Sales)`
  - `promo_share = mean(Promo)` sobre tiendas abiertas
  - `open_share = mean(Open)`
  - `school_holiday_share = mean(SchoolHoliday)`
  - `state_holiday` = el valor más frecuente del día
  - `promo2_share = mean(promo2_active)`
  - frecuencia diaria completa (`asfreq("D")`); los huecos se documentan y se rellenan sin usar el futuro.
- `build_exog(series_df) -> pd.DataFrame`: exógenas históricas con dummies `sh_a`, `sh_b`, `sh_c` y Fourier, en el orden de `exog_cols`.
- `build_future_exog(...)`: firma **exacta** de la sección 5.5. Sus valores por defecto: promo = patrón histórico (mismos días de la semana y la misma alternancia quincenal de las últimas 8 semanas), `open_share` = promedio histórico por día de la semana, festivos = ninguno salvo los que se pasen.
- Exportar `artifacts/series/history.parquet` con todos los niveles.

Tests (`tests/test_features.py`): Promo2 activo en un caso conocido de "Jan,Apr,Jul,Oct"; `build_future_exog` devuelve las columnas de `exog_cols` en orden, sin NaN y con el índice pedido; ninguna función usa datos posteriores a la fecha de corte.

**Hecho cuando:** B puede hacer `from src.features.exog import build_future_exog` y obtener un DataFrame válido para `total`.

---

### Tarea A4: análisis de estacionalidad (`notebooks/02_estacionalidad.ipynb`)

Sobre `total` y cada `type_*`, con datos de **entrenamiento**:
- [ ] Descomposición STL (`statsmodels.tsa.seasonal.STL(y, period=7, robust=True)`), y otra con period=365 sobre la serie semanal.
- [ ] ACF y PACF (lags 0–60) de y, de la diferencia simple y de la diferencia estacional (lag 7).
- [ ] ADF (`adfuller`) y KPSS (`kpss`) sobre y, Δy y Δ₇y: tabla de p-valores e interpretación (H0 de ADF = no estacionaria; H0 de KPSS = estacionaria).
- [ ] Periodograma (`scipy.signal.periodogram`) para confirmar los periodos de 7 y ~365 días.
- [ ] Efectos de calendario: ventas promedio por día de la semana, por mes, antes y después de festivos, y con y sin promo.
- [ ] **Conclusión escrita:** d, D, s elegidos y por qué; por qué la estacionalidad anual va con Fourier y no con SARIMA s = 365.

---

### Tarea A5: métricas y baseline (`src/evaluation/`, `src/models/baseline.py`)

`metrics.py` (lo usan los dos; su firma es parte del contrato de hecho):
```python
def rmse(y, y_hat) -> float: ...
def mae(y, y_hat) -> float: ...
def mape(y, y_hat) -> float:
    """MAPE en %, solo sobre posiciones con y > 0."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    m = y > 0
    return float(np.mean(np.abs((y[m] - y_hat[m]) / y[m])) * 100)
def compute_all(y, y_hat) -> dict[str, float]:
    return {"rmse": rmse(y, y_hat), "mae": mae(y, y_hat), "mape": mape(y, y_hat)}
```
`backtest.py`: funciones `split(series)` usando `config` y `write_forecast(df, model)` / `write_metrics(dict, model)` que validan el esquema de la sección 5 antes de escribir.

`baseline.py`: ingenuo estacional, `y_hat(t) = y(t − 7)`, aplicado de forma recursiva para el horizonte de 42 días desde el origen (para val: origen = TRAIN_END; para test: origen = VAL_END; para future: origen = TEST_END).

Tests: métricas contra casos calculados a mano; MAPE ignora los ceros; `write_forecast` rechaza columnas faltantes.

**Hecho cuando:** existen `artifacts/forecasts/naive.parquet` y `artifacts/metrics/naive.json` para todos los niveles. Avisa a B.

---

### Tarea A6: SARIMA (`notebooks/03_baseline_sarima.ipynb`, `src/models/arima.py`)

- [ ] Sobre `log1p(y)` de entrenamiento, proponer (p,d,q)(P,D,Q)₇ a partir de ACF/PACF (tarea A4).
- [ ] Búsqueda pequeña por AIC (p,q ∈ {0,1,2}, P,Q ∈ {0,1}) con `SARIMAX(..., exog=None)`, y comparar con `statsforecast.models.AutoARIMA(season_length=7)`.
- [ ] Elegir con el **RMSE de validación**, no solo con el AIC.
- [ ] Diagnóstico: `res.plot_diagnostics()`, Ljung-Box (`acorr_ljungbox`) sobre los residuos.
- [ ] Escribir `artifacts/forecasts/sarima.parquet` (val y test) y las métricas. Para test, reajustar con train + val y pronosticar 42 días.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = SARIMAX(np.log1p(y_tr), order=order, seasonal_order=seas,
              enforce_stationarity=False, enforce_invertibility=False)
res = mod.fit(disp=False)
fc = res.get_forecast(steps=HORIZON)
y_hat = np.expm1(fc.predicted_mean); ci = np.expm1(fc.conf_int(alpha=0.05))
```

---

### Tarea A7: SARIMAX con exógenas (`notebooks/04_sarimax_exogenas.ipynb`)

- [ ] El mismo flujo de A6 agregando `exog=build_exog(...)` para `total` y `type_a…d`.
- [ ] Tabla de coeficientes: el de `promo_share` traducido a negocio (en log, `exp(β) − 1` ≈ % de aumento de ventas con promo total). Comparar entre StoreType → ¿quién responde mejor a la promo?
- [ ] Ljung-Box y gráfico de residuos.
- [ ] Validación con origen móvil: `StatsForecast(models=[AutoARIMA(season_length=7)], freq="D").cross_validation(df, h=42, n_windows=4, step_size=42)` con exógenas. Reportar media ± desviación de las métricas.
- [ ] Para las 5 tiendas: AutoARIMA de statsforecast con exógenas, en la misma tabla.
- [ ] Escribir `artifacts/forecasts/sarimax.parquet` (val, test **y future**) y las métricas.
- [ ] **Modelo para el simulador:** reajustar con **todos** los datos (hasta TEST_END) con los órdenes elegidos, y guardar `res.save(ARTIFACTS/"models"/f"sarimax_{uid}.pkl")` + el `_meta.json` de la sección 5.4.
- [ ] Probar que se carga y pronostica como lo hará la app:
```python
from statsmodels.iolib.smpickle import load_pickle
res = load_pickle(path)
X = build_future_exog("total", pd.date_range(FUTURE_START, FUTURE_END))
fc = res.get_forecast(steps=len(X), exog=X[meta["exog_cols"]])
```
- [ ] Tamaño de cada `.pkl` < 25 MB. Si pesa más, evalúa `remove_data=True` con el **protocolo 6.2** y verifica que `get_forecast` siga funcionando después.

**Hecho cuando:** existe `tests/test_contract_a.py`, que carga cada pkl, llama a `build_future_exog` y obtiene 42 predicciones finitas y positivas.

---

### Tarea A8: evaluación final (semana 4)

- [ ] Correr **una sola vez** todas las predicciones de test (ya generadas) y la tabla final: modelo × nivel × RMSE/MAE/MAPE, más si supera al baseline. Lee `artifacts/metrics/gru.json` de B (solo lectura).
- [ ] Prueba de Wilcoxon (`scipy.stats.wilcoxon`) entre los errores absolutos diarios de SARIMAX y GRU, y de SARIMAX vs baseline, en `total`. Guardar en `artifacts/metrics/comparison.json` (archivo tuyo).
- [ ] Errores por tipo de día (promo / no promo, festivo, inicio de mes).
- [ ] Redactar en el informe las secciones: comprensión de datos, preparación, estacionalidad, ARIMA/SARIMAX, evaluación y conclusiones de negocio para marketing.
- [ ] Entregar a B el texto de las secciones de metodología para el README y para la página Inicio de la app (B lo pega; tú no editas `app/` ni `README.md`).

---

### Cronograma de Persona A

| Semana | Tareas | Sincronización con B |
| --- | --- | --- |
| 1 | A0, A1, A2, A3 (parquet + history) | Viernes: entregar parquet, history y `build_future_exog` en borrador |
| 2 | A4, A5, A6 | Mitad: métricas listas · Viernes: baseline + SARIMA en el dashboard |
| 3 | A7 | Mitad: pkl + meta para el simulador · Viernes: app completa en `dev` |
| 4 | A8 + informe | Mitad: congelar código · Final: ensayo de la demo |

### Lo que NO haces (área de Persona B)
`src/models/gru.py`, `notebooks/05_gru.ipynb`, todo `app/`, `.github/`, `pyproject.toml`, `.pre-commit-config.yaml`, `README.md`, `artifacts/**/gru.*`, `tests/test_gru.py`, `tests/test_app_loaders.py`, `tests/test_contract_b.py`. Si algo de eso necesita un cambio → regla 6.1.
