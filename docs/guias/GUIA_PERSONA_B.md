# Guía Persona B: GRU, aplicación web y despliegue

> Proyecto final de Inteligencia de Minería de Datos · Predicción de demanda Rossmann con ARIMA/SARIMAX y GRU + app web.
> **Este archivo es para Persona B.** La guía de Persona A está en `docs/guias/GUIA_PERSONA_A.md`.
> **Regla de oro para Claude Code:** no modificar archivos de Persona A ni archivos compartidos sin confirmación explícita, y comparar toda mejora contra este contexto antes de aplicarla (sección 6).

## 0. Cómo usar este archivo con Claude Code

1. Clonen el repositorio y copien las dos guías en `docs/guias/` (`GUIA_PERSONA_A.md` y `GUIA_PERSONA_B.md`). Las dos se versionan para que cada uno sepa qué hace el otro.
2. Al abrir Claude Code en el repo, el primer mensaje de cada sesión debe ser:
   > Trabajo como **Persona B**. Lee `docs/guias/GUIA_PERSONA_B.md` completo antes de hacer cualquier cosa y respeta sus restricciones.
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
2. **Archivos de la otra persona (`[A]`): NO los toques.** Si crees que necesitan un cambio:
   - detente y explícale al usuario qué cambiarías, por qué y qué impacto tiene;
   - pide confirmación explícita con este formato: *"¿Confirmas que modifique `<ruta>` de Persona A? (sí/no)"*;
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
### P-<nn>: <título> (<fecha>, detectada por Persona B)
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

- Rama por tarea: `feature/b-<tema>` desde `dev`. PR hacia `dev`; `dev → main` solo cuando la app funcione.
- Commits convencionales: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `cross:` (cambio autorizado en el área del otro).
- Antes de cada commit corre `ruff check . && black --check . && pytest -q`.
- Código definitivo en `src/` o `app/`; los notebooks solo importan desde `src/` y no definen lógica que otra parte necesite.
- Funciones con docstring y tipos; nada de rutas absolutas (usa `src/config.py`).
- Al terminar una tarea, marca su casilla en esta guía (sección de tareas) y resume en el PR: qué se hizo, cómo se probó y qué métricas cambiaron.
- Si una instrucción del usuario contradice esta guía, señala la contradicción y pide confirmación antes de seguir.

---

## 7. Tu rol: Persona B, GRU, aplicación web y despliegue

Eres dueña o dueño del **modelo GRU, la infraestructura del repo, la aplicación web y el despliegue** (fase de despliegue de CRISP-DM y parte del modelado). Lo clave es **no depender de los tiempos de Persona A**: la app se construye primero con datos falsos que cumplen el contrato y luego se cambian por los reales sin tocar la lógica.

**Lo que recibes de Persona A (y lo que haces mientras tanto):**

| Recibes | Fecha | Mientras tanto usa |
| --- | --- | --- |
| `src/config.py` | Día 1 | — (se hace juntos) |
| `data/processed/rossmann.parquet` + `src/features/*` | Fin de semana 1 | Prepara el código de ventanas con datos sintéticos |
| `artifacts/series/history.parquet` | Fin de semana 1 | `app/mock_data.py` |
| `src/evaluation/metrics.py` | Mitad de semana 2 | No escribas tus propias métricas: espera o usa un stub local en `src/models/gru.py` marcado `# TODO quitar` |
| `artifacts/models/sarimax_*.pkl` + `build_future_exog` | Mitad de semana 3 | Un simulador falso en `app/mock_data.py` (efecto de promo fijo del +X %) |

---

### Tarea B0: arranque e infraestructura (semana 1)
- [ ] Día 1 con A: objetivo, contrato y fechas. Revisar y aprobar el PR de `src/config.py`.
- [ ] Crear el repo en GitHub con la estructura de la sección 4 (carpetas vacías con `.gitkeep`), ramas `main` y `dev`, y protección de `main` (PR obligatorio).
- [x] `.gitignore` (compartido, primera versión tuya): `data/`, `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`, `.env`, `CLAUDE.md`. **No** ignores `artifacts/`.
- [x] `pyproject.toml` con la configuración de ruff y black (línea de 100 caracteres) y `.pre-commit-config.yaml` (ruff, black, end-of-file-fixer, check-added-large-files con `--maxkb=25000`).
- [x] `.github/workflows/ci.yml`:
```yaml
name: ci
on: [pull_request, push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r app/requirements.txt pytest ruff black
      - run: ruff check . && black --check .
      - run: pytest -q -m "not slow"
```
  (Los tests que necesitan TensorFlow o datos crudos se marcan `@pytest.mark.slow`.)
- [ ] GitHub Projects con una tarjeta por tarea A* y B* de las dos guías.
- [x] `docs/DECISIONES.md` y `docs/PROPUESTAS.md` con su encabezado y D-00 (archivos de forecast separados por modelo).

**Hecho cuando:** un PR de prueba dispara la CI en verde.

---

### Tarea B1: esqueleto de la app con datos falsos (semana 1)

Estructura:
```
app/
├─ Home.py                    # Inicio / metodología
├─ pages/
│  ├─ 1_Dashboard.py
│  ├─ 2_Escenarios.py
│  └─ 3_Series.py
├─ utils/
│  ├─ loaders.py              # leer y validar artifacts
│  ├─ charts.py               # figuras Plotly reutilizables
│  └─ sidebar.py              # selector común: nivel, modelo, horizonte
├─ mock_data.py               # genera artifacts falsos con el esquema del contrato
└─ requirements.txt           # streamlit, pandas, pyarrow, plotly, statsmodels, numpy, scipy (SIN tensorflow)
```

- Al inicio de cada página, para poder importar `src/`:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # en Home.py: parents[1]
```
- `loaders.py`:
```python
REQUIRED = ["unique_id", "ds", "y", "y_hat", "lo_95", "hi_95", "model", "split"]

@st.cache_data
def load_forecasts() -> pd.DataFrame:
    files = sorted((ARTIFACTS / "forecasts").glob("*.parquet"))
    if not files:                      # todavía no hay modelos reales
        return mock_data.forecasts()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Contrato roto, faltan columnas: {missing}")
    return df
```
  Igual para `load_metrics()` (une `artifacts/metrics/*.json`) y `load_history()`.
- `mock_data.py` genera series sintéticas con estacionalidad semanal y anual y un efecto de promo, **con el mismo esquema de la sección 5**. Así, cuando lleguen los artifacts reales, la app funciona sin cambios.
- Primer despliegue en Streamlit Community Cloud apuntando a `app/Home.py` (rama `dev` para pruebas, `main` para la versión final). La plataforma busca el `requirements.txt` en la carpeta del archivo de entrada; verifica en los logs del despliegue que tomó `app/requirements.txt`.

Tests (`tests/test_app_loaders.py`): los loaders con mock devuelven el esquema correcto; un parquet sin columnas lanza error.

**Hecho cuando:** la URL pública abre las 4 páginas con datos falsos.

---

### Tarea B2: GRU, preparación (semana 2, `src/models/gru.py`)

Con `history.parquet` o con el parquet procesado y las funciones de `src/features` de A (**solo importar, no modificar**):
- [x] Entradas por día: `log1p(y)` escalado, `promo_share`, `open_share`, `school_holiday_share`, `sh_a/b/c`, día de la semana (one-hot) y mes (seno/coseno).
- [x] Ventanas: **56 días pasados** → **42 días futuros**, con una segunda entrada de exógenas futuras conocidas (calendario de promos, festivos, `open_share`).
```python
def make_windows(arr_past, arr_fut_exog, y, lookback=56, horizon=42, end_idx=None):
    """Devuelve X_past (n,56,f), X_fut (n,42,k), Y (n,42).
    Solo crea ventanas cuyo objetivo termina en o antes de end_idx (evita fuga)."""
```
- [x] `MinMaxScaler` ajustado **solo** con las filas ≤ TRAIN_END; guardar el escalador con joblib en `artifacts/models/gru_scaler.joblib` (archivo tuyo).
- [x] Tests (`tests/test_gru.py`, marcados `slow` si importan TensorFlow): formas correctas; ninguna ventana de entrenamiento tiene objetivo después de TRAIN_END.

### Tarea B3: GRU, entrenamiento (semanas 2–3, `notebooks/05_gru.ipynb` en Colab/Kaggle)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(SEED)

past = keras.Input(shape=(56, n_feat), name="past")
fut  = keras.Input(shape=(42, n_exog), name="future_exog")
x = layers.GRU(64, return_sequences=True)(past)
x = layers.Dropout(0.2)(x)
x = layers.GRU(32)(x)
f = layers.Flatten()(fut)
h = layers.Concatenate()([x, f])
h = layers.Dense(64, activation="relu")(h)
out = layers.Dense(42, name="y")(h)
model = keras.Model([past, fut], out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber())
cb = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
      keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)]
hist = model.fit([Xp_tr, Xf_tr], Y_tr, validation_data=([Xp_va, Xf_va], Y_va),
                 epochs=300, batch_size=32, callbacks=cb)
```
- [x] **v1:** serie `total`. Con ~900 días hay pocas ventanas y riesgo de sobreajuste: grafica las curvas de aprendizaje (pérdida de train vs val) y coméntalas.
- [ ] ~~**v2 (si v1 sobreajusta o rinde menos que SARIMAX):**~~ no hizo falta: con D-13 la v1 supera a SARIMAX en validación. modelo global entrenado con ventanas de `type_a…d` (y de las tiendas) escaladas por serie, evaluado sobre cada nivel. Antes de hacerlo, regístralo como propuesta (6.2), porque cambia el tiempo del cronograma.
- [x] 2–3 variantes (unidades 32/64/128, lookback 28/56, dropout) registradas en `experiments/experiments.csv`: fecha, modelo, parámetros, MAPE y RMSE de validación. Se elige **por validación**.
- [x] Predicciones con origen fijo, igual que A:
  - `val`: entrada = 56 días hasta TRAIN_END → 42 días de validación;
  - `test`: reentrenar (o seguir) con datos hasta VAL_END → 42 días de prueba;
  - `future`: datos hasta TEST_END + `build_future_exog` de A → 42 días futuros.
- [x] Revertir el escalado y log1p, y escribir `artifacts/forecasts/gru.parquet` (`lo_95`/`hi_95` = NaN o un intervalo por MC Dropout si sobra tiempo) y `artifacts/metrics/gru.json` con `compute_all` de A.
- [ ] Guardar el modelo `.keras` **fuera** de `artifacts/` (en tu Drive), o en `artifacts/models/gru.keras` solo si pesa < 25 MB. La app no lo carga.

`tests/test_contract_b.py`: `gru.parquet` cumple el esquema, tiene 42 filas por `unique_id` y por `split` y no tiene NaN en `y_hat`.

---

### Tarea B4: página Dashboard de predicción (semana 2–3)

- Barra lateral (`utils/sidebar.py`): nivel (`unique_id`), modelos (selección múltiple), horizonte (1–42), periodo (`test` o `future`).
- Tarjetas `st.metric`: ventas previstas totales del horizonte, variación frente a los mismos días del año anterior (desde history), día pico previsto, MAPE de test del modelo.
- Gráfico Plotly: últimos 90 días reales + predicción + banda `lo_95`–`hi_95` (relleno semitransparente).
- Pestaña "Comparar modelos": todas las líneas sobre `test` + tabla de métricas (desde `load_metrics()`), resaltando el mejor por métrica + el resultado de Wilcoxon si existe `artifacts/metrics/comparison.json`.
- `st.download_button` con el CSV del pronóstico filtrado.

### Tarea B5: página Simulación de escenarios (semana 3)

- Entradas del usuario para el periodo `future`:
  - días con promoción: `st.date_input` de rango + `st.multiselect` de días de la semana, o los escenarios predefinidos "Sin promociones", "Patrón histórico" y "Promo intensiva (lun–vie todas las semanas)";
  - festivos escolares (rango de fechas) y festivos estatales puntuales;
  - `promo2_share` (slider 0–1, por defecto el histórico).
- Lógica (la única inferencia en vivo):
```python
@st.cache_resource
def load_sarimax(uid):
    meta = json.loads((ARTIFACTS/"models"/f"sarimax_{uid}_meta.json").read_text())
    return load_pickle(ARTIFACTS/"models"/f"sarimax_{uid}.pkl"), meta

res, meta = load_sarimax(uid)
dates = pd.date_range(FUTURE_START, periods=horizon, freq="D")
X_base = build_future_exog(uid, dates)                         # patrón histórico
X_scen = build_future_exog(uid, dates, promo_days=..., school_holiday_days=..., ...)
f_base = res.get_forecast(steps=horizon, exog=X_base[meta["exog_cols"]])
f_scen = res.get_forecast(steps=horizon, exog=X_scen[meta["exog_cols"]])
# si meta["log_target"]: aplicar np.expm1 a la media y a los intervalos
```
- Salidas: dos curvas (base vs escenario) con intervalos, ventas incrementales en € y %, tabla por semana y por día de la semana.
- Aviso visible: el simulador usa SARIMAX y solo vale para las variables que el modelo incluye y dentro de los rangos históricos. La GRU no participa del simulador (sus predicciones son fijas).
- Hasta que A entregue los pkl: `mock_data.simulate(...)` con un efecto de promo fijo y una etiqueta "DATOS DE PRUEBA".

### Tarea B6: página Visualización temporal (semana 3)

- Serie histórica (`load_history()`) con `rangeslider` de Plotly y selector de agregación diaria/semanal/mensual (`resample`).
- Descomposición STL calculada en la app (`statsmodels.tsa.seasonal.STL(period=7)`) en 4 subgráficos, con `@st.cache_data`.
- Heatmap día de la semana × mes de la venta promedio; barras de venta promedio con y sin promo por StoreType.
- ACF/PACF (`statsmodels.tsa.stattools.acf/pacf`) como gráficos de barras de Plotly.

### Tarea B7: Inicio, README y despliegue final (semana 4)

- `Home.py`: objetivo SMART, fases CRISP-DM con lo que se hizo en cada una, descripción de los datos, limitaciones (datos hasta 2015, sin clima, simulador lineal en log). Los textos de metodología los entrega A; tú los pegas.
- `README.md`: descripción, URL pública, capturas, cómo reproducir (`make_dataset` → notebooks → artifacts → `streamlit run app/Home.py`), estructura, dueños y tabla final de métricas.
- Merge `dev → main` con la CI en verde; verificar la app en una ventana de incógnito (despierta tras inactividad, sin errores en los logs, memoria < 1 GB).
- Ensayo de la demo con A: guion de 5 minutos (problema → datos → estacionalidad → modelos → métricas → escenario de promo).

---

### Cronograma de Persona B

| Semana | Tareas | Sincronización con A |
| --- | --- | --- |
| 1 | B0, B1 (app con mock + primer despliegue) | Viernes: recibir parquet y history, revisar el contrato |
| 2 | B2, B3 v1, B4 con el baseline real | Mitad: usar `metrics.py` de A · Viernes: primera tabla comparativa |
| 3 | B3 ajuste/v2, B5 con el SARIMAX real, B6 | Mitad: recibir pkl + meta · Viernes: app completa en `dev` |
| 4 | B7 | Mitad: congelar código · Final: ensayo de la demo |

### Lo que NO haces (área de Persona A)
`src/data/`, `src/features/`, `src/models/baseline.py`, `src/models/arima.py`, `src/evaluation/`, `notebooks/01–04`, `artifacts/**/naive|sarima|sarimax.*`, `artifacts/models/sarimax_*`, `artifacts/series/history.parquet`, `artifacts/metrics/comparison.json`, `tests/test_data.py`, `tests/test_features.py`, `tests/test_metrics.py`, `tests/test_contract_a.py`. Si `build_future_exog`, las métricas o el history no te sirven tal como están → regla 6.1 (proponer, no editar).
