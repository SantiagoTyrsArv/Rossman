# Registro de decisiones

> Archivo compartido: **solo agregar** entradas. No se editan ni borran las del compañero.
> Formato: `D-<nn>: qué, por qué, alternativa descartada, impacto medido`.

### D-00: un archivo de pronóstico por modelo (Persona B, 2026-11-15)
- **Qué:** cada modelo escribe `artifacts/forecasts/<modelo>.parquet` y `artifacts/metrics/<modelo>.json`; la app los concatena.
- **Por qué:** un único `forecasts.parquet` escrito por A y B generaría conflictos de Git en un binario.
- **Alternativa descartada:** `forecasts.parquet` compartido.
- **Impacto:** ninguno en el esquema de columnas (sección 5.1 del contrato).

### D-01: serie diaria con domingos, s = 7 y `open_share` (por defecto, a validar por A en el EDA)
- **Qué:** se conservan los domingos; estacionalidad semanal s = 7; exógena `open_share`. MAPE solo con y > 0.
- **Estado:** pendiente de validación por Persona A.

### D-02: la CI instala scikit-learn y joblib además de `app/requirements.txt` (Persona B, 2026-11-15)
- **Qué:** la línea `pip install` de `.github/workflows/ci.yml` agrega `scikit-learn joblib`.
- **Por qué:** `src/models/gru.py` usa `MinMaxScaler` y `joblib`, y sus tests (no lentos) corren en la CI. La app desplegada no los necesita, así que no se agregan a `app/requirements.txt`.
- **Alternativa descartada:** agregarlos a `app/requirements.txt` (infla el despliegue sin necesidad).
- **Impacto:** solo la CI.

### D-03: las guías de trabajo no se versionan (Persona B, 2026-11-15)
- **Qué:** `GUIA_PERSONA_A.md` y `GUIA_PERSONA_B.md` quedan fuera de Git (`.gitignore`).
- **Por qué:** decisión explícita del equipo; contradice el paso 0.1 de las guías (que pedía versionarlas en `docs/guias/`).
- **Impacto:** cada persona mantiene su copia local.

### D-04: las guías sí se versionan en `docs/guias/` (Persona B, 2026-11-15) · reemplaza a D-03
- **Qué:** `docs/guias/GUIA_PERSONA_A.md` y `docs/guias/GUIA_PERSONA_B.md` entran a Git, como indica el paso 0.1 de las guías.
- **Por qué:** decisión del equipo para que cada uno vea qué hace el otro.

### D-05: Persona B apoya a Persona A en las tareas A0–A3 (equipo, 2026-11-15)
- **Qué:** por acuerdo entre A y B, Persona B implementa A0 (config), A1 (carga y limpieza), A2 (EDA) y A3 (exógenas, series agregadas y `history.parquet`).
- **Cómo:** commits con prefijo `cross:` y PR hacia `dev` para que A los revise. A4 en adelante sigue siendo solo de A.
- **Impacto:** adelanta las entregas que bloqueaban la app y la GRU (P-01, P-03).

### D-06: `StateHoliday` como texto en {0, a, b, c} (A1, hecho por B, 2026-11-15)
- **Qué:** se lee con `dtype=str` y se normaliza (`0`, `0.0`, vacío → `"0"`); cualquier otro valor lanza error.
- **Por qué:** el CSV mezcla `0` numérico y `'0'` texto, lo que crea dos categorías falsas.

### D-07: días abiertos con ventas 0 se marcan, no se borran (A1)
- **Qué:** columna `anomaly_open_zero = (Open == 1) & (Sales == 0)`.
- **Por qué:** son pocos y su tratamiento (excluir o no del entrenamiento) se decide en el EDA (A2). En las series agregadas su efecto es despreciable.

### D-08: `CompetitionDistance` nulo → mediana + bandera `comp_dist_missing` (A1)
- **Alternativa descartada:** borrar las tiendas (perdería ventas reales del total).

### D-09: competencia sin fecha de apertura → se asume abierta desde siempre (A1)
- **Qué:** año 1900, mes 1, y bandera `comp_open_assumed`.
- **Por qué:** si hay distancia pero no fecha, lo más probable es que el competidor ya existiera antes de 2013.

### D-10: relleno del hueco de 2014 sin usar el futuro (A1)
- **Qué:** cada (tienda, fecha) faltante se crea con `imputed = 1`. Promo, SchoolHoliday y StateHoliday = moda del mismo día en las demás tiendas; Open = 1 si la mayoría de su StoreType abrió; Sales = promedio de la propia tienda en los 56 días **anteriores** al hueco con el mismo día de la semana y estado de promo. `Customers` queda NaN.
- **Por qué:** sin relleno, el total y el tipo de esas tiendas caen artificialmente ~180 tiendas durante medio año, lo que el modelo leería como un cambio de nivel real.
- **Alternativa descartada:** interpolar entre antes y después del hueco (usa información futura).
- **Impacto:** a validar en el EDA (A2) comparando el total con y sin relleno.
- **Impacto medido (EDA):** 33 121 tienda-días rellenados (180 tiendas en el 2.º semestre de 2014 y 1 tienda el 01/01/2013). Sin relleno, la venta total del 2.º semestre de 2014 quedaría un 15 % por debajo (1 042 M€ frente a 1 231 M€).

### D-11: `SELECTED_STORES = [9, 769, 504, 960, 693]` (A2, hecho por B, 2026-11-15)
- **Qué:** una tienda por StoreType (9 = a, 769 = b, 504 = c, 960 = d), con historia completa, sin días abiertos con venta 0 y con venta cercana a la mediana de su tipo, más la **693** (tipo d), que tiene la mayor respuesta a la promo (~+100 % de lunes a viernes) entre las tiendas con venta ≥ la mediana global.
- **Alternativa descartada:** la tienda 198 responde igual a la promo, pero vende ~2 900 €/día; en una tienda tan chica el MAPE es muy inestable.

### D-01 validada (EDA, 2026-11-15)
- Solo el 2,4 % de los tienda-días de domingo están abiertos y el domingo aporta el 0,5 % de la venta (el tipo b abre el 98 % de los domingos). Se mantiene la serie diaria con domingos, s = 7 y `open_share`.

### D-12: versiones fijas de pandas, numpy y statsmodels en `app/requirements.txt` (aplica P-08, Persona B, 2026-11-18)
- **Qué:** `pandas==2.3.3`, `numpy==2.4.6` y `statsmodels==0.15.0`, las versiones con las que A guardó `artifacts/models/sarimax_*.pkl`. El resto de dependencias sigue con rango.
- **Por qué:** con rangos, la CI ya instalaba pandas 3.1.1. Un pickle de statsmodels guarda objetos de pandas y numpy, así que un cambio de versión mayor en el despliegue puede impedir que el simulador cargue los modelos.
- **Alternativa descartada:** dejar los rangos (`>=`). Hoy funciona, pero cualquier versión nueva puede romper el simulador sin aviso.
- **Impacto medido:** las tres versiones tienen ruedas para Python 3.11 (CI) y 3.12. Si A regenera los pkl con otras versiones, hay que actualizar las tres juntas.

### D-13: GRU con decodificador por día (aplica P-09, Persona B, 2026-11-18)
- **Qué:** `build_model` deja de aplanar las exógenas futuras. El estado del codificador GRU se repite en los 42 días, se une a las exógenas de cada día y pasa por un decodificador GRU + `TimeDistributed(Dense)` con pesos compartidos en todo el horizonte.
- **Por qué:** la versión aplanada no aprendía (underfitting: MAPE 55,9 % dentro del entrenamiento) y fallaba en los festivos.
- **Alternativa descartada:** la v2 de la guía (modelo global con `type_*` y tiendas). Toma más horas y no hizo falta.
- **Impacto medido (validación, `total`):** MAPE de 62,7 % a 7,6 % y RMSE de 2 389 656 a 658 021 (SARIMAX: 8,9 % / 738 378). Variante elegida por validación: 64 unidades, lookback 56, dropout 0,2, 40 épocas. En prueba, que se usó una sola vez: MAPE 5,5 %, RMSE 580 806.

### D-14: diferenciación s = 7, D = 1 y d según ADF/KPSS (A4, Persona A, 2026-11-18)
- **Qué:** D = 1 en todos los niveles. d = 1 solo si ADF no rechaza raíz unitaria o KPSS rechaza estacionariedad (`choose_d`); en los agregados, solo el tipo b.
- **Por qué:** la ACF de `log1p(y)` no decae en múltiplos de 7 (0,80 en el lag 7) y Δ₇y la limpia. La estacionalidad anual va con Fourier: 2,35 años no alcanzan para s = 365.
- **Alternativa descartada:** SARIMA con s = 365.

### D-15: selección en dos pasos y prueba usada una sola vez (A6-A7, Persona A, 2026-11-18)
- **Qué:** `select` usa solo datos hasta `VAL_END`: 36 candidatos más AutoARIMA, 5 finalistas por AIC y gana el menor RMSE de validación. Los órdenes se versionan en `artifacts/models/arima_selection.json` antes de ejecutar `final`.
- **Alternativa descartada:** elegir solo por AIC.
- **Impacto medido:** SARIMAX supera al ingenuo en RMSE de validación en los 10 niveles.

### D-16: cierres totales como faltantes en `type_a` y `type_d` (aplica P-05, aprobada por Persona A, 2026-11-18)
- **Qué:** si al menos el 5 % de los días tiene apertura parcial, los días con `y = 0` y `open_share = 0` no se usan para ajustar. Todo pronóstico SARIMAX con `open_share = 0` vale 0 en los parquet.
- **Por qué:** `log1p(0)` queda muy lejos de un domingo con pocas tiendas abiertas y `open_share` no puede explicar ese salto.
- **Alternativa descartada:** mantener esos días, como indicaba el flujo original.
- **Impacto medido en validación (SARIMAX):** `type_a` pasa de RMSE 1 186 937 y MAPE 28,8 % a 864 171 y 19,0 %; `type_d`, de 541 852 y 27,1 % a 340 163 y 17,5 %.

### D-17: `sarimax_*.pkl` compactos (aplica P-06, aprobada por Persona A, 2026-11-18)
- **Qué:** el pkl se guarda desde un modelo limpio filtrado con `conserve_memory`, sin las matrices históricas.
- **Alternativas descartadas:** `remove_data=True` (falla al pronosticar) y `low_memory=True` (intervalos NaN).
- **Impacto medido:** de 33-46 MB a un máximo de 0,67 MB por modelo (6,6 MB en total), con pronóstico e intervalos idénticos. El pkl no sirve para `summary()`.
