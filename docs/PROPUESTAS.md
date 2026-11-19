# Propuestas y dependencias pendientes

> Archivo compartido: **solo agregar** entradas.
> Plantilla:
> ```markdown
> ### P-<nn>: <título> (<fecha>, detectada por Persona X)
> - Situación actual:
> - Propuesta:
> - Comparación: <tabla resumida>
> - Recomendación de Claude:
> - Afecta a: solo mi área | área del otro | contrato
> - Decisión del equipo: pendiente
> ```

### P-01: dependencia de `src/config.py` (2026-11-15, detectada por Persona B)
- **Situación actual:** `src/config.py` (compartido, lo crea A en la tarea A0) todavía no existe.
- **Propuesta:** mientras tanto, `app/utils/settings.py` y `src/models/gru.py` intentan `from src.config import ...` y, si falla, usan una copia local de las constantes de la sección 5.6 del contrato. Cuando exista `src/config.py` se usa automáticamente; el respaldo local se borra después.
- **Afecta a:** solo mi área (no se crea nada en `src/config.py`).
- **Decisión del equipo:** resuelta (2026-11-15): `src/config.py` creado con el contenido de 5.6 (A0, D-05). El respaldo local se puede quitar cuando A apruebe el PR.

### P-02: dependencia de `src/evaluation/metrics.py` y `backtest.py` (2026-11-15, detectada por Persona B)
- **Situación actual:** `compute_all` y `write_forecast` de A no existen aún.
- **Propuesta:** `src/models/gru.py` importa `compute_all` de A si existe; si no, usa un stub local marcado `# TODO quitar` con la misma fórmula (MAPE solo con y > 0). Las métricas definitivas de `gru.json` se regeneran con la función de A antes de la evaluación final.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** pendiente.

### P-03: dependencia de `history.parquet`, `sarimax_*.pkl` y `build_future_exog` (2026-11-15, detectada por Persona B)
- **Situación actual:** no existen todavía.
- **Propuesta:** la app usa `app/mock_data.py` (mismo esquema del contrato) y lo indica con el aviso "DATOS DE PRUEBA". El simulador cambia solo a SARIMAX real cuando existen el pkl, el meta y `build_future_exog`.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** parcialmente resuelta (2026-11-15): ya existen `history.parquet` y `build_future_exog` (A3). Faltan `sarimax_*.pkl` y su meta (A7).

### P-04: calendario de promos de `test.csv` como escenario base del futuro (2026-11-15, detectada por Persona B)
- **Situación actual:** `build_future_exog` usa por defecto el patrón histórico (alternancia quincenal desde la última semana con promo, 27/07/2015 → promo la semana del 10/08). El calendario real planificado en `test.csv` difiere en 34 de 48 días abiertos: tiene promo las semanas del 03/08, 17/08, 31/08 y 14/09 (rompe la alternancia, como ya pasó el 27/04 y el 04/05).
- **Propuesta:** usar el calendario de `test.csv` como `promo_days` por defecto del periodo futuro (o como escenario predefinido "Calendario planificado" en la app). Es información conocida de antemano (promos planificadas), no fuga.
- **Comparación:**

| Criterio | Actual | Propuesta |
| --- | --- | --- |
| Requerimientos | Cumple | Cumple; escenario base más realista |
| Gratis / despliegue | Sí | Sí (exportar un CSV pequeño a `artifacts/`) |
| Cronograma | 0 h | ~1 h |
| Sin fuga / partición temporal | Sí | Sí: el calendario de promos es un plan conocido; no usa Sales ni Customers futuros |
| Cambia el contrato | No | Sí: comportamiento por defecto de `build_future_exog` (5.5) |
| Afecta al otro | — | Sí (función de A, la usan el simulador y la GRU) |
| Evidencia | — | 34/48 días difieren del calendario real |
| Explicable en el curso | Sí | Sí (exógenas conocidas de antemano) |
| Riesgo | Ninguno | Bajo |

- **Recomendación de Claude:** aplicar como **escenario predefinido** en la app ("Calendario planificado") sin cambiar el valor por defecto de `build_future_exog`; así no se rompe el contrato.
- **Afecta a:** contrato y área de A (si se cambia el valor por defecto); solo área de B (si es un escenario de la app).
- **Decisión del equipo:** pendiente.

### P-05: no entrenar con los días de cierre total en `type_a` y `type_d` (2026-11-18, detectada por Persona A)
- **Situación actual:** en esos tipos, un festivo con todas las tiendas cerradas vale `log1p(0) = 0`, mientras que un domingo con pocas tiendas abiertas vale entre 9 y 10. `open_share` es lineal y no puede explicar ese salto (0 frente a ~0,01), así que esos días disparan la varianza del modelo (`sigma2` de 1,6 frente a 0,13 en `type_a`).
- **Propuesta:** en los niveles con días de apertura parcial (`0 < open_share < 0,5`) y días con `y = 0`, tratar esos ceros como faltantes al ajustar (el filtro de Kalman los admite) y pronosticar 0 cuando `open_share = 0`. Afecta solo a `type_a` y `type_d`.
- **Comparación (solo validación; la prueba no se ha usado):**

| Criterio | Actual | Propuesta |
| --- | --- | --- |
| Requerimientos | Cumple | Cumple |
| Gratis / despliegue | Sí | Sí, sin cambios |
| Cronograma | 0 h | ~1 h |
| Sin fuga / partición temporal | Sí | Sí: solo cambia cómo se usa el entrenamiento; `open_share` ya se conoce de antemano |
| Cambia el contrato | No | No |
| Afecta al otro | No | No |
| RMSE val SARIMAX `type_a` | 1 186 937 (MAPE 28,8 %) | 910 692 (MAPE 19,8 %), −23,3 % |
| RMSE val SARIMAX `type_d` | 541 852 (MAPE 27,1 %) | 340 163 (MAPE 17,5 %), −37,2 % |
| RMSE val SARIMA `type_a` / `type_d` | 1 877 945 / 1 076 608 | 1 262 880 / 771 206 (−32,8 % / −28,4 %) |
| Explicable en el curso | Sí | Sí (tratamiento de valores atípicos estructurales) |
| Riesgo | Modelo mal calibrado en dos niveles | Bajo; en la app, un día de cierre total mostraría una venta pequeña, no 0 exacto |

- **Recomendación de Claude:** aplicar ahora, antes de generar los pronósticos de prueba.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** pendiente.

### P-06: guardar los `sarimax_*.pkl` sin las matrices históricas del filtro (2026-11-18, detectada por Persona A)
- **Situación actual:** con `results.save` normal cada pkl pesa entre 33 y 46 MB, por encima del límite de 25 MB de la tarea A7 (unos 400 MB para los 10 niveles). `remove_data=True` falla al pronosticar y `low_memory=True` deja los intervalos en NaN.
- **Propuesta:** reconstruir el resultado final con `model.filter(params, cov_type="none", conserve_memory=...)`, conservando solo lo necesario para pronosticar.
- **Comparación:**

| Criterio | Actual | Propuesta |
| --- | --- | --- |
| Tamaño por modelo | 33-46 MB | ~0,65 MB |
| Pronóstico e intervalo tras cargar | Referencia | Idénticos (diferencia 0 en 3 niveles probados) |
| Gratis / despliegue | Riesgo por peso | Sí |
| Cronograma | — | ~0,5 h |
| Sin fuga / partición temporal | Sí | Sí |
| Cambia el contrato | No | No: sigue siendo `results.save` y `load_pickle` |
| Afecta al otro | No | No: `get_forecast` y `conf_int` funcionan igual |
| Riesgo | Repositorio pesado | El pkl ya no sirve para `summary()`; los coeficientes se reportan en el notebook 04 |

- **Recomendación de Claude:** aplicar ahora.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** pendiente.

### P-07: festivos escolares por defecto en `build_future_exog` (2026-11-18, detectada por Persona A)
- **Situación actual:** por defecto `school_holiday_share = 0` en todo el futuro, pero entre el 01/08 y el 11/09/2014 fue 0,47 de media. Con el modelo de `total`, usar 0 en validación sube el RMSE de 738 378 a 780 637 (+5,7 %), aun cuando allí la media real era solo 0,07.
- **Propuesta:** usar por defecto la proporción del mismo periodo del año anterior.
- **Comparación:** cumple los requerimientos, no usa datos futuros y toma ~1 h, pero cambia el comportamiento por defecto de la sección 5.5 y, con él, el escenario base del simulador y las entradas de la GRU.
- **Recomendación de Claude:** aplicar cuando A y B lo acuerden; mientras tanto se mantiene lo que dice la guía.
- **Afecta a:** contrato y área del otro.
- **Decisión del equipo:** pendiente.

### P-08: fijar versiones en `app/requirements.txt` (2026-11-18, detectada por Persona A)
- **Situación actual:** la app instala `statsmodels>=0.14.2`, `pandas>=2.2` y `numpy>=1.26`. Los pkl se generan con statsmodels 0.15.0, pandas 2.3.3 y numpy 2.4.6, y un pickle puede no cargar con otras versiones.
- **Propuesta:** fijar esas tres versiones en `app/requirements.txt`.
- **Comparación:** no cambia métricas ni contrato, toma minutos y evita que el simulador falle en el despliegue.
- **Recomendación de Claude:** aplicar antes de desplegar; lo decide Persona B porque el archivo es suyo.
- **Afecta a:** área del otro.
- **Decisión del equipo:** pendiente.

### P-09: decodificador por día en la GRU (2026-11-18, detectada por Persona B)
- **Situación actual:** la GRU v1 (`build_model`) aplana las exógenas futuras (42 × 15 = 630 entradas) y las pasa por una sola `Dense`. Así tiene que aprender por separado, para cada una de las 42 posiciones del horizonte, el efecto de festivo, domingo o promo. Con 761 ventanas no le alcanza: hay **underfitting** (MAPE dentro del entrenamiento 55,9 %, pérdida de train ≈ val) y en validación da un MAPE de 62–96 %. En los festivos (14/05 y 25/05) predice 2,3–2,9 M€ cuando la venta real es ~0,3 M€.
- **Propuesta:** el codificador GRU se mantiene igual. Su estado se repite en los 42 días (`RepeatVector`) y se concatena con las exógenas de cada día. Después pasa por un decodificador `GRU(units, return_sequences=True)` y `TimeDistributed(Dense)`, que aplica los mismos pesos a cada día del horizonte. No cambian las entradas, las ventanas, el escalador ni las salidas.
- **Comparación (solo validación, `total`; la prueba no se usó para decidir):**

| Criterio | Actual | Propuesta |
| --- | --- | --- |
| Requerimientos (GRU, RMSE/MAE/MAPE) | Cumple | Cumple |
| Gratis / despliegue | Sí (la app no carga la GRU) | Sí, sin cambios |
| Cronograma | — | ~1 h |
| Sin fuga / partición temporal | Sí | Sí: mismas ventanas y escalador ajustado solo con train |
| Cambia el contrato | No | No: `gru.parquet` y `gru.json` tienen el mismo esquema |
| Afecta al otro | No | No |
| MAPE / RMSE val | 62,7 % / 2 389 656 (mejor variante) | 7,5 % / 666 867 (64 unidades); 8,8 % / 750 320 (32 unidades) |
| MAPE en entrenamiento | 55,9 % | 7,9 % |
| Referencia SARIMAX val `total` | 8,9 % / 738 378 | — |
| Explicable en el curso | Sí | Sí: arquitectura codificador–decodificador (seq2seq) |
| Riesgo | Modelo inservible para la comparación final | Bajo; tests de forma y de contrato sin cambios |

- **Recomendación de Claude:** aplicar ahora, antes de generar los pronósticos de prueba. Se mantiene la búsqueda de variantes (unidades, lookback, dropout) eligiendo por validación.
- **Afecta a:** solo mi área (`src/models/gru.py`, `notebooks/05_gru.ipynb`).
- **Decisión del equipo:** aprobada por Persona B el 2026-11-18 (área propia). Aplicada como D-13.
