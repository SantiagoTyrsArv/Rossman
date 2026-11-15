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
