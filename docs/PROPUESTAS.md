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
- **Decisión del equipo:** pendiente (A debe crear `src/config.py` con el contenido exacto de 5.6).

### P-02: dependencia de `src/evaluation/metrics.py` y `backtest.py` (2026-11-15, detectada por Persona B)
- **Situación actual:** `compute_all` y `write_forecast` de A no existen aún.
- **Propuesta:** `src/models/gru.py` importa `compute_all` de A si existe; si no, usa un stub local marcado `# TODO quitar` con la misma fórmula (MAPE solo con y > 0). Las métricas definitivas de `gru.json` se regeneran con la función de A antes de la evaluación final.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** pendiente.

### P-03: dependencia de `history.parquet`, `sarimax_*.pkl` y `build_future_exog` (2026-11-15, detectada por Persona B)
- **Situación actual:** no existen todavía.
- **Propuesta:** la app usa `app/mock_data.py` (mismo esquema del contrato) y lo indica con el aviso "DATOS DE PRUEBA". El simulador cambia solo a SARIMAX real cuando existen el pkl, el meta y `build_future_exog`.
- **Afecta a:** solo mi área.
- **Decisión del equipo:** pendiente.
