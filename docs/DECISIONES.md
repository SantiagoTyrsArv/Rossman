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
