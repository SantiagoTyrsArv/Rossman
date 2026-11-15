# Rossmann: predicción de demanda con SARIMAX y GRU

Proyecto final de Inteligencia de Minería de Datos. Pronostica las ventas diarias de Rossmann
(Kaggle *Rossmann Store Sales*) para los próximos 42 días a nivel total, por tipo de tienda y para
5 tiendas seleccionadas, e incluye una app web con dashboard, simulación de escenarios de promoción
y visualización temporal. Metodología: CRISP-DM.

> **Estado:** en desarrollo. La app funciona con datos de prueba hasta que lleguen los artifacts reales.

- **App pública:** _pendiente del primer despliegue en Streamlit Community Cloud_

## Estructura

| Ruta | Contenido | Dueño |
| --- | --- | --- |
| `src/data`, `src/features` | carga, limpieza y variables exógenas | A |
| `src/models/baseline.py`, `arima.py` | baseline ingenuo estacional, SARIMA/SARIMAX | A |
| `src/models/gru.py`, `notebooks/05_gru.ipynb` | modelo GRU | B |
| `src/evaluation` | métricas (RMSE, MAE, MAPE) y backtest | A |
| `artifacts/` | pronósticos, métricas y modelos que lee la app | A y B |
| `app/` | aplicación Streamlit | B |
| `docs/` | decisiones, propuestas y contrato de datos | compartido |

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # entorno de entrenamiento
python -m src.data.make_dataset                      # descarga y procesa los datos
# notebooks 01–05 → generan artifacts/
pip install -r app/requirements.txt
streamlit run app/Home.py
```

Sin artifacts reales la app genera datos de prueba con el mismo esquema (`app/mock_data.py`).

## Calidad

```bash
ruff check . && black --check . && pytest -q -m "not slow"
```

## Métricas finales

_Se completa con la evaluación final (tarea A8)._
