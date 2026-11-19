# Informe de Persona A: datos, modelos estadísticos y evaluación

Pronóstico de ventas diarias de Rossmann a 42 días, siguiendo CRISP-DM. Las tareas A0 a A3 las
implementó Persona B por acuerdo del equipo (D-05) y Persona A las revisó.

**Resultado principal.** SARIMAX supera al baseline y cumple MAPE < 15 % en prueba en los 10 niveles
(total, 4 tipos de tienda y 5 tiendas). En el total, su MAPE es 6,81 % frente a 17,41 % del baseline.
La GRU, disponible solo para el total, logra 5,49 %, pero su diferencia con SARIMAX no es
estadísticamente significativa.

## 1. Comprensión de los datos

- **Fuente:** Rossmann Store Sales (Kaggle): 1 017 209 filas, 1 115 tiendas y 942 días, del
  01/01/2013 al 31/07/2015.
- **Hueco principal:** 180 tiendas no tienen registros del 01/07/2014 al 31/12/2014.
- **Otros problemas:** faltan `CompetitionDistance` en 3 tiendas y la fecha de apertura del competidor
  en 354. Hay 54 días abiertos con venta 0.
- **Patrones:** el lunes es el día más alto, el domingo casi desaparece salvo en el tipo b y diciembre
  es el mes más fuerte (+17 %). La distribución de ventas tiene cola larga a la derecha.
- **`Customers`:** su correlación con ventas es 0,82, pero no se conoce de antemano. Solo se usa de
  forma descriptiva; nunca como variable del modelo.

## 2. Preparación de los datos

| Decisión | Tratamiento | Motivo |
| --- | --- | --- |
| Hueco de 2014 (D-10) | 33 121 tienda-días rellenados solo con información anterior | Sin relleno, la venta del 2.º semestre de 2014 quedaría 15 % por debajo |
| `StateHoliday` (D-06) | Texto normalizado a {0, a, b, c} | El CSV mezcla número y texto |
| Abierto con venta 0 (D-07) | Se marcan, no se borran | Son pocos y no afectan a las series agregadas |
| Competencia (D-08, D-09) | Mediana y banderas | Evita perder tiendas |
| Domingos (D-01) | Se conservan, con `open_share` | Mantiene un calendario continuo |

**Partición temporal:** entrenamiento hasta el 08/05/2015; validación del 09/05 al 19/06; prueba del
20/06 al 31/07; futuro del 01/08 al 11/09/2015. Cada bloque de evaluación tiene 42 días.

**Variables exógenas (13):** proporción de tiendas con promoción, abiertas, en festivo escolar y con
Promo2; tres indicadores de festivo estatal; y seis términos de Fourier para el ciclo anual.

## 3. Estacionalidad

- **Semanal:** domina la serie, con fuerza STL de 0,65 (tipo b) a 0,83 (tipo c) y 0,80 en el total. La
  autocorrelación en el lag 7 es 0,80 y la diferencia estacional la corrige. Por eso **s = 7 y D = 1**
  (D-14).
- **Tendencia:** ADF y KPSS indican **d = 0** en el total y en los tipos a, c y d, y **d = 1** en el
  tipo b. En las tiendas se aplica la misma regla automáticamente.
- **Anual:** con 2,35 años, el periodograma no puede aislar 365 días. La evidencia viene de comparar
  cada semana con la misma semana del año anterior (r = 0,42). Se modela con **Fourier**, porque
  s = 365 gastaría un año de datos y volvería el ajuste lento e inestable.

## 4. Modelos ARIMA y SARIMAX

Todos se ajustan sobre `log1p(ventas)` y los pronósticos se revierten a euros.

1. **Selección (D-15):** con datos hasta el fin de validación se prueban 36 candidatos por nivel y
   modelo, más AutoARIMA. Pasan los 5 mejores por AIC más AutoARIMA, y gana el menor RMSE de validación.
2. **Congelación:** los órdenes se guardaron antes de generar los pronósticos de prueba.
3. **Modelo final:** se reestima hasta el fin de cada periodo y la prueba se evalúa una sola vez.

Dos mejoras aprobadas:

- **Cierres totales (P-05, D-16):** en `type_a` y `type_d`, los días con todas las tiendas cerradas no
  se usan para ajustar. Tras volver a elegir los órdenes, el RMSE de validación de SARIMAX baja de
  1 186 937 a 864 171 en `type_a` y de 541 852 a 340 163 en `type_d`.
- **Modelos compactos (P-06, D-17):** cada archivo del simulador pasa de 33-46 MB a menos de 0,7 MB,
  con pronósticos e intervalos idénticos.

| Nivel | Orden SARIMAX (p,d,q)(P,D,Q) con s = 7 |
| --- | --- |
| total | (2,0,0)(0,1,1) |
| type_a | (0,0,0)(2,0,1) |
| type_b | (2,1,2)(0,1,1) |
| type_c | (2,0,2)(0,1,1) |
| type_d | (2,0,1)(0,1,1) |
| store_9 | (2,0,0)(1,1,1) |
| store_769 | (2,1,1)(1,1,1) |
| store_504 | (2,0,1)(1,1,1) |
| store_960 | (2,0,0)(0,1,1) |
| store_693 | (2,0,1)(1,1,1) |

## 5. Evaluación

### 5.1 Prueba por nivel (MAPE %)

| Nivel | Baseline | SARIMA | SARIMAX | GRU |
| --- | ---: | ---: | ---: | ---: |
| total | 17,41 | 17,08 | **6,81** | **5,49** |
| type_a | 18,72 | 22,29 | **13,74** | — |
| type_b | 9,79 | 8,51 | **3,55** | — |
| type_c | 19,56 | 33,43 | **9,29** | — |
| type_d | 20,74 | 19,87 | **11,21** | — |
| store_9 | 22,17 | 36,11 | **10,02** | — |
| store_769 | 10,09 | 11,12 | **7,74** | — |
| store_504 | 15,68 | 25,28 | **10,60** | — |
| store_960 | 28,76 | 29,79 | **8,51** | — |
| store_693 | 58,31 | 30,65 | **10,48** | — |

En RMSE, SARIMAX supera al baseline en los 10 niveles, con mejoras de 18,5 % a 67,8 %. SARIMA solo lo
supera en 4 de 10, lo que muestra el valor de incluir promoción y calendario.

### 5.2 Total: comparación de modelos en prueba

| Modelo | RMSE | MAE | MAPE % |
| --- | ---: | ---: | ---: |
| Baseline | 1 775 080 | 1 179 019 | 17,41 |
| SARIMA | 1 734 123 | 1 228 501 | 17,08 |
| SARIMAX | 571 130 | 409 097 | 6,81 |
| GRU | 580 806 | 354 428 | 5,49 |

**Prueba de Wilcoxon sobre errores absolutos diarios (n = 42):**

- SARIMAX frente al baseline: p = 0,0029. SARIMAX es significativamente mejor.
- SARIMAX frente a GRU: p = 0,29. **No hay diferencia significativa.** La GRU gana en MAE y MAPE;
  SARIMAX, por poco, en RMSE.

### 5.3 Error por tipo de día (total, MAPE %)

| Tipo de día | Días | Baseline | SARIMA | SARIMAX | GRU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Con promoción | 15 | 5,70 | 27,23 | 6,74 | 5,16 |
| Sin promoción | 21 | 29,39 | 8,79 | 5,51 | 4,84 |
| Inicio de mes | 5 | 3,13 | 22,04 | 7,61 | 4,24 |
| Domingo | 6 | 4,78 | 20,71 | 11,52 | 8,61 |

El baseline repite la última semana conocida, que tuvo promoción de lunes a viernes. Por eso acierta en
las semanas con promoción y falla mucho en las demás. Sus buenos resultados en estos subconjuntos no
compensan su error global.

**Festivos.** La prueba no contiene festivos estatales, así que se revisaron los 3 de validación:

| Fecha | Tiendas abiertas | Venta real | SARIMAX | GRU | Baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14/05/2015 | 2,9 % | 289 247 | 238 383 | 320 691 | 8 550 616 |
| 25/05/2015 | 3,0 % | 299 473 | 282 339 | 270 883 | 10 995 906 |
| 04/06/2015 | 38,7 % | 3 534 231 | 1 516 547 | 2 140 396 | 8 550 616 |

Los festivos casi totales se pronostican bien. El festivo regional del 04/06, con cierre parcial, sigue
siendo difícil: SARIMAX se queda 57 % por debajo y la GRU 39 %.

### 5.4 Limitaciones

- **La prueba es algo favorable:** no tiene festivos. En validación, que sí los tiene, `type_a` (19,0 %)
  y `type_d` (17,5 %) **no cumplen** la meta del 15 %. El total da 8,9 %.
- **Exógenas conocidas:** validación y prueba usan el calendario real de promociones, aperturas y
  festivos. Si el plan cambia, el error será mayor.
- **Validación con origen móvil:** en 4 ventanas, el MAPE medio del total es 12,3 % (6,8 % a 21,5 %).
  `type_a` (50,1 %) y `type_d` (61,9 %) salen muy altos, pero esta prueba usa AutoARIMA sin el
  tratamiento de cierres totales, así que no mide el modelo final.
- **Residuos:** Ljung-Box rechaza ausencia de autocorrelación en algunos lags de todos los niveles. Los
  p-valores y los intervalos deben leerse con cautela.
- **Intervalos:** la cobertura del IC 95 % en prueba va de 0,95 a 1,00, correcta o algo conservadora.
- **Sin conciliación:** en el futuro, la suma de los tipos (261,3 M€) es 2,6 % menor que el total
  (268,4 M€).
- **Escenario futuro:** usa festivos escolares en 0 y promoción según el patrón histórico. Las propuestas
  P-04 y P-07 siguen pendientes de decisión del equipo.
- **Verificación:** las tareas A1 y A2 se revisaron con `history.parquet` y sus pruebas, no contra los
  CSV originales, que no estaban disponibles al revisar.

## 6. Conclusiones para marketing

Los efectos son asociaciones ajustadas por calendario, no efectos causales probados, y **no incluyen
costos ni margen**. Indican dónde sube la venta, no dónde la promoción es más rentable.

| Nivel | Venta asociada a la promoción | IC 95 % |
| --- | ---: | ---: |
| Total | +38,9 % | 34,9 a 43,0 |
| Tipo c | +54,2 % | 35,1 a 76,0 |
| Tipo d | +44,6 % | 37,4 a 52,1 |
| Tipo a | +36,0 % | 27,6 a 44,8 |
| Tipo b | +18,5 % | 16,6 a 20,5 |

1. **La promoción es la palanca más clara:** se asocia con cerca de 39 % más venta en el total.
2. **El tipo b responde bastante menos.** Es la candidata natural para revisar si necesita la misma
   intensidad. Entre a, c y d los intervalos se solapan, así que no conviene ordenarlos.
3. **Volumen y respuesta son cosas distintas:** en la proyección, el tipo a concentra 54 % de la venta
   de los tipos y el d 30 %. Allí un mismo porcentaje mueve más euros.
4. **Conviene decidir por tienda:** la respuesta va de +10,2 % (`store_769`) a +97,2 % (`store_693`).
5. **Planear con el calendario:** el lunes es el día más fuerte, diciembre el mejor mes y hay más venta
   antes y después de los festivos.
6. **Siguiente paso:** probar la promoción con grupos comparables de tiendas e incluir el margen, para
   medir su efecto real y su rentabilidad.

## 7. Reproducción

```bash
python -m src.models.arima select   # elige órdenes, sin usar la prueba
python -m src.models.arima cv       # validación con origen móvil
python -m src.models.arima final    # genera pronósticos, métricas y modelos
python -m src.evaluation.compare    # tabla final y Wilcoxon
```

No se debe volver a ejecutar `select` ni cambiar modelos a partir de los resultados de prueba.

## 8. Texto para Persona B (README y página Inicio)

> **Metodología.** Seguimos CRISP-DM con datos de Rossmann, de enero de 2013 a julio de 2015. Limpiamos
> el hueco de 180 tiendas en 2014 sin usar información futura y separamos los datos por tiempo:
> entrenamiento, validación y prueba, estos dos últimos de 42 días cada uno.
>
> Comparamos un baseline que repite la última semana, SARIMA, SARIMAX y una red GRU. SARIMAX usa la
> estacionalidad semanal, términos anuales de Fourier y variables conocidas de antemano: promociones,
> aperturas y festivos. Nunca usa el número de clientes, porque no se conoce al pronosticar. Los órdenes
> se eligieron solo con validación y la prueba se usó una vez.
>
> **Resultados.** SARIMAX supera al baseline y logra un MAPE menor a 15 % en los 10 niveles evaluados. En
> la venta total, el MAPE es 6,8 % frente a 17,4 % del baseline. La GRU logra 5,5 % en el total, pero su
> diferencia con SARIMAX no es estadísticamente significativa (Wilcoxon, p = 0,29). El periodo de prueba
> no incluye festivos, y en validación los tipos a y d quedaron en 19,0 % y 17,5 %.
>
> **Cómo leer el simulador.** La promoción se asocia con cerca de 39 % más venta en el total, y el efecto
> varía por tipo y tienda. Es una estimación histórica: no prueba causalidad ni considera costos o margen.
> Los datos terminan el 31/07/2015.
