# Guía de estudio: preguntas probables del profesor

Respuestas cortas con las cifras reales del proyecto. Para el detalle, ver
`docs/INFORME_PERSONA_A.md`, `docs/DECISIONES.md` y `notebooks/`.

---

## 0. Cifras que hay que saber de memoria

| Qué | Valor |
| --- | --- |
| Datos | 1 017 209 filas, 1 115 tiendas, 942 días (01/01/2013–31/07/2015) |
| Hueco | 180 tiendas sin datos del 01/07 al 31/12/2014 → 33 121 tienda-días rellenados |
| Partición | train hasta 08/05/2015 · val 09/05–19/06 · test 20/06–31/07 · futuro 01/08–11/09 |
| Horizonte | 42 días (6 semanas) |
| Niveles | 10: total, 4 tipos (a–d), 5 tiendas (9, 769, 504, 960, 693) |
| Exógenas SARIMAX | 13: promo, apertura, festivo escolar, Promo2, 3 festivos estatales, 6 Fourier |
| MAPE test total | baseline 17,4 % · SARIMA 17,1 % · **SARIMAX 6,8 %** · **GRU 5,5 %** |
| RMSE test total | baseline 1 775 080 · SARIMAX **571 130** · GRU 580 806 |
| Wilcoxon | SARIMAX vs baseline p = 0,003 (sí) · SARIMAX vs GRU p = 0,29 (no) |
| Meta MAPE < 15 % | SARIMAX la cumple en los 10 niveles de prueba |
| Efecto promo | total +38,9 % · a +36,0 % · b +18,5 % · c +54,2 % · d +44,6 % |
| SARIMAX total | (2,0,0)(0,1,1)₇ |
| GRU | 56 → 42 días, 48 961 parámetros, val 7,6 % (antes 62,7 %) |

---

## 1. Negocio y objetivo

**¿Qué problema resuelve el proyecto?**
Pronostica las ventas diarias de las 6 semanas siguientes y estima cuánto cambian con promociones, para
que marketing decida dónde y cuándo invertir.

**¿Cuál es el objetivo SMART y se cumplió?**
En 4 semanas, pronosticar 42 días a nivel total, por tipo y para 5 tiendas, superando al baseline con
MAPE < 15 % en prueba, y publicarlo en una app con escenarios. Sí: SARIMAX lo cumple en los 10 niveles
y la app está publicada.

**¿Por qué 42 días?**
Son 6 semanas completas: cubren un ciclo de planificación de campañas y el mismo número de cada día de
la semana, lo que hace comparables los periodos.

**¿Por qué esas 5 tiendas?**
Una por tipo (9 = a, 769 = b, 504 = c, 960 = d), con historia completa y venta cercana a la mediana de
su tipo, más la 693, la de mayor respuesta a la promoción (~+100 %) entre las tiendas grandes (D-11).

---

## 2. Comprensión de los datos

**¿Qué problemas de calidad encontraron? (ISO/IEC 25012)**

| Dimensión | Hallazgo |
| --- | --- |
| Completitud | 180 tiendas sin datos en el 2.º semestre de 2014; nulos en `CompetitionDistance` (3 tiendas) y en la fecha del competidor (354) |
| Consistencia | `StateHoliday` mezcla `0` numérico y `'0'` texto |
| Exactitud | 54 días con la tienda abierta y venta 0 |
| Credibilidad | Sin valores imposibles; los cierres explican los ceros |
| Actualidad | Los datos terminan en 2015: el «futuro» es agosto–septiembre de 2015 |

**¿Qué patrones vieron en el EDA?**
El lunes es el día más alto, el domingo casi no hay ventas salvo en el tipo b, diciembre es el mes más
fuerte (+17 %) y la distribución tiene cola larga a la derecha (por eso `log1p`).

**¿Por qué no usaron `Customers` si tiene correlación 0,82 con las ventas?**
Porque al pronosticar no se conoce cuántos clientes vendrán: usarlo sería **fuga de datos**. Solo se
usa de forma descriptiva.

---

## 3. Preparación

**¿Cómo trataron el hueco de 2014?**
Se crearon los tienda-días faltantes (marcados con `imputed = 1`). La venta es el promedio de la propia
tienda en los 56 días **anteriores** con el mismo día de la semana y estado de promo. Sin relleno, el
total caía 15 % y el modelo lo habría leído como un cambio real (D-10).

**¿Por qué no interpolaron?**
La interpolación usa datos posteriores al hueco: es información del futuro.

**¿Qué hicieron con los domingos y festivos?**
Se conservan en la serie (calendario continuo, necesario para s = 7) y se explica con `open_share`, la
proporción de tiendas abiertas, que se conoce de antemano (D-01).

**¿Cómo partieron los datos y por qué así?**
Por tiempo, nunca al azar: en series de tiempo, una partición aleatoria mete el futuro en el
entrenamiento. Validación para decidir y prueba usada **una sola vez** al final.

**¿Cuáles son las variables exógenas y por qué esas?**
Proporción de tiendas en promo, abiertas, en festivo escolar y con Promo2; tres indicadores de festivo
estatal (a, b, c) y 6 términos de Fourier (3 pares seno/coseno con periodo 365,25). Todas se conocen
antes de la fecha a pronosticar.

**¿Qué diferencia hay entre Promo y Promo2?**
Promo es la promoción diaria de la tienda. Promo2 es una promoción continua en la que participan
algunas tiendas, que se reactiva en meses fijos (`PromoInterval`).

---

## 4. Estacionalidad y estacionariedad

**¿Cómo eligieron s, d y D?**
- **s = 7:** ACF de 0,80 en el lag 7 y fuerza estacional STL de 0,80 en el total.
- **D = 1:** la diferencia estacional (Δ₇) elimina esa autocorrelación.
- **d:** con ADF y KPSS. d = 0 en el total y en los tipos a, c y d; d = 1 en el tipo b (D-14).

**¿Qué hipótesis contrastan ADF y KPSS?**
- **ADF:** H0 = hay raíz unitaria (no estacionaria). Rechazar → estacionaria.
- **KPSS:** H0 = es estacionaria. Rechazar → no estacionaria.
Se usan juntas porque tienen hipótesis opuestas. Regla: d = 1 si ADF no rechaza **o** KPSS rechaza.

**¿Para qué sirve la descomposición STL?**
Separa la serie en tendencia, estacionalidad y residuo de forma robusta a atípicos. Sirve para medir
la fuerza de cada componente.

**¿Por qué la estacionalidad anual va con Fourier y no con s = 365?**
Con 2,35 años de datos, s = 365 consume un año entero en la diferencia estacional, el modelo se vuelve
lento e inestable y el periodograma no alcanza a aislar ese periodo. Fourier representa el ciclo anual
con solo 6 variables.

**¿Qué muestran la ACF y la PACF?**
La ACF tiene picos en 7, 14, 21… (estacionalidad semanal). Tras Δ₇, los picos restantes orientan los
órdenes AR/MA, que luego se afinaron con la búsqueda por AIC.

---

## 5. ARIMA, SARIMA y SARIMAX

**¿Qué significa SARIMAX(2,0,0)(0,1,1)₇?**
AR de orden 2 (depende de los 2 días anteriores), sin diferencia simple, diferencia estacional de
lag 7 y un término MA estacional; la «X» indica que incluye exógenas.

**¿Cuál es la diferencia entre SARIMA y SARIMAX?**
SARIMAX agrega exógenas. Por eso SARIMA falla tanto (17,1 % en el total): no sabe cuándo hay promoción
ni cuándo cierran las tiendas.

**¿Por qué modelaron `log1p(ventas)`?**
Estabiliza la varianza, convierte efectos multiplicativos (la promo sube un %) en aditivos y evita
pronósticos negativos. `log1p` en lugar de `log` admite ceros.

**¿Cómo se interpreta el coeficiente de promo?**
En log, `exp(β) − 1` es el cambio porcentual de pasar de 0 a 100 % de tiendas en promo: +38,9 % en el
total (IC 95 % de 34,9 a 43,0).

**¿Cómo eligieron los órdenes?**
Con datos hasta el fin de validación: 36 candidatos (p, q ∈ {0,1,2}; P, Q ∈ {0,1}) más AutoARIMA. Pasan
los 5 mejores por AIC y gana el menor RMSE de validación. Los órdenes se congelaron antes de mirar la
prueba (D-15).

**¿Por qué no elegir solo por AIC?**
El AIC mide el ajuste dentro de la muestra a un paso. Lo que importa es pronosticar 42 días fuera de
la muestra, y eso lo mide el RMSE de validación.

**¿Qué es el test de Ljung-Box y qué les dio?**
Contrasta H0 = los residuos no tienen autocorrelación (son ruido blanco). En algunos lags se rechaza en
todos los niveles: queda estructura sin capturar, así que los p-valores y los intervalos deben leerse
con cautela.

**¿Qué hicieron con los días de cierre total en los tipos a y d?**
Allí `log1p(0) = 0` queda muy lejos de un domingo con pocas tiendas abiertas, y `open_share` no puede
explicar ese salto. Se tratan como faltantes al ajustar (el filtro de Kalman lo permite) y se pronostica
0 cuando `open_share = 0`. En validación, el RMSE de `type_a` bajó de 1 186 937 a 864 171 (D-16).

**¿Por qué el modelo de `type_a` no tiene diferencia estacional?**
Lo eligió AutoARIMA, (0,0,0)(2,0,1)₇, porque tuvo menor RMSE de validación que los candidatos de la
rejilla. La estacionalidad la captan los términos AR/MA estacionales.

---

## 6. GRU

**¿Cómo es la arquitectura?**
Codificador–decodificador: el codificador (GRU 64 → Dropout → GRU 32) lee los 56 días pasados (16
variables). Su estado se repite en los 42 días futuros, se une a las 15 exógenas de cada día y un
decodificador GRU + `TimeDistributed(Dense)` predice cada día. 48 961 parámetros, pérdida Huber, Adam.

**¿Qué pasó con la primera versión?**
Aplanaba las exógenas futuras (42 × 15 = 630 entradas) en una sola capa densa y no aprendía:
**underfitting**, con MAPE de 55,9 % en entrenamiento y 62,7 % en validación. Fallaba en los festivos
(predecía 2,9 M€ cuando la venta real era 0,3 M€). El decodificador comparte pesos en los 42 días, así
que aprende una sola vez qué significa festivo, domingo o promo: validación 7,6 % (P-09, D-13).

**¿Cómo saben que no hay sobreajuste?**
Las curvas de pérdida de entrenamiento y validación bajan juntas y no se separan. Además,
`EarlyStopping` restaura la mejor época (la 40).

**¿Cómo evitaron fuga de datos en la GRU?**
El `MinMaxScaler` se ajustó solo con datos de entrenamiento y las ventanas de entrenamiento no tienen
objetivo posterior al corte. No usa `Customers`.

**¿Qué variantes probaron?**
32, 64 y 128 unidades, ventana de 28 o 56 días y dropout 0,2 o 0,3. Se eligió por MAPE de validación:
64 unidades, 56 días, dropout 0,2.

**¿Por qué la GRU solo para el total?**
La guía pedía primero una versión para el total y pasar a un modelo global (con los tipos y tiendas)
solo si rendía menos que SARIMAX. No fue necesario: ya le ganaba en validación.

**¿Por qué la GRU no está en el simulador?**
Requeriría TensorFlow en la app (demasiado pesado para Streamlit Cloud) y no tiene coeficientes
interpretables. SARIMAX es liviano y su efecto de promo se explica con un coeficiente.

**¿Qué son los intervalos de la GRU?**
Se obtienen con MC Dropout (dropout activo al predecir, 100 repeticiones). Salen muy estrechos porque
el dropout solo está en el codificador: no son un 95 % calibrado.

---

## 7. Evaluación

**¿Qué mide cada métrica?**
- **RMSE:** raíz del error cuadrático medio; penaliza más los errores grandes. En euros.
- **MAE:** error absoluto medio; más robusto a atípicos. En euros.
- **MAPE:** error porcentual medio; sin escala, sirve para comparar niveles distintos.

**¿Por qué el MAPE excluye los días con venta 0?**
Dividir por 0 es infinito: los días cerrados harían explotar el MAPE. RMSE y MAE sí incluyen esos días.

**¿Qué es el baseline y por qué compararse con él?**
El ingenuo estacional: repite la venta del mismo día de la semana anterior. Si un modelo no le gana,
no aporta valor.

**¿Por qué Wilcoxon y no una prueba t?**
Compara los errores absolutos diarios de dos modelos **pareados por día** (n = 42) sin suponer
normalidad. Los errores tienen cola larga, así que la prueba t no es adecuada.

**¿Qué concluye Wilcoxon?**
SARIMAX es significativamente mejor que el baseline (p = 0,003). SARIMAX y la GRU no se distinguen
(p = 0,29): la GRU gana en MAE y MAPE, SARIMAX por poco en RMSE.

**¿Qué es la validación con origen móvil y qué dio?**
Se repite el pronóstico de 42 días desde 4 puntos de corte distintos para ver si el error es estable.
Total: MAPE medio 12,3 % (entre 6,8 % y 21,5 %). Usa AutoARIMA sin el tratamiento de cierres, así que
no mide exactamente el modelo final, pero muestra que el error depende del periodo.

**¿Cómo se comportan los modelos en festivos?**
La prueba no tiene festivos. En validación, los festivos casi totales (14/05 y 25/05) se pronostican
bien, pero el regional del 04/06 (39 % de tiendas abiertas) es difícil: SARIMAX se queda 57 % abajo.

**¿Por qué el baseline acierta en semanas con promo?**
Repite la última semana conocida, que tuvo promo. Acierta cuando la semana se repite (5,7 % con
promo) y falla cuando no (29,4 % sin promo).

**¿Están bien calibrados los intervalos de SARIMAX?**
En prueba cubren entre 95 % y 100 % de los valores reales: correctos o algo conservadores.

---

## 8. App y despliegue

**¿Cómo está construida la app?**
Streamlit con 4 páginas (Inicio, Dashboard, Escenarios, Series), desplegada en Streamlit Community
Cloud desde `main`. Los modelos se entrenan fuera de la app y ella solo lee archivos precalculados
(`artifacts/`). La única inferencia en vivo es el SARIMAX del simulador.

**¿Cómo funciona el simulador?**
Carga el SARIMAX ya ajustado del nivel elegido, arma las exógenas futuras con el escenario (días de
promo, festivos, Promo2) y pronostica base y escenario. La diferencia es la venta incremental.

**¿Por qué el escenario «Personalizado» puede vender menos que la base?**
Los días elegidos **reemplazan** el calendario histórico (unos 15 días de promo en 42). Si eliges menos
días, hay menos promo que en la base.

**¿Cómo se coordinaron dos personas sin pisarse?**
Un contrato de datos (`docs/CONTRATO_DATOS.md`) con el formato exacto de cada archivo, un archivo por
modelo, ramas por tarea, PR a `dev`, CI con tests y `main` protegida. Las mejoras se proponen y
documentan (`PROPUESTAS.md`, `DECISIONES.md`).

**¿Qué problema técnico hubo en el despliegue?**
Los modelos se guardan con pickle, que depende de las versiones de las librerías. Se fijaron pandas,
numpy y statsmodels en `app/requirements.txt` (D-12) y se compactaron los modelos de ~40 MB a 0,7 MB
(D-17).

---

## 9. Conclusiones para marketing

- La promoción es la palanca más clara: +39 % en el total.
- El tipo b responde menos (+18,5 %); entre a, c y d los intervalos se solapan.
- Volumen y respuesta son distintos: el tipo a concentra 54 % de la venta de los tipos, así que el
  mismo % mueve más euros.
- Conviene decidir por tienda: de +10 % (tienda 769) a +97 % (tienda 693).
- Planear con el calendario: lunes, diciembre y los días alrededor de festivos.

---

## 10. Limitaciones y trabajo futuro

- Los efectos de promo son **asociaciones**, no causales, y no incluyen costos ni margen.
- La prueba no tiene festivos; en validación los tipos a (19,0 %) y d (17,5 %) no cumplen el 15 %.
- En validación y prueba se usaron las exógenas reales (promo y aperturas): si el plan cambia, el
  error crece.
- Residuos con algo de autocorrelación (Ljung-Box).
- Sin conciliación jerárquica: en el futuro, la suma de los tipos es 2,6 % menor que el total.
- Sin clima ni precios; datos solo hasta 2015.
- **Trabajo futuro:** experimentos de promo con grupos de control, incluir margen, conciliación
  jerárquica y un modelo GRU global. Pendientes de decisión: P-04 y P-07.

---

## 11. Preguntas trampa

**«Si la GRU tiene menor MAPE, ¿por qué no es el modelo principal?»**
La diferencia no es significativa (p = 0,29), SARIMAX gana en RMSE, cubre los 10 niveles (la GRU solo
el total), es interpretable y corre en la app sin TensorFlow.

**«¿No usaron la validación dos veces en la GRU?»**
Sí: para `EarlyStopping` y para elegir la variante. Eso hace optimista el error de validación, pero
la prueba quedó limpia y se usó una sola vez.

**«¿Volver de log con `expm1` da la media?»**
No exactamente: da la mediana del pronóstico, que puede quedar algo por debajo de la media. No se
corrigió; es una limitación menor.

**«Wilcoxon supone independencia y los errores diarios están autocorrelacionados.»**
Es cierto. El resultado es orientativo; una alternativa más rigurosa es la prueba de Diebold-Mariano.

**«¿El 6,8 % es el error que tendrán en la vida real?»**
No del todo: la prueba no tiene festivos y usa el calendario real de promo. La validación con origen
móvil (12,3 % de media en el total) es una referencia más prudente.

**«¿Rellenar el hueco de 2014 no inventa datos?»**
Sí, por eso se marcan con `imputed = 1` y se usan solo datos anteriores. La alternativa (dejar el
hueco) era peor: el total caía 15 % de forma artificial.

**«¿Por qué `open_share` puede usarse si es del futuro?»**
La apertura de tiendas se planifica de antemano (horarios y festivos), a diferencia de `Customers`.

**«¿Cambiaron algo después de ver la prueba?»**
No. Los órdenes se congelaron en `arima_selection.json` antes de generar la prueba, y la arquitectura
de la GRU se eligió con validación.

---

## 12. Glosario rápido

| Término | Significado |
| --- | --- |
| CRISP-DM | Metodología: negocio → datos → preparación → modelado → evaluación → despliegue |
| Fuga de datos | Usar información que no estaría disponible al pronosticar |
| Estacionaria | Media y varianza constantes en el tiempo |
| AIC | Criterio de ajuste que penaliza la complejidad del modelo |
| Fourier | Pares seno/coseno que representan un ciclo largo con pocas variables |
| Huber | Pérdida cuadrática para errores chicos y lineal para grandes (robusta) |
| EarlyStopping | Detiene el entrenamiento cuando la validación deja de mejorar |
| MC Dropout | Repetir la predicción con dropout activo para estimar incertidumbre |
| Ingenuo estacional | y(t) = y(t − 7) |
