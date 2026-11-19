# Guion de la demo (5 minutos)

App: <https://rossman-n2q2axurd2uksg9sonyqwv.streamlit.app/>

## Antes de empezar (10 minutos antes)

- [ ] Abrir la app y recorrer las 4 páginas: si estuvo dormida, la primera carga tarda 30–60 s.
- [ ] Dejar cada página en una pestaña: **Home**, **Dashboard**, **Escenarios**, **Series**.
- [ ] En Escenarios, dejar seleccionado **Patrón histórico** y nivel **Total cadena**.
- [ ] Tener abiertas las capturas de `docs/img/` como plan B si falla la conexión.
- [ ] Repartir los bloques: **A** habla de datos, estacionalidad y SARIMAX; **B** de la GRU y la app.

## Estructura

| Tiempo | Bloque | Quién | Pantalla |
| --- | --- | --- | --- |
| 0:00–0:40 | 1. Problema | B | Home |
| 0:40–1:30 | 2. Datos | A | Home (tabla de partición) |
| 1:30–2:15 | 3. Estacionalidad | A | Series |
| 2:15–3:15 | 4. Modelos y métricas | A + B | Dashboard |
| 3:15–4:30 | 5. Escenario de promo | B | Escenarios |
| 4:30–5:00 | 6. Cierre | A | Escenarios |

---

## 1. Problema (0:00–0:40) · Home

> Rossmann es una cadena de más de 1 100 tiendas. Marketing necesita saber cuánto se va a vender en las
> próximas 6 semanas y cuánto mueve una promoción, para decidir dónde y cuándo invertir.
>
> Nuestro objetivo SMART fue pronosticar las ventas diarias de los próximos **42 días** a nivel total,
> por tipo de tienda y para 5 tiendas, **superando al baseline y con MAPE menor a 15 %**. Lo cumplimos.

**Mostrar:** el bloque de objetivo y las 4 preguntas de negocio.

## 2. Datos (0:40–1:30) · Home

> Usamos el dataset de Kaggle: un millón de filas, de enero de 2013 a julio de 2015. Seguimos CRISP-DM.
>
> Encontramos un problema importante: **180 tiendas no tienen datos en el segundo semestre de 2014**.
> Las rellenamos usando solo información anterior al hueco; sin eso, la venta caía 15 % de forma
> artificial.
>
> Dos reglas que no negociamos: **no usar `Customers`**, porque no se conoce el futuro y sería fuga de
> datos, y **partir por tiempo, nunca al azar**: entrenamiento, validación de 42 días y prueba de 42
> días. La prueba se usó **una sola vez**.

**Mostrar:** la tabla «Datos y partición temporal».

## 3. Estacionalidad (1:30–2:15) · Series

> La serie tiene un patrón semanal muy fuerte: el lunes es el día más alto y el domingo casi no hay
> ventas. La autocorrelación en el lag 7 es 0,80.
>
> Por eso el modelo usa estacionalidad **s = 7** con una diferencia estacional. La estacionalidad anual
> (diciembre vende 17 % más) la modelamos con **términos de Fourier**, porque con 2,3 años de datos un
> SARIMA con s = 365 no es viable.

**Mostrar:** bajar hasta la descomposición STL y luego al correlograma ACF (picos cada 7 días).

## 4. Modelos y métricas (2:15–3:15) · Dashboard

**A:**
> Comparamos cuatro modelos: un baseline que repite la semana anterior, SARIMA, SARIMAX con variables
> que se conocen de antemano —promoción, apertura, festivos— y una red GRU.
>
> Los órdenes se eligieron con validación, por AIC y luego por el RMSE de validación.

**Mostrar:** el gráfico con SARIMAX y GRU sobre los datos reales (línea punteada).

**B:**
> La GRU es una red codificador–decodificador: lee 56 días y predice los 42 siguientes usando el
> calendario futuro. La primera versión no aprendía; al cambiar la arquitectura, el MAPE de validación
> bajó de 63 % a 7,6 %.

**Mostrar:** la pestaña **Comparar modelos**.

> En prueba, en el total: **baseline 17,4 %, SARIMAX 6,8 %, GRU 5,5 %**. SARIMAX es significativamente
> mejor que el baseline (Wilcoxon, p = 0,003) y cumple la meta en los 10 niveles. Entre SARIMAX y la
> GRU **no hay diferencia significativa** (p = 0,29).

## 5. Escenario de promo (3:15–4:30) · Escenarios

> Ahora la pregunta de marketing: ¿qué pasa si hacemos promoción de lunes a viernes todas las semanas
> de agosto y septiembre?

**Hacer:** seleccionar **Promo intensiva (lun–vie todas las semanas)**.

> El simulador vuelve a calcular en vivo con SARIMAX: pasamos de **268 a 306 millones**, unos
> **37 millones más (+14 %)**. Ojo: los días elegidos **reemplazan** el calendario histórico, que ya
> traía unos 15 días de promo.

**Hacer:** cambiar el nivel a **Tipo b** y luego a **Tipo c** para comparar el incremento.

> No todas las tiendas responden igual: con la misma promo intensiva, el **tipo b sube solo +6 %** y el
> **tipo c +18 %**. En el modelo, un día con todas las tiendas en promo se asocia con +18,5 % de venta
> en el tipo b y +54 % en el c.

Lo que debería verse en pantalla con «Promo intensiva»:

| Nivel | Incremento | En euros |
| --- | --- | --- |
| Total cadena | +14,0 % | +37,5 M€ |
| Tipo a | +13,4 % | +19,0 M€ |
| Tipo b | +6,2 % | +0,5 M€ |
| Tipo c | +17,9 % | +6,1 M€ |
| Tipo d | +15,5 % | +12,2 M€ |

## 6. Cierre (4:30–5:00)

> Conclusiones: la promoción es la palanca más clara, cerca de **+39 % en el total**; conviene decidir por
> tipo y por tienda, y planear con el calendario.
>
> Limitaciones: son asociaciones, no efectos causales, y no incluyen costos ni margen. El siguiente paso
> sería probar la promoción con grupos de tiendas comparables.

---

## Si algo falla

| Problema | Qué hacer |
| --- | --- |
| La app tarda en cargar | Seguir hablando del bloque actual; abrir las pestañas que ya estaban cargadas |
| La app no carga | Mostrar `docs/img/dashboard.png`, `escenarios.png` y `series.png` |
| Se acaba el tiempo | Saltar el cambio a Tipo b/c del bloque 5 y pasar al cierre |
| Preguntan algo fuera del guion | Ver `docs/GUIA_ESTUDIO.md` |
