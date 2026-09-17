# Ajuste de marea juzgado por mareógrafos y batimetría: MAREA frente a Granadeiro

Estado a 2026-09-17. Notebooks: `tide_boundary_comparison_<sitio>.ipynb`
(escalda, ferrol, ems, wadden); productos en `products_<sitio>/comparison_<tag>/`;
figuras en `docs/figures/tide_comparison/`; verdad batimétrica en
`data_v4/truth/vaklodingen/`.

---

## 0 · La pregunta, en llano

Para convertir imágenes de satélite en un mapa de cotas de una llanura
mareal hay que saber, en cada imagen, a qué altura estaba el agua. Esa
altura sale de un *contorno*: un modelo de marea oceánico, un mareógrafo de
puerto, o la media de los dos. El contorno vale en la boca. Dentro del
estuario la marea llega más tarde, a veces más de una hora, y si se lee el
contorno a la hora de la imagen se le asigna a cada píxel un nivel que no
era el suyo.

Dos métodos intentan medir ese retraso a partir de las propias imágenes y
leer el contorno "en el reloj correcto":

* **MAREA** (nuestro): divide la llanura en seis bandas por distancia a la
  boca y mide un retraso por banda, mirando solo si cada píxel salió mojado
  o seco en cada imagen. Aplica el retraso solo si supera 10 minutos y lo
  anula si sale negativo (un interior que se adelanta al mar es imposible).
* **Granadeiro et al. 2021**: mide un retraso por píxel ajustando una
  curva logística al infrarrojo cercano en subida y en bajada de marea, y
  suaviza el resultado con un spline.

Este documento contesta tres preguntas con datos que ninguno de los dos
métodos vio: ¿acierta el retraso? (mareógrafos interiores), ¿mejora el
mapa? (batimetría oficial holandesa), y ¿cuál de los dos lo hace mejor?

---

## 1 · Los ingredientes

### Los sitios

Solo sirven estuarios con al menos dos mareógrafos: uno en la boca para el
contorno y otro dentro como juez. El juez **nunca** entra en el contorno ni
en ningún ajuste.

| sitio | contorno (consenso de) | juez | distancia del juez a la boca | píxeles intermareales | escenas 2023-25 |
|---|---|---|---|---|---|
| Westerschelde (escalda) | EOT20 + Vlissingen + Breskens | Terneuzen | 21,9 km | 73 764 @ 20 m | 464 |
| Ría de Ferrol | EOT20 + Ferrol1 | Ferrol2 | 10,1 km | 103 759 @ 10 m | 461 |
| Ems-Dollard | EOT20 + Borkum | Delfzijl | 24,4 km | 477 745 @ 20 m | 462 |
| Wadden (Vlie) | EOT20 + Terschelling | Harlingen | 18,3 km | 367 076 @ 20 m | 462 |

"Distancia a la boca" es siempre la distancia geodésica *por el agua*
desde el borde marino del recuadro, no en línea recta.

### El contorno

Una sola función, `make_boundary(aoi, tide_model, n_gauges, exclude)`, con
dos interruptores: el modelo de marea (o ninguno) y el número de mareógrafos
más cercanos (o ninguno). Si se dan los dos, el contorno es la media de
ambos. Los mareógrafos se limpian de picos y mesetas y nunca se interpolan
a través de huecos de más de una hora. El mismo objeto alimenta a MAREA, a
Granadeiro y a la inversión sin reloj: entre filas cambia el método, nunca
el dato.

### Los cinco productos que se comparan

Todos parten del **mismo registro**: los píxeles intermareales de la
extracción de MAREA (frecuencia de agua entre 0,10 y 0,90 con al menos 30
observaciones claras, sin recortar al polígono), las mismas escenas y el
mismo contorno. Cambia una cosa por fila:

| producto | reloj con el que se lee el contorno | estimador de la cota |
|---|---|---|
| **sin reloj** | la hora de la imagen (τ = 0) | nuestra sigmoide sobre NDWI (`marea.invert_series`) |
| **MAREA aplicado** | el retraso de la banda del píxel, si supera el umbral y no está vetado | la misma sigmoide |
| **MAREA sin umbral** | el retraso ajustado de la banda, aunque sea pequeño o negativo (fila de sensibilidad, no producto) | la misma sigmoide |
| **cruzada** | el retraso por píxel de Granadeiro | la misma sigmoide |
| **Granadeiro** | el retraso por píxel de Granadeiro | su logística sobre NIR |

Las tres primeras solo se diferencian por el reloj: son el experimento
"con y sin ajuste MAREA". La cruzada separa lo que aporta el reloj de
Granadeiro de lo que aporta su estimador.

Ningún producto rellena huecos. Donde un píxel no se pudo invertir (el
ajuste no converge, o la amplitud o el número de observaciones no pasan las
guardas de `invert_series`: al menos 8 observaciones, contraste $b$ entre
0,15 y 2,5, nivel seco $|a| < 2,5$) el mapa queda en blanco. Por eso dos
mapas del mismo sitio pueden tener distinto número de píxeles: **un reloj
mejor hace converger más píxeles**, no rellena nada.

---

## 2 · Las métricas, con su fórmula y sus condiciones

### 2.1 Error de nivel en el juez (la métrica que decide)

El mareógrafo juez midió el nivel $h^{\mathrm{juez}}(t)$ cada 10 minutos
durante 2023-2025 (unas 147 000 muestras, tras quitar picos y huecos). Cada
método afirma conocer el retraso $\tau$ entre la boca y ese punto, así que
predice el nivel allí leyendo el contorno $\tau$ minutos antes:

$$
\hat h(t) = h^{\mathrm{contorno}}(t - \tau), \qquad
d(t) = \hat h(t) - h^{\mathrm{juez}}(t),
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_t \big(d(t) - \mathrm{mediana}(d)\big)^2 } .
$$

Se resta la mediana de $d$ porque el datum del mareógrafo (cero del puerto)
no es el del contorno (nivel medio del modelo): esa constante no la puede
ver ningún método y no es un error de reloj. Lo que queda mide fase,
amplitud y marejada mal asignadas.

Condiciones: rejilla de 10 min sobre todo 2023-2025 (nada se ajustó al
juez, así que no hace falta reservar un tramo); el contorno se lee con
interpolación lineal de una serie de 5 min; $\tau$ del método en el juez es,
para MAREA, el perfil por bandas interpolado en la distancia del juez a la
boca, y para Granadeiro la mediana de su mapa de retrasos en los 200
píxeles intermareales más cercanos al mareógrafo.

**Techo**: el mismo RMSE con el mejor $\tau$ posible, barrido de −30 a +180
min en pasos de 2,5 min *contra el propio juez*. Es lo máximo que cualquier
reloj podría conseguir; el resto del error es marejada y armónicos locales
que el contorno no lleva.

### 2.2 Cuánto cambia el mapa (comparación entre productos)

Sobre el **subconjunto común** (píxeles que todos los productos resuelven)
y tomando como referencia el mapa sin reloj $z^{0}$:

$$
\Delta_i = z_i - z^{0}_i, \qquad
\text{desplazamiento} = \mathrm{mediana}(\Delta), \qquad
\mathrm{RMS}_{\text{centrado}} = \sqrt{\frac{1}{n}\sum_i \big(\Delta_i - \mathrm{mediana}(\Delta)\big)^2},
$$

más la correlación de Pearson entre $z$ y $z^{0}$ y la fracción de píxeles
que se mueven más de 10 cm. Esto **no** dice quién acierta, solo cuánto se
parecen dos mapas: para eso están 2.3 y 2.4.

### 2.3 Error contra la batimetría oficial (Vaklodingen, sitios holandeses)

Las *Vaklodingen* de Rijkswaterstaat son la batimetría y topografía
intermareal anual en rejilla de 20 m (datum NAP). Para cada píxel de
nuestros productos se toma la celda más cercana y su levantamiento más
reciente entre 2019 y 2025, y se calcula

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i \big(\Delta_i - \mathrm{mediana}(\Delta)\big)^2},
\qquad \Delta_i = z_i - z^{\mathrm{verdad}}_i,
$$

$$
\text{pendiente} = \text{coeficiente OLS de } (z_i - \mathrm{mediana}\, z) \text{ sobre } (z^{\mathrm{verdad}}_i - \mathrm{mediana}\, z^{\mathrm{verdad}}).
$$

Otra vez se centra en mediana (NAP frente al datum del contorno). La
pendiente dice si el mapa reproduce el relieve (1) o lo comprime (< 1).

Condiciones: solo verdad dentro del rango mareal que los productos pudieron
muestrear (el rango de niveles de las escenas, ±0,25 m). Una celda de 20 m
junto al borde de la llanura cae a menudo en el canal a −8 m, y ningún
producto puede ni debe representar eso: con esas celdas el Westerschelde
daba 1,4 m de RMSE; sin ellas, 0,43. Se excluyen 3 528, 2 491 y 14 608
píxeles (Escalda, Wadden, Ems). El levantamiento del Dollard alemán es de
2020, cinco años antes que nuestras escenas. Se comprueba además con un
adelgazamiento 5 × 5 (un píxel por bloque de 100 m): mueve los RMSE 0,02-0,06
m y ningún orden.

### 2.4 Error contra RTK (Villaviciosa)

Mismas fórmulas que 2.3, con los 181 píxeles RTK del split de desarrollo
(el 35 % de bloques de 150 m reservado sigue sellado). El LiDAR del IGN no
es verdad válida en Villaviciosa: sobre la llanura toma siete valores
enteros y el 83 % es +2,00 m, la lámina de agua del vuelo.

---

## 3 · Resultados

### 3.1 ¿Acierta el retraso? El juez

| | Escalda / Terneuzen | Ferrol / Ferrol2 | Ems / Delfzijl | Wadden / Harlingen |
|---|---|---|---|---|
| contorno tal cual (sin reloj) | 0,356 | 0,112 | 0,620 | 0,562 |
| **+ reloj MAREA aplicado** | 0,356 (τ = 0, bajo umbral) | 0,112 (τ = 0, veto físico) | **0,360 (τ = +71 min)** | **0,301 (τ = +128 min)** |
| + reloj MAREA sin umbral | 0,310 (τ = +7) | 0,243 (τ = −25) | 0,360 | 0,301 |
| + retraso de Granadeiro | 0,455 (τ = +52) | 0,253 (τ = −26) | 0,446 (τ = +31) | 0,556 (τ = +1) |
| techo (mejor τ posible) | 0,271 (+20) | 0,108 (+5) | 0,357 (+65) | 0,238 (+95) |

Lectura:

* **Ems**: MAREA mide 0 → +104 min a lo largo de 35 km, aplica +71 en
  Delfzijl; el mareógrafo dice +65. Recupera el 99 % de lo alcanzable.
* **Wadden**: MAREA aplica +128 min donde el juez quiere +95: se pasa, pero
  recupera el 81 %. Granadeiro lee +1 min y no gana nada.
* **Escalda**: hay 20 min reales; MAREA mide +7, que no supera el umbral, y
  el producto no cambia. Granadeiro dice +52 y empeora.
* **Ferrol**: no hay retraso (techo +5 min, 4 mm). Los dos métodos ven un
  retraso negativo espurio de −25 min; MAREA lo veta, Granadeiro lo aplica
  y dobla el error.

**Figuras `<sitio>_judge_sweep.png`**: la curva es el RMSE en el juez para
cada retraso barrido; las verticales son los retrasos que cada método
propone. Cuanto más cerca del mínimo cae una vertical, mejor mide ese
método. En el Ems la vertical de MAREA cae casi en el mínimo; en el
Wadden se pasa a la derecha; en el Escalda se queda a la izquierda.

![Escalda](figures/tide_comparison/escalda_judge_sweep.png)
![Ems](figures/tide_comparison/ems_judge_sweep.png)
![Wadden](figures/tide_comparison/wadden_judge_sweep.png)
![Ferrol](figures/tide_comparison/ferrol_judge_sweep.png)

Qué contorno es mejor antes de aplicar ningún método (tal cual, en el juez):

| sitio | solo modelo | solo mareógrafos | consenso |
|---|---|---|---|
| Escalda | 0,504 | **0,280** | 0,356 |
| Ferrol | 0,154 | **0,096** | 0,112 |
| Ems | 0,771 | **0,497** | 0,620 |
| Wadden | 0,614 | **0,538** | 0,562 |

El mareógrafo solo gana siempre: lleva la marejada real. El consenso la
diluye a la mitad. Merece la pena solo como red cuando el mareógrafo tiene
huecos (Vlissingen está apagado 18 meses).

### 3.2 ¿Cuánto cambia el mapa? Con y sin MAREA

Referencia: el mapa sin reloj. Subconjunto común de cada sitio.

| sitio | producto | px resueltos | desplazamiento mediano (m) | RMS centrado (m) | corr | % px que se mueven > 10 cm |
|---|---|---|---|---|---|---|
| Escalda | sin reloj | 65 256 | — | — | — | — |
| | MAREA aplicado | 64 258 | 0,000 | 0,000 | 1,000 | 0 % |
| | MAREA sin umbral | 64 280 | 0,000 | 0,064 | 0,998 | 3 % |
| | cruzada | 65 126 | 0,000 | 0,258 | 0,963 | 30 % |
| | Granadeiro | 47 189 | +0,069 | 0,654 | 0,810 | 81 % |
| Ferrol | sin reloj | 54 458 | — | — | — | — |
| | MAREA aplicado | 52 916 | 0,000 | 0,000 | 1,000 | 0 % |
| | MAREA sin umbral | 54 436 | 0,000 | 0,153 | 0,972 | 10 % |
| | cruzada | 55 710 | 0,000 | 0,148 | 0,973 | 12 % |
| | Granadeiro | 52 587 | −0,175 | 0,481 | 0,717 | 80 % |
| Ems | sin reloj | 418 459 | — | — | — | — |
| | **MAREA aplicado** | **428 414** | 0,000 | **0,406** | 0,577 | **69 %** |
| | cruzada | 404 692 | −0,063 | 0,364 | 0,611 | 68 % |
| | Granadeiro | 105 390 | −0,240 | 0,342 | 0,573 | 85 % |
| Wadden | sin reloj | 308 458 | — | — | — | — |
| | **MAREA aplicado** | **359 382** | +0,253 | **0,495** | 0,481 | **75 %** |
| | cruzada | 308 614 | +0,253 | 0,522 | 0,508 | 84 % |
| | Granadeiro | 66 757 | +0,363 | 0,549 | 0,411 | 89 % |

En Escalda y Ferrol la fila "MAREA aplicado" es idéntica al mapa sin reloj
porque el producto no aplicó ningún retraso (por eso RMS = 0 y corr = 1:
son el mismo mapa). En Ems y Wadden el reloj mueve tres de cada cuatro
píxeles más de 10 cm y añade un 2 % y un 17 % de píxeles resueltos.

**Figuras `<sitio>_dem_without_vs_with.png`** (cuatro paneles): arriba,
el mapa sin reloj y el mapa MAREA, mismos píxeles de entrada, misma escala
de color; blanco = píxel no resuelto por las guardas, no hay relleno.
Abajo, la diferencia MAREA − sin reloj (izquierda, el producto) y la
diferencia con los relojes sin umbral (derecha, sensibilidad). En Escalda y
Ferrol el panel de abajo a la izquierda es plano por construcción.

![Ems](figures/tide_comparison/ems_dem_without_vs_with.png)
![Wadden](figures/tide_comparison/wadden_dem_without_vs_with.png)
![Escalda](figures/tide_comparison/escalda_dem_without_vs_with.png)

**Figuras `<sitio>_granadeiro_maps.png`**, cuatro paneles, todos sobre los
mismos píxeles de entrada que los mapas anteriores:

* *Arriba a la izquierda, el mapa de retrasos de Granadeiro.* Su método mide
  el retraso solo en una muestra de ~1 000 píxeles cercanos al nivel medio
  y extiende esos valores a toda la llanura con un spline de placa delgada.
  El color es el retraso en minutos. Las manchas violetas y negras (−100 a
  −150 min) que aparecen en los bordes y en charcas aisladas no son medidas:
  son el spline extrapolando fuera de la muestra. Ese mapa es el que la fila
  "cruzada" mete en nuestra inversión.
* *Arriba a la derecha, su mapa de cotas* (la inflexión de la logística
  sobre NIR). Tiene muchos menos píxeles porque su ajuste por píxel no
  converge donde hay pocas escenas admisibles (33-120 según el sitio,
  frente a 460 en nuestro registro), y en el Ems y el Wadden además solo se
  ajustó 1 de cada 4 píxeles (`px_stride`, declarado), porque la logística
  por píxel sobre 400 000 píxeles cuesta 13 horas. **Por eso en la primera
  versión de estas figuras ese panel y el de diferencia salían grises**: una
  retícula con tres píxeles vacíos de cada cuatro se ve como el fondo del
  sombreado. En la versión actual cada píxel ajustado se pinta sobre su
  bloque de 4 × 4 solo para verlo (el título lo dice); las tablas usan
  únicamente los píxeles ajustados.
* *Abajo a la izquierda, la cruzada*: su retraso, nuestra inversión. Tiene
  tantos píxeles como el mapa sin reloj porque el estimador es el nuestro.
* *Abajo a la derecha, Granadeiro − MAREA*, píxel a píxel, misma escala de
  ±0,5 m que las diferencias anteriores.

**Figuras `<sitio>_granadeiro_vs_marea_scatter.png`**: cota de Granadeiro
(izquierda) y de la cruzada (derecha) contra la de MAREA, píxel a píxel,
subconjunto común, densidad en escala logarítmica; la diagonal discontinua
es acuerdo perfecto. Una nube pegada a la diagonal dice que los dos mapas
ordenan el relieve igual; una nube desplazada dice que difieren en datum
(la inflexión de la logística NIR no es el mismo cero que el umbral
mojado/seco); una nube ancha dice que difieren píxel a píxel.

![Escalda](figures/tide_comparison/escalda_granadeiro_maps.png)
![Ems](figures/tide_comparison/ems_granadeiro_maps.png)
![Wadden](figures/tide_comparison/wadden_granadeiro_maps.png)
![Ferrol](figures/tide_comparison/ferrol_granadeiro_maps.png)

### 3.3 ¿Mejora el mapa? La batimetría oficial

| sitio | producto | RMSE (m) | pendiente | n | común: RMSE | pendiente |
|---|---|---|---|---|---|---|
| Westerschelde | sin reloj | 0,434 | 0,767 | 62 153 | 0,383 | 0,800 |
| | MAREA aplicado (= sin reloj) | 0,433 | 0,767 | 62 111 | 0,383 | 0,800 |
| | cruzada | 0,507 | 0,715 | 61 952 | 0,442 | 0,757 |
| | Granadeiro | 0,777 | 0,780 | 45 005 | 0,763 | 0,789 |
| Wadden / Vlie | sin reloj | 0,474 | 0,741 | 302 001 | 0,464 | 0,763 |
| | **MAREA aplicado** | **0,354** | **0,833** | 349 043 | **0,261** | **1,046** |
| | cruzada | 0,440 | 0,911 | 304 083 | 0,381 | 1,033 |
| | Granadeiro | 0,477 | 0,645 | 65 576 | 0,414 | 0,798 |
| Ems-Dollard | sin reloj | 0,565 | 0,243 | 404 021 | 0,556 | 0,221 |
| | **MAREA aplicado** | **0,378** | **0,637** | 412 415 | **0,338** | **0,666** |
| | cruzada | 0,412 | 0,567 | 388 392 | 0,392 | 0,570 |
| | Granadeiro | 0,531 | 0,387 | 98 953 | 0,405 | 0,501 |

Subconjuntos comunes: 44 750, 60 249 y 93 273 píxeles.

**Figura `vaklodingen_by_band.png`**: arriba el RMSE y abajo la pendiente
contra la batimetría, por banda de distancia a la boca, para los cuatro
productos. Lo que hay que mirar es la separación entre la línea gris (sin
reloj) y la roja (MAREA) según nos alejamos de la boca.

![Por bandas](figures/tide_comparison/vaklodingen_by_band.png)

Lectura por bandas (RMSE / pendiente / sesgo mediano):

* **Wadden, bandas de 8 y 11 km** (reloj 79-88 min): sin reloj 0,39 / 0,97
  / −0,23 m y 0,34 / 0,84 / −0,43 m; con MAREA 0,17 / 1,02 / +0,01 m y 0,17
  / 1,10 / 0,00 m. El reloj elimina un sesgo de nivel de 20-40 cm y deja el
  relieve a escala 1:1. En las bandas interiores (16-21 km), donde MAREA
  se pasa de retraso y la verdad es de 2022, todos los productos quedan en
  0,3-0,4 de pendiente.
* **Ems**: sin reloj, más allá de 29 km la pendiente es 0,03-0,10, es decir,
  el mapa no guarda relación con el relieve; MAREA la sube a 0,5-0,6 y baja
  el RMSE a la mitad (0,70 → 0,44, 0,56 → 0,35, 0,53 → 0,34). Los 0,35-0,44
  m que quedan en el Dollard son en parte una llanura muy móvil medida en
  2020.
* **Westerschelde**: sin reloj y MAREA coinciden (0,43 m, pendiente 0,77),
  con un sesgo de +0,14 a +0,27 m más allá de 10 km: es el retraso de 20 min
  que MAREA no midió, y vale unos 20 cm. Sobre el estimador la comparación
  es limpia: Granadeiro 0,78 m frente a nuestros 0,43 en sus propios
  píxeles.

### 3.4 Villaviciosa contra RTK: el estimador, no el reloj

`experiments/c5_validation_matrix.py`, split de desarrollo, época 2023-25.

| reloj × estimador | RMSE (m) | pendiente | n | común 53 px: RMSE | pendiente |
|---|---|---|---|---|---|
| sigmoide · sin reloj | 0,163 | 0,747 | 120 | 0,127 | 0,848 |
| sigmoide · relojes MAREA aplicados | 0,163 | 0,747 | 120 | 0,127 | 0,848 |
| sigmoide · relojes MAREA sin umbral | 0,169 | 0,757 | 120 | 0,121 | 0,869 |
| sigmoide · retraso de Granadeiro (cruzada) | 0,233 | 0,524 | 120 | 0,205 | 0,607 |
| logística · sin reloj (Granadeiro) | 0,315 | 1,081 | 85 | 0,367 | 1,218 |
| logística · su retraso (Granadeiro) | 0,279 | 0,356 | 100 | 0,304 | 0,364 |
| HSR 2023-25 publicado | 0,199 | 0,759 | 146 | 0,130 | 0,898 |
| escalón tipo DEA publicado | 0,155 | 0,761 | 88 | 0,129 | 0,822 |
| logística 10 años · sin retraso (publicado) | 0,141 | 0,878 | 120 | 0,112 | 0,980 |
| producto MAREA publicado (ajuste decadal) | 0,240 | 0,864 | 119 | 0,139 | 0,956 |

En Villaviciosa el RTK está donde MAREA mide τ ≈ 0, así que el reloj no
puede cambiar nada: este sitio valida el estimador. Con tres años de
escenas nuestra sigmoide (0,163) gana a la logística de Granadeiro (0,315);
con diez años su logística es la mejor de todas (0,141). El mapa de
retrasos de Granadeiro perjudica a cualquier estimador (pendiente 0,75 →
0,52 en el nuestro). Salvedad destapada aquí: el producto MAREA publicado
de Villaviciosa se ajustó con las 1 379 escenas de diez años, no con la
época 2023-25 (0,240 frente a 0,163 en los mismos píxeles); hay que
reeditarlo por épocas.

---

## 4 · Conclusión

1. **El retraso interior existe y MAREA lo mide desde la imagen** donde
   importa: +71 min en el Ems (el juez dice +65) y +128 en el Wadden (el
   juez dice +95). Con eso el error de nivel en el interior baja de 0,62 a
   0,36 m y de 0,56 a 0,30 m, cerca del máximo que cualquier reloj podría
   lograr (0,357 y 0,238).
2. **Y el mapa mejora contra una batimetría independiente**: en el Wadden
   el RMSE baja de 0,47 a 0,35 m (0,46 → 0,26 en el subconjunto común) y la
   pendiente sube de 0,76 a 1,05; en el Ems, de 0,57 a 0,38 m (0,56 → 0,34)
   y de 0,22 a 0,67. Sin reloj, el fondo del Ems no era un mapa.
3. **Donde no hay retraso medible, MAREA no toca nada**, y eso es correcto:
   en Ferrol el veto físico evita aplicar un retraso espurio que Granadeiro
   sí aplica y que dobla el error. El coste de esa prudencia se ve en el
   Escalda: 20 min reales que MAREA mide como 7 y descarta, unos 20 cm de
   sesgo que se quedan en el mapa.
4. **Granadeiro no gana en ningún sitio**: su retraso recupera el 66 % de lo
   alcanzable en el Ems y el 0 % en el Wadden, empeora Escalda y Ferrol, y
   su estimador logístico sobre NIR es peor que nuestra sigmoide con tres
   años de escenas (0,78 frente a 0,43 m contra la batimetría del
   Westerschelde; 0,315 frente a 0,163 contra RTK). Solo con diez años de
   escenas y sin retraso alguno su estimador es el mejor, en Villaviciosa.
5. **El mejor contorno es el mareógrafo más cercano**, no el consenso con
   el modelo: en los cuatro sitios el mareógrafo solo da el menor error en
   el juez. El consenso solo compensa como red frente a huecos.
6. **Queda por corregir**: el umbral de 10 min se comió el Escalda y el
   bandeado por cuantiles pone al juez en el borde de bandas anchas en
   Escalda y Wadden (bandas más finas cerca del juez es la prueba
   siguiente); el producto MAREA de Villaviciosa debe reeditarse por
   épocas.

---

## 5 · Lo que hubo que arreglar por el camino (todo en la librería)

* Los registros IOC llevan picos centinela a −10 m y **mesetas** (Ferrol1
  escribió +4 m durante días): `gauges.despike`, mediana móvil más un tope
  de 2,5 m sobre el residuo del ajuste armónico del propio registro.
* Los huecos de mareógrafo (Vlissingen, 18 meses) se puenteaban con una
  recta: `GaugeBoundary` devuelve NaN en huecos > 1 h; el consenso cae al
  modelo.
* El catálogo Sentinel-2 devolvía listas de horas de paso **parciales o
  vacías** durante caídas, en silencio: `get_overpass_times` reintenta la
  búsqueda entera y lanza error; los notebooks cachean las horas en disco.
* La distancia a la boca se medía desde todos los bordes de la imagen:
  `mouth_side`.
* El tope de búsqueda del retraso de MAREA sube de 120 a 180 min.
* Granadeiro guarda puntos de control en sus dos etapas lentas; `px_stride`
  para llanuras muy grandes; su regla de nube se evalúa sobre la llanura y
  no sobre el marco (su regla dejaba 11 escenas de subida y necesita 12).
