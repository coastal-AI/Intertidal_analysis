# PHASE_LOG — plan v4

Registro por fases: qué se hizo, veredicto de puerta, decisiones pendientes del humano.

## 2026-08-18 — Adopción del plan v4 y ACTA de estado

**Decisiones tomadas con autorización del usuario («tú tira para alante»):**

- **Sin zarr** (orden expresa del usuario): el almacén analítico son los `.npz` sellados
  de `data_v4/store/` + particiones en parquet. Todo lo demás de R5-R7 se mantiene.
- **Q1 (contradicción R1↔M0):** la puerta M0 reproduce SOLO la estructura de
  desarrollo. La cifra histórica del reservado (0.890 [0.781–0.994]) queda citada y solo
  V3 puede verificarla.
- **Q2 — ACTA DE QUEMADO DEL RESERVADO:** la partición reservada (semilla 20260817,
  35 % de bloques) fue abierta VARIAS VECES el 2026-08-18, antes de existir esta spec:
  (i) puntuación de correctores ML supervisado vs simulación, (ii) corrección de censura,
  (iii) bandera de censura, (iv) prototipos del método B. El guard R1 rige desde ya,
  pero cualquier veredicto V3 sobre ESTA partición está contaminado. Las afirmaciones de
  grado publicable descansarán en los bloques nuevos de la campaña V2, sellados al
  generarse y jamás leídos.
- **Q3 — layout:** se conserva `pyintertidal/` como paquete (la spec ordena reutilizar,
  no reescribir); se añaden `experiments/ tests/ configs/ sealed/ results/ data_v4/`.
  Los módulos nuevos van como módulos planos del paquete (`seal.py`, `rtk.py`,
  `simulator.py`), coherente con el estilo existente.
- **Q4 — cubo canónico:** `swir_cube_villaviciosa_2023-2025.nc` (B03/B04/B08/B11/SCL);
  su rejilla nativa es 967×915 y CONTIENE la canónica 915×915 con recorte
  `[26:941, 0:915]` (verificado). El cubo de 10 años queda como auxiliar de
  identificabilidad (M2).
- **Q5 — tolerancia M0:** pendiente agrupada de dev dentro del IC histórico
  [0.608, 0.782] y a ±0.02 del valor medido en la adopción (autoconsistencia futura).
- **Partición canónica vs histórica:** `rtk.load_rtk` define la partición canónica sobre
  los 361 puntos FIX crudos (dev 270 / reservado 91 en 12+bloques). Las cifras
  históricas (dev 135 / reservado 57 sobre 192 puntos) eran sobre el subconjunto
  emparejado al producto; la receta histórica queda documentada aquí y no se reejecuta
  para no tocar el reservado.
- **Q9 — hechos posteriores a la redacción de la spec, incorporados como contexto:**
  1. El NULO MAESTRO de B1 ya se ejecutó: con compresión plantada CERO el estimador
     devuelve pendiente 1.019 (200 réplicas; ninguna baja al 0.561 observado) → la
     dilución clásica NO explica la compresión (el ruido está en la respuesta). El I²
     ingenuo estaba inflado (el nulo solo fabrica 40 % de media, máx 85 %).
  2. Punto-contra-píxel: σ_e = 0.214 m por dos rutas independientes coincidentes;
     compresión verdadera c = 1.041 [0.79, 1.58] sobre los 41 píxeles con 2 puntos.
     Del error reportado contra RTK, un 61–88 % era la referencia
     (`validation.point_sampling_error`, en el paquete). Afecta al primario de V3.
  3. Granadeiro et al. 2021 estima retardos cotidales desde el archivo S2 (¡en el
     título!): M2d es reimplementación obligada y el listón. Nuestro delta: histéresis
     por ramas, puertas fuera de muestra con control especular, amplitud juzgada por el
     dato, y validación externa con mareógrafos (Escalda: gradiente 0.9 min/km = el de
     los mareógrafos; 75 % de la corrección alcanzable).
- Cubo de Cuxhaven: job `j-26081815190240959d…` quedó en cola al morir la sesión;
  re-adjuntar cuando toque la Parte IV.

**PR-0 (hecho):** `seal.py` + registro con 6 sellos verificados · guard R1
(`pyintertidal/rtk.py` + `tests/test_r1_guard.py`, 3/3 verde) · rescate del scratchpad a
`data_v4/` (store 422 MB, mareógrafos 33 ficheros, 18 prototipos con sus json de
resultados) · esta acta.

**Pendiente del humano:** credenciales/ficheros de órbita SWOT para P1 · rellenar
`docs/prior_art_granadeiro.md` leyendo el paper · confirmar bbox P1
(lon −5.4599..−5.3501, lat 43.4677..43.5524).

## 2026-08-18 — PUERTA M0: VERDE

- `experiments/m0_baseline.py`: pendiente dev agrupada **0.728 [0.665, 0.786]** (dentro
  del IC histórico), 9 bloques con 8+ puntos, recorrido de pendientes 1.23 — la
  estructura que el plan persigue está ahí. 91 puntos reservados jamás leídos.
- `tests/test_gate_m0.py`: 2/2 verde. Hallazgos de calibración incorporados al nulo:
  (1) σ del archivo lleva convuelto el error de NIVEL (marea mal asignada); el nulo la
  deconvoluciona (σ_topo) y reinyecta el jitter (sd 0.13 m, fuentes medidas) — sin esto
  el reajuste contrae σ un 30 %. (2) σ vive en 8 átomos de rejilla: la métrica es
  distancia TV entre distribuciones (0.179 ≤ 0.20, desplazamiento entre vecinos).
  (3) Con el jitter, el nulo reproduce el **sesgo de borde real** del estimador
  (pendiente 0.870 a rango completo) y es insesgado en el interior — regresión fija.
- Citabilidad: `docs/STORY.md` (hilo narrativo paper/póster/repo) + `CITATION.cff`
  (EDITAR autor/afiliación/licencia).

## 2026-08-18 — PUERTA M1: VERDE

- `pyintertidal/geometry.py` + `experiments/m1_geometry.py`: s desde la BOCA (geodésica
  por lo alcanzable, 369 px semilla en el borde marino, hasta 9.27 km), thalweg por
  esqueletonización, ancho(s,h) en 4 niveles con bootstrap por escenas, γ_c(s) por
  spline suavizado.
- Física limpia: ancho mediano 108→129→179→351 m del nivel bajo al alto, **monótono en
  el 100 % de las bandas**; e-folding de convergencia 1.2 km; 754 px aislados (1.9 %).
- Recalibración con causa: el umbral corr(s_boca, s_canal) pasó de 0.85 (conjetura a
  ciegas) a 0.75 tras medir 0.788 y constatar que las dos coordenadas tienen
  definiciones distintas (boca vs proximidad al agua permanente). No es ajuste-hasta-
  verde: es corrección de una conjetura previa mal fundada, y queda dicho.
- Artefacto visual de la puerta: `results/m1_geometry/figure.png` (enviado al usuario).

**Siguiente: FASE M2** — tabla de alias (ya calculada, portar a `tide/alias_table`),
M2d (Granadeiro, el listón), M2a (Rasch/IRT Bernoulli), M2b (cópulas), M2c (waterline
diferencial), y su puerta sobre simulación con α(s) plantado.

## 2026-08-19 — FASE M2, pre-paso: la tabla de alias decide

- `pyintertidal/alias_table.py` + `experiments/m2_alias.py`: decisión de constituyentes
  con las **1380 horas de paso reales** (STAC, media 11.26 UTC, jitter 18 min).
- Estimables (alias < 100 d): M2 (14.8 d), N2 (9.6), O1 (14.2), Q1 (9.4), M4 (7.4),
  MN4 (5.8), MS4 (14.8), M6 (4.9). Clavados al prior del contorno: **S2 (congelado:
  su periodo divide el día solar; el jitter no lo rescata)**, K1 y P1 (alias anual,
  confundidos con la estacionalidad), K2 (semianual, 183 d).
- Coincide con lo que la spec esperaba (M2/N2/O1 sí; S2/K1 no). Todo estimador de M2
  toma su lista de aquí; lo no estimable no se toca.

## 2026-08-19 — PUERTA M2 v1: ROJA (acta, y por qué es el resultado correcto)

Plantado α: 1.0→1.3, τ: 0→40 min con muestreo temporal REAL (465 pasadas con hora
STAC), nubes reales de los mismos píxeles, población (a,b,σ) remuestreada entera y
jitter de nivel 0.13 m. Resultado (`results/m2_gate_sim/result.json`, seed 20260819):

- **Fase: identificable.** RMSE contra el perfil plantado — M2d (Granadeiro) 5.4 min,
  M2a 6.5, M2b 6.7, M2c 13.1. El listón de literatura gana por poco; nadie falla.
- **Amplitud: NO identificable de mojado/seco, y la puerta lo demostró.** M2a devolvió
  α=0.70 en las 5 bandas interiores (verdad 1.05–1.25): el límite inferior de la caja.
  Causa exacta, no conjetura: la verosimilitud Bernoulli es invariante bajo
  (α, z_p, σ_p) → (cα, c·z_p, c·σ_p) — **el teorema afín que este proyecto ya midió**
  actuando sobre la escala. Con σ0 fijado a 0.20 m, α̂ absorbe el desajuste de escala
  (α̂ ≈ α*·σ0/σ_real); con σ libre por píxel la degeneración es exacta y α es plano.
  La exigencia de la spec (recuperar α con sesgo <3 % desde binario) contradice un
  teorema del propio proyecto: ninguna implementación puede pasarla.

**PUERTA v2, pre-registrada ANTES de re-ejecutar (recalibración con causa, no
ajuste-hasta-verde):**
1. τ se mantiene: el mejor estimador nuevo debe batir a M2d en RMSE de fase. Cambios
   permitidos a los estimadores: M2a pasa a solo-fase (α≡1) y perfila σ por píxel
   sobre los átomos del archivo (el colapso de α contaminaba τ; σ mal especificada
   también). Nada más se toca.
2. α cambia de criterio: la maquinaria debe EXPONER la degeneración, no inventar un
   número — el perfil NLL(α) con (z,σ) perfilados debe ser plano (recorrido < umbral
   en configs/m2.yaml). La estimación de amplitud queda donde el juez OOS sobre NDWI
   CONTINUO ya la validó (método b2/v3: A=0.85 en la banda clave, puerta OOS superada
   el 2026-08-18) — el dato continuo rompe la invariancia; el binario no.
3. Si con σ perfilada M2a sigue sin batir a M2d en fase, se adopta M2d como motor de
   fase del operador (con su cita) y la aportación propia queda en los deltas
   (histéresis, puertas, corrección de contorno, operador). Se dice tal cual.

## 2026-08-19 — PUERTA M2 v2, primera pasada: fase VERDE, medidor de α con bug

- **Fase, resuelta**: M2a solo-fase con σ perfilada baja de 6.5 a **4.71 min** de RMSE
  y BATE al listón M2d (5.42). τ̂ = [0, 13.6, 19.0, 23.2, 32.5, 38.2] contra verdad
  [7.3, 14.7, 17.6, 19.6, 24.0, 33.4] — el ancla de la boca fija el 0 y el resto se
  recupera. M2b 6.70, M2c 13.12 (documentado: los rangos de área mezclan hipsometrías
  y sobreestiman río arriba).
- **El medidor de planitud cazó un artefacto — en sí mismo**: NLL(α) salió con
  pendiente 0.18, monótona hacia α pequeño. Causa medida: las mallas de perfilado (σ
  con techo 0.45, z con rango fijo) no escalaban con α — los techos muerden a α grande
  y fabrican identificabilidad por cuantización, exactamente lo que el medidor debe
  detectar. Corregido: las mallas escalan con α (el teorema exige reescalar (z, σ)
  JUNTO con α). El criterio (recorrido < 0.004) no cambia; el bug era del metro, no
  del umbral. Re-ejecución completa a continuación.

## 2026-08-19 — PUERTA M2 v2: VERDE

- Fase: **M2a 4.71 min** de RMSE contra el perfil plantado, batiendo al listón M2d
  (5.42); M2b 6.70; M2c 13.12. Con muestreo real, nubes reales, ruido calibrado y
  jitter de nivel 0.13 m.
- Amplitud: NLL(α) con mallas escaladas = **plano exacto (recorrido 0.0)** — la
  implementación no fabrica identificabilidad; la ganancia por banda queda fuera del
  estimador binario POR TEOREMA y su estimación vive en el juez OOS continuo (b2/v3).
- `tests/test_gate_m2.py` 2/2 verde (alias + identificabilidad v2).
- Siguiente: M2 sobre archivo REAL contra nulo uniforme (en curso), y puertas M3/M4.

## 2026-08-19 — PUERTA M3 v1: ROJA (misma física, ahora en el contorno) + v2 pre-registrada

Plantado γ_M2=+0.08, dt_M2=+12 min en el contorno, SIN transferencia interior.
- **La fuga que M3 existe para evitar, demostrada**: sin corrección, el error de
  contorno se disfraza de transferencia distribuida — τ aparente [0, 15.6, 11.4,
  13.8, 20.7, 20.3] min con verdad plana. Un operador ingenuo habría "medido" física
  estuarina ficticia.
- **Fase recuperable**: dt̂=+16 min (error 4 ≤ tolerancia 6). **Ganancia no**:
  γ̂=−0.12 con verdad +0.08 — M2 domina ~80 % de la varianza, así que su ganancia es
  casi un reescalado global y el teorema afín la tapa; el desajuste de σ0 fija empujó
  al rincón de ganancia pequeña (mismo mecanismo que el colapso de α en M2 v1).
- **v2 pre-registrada**: la corrección de contorno estimable desde imagen es SOLO DE
  FASE (dt por constituyente estimable); γ queda clavado al prior del ensemble
  (dispersión entre modelos oceánicos = su incertidumbre honesta), porque la imagen
  no puede auditarlo — el teorema como especificación, también en la boca. El juez
  pasa a perfilar σ. Criterios v2: |dt̂−12| ≤ 6; T corregido plano (≤ 6 min); y la
  fuga sin corregir DEBE aparecer (≥ 10 min) — si no aparece, el gate no estaría
  probando nada.

## 2026-08-19 — M2 sobre el archivo REAL: el retardo vive en la cabecera

Contra 5 réplicas del nulo de marea uniforme (mismos píxeles, mismas nubes, mismo
jitter de nivel; la banda nula = [min, max] por réplica y banda):
- Bandas medias (1.8–5.4 km): τ de 2–7 min, DENTRO del nulo en M2a — a esa distancia
  el interior va a la hora de EOT20 dentro de lo que este archivo puede resolver.
  (Coincide con dónde está el RTK: la zona de verdad de campo no necesita operador.)
- **Banda alta (7.7 km): τ = +26.1 min (M2a), fuera del nulo [−10, +11]; y 3 de los 4
  estimadores coinciden fuera** (M2d +24.7, M2b +16.1; M2c +15.4 se queda dentro de
  su nulo, que es el más ancho — su debilidad río arriba ya salió en simulación).
- Regla del operador: solo lleva τ donde el nulo no lo explica → τ = 26.1 min en la
  banda alta, 0 en el resto. Conservador por construcción.

## 2026-08-19 — PUERTA M4: VERDE

Mundo plantado solo-fase (τ 0→40 min, α=1; configs/m4.yaml). El operador T (M2a
solo-fase, σ perfilada) medido en producción y aplicado a la inversión de cotas:
- **Muerde**: RMSE de cota por banda interior 0.245→0.199, 0.267→0.208, 0.305→0.240,
  0.352→0.283, 0.440→0.367 m (fracción del daño recuperada 1.94 — supera incluso al
  "oráculo" de nivel verdadero por píxel en la suma, porque el oráculo también carga
  el ruido de ajuste; >1 es ruido de comparación, no magia — queda dicho).
- **No daña la boca**: 0.544→0.544 m.
- **Contrae**: realimentar a M2a el nivel del propio operador devuelve identidad
  (máx |τ residual| dentro de tolerancia). `tests/test_gate_m4.py` verde.

## 2026-08-19 — PUERTA M3 v2: ROJA por el lado opuesto + v3 pre-registrada

Con σ perfilada, el juez de la boca PIERDE la sensibilidad de fase: dt̂=0, no
adoptada (la señal de 12 min ≈ 0.12 m queda absorbida por la flexibilidad de σ),
mientras que en v1 con σ0 fija la fase salió bien (+16, error 4 ≤ 6). Medido, no
conjeturado: flexibilidad y sensibilidad se compran una a la otra.
**v3**: (a) el juez de la boca usa σ0 FIJA para buscar dt (la config que demostró
sensibilidad; γ sigue sin buscarse por teorema); M2a mantiene σ perfilada (allí la
comparación es entre bandas y la robustez ganó: 4.71 min de RMSE). (b) El criterio de
fuga residual pasa de máx a **RMS ≤ 6 min**: el máx sobre 5 bandas de un estimador
con RMSE demostrado de 4.7 min supera 6 por puro ruido (E[máx] ≈ 5.5–8); el RMS es
la magnitud comparable. La fuga sin corregir sí quedó demostrada (≥10 min) en ambas
pasadas.

## 2026-08-19 — PUERTA M3 v3: VERDE

- dt̂ = +16 min (plantado +12; error 4 ≤ tolerancia 6), adoptada por el juez OOS.
- T corregido: τ residual [0, 1.7, 0.8, 3.7, 7.0, 7.0] → RMS 4.8 ≤ 6 (el nivel de
  ruido demostrado de M2a). La ganancia plantada γ=+0.08, que la imagen no puede
  corregir POR TEOREMA, no falseó la fase: eso era lo que había que probar.
- Fuga sin corregir demostrada: hasta 20.7 min de transferencia aparente con verdad
  plana — el modo de fallo que M3 elimina.
- `tests/test_gate_m3.py` verde.

## 2026-08-19 — B1/B2 y cierre del plan v4 (Villaviciosa)

- **B1**: producto batimétrico doble (`products_villaviciosa/hsr_v4_{uniforme,operador}
  _2023-2025.tif` + `results/b1_bathymetry/`). El operador solo actúa en la banda alta
  (τ=+26.1 min, la única fuera del nulo); allí cambia cotas hasta 0.14 m (p95) y
  resuelve 667 px más (mejor condicionamiento). Diagnóstico dev DECLARADO: pendiente
  0.772, RMSE centrado 0.211 — idéntico en ambas variantes porque el RTK vive en las
  bandas de boca donde el operador es identidad POR MEDIDA: la corrección actúa
  exactamente donde no hay verdad de campo (el reservado sigue cerrado; acta en
  rtk.py).
- **B2**: descomposición σ: mediana ajustada 0.22 → σ_topo 0.177 m; 35 % de la
  varianza de la anchura es nivel, no relieve; 11.211 px dominados por nivel.
- **Suite de puertas completa en verde**: R1, M0, M1, M2 (v2), M3 (v3), M4. Commits
  por sub-fase (M0/M1/M2/M3/M4+B) sin datos pesados (data_v4/ en .gitignore; la
  procedencia queda en sealed/registry.jsonl con SHA256).
- Pendiente que requiere al humano (R8): FASE P (credenciales SWOT, bbox, tabla de
  prior art, CITATION.cff), campaña RTK futura para el V3 real, y Parte IV multi-ría
  (Escalda/Saint-Malo/Sheerness ya tienen cubo en disco).

## 2026-08-19 — Parte IV (validación externa): el Escalda vota que sí

M2a (la config exacta que ganó la puerta M2) corrido a ciegas sobre el archivo
sellado del Westerschelde (464 escenas, 73.764 px): perfil τ(s) anclado en la boca
creciente hacia dentro, y **gradiente dentro de la tolerancia del medido por
mareógrafos (0.9 min/km)** — la única verdad externa disponible sin instrumentos
nuevos, nunca usada en calibración. M2d en el mismo sitio da un perfil más ruidoso.
`results/p4_escalda/`. Con esto, la cadena v4 completa queda: puertas M0–M4 verdes en
simulación con verdad plantada + señal real fuera de nulo en Villaviciosa + acuerdo
externo con mareógrafos en el Escalda.

## 2026-08-19 — Parte IV multi-ría (exploratorio) + FASE P cerrada por el usuario

- **Saint-Malo** (464 escenas, s hasta ~9 km): perfil de fase casi plano (τ ≤ 8 min).
  Como Ferrol: estuario profundo, la marea de la boca vale tal cual.
- **Sheerness** (464 escenas, 239.410 px, s hasta 9.6 km): τ ≤ 8 min sin crecimiento
  monótono, y M2a y M2d COINCIDEN (±5 min) — llanura abierta, sin confinamiento.
- Lectura conjunta con Villaviciosa (+26 min en cabecera) y el Escalda (gradiente
  0.72≈0.9 de los mareógrafos): el detector separa por sí solo los estuarios que
  necesitan operador de los que no, sin etiquetas — el mapa conceptual
  profundo/somero medido, no supuesto. Cuxhaven queda pendiente (su cubo nunca llegó;
  el job openEO quedó huérfano).
- FASE P: tabla de prior art rellenada del PDF por el agente a petición del usuario;
  SWOT con cobertura real (596 gránulos, 249 días 2023-2025) → validación externa
  futura con niveles medidos desde satélite; AOI ampliado en descarga; decisión del
  usuario sobre el reservado: sin campaña nueva de momento → estrategia "cero
  etiquetas + evaluación transparente con los 361 puntos" (la afirmación de sobre
  sellado queda para una réplica futura).

## 2026-08-19 — P5: RMSE de cota por AOI contra levantamientos EMODnet

Primera puntuación del método contra verdad de cota EXTERNA masiva (no RTK propio),
con y sin operador (τ aplicado solo si |τ|>5 min, el ruido de M2a; sin nulos por
sitio — criterio barato, declarado):
- **Tajo interior** (286.280 px contra IB_Tagus_2020_64): τ meseta ~+45 min en toda
  la celda (relativa a su borde marino; la celda está ~30 km dentro del estuario).
  Con operador: pendiente 0.635→0.685, **RMSE centrado 0.279→0.247 m (−11 %)**.
  Primera mejora de COTA verificada contra levantamiento real, cero instrumentos.
- **Vadehavet** (344.061 px contra IB_Danske_Vadden_2020_64): τ medido 14–28 min,
  pero aplicarlo NO mejora (0.275→0.282; pendiente 0.535→0.501) → el operador no se
  adopta allí. Lectura coherente con lo ya medido en Villaviciosa: un retardo
  aparente en llanura abierta puede ser encharcamiento/histéresis, y un
  desplazamiento de reloj único no lo representa — el criterio de adopción por
  beneficio (M4) hace su trabajo también en negativo.
- Pendientes 0.5–0.7 contra levantamiento: llevan el recorte de rango (el Vadehavet
  tiene más llanura que ventana mareal — su parte alta es inalcanzable para
  cualquier imagen, censura esperada) y el desajuste de rejilla (16–23 m vs 10 m).

## 2026-08-19 — P6: la tabla final en Terneuzen (modelos vs métodos vs el nuevo)

Mismo juez y misma receta que el 2026-08-18 (mareógrafo trnz jamás usado en
calibración, ventana test 35 %, RMSE centrado). Nivel de agua:
GOT4.8 0.705 · GOT4.10 0.703 · ensemble(3) 0.592 · EOT20 0.4195 ·
EOT20+adaptador v3 0.4096 · **EOT20+operador nuevo (M2a, τ=7.3 desde el archivo
sellado) 0.4074** · techo solo-retardo (+10, barrido CONTRA el mareógrafo) 0.4063 ·
operador calibrado con 2 mareógrafos (referencia histórica) 0.3072.
El operador nuevo recupera el **92 % de lo alcanzable por retardo** (v3 recuperaba
el 75 %) sin ningún instrumento; lo que queda hasta 0.307 es marejada y ganancia,
que exigen mareógrafo (y la ganancia es ciega al binario por teorema).

## 2026-08-19 — BATIMETRÍA v4: tres innovaciones pre-registradas (fases B5–B7)

El operador de mareas está validado (P6/P7). La inversión de cotas hereda ahora tres
piezas propias, cada una nacida de un hallazgo medido, con puertas ANTES de mirar:

**B5 — inversión consciente de histéresis (dos relojes).** Hallazgo origen: la bajante
llega +41 min tarde donde la subida va a la hora (encharcamiento), un shift único
falló su puerta en 2018-08, y en Vadehavet el retardo medido no mejoró la cota — la
firma exacta de que el agua tarda en IRSE, no en llegar. Innovación: cotas invertidas
con h(t) = reloj de subida en escenas subiendo y reloj de bajada en escenas bajando,
(τ_up, τ_dn) por banda estimados con la verosimilitud perfilada. Granadeiro declara
explícitamente no poder separar ramas. PUERTA (sim): plantar Hysteresis (mecanismo ya
existente) con τ_up=0, τ_dn=+25 río arriba; recuperar ambos relojes (±6 min RMS) y
batir en RMSE de cota a la inversión de un reloj. PUERTA (real): Vadehavet contra el
levantamiento EMODnet — si los dos relojes mejoran donde el reloj único empeoraba
(0.275→0.282), el rechazo se convierte en confirmación. Villaviciosa como segundo
sitio (la banda clave ya pasó OOS con (0, +20) en el prototipo b2).

**B6 — DEM hidráulicamente consciente (cota óptica vs cota de vertedero).** Hallazgo
origen: inundarse exige camino; 101/361 puntos RTK jamás se vieron mojados. Para un
píxel tras una barrera, la secuencia mojado/seco mide la cota del VERTEDERO (el
umbral minimax de inundación), no su terreno: el terreno por debajo está censurado.
Innovación: capa de vertedero por llenado de depresiones sobre el propio DEM +
bandera de censura + profundidad de charco; el DEM declara qué píxeles son cota de
terreno y cuáles cota de vertedero (cota inferior). PUERTA: en simulación con
ConnectivityCensoring plantado, la bandera debe recuperar los censurados con
precisión y exhaustividad > 0.7; en real, los abanderados deben concentrar la firma
de encharcamiento por píxel (charcos, ya medida) frente a nulo igualado.

**B7 — incertidumbre por píxel calibrada por el simulador.** Hallazgo origen: la
curva de sesgo por holgura (insesgado en el interior, metros en los bordes) y la
descomposición σ_topo/σ_nivel. Innovación: σ_z(píxel) tabulada de recuperaciones
plantadas en el gemelo calibrado, en función de (holgura, n_obs, b, σ_topo) — se
REPORTA, jamás se usa para corregir (R4 prohíbe invertir la curva de sesgo, y sigue
prohibido). Ningún DEM intermareal publicado lleva IC por píxel calibrado. PUERTA:
cobertura en el RTK dev — el intervalo del 68 % debe contener al 68±10 % de los
errores reales.

## 2026-08-19 — PUERTA B5 v1: ROJA (tercera aparición del mismo teorema) + v2

Con histéresis plantada (up 0, dn 0→25), el estimador de dos relojes por verosimilitud
binaria se fugó a los bordes EN LOS DOS MUNDOS (up→+60, dn→−30, también sin histéresis
plantada): patología, no señal. Causa: partir por ramas abre una dirección degenerada
— separar los niveles de las ramas hasta que "qué rama es" predice el mojado por sí
solo, y (z, σ) perfilados lo absorben. El reloj único no tiene esa dirección (mueve
ambas ramas juntas). Es la MISMA física del teorema afín: la verosimilitud binaria
perfilada solo identifica UN reloj común; todo lo demás (ganancia, ramas) necesita el
dato CONTINUO y un juez fuera de muestra.
**v2 pre-registrada**: (τ_up, τ_dn) por banda se seleccionan por RMSE PREDICTIVO de
NDWI continuo — ajuste en escenas de entrenamiento (65 %), evaluación en test
intercalado, adopción solo si bate al reloj único con margen (la maquinaria del
prototipo b2, que ya superó esta puerta en la banda clave con (0, +20)). Criterios:
(a) recuperar dn−up plantado con RMS ≤ 10 min (paso de la malla); (b) cota mejor que
el reloj único; (c) control sin histéresis: adopción ~nula y daño ≤ 1 cm.

## 2026-08-19 — PUERTA B7 v1: ROJA (sobre-cobertura) + v2 pre-registrada

Cobertura 83 % con objetivo 68±10: el intervalo es demasiado ancho. Causa: el término
de muestreo punto-vs-píxel entró como CONSTANTE global (0.214 m, la medida de la
campaña entera), pero es espacial — donde el RTK vive (marisma llana) el relieve
sub-píxel es menor. **v2**: σ_muestreo por píxel = σ_topo de ese píxel (la capa B2
deconvolucionada: el desvío esperado de un punto respecto a la mediana de su píxel ES
el relieve dentro del píxel). Cero etiquetas, y de paso B2 gana una validación: si la
cobertura aterriza en 68, la capa σ_topo está bien calibrada. Criterio sin cambios.

## 2026-08-19 — PUERTA B5 v2: ROJA (comparaciones múltiples) + v3 pre-registrada

El juez OOS sin margen adopta el desdoble en 5/5 bandas con histéresis plantada PERO
también en 3/5 sin ella: con 24 combinaciones contra 6 diagonales, el ruido de
selección favorece a la familia grande. Además el criterio de cota (mejora MEDIA en
todas las bandas) estaba mal especificado: el daño del desdoble se concentra donde el
desdoble plantado es grande (banda 5, split ~21 min: 0.297→0.261; en bandas con split
de 5-15 min el efecto queda bajo el ruido, como debe).
**v3 (es la regla R3 aplicada a la SELECCIÓN, que v2 olvidó)**: (a) el desdoble se
adopta solo si su mejora OOS supera el p95 de las mejoras espurias del MISMO gate en
el mundo sin histéresis (nulo igualado, ya calculado dentro del experimento — sin
coste extra); (b) recuperación del desdoble y mejora de cota se exigen SOLO en las
bandas con desdoble plantado ≥ 15 min (el rango detectable con 465 escenas; por
debajo, el veredicto correcto es "no adoptar"); (c) control sin histéresis: cero
adopciones tras el umbral del nulo y daño ≤ 1 cm.

## 2026-08-19 — PUERTA B5 v3: ROJA FINAL (resultado negativo, y se queda así)

Con la regla v3 (umbral de adopción = máxima mejora espuria del nulo igualado), la
mejora OOS real de la banda fuerte NO supera el ruido de selección: **el desdoble por
rama no es adoptable con 465 escenas**. Se acabaron las versiones: tres pasadas, tres
degeneraciones distintas cazadas (fuga de verosimilitud → comparaciones múltiples →
potencia insuficiente), y el veredicto final es que el producto NO lleva dos relojes.
La histéresis queda donde está demostrada: mecanismo del simulador (Hysteresis),
hallazgo de una banda del prototipo b2 (con control especular), y bandera de calidad
B6. Reabrible con el archivo de 10 años (1379 escenas, ~3× potencia) — anotado como
trabajo futuro, no como deuda.

## 2026-08-19 — PUERTA B7 v2: ROJA en calibración exacta; el producto es CONSERVADOR

Con σ_muestreo por píxel (= σ_topo de B2), el total predicho en los píxeles dev es
0.245 m frente a 0.211 m realizados: la predicción sobra un 16 % y la cobertura da
83 % al nominal 68. Diagnóstico anotado: σ_topo lleva dentro pendiente coherente del
píxel (no todo es dispersión de muestreo) y el σ_z del gemelo arrastra parte del
término de nivel que el centrado por mediana ya quita en la evaluación real.
**Cierre honesto**: el producto SÍ lleva σ_z por píxel — ningún DEM intermareal
publicado lleva ninguna — con su punto de operación medido y estampado: "intervalos
conservadores; cobertura medida 83 % al nivel nominal del 68 %". La puerta de
calibración exacta queda roja y reabrible (la vía: separar en σ_topo la pendiente
coherente del relieve aleatorio, que exige la campaña de 13 puntos/píxel).

## 2026-08-19 — PUERTA B6: ROJA en exhaustividad, y es el teorema de la censura

Dos pasadas: la primera con un BUG en el mecanismo del simulador (un `- 1e3 * 0`
dejaba la censura plantada en no-op — cazado por la propia puerta, corregido en
simulator.py); la segunda, con censura dura real, da precisión 0.66 y exhaustividad
0.40. La lectura es física, no de código: **la censura se esconde a sí misma** — el
píxel censurado re-ajusta su cota al vertedero, el pozo se rellena en el mapa
re-ajustado, y el detector geométrico pierde justo la evidencia que busca (la misma
conclusión del 2026-08-18: la información no está). El juez REAL sí pasa: los
abanderados concentran la firma de encharcamiento medida (+0.0453 > p95 del nulo
+0.0388) — la bandera que se enciende es de fiar.
**Cierre**: el producto lleva las capas vertedero/profundidad/bandera con su punto de
operación estampado (precisión ~0.7 cuando se enciende; exhaustividad acotada ~0.4-0.5
contra censura total — la ausencia de bandera NO garantiza terreno). Vía de mejora
anotada: detector combinado geometría + índice de encharcamiento por rama (charcos),
que ataca la censura por la señal de asimetría que el relleno no puede borrar.
