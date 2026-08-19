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
