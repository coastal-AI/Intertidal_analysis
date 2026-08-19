# INVENTORY — activos existentes contra las fases del plan v4

Fecha: 2026-08-18. Estado real verificado en disco, no de memoria.
Leyenda: **PAQ** = en el paquete `pyintertidal/` (producción) · **SCR** = prototipo en
scratchpad de sesión (directorio temporal, **se pierde si no se porta**) ·
**HECHO** = resultado ya obtenido hoy, con números · **FALTA** = no existe.

## 0. Datos en disco

| activo | ruta | estado |
|---|---|---|
| Cubo Villaviciosa 10 años (B03/B08/SCL, 10 m, 1379 t, 915×915) | `ndwi_cube_villaviciosa_grande_10y.nc` (3.09 GB) | PAQ |
| Cubo Villaviciosa v2 (B03/**B04**/B08/**B11**/SCL, 10 m, 465 t, **967×915**) | `swir_cube_villaviciosa_2023-2025.nc` (1.85 GB) | PAQ — ver nota rejilla |
| Cubo Escalda 20 m (464 t, 734×1051) | `ndwi_cube_escalda_2023-2025_20m.nc` (1.13 GB) | PAQ |
| Cubos Saint-Malo / Sheerness 20 m | `ndwi_cube_{stmalo,sheerness}_2023-2025_20m.nc` | PAQ |
| Cubo Cuxhaven | job openEO `j-26081815190240959d…` en cola al morir la sesión | FALTA re-adjuntar |
| Cubos Tejo / Vadehavet / Santander (épocas antiguas) | `ndwi_cube_{tejo,vadehavet,santander}_*.nc` | PAQ (Parte IV) |
| RTK Villaviciosa crudo | `data/villaviciosa_rtk_gnss.csv` — 361 FIX, 248 px, 54 px con 2 puntos, 23 con 3+ | PAQ |
| Series intermareales extraídas (Y, C, keep, fechas) | `scratchpad/ndwi_intermareal.npz` (175 MB), `swir_intermareal.npz` (148 MB), `escalda_intermareal.npz` | **SCR — portar** |
| Distancia geodésica por canal + máscaras mar/alcanzable | `scratchpad/canal.npz` | **SCR — portar** |
| Mareógrafos IOC en caché (bon2, sev2, fer1/2, vlis, trnz, tdan/tdas, goer/gohi, hono, keehi, barc/2, leha, diep…) | `scratchpad/ioc_*.json` | **SCR — portar** |
| Horas reales de paso S2 (STAC; el cubo solo guarda FECHA) | vía `pyintertidal.overpass.get_overpass_times` | PAQ |

**Nota rejilla (institucional, costó dos incidentes):** el cubo v2 bajó en 967×915; contiene
la rejilla canónica 915×915 EXACTA con desplazamiento de 26 filas en y (verificado);
recortar `[26:941, 0:915]`. Todo ingest debe `assert` contra la rejilla canónica —
un desajuste silencioso invalidó una mañana entera de resultados de ML.

## 1. Paquete `pyintertidal/` → arquitectura §4

| módulo existente | destino §4 | notas |
|---|---|---|
| `cube.py` (SentinelCube, `extra_bands`, `last_job_id`) | data/ | descarga openEO; CDSE limita a **1 conexión** |
| `scenes.py` (connect), `overpass.py` (horas STAC) | data/ | paso real = 11:21 UTC, no 11:00 |
| `water.py`, `stability.py`, `frequency.py` | data/, optics/ | detectores e intermareal |
| `elevation.py` — `_fit_block` (forma cerrada), `fit_hsr`, `fit_step`, `extract_intertidal_ndwi` | inference/ | baseline §1; **guardas de amplitud** `0.15<b<2.5`, `|a|<2.5` aprendidas hoy (predicción congelada explota sin ellas) |
| `tides.py`, `tidemodels.py`, `tidecheck.py` (nearest_ocean_km, phase_lag, range_dependence) | tide/ | |
| **`boundary.py`** — BoundaryProvider, PyTMDBoundary, GaugeBoundary, EnsembleBoundary, `fit_harmonics`, `alias_periods`, **EPOCH fijo** | tide/ | **M3 casi entera ya existe**; falta ClimatologyBoundary y la corrección (1+γ_k, Δφ_k) con priores. BUG resuelto: fases ancladas a época fija (referirlas al inicio de la serie desfasa entre ventanas) |
| **`estuary.py`** — EstuaryTransfer (`from_gauges` VALIDADO, `from_geometry` SIN VALIDAR, `residual_gain` medido, `validate_against_gauge` con 3 términos) | tide/ | el 3er término (solo-armónicos) es obligatorio: sin él se atribuyó al operador un 96 % de mejora que era filtrado |
| `validation.py` — `compare_dems`, `transect_profile`, **`point_sampling_error`** | stats/ | descomposición punto-vs-píxel (ver §3 HECHO) |
| `gauges.py`, `hypsometry.py`, `mosaic.py`, `shoreline.py`, `aoi.py` | geometry/, data/ | M1 y Parte IV |
| `campaign_north_coast.ipynb` (116 celdas; `NameError` de `rows` corregido) | Parte IV | |

## 2. Prototipos en scratchpad → portar (se pierden con la sesión)

| prototipo | destino v4 | qué hace |
|---|---|---|
| `sesgo.py` / `modelo.py::simulate_training_set` | simulator/ | **el tribunal**: marea real + (a,b,σ,ruido) remuestreados por píxel ENTERO + sistemático de escena + cotas plantadas. Falta API de mecanismos enchufables |
| `lamina.py::flood_threshold` | optics/ + simulator/ | censura por conectividad (minimax por reconstrucción morfológica; probado en sintético) |
| `nulo_bloques.py` | stats/ | **NULO MAESTRO B1 — YA EJECUTADO** (ver §3) |
| `atenuacion.py`, `varianzas.py` | stats/ | desatenuación k=1 vs k=2; σ_e por dos rutas independientes |
| `forma_cx.py` | stats/ | regresión de c(x) con interacciones contra nulo; **sellada ad-hoc sha256 `adadcb0ed33d3f39`** — migrar a `seal.py` |
| `retardo_imagen.py`, `escalda_valida.py` (lag_for, barrido h(t−τ), calibración sintética) | tide/ (base de M2c/M2d) | retardo por rangos de área inundada; precisión ±4 min calibrada |
| `metodo_b.py`, `metodo_b2.py` | tide/ (pariente de M2a) | warp (A, τ_subida, τ_bajada) por banda, selección en train, juez fuera de muestra + control especular. Perfila z con `_fit_block` (gaussiano; M2a pide Bernoulli — cambio menor) |
| `charcos.py` | optics/ | índice de encharcamiento por píxel a niveles igualados, con nulo por permutación de ramas |
| `doble_indice.py` | optics/ B2 | δ̂ = ẑ_MNDWI − ẑ_NDWI con nulo de dos-ajustes-de-la-misma-verdad |
| `busca_mareografos.py`, `pares_mundo.py`, `curva_estuarios.py`, `vs_modelos.py`, `contornos.py` | tide/, Parte IV | admitancia multi-estuario IOC; comparación EOT20/GOT/ensemble/operador contra 17 mareógrafos |
| `hora_paso.py` | tide/alias_table.py | **tabla de alias ya calculada** (ver §3) |

## 3. Resultados ya obtenidos que pre-cargan puertas del v4

| puerta v4 | estado | números |
|---|---|---|
| M0 (estructura por bloques) | números de referencia medidos | dev 0.698 [0.608–0.782]; ~~reservado 0.890~~ ⚠ ver pregunta bloqueante Q1: reproducir 0.890 exige leer el reservado y violaría R1 |
| M2 alias_table | **HECHO** | estimables M2 (14.8 d), N2 (9.6), O1 (14.2), Q1, M4, M6; **S2 invisible por construcción**; K1/P1 alias anual |
| M2d (Granadeiro) | pariente ya validado externamente | retardo desde imagen: Villaviciosa perfil 0→+74 min (±4 min calibrado); **Escalda: gradiente 0.9 min/km = mareógrafos; recupera el 75 % de la corrección alcanzable contra Terneuzen** |
| M2a (parcial) | prototipo con puerta pasada | banda 0.98–3.53 km: warp (A=0.85, τ_s=0, τ_b=+20) **PASA fuera de muestra con control especular** (0.1996→0.1924; control 0.2112). Histéresis medida, no impuesta |
| M3 (contorno) | módulos hechos + medición | EOT20/GOT4.10/GOT4.8 contra 17 mareógrafos: en costa empatan (~0.075 m); dentro de estuario EOT20 0.222 gana a GOT 0.32 y **al ensemble de series (interferencia destructiva de fases — promediar armónicos, no series)** |
| B1 (nulo de dilución) | **HECHO — puerta cerrada** | estimador insesgado con compresión plantada 0: pendiente 1.019, ninguna de 200 réplicas baja al 0.561 observado; I² ingenuo inflado (nulo solo ya da 40 %, máx 85 % vs 86 % observado). Dilución clásica descartada (ruido en la respuesta, no en la predictora) |
| B1 (regresión de c(x)) | **HECHO — sellado** | ninguna covariable disponible modula c(x) (todas dentro del nulo); solo la compresión global (z=−3.48). sha `adadcb0ed33d3f39` |
| B2 (doble índice) | **HECHO** | δ real (1.44× el nulo, MNDWI −0.120 m vs NDWI) pero **no predice el error de campo** (r=+0.010) ni sigue turbidez (+0.066) |
| — (no en v4, afecta a B1/V3) | **HECHO** | punto-vs-píxel: σ_e=0.214 m por dos rutas independientes coincidentes; **c verdadera = 1.041 [0.79, 1.58]**; del error reportado, 61–88 % era la referencia. `validation.point_sampling_error` en el paquete |

## 4. Ausencias netas (construir de cero)

`seal.py` y `sealed/` · guard R1 con `allow_reserved` + test de imports · ingest zarr
particionado por bloque · layout `src/` + configs YAML + `tests/test_gate_*` ·
M2a Bernoulli completo (armónica en t × spline en s, L-BFGS-B ~20–40 params; hoy solo
existe la versión por-bandas con warp de 3 parámetros) · M2b cópulas · M2c waterline
diferencial · EDO Toffolon–Savenije (M4 prior) · EM conjunto B3 · P1 SWOT ·
ClimatologyBoundary · corrección de contorno con priores.

## 5. Memoria institucional (errores que ya costaron horas — no repetir)

1. Rejillas: `assert` siempre; 3 incidentes hoy (887×673, 967×915, 588×1852).
2. El cubo guarda **fecha, no hora**: unir horas STAC; el paso real es 11:21±10 UTC.
3. Fases armónicas: referencia a **EPOCH fijo**, jamás al inicio de la serie.
4. `harmonics(constituents, ...)` posicional: el orden de parámetros es contrato.
5. Amplitudes (a, b) sin acotar explotan en predicción congelada.
6. CDSE: 1 conexión; descargas en serie; float32 (OOM con 0.47 GB libres mató una corrida).
7. p ingenuos inválidos (píxeles autocorrelados): dos resultados "p=1e-37" murieron hoy contra nulos igualados.
8. Comparar operadores contra la serie cruda regala el filtrado: 3 términos siempre.

## Añadido en la fase M2–M4 (2026-08-19)

Módulos nuevos del paquete:
- `alias_table.py` — decisión de constituyentes con horas de paso reales.
- `tide_estimators.py` — ShiftBank, bandas por cuantiles, M2a (Rasch solo-fase,
  σ perfilada), M2b (concordancia), M2c (waterline por rangos), M2d (Granadeiro),
  `alpha_nll_profile` (el medidor de la degeneración afín, con mallas que escalan).
- `boundary_correction.py` — descomposición armónica del contorno, corrección
  (1+γ_k, dt_k) solo en estimables, `ClimatologyBoundary`, juez OOS de la boca.
- `operator.py` — `InteriorTide`, el mareógrafo distribuido con `describe()`.

Experimentos (cada uno escribe result.json con SHA de entradas + figura):
- `m2_alias`, `m2_gate_sim` (verdad plantada), `m2_real` (archivo real vs nulo
  uniforme), `m3_gate_sim` (error de contorno plantado), `m4_gate_sim` (beneficio +
  contracción), `b1_bathymetry` (producto con/sin operador + diagnóstico dev
  DECLARADO + GeoTIFFs), `b2_sigma` (descomposición σ_topo/σ_nivel),
  `v3_open_reserved` (solo humano, doble salvaguarda).

Tests de puerta: `test_gate_m2/m3/m4` sobre los result.json (más los previos M0/M1
y el guard R1).
