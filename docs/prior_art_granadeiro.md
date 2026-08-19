# Prior art — Granadeiro et al. 2021 (P3)

Paper: *Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and Exposure
Periods over Large Intertidal Areas*, Remote Sensing 13(2):320, doi:10.3390/rs13020320.

Rellenada el 2026-08-19 leyendo el PDF completo (no de memoria). Donde el paper no
trata algo se escribe «no lo trata» — esas filas son la lista de nuestros deltas.

| campo | ellos (Granadeiro 2021) | nosotros (v4) |
|---|---|---|
| Sitio(s) y extensión | Archipiélago de Bijagós (Guinea-Bissau), ~1200 km² de intermareal repartidos en ~100 km; régimen semidiurno, vivas 0.3–4.8 m | Villaviciosa (ría de 3.5 km, caso confinado) + Escalda (validación con mareógrafos) + celdas costa norte definidas |
| Nº de escenas y preprocesado | 35 escenas L1C (2017–18 y 2019–20, nubes <10 %), corrección atmosférica ACOLITE, e inter-calibración radiométrica entre escenas por regresión de eje mayor sobre píxeles estables (agua NIR<0.05, tierra >0.2) | 465–1379 escenas L2A por sitio (todo el archivo útil, no una selección); el sistemático de escena entera se mide y entra en el nulo (sd_scene), no se corrige por regresión |
| Qué estima exactamente | Topografía 10 m (logística de 4 parámetros sobre NIR), retardos cotidales, y mapas de periodo de exposición (Ec. 5 sinusoidal) | Topografía 10 m + perfil de fase τ(s) del interior + corrección de fase del contorno + descomposición σ_topo/σ_nivel; la exposición no es objetivo |
| Cómo estima el retardo | Ajustes logísticos SEPARADOS con escenas de subida y de bajada; barrido de retardos −90..+90 min (paso 5); se elige el que minimiza la diferencia de cotas subida-vs-bajada; en 50.000 píxeles aleatorios de cota media (2.47±0.25 m); luego un GAM (splines, lon/lat) suaviza el campo | Reimplementado como listón (M2d). Nuestro motor (M2a) es distinto: verosimilitud Bernoulli con cotas y anchuras PERFILADAS por píxel, ancla en la boca, todos los píxeles (no solo cota media) — bate a M2d en la puerta con verdad plantada (4.71 vs 5.42 min) |
| ¿Retardo simétrico entre ramas o histéresis? | Simétrico por construcción (un solo retardo que iguala ambas ramas); admiten como limitación no poder separar subida/bajada ni vivas/muertas | Histéresis medida por separado (bajada +41 min donde subida ~0 en el prototipo; mecanismo Hysteresis en el simulador); el operador admite dos relojes |
| ¿Estima amplificación/ganancia A(s)? | No lo trata (asume la amplitud del punto de referencia en todo el archipiélago) | Demostramos que de mojado/seco NO puede estimarse (teorema afín, puerta M2 v2 con NLL(α) plano) — tampoco ellos podrían; solo el dato continuo la respalda fuera de muestra donde hay física (A=0.85 en la banda clave) |
| Nivel de agua de referencia | Tablas de marea del puerto de Bubaque (un único punto, del Instituto Hidrográfico portugués, base de los 60) + interpolación cosenoidal entre pleamar y bajamar (Ec. 1) | BoundaryProvider intercambiable (modelo global pyTMD, mareógrafo, ensemble, climatología armónica) + corrección de fase auditada desde la propia imagen (M3) |
| ¿Tratan hora real de paso vs nominal? | Usan la hora exacta de adquisición en la interpolación, pero no lo discuten como fuente de error | Medido: 0.134 m de error por usar la hora nominal; horas reales vía STAC para todo |
| ¿Alias de constituyentes (S2 invisible)? | No lo trata (con tablas de marea no hay armónicos que estimar) | Tabla de alias explícita con las horas reales: S2 congelado, K1/P1 anuales, M2/N2/O1/Q1/M4/M6 estimables |
| Resolución espacial del retardo | 50.000 píxeles muestreados → campo suave lon/lat por GAM (líneas cotidales) | Bandas por cuantiles de la coordenada geodésica desde la boca (la geometría del estuario, no lon/lat); el confinamiento de una ría es justo el caso donde lon/lat no sirve |
| Precisión declarada del retardo y contra qué verdad | Diferencia media absoluta 6.6 min contra 4 puntos del Instituto Hidrográfico (máx 15 min en Abú); referencia posiblemente desactualizada (obs. de los 60) | Con verdad PLANTADA y muestreo real: 4.71 min (M2a); externa: gradiente 0.72 min/km contra 0.9 de los mareógrafos del Escalda |
| Validación fuera de muestra (train/test) | No lo trata (sin partición temporal; validan el producto final de exposición contra 66 cámaras time-lapse, r²=0.94) | Sí: OOS temporal 65/35 en el juez de amplitud; juez OOS intercalado en M3; test abierto una vez |
| Nulos / controles estadísticos | No lo trata (sin nulos; la validación de campo es directa) | Todo veredicto contra nulo igualado: simulador calibrado (píxeles enteros + sistemático de escena + jitter de nivel), banda nula de 5 réplicas, controles de signo/especulares |
| Umbral mojado/seco y sensor de agua | Logística continua sobre reflectancia NIR; el NDWI solo para DELIMITAR el intermareal (sd temporal > 0.2) | NDWI continuo para el sigmoide (con σ en metros interpretable como relieve sub-píxel) y binario NDWI>0 para los estimadores de fase |
| ¿Censura por conectividad / encharcamiento? | Lo observan como sesgo (película de agua y charcos en fango bajo inflan la exposición a cotas <2.2 m) pero no lo modelan | Mecanismo ConnectivityCensoring en el simulador; el encharcamiento es parte del modelo de histéresis, no solo una disculpa |
| ¿Batimetría del canal desde celeridad? | No lo trata | h̄(s) = c²/g como subproducto del perfil de fase; test de variedad hidráulica (Escalda 18.5 m/s → canal dragado plausible) |
| ¿Validación punto-vs-píxel del terreno? | No lo trata (sus cámaras van por GPS de ~4 m y validan exposición, no cotas) | Sí: σ_muestreo=0.214 m medido por dos rutas independientes; c=1.041 — la "compresión" de esta literatura es artefacto de comparar punto con mediana de píxel; `point_sampling_error` en el paquete |
| Qué NO estima / limitaciones que admiten | Rango de cotas limitado por las mareas observadas (1.04–4.69 m); no ve lo más bajo; sesgo por fango húmedo; retardo promedia ramas y vivas/muertas; referencia de marea antigua | Ganancia no estimable de binario (teorema, expuesto); amplitud solo donde el OOS continuo la respalde; reservado RTK actual quemado para el veredicto sellado final |

**Síntesis para la introducción del paper**: Granadeiro et al. demostraron que el
archivo S2 contiene los retardos cotidales y los extrajeron con un método
simétrico, suavizado en lon/lat y validado contra tablas antiguas. Nuestros deltas:
(1) motor de verosimilitud que bate al suyo con verdad plantada y muestreo real;
(2) histéresis (dos relojes) en vez de retardo único; (3) geometría de estuario
(s desde la boca) en vez de lon/lat; (4) teorema de qué NO puede estimarse
(ganancia) y su demostración; (5) nulos igualados y puertas pre-registradas para
cada afirmación; (6) contorno intercambiable con corrección de fase auditada;
(7) marco de validación punto-vs-píxel que explica la "compresión" publicada.
