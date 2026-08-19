# Prior art — Granadeiro et al. 2021 (P3)

Paper: *Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and Exposure
Periods over Large Intertidal Areas*, Remote Sensing 13(2):320, doi:10.3390/rs13020320.

**Instrucciones:** esta tabla la rellena el humano leyendo el paper (el agente no debe
citarlo de memoria). Es el listón de M2d y la referencia obligada del paper 1. Rellenar
TODOS los campos; donde el paper no diga nada, escribir «no lo trata» — esa columna es
la lista de nuestros deltas.

| campo | ellos (Granadeiro 2021) | nosotros (v4) |
|---|---|---|
| Sitio(s) y extensión | _por rellenar_ | Villaviciosa (3.5 km) + Escalda (dev con verdad) + 116 celdas costa norte |
| Qué estima exactamente (topografía / retardos / exposición) | _por rellenar_ | topografía 10 m + T(A_k, φ_subida, φ_bajada por tramo) + h̄(s) del canal |
| Cómo estima el retardo (¿minimización de discrepancia flujo/reflujo? ¿por zonas?) | _por rellenar_ | verosimilitud perfilada con juez fuera de muestra + control especular |
| ¿Retardo simétrico entre ramas o histéresis? | _por rellenar_ | histéresis medida: (τ_subida, τ_bajada) separados por banda |
| ¿Estima amplificación/ganancia A(s)? | _por rellenar_ | sí, juzgada por el dato (banda alta Villaviciosa: A=0.85 pasa la puerta) |
| Resolución espacial del retardo (zonas/píxel) | _por rellenar_ | bandas por cuantiles de distancia de canal |
| Precisión declarada del retardo y contra qué verdad | _por rellenar_ | ±4 min (calibración sintética); Escalda contra mareógrafos: gradiente 0.9 min/km coincide |
| Validación fuera de muestra (train/test temporal): ¿sí/no? | _por rellenar_ | sí, 65/35 temporal, test abierto una vez |
| Nulos / controles estadísticos | _por rellenar_ | control especular, nulos igualados en ruido, permutación por bloques |
| Umbral mojado/seco y sensor de agua | _por rellenar_ | NDWI>0 (binario) y sigmoide continuo (perfilado) |
| Marea de contorno usada (modelo/mareógrafo) | _por rellenar_ | BoundaryProvider intercambiable; EOT20 medido mejor en estuario (0.22 vs 0.32 GOT) |
| ¿Tratan hora real de paso vs hora nominal? | _por rellenar_ | sí: 11:21 UTC real vía STAC (+21 min vs nominal) |
| ¿Alias de constituyentes (S2 invisible)? | _por rellenar_ | tabla de alias explícita; S2 clavado al contorno |
| ¿Batimetría del canal desde celeridad? | _por rellenar_ | h̄(s) = c²/g como subproducto; test de variedad hidráulica |
| ¿Validación punto-vs-píxel (representatividad del RTK)? | _por rellenar_ | sí: σ_e=0.214 m; c=1.041; `point_sampling_error` |
| Qué NO estima / limitaciones que admiten | _por rellenar_ | — |
