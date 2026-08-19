# STORY — el hilo narrativo del método (para paper, póster y repo)

Este documento es el argumento científico en orden lógico, no cronológico. Cada
afirmación lleva su evidencia y su artefacto reproducible. El texto final de los papers
lo escribe el humano; esto es el esqueleto que no debe perderse.

## 1. El problema, en una frase

Un archivo de satélite óptico convierte la marea en un escáner del relieve intermareal:
cada píxel se moja cuando el agua supera su cota, así que 465 fotos fechadas son 465
preguntas de sí/no sobre cada punto del fango. El método estándar (sigmoide por píxel
contra la marea de un modelo oceánico) es *prior art* — Catalão & Nico 2017, Bué 2020,
Granadeiro 2021, Chen & Wang 2025 — y este proyecto NO reclama su invención.

## 2. Lo que la literatura reporta y nadie explicaba: la compresión

Todos los métodos recuperan el relieve *comprimido* (pendiente contra terreno < 1;
aquí: 0.73–0.78 según partición). La sospecha universal es «la marea dentro del
estuario no es la del modelo». Este proyecto la persiguió con disciplina de nulos y el
resultado es la columna vertebral del paper:

| hipótesis sobre la compresión | veredicto | evidencia |
|---|---|---|
| lámina inclinada (3 variantes) | muerta | nulos sintéticos; explica ≤1.5 % |
| marejada (barómetro inverso) | muerta | no supera a marejada barajada |
| censura del techo de marea | retractada | análisis circular (holgura ≡ −cota) |
| deriva morfológica (2 años imagen-campaña) | muerta | 3 épocas de igual n, IC cruza 0 |
| dilución por regresión | muerta | el ruido está en la respuesta; nulo maestro: pendiente 1.019 |
| **muestreo punto-contra-píxel de la validación** | **CONFIRMADA** | emparejado k=1 vs k=2: c = 1.041 [0.79, 1.58]; σ_e = 0.214 m por 2 rutas independientes |

**Tesis 1 (paper de validación):** buena parte de la «compresión» que reporta esta
literatura es un artefacto de comparar un punto de GNSS con la mediana de un píxel de
10 m. Del error reportado contra RTK, el 61–88 % era la referencia, no el método.
Herramienta citable: `pyintertidal.validation.point_sampling_error`. Receta de campaña
que lo arregla: ~18 puntos por píxel.

## 3. El mareógrafo distribuido: la marea interior desde el propio archivo

Aun así la marea interior difiere de la oceánica, y es medible SIN instrumentos
tratando los píxeles como mareógrafos binarios (la senda que abrió Granadeiro 2021 con
retardos cotidales; tabla ellos/nosotros en `docs/prior_art_granadeiro.md`).

Nuestros deltas sobre ese listón, cada uno con puerta pre-registrada:

1. **Histéresis por ramas** (τ_subida ≠ τ_bajada por tramo): el encharcamiento se mide,
   no se asume. En Villaviciosa la banda alta pasa la puerta fuera de muestra con
   (τ_s=0, τ_b=+20 min) y control especular en contra.
2. **Amplitud juzgada por el dato**: A=0.85 (amortiguación) aporta donde hay física y
   es dirección plana donde no — el juez es la verosimilitud en escenas retenidas.
3. **Validación externa con mareógrafos que el método nunca ve**: Escalda — gradiente
   de retardo de la imagen 0.9 min/km = el de los mareógrafos; recupera el 75 % de la
   corrección de nivel alcanzable en Terneuzen con cero instrumentos.
4. **Tabla de alias explícita**: S2 (constituyente solar) es invisible por construcción
   para un sensor heliosíncrono; M2/N2/O1/Q1/M4/M6 estimables. Límite duro del campo.
5. **Batimetría del canal de regalo**: h̄(s) = c²/g desde la celeridad del retardo, con
   test de variedad hidráulica (Escalda: 35 m, plausible; donde da valores imposibles,
   eso ES el mapa de charco, y se reporta como tal).

## 4. El tribunal: por qué esto es creíble

Nada se afirma contra cero; todo se afirma contra un **nulo igualado** fabricado por un
simulador calibrado desde el archivo (marea real, píxeles remuestreados enteros, ruido
por píxel + sistemático de escena + incertidumbre de nivel deconvuelta). El nulo
reproduce las marginales del archivo (puerta M0.2) y es insesgado en el interior de la
ventana de marea — y reproduce el sesgo de borde real del estimador, que es exactamente
lo que un tribunal debe hacer. Dos resultados con p≈1e-37 murieron hoy contra nulos
bien igualados: esa es la razón de existir de esta arquitectura.

**Tesis 2 (paper del método):** corrección por mecanismo del nivel interior + el marco
de validación (puertas fuera de muestra, controles especulares, sellado de predicciones,
reservado intocable) como contribución metodológica reutilizable.

## 5. Cero etiquetas, y por qué

Entrenar un corrector con 64 etiquetas RTK es significativamente PEOR (−0.038 m,
IC95 [−0.065, −0.014]) que entrenarlo con cero etiquetas sobre el simulador calibrado.
El diseño entero es transferible a costas sin campañas: ese es el argumento de escala
(116 celdas de la costa norte, Parte IV).

## 6. Reproducibilidad práctica

- datos sellados (`sealed/registry.jsonl`, SHA256+fecha, append-only);
- particiones deterministas (semillas en `configs/`), reservado con guard R1;
- cada puerta es un test (`tests/test_gate_*.py`); cada experimento un script con
  config y resultados versionados por hash;
- sin nada cableado: umbrales y tolerancias viven en `configs/*.yaml` con su porqué.

## Tesis 3 (fase M2): la marea interior se puede FECHAR desde el archivo, pero no ESCALAR

Lo que un archivo binario de mojado/seco puede y no puede medir de la marea interior,
demostrado con verdad plantada (no argumentado):

- **La fase sí.** Cuatro estimadores independientes — verosimilitud Rasch (M2a),
  concordancia por píxel (M2b), rangos de área inundada (M2c) y el método de
  literatura de discrepancia flujo/reflujo (M2d, Granadeiro 2021) — recuperan un
  perfil de retardo plantado de 0→40 min con errores de 5–13 min, bajo muestreo
  temporal real, nubes reales y ruido calibrado del archivo.
- **La amplitud no, y es un teorema, no una limitación de implementación.** La
  verosimilitud de respuestas binarias es exactamente invariante bajo
  (α, z_p, σ_p) → (c·α, c·z_p, c·σ_p): la ganancia por banda es indistinguible de un
  reescalado de las cotas y anchuras locales. La puerta v1 lo hizo visible del modo
  más instructivo: el optimizador gastó α en compensar σ mal especificada y devolvió
  el límite de la caja. La puerta v2 exige exponer la curva NLL(α) plana en vez de
  reportar un número inventado.
- **Consecuencia de diseño**: el operador T corrige fase (identificable, validada
  externamente en el Escalda); la amplitud solo entra donde el dato CONTINUO la
  respalda fuera de muestra (el juez OOS del método b2: A=0.85 en la única banda
  donde añade poder predictivo). Los métodos publicados que reportan ganancias
  mareales desde waterlines binarias deberían pasar por esta misma puerta.

## Tesis 4 (fases M3–M4): el error del contorno se corrige en fase, se acota en ganancia, y el operador muerde

- **La fuga, demostrada antes de evitada**: +12 min de error de fase en el modelo
  oceánico de la boca se disfrazan de "transferencia estuarina" distribuida (hasta
  +21 min aparentes con verdad plana). Cualquier trabajo que mida transferencias
  contra un modelo global sin auditar su fase en la boca puede estar publicando el
  error del modelo como física del estuario.
- **La auditoría es asimétrica, y eso es un resultado**: la FASE del contorno se
  recupera desde la propia imagen (dt̂ con error de 4 min, juez OOS en la boca); la
  GANANCIA no puede auditarse desde binario (teorema afín, medido dos veces) y queda
  acotada por el prior del ensemble (la dispersión entre modelos oceánicos
  independientes). Imagen audita fase; ensemble audita ganancia.
- **El operador final es una composición inspeccionable** (`InteriorTide.describe()`):
  contorno (cualquier proveedor) + corrección de fase de la boca + perfil τ(s) medido
  por verosimilitud con ancla en la boca — y solo lleva señal donde 5 réplicas del
  nulo uniforme no la explican. En la puerta M4 recupera el daño plantado (RMSE de
  cota 0.44→0.37 m en la banda alta) sin tocar la boca y siendo punto fijo.
- **En Villaviciosa real**: el interior va a la hora de EOT20 hasta ~5.4 km (τ dentro
  del nulo, donde vive el RTK) y llega +26 min tarde a 7.7 km — decenas de cm de
  nivel mal asignado por escena en la cabecera, ahora corregidos por un operador que
  no vio ni un mareógrafo ni una etiqueta.
