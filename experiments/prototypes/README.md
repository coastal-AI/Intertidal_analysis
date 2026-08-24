# Prototypes — the exploratory record (pre-v4)

Every script in this folder is preserved **verbatim** from the exploratory
phase of the project (August 2026, before the v4 gate methodology). Names
and comments are in Spanish because that is how they were written; nothing
has been edited, because their value is exactly that they are the record:
what was tried, what worked, what failed, and in which order. The polished,
gated versions of everything that survived live one level up in
`experiments/` and in `pyintertidal/`; the failures are documented in
`docs/PHASE_LOG.md` and `research/06_negative_results.ipynb`.

A map of the main families:

**The tide adapter (what became MAREA)**
`retardo_imagen.py` (the first image-only lag estimator),
`retardo_adaptativo.py`, `desfase*.py`, `escalda_valida.py` /
`extrae_escalda.py` / `descarga_escalda.py` (the Scheldt external
validation), `guadalquivir.py`, `pares_costeros.py` / `pares_mundo.py`
(the 13 worldwide gauge pairs), `busca_mareografos.py`, `vs_modelos.py`,
`ensemble_tide.py` / `ensemble_fit.py`, `eot20*.py`, `marea_cmp.py`.

**The compression investigation (what became the point-vs-pixel finding)**
`sesgo.py`, `atenuacion.py`, `artefacto.py`, `campo.py`, `forma_cx.py`,
`pendiente_confundente.py`, `nulo_bloques.py`, `varianzas.py`,
`vivas_muertas.py`, `deriva.py`, `marejada.py`, `techo.py`, `censura.py`,
`hora_paso.py`, `doble_indice.py`.

**Water-surface estimation attempts (retired with controls)**
`lamina.py` (the tilted-sheet solver), `mareografo.py` (the per-date free
level that the affine theorem killed), `metodo_b.py` / `metodo_b2.py`,
`nivel_desde_imagenes.py`, `charcos.py` (the per-pixel ponding index),
`histeresis.py`, `warp.py`, `canal.py` (geodesic channel distance —
promoted into `pyintertidal.geometry`).

**Elevation estimators and baselines**
`hsr_prototype.py`, `hsr_v3.py`, `hsr_epoca.py`, `dea_fiel.py` (the
faithful DEA replica), `bathymetry_orig.py` (the original dictionary
code), `ml_honesto.py` / `ml_ricas.py` / `ml_vs_hsr.py` (the label-using
baselines the zero-label method is measured against), `test_step_siq.py`.

**The simulator (what became `pyintertidal.simulator`)**
`modelo.py`, `sigma.py`, `prueba_sigma.py`, `snr.py`, `repetibilidad.py`.

**Validation machinery**
`holdout.py` (the 150 m-block reserved split, SEED 20260817),
`perfil_gnss.py`, `transectos.py`, `paso3_lidar.py`, `criba_lidar.py`,
`zona_lidar_util.py`.

**Site downloads, detectors, figures, notebook tooling**
`descarga_*.py`, `tajo*.py`, `dk_*.py`, `sdr*.py`, `scal_*.py`,
`rias_cortas.py`, `tres_detectores.py`, `scl_vs_ndwi.py`, `fig_*.py`,
`figuras.py`, `build_*_nb.py`, `wire_*.py`, and assorted one-off checks
(`test_*.py`, `verify*.py`, `diagnostico.py`).

These scripts assume data files and working directories from their day and
are **not expected to run as-is** — they are kept as provenance, not as
tools.
