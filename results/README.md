# results/: the record of the development phase

These files are the verdicts and outputs of the development experiments
(August to September 2026), kept as the record of what the authors had seen
before the pre-registration lock of validation v2. **They are not results of
validation v2**, the repository draws no conclusion from them, and where
they overlap with validation v2, validation v2 supersedes them. The outputs
of validation v2 and of the method comparison (v14-v19) are not in this
repository while that work is in progress.

| path | source | contents |
|---|---|---|
| `<experiment>/` (the m, b and p series: `m0_baseline/` to `p13_sigma_uncertainty/`) | `python -X utf8 -m experiments.<experiment>` | `result.json`: the verdict of one development experiment, with its pre-registered criterion, the numbers it was judged on and the sha256 of every input; many folders also hold a `figure.png` and small arrays (`.npz`) |
| `p4_<site>/`, `p5_<site>/` | `experiments/p4_site.py` (`p4_scheldt.py` for `p4_escalda/`), `experiments/p5_rmse_aoi.py` | the same experiment at several external estuaries, one folder per site |
| `north_coast_census/result.json` | aggregated from the per-cell MAREA results of the north-coast campaign (read by `experiments/paper_fig_census.py`) | one entry per campaign cell that had a result (68 cells) |
| `c1_dev_comparison.json`, `c5_validation_matrix.json` | `experiments/c1_compare_elevation.py`, `experiments/c5_validation_matrix.py` | the development comparison with Granadeiro et al. (2021) |
| `map_vs_plain.json`, `v1_vaklodingen_validation.json` | the September comparison ([`../docs/tide_adjustment_comparison.md`](../docs/tide_adjustment_comparison.md)); `experiments/v1_vaklodingen_validate.py` | how far each delayed map of that comparison moves from the map with no delay; its scores against the Vaklodingen surveys |
| `scene_rule_study_2026-09-30/` | `experiments/v11_local_lag.py` | MAREA's lag fitted on the pixels around an inner tide gauge, against the lag of the gauge's section, with and without the transition-zone scene rule |
| `santander_*` | read by [`../archive/research/08_santander_scl_marsh_audit.ipynb`](../archive/research/08_santander_scl_marsh_audit.ipynb) | the audit of the scene classification over the salt marshes of the Bahía de Santander |
| `v3_aperturas.jsonl` | `experiments/v3_open_reserved.py` | the log of every opening of the reserved RTK points (*aperturas* = openings): one opening so far, on 2026-08-21, which scored four development maps ([`../data/README.md`](../data/README.md)) |

The dated account of each experiment is
[`../docs/PHASE_LOG.md`](../docs/PHASE_LOG.md); the scripts and their
questions are listed in [`../experiments/README.md`](../experiments/README.md),
section 4. Some folder names and keys are in Spanish (*comparativa* =
comparison, *tabla de mareógrafos* = table of tide gauges, *incertidumbre*
= uncertainty).

Each verdict was reached on development data with the version of the method
of its date; the method has changed since (see the status notes of
[`../docs/README.md`](../docs/README.md)).
