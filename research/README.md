# MAREA — research notebooks

Each notebook reproduces one line of reasoning from the MAREA project
(*Mudflat Altimetry via Remotely-Estimated tides from the Archive*), reading
only the stored experiment artifacts under `results/` and `runs/` — nothing
heavy is recomputed, so the whole folder executes in seconds and the outputs
shown are exactly the numbers each conclusion was drawn from. Rebuild and
re-execute everything with:

```bash
python research/_build_notebooks.py
```

| Notebook | The deduction it reproduces |
|---|---|
| `01_interior_tide_identifiability` | The interior tide can be *dated* from wet/dry imagery (~5 min, beating the literature baseline) but not *scaled* — the affine theorem, executable. Plus the constituent alias table and the real-archive verdict (Villaviciosa runs late only at the head: +26 min). |
| `02_boundary_audit` | A +12 min boundary-model error masquerades as estuarine physics unless audited from the mouth pixels; phase is recoverable (±4 min), gain is not (ensemble prior). |
| `03_operator_and_gauges` | The distributed tide gauge scored against real gauges: contraction gate, the Terneuzen ladder (92 % of what any lag can give, zero instruments), the Scheldt gradient match, and the seven-gauge model table. |
| `04_bathymetry_validation` | Elevation vs external truth: Tagus −11 %, Aveiro +24..+53 mm where the lag grows, Vadehavet correctly declining, Villaviciosa RTK tie with the point-vs-pixel decomposition. |
| `05_uncertainty_hydraulics` | The layers no published intertidal DEM carries: σ_topo/σ_level decomposition, calibrated per-pixel σ_z (conservative: 83 % coverage at 68 % nominal), spill/ponding hydraulics. |
| `06_negative_results` | The three demonstrated limits: binary gain blindness, per-limb clocks not adoptable at 465 scenes, and self-hiding censoring. |
| `07_north_coast_census` | The 116-cell campaign: hypsometric integrals with uncertainty and the first interior-tide census of the northern Spanish coast (progressive — re-run as cells land). |

Provenance rules of the project (enforced by `tests/`): every threshold lives
in `configs/*.yaml` with its rationale; every verdict has a matched null;
every recalibration is pre-registered in `docs/PHASE_LOG.md`; every result
file carries the SHA-256 of its inputs.
