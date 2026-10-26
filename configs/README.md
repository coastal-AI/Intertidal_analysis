# configs/: thresholds fixed before the experiments that use them

Every threshold of the development phases lives in one of these files, each
value with its rationale in a comment, so that no number is hidden in the
code. Each file was written before the experiment that reads it was run;
changing a value is a decision recorded in
[`../docs/PHASE_LOG.md`](../docs/PHASE_LOG.md) with its rationale (as
`m0.yaml` states). Keys and comments are partly in Spanish, for example
*umbral* = threshold, *puerta* = gate, *plantado* = planted, *banda* = band.

| file | phase | read by |
|---|---|---|
| `m2.yaml` | the delay estimator: the seed, the per-pixel widths and the grid of candidate elevations that the delay fit profiles over, the sample of pixels per section | **the production engine** `pyintertidal.marea.reconstruct`; the validation v2 harness (`experiments/benchmark/`); the method-comparison scripts v14, v16 and v19; many development scripts |
| `m0.yaml` | gate M0: the reference values fixed when plan v4 was adopted (2026-08-18) | `experiments/m0_baseline.py`, `experiments/b2_sigma.py`, `tests/test_gate_m0.py` |
| `m1.yaml` | phase M1: the geometry of the estuary read from the archive | `experiments/m1_geometry.py`, `tests/test_gate_m1.py` |
| `m3.yaml` | phase M3: a boundary tide that can be corrected, and which tidal constituents may be | `experiments/m3_gate_sim.py`, `tests/test_gate_m3.py` |
| `m4.yaml` | phase M4: the distributed tide-gauge operator and its gate | `experiments/m4_gate_sim.py`, `tests/test_gate_m4.py` |
| `b5.yaml` | phase B5: separate delays for the flood and the ebb | `experiments/b5_gate_sim.py` |
| `b6.yaml` | phase B6: spill and ponding layers of the elevation map | `experiments/b6_hydraulic_dem.py` |
| `b7.yaml` | phase B7: a per-pixel uncertainty from planted recoveries in a calibrated simulator | `experiments/b7_incertidumbre.py` |

The phases are described in
[`../experiments/README.md`](../experiments/README.md) (section 4, the
development record) and dated in `docs/PHASE_LOG.md`.

**Do not edit `m2.yaml`.** The validation v2 harness hashes it into the code
key of every MAREA product it makes, so any change, even to a comment, marks
those products as stale. To try other values, pass them as arguments to
`pyintertidal.tide_estimators.m2a_rasch`, as the first example of the
[repository README](../README.md#minimal-examples) does.
