# Intertidal Analysis — MAREA

Mapping intertidal topography from optical satellite time series, with the
tide measured from the imagery itself.

**MAREA** (*Mudflat Altimetry via Remotely-Estimated tides from the
Archive*): every published intertidal-topography method assumes one uniform
water level per scene, taken from an ocean tide model at the estuary mouth.
Inside an estuary that is false — the tide arrives late and deformed. MAREA
treats every intertidal pixel as a binary threshold sensor, measures the
interior tide's phase lag by profiled Bernoulli likelihood, corrects the
elevation inversion with it only where the correction beats a matched null,
and delivers a DEM with calibrated per-pixel uncertainty, declared
hydraulics, and demonstrated limits.

## Repository layout

```
Intertidal_analysis/
├── pyintertidal/          The package: one function per scientific step,
│                          explicit parameters, no run_everything().
├── intertidal_topography_villaviciosa.ipynb
│                          The worked study: the complete pipeline on one
│                          estuary, every step shown in a cell.
├── research/              The evidence: seven executed notebooks, one per
│                          deduction, including the negative results.
├── experiments/           Gated experiments (m0–m4, b1–b7, p0–p8), the
│                          campaign runner, and prototypes/ — the verbatim
│                          exploratory record that preceded them.
├── configs/               Pre-registered gate criteria with rationale.
├── tests/                 Guards (R1 reserved-data lock) and pinned gate
│                          verdicts.
├── docs/                  PHASE_LOG.md — the full audit trail: every gate,
│                          recalibration and verdict, failures included.
├── deliverables/          What accompanies the paper: poster, flagship
│                          DEM, north-coast census, method animation, and
│                          release snapshots of package + experiments.
├── poster/                A0 poster source (beamerposter).
└── intertidal/, stand-by/ Legacy code predating the project, kept as-is.
```

## Installation

```bash
git clone https://github.com/coastal-AI/Intertidal_analysis.git
cd Intertidal_analysis
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Data access uses openEO (Copernicus Data Space; free account, OIDC login on
first use) and global tide models via eo-tides/pyTMD (EOT20 by default).

## Quick start

The study notebook is the guided path. The campaign equivalent is one call:

```python
from pyintertidal import marea
result = marea.reconstruct("ndwi_cube_<site>.nc", out_dir="products_<site>")
```

which writes the DEM, the interior-tide profile, the hypsometry with
uncertainty, and a self-describing `result.json` with input hashes.

## Conventions

* English throughout — code, comments, docs, file names. Two deliberate
  exceptions: keys of previously sealed artifacts (`result.json`,
  `configs/*.yaml`, stored `.npz`) keep the names they were recorded with,
  and `experiments/prototypes/` is preserved verbatim as the historical
  record.
* No label leaks: the reserved RTK set is locked by a runtime guard
  (`tests/test_r1_guard.py`); the only opener is
  `experiments/v3_open_reserved.py`, run by a human, and every opening is
  logged.
* Every adopted mechanism passed a pre-registered gate against a matched
  null; every retired one is documented with its control.
