# STORY — the narrative thread of the method (for paper, poster and repo)

This document is the scientific argument in logical, not chronological, order. Every
claim carries its evidence and its reproducible artifact. The final text of the papers
is written by the human; this is the skeleton that must not be lost.

## 1. The problem, in one sentence

An optical satellite archive turns the tide into a scanner of the intertidal relief:
each pixel gets wet when the water rises above its elevation, so 465 dated photos are 465
yes/no questions about every point of the mudflat. The standard method (per-pixel sigmoid
against the tide of an ocean model) is *prior art* — Catalão & Nico 2017, Bué 2020,
Granadeiro 2021, Chen & Wang 2025 — and this project does NOT claim its invention.

## 2. What the literature reports and nobody explained: the compression

All methods recover the relief *compressed* (slope against terrain < 1;
here: 0.73–0.78 depending on the partition). The universal suspicion is "the tide inside
the estuary is not the model's". This project chased it with null discipline and the
result is the backbone of the paper:

| hypothesis about the compression | verdict | evidence |
|---|---|---|
| tilted water surface (3 variants) | dead | synthetic nulls; explains ≤1.5% |
| surge (inverse barometer) | dead | does not beat shuffled surge |
| tide-ceiling censoring | retracted | circular analysis (clearance ≡ −elevation) |
| morphological drift (2 years imagery-campaign) | dead | 3 epochs of equal n, CI crosses 0 |
| regression dilution | dead | the noise is in the response; master null: slope 1.019 |
| **point-vs-pixel sampling of the validation** | **CONFIRMED** | matched k=1 vs k=2: c = 1.041 [0.79, 1.58]; σ_e = 0.214 m by 2 independent routes |

**Thesis 1 (validation paper):** a good part of the "compression" this
literature reports is an artifact of comparing a GNSS point with the median of a 10 m
pixel. Of the error reported against RTK, 61–88% was the reference, not the method.
Citable tool: `pyintertidal.validation.point_sampling_error`. Campaign recipe
that fixes it: ~18 points per pixel.

## 3. The distributed tide gauge: the interior tide from the archive itself

Even so the interior tide differs from the oceanic one, and it is measurable WITHOUT
instruments by treating the pixels as binary tide gauges (the path opened by Granadeiro
2021 with cotidal lags; them/us table in `docs/prior_art_granadeiro.md`).

Our deltas over that bar, each with a pre-registered gate:

1. **Per-limb hysteresis** (τ_subida ≠ τ_bajada per reach): ponding is measured,
   not assumed. In Villaviciosa the high band passes the out-of-sample gate with
   (τ_s=0, τ_b=+20 min) and the specular control against it.
2. **Amplitude judged by the data**: A=0.85 (damping) contributes where there is physics
   and is a flat direction where there is not — the judge is the likelihood on held-out
   scenes.
3. **External validation with tide gauges the method never sees**: Scheldt — imagery lag
   gradient 0.9 min/km = the tide gauges'; recovers 75% of the achievable
   level correction at Terneuzen with zero instruments.
4. **Explicit alias table**: S2 (solar constituent) is invisible by construction
   for a sun-synchronous sensor; M2/N2/O1/Q1/M4/M6 estimable. A hard limit of the field.
5. **Channel bathymetry for free**: h̄(s) = c²/g from the celerity of the lag, with a
   hydraulic plausibility test (Scheldt: 35 m, plausible; where it gives impossible
   values, that IS the puddle map, and it is reported as such).

## 4. The tribunal: why this is credible

Nothing is claimed against zero; everything is claimed against a **matched null**
manufactured by a simulator calibrated from the archive (real tide, pixels resampled
whole, per-pixel noise + scene-wide systematic + deconvolved level uncertainty). The null
reproduces the archive's marginals (gate M0.2) and is unbiased in the interior of the
tidal window — and reproduces the estimator's real edge bias, which is exactly
what a tribunal must do. Two results with p≈1e-37 died today against well-matched
nulls: that is this architecture's reason to exist.

**Thesis 2 (method paper):** mechanism-based correction of the interior level + the
validation framework (out-of-sample gates, specular controls, sealed predictions,
untouchable holdout) as a reusable methodological contribution.

## 5. Zero labels, and why

Training a corrector with 64 RTK labels is significantly WORSE (−0.038 m,
CI95 [−0.065, −0.014]) than training it with zero labels on the calibrated simulator.
The whole design is transferable to coasts without campaigns: that is the scaling
argument (116 cells of the northern coast, Part IV).

## 6. Practical reproducibility

- sealed data (`sealed/registry.jsonl`, SHA256+date, append-only);
- deterministic partitions (seeds in `configs/`), holdout with guard R1;
- every gate is a test (`tests/test_gate_*.py`); every experiment a script with
  config and results versioned by hash;
- nothing hard-wired: thresholds and tolerances live in `configs/*.yaml` with their why.

## Thesis 3 (phase M2): the interior tide can be TIMED from the archive, but not SCALED

What a binary wet/dry archive can and cannot measure of the interior tide,
demonstrated with planted truth (not argued):

- **Phase, yes.** Four independent estimators — Rasch likelihood (M2a),
  per-pixel concordance (M2b), flooded-area ranks (M2c) and the literature
  method of flood/ebb discrepancy (M2d, Granadeiro 2021) — recover a
  planted lag profile of 0→40 min with errors of 5–13 min, under real temporal
  sampling, real clouds and calibrated archive noise.
- **Amplitude, no, and it is a theorem, not an implementation limitation.** The
  likelihood of binary responses is exactly invariant under
  (α, z_p, σ_p) → (c·α, c·z_p, c·σ_p): per-band gain is indistinguishable from a
  rescaling of the local elevations and widths. Gate v1 made it visible in the most
  instructive way: the optimizer spent α compensating a misspecified σ and returned
  the bound of the box. Gate v2 requires exposing the flat NLL(α) curve instead of
  reporting an invented number.
- **Design consequence**: the operator T corrects phase (identifiable, validated
  externally on the Scheldt); the amplitude only enters where the CONTINUOUS data
  backs it out of sample (the OOS judge of method b2: A=0.85 in the only band
  where it adds predictive power). Published methods that report tidal gains
  from binary waterlines should pass through this same gate.

## Thesis 4 (phases M3–M4): the boundary error is corrected in phase, bounded in gain, and the operator bites

- **The leak, demonstrated before being prevented**: +12 min of phase error in the
  ocean model at the mouth masquerades as distributed "estuarine transfer" (up to
  +21 apparent min with flat truth). Any work that measures transfers
  against a global model without auditing its phase at the mouth may be publishing the
  model's error as estuary physics.
- **The audit is asymmetric, and that is a result**: the boundary's PHASE is
  recovered from the imagery itself (dt̂ with 4-min error, OOS judge at the mouth); the
  GAIN cannot be audited from binary (affine theorem, measured twice) and stays
  bounded by the ensemble prior (the spread between independent ocean
  models). Imagery audits phase; the ensemble audits gain.
- **The final operator is an inspectable composition** (`InteriorTide.describe()`):
  boundary (any provider) + mouth phase correction + τ(s) profile measured
  by likelihood with anchor at the mouth — and it only carries signal where 5 replicates
  of the uniform null do not explain it. In gate M4 it recovers the planted damage (elevation
  RMSE 0.44→0.37 m in the high band) without touching the mouth and being a fixed point.
- **In real Villaviciosa**: the interior runs on EOT20 time up to ~5.4 km (τ within
  the null, where the RTK lives) and arrives +26 min late at 7.7 km — tens of cm of
  level misassigned per scene at the head, now corrected by an operator that
  never saw a tide gauge or a label.
