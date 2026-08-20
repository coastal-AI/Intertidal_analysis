# Prior art — Granadeiro et al. 2021 (P3)

Paper: *Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and Exposure
Periods over Large Intertidal Areas*, Remote Sensing 13(2):320, doi:10.3390/rs13020320.

Filled in on 2026-08-19 by reading the full PDF (not from memory). Where the paper does
not address something, "does not address it" is written — those rows are the list of our
deltas.

| field | them (Granadeiro 2021) | us (v4) |
|---|---|---|
| Site(s) and extent | Bijagós Archipelago (Guinea-Bissau), ~1200 km² of intertidal spread over ~100 km; semidiurnal regime, springs 0.3–4.8 m | Villaviciosa (3.5 km ría, the confined case) + Scheldt (validation with tide gauges) + northern-coast cells defined |
| Number of scenes and preprocessing | 35 L1C scenes (2017–18 and 2019–20, clouds <10%), ACOLITE atmospheric correction, and radiometric inter-calibration between scenes by major-axis regression on stable pixels (water NIR<0.05, land >0.2) | 465–1379 L2A scenes per site (the whole usable archive, not a selection); the whole-scene systematic is measured and enters the null (sd_scene), not corrected by regression |
| What exactly it estimates | 10 m topography (4-parameter logistic on NIR), cotidal lags, and exposure-period maps (sinusoidal Eq. 5) | 10 m topography + interior phase profile τ(s) + boundary phase correction + σ_topo/σ_nivel decomposition; exposure is not a goal |
| How it estimates the lag | SEPARATE logistic fits with rising and falling scenes; lag sweep −90..+90 min (step 5); the one minimizing the rise-vs-fall elevation difference is chosen; on 50,000 random mid-elevation pixels (2.47±0.25 m); then a GAM (splines, lon/lat) smooths the field | Reimplemented as the bar (M2d). Our engine (M2a) is different: Bernoulli likelihood with per-pixel PROFILED elevations and widths, anchor at the mouth, all pixels (not just mid-elevation) — beats M2d in the planted-truth gate (4.71 vs 5.42 min) |
| Lag symmetric between limbs, or hysteresis? | Symmetric by construction (a single lag that equalizes both limbs); they admit as a limitation not being able to separate rise/fall or springs/neaps | Hysteresis measured separately (fall +41 min where rise ~0 in the prototype; Hysteresis mechanism in the simulator); the operator admits two clocks |
| Does it estimate amplification/gain A(s)? | Does not address it (assumes the reference point's amplitude across the whole archipelago) | We prove it CANNOT be estimated from wet/dry (affine theorem, gate M2 v2 with flat NLL(α)) — they could not either; only the continuous data backs it out of sample (A=0.85 in the key band) |
| Reference water level | Tide tables of the port of Bubaque (a single point, from the Portuguese Hydrographic Institute, 1960s base) + cosine interpolation between high and low water (Eq. 1) | Interchangeable BoundaryProvider (global model pyTMD, tide gauge, ensemble, harmonic climatology) + phase correction audited from the imagery itself (M3) |
| Do they handle real vs nominal overpass time? | They use the exact acquisition time in the interpolation, but do not discuss it as an error source | Measured: 0.134 m of error from using the nominal time; real times via STAC for everything |
| Constituent aliasing (S2 invisible)? | Does not address it (with tide tables there are no harmonics to estimate) | Explicit alias table with the real times: S2 frozen, K1/P1 annual, M2/N2/O1/Q1/M4/M6 estimable |
| Spatial resolution of the lag | 50,000 sampled pixels → smooth lon/lat field by GAM (cotidal lines) | Quantile bands of the geodesic coordinate from the mouth (the estuary's geometry, not lon/lat); the confinement of a ría is exactly the case where lon/lat does not work |
| Declared lag precision, and against what truth | Mean absolute difference 6.6 min against 4 points of the Hydrographic Institute (max 15 min at Abú); reference possibly outdated (1960s obs.) | With PLANTED truth and real sampling: 4.71 min (M2a); external: gradient 0.72 min/km against 0.9 of the Scheldt tide gauges |
| Out-of-sample validation (train/test)? | Does not address it (no temporal partition; they validate the final exposure product against 66 time-lapse cameras, r²=0.94) | Yes: temporal OOS 65/35 in the amplitude judge; interleaved OOS judge in M3; test opened once |
| Nulls / statistical controls | Does not address it (no nulls; the field validation is direct) | Every verdict against a matched null: calibrated simulator (whole pixels + scene systematic + level jitter), 5-replicate null band, sign/specular controls |
| Wet/dry threshold and water sensor | Continuous logistic on NIR reflectance; NDWI only to DELIMIT the intertidal (temporal sd > 0.2) | Continuous NDWI for the sigmoid (with σ in meters interpretable as sub-pixel relief) and binary NDWI>0 for the phase estimators |
| Connectivity censoring / ponding? | They observe it as a bias (water film and puddles on low mudflat inflate exposure at elevations <2.2 m) but do not model it | ConnectivityCensoring mechanism in the simulator; ponding is part of the hysteresis model, not just an apology |
| Channel bathymetry from celerity? | Does not address it | h̄(s) = c²/g as a by-product of the phase profile; hydraulic plausibility test (Scheldt 18.5 m/s → plausible dredged channel) |
| Point-vs-pixel validation of the terrain? | Does not address it (their cameras use ~4 m GPS and validate exposure, not elevations) | Yes: σ_muestreo=0.214 m measured by two independent routes; c=1.041 — this literature's "compression" is an artifact of comparing a point with a pixel median; `point_sampling_error` in the package |
| What it does NOT estimate / limitations they admit | Elevation range limited by the observed tides (1.04–4.69 m); does not see the lowest; wet-mud bias; the lag averages limbs and springs/neaps; old tide reference | Gain not estimable from binary (theorem, exposed); amplitude only where the continuous OOS backs it; current RTK holdout burned for the final sealed verdict |

**Synthesis for the paper's introduction**: Granadeiro et al. demonstrated that the
S2 archive contains the cotidal lags and extracted them with a symmetric method,
smoothed in lon/lat and validated against old tables. Our deltas:
(1) a likelihood engine that beats theirs with planted truth and real sampling;
(2) hysteresis (two clocks) instead of a single lag; (3) estuary geometry
(s from the mouth) instead of lon/lat; (4) a theorem of what CANNOT be estimated
(gain) and its proof; (5) matched nulls and pre-registered gates for
every claim; (6) an interchangeable boundary with audited phase correction;
(7) a point-vs-pixel validation framework that explains the published "compression".
