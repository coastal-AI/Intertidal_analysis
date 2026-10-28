# MAREA variants, addendum of 2026-10-28, before any window is computed with them

This addendum to `experiments/variants_preregistered_2026-10-27.md` fixes, before the numbers exist, four more MAREA
variants (V11-V14), the run of the already declared V4, and the bootstrap the judging rule asks for. It is committed
before any of them is computed on any window, so the git history dates it. Nothing here changes the published method;
adoption is the first author's decision. Every constant below is fixed here, a priori; none is chosen per site or by
looking at the truth.

## Why these four

The attribution (notebook 05, the confirmatory windows of 2026-10-28) found what the other methods do well that MAREA
does not, and what MAREA does that they do not:

- they read the elevation at the wet/dry transition, not at the midpoint of the NDWI magnitude, which NDWI's rise with
  water depth pulls upward (MAREA's bias +0.21-0.26 m at Westerschelde); MAREA-0 and MAREA-B already do this;
- they assume no curve shape (DEA: a rolling median along the tide; sampled exceedance: the wet frequency);
- DEA abstains on 9-14 % of the Westerschelde pixels, the high flats wet 7-10 % of the time, where every method errs
  0.6-0.8 m; MAREA maps every pixel and its sigma is not calibrated (31-65 % of the pixels within 1 sigma);
- DEA calls water NDWI > 0.1; every method with wet = NDWI > 0 is low at Ems (-0.10 to -0.18 m), DEA is not;
- MAREA's delay is its own, transferable contribution: with it, LT2D, sampled exceedance and DEA improve at the Wadden
  (DEA 0.459 -> 0.247 m at n = 100) and DEA with MAREA's delay is the best of all at Ems.

V11 drops the curve shape, V12 uses the NDWI magnitude without its bias, V13 gives MAREA a likelihood interval, an
abstention rule and an error-coverage comparison of every method, V14 replaces the fixed water threshold by one
learnt from each scene. V4 (declared 2026-10-27) is the robust loss.

## Common settings

As the variants file: the same windows, truth, pixels F and configuration; MAREA's delay tau_px (V0's band lags of
the window, its smoothed profile, to the minute); h = h_B(t - tau_px); clear = SCL 4-7; wet = clear and NDWI > 0
(a clear look with an undefined NDWI is dry) unless V14 says otherwise; the guards of V2: at least 8 clear looks with
at least one wet and one dry; the z grid of `marea.invert_series` (50 levels between the window's lowest and highest
level) and sigma in `marea.SG_GRID` wherever a grid is used.

## Variants

| id | name | definition |
|---|---|---|
| V11 | MAREA-M (monotone) | per pixel, the isotonic (non-decreasing) regression of the wet indicator of its clear looks on h, by pool-adjacent-violators, looks at equal h pooled first; p = the fitted wet probability. Elevation z = (h_lo + h_hi)/2, h_hi the lowest level with p >= 0.5 and h_lo the highest level below it; if every level has p >= 0.5, z = the lowest level (censored low); if none does, z = the highest level (censored high); a censored pixel keeps its value and is flagged. No curve shape is assumed |
| V11d | MAREA-M, no delay | V11 with tau = 0 (diagnostic: the delay's share) |
| V12 | MAREA-P (depth-aware) | per pixel, least squares of NDWI = a + Phi(u) (b + c (h - z)), u = (h - z)/sigma, over the clear looks with a defined NDWI: (a, b, c) in closed form at each (z, sigma) of the grid, with c >= 0 (where the unconstrained c is negative, that grid point takes the c = 0 fit, MAREA's model); the best grid point by the loss. Elevation z, the level where the pixel is half covered; the rise of NDWI with depth is c, not the midpoint. Guards of `invert_series` (N >= 8, 0.15 < b < 2.5, abs(a) < 2.5). With c = 0 it is V0 |
| V13 | likelihood interval and abstention | for MAREA-B (V2): the profile log-likelihood lp(z) = max over sigma of V2's log-likelihood, on V2's grid; the 95 % interval {z : lp(z) >= max lp - 1.9207} (chi-square 1 d.f., 0.95, halved); bracketed = the interval touches neither end of the grid. V13a = MAREA-B with no value where the interval is not bracketed (the observed levels do not constrain the elevation). Reported: V13a's scores and coverage; the share of pixels with abs(p - t) within the interval's half-width (nominal 0.95); and, for EVERY method, error-coverage curves on a common, method-independent reliability r = min(number of wet looks, number of dry looks) in the window: at coverages 100, 95, 90, 85 and 80 % of F, the pixels ranked by r (largest first, ties by pixel index), the first ceil(q |F|) kept, each method scored on its pixels with a value there and on the pixels common to all |
| V14 | per-scene water threshold (factor) | for each scene, t_s = the Otsu threshold (256 bins on [-1, 1]) of the NDWI of its clear extraction pixels with a defined NDWI; when t_s falls outside [-0.2, 0.3] (a scene without two classes, e.g. all water at high tide) t_s = 0. Wet = clear and NDWI > t_s. A factor applied to EVERY wet/dry estimator, each reported as "+ Otsu": MAREA-B, MAREA-BL, MAREA-M, LT2D (E4), sampled exceedance (E5c) and the wet/dry bracket (E12); the competitors receive it as NDWI - t_s per scene, so that their own rule wet = NDWI > 0 is wet = NDWI > t_s without changing their code. MAREA's delay fit is unchanged (V0's) |
| V4 | robust fit (declared 2026-10-27) | as declared: V0 and V1 with a Huber loss on the NDWI residuals (delta = 0.10) in place of least squares, and weight 0.5 for looks of SCL class 7; computed by iteratively reweighted least squares on `_fit_block`'s grid: weights w = clear x s x min(1, delta/abs(r)), s = 0.5 for SCL 7, r the residual of the current fit, starting from the weighted least-squares fit, until the grid point and (a, b) repeat or 20 iterations; V4 = the midpoint z, V4-0 = the NDWI = 0 crossing (as V1); invert_series' guards. V4c (CLD > 20 % dropped) is not computed: CLD exists only in the nine-band stores, which do not cover the confirmatory windows |

## How they are judged

- As the variants file: RMSE, bias = median(p - t), centred RMSE, on each product's pixels and on the common pixels,
  per site and n. A variant improves MAREA (V0) when, on the confirmatory windows (n = 125, 100, 75), both RMSE and
  centred RMSE decrease at two of the three sites or more and no site gets worse by more than 0.02 m in either.
- The bootstrap the rule asks for, fixed now: the paired difference variant - V0 on the pixels both have, resampled
  by blocks of 32 x 32 pixels of the grid (320 m), 1,000 resamples, seed 20261028, the 95 % percentile interval of
  delta RMSE and delta centred RMSE per site and window.
- Also reported, not part of the rule: each variant against MAREA-B and against the best of the eight other methods,
  on the common pixels and on the error-coverage curves of V13.
- The truth is the one-row Vaklodingen of the n-tables (Westerschelde and Ems: the cells surveyed in 2025; Wadden:
  2023-2025 together, the first author's choice of 2026-10-28, made after the per-year confirmatory run).
- The development windows (n = 60, 50, 40) are computed first and only to find bugs; any change of a definition after
  them is written here, dated, before the confirmatory windows are computed.

## Not computed now

V3 (MNDWI) and V4c need B11 and CLD, which only the incomplete nine-band stores have. V5 (3 x 3 median), V6 (flood
and ebb delays), V7 (amplitude per band) and V8 (recency weights; notebook 04's preliminary test on the development
windows showed no gain at Westerschelde) stay declared and are computed later.
