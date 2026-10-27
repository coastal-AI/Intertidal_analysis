# MAREA variants declared on 2026-10-27, before any confirmatory window is computed

This file fixes, before the numbers exist, which MAREA variants will be tried, how each is defined
and how it will be judged. It is committed before the pre-2023 images enter any n-table, so the git
history dates it. Nothing here changes the published method; adoption is the first author's decision.

## Windows and truth

- **Development windows** (already seen): the n-tables of `notebooks/validation/01_fixed_config_n_tables_site.ipynb`
  with n = 40, 50, 60 (Westerschelde, Wadden 2025; Ems n = 40, 50). MAREA-0 was chosen on these windows.
- **Confirmatory windows** (not yet computed): the same notebook, same truth (Vaklodingen cells surveyed in
  2023-2025, the most recent survey per cell, dated to the end of its year), same pixels F, with
  n = 75, 100, 125, using the pre-2023 images being downloaded. Every variant below is judged on these.
- Configuration common to all methods unless a variant says otherwise: the production screening, clear mask
  SCL 4-7, NDWI = (B03 - B08)/(B03 + B08), wet = NDWI > 0, the production boundary.

## Variants

| id | name | definition |
|---|---|---|
| V0 | MAREA (reference) | as published: delay per 3-km band (profiled Bernoulli likelihood, monotone, 0 at the mouth), per-pixel least-squares fit NDWI = a + b Phi((h(t - tau) - z)/sigma) on the grid of `marea.invert_series`, elevation = the midpoint z, no quality rule |
| V1 | MAREA-0 | the same fit; elevation h0 = z + sigma Phi^-1(-a/b), where the fitted curve crosses NDWI = 0; no value when 0 is not strictly between a and a + b. Declared after the development windows (post hoc) |
| V2 | MAREA-B | the same delay; per pixel, the Bernoulli (probit) likelihood of the wet/dry record W against h(t - tau_px) on its clear looks, P(wet) = Phi((h - z)/sigma), maximised on the grid of `invert_series` (50 levels between the window's lowest and highest level, sigma in `marea.SG_GRID`); elevation = z, the level of P = 0.5; ties (complete separation) resolved by the median z of the tied grid points; at least 8 clear looks with at least one wet and one dry |
| V2d | MAREA-B, no delay | V2 with tau = 0 (diagnostic: the contribution of the delay to V2) |
| V3 | water index MNDWI | configuration factor for EVERY method: MNDWI = (B03 - B11)/(B03 + B11), B11 resampled to the 10 m grid as stored, wet = MNDWI > 0; only where B11 exists (the nine-band stores, 2023-2025, so n <= 60) |
| V4 | robust fit | V0 and V1 with a Huber loss on the NDWI residuals (delta = 0.10) in place of least squares, and weight 0.5 for looks of SCL class 7; with CLD available (nine-band stores) also looks with CLD > 20 % dropped (V4c) |
| V5 | spatial median | post-processing factor for EVERY method: 3 x 3 median of the elevation map over pixels with a value (a pixel keeps no value if it had none) |
| V6 | flood and ebb delays | two delays per band, one fitted on the rising-tide images and one on the falling-tide images (rising: the boundary level 30 min later is higher), each image's levels read with the delay of its limb; judged against the gauges (notebook 02, development judges only, W3 stays blinded) and in the n-tables |
| V7 | amplitude per band | the boundary level of each band scaled by the ratio of the EOT20 amplitude at the band's centroid to that at the boundary point, then delayed; never calibrated on validation gauges |
| V8 | recency weights | V0 and sampled exceedance with weights exp(-dt/T), T = 1 year, dt the image's age before the window's last image (as notebook 04) |
| V9 | sigma as uncertainty | not an estimator: for V0 and V1, the share of pixels with abs(p - t) <= sigma and <= 2 sigma, and the Spearman correlation of abs(p - t) with sigma |
| V10 | MAREA-B with lapse rates | V2 with a contamination term: P(wet) = e + (1 - 2e) Phi((h - z)/sigma), e on the grid {0, 0.02, 0.05, 0.10} maximised with z and sigma (the psychometric lapse rate), so a few spurious wet looks at low tide or dry looks at high tide do not drag z; added 2026-10-27 after notebook 05's development gallery, before any confirmatory window |

## How they are judged

- Scores as `scores()` in notebook 01: RMSE, bias = median(p - t), centred RMSE, slope; on each product's pixels
  and on the common pixels; per site and n.
- A variant improves MAREA when, on the confirmatory windows, both RMSE and centred RMSE decrease at two of the
  three sites or more and no site gets worse by more than 0.02 m in either, with a pixel-block bootstrap interval
  of the paired difference reported for each site.
- Factors applied to every method (V3, V5) are reported for every method; MAREA's rank among the methods is
  reported with and without them.
- The development windows are reported as development, never as evidence.
