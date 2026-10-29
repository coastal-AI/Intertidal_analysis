# Validation v3, the Cantabrian estuaries: design of 2026-10-29, before any window is computed

The first author refocused the validation on 2026-10-29: Spanish Cantabrian estuaries, PNOA LiDAR as the elevation
truth, and windows centred on the LiDAR flight. This file fixes the design before any number exists; the git history
dates it. Every constant is fixed here; none is chosen per site or by looking at the truth. The Dutch validation
(notebooks 01 and 05 on Westerschelde, Wadden, Ems) is kept as it is, as a supplementary set.

## Sites and truth

The truth files are those the second author (F. García-González) listed on 2026-10-29, used unless he states that
the flight was at high water:

| site | coverage | product, sheets | his note | use |
|---|---|---|---|---|
| Villaviciosa | 2nd, 2020 | MDT02 ETRS89 HU30 0015-3, 0030-1 | low water | truth |
| Mundaka (Urdaibai) | 2nd, 2020 | MDT02 ETRS89 HU30 0038-3, 0038-4 | very good low water on the right bank, noise on the left bank | truth |
| Mundaka (Urdaibai) | 3rd, 2023 | MDT50CM ETRS89 H30 0038 4-5, 4-6, 4-7, 5-5, 5-6, 5-7, 5-8, 6-8; 0062 5-1 | good low water | truth |
| Foz | 2nd, 2020 | MDT02 ETRS89 HU29 0009-4 | low water, a composite of two flights at different low waters | truth |
| Eo | 2nd, 2020 | MDT02 ETRS89 HU29 0010-3, 0010-4, 0025-1 | low water tending to mid tide | truth |
| Santander | 2nd 2020, 3rd 2023 | none | high water, no useful data | maps only |
| Santoña | 2nd 2020, 3rd 2023 | none | high water | maps only |

**Truth date.** The flight month is not known yet. Provisional truth date: 15 July of the coverage year (2020-07-15,
2023-07-15), the middle of the summer when the flights are scheduled (the second author's rule). When the CNIG
metadata give the flight dates, they replace it in a dated amendment written before the windows are recomputed.

**Truth processing** (`experiments/benchmark/truth.py`, `pnoa_ground` and `aggregate_ground`, unchanged): the sheets
of a site and coverage are mosaicked; water at flight time is masked (4-connected patches of at least 50 cells whose
values round to the same mm; and, when a flight time is known, every cell below the flight-time water level + 0.15 m);
the remaining ground cells are averaged to the site's 10 m Sentinel-2 grid. Mundaka's two coverages are two truths,
each with its own windows. The heights are orthometric (REDNAP); the levels are the boundary's (EOT20 with the site's
gauges), so a datum offset common to every method may appear as bias: the centred RMSE does not see it, and the bias is
reported as it is.

## Windows

For each n in (20, 40, 60, 80, 100, 120), the window is the n screened scenes nearest in time to the truth date d
(sorted by |t - d|, ties by the earlier t): the smallest window [d - T, d + T] symmetric in time that holds n scenes.
Its half-width T and the scenes before and after d are reported. A window is computed only if the cube covers
[d - T, d + T] entirely; otherwise it is not reachable.

## The pixels scored

As in the n-tables (the production extraction's pixels; the truth inside the no-delay product's range +- 0.25 m;
a larger F is the seeded 200,000-pixel subsample), and in addition, the first author's request: **a pixel is scored
only when neither class dominates its record in the window**: its wet share f = wet looks / clear looks (wet = clear and
NDWI > 0) satisfies 0.10 <= f <= 0.90. The same pixels for every method. The sensitivity is reported: the scores at
q = 0, 0.05, 0.10, 0.15 and 0.20 (0.10 <= f <= 0.90 being q = 0.10).

## Methods

The ten of the n-tables (MAREA, MAREA-0, NIDEM, DEA, LT2D, F2T, Tseng, SE, CXR, WDB), MAREA-B, MAREA-BL and V15s
(MAREA-S with the smoothed delay, the one-model MAREA of the variants addendum), as notebook 01 and notebook 05 run
them, with the site's production configuration (screening, clear mask SCL 4-7, NDWI, the boundary).

## Outputs

1. Per site, truth and n: RMSE, bias = median(p - t) and centred RMSE, on each method's pixels and on the pixels common
   to all; tables as the short report's, and error curves against n in one figure.
2. Per site and method, a map of the per-pixel error p - t on the 10 m grid (diverging colour scale, the same limits for
   every method of a site), at n = 100 and at the smallest n, with the decomposition of the mean squared error
   (offset, relief, scatter) of each method.
3. The class-dominance sensitivity: the scores against q.

## Judging

Descriptive, per site and n, against the best of the eight other methods; the variants against MAREA (V0) with the
rule of the variants files. Santander and Santoña get elevation maps without scores.
