# products_marea/: the 10 m MAREA product of the Ría de Villaviciosa

A copy of the 10 m production run of MAREA at the Ría de Villaviciosa
(Asturias, Spain): 97 screened Sentinel-2 scenes of 2023-2025, with the tide
at the mouth from the EOT20 model alone (no tide gauge). It was made before
the quality rule of 2026-10-06 existed, so no bracket rule was applied (its
`result.json` has no `quality_min_bracket` key). It is a product of the
method, not a validation result.

| file | contents |
|---|---|
| `marea.npz` | the per-pixel arrays: elevation `z`, its width `sigma` (the uncertainty the poster shows), the distance from the mouth `s_km`, the 3-km section `band` and the delay applied to each pixel `tau_px_min`; per section, the fitted and applied delays and their bootstrap interval; the hypsometric curve with its envelope |
| `result.json` | the run in numbers: status, scene count and dates, pixel counts, sections and their delays, the range of mouth levels searched, the sha256 of the input cube, run time (keys partly in Spanish) |
| `figure.png` | three panels with Spanish titles: the elevation map (*cota*), the fitted and applied delay against distance from the mouth (*marea interior*), and the hypsometric curve (*hipsometría*) |
| `overpass_times.json` | the acquisition time (UTC) of every Sentinel-2 scene of the site from 2016 to 2025, from the catalog |

Every key and its units are described in
[`../docs/products.md`](../docs/products.md#2-the-marea-product-mareanpz-resultjson-figurepng),
section 2; what `sigma` is and how it is computed, in the "Uncertainty"
paragraph of [`../docs/METHOD.md`](../docs/METHOD.md).

## Using it

* **Elevations** are in meters on the vertical scale of the EOT20 tide
  heights (about mean sea level), not on the Spanish national datum.
* **Pixels.** The arrays hold only the intertidal pixels. `keep` gives
  their flat indices in the frame `shape` = (915, 915), which is the
  canonical 10 m grid of Villaviciosa (EPSG:32630) stored in
  [`../products_villaviciosa/`](../products_villaviciosa/README.md).
  Pixels without an estimate are NaN.
* The second example of the [repository README](../README.md#minimal-examples)
  opens this product and writes its elevation as a GeoTIFF.
* **Emersion time** is not stored: for a pixel at elevation z it is
  `1 - pyintertidal.hydroperiod.hydroperiod(z, h)`, with h the mouth tide
  read the pixel's delay earlier over a continuous record.

To make the same kind of product for another site, run
`pyintertidal.marea.reconstruct` on a downloaded cube (the repository
README, example 3).
