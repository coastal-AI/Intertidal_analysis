# products_marea/: the 10 m MAREA product of the Ría de Villaviciosa

The 10 m production run of MAREA at the Ría de Villaviciosa (Asturias,
Spain): 97 screened Sentinel-2 scenes of 2023-2025, with the tide at the
mouth from the EOT20 model alone (no tide gauge), read at the centroid of the
registered site polygon. It was made with `quality_min_bracket` = 0, before
the quality rule of 2026-10-06 existed: its `result.json` has no
`quality_min_bracket` key, which means no bracket rule. It is a product of
the method, not a validation result.

| file | contents |
|---|---|
| `marea.npz` | the per-pixel arrays: elevation `z`, its width `sigma`, the distance from the mouth `s_km`, the 3-km section `band` and the delay applied to each pixel `tau_px_min`; per section, the fitted and applied delays and their bootstrap interval; the hypsometric curve with its envelope |
| `result.json` | the run in numbers: status, scene count and dates, pixel counts, sections and their delays, the range of mouth levels searched, the sha256 of the input cube, run time (keys partly in Spanish) |
| `figure.png` | three panels with Spanish titles: the elevation map (*cota*), the fitted and applied delay against distance from the mouth (*marea interior*), and the hypsometric curve (*hipsometría*) |
| `overpass_times.json` | the acquisition time (UTC, datatake start) of every Sentinel-2 scene of the site from 2016 to 2025, from the STAC catalogue |

Every array and key, with its type, shape and units, and the Spanish keys
translated: [`../guide/DATA.md`](../guide/DATA.md). What `sigma` is and how it is
computed: [`../guide/METHOD.md`](../guide/METHOD.md). Heights, times and delays
follow the conventions of [`../scripts/marea_tools.py`](../scripts/marea_tools.py).

## Using it

* **Elevations** are in metres on the vertical scale of the EOT20 tide
  heights (about mean sea level), not on the Spanish national datum.
* **Pixels.** The arrays hold only the intertidal pixels. `keep` gives
  their flat indices in the frame `shape` = (915, 915), which is the
  canonical 10 m grid of Villaviciosa (EPSG:32630) stored in
  [`../products_villaviciosa/`](../products_villaviciosa/README.md).
  Pixels without an estimate are NaN.
* **GeoTIFFs and emersion time** are not stored here. Notebook 01
  ([`../notebooks/01_marea_villaviciosa.ipynb`](../notebooks/01_marea_villaviciosa.ipynb))
  maps this product, with its emersion time; `scripts/marea_tools.py`
  writes its layers as GeoTIFFs (`export_geotiffs`) and computes the
  emersion time (`emersion_time`).
* **Reproduce it** with [`../guide/REPRODUCE.md`](../guide/REPRODUCE.md):
  offline from [`../replication/`](../replication/README.md), or from the
  authors' cube, always with `quality_min_bracket=0`. On 2026-10-27 both
  routes gave arrays identical to the ones here. The scripts write into
  `runs/`, never into this folder.

How the method works, step by step:
[`../guide/METHOD.md`](../guide/METHOD.md).
