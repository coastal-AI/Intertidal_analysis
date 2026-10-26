# products_villaviciosa/: the canonical 10 m grid of Villaviciosa

One data file, kept for its grid:

| file | contents |
|---|---|
| `hsr_eot20_2023-2025.tif` | a single-band float32 GeoTIFF, 915 x 915 pixels of 10 m in EPSG:32630 (UTM zone 30N), upper-left corner at (301020, 4824840); NaN outside the mapped pixels |

Its grid is the **canonical 10 m grid of the Ría de Villaviciosa**:

* `pyintertidal.rtk.load_rtk()` places the RTK-GNSS survey
  ([`../data/`](../data/README.md)) on it;
* the MAREA product in [`../products_marea/`](../products_marea/README.md)
  is on the same frame, so the second example of the
  [repository README](../README.md#minimal-examples) uses this file to
  write that product as a GeoTIFF.

Its values are an elevation map of the development phase, made with the
earlier HSR method (`pyintertidal.elevation.fit_hsr`, EOT20, 2023-2025).
They are not the MAREA product and not a validation result; use the file
for its grid. ([`../docs/products.md`](../docs/products.md), section 6.)
