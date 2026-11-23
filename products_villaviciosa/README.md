# products_villaviciosa/: the canonical 10 m grid of Villaviciosa

One data file, kept for its grid only:

| file | contents |
|---|---|
| `hsr_eot20_2023-2025.tif` | a single-band float32 GeoTIFF, 915 x 915 pixels of 10 m in EPSG:32630 (UTM zone 30N), upper-left corner at (301020, 4824840); NaN outside the mapped pixels |

Its grid is the **canonical 10 m grid of the Ría de Villaviciosa**. The
MAREA product in [`../products_marea/`](../products_marea/README.md) is on
the same frame, so the notebooks and the scripts use this file as the grid
reference when they write that product as GeoTIFFs (`grid_reference` in
`scripts/sites/villaviciosa.yaml`; `pyintertidal.raster.grid_of`).

Its values are an older elevation map of the development phase, made with
the earlier HSR method (`pyintertidal.elevation.fit_hsr`, EOT20, 2023-2025).
They are not the MAREA product and not a validation result: use the file
for its grid only.
