# Documentation figures

The figures in this folder are drawn by [`../make_doc_figures.py`](../make_doc_figures.py).
Run it from any folder:

```
python -X utf8 docs/figures/make_doc_figures.py               # every figure
python -X utf8 docs/figures/make_doc_figures.py sites_map     # one or more, by name
python -X utf8 docs/figures/make_doc_figures.py --list
```

Every figure is a 200 dpi PNG. The script reads only files in the repository.
Every color comes from one muted palette, the constants at the top of the
script (the map ramps included; no library default color map is used).
Some figures use a synthetic tide, `SYNTH` in the script: M2 1.30 m, S2 0.45 m,
N2 0.27 m, K1 0.07 m and O1 0.06 m, with the S2 trough at 11:00 UTC. Others use
synthetic noise drawn with the fixed seed `SEED`. Each figure's note says which
of its numbers are synthetic. No figure shows a validation result.

## Figures drawn here

| file | what it shows | source data | function |
|---|---|---|---|
| `sites_map.png` | The five validation sites on the Atlantic coast of Europe. A table gives each site's type and its role: the elevation check (PNOA LiDAR or Vaklodingen), the number of delay-check tide gauges inside the estuary, and the tide used at the mouth. | `pyintertidal.sites` (polygon centroids); `experiments/benchmark/config.py` `ELEVATION_TRUTH` and `judges.py` `HEADLINE`, read as literals; `docs/validation_v2_spec.md` §1.1; `../data/ne_10m_land_europe_atlantic.geojson` | `fig_sites_map` |
| `tide_delay_concept.png` | (a) Three days of the mouth tide h(t) and the delayed tide inside h(t − τ), with τ = 60 min. Real Sentinel-2 passes are marked, and gray bands show when the mouth tide gives a flat at z = 0.50 m the wrong state. (b, c) The 97 looks of the record, wet or dry, placed by the level read at the mouth and by the level inside. The looks the mouth level gets wrong are circled. | synthetic tide at the real acquisition times (`products_marea/overpass_times.json`, `result.json` `record_dates`) | `fig_tide_delay_concept` |
| `sentinel_sampling.png` | (a) One month of tide with every Sentinel-2 pass, all near 11:10 UTC. (b) The distribution of the tide at all times against the tide at the 97 kept looks. The tide stands above the highest look 12.7 % of the time. The note gives the range of the real record: −2.17 to +1.15 m. | synthetic tide at the real acquisition times; `result.json` `rango_marea_m` | `fig_sentinel_sampling` |
| `lag_profile_villaviciosa.png` | Delay against distance from the mouth along the water. It shows the fitted delay of each 3-km section with its 95 % bootstrap interval and the 30 bootstrap fits, the applied delay (equal to the fitted one: there is no detection threshold), and the delay applied to every pixel after smoothing. The anchor τ = 0 at the mouth and the inland rise that never decreases are visible. | `result.json` (`centros_km`, `tau_min`, `tau_usado_min`, `tau_lo_min`, `tau_hi_min`); `marea.npz` (`tau_boot_min`, `s_km`, `tau_px_min`, `band`); `marea.TAU_DETECT_MIN` | `fig_lag_profile_villaviciosa` |
| `flooding_curve.png` | The pixel model NDWI = a + b·Φ((h − z)/σ). (a) Synthetic looks with the fitted curve, with z, σ, a and a + b marked; the guards, and the quality rule (at least 3 clear looks below z and 3 above). (b) The grid search: the sum of squared residuals over 50 z × 8 σ, with a and b solved in closed form. (c) A pixel near the top of the sampled range, which the quality rule rejects. | synthetic looks (seeded) at the real acquisition times; the fit is `pyintertidal.elevation._fit_block` with the defaults of `marea.invert_series` and `marea.SG_GRID` | `fig_flooding_curve` |
| `product_maps_villaviciosa.png` | Four maps of the product on a land/water background, in UTM 30N with a scale bar and north: (a) elevation z, (b) σ, (c) distance from the mouth s with the section edges, (d) the delay applied to each pixel. | `marea.npz` (`z`, `sigma`, `s_km`, `band`, `tau_px_min`, `keep`, `shape`); the grid from the header of `products_villaviciosa/hsr_eot20_2023-2025.tif`; `../data/villaviciosa_water_background.tif` | `fig_product_maps_villaviciosa` |
| `product_distributions.png` | (a) Area at each of the 50 elevation levels. (b) The hypsometric curve with its σ envelope. (c) The share of pixels at each of the 8 σ values. (d) The median and quartiles of z along the channel. | `marea.npz` (`z`, `sigma`, `s_km`, `hyp_z`, `hyp_area`, `hyp_lo`, `hyp_hi`); `result.json` (`hipsometria`, `rango_marea_m`) | `fig_product_distributions` |
| `affine_identifiability.png` | The affine theorem of `docs/METHOD.md`. A tide scaled by a gain α = 1.2, over a pixel at z = 1.2 m with σ = 0.24 m, gives the same periods of emersion and the same emersion time as α = 1.0 over z = 1.0 m with σ = 0.20 m. Its wet/dry probabilities are also identical. | synthetic tide; the two worlds of `docs/METHOD.md` | `fig_affine_identifiability` |
| `screening_timeline.png` | Sentinel-2 scenes per month in 2023-2025: every acquisition over the site (466) against the scenes kept after cloud screening (97), with yearly totals. | `overpass_times.json`; `result.json` `record_dates` | `fig_screening_timeline` |
| `sections_villaviciosa.png` | The three 3-km sections over the ría, colored by section, with the edges at s = 3 and 6 km and the delay of each section with its 95 % interval. It replaces `../bands_map.png`. | `marea.npz` (`band`, `s_km`); `result.json` (`tau_usado_min`, `tau_lo_min`, `tau_hi_min`, `s_max_km`); `../data/villaviciosa_water_background.tif` | `fig_sections_villaviciosa` |

Sections are numbered from 0, as in the code (`band` in `marea.npz`) and in
`docs/METHOD.md`.

The maps show only what `marea.npz` stores. Two quantities are not mapped:

- **Emersion time.** Computing it honestly needs a continuous tide series at
  the mouth, read each pixel's delay earlier, over the whole record (for
  example EOT20 every 10 min for 2023-2025). The levels at the image times
  are not enough. The repository holds neither that series nor the EOT20
  model files, and a synthetic tide would give a map that is not the
  product. `products_marea/README.md` explains how to compute it.
- **The number of clear looks per pixel.** The product does not store it.

## Existing figures that already do the job

These figures are not redrawn here.

| file | what it shows |
|---|---|
| `../../../poster/figures/cg/method_flow.png` | The method in six cards, following one pixel: looks, delayed tide, section delay, elevation, emersion time, maps. |
| `../../../poster/figures/cg/method_pipeline.png` | From images to elevation, as a five-step strip. |
| `../../../poster/figures/cg/pixel_gauge.png` | One real pixel of Villaviciosa: its looks against the mouth tide and its flooding curve. |
| `../../../poster/figures/cg/b1_tide_arrives_late.png` | Measured tide at Borkum (mouth), Delfzijl and Emden on one day: the tide inside arrives 58 and 86 min late. |
| `../../../poster/figures/cg/b2_decimetres_hours.png` | Ground elevation against time in the air, from the Gijón gauge: 30 cm of ground is about one hour more in the air per tide. |
| `../../../poster/figures/cg/ctx_lowtide.png`, `ctx_hightide.png` | The Ría de Villaviciosa at low and at high tide (Sentinel-2 near-infrared). |
| `../../../poster/figures/cg/sites_map.png` | The poster's sites map, with a small map of each site showing its flats and in-estuary gauges. |
| `../pixel_model_schematic.png` | One pixel in cross-section at three tide stages, and the S-curve they trace. |
| `../sampling_schematic.png` | Sun-synchronous sampling: M2 is seen as a 14.8-day wave, and S2 stays frozen at one phase. |
| `../interior_tide_schematic.png` | Inside an estuary the tide is late, and the level read at the mouth is more wrong inland. |
| `../datacube.svg` | The Villaviciosa data cube: bands against time. |
| `../onepixel/*.png` | The one-pixel walkthrough (`docs/one_pixel_marea_example.md`). |
| `../workflow.png` | The archive, the wet/dry classification, one pixel's record and the elevation map. Caveat: the caption of panel (d), "where the interior clock was adopted", describes the adoption gate, which was removed on 2026-09-23. |
| `../bands_map.png` | Superseded: it shows six quantile bands of an earlier product. Use `sections_villaviciosa.png`. |

## Data files (`../data/`)

| file | contents | rebuilt by |
|---|---|---|
| `ne_10m_land_europe_atlantic.geojson` (112 kB) | Natural Earth 1:10m land and land borders (public domain), clipped to 14° W - 13° E and 40 - 58° N, simplified to 0.01°. | `make_doc_figures.py --build-data NE_DIR WATER_FREQUENCY_TIF` |
| `villaviciosa_water_background.tif` (20 kB) | uint8 on the canonical 10 m grid. 1 = water (Sentinel-2 water frequency ≥ 0.5, outside the product's intertidal pixels), 0 = land. Contains modified Copernicus Sentinel data. | same command, from `products_villaviciosa/water_frequency.tif` |
