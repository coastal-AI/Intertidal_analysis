# poster/: the CARTOGRASS 2026 poster

**MAREA: a method for mapping intertidal topography using Sentinel-2 openEO
data cubes.** Jorge Alías Álvarez, Fernando García-González, Pablo González.
CARTOGRASS 2026, Simposio Ibérico de Cartografía y Seguimiento de Praderas
Marinas, Cartagena, 24-26 November 2026. A0 portrait (841 x 1189 mm).

* Latest draft: [`CARTOGRASS2026_poster_draft_2026-10-20.pdf`](CARTOGRASS2026_poster_draft_2026-10-20.pdf).
* The Results panels are left blank and the Conclusions banner names only
  what each message will be about ([pending]); both are filled only from the
  results of validation v2, which is in progress
  ([`../docs/validation_v2_spec.md`](../docs/validation_v2_spec.md)). The
  "Pilot result" box is earlier work, from the submitted abstract.
* The QR code of the bottom strip (`content/cg/blk_foot_qr.tex`) opens the
  repository, `https://github.com/coastal-AI/Intertidal_analysis`.

## How the sources are organised

| path | what it holds |
|---|---|
| `cartograss2026.tex` | the sheet: page size, the four-column grid, every row and panel position in millimetres, and the order of the blocks |
| `content/cg/text_*.tex` | every word on the sheet, defined as macros (`text_header`, `text_context`, `text_methods`, `text_results`, `text_conclusions`, `text_footer`, `text_captions`); `text_all.tex` loads them. Comments beside each text record why it says what it says and where each number comes from |
| `content/cg/blk_*.tex` | the blocks that place those texts and figures in the panels (header, context, concept, materials and methods, results, conclusions, bottom strip) |
| `content/cg/cg_macros.tex` | the layout macros of this poster (`\cgpanel`, `\cgfig`, `\cgreserved`, ...); they log the geometry of every panel to the build log |
| `preamble/*.tex` | packages, colours, theme and tables, shared with the earlier poster draft |
| `figures/cg/` | the figures, as PDF (vector, placed on the sheet) and PNG (300 dpi) |
| `tools/cg/` | the scripts that draw the figures, and `build_poster.py` |

## Build

From the repository root:

```bash
python -X utf8 poster/tools/cg/build_poster.py            # draft: poster/build/, copied to poster/CARTOGRASS2026_poster_draft.pdf
python -X utf8 poster/tools/cg/build_poster.py --final    # also fails on missing figures, drafts and placeholders
python -X utf8 poster/tools/cg/build_poster.py --print    # the file for the print shop
```

It runs pdflatex twice and reports the geometry of every panel, any
overflow, and the placeholders still on the sheet. It needs a TeX
distribution with beamerposter and the Fira Sans fonts; the authors use
MiKTeX, whose folder is set at the top of `build_poster.py`
(and in `tools/cg/b_common.py`, `fig_method_pipeline.py` and
`fig_pixel_gauge.py`, which also call pdflatex or kpsewhich).

Each figure script is run the same way, for example
`python -X utf8 poster/tools/cg/fig_method_flow.py`; its docstring gives its
options, inputs and outputs, and most also write the numbers behind the
figure to `poster/build/cg/<figure>.json`. Most scripts read inputs that are
not in the repository: the 10 m netCDF cube of Villaviciosa, the sites'
water-frequency and intertidal-mask rasters (`products_*/`), the Villaviciosa
production record (`results/server_10m/products_marea/marea_eot20_g0_2023-2025/record.json`),
and the gauge records and station lists under `data_v4/gauges/`; basemap
tiles and fonts are downloaded into `tools/cg/cache/`, which git ignores.

## Figures

Figures placed on the sheet:

| file (PDF and PNG) | made by | alt text / caption |
|---|---|---|
| `figures/cg/method_flow` | `tools/cg/fig_method_flow.py` (the Ems-Dollard outline from `products_ems/water_frequency.tif`, a 20 m raster not in the repository, cached in `poster/build/cg/method_flow_ems.npz`) | How MAREA works, followed for one imaginary pixel, in six cards: (1) seen many times in a Sentinel-2 data cube, each cloud-free look wet or dry; (2) the tide at the mouth reaches the pixel's 3-km section later, and the pixel is wet when that water stands above its ground; (3) the section's delay is the one that best splits the wet and dry looks of its pixels; (4) the midpoint of the flooding curve on the corrected tide is the elevation, its width the uncertainty; (5) emerged time is the share of time the corrected tide stays below the ground; (6) the maps of elevation, uncertainty and emerged time. Schematic: simulated looks and tide, example delays; the outline and the 3-km sections are those of the Ems-Dollard from Sentinel-2. |
| `figures/cg/pixel_gauge` | `tools/cg/fig_pixel_gauge.py` (Villaviciosa cube, `products_marea/overpass_times.json`, the record dates in `results/server_10m/products_marea/marea_eot20_g0_2023-2025/record.json`, EOT20) | One real 10 m pixel near the mouth of the Ría de Villaviciosa, 2023-2025: each dot is one cloud-free look of MAREA's record, placed at the mouth tide of its image (EOT20, metres above mean sea level) and coloured teal when the water index is above 0 (wet) and sand when it is 0 or below (dry). The line is the flooding curve; its midpoint is marked z, the shaded band its width. About 3 km from the mouth, so the delay is close to zero. The z drawn is an illustration, not a result. Contains modified Copernicus Sentinel data (2023-2025). |
| `figures/cg/sites_map_strip` | `tools/cg/fig_sites_map.py --layout strip` (the sites' `products_*/water_frequency.tif` and `intertidal_mask.tif`, `data_v4/gauges/estuary_truth/stations.json`, Natural Earth) | Study sites: an overview of the Atlantic coast from north-west Spain to the German Bight with the five sites numbered, and one small map per site: (1) Ría de Villaviciosa, (2) Ría de Ferrol, (3) Westerschelde, (4) Wadden Sea (Vlie basin), (5) Ems-Dollard. Sand: intertidal flats; pale blue: water; teal triangles: tide gauges inside the sites, used only as checks. Basemap: Natural Earth. |
| `figures/cg/b1_tide_arrives_late` | `tools/cg/b1_tide_arrives_late.py` (gauge records) | Ems estuary, 11 July 2025 (UTC): measured water level at Borkum (the mouth gauge, IOC), Delfzijl (Rijkswaterstaat) and Emden (WSV), each relative to its 2025 mean. Inside the estuary high water arrives later and higher; two arrows mark how much later it arrives at Delfzijl and Emden, and the tidal range of each gauge is printed in the key. |
| `figures/cg/ctx_lowtide`, `figures/cg/ctx_hightide` | `tools/cg/fig_ctx_tide_pair.py` (Villaviciosa cube, `products_villaviciosa/intertidal_mask.tif` to rank the scenes, Gijón gauge as a check) | The Ría de Villaviciosa in Sentinel-2 near-infrared (band B08, 10 m), the same frame at low tide (flats emerged) and at high tide (flats flooded), two scenes one week apart in 2025 (dates printed on the panels). Water is dark, exposed wet flats mid-grey, vegetated land bright. Contains modified Copernicus Sentinel data (2025). |

Figures kept for a later version (not placed on the current sheet):

| file (PDF and PNG) | made by | alt text / caption |
|---|---|---|
| `figures/cg/b2_decimetres_hours` | `tools/cg/b2_decimetres_hours.py` (Gijón gauge, IOC) | Decimetres become hours: the share of time in the air of ground at each elevation, from one year (2024) of the Gijón tide gauge (IOC, Puertos del Estado). Its message is now one line of the Objective box. |
| `figures/cg/sites_map` | `tools/cg/fig_sites_map.py` | The study-site map in its larger layout (overview and five site maps), of which the strip above is the placed version. |
| `figures/cg/method_pipeline`, `figures/cg/method_pipeline_wide` | `tools/cg/fig_method_pipeline.py` (TikZ source `tools/cg/cg_method_pipeline.tex`) | The MAREA pipeline in five steps, as icons (no data), at two widths; replaced by the method flow chart. |
| `figures/cg/pixel_gauge_strip`, `figures/cg/pixel_gauge_wide` | `tools/cg/fig_pixel_gauge.py` (other presets) | Earlier layouts of the one-pixel figure. |

## The earlier poster draft (August-September 2026)

The sources of an earlier MAREA poster (August-September 2026), superseded
by this sheet, are in [`../archive/poster_marea_2026-08/`](../archive/poster_marea_2026-08/)
(`poster.tex`, `poster-a3.tex`, `content/`, `tikz/`, `tools/`). It does not
build from there: its preamble stays in `preamble/`, now used by the
CARTOGRASS poster. Its figures (several with Spanish labels, all from the
development phase):

| file | alt text |
|---|---|
| `../archive/poster_marea_2026-08/figures/02_curvas_reales.png` | Three real pixels, low to high on the flat: water index against tide level with the fitted S-curve of each, whose centre is the elevation. |
| `../archive/poster_marea_2026-08/figures/27_perfil_gnss.png` | Elevation along the walked route of the RTK-GNSS survey of Villaviciosa (14 August 2026) and four earlier estimators, the error of each along the route, and the route drawn on an earlier elevation map. |
| `../archive/poster_marea_2026-08/figures/29_santander_tres.png` | Bay of Santander: three water detectors on the same observations, and two maps of their differences. |
| `../archive/poster_marea_2026-08/figures/census_map.png` | The north-coast campaign: the 10-km cells along the Cantabrian and Galician coast, coloured by the verdict of the unattended run. |
| `../archive/poster_marea_2026-08/figures/dem_v4.png` | An earlier MAREA elevation map of the Ría de Villaviciosa at 10 m (development version 4). |
| `../archive/poster_marea_2026-08/figures/escalda.png` | Westerschelde: delay against distance along the estuary axis from two estimators, and the gradient between the tide gauges (Spanish labels). |
| `../archive/poster_marea_2026-08/figures/fig_ladder.png` | Bar chart of the water-level error at the Terneuzen tide gauge (Westerschelde) for several sources of the level: two GOT models, a three-model ensemble, EOT20 alone, EOT20 read with MAREA's delay, the best pure delay (which needs the gauge) and an operator calibrated with two gauges (development phase). |
| `../archive/poster_marea_2026-08/figures/fig_m2_recovery.png` | Simulation: a planted delay profile along the channel and its recovery by MAREA and by a baseline estimator. |
| `../archive/poster_marea_2026-08/figures/m2_gate.png` | Simulation gate M2 (Spanish labels): planted and recovered delay for four estimators, and the likelihood against an imposed tidal gain, flat as the affine invariance predicts. |
| `../archive/poster_marea_2026-08/figures/sigma_z.png` | Per-pixel elevation uncertainty of an earlier Villaviciosa product: the map of sigma_z calibrated in the simulator, and its look-up table by tidal headroom and number of observations (Spanish labels). |
