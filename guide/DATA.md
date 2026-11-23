# Data: the maps, their heights and their files

* **What this page is for:** what the maps and files hold, how their heights are given,
  their formats and their licences.
* **Who it is for:** users of the map viewer and of its data files (GIS analysts,
  ecologists, managers), and readers of the product files.

## 1. The six sites

| site | region | images | period of the images | tide at the mouth | pixels with an elevation |
|---|---|---|---|---|---|
| Ría de Foz | Galicia | 140 | 2015-07-25 to 2019-12-31 | EOT20 and the Ferrol 2 gauge (80 km away) | 32,039 |
| Ría del Eo | Galicia and Asturias | 114 | 2017-01-25 to 2023-12-30 | EOT20 and the Ferrol 2 gauge (98 km away) | 52,562 |
| Ría de Villaviciosa | Asturias | 97 | 2023-01-06 to 2025-12-28 | EOT20 alone | 29,728 |
| Bahía de Santander | Cantabria | 51 | 2023-01-06 to 2025-12-28 | EOT20 and the Santander gauge (5 km away) | 90,888 |
| Marismas de Santoña | Cantabria | 116 | 2023-01-03 to 2025-12-28 | EOT20 and the Santander gauge (27 km away) | 84,904 |
| Ría de Mundaka (Urdaibai) | Basque Country | 197 | 2022-01-01 to 2026-10-31 | EOT20 and the Bilbao 3 gauge (30 km away) | 37,941 |

* **Images:** the screened Sentinel-2 L2A scenes of each map's record. The periods
  differ between sites; each map describes its own period.
* **Tide at the mouth:** the EOT20 model at the site. Where a gauge is named, the tide
  is the mean of EOT20 and that IOC tide gauge, with the gauge's median removed (EOT20
  alone where the gauge record has gaps).
* **Pixels:** 10 m pixels of 100 m² each.
* **Delay:** at the Ría del Eo and the Marismas de Santoña every section has a delay of
  0 min, and at the Bahía de Santander at most 1 min. There the maps use the tide at the
  mouth unchanged.
* **Villaviciosa:** the map is the product in
  [`products_marea/`](../products_marea/README.md), which
  [`REPRODUCE.md`](REPRODUCE.md) rebuilds bit for bit.

## 2. The layers

| layer | units | what it is |
|---|---|---|
| Elevation, z | m | The water level at which the pixel turns from dry to wet, read on the tide that reaches it: the tide at the mouth, delayed by the pixel's τ. |
| Uncertainty, σ | m | The width of the pixel's dry-to-wet transition: one of eight values, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45 and 0.65 m. A larger σ is a less sharp transition. It is not an accuracy measured against a survey. |
| Emersion time | % of time | The share of the map's period during which the pixel is out of the water: how often its delayed tide, h(t − τ), stays at or below z, with the tide taken every minute. |
| Delay, τ | min | How much later than at the mouth the tide reaches the pixel: one delay per 3-km section of distance from the mouth along the water, smoothed along that distance (Gaussian kernel of 3 km), 0 at the mouth, rounded to the minute. Each site has its own colour scale. At Mundaka the layer also draws lines of equal delay ([section 4](#4-tide-lines)). |
| Tide lines | m | Contours of the elevation at the tidal datums ([section 4](#4-tide-lines)). |

MAREA estimates elevations only inside the intertidal mask: the pixels seen wet in 10 to
90 % of more than 30 clear looks. Inside it, a pixel without an elevation failed a guard
of the fit ([`METHOD.md`](METHOD.md#5-from-sections-to-every-pixel)). The delay is
defined on the whole mask.

## 3. Heights: NMMA and the constant C

MAREA's elevations are on the zero of the tide used to make each map: the mean level of
EOT20 at Villaviciosa, and the mean of EOT20 and the median-removed gauge at the other
sites. That zero is close to the local mean sea level over the map's period, but it is
not a national datum.

The map viewer shows heights in **NMMA**, metres above the *Nivel Medio del Mar en
Alicante*, the mean sea level at Alicante that is the zero of Spanish levelling (REDNAP):

**height in NMMA = height on the map's zero + C**

C is the height in NMMA of the map's zero. For each image of a map, it is read from a
Puertos del Estado tide gauge whose zero is tied to NMMA, as the 25-hour mean of the
difference between the gauge and the map's tide around the image. C is the mean over the
map's images that have gauge data. Its 1σ uncertainty combines the tie of the gauge to
NMMA, the geoid model (EGM08-REDNAP), the transfer from the gauge to the estuary and the
spread over the images.

| site | C (m) | 1σ (m) | gauge of the tie |
|---|---|---|---|
| Ría de Foz | 0.3151 | 0.047 | Ferrol 2 |
| Ría del Eo | 0.3240 | 0.047 | Ferrol 2 |
| Ría de Villaviciosa | 0.3778 | 0.055 | Ferrol 2 (91 of the 97 images) |
| Ría de Mundaka (Urdaibai) | 0.3742 | 0.042 | Bilbao 3 |
| Bahía de Santander | – | – | none yet: heights above the tide model's mean |
| Marismas de Santoña | – | – | none yet: heights above the tide model's mean |

* The poster rounds the Villaviciosa constant to 0.38 m.
* The data files (the viewer's value images and the product files) keep the map's zero:
  add C to read them in NMMA.

## 4. Tide lines

Two kinds of tide line are drawn: contours at the tidal datums, in the **Tide lines**
layer, and, at Mundaka, lines of equal delay, with the **Delay** layer.

**Contours at the tidal datums.** Each datum is a level of the tide at the site. The line
joins the places whose elevation equals that level: the ground below it is covered for
longer, the ground above it for less. The datums come from the EOT20 model at each map's
tide point, every 10 minutes over the 19 years 2007–2025, longer than the 18.6-year cycle
of the Moon's orbit.

| datum | name | definition used here |
|---|---|---|
| HAT | highest astronomical tide | the highest high water of the 19 years |
| MHWS | mean high water springs | the mean high water of the spring tides¹ |
| MHWN | mean high water neaps | the mean high water of the neap tides¹ |
| MSL | mean sea level | the mean level of the 19 years (0 on the map's zero) |
| MLWN | mean low water neaps | the mean low water of the neap tides¹ |
| MLWS | mean low water springs | the mean low water of the spring tides¹ |
| LAT | lowest astronomical tide | the lowest low water of the 19 years |

¹ From the series' own high and low waters. A tidal day is two successive tidal cycles;
its high water is the mean of its two high waters, and likewise for its low water. The
spring tides are the tidal day of largest range in each fortnight, and the neap tides the
tidal day of smallest range.

The contours are drawn on MAREA's elevation, lightly smoothed (a Gaussian of one pixel
that ignores empty pixels, with holes of up to 9 pixels filled). Lines shorter than 100 m
are dropped, and the rest are simplified to 2.5 m. A datum is drawn only where the map
has values on both sides of it: between the 1st and the 99th percentiles of the mapped
elevation. The images never see the highest tides
([`METHOD.md`](METHOD.md#9-what-the-images-cannot-tell)), so MHWS and HAT lie above every
mapped pixel and are not drawn, and LAT lies below them. The map's legend lists all seven
datums and marks those that are drawn.

For example, at Villaviciosa the datums are, in m above NMMA: LAT −2.02, MLWS −1.47,
MLWN −0.46, MSL +0.38, MHWN +1.19, MHWS +2.20 and HAT +2.79.

**Lines of equal delay (Mundaka).** In the Ría de Mundaka the delay grows to more than
an hour, so the **Delay** layer also draws lines of equal delay every 10 minutes: the
places the tide reaches the same number of minutes after the mouth (co-tidal lines).
They follow the delay of each pixel, from the same smooth profile of the section delays.

**Files.** For each site, `tide_lines.geojson` holds the contours (properties `datum`,
`level_m` on the map's zero, and `length_m`), and `tide_levels.json` the seven levels and
whether each is drawn. At Mundaka, `cotidal.geojson` holds the lines of equal delay
(properties `tau_min`, `s_km` and `length_m`). Coordinates are WGS84, with 5 decimals.

## 5. The tide at each image

Notebook [`02_tide_per_image`](../notebooks/02_tide_per_image.ipynb) places each of the 97
Sentinel-2 images of the Villaviciosa record at the height of the tide when the satellite
passed, says whether the tide was rising or falling, gives the delayed level that each
section saw, and says how much of the tide range the images cover. The images are the
usable ones of the method's screening
([`METHOD.md`](METHOD.md#2-one-pixel-seen-many-times)).

## 6. Map viewer files

* **Grid.** The layers of a site share one grid in Web Mercator (EPSG:3857), of about
  6.89 m, which is 5 m on the ground at these latitudes. Each 10 m pixel of a product is
  2 × 2 grid pixels; the values are not resampled.
* **Overlay images.** Coloured RGBA PNG, for display only: at twice the grid's resolution
  (2.5 m on the ground), each overlay pixel interpolated bilinearly from the grid pixels
  around it that have a value, the edge of the data faded over one grid pixel, and
  transparent where there is no value. Colours: ColorBrewer scales (elevation YlGnBu,
  uncertainty Purples, emersion time Greens, delay Oranges). Overlays wider or taller
  than 4096 pixels come in tiles. The readout uses the value images.
* **Value images.** 8-bit RGB PNG that store the values for the pixel readout:
  code = 256 × R + G (B = 0), and value = offset + scale × code, with code 0 meaning no
  value. The offset and scale of each layer are in the site's `manifest.json`. The PNGs
  carry no colour profile, so they decode losslessly. In a browser, decode them with
  `createImageBitmap`, with `colorSpaceConversion` and `premultiplyAlpha` set to `none`.
* **Tide lines.** GeoJSON in WGS84 ([section 4](#4-tide-lines)).
* **Heights.** The value images hold elevations on the map's zero: add C for NMMA
  ([section 3](#3-heights-nmma-and-the-constant-c)).
* **In this branch.** The files are in [`site/public/data/`](../site/public/data/), and
  the built copy the viewer serves is under [`docs/data/`](../docs/data/).

## 7. The Villaviciosa product files

[`products_marea/`](../products_marea/README.md) holds the 10 m MAREA product of the Ría
de Villaviciosa: the 97 screened scenes of 2023–2025, with the tide at the mouth from
EOT20 alone, read at the centroid of the registered site polygon (43.5090 N, 5.4161 W).
It was made before the quality rule existed (`quality_min_bracket` 0). Its heights are on
EOT20's scale (add C = 0.3778 m for NMMA).

**`marea.npz`.** The per-pixel arrays are 1-D over the 32,120 intertidal pixels; `keep`
gives their flat row-major indices in the frame `shape` = (915, 915).

| key | units | what it is |
|---|---|---|
| `z` | m | elevation; NaN where a guard rejects the fit (29,728 pixels have a value) |
| `sigma` | m | width of the flooding curve, one of the eight values; NaN exactly where `z` is |
| `tau_px_min` | whole min | delay applied to each pixel (0 to 16 min here) |
| `s_km` | km | distance from the mouth along the water; NaN where water cannot reach |
| `band` | index | 3-km section of each pixel, 0 at the mouth; −1 without a water path |
| `keep`, `shape` | index, px | flat index of each intertidal pixel; the frame (rows, columns) |
| `tau_min`, `tau_usado_min` | min | fitted and applied delay of each section (*usado*: applied) |
| `tau_lo_min`, `tau_hi_min`, `tau_boot_min` | min | 95 % interval of each section's delay, and the 30 bootstrap fits |
| `hyp_z`, `hyp_area`, `hyp_lo`, `hyp_hi` | m, km² | hypsometric curve (area below each level) and its σ envelope |

**`result.json`.** The run in numbers. Its keys are partly in Spanish: `sitio` (site),
`estado` (status), `n_escenas` (scenes in the record), `n_px` (intertidal pixels),
`n_px_cota` (pixels with an elevation), `centros_km` (section centres), `tau_min` and
`tau_usado_min` (fitted and applied section delays), `tau_px_median_min` and
`tau_px_max_min` (median and largest pixel delay), `rango_marea_m` (the tide range the fit
searches), `hipsometria` (hypsometric integral and mapped area), `con_operador` (a delay
was applied), `veto_fisico` (a negative delay was set to 0), `inputs_sha` (sha256 of the
inputs) and `duracion_s` (run time).

**Other files.** `figure.png` (the elevation map, the delay against distance and the
hypsometric curve, with Spanish panel titles) and `overpass_times.json` (the UTC
acquisition time of every Sentinel-2 scene of the site, 2016–2025).

**On the map.** The frame is the grid of
[`products_villaviciosa/hsr_eot20_2023-2025.tif`](../products_villaviciosa/README.md):
915 × 915 pixels of 10 m in EPSG:32630 (UTM zone 30N), upper-left corner at
(301020, 4824840). That file's values are an older map; use it for its grid only.

```python
from pathlib import Path

import numpy as np
from pyintertidal.raster import grid_of, write_geotiff

prod = np.load("products_marea/marea.npz")
H, W = prod["shape"]
z = np.full(H * W, np.nan, np.float32)
z[prod["keep"]] = prod["z"]                      # elevation (m) of each intertidal pixel
z = z.reshape(H, W)

transform, crs, shape = grid_of("products_villaviciosa/hsr_eot20_2023-2025.tif")
assert shape == (H, W)
Path("runs").mkdir(exist_ok=True)                # git ignores runs/
write_geotiff("runs/marea_villaviciosa_2023-2025_z.tif", z, transform, crs)
```

`scripts/marea_tools.export_geotiffs` writes `z`, `sigma`, `tau_px`, `s_km` and `band` in
one call; `scripts/marea_tools.emersion_time` computes the emersion time from `z`,
`tau_px_min` and a tide series such as the kit's 10-minute EOT20 series.

## 8. The notebook sample

[`data/`](../data/README.md) holds a real block of 270 × 270 pixels of 10 m (2.7 km ×
2.7 km) of the Ría de Villaviciosa, cut from the cube the product was made from. It
covers part of the central flats, the main channel and the inner reach. It holds:

* the 97 scenes of 2023–2025: bands B03, B08 and SCL, and the clear-sky flag;
* the tide at each image, also read every whole minute from −45 to 180 min late;
* the product's elevation, uncertainty, delay, distance from the mouth and sections for
  the block;
* the EOT20 tide every 10 minutes over 2023–2025.

Its variables and checks are described in [`data/README.md`](../data/README.md).
Notebook [`01_marea_villaviciosa`](../notebooks/01_marea_villaviciosa.ipynb) uses it.

## 9. Licences and attribution

* **MAREA's maps, the product, the kit and the sample:** CC BY 4.0
  ([`LICENSE-docs.md`](../LICENSE-docs.md)). Suggested attribution: "MAREA maps by
  J. Alías Álvarez, F. García-González and P. González, CC BY 4.0. Contains modified
  Copernicus Sentinel data (2015–2026)."
* **Sentinel-2:** Copernicus data, free and open; "Contains modified Copernicus Sentinel
  data (2015–2026)", processed with openEO on the Copernicus Data Space Ecosystem.
* **EOT20:** Hart-Davis et al. (2021), DGFI-TUM; data on SEANOE,
  <https://doi.org/10.17882/79489>, CC BY 4.0. Its values are in the kit and the sample.
* **Tide gauges:** IOC Sea Level Station Monitoring Facility (VLIZ, IOC-UNESCO); Ferrol 2,
  Santander and Bilbao 3 are stations of Puertos del Estado.
* **Website basemaps:** PNOA orthophoto and IGN base map, © Instituto Geográfico
  Nacional de España, CC BY 4.0 (scne.es). They are loaded at run time from IGN's
  services and not redistributed.
* **3D terrain**, where the map viewer offers it (wide screens): elevation tiles loaded at
  run time, with their attribution on the map.
* **Website code:** the bundled libraries keep their licences, collected in
  [`docs/licenses/`](../docs/licenses/).

## 10. Seagrass: ICE CREAMS

ICE CREAMS classifies intertidal vegetation, seagrass included, from Sentinel-2. Its
European intertidal seagrass map is not included here; it can be viewed and downloaded
from the links below.

* Viewer: <https://s2maxextenticecreams.projects.earthengine.app/view/europeanintertidalseagrass>
* Paper: Davies, B. F. R., Oiry, S., Roca, M., Rosa, P., Zoffoli, M. L., Poursanidis, D.,
  Dolch, T., Ondiviela, B., Galván, C., Brito, A. C., Lilley, R. J., Govers, L. L.,
  Gade, M., Barillé, L., Gernez, P. (2026). An initial map of European intertidal
  seagrass. *Remote Sensing of Environment* 333, 115116.
  <https://doi.org/10.1016/j.rse.2025.115116>
* Data: <https://doi.org/10.5281/zenodo.17523642> (CC BY 4.0).
