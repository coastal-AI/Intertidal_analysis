# Data: the maps, their heights and their files

* **What this page is for:** what the maps and files hold, how their heights are given,
  their formats and their licences.
* **Who it is for:** users of the map viewer and of its data files (GIS analysts,
  ecologists, managers), and readers of the product files.

## 1. The sites

The map viewer shows fourteen estuaries. Four are drawn inside an intertidal mask cleaned
by hand:

| site | region | images | period of the images | tide at the mouth | pixels with an elevation | pixels on the map |
|---|---|---|---|---|---|---|
| Ría de Foz | Galicia | 140 | 2015-07-25 to 2019-12-31 | EOT20 and the Ferrol 2 gauge (80 km away) | 32,039 | 30,720 |
| Ría del Eo | Galicia and Asturias | 114 | 2017-01-25 to 2023-12-30 | EOT20 and the Ferrol 2 gauge (98 km away) | 52,562 | 42,964 |
| Ría de Villaviciosa | Asturias | 97 | 2023-01-06 to 2025-12-28 | EOT20 alone | 29,728 | 26,964 |
| Ría de Mundaka (Urdaibai) | Basque Country | 197 | 2022-01-01 to 2026-10-31 | EOT20 and the Bilbao 3 gauge (30 km away) | 37,941 | 34,038 |

The other ten are **preliminary**: each is drawn inside an intertidal mask made by a rule,
which nobody checked by hand, and none has an accuracy assessment, because no survey of
the ground was compared with them. The viewer writes "preliminary" next to their names.

| site | region | images | period of the images | tide at the mouth | pixels with an elevation | pixels on the map |
|---|---|---|---|---|---|---|
| Ría de Corme e Laxe | Galicia | 131 | 2021-01-02 to 2026-10-13 | EOT20 alone | 48,929 | 46,921 |
| Ría de Noia (inner part) | Galicia | 143 | 2021-01-02 to 2026-11-02 | EOT20 alone | 51,521 | 49,281 |
| Ría de Betanzos | Galicia | 189 | 2022-01-12 to 2026-11-06 | EOT20 alone | 48,009 | 41,372 |
| Ría de Ferrol | Galicia | 171 | 2022-01-14 to 2026-09-28 | EOT20 and the Ferrol 2 gauge (2 km away) | 53,200 | 45,188 |
| Ría de Ortigueira | Galicia | 130 | 2022-01-12 to 2026-11-06 | EOT20 and the Ferrol 2 gauge (40 km away) | 93,149 | 92,485 |
| Ría de Tina Menor | Cantabria | 123 | 2021-01-06 to 2026-11-06 | EOT20 and the Santander 2 gauge (56 km away) | 11,905 | 10,669 |
| Ría de San Vicente de la Barquera | Cantabria | 117 | 2021-01-06 to 2026-11-06 | EOT20 and the Santander 2 gauge (50 km away) | 33,381 | 32,245 |
| Ría de Mogro | Cantabria | 111 | 2021-01-16 to 2026-10-14 | EOT20 and the Santander 2 gauge (13 km away) | 16,288 | 15,312 |
| Bahía de Santander | Cantabria | 99 | 2021-03-02 to 2026-10-14 | EOT20 and the Santander 2 gauge (5 km away) | 93,965 | 90,726 |
| Marismas de Santoña | Cantabria | 176 | 2022-01-06 to 2026-10-31 | EOT20 and the Santander 2 gauge (26 km away) | 118,843 | 116,681 |

* **Images:** the screened Sentinel-2 L2A scenes of each map's record. The periods
  differ between sites; each map describes its own period.
* **Tide at the mouth:** the EOT20 model at the site. Where a gauge is named, the tide
  is the mean of EOT20 and that IOC tide gauge, with the gauge's median removed (EOT20
  alone where the gauge record has gaps). For the preliminary sites the gauge records
  used end in 2025, and the Santander record starts in 2023, so at the seven preliminary
  sites with a gauge it takes part at 45 to 78 % of the images.
* **Pixels:** 10 m pixels of 100 m² each.
* **On the map:** the map viewer draws every layer only on the pixels with an elevation
  inside an intertidal mask, on the same 10 m grid. At the first four sites that mask was
  cleaned by hand by the second author. The cleaning removes the water-detection noise at
  sea, around islands and along channel edges. Gaps on the flats are pixels of that mask
  without an elevation: outside the product's own mask, or a fit that failed a guard.
* **The mask of the preliminary sites** is the product's own intertidal mask cleaned by a
  fixed rule, without hand editing: a 3 × 3 median filter removes isolated pixels, then
  only the pieces connected to the estuary, which reaches the open sea at its mouth, are
  kept. The rule removes 1 to 15 % of the pixels with an elevation. It keeps some noise
  that a hand cleaning would remove, such as mussel rafts in the Rías de Betanzos and
  Noia and specks on rocky shores and at sea. Gaps on the flats are larger than at the
  first four sites: 13 to 48 % of the mask has no elevation, mostly channel and harbour
  margins that stay wet at the tides the images see.
* **What to keep in mind at the preliminary sites:**
  * Ferrol: only about half of its mask has an elevation; the map is patchy along the
    port and dock shores. Tina Menor: about two thirds.
  * Noia: the box covers only the inner part of the ría, so its distances, and its
    delays, start at the box's western edge, not at the mouth of the ría.
  * Mogro: the estuary has two 3-km sections. MAREA estimates 54 min for the inner
    channel (95 % interval 36 to 85 min) and 0 for the outer one; the smooth curve
    through them gives the outer flats delays of 10 to 20 min.
  * Corme: MAREA estimates 46 min for its innermost section; that delay has not been
    checked against measurements.
  * The constants C of Betanzos, Corme and Noia come from a gauge outside the estuary, and
    those of the five Cantabrian sites from the coast's mean sea level of 2011
    ([section 3](#3-heights-nmma-and-the-constant-c)).
* **Delay:** at the Ría del Eo, the Ría de Tina Menor, the Bahía de Santander and the
  Marismas de Santoña every section has a delay of 0 min; at the Ría de Ferrol every
  section has less than half a minute (0.4 min in the innermost one). There every pixel has
  a delay of 0 min, and the map uses the tide at the mouth unchanged. At the Ría de Mundaka
  MAREA estimates delays of 0, 9.6, 23.4 and 96.7 min for the four sections; the delays
  of the pixels, read from a smooth curve through them, reach 80 min, so the map shows less
  delay in the inner reach than the last section's fit
  ([`METHOD.md`](METHOD.md#5-from-sections-to-every-pixel)).
* **Villaviciosa:** the map is drawn from the product in
  [`products_marea/`](../products_marea/README.md), inside the hand-cleaned mask and on
  the viewer's grid. [`REPRODUCE.md`](REPRODUCE.md) rebuilds that product array by array.
* **What this branch can rebuild.** Only the Villaviciosa product. The maps of Foz, Eo
  and Mundaka and of the ten preliminary sites, the hand-cleaned and the rule-made masks,
  the constants C, the tidal datums and their contours, and the emersion layers of the
  viewer were made with the same library outside this branch, and are published here as
  results. `site/scripts/data_layers.py`
  only turns them into the viewer's files.

## 2. The layers

| layer | units | what it is |
|---|---|---|
| Elevation, z | m | The water level at which the pixel turns from dry to wet, read on the tide that reaches it: the tide at the mouth, delayed by the pixel's τ. |
| Uncertainty, σ | m | The width of the pixel's dry-to-wet transition: one of eight values, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45 and 0.65 m. A larger σ is a less sharp transition. It is neither a standard error of z nor an accuracy measured against a survey. |
| Emersion time | % of time | The share of the map's period during which the pixel is out of the water: how often its delayed tide, h(t − τ), stays at or below z, with the tide taken every minute. |
| Delay, τ | min | How much later than at the mouth the tide reaches the pixel: one delay per 3-km section of distance from the mouth along the water, then a smooth curve through the section delays and a point of 0 min at the mouth (Gaussian weights of standard deviation 3 km), never decreasing inland and rounded to the minute. Pixels near the mouth already receive about a minute; the innermost pixels receive less than the last section's fitted delay ([`METHOD.md`](METHOD.md#5-from-sections-to-every-pixel)). Each site has its own colour scale. At Mundaka the layer also draws lines of equal delay ([section 4](#4-tide-lines)). |
| Tide lines | m | Contours of the elevation at the tidal datums ([section 4](#4-tide-lines)). |

MAREA estimates elevations only inside the intertidal mask: the pixels seen wet in 10 to
90 % of more than 30 clear looks. Inside it, a pixel without an elevation failed a guard
of the fit ([`METHOD.md`](METHOD.md#5-from-sections-to-every-pixel)). The delay is
defined on the whole mask. The map viewer draws every layer, the delay included, only
on the pixels on the map ([section 1](#1-the-sites)).

The viewer's legend gives the emersion time as the % of hours out of the water.

## 3. Heights: NMMA and the constant C

MAREA's elevations are on the zero of the tide used to make each map: the mean level of
EOT20 where the tide is EOT20 alone (Villaviciosa, Corme, Noia and Betanzos), and the mean
of EOT20 and the median-removed gauge at the other sites. That zero is close to the local
mean sea level over the map's period, but it is not a national datum.

The map viewer shows heights in **NMMA**, metres above the *Nivel Medio del Mar en
Alicante*, the mean sea level at Alicante that is the zero of Spanish levelling (REDNAP):

**height in NMMA = height on the map's zero + C**

C is the height in NMMA of the map's zero. For each image of a map, it is read from a
Puertos del Estado tide gauge whose zero is tied to NMMA, as the 25-hour mean of the
difference between the gauge and the map's tide around the image. C is the mean over the
map's images that have gauge data. Its 1σ uncertainty combines the tie of the gauge to
NMMA, the geoid model (EGM08-REDNAP), the transfer from the gauge to the estuary and the
spread over the images. Where the map's tide is EOT20 alone, the same 25-hour means of the
Ferrol 2 gauge give C by the rule used at Villaviciosa, although the gauge lies outside the
estuary. The 1σ of the preliminary sites is not measured: it is taken from the sites of
the first table whose C was built the same way (Foz and Eo with a gauge in the tide,
Villaviciosa with EOT20 alone). C was computed outside this branch, from gauge records
that the branch does not redistribute; the library does not compute it.

| site | C (m) | 1σ (m) | gauge of the tie |
|---|---|---|---|
| Ría de Foz | 0.3151 | 0.047 | Ferrol 2 |
| Ría del Eo | 0.3240 | 0.047 | Ferrol 2 |
| Ría de Villaviciosa | 0.3778 | 0.055 | Ferrol 2 (91 of the 97 images) |
| Ría de Mundaka (Urdaibai) | 0.3742 | 0.042 | Bilbao 3 |
| Ría de Corme e Laxe (preliminary) | 0.3205 | 0.055 | Ferrol 2, 62 km away (106 of the 131 images) |
| Ría de Noia, inner part (preliminary) | 0.3112 | 0.055 | Ferrol 2, 94 km away (113 of the 143 images) |
| Ría de Betanzos (preliminary) | 0.3258 | 0.055 | Ferrol 2, 16 km away (143 of the 189 images) |
| Ría de Ferrol (preliminary) | 0.3359 | 0.047 | Ferrol 2 (130 of the 171 images) |
| Ría de Ortigueira (preliminary) | 0.3436 | 0.047 | Ferrol 2 (92 of the 130 images) |
| Ría de Tina Menor (preliminary) | 0.3739 | 0.055 | Santander 2, 56 km away (56 of the 123 images) |
| Ría de San Vicente de la Barquera (preliminary) | 0.3737 | 0.055 | Santander 2, 50 km away (51 of the 117 images) |
| Ría de Mogro (preliminary) | 0.3771 | 0.055 | Santander 2, 13 km away (58 of the 111 images) |
| Bahía de Santander (preliminary) | 0.3689 | 0.055 | Santander 2 (50 of the 99 images) |
| Marismas de Santoña (preliminary) | 0.3750 | 0.055 | Santander 2, 26 km away (104 of the 176 images) |

* The poster rounds the Villaviciosa constant to 0.38 m.
* At the five Cantabrian preliminary sites (Tina Menor, San Vicente de la Barquera, Mogro,
  Santander and Santoña) the tide includes the Santander 2 gauge, whose benchmark has only a
  1998 tie to NMMA. Its zero is tied instead through the coast's REDNAP-tied mean sea level
  of 2011, with the gauge's REDMAR datum sheet: the gauge's mean level of 2011 is put at that
  sea level. With the 1998 tie, C would be about 0.10 m higher. The 1σ of these sites is
  taken from Villaviciosa, whose zero rests on the same assumption.
* The data files (the viewer's value images and the product files) keep the map's zero:
  add C, where the site has one, to read them in NMMA.

## 4. Tide lines

Two kinds of tide line are drawn: contours at the tidal datums, in the **Tide lines**
layer, and, at Mundaka, lines of equal delay, with the **Delay** layer.

**Contours at the tidal datums.** Each datum is a level of the tide at the site. The line
joins the places whose elevation equals that level: the ground below it is covered for
longer, the ground above it for less. The datums come from the EOT20 model at each map's
tide point, every 10 minutes over the 19 years 2007–2025, longer than the 18.6-year cycle
of the Moon's orbit. They and the contours were computed outside this branch.

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

The contours are drawn on MAREA's elevation on the pixels on the map
([section 1](#1-the-sites)), lightly smoothed (a Gaussian of one pixel
that ignores empty pixels, with holes of up to 9 pixels filled). Lines shorter than 100 m
are dropped, and the rest are simplified to 2.5 m. A datum is drawn only where the map
has values on both sides of it: between the 1st and the 99th percentiles of the mapped
elevation. Only MLWN, MSL and MHWN are drawn: all three at Corme, Noia, Betanzos,
Ortigueira, Foz, Eo and Villaviciosa, MLWN and MSL at Ferrol, at the five Cantabrian sites
and at Mundaka; the other datums lie outside that range. The images never see the highest tides
([`METHOD.md`](METHOD.md#9-what-the-images-cannot-tell)), so MHWS and HAT lie above it,
and MLWS and LAT below it. The map's legend lists all seven datums and marks those that
are drawn.

For example, at Villaviciosa the datums are, in m above NMMA: LAT −2.02, MLWS −1.47,
MLWN −0.46, MSL +0.38, MHWN +1.19, MHWS +2.20 and HAT +2.79.

**Lines of equal delay (Mundaka).** They are drawn at Mundaka only, not at the preliminary
sites. In the Ría de Mundaka MAREA estimates that the delay
grows to more than an hour, so the **Delay** layer also draws lines of equal delay every
10 minutes: the places the tide reaches the same number of minutes after the mouth
(co-tidal lines). They follow the delay of each pixel, from the same smooth profile of the
section delays, on the pixels on the map only, so they stop at the channels. Pieces
shorter than 20 m are dropped: no line is drawn beyond 50 min, where the flats of the
inner reach are a few pixels wide.

**Files.** For each site, `tide_lines.geojson` holds the contours, with the properties
`datum`, `level_m` (on the map's zero), `level_nmma_m` (above NMMA), `units`, `label`
(for example "MLWN −0.46 m", in NMMA), `length_m` (before simplification) and `source`,
and a `metadata` block (the definitions, C, the licence and the attribution). The seven
levels, and whether each is drawn, are in the site's `manifest.json`: the `tide_lines`
layer, field `levels` (`id`, `name`, `level` on the map's zero, `nmma`, `drawn`). At
Mundaka, `cotidal.geojson` holds the lines of equal delay, with the properties `tau_min`,
`units`, `s_km` (where the delay first reaches that value, km from the mouth), `label`,
`length_m` and `source`. Coordinates are WGS84, with 5 decimals.

## 5. The tide at each image

Notebook [`02_tide_per_image`](../notebooks/02_tide_per_image.ipynb) places each of the 97
Sentinel-2 images of the Villaviciosa record at the height of the tide when the satellite
passed, says whether the tide was rising or falling, gives the delayed level that each
section saw, and says how much of the tide range the images cover. The images are the
usable ones of the method's screening
([`METHOD.md`](METHOD.md#2-one-pixel-seen-many-times)).

## 6. Map viewer files

* **Grid.** The layers of a site share one grid in Web Mercator (EPSG:3857), of about
  6.89 m, which is 5 m on the ground at these latitudes, over the pixels on the map plus
  200 m. Each 10 m pixel of a product is about 2 × 2 grid pixels, assigned by nearest
  neighbour (no interpolation).
* **Overlay images.** Coloured RGBA PNG, for display only: at twice the grid's resolution
  (2.5 m on the ground), each overlay pixel interpolated bilinearly from the grid pixels
  around it that have a value, the edge of the data faded over one grid pixel, and
  transparent where there is no value. Colours, those of the poster's maps: elevation
  matplotlib's viridis (dark purple low, yellow high), uncertainty ColorBrewer Purples,
  emersion time Greens, delay Oranges without its lightest 15 %. Overlays wider or
  taller than 4096 pixels come in tiles. The readout uses the value images.
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

**A layer as a GeoTIFF.** From the repository root, this turns a value image into a
GeoTIFF on the viewer's grid, with the elevations in NMMA:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import from_origin

from pyintertidal.raster import write_geotiff

site, layer_id = "villaviciosa", "elevation"           # any site of site/public/data/sites/
folder = f"site/public/data/sites/{site}"
manifest = json.load(open(f"{folder}/manifest.json", encoding="utf-8"))
layer = next(L for L in manifest["layers"] if L["id"] == layer_id)
v = layer["values"]
rgb = np.asarray(Image.open(f"{folder}/{v['file']}").convert("RGB"), np.int64)
code = 256 * rgb[..., 0] + rgb[..., 1]                 # 0 means no value
for col, row, expected in v["check"]:                  # the manifest's own probes
    assert code[row, col] == expected
value = np.where(code > 0, v["offset"] + v["scale"] * code, np.nan).astype(np.float32)
C = manifest["heights"]["C"]                           # None at the sites without an NMMA link
if layer.get("heights") and C is not None:             # elevations: on the map's zero, + C for NMMA
    value = value + np.float32(C)
g = manifest["grid"]
transform = from_origin(g["x0"], g["y0"], g["res"], g["res"])   # top-left corner, Web Mercator metres
Path("runs").mkdir(exist_ok=True)                      # git ignores runs/
write_geotiff(f"runs/{site}_{layer_id}.tif", value, transform, "EPSG:3857")
```

The GeoTIFF keeps the viewer's 6.89 m Web Mercator grid; reproject it for other uses.

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
| `hyp_z`, `hyp_area`, `hyp_lo`, `hyp_hi` | m, km² | hypsometric curve (area below each level) and its envelope: the 5 to 95 % range of 30 curves with each z moved at random by its σ, a sensitivity band rather than a confidence interval |

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
* **Tide gauges:** IOC Sea Level Station Monitoring Facility (VLIZ/IOC),
  <https://doi.org/10.14284/482>, for non-commercial use (CC BY-NC 4.0); Ferrol 2 and
  Bilbao 3 are stations of Puertos del Estado.
* **Website basemaps:** PNOA orthophoto and IGN base map, © Instituto Geográfico
  Nacional de España, CC BY 4.0 (scne.es). They are loaded at run time from IGN's
  services and not redistributed.
* **3D terrain**, where the map viewer offers it (wide screens): elevation tiles loaded at
  run time, with their attribution on the map.
* **Website code:** the bundled libraries keep their licences, collected in
  [`docs/licenses/`](../docs/licenses/).
