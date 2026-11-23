# Notebook sample: Ría de Villaviciosa

A small real dataset for the example notebooks in `notebooks/`. It is a block of 270 × 270 pixels of 10 m
(2.7 km × 2.7 km) of the Ría de Villaviciosa (Asturias, Spain), cut from the Sentinel-2 cube that MAREA's 10 m
product of the ría (`products_marea/`) was made from. It holds everything needed to run MAREA's elevation step and
to compare the result with that product. The block covers the western part of the central flats, the main channel
and the inner reach. Across it, the delay MAREA applied grows from 5 to 15 minutes.

| File | Size | Contents |
|---|---|---|
| `villaviciosa_sample.nc` | 18.4 MB | netCDF-4 (CF-1.8): the data, variables below |
| `villaviciosa_tide_levels.csv` | 37 kB | the water level of each of the 97 scenes at the overpass (`tide_m`) and for pixels 1 to 16 min late (`tide_m_delay_01min` to `tide_m_delay_16min`) |
| `README.md` | | this file |
| `SHA256SUMS.txt` | | the sha256 of the three files above |

## Variables of `villaviciosa_sample.nc`

| Name | Dimensions | Contents |
|---|---|---|
| `t` | t (97) | overpass instant of each scene, UTC: the record of MAREA's product |
| `x`, `y` | x, y (270 each) | pixel centres, m, WGS 84 / UTM zone 30N (EPSG:32630); `crs` is the grid mapping |
| `B03`, `B08` | t, y, x | Sentinel-2 L2A band 3 (green) and band 8 (near infrared): surface reflectance × 10000, stored as int16 exactly as in the cube; −32768 means no value; values below 0 occur over dark water |
| `SCL` | t, y, x | Sentinel-2 L2A scene classification, classes 0 to 11 |
| `clear` | t, y, x | 1 where SCL is 4, 5, 6 or 7: the looks MAREA uses |
| `tide_level` | t | EOT20 water level at the boundary point at each overpass, m |
| `tide_level_delayed` | t, delay (226: −45 to 180 min) | the same level read `delay` minutes before the overpass, m: what a pixel whose water is that many minutes late sees |
| `intertidal` | y, x | 1 on the 15,064 pixels of this block that MAREA's product inverted |
| `z_marea` | y, x | the product's elevation, m, on the scale of the EOT20 levels (about mean sea level); 14,171 pixels hold a value |
| `sigma_marea` | y, x | the product's width of the fitted flooding curve, m, shown as the uncertainty of z (not a standard error) |
| `delay_marea` | y, x | the delay the product applied to each pixel, min, from a smooth curve through the section delays: 5 to 15 here, 0 on the pixels not connected to the mouth |
| `s_km` | y, x | distance from the mouth along the water, km |
| `band_marea` | y, x | the 3-km section of each pixel, −1 where it has no distance |
| `band_centre_km`, `band_delay_min`, `band_delay_lo_min`, `band_delay_hi_min` | band (3) | the product's section delays and their 95 % scene-bootstrap interval; section 0 is the mouth, pinned at 0 |
| `tide_series` | series_time (157,824) | EOT20 water level at the boundary point every 10 minutes, 2023-01-01 to 2025-12-31, m |

NDWI is not stored. Compute it as MAREA does, in float32: NDWI = (B03 − B08) / (B03 + B08), NaN where
B03 + B08 = 0. Over dark water NDWI can lie far outside [−1, 1] (here from −197 to 125, on 1.45 % of the values),
because the reflectances can be negative there.

## Where and when

- **Where.** Ría de Villaviciosa, Asturias, northern Spain. The block is rows 330 to 599 and columns 290 to 559 of
  the product's 915 × 915 frame. Its edges are x 303,920 to 306,620 m and y 4,818,840 to 4,821,540 m (UTM 30N),
  about 43.497 to 43.522 N and 5.392 to 5.426 W. The water levels are read at the study area's centroid,
  43.50902 N 5.41613 W.
- **When.** The 97 Sentinel-2 scenes of the product's record, 2023-01-06 to 2025-12-28: the scenes of 2023 to 2025
  with at most 10 % cloud over the transition zone. They were acquired between 11:06 and 11:25 UTC. The tide
  series covers 2023 to 2025.

## How it was made and checked

The block was cut from the record's scenes in the cube. EOT20 was read at the scenes' overpass instants, shifted by
every whole minute from −45 to 180, at the point the product used, and the product's per-pixel layers were copied.
Nothing was downloaded. The checks:

1. NDWI and the clear looks computed from the sample equal the looks the product was run on, bit for bit, on all
   15,064 pixels.
2. MAREA on the sample reproduces the product exactly. `pyintertidal.marea.invert_series`, run per pixel with
   `tide_level_delayed` at the pixel's `delay_marea`, searching elevations between the minimum and the maximum of
   `tide_level` and with no quality rule, gives the product's z and sigma on all 15,064 pixels, bit for bit.
3. The CSV levels, written with 17 significant digits, read back equal to the netCDF table, bit for bit.
4. The delay belongs to the whole ría. The delay estimator (`pyintertidal.tide_estimators.m2a_rasch`), run on
   the whole ría, returns the product's section delays bit for bit: 0, 5.05 and 20.74 min. Run on this block alone,
   it sees part of two sections and none of the mouth section, and returns 4.16 and 22.71 min. Both lie inside the
   product's 95 % intervals (0 to 17.9 and 13.1 to 28.7 min), but 573 of the 15,064 elevations change with them. To
   reproduce the product, use `delay_marea`.
5. The two bands, not NDWI, are stored. netCDF has no float16, and NDWI rounded to float16 or to int8 changes some
   elevations. The two int16 bands are the cube's own values and keep the file under 20 MB.

## How to use it

Run from the repository root, with the `pyintertidal` library of this repository:

```python
import numpy as np
import xarray as xr
from pyintertidal import marea

s = xr.open_dataset("data/villaviciosa_sample.nc")
g, n = s["B03"].values, s["B08"].values                # float32, NaN = no value
with np.errstate(invalid="ignore", divide="ignore"):
    ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
rows, cols = np.nonzero(s["intertidal"].values == 1)   # MAREA's pixels
Y = np.nan_to_num(ndwi[:, rows, cols], nan=0.0).astype(float)
C = (s["clear"].values[:, rows, cols] == 1).astype(float)
delay = s["delay_marea"].values[rows, cols]
lo, hi = float(s["tide_level"].min()), float(s["tide_level"].max())
z = np.full(len(rows), np.nan)
for d in np.unique(delay):
    j = delay == d
    h = s["tide_level_delayed"].sel(delay=int(d)).values
    z[j], _ = marea.invert_series(Y[:, j], C[:, j], h, lo, hi)
# z.astype("float32") equals s["z_marea"].values[rows, cols], NaN included
```

`notebooks/01_marea_villaviciosa.ipynb` runs this and more. The delay fit of check 4 uses the settings of
`configs/m2.yaml`: sigma0 0.20 m, sigma profile 0.06, 0.10, 0.15, 0.22, 0.32 and 0.45 m, 60 elevation points, at
most 1200 pixels per section, delays bounded to −45 to 180 min, monotone, and the random generator seeded with
20260850. The delay fit reads the tide at 16 delays (−45 to 180 min, every 15 min) and interpolates between them;
those levels are columns of `tide_level_delayed` too.

## Caveats

- The satellite passes between 11:06 and 11:25 UTC, so the record never sees the highest tides. The overpass
  levels span −2.17 to +1.15 m, the continuous series −2.40 to +2.41 m. MAREA only searches elevations inside the
  record's range (global attribute `level_range_m`).
- The block holds part of the ría only. See check 4 before re-estimating the delay from it.

## Licences and attribution

This sample: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Cite the repository it comes with and keep
the attributions below.

- **Sentinel-2** (`B03`, `B08`, `SCL`, `clear`): Contains modified Copernicus Sentinel data 2023-2025, accessed
  through the Copernicus Data Space Ecosystem (openEO). Copernicus Sentinel data are free, full and open; see the
  Copernicus Sentinel data legal notice.
- **Water levels** (`tide_level`, `tide_level_delayed`, `tide_series`, the CSV): EOT20 ocean tide model, CC BY 4.0.
  Hart-Davis, M., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M. and Seitz, F. (2021): EOT20 - A global
  Empirical Ocean Tide model from multi-mission satellite altimetry. SEANOE, https://doi.org/10.17882/79489. Paper:
  Earth System Science Data 13, 3869-3884, https://doi.org/10.5194/essd-13-3869-2021.
- **MAREA layers** (`z_marea`, `sigma_marea`, `delay_marea`, `s_km`, `band_*`): the authors' product, CC BY 4.0
  with this sample.
