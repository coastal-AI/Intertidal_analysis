# MAREA: a method for mapping intertidal topography using Sentinel-2 openEO data cubes

*The tide arrives late: MAREA reads the delay from satellite images and corrects the maps for it.*

Research code behind the CARTOGRASS 2026 poster (Cartagena, 24-26 November 2026).

* **What it gives:** maps of intertidal elevation, with its uncertainty, and
  of emerged time (emersion time: the share of time the ground spends out of
  the water), at 10 m from the Sentinel-2 archive, with the delay of the tide
  inside estuaries and tidal basins corrected.
* **Status:** the validation is in progress (October 2026). This page
  describes the method and the validation design and reports no validation
  result.
* **Poster:** [PDF, A0, 2.3 MB](poster/CARTOGRASS2026_poster_draft_2026-10-20.pdf)
* **Cite:** [how to cite](#how-to-cite) · **Contact:** Jorge Alías Álvarez,
  jorge.alias.alvarez@gmail.com

## What MAREA is

MAREA maps the elevation of intertidal flats, and the share of time each spot
spends out of the water (emerged time), from the Sentinel-2 archive. A 10 m
pixel on a tidal flat is seen many times by the satellite, at different
stages of the tide; whether it looks wet or dry in each image, together with
the water level at the time of that image, fixes its elevation
(Bishop-Taylor et al. 2019; Granadeiro et al. 2021). Global tide models give
the tide at the open coast, but inside estuaries and tidal basins the tide
arrives up to 1.5-2 hours later, so maps made with the coastal tide misjudge
when the flats flood. Granadeiro et al. (2021) showed that the Sentinel-2
archive holds this delay. MAREA measures it section by section along the
channel: one delay for each 3-km section (zero at the mouth, never
decreasing inland), read from the images themselves. It then reads the tide
at the mouth that many minutes earlier, and only then works out the
elevation of every pixel. The images come as openEO data cubes from the
Copernicus Data Space Ecosystem; the tide at the mouth comes from the global
tide model EOT20 and, where they exist, from tide gauges at the mouth.
(*Marea* is Spanish for tide; in the code the name reads *Mudflat Altimetry
via Remotely-Estimated tides from the Archive*.)

## The poster

**MAREA: a method for mapping intertidal topography using Sentinel-2 openEO
data cubes.** Jorge Alías Álvarez¹, Fernando García-González², Pablo
González¹. ¹University of Oviedo, Spain; ²Instituto Español de Oceanografía
(IEO-CSIC), Spain. CARTOGRASS 2026, Simposio Ibérico de Cartografía y
Seguimiento de Praderas Marinas, Cartagena, 24-26 November 2026.
Authors as on the poster; the affiliations are being confirmed with the
co-authors.

* The poster as a PDF (A0, 2.3 MB; draft of 20 October 2026):
  [`poster/CARTOGRASS2026_poster_draft_2026-10-20.pdf`](poster/CARTOGRASS2026_poster_draft_2026-10-20.pdf).
  Its Results panels are left blank, and its Conclusions marked pending,
  until the validation is complete.
* The QR code in the poster's bottom strip opens this page.
* Sources, figures and the scripts that draw them: [`poster/`](poster/README.md).

## How MAREA works: follow one pixel

![Flow chart in six cards: (1) the pixel is seen many times in a Sentinel-2 data cube and each cloud-free look says wet or dry; (2) the tide at the mouth reaches the pixel's 3-km section some minutes later, and the pixel is wet when that water stands above its ground; (3) the section's delay is the one that best splits the wet and dry looks of its pixels; (4) the midpoint of the pixel's flooding curve on the corrected tide is its elevation and the width of the curve its uncertainty; (5) its emerged time is the share of time the corrected tide stays below its ground; (6) all pixels give maps of elevation, uncertainty and emerged time.](poster/figures/cg/method_flow.png)

*How MAREA works, followed for one imaginary pixel (the method panel of the
poster). Schematic: the looks and the tide are simulated and the delays shown
are examples, not results; the outline and the 3-km sections are those of the
Ems-Dollard, from Sentinel-2. Drawn by `poster/tools/cg/fig_method_flow.py`.
Tap the figure to enlarge it.*

### Step by step, with the code

Each step first says what happens; the names in `code font` point to the
functions and files that do it, for readers who want to reuse the code.

1. **Images.** Sentinel-2 L2A at 10 m, 2023-2025 (Sentinel-2A, 2B and 2C),
   downloaded as openEO data cubes from the Copernicus Data Space Ecosystem.
   A *look* is a cloud-free view of the pixel: its scene classification
   (SCL, the map of clouds, shadows, water and land that comes with every
   Sentinel-2 L2A image) is one of the classes 4-7 (vegetation, bare
   ground, water, unclassified: not cloud, shadow or snow), in a scene with
   little cloud (at most 5 % cloudy over the frame or, failing that, 10 %
   over the transition zone between permanent water and land). In each look
   the pixel is wet when the water index NDWI = (B03 - B08) / (B03 + B08),
   from green and near-infrared light, is above 0. Code:
   `pyintertidal.cube.SentinelCube`
   and `experiments/download_cube_zarr.py` (download),
   `pyintertidal.stability.reference_and_clouds` and `filter_dates` (the
   scene screening), `pyintertidal.marea.extract` (the looks).
2. **Tide at the mouth, h(t).** The water level at each image's acquisition
   time (the catalog time of the scene, `overpass_times.json`) from EOT20
   (through eo-tides and pyTMD), averaged with the tide gauges at the mouth
   where they exist (four of the five study sites;
   `pyintertidal.boundary.make_boundary`). Heights are in meters on the
   vertical scale of that tide series (about mean sea level), not on a
   national datum such as Alicante (Spain) or NAP (the Netherlands); a
   comparison with a survey first removes the median offset and reports it
   separately.
3. **The delay of its 3-km section, τ.** The distance from the mouth is
   measured along the water (`pyintertidal.geometry.along_distance`), and the
   intertidal zone is cut into sections of 3 km of that distance
   (`pyintertidal.tide_estimators.make_bands_fixed`; a section with fewer
   than 500 pixels is merged with the next). Each section gets one delay τ:
   the one for which the mouth tide read τ minutes earlier, h(t - τ), best
   explains the wet and dry looks of all its pixels together. In plain words:
   with the right delay, the looks taken while the water stood above a
   pixel's ground come out wet and those taken while it stood below come out
   dry; with a wrong delay, wet and dry looks mix (panel 3 of the figure, "the
   cleanest split"). The fit uses a seeded random sample of at most 1,200
   pixels per section (`configs/m2.yaml`) and a profiled Bernoulli
   likelihood: for each candidate τ, every pixel refits its own elevation
   and transition width, so that no delay is judged with ground heights
   fitted for another (`pyintertidal.tide_estimators.m2a_rasch`). τ is zero
   at the mouth and can only grow inland; refitting it on 30 resamples of the
   scenes (a bootstrap) gives its 95 % interval. Each pixel then receives its
   own delay, read at its distance from the mouth on a smooth curve through
   the section delays (a Gaussian kernel of 3-km bandwidth along the
   channel), rounded to the minute. MAREA shifts the tide in time, not in
   height: wet/dry images can time the tide but cannot measure its
   amplification ([`docs/METHOD.md`](docs/METHOD.md)).
4. **Its elevation, z.** On the corrected tide, the pixel's water index
   follows an S-shaped flooding curve: low (the dry pixel) while the water
   is below its ground, high (the flooded pixel) once the water is above it.
   The curve is NDWI = a + b Φ((h(t - τ) - z) / σ), where Φ is the S-shaped
   cumulative normal distribution, a the pixel's dry level and b the jump
   from dry to wet; it is fitted by least squares
   (`pyintertidal.marea.invert_series`). Its midpoint z is the elevation
   (meters, on the same scale as h). Its width σ says how gradually the
   pixel floods: one of eight steps from 0.03 to 0.65 m, and what the poster
   shows as the uncertainty. Whether σ is a calibrated error bar is one of
   the open questions; the "Uncertainty" paragraph of
   [`docs/METHOD.md`](docs/METHOD.md) says what σ is and how it is computed.
   A pixel is left empty unless it has at least eight cloud-free looks, a
   plausible curve (wet-dry contrast b between 0.15 and 2.5, |a| < 2.5) and, since
   2026-10-06, at least three looks with the water below its ground and three
   above it. That last condition is the quality rule
   `marea.QUALITY_MIN_BRACKET = 3`, the default of `reconstruct` and of
   validation v2. The 10 m production products of the study sites, among them
   the one in `products_marea/`, were made before the rule existed: their
   `result.json` has no `quality_min_bracket` key, which means no bracket
   rule. The method-comparison scripts v14-v19 score them with that earlier
   rule and report the rule of three as a variant.
5. **Its emerged time, E(z).** The share of time the corrected tide stays
   below the ground, h(t - τ) < z, over a continuous tide record (every time
   step, not only the image times): `1 - pyintertidal.hydroperiod.hydroperiod(z, h)`.
6. **All pixels: the maps.** Elevation with its uncertainty, and emerged
   time: a base layer for the potential habitat of *Zostera noltei*, whose
   limits ecologists set.

`pyintertidal.marea.reconstruct(cube, out_dir)` runs steps 1-4 on a
downloaded cube and writes `marea.npz`, `result.json` and `figure.png`
([`docs/products.md`](docs/products.md) describes every product). Two parts
of steps 1-2 are left to the caller:

* **The scene screening.** `reconstruct` uses every scene of the cube that
  has an acquisition time (from `overpass_times`, or else from the STAC
  catalog). The production runs screened the scenes first
  (`pyintertidal.stability.reference_and_clouds` and `filter_dates`) and
  passed only the times of the screened scenes. Its default `min_scenes=80`
  refuses shorter records; the production runs used 40.
* **The mouth gauges.** By default its tide at the mouth is EOT20 alone, as at
  Villaviciosa. For the average with the mouth gauges used at the other four
  sites, pass `boundary=pyintertidal.boundary.make_boundary(aoi, n_gauges=1)`
  (two at the Westerschelde; `exclude=` names the gauges kept as checks).
  Each study site's choices (gauges, the gauges held out as checks, the sea
  edge of the frame) are in `experiments/benchmark/config.py` (`BOUNDARY`,
  `HELD_OUT`, `MOUTH_SIDE`).

The derivation of the pixel model and of the delay estimator is in
[`docs/METHOD.md`](docs/METHOD.md); a worked example on one real pixel, in
the development state of September 2026, is in
[`docs/one_pixel_marea_example.md`](docs/one_pixel_marea_example.md).

## Validation: design and status

**Status (26 October 2026): in progress.** The specification is
[`docs/validation_v2_spec.md`](docs/validation_v2_spec.md). Its code is
committed up to milestone V2: the inputs, the processing settings, the water
levels, the reference data and the command line (V0, V1), and the comparison
of the delay with the tide gauges inside the estuaries (V2). The runs on the
10 m data cubes are under way; their results will be added to this
repository, and to the poster's Results and Conclusions, when they are
complete.

The design, in short:

* **Two questions.** Is MAREA's delay a good estimate of the delay that tide
  gauges inside the estuary measure, compared with other ways of estimating
  it (from images, from global and regional tide models, from physics)? And
  is MAREA a good method for intertidal elevation compared with the state of
  the art, and where? The non-circular test of the delay is whether the
  correction improves the elevation against independent surveys (MAREA
  against the same inversion with no delay).
* **Same inputs for every method**, across a pre-registered space of
  processing settings (cloud handling, water index, water detection, band
  preprocessing, time window, number of images); every competitor also gets
  an "as published" row; deviations are declared.
* **Five sites**: the Ría de Villaviciosa and the Ría de Ferrol (Spain), the
  Westerschelde, the Vlie basin of the Wadden Sea and the Ems-Dollard (the
  Netherlands and Germany).
* **Independent checks, never used to make the maps**: an RTK-GNSS survey and
  PNOA airborne LiDAR at Villaviciosa; Rijkswaterstaat's Vaklodingen depth
  surveys at the Westerschelde, Wadden Sea and Ems-Dollard sites; tide gauges
  inside four of the sites.
* **Home advantage, declared**: MAREA's settings were developed on these same
  sites, gauges and surveys. The specification lists every check that
  influenced a setting (the comparison with the Harlingen gauge widened the
  range of delays the fit may try), reports every delay result also without
  those gauges, and lists the results the authors had seen before the
  pre-registration lock (§1.7 and §9.9).
* **The RTK survey**: validation v2 reads only its development points.
  `pyintertidal.rtk.load_rtk` returns those only, `tests/test_r1_guard.py`
  checks that no other code asks for the rest, and only
  `experiments/v3_open_reserved.py` may open the reserved part, logging every
  opening to `results/v3_aperturas.jsonl`. That log holds one run, on
  2026-08-21, which scored four development maps made before the current
  MAREA products; the log records that this run consumed the reserved set,
  and `docs/PHASE_LOG.md` reports it with its numbers. The reserved points
  are therefore not an untouched blind test; validation v2 does not score
  them. The survey files in `data/` hold every point: the guard binds this
  repository's code, not the files.
* **Only 10 m results will be reported**, and a method that cannot give a
  result at a site is reported as NOT COMPUTABLE there; its rules are never
  relaxed to obtain one.

The "Pilot result" box on the poster is earlier work, from the submitted
abstract (the poster restates the abstract's percentage against the mean
spring range); it is not part of validation v2. Alongside the validation, the
method-comparison scripts `experiments/v14`-`v19` (number of scenes, length
of the time window, errors by section and by elevation, computing cost, and
tables at one fixed configuration) are running on the existing 10 m inputs of
each site; their outputs are not in this repository.

## Repository map

| path | what is there |
|---|---|
| [`pyintertidal/`](pyintertidal/README.md) | the MAREA library: data cubes, water detection, tides, the delay estimator, the per-pixel inversion, derived products |
| [`experiments/`](experiments/README.md) | the scripts: validation v2 ([`validation_v2.py`](experiments/validation_v2.py) and [`benchmark/`](experiments/benchmark/README.md)), the method comparison on the existing 10 m inputs (`v14`-`v19`, [`fixedcfg/`](experiments/fixedcfg/README.md)), and the development record |
| [`tests/`](tests/README.md) | unit tests on small synthetic data, and the guard of the reserved RTK points |
| [`configs/`](configs/README.md) | pre-registered thresholds (`configs/m2.yaml` is read by the production engine) |
| [`data/`](data/README.md) | the RTK-GNSS field survey of the Ría de Villaviciosa |
| [`docs/`](docs/README.md) | the method, the validation specification, the products guide, worked examples and their figures |
| [`notebooks/`](notebooks/README.md) | the notebook that explores the validation cubes, and the one-pixel walkthrough |
| [`poster/`](poster/README.md) | the CARTOGRASS 2026 poster: LaTeX sources, figures, the scripts that draw them, the PDF |
| [`products_marea/`](products_marea/README.md) | the 10 m MAREA product of the Ría de Villaviciosa (97 screened scenes of 2023-2025, EOT20 at the mouth, made before the quality rule of step 4), used by the second example below |
| [`products_villaviciosa/hsr_eot20_2023-2025.tif`](products_villaviciosa/README.md) | the canonical 10 m grid of Villaviciosa (EPSG:32630), on which `pyintertidal.rtk.load_rtk` places the RTK survey and the second example writes its GeoTIFF |
| `products_ferrol/`, `products_{escalda,wadden,ems}_10m/` | `overpass_times.json`: the acquisition time of every scene of the Ferrol, Dutch and German sites |
| [`sealed/`](sealed/README.md) | the registry of sealed data files (`pyintertidal.seal`) |
| [`data_v4/gauges/ioc_stations.json`](data_v4/README.md) | the IOC station list read by `pyintertidal.boundary.make_boundary` |
| [`results/`](results/README.md) | verdicts of the development experiments, kept as the record of what had been seen before the validation's pre-registration lock; not validation results |
| [`archive/`](archive/README.md) | legacy code, notebooks and the earlier poster draft, kept for the record and not maintained |

## Installation

```bash
git clone --depth 1 https://github.com/coastal-AI/Intertidal_analysis.git   # current files only; a full clone also fetches the working branch's history (several hundred MB)
cd Intertidal_analysis
python -m venv .venv            # Python 3.13 (the validation runs use 3.13.3)
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install numpy scipy pandas xarray rasterio shapely matplotlib requests   # enough for the examples
```

* `import pyintertidal` needs numpy, scipy, pandas, xarray, rasterio,
  shapely, matplotlib and requests (enough for examples 1 and 2);
  `marea.reconstruct` on a real cube also needs PyYAML, pyproj, netCDF4 or
  h5netcdf, pyTMD and eo-tides.
* The full, pinned environment of the validation runs is
  [`requirements-cluster.txt`](requirements-cluster.txt) (openEO, pyTMD and
  eo-tides, geopandas, scikit-image, netCDF4, zarr, and more). It pins a CPU
  build of PyTorch for one optional cloud mask, so install in this order (the
  PyPI builds of torch and torchvision are the CUDA ones):

  ```bash
  pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements-cluster.txt
  ```
* `requirements.txt` is an older, shorter list that no longer matches that
  environment.
* Only for parts of the validation: R 4.4 with nplr, lmodel2 and mgcv (the
  re-implementation of Granadeiro et al. 2021), and a separate environment
  with DEA Intertidal 2.5.3 (`experiments/fixedcfg/`).

## Minimal examples

Run from the repository root.

**1. MAREA on synthetic data** (no download, about 15 s). Two 3-km sections:
at the mouth the tide is on time, inland it arrives 60 minutes late. The
script plants that delay, recovers it from the wet/dry looks alone, and
inverts the elevations with and without it.

<details>
<summary>Show the code (Python, 65 lines)</summary>

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

from pyintertidal import marea, tide_estimators as te
from pyintertidal.hydroperiod import hydroperiod

rng = np.random.default_rng(0)

# 1. Images: 300 acquisitions near 11:00 UTC over three years (a sun-synchronous satellite).
t0 = pd.Timestamp("2023-01-01 11:00")
days = np.sort(rng.choice(3 * 365, 300, replace=False))
times = t0 + pd.to_timedelta(days, "D") + pd.to_timedelta(rng.normal(0, 5, days.size), "min")
hours = np.asarray((times - t0) / pd.Timedelta(hours=1))


# 2. Tide at the mouth, h(t): a synthetic semidiurnal tide in metres above mean sea level.
def mouth_tide(t_hours):
    return 1.2 * np.cos(2 * np.pi * t_hours / 12.4206) + 0.3 * np.cos(2 * np.pi * t_hours / 12.6583)


# 3. Two sections of channel: at the mouth the tide is on time; inland it arrives 60 min late.
planted_delay_min = np.array([0.0, 60.0])
section = np.repeat([0, 1], 400)                      # 400 pixels per section
z_true = rng.uniform(-0.9, 0.9, section.size)         # ground elevation of each pixel (m)
water = np.stack([mouth_tide(hours - planted_delay_min[k] / 60.0) for k in section], axis=1)
clear = rng.random(water.shape) < 0.7                 # cloud-free looks (70 %)
ndwi = (-0.3 + 0.6 * norm.cdf((water - z_true) / 0.10)
        + rng.normal(0.0, 0.05, water.shape))         # water index of each look
wet = clear & (ndwi > 0)                              # each look: wet or dry

# The tide at the mouth read tau minutes earlier, for every tau the fit may try.
shifts = np.arange(-45.0, 181.0, 15.0)
bank = te.ShiftBank(shifts, [mouth_tide(hours - s / 60.0) for s in shifts])

# 4. One delay per section, from the wet/dry looks alone (zero at the mouth, never decreasing inland).
fit = te.m2a_rasch(wet, clear, bank, section, np.array([1.5, 4.5]), 2,
                   tau_bounds=(-45.0, 180.0), sg_grid=marea.SG_GRID,
                   rng=np.random.default_rng(1))
print("planted delay per section (min):", planted_delay_min)
print("fitted delay per section (min): ", np.round(fit["tau"], 1))

# 5. Each pixel's elevation: the midpoint of its flooding curve on the corrected tide.
Y = np.where(clear, ndwi, 0.0)
lo, hi = bank.at(0.0).min(), bank.at(0.0).max()
z_marea = np.full(section.size, np.nan)
z_nodelay = np.full(section.size, np.nan)
for k, tau in enumerate(fit["tau"]):
    cols = section == k
    z_marea[cols], _ = marea.invert_series(Y[:, cols], clear[:, cols], bank.at(tau), lo, hi,
                                           min_bracket=marea.QUALITY_MIN_BRACKET)
    z_nodelay[cols], _ = marea.invert_series(Y[:, cols], clear[:, cols], bank.at(0.0), lo, hi,
                                             min_bracket=marea.QUALITY_MIN_BRACKET)
inland = section == 1
for label, z in (("without the delay", z_nodelay), ("with the fitted delay", z_marea)):
    err = z[inland] - z_true[inland]
    print(f"inland section, {label}: RMS elevation error {np.sqrt(np.nanmean(err ** 2)):.3f} m, "
          f"{np.isfinite(err).sum()} of {inland.sum()} pixels mapped")

# 6. Emerged time: share of time the corrected tide stays below the ground, over one year
#    of continuous tide (every 10 min), not only at image times.
year = np.arange(0.0, 365 * 24, 1 / 6)
emerged = 1.0 - hydroperiod(z_marea[inland], mouth_tide(year - fit["tau"][1] / 60.0))
print("inland section, emerged time of the lowest and highest mapped pixels:",
      np.round([emerged[np.nanargmin(z_marea[inland])], emerged[np.nanargmax(z_marea[inland])]], 2))
```

</details>

With Python 3.13.3 and the versions pinned in `requirements-cluster.txt` it
prints the following. These numbers describe synthetic data only: they are
not a validation result, and the 0.191 m is not the 0.19 m of the poster's
pilot box.

```
planted delay per section (min): [ 0. 60.]
fitted delay per section (min):  [ 0.  59.8]
inland section, without the delay: RMS elevation error 0.191 m, 400 of 400 pixels mapped
inland section, with the fitted delay: RMS elevation error 0.019 m, 400 of 400 pixels mapped
inland section, emerged time of the lowest and highest mapped pixels: [0.22 0.78]
```

**2. Open the Villaviciosa product kept in the repository** and write its
elevation as a GeoTIFF.

```python
import json

import numpy as np

from pyintertidal.raster import grid_of, write_geotiff

prod = np.load("products_marea/marea.npz")
H, W = prod["shape"]
z = np.full(H * W, np.nan, np.float32)
z[prod["keep"]] = prod["z"]                 # elevation (m, tide-model scale) of each intertidal pixel
z = z.reshape(H, W)

info = json.load(open("products_marea/result.json", encoding="utf-8"))
print(info["n_escenas"], "scenes;", info["n_px_cota"], "of", info["n_px"], "intertidal pixels have an elevation")

# marea.npz stores pixel indices on the site's canonical 10 m grid (EPSG:32630),
# which is the grid of this GeoTIFF:
transform, crs, shape = grid_of("products_villaviciosa/hsr_eot20_2023-2025.tif")
assert shape == (H, W), f"marea.npz frame {H}x{W} is not the canonical 10 m grid {shape}"
write_geotiff("villaviciosa_marea_elevation.tif", z, transform, crs)
```

**3. Real data.** To map a site from scratch you need:

* **A Sentinel-2 L2A cube** with the bands B03, B08 and SCL: a free account on
  the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
  and an openEO login (OIDC, in the browser on first use). For example
  `cube = pyintertidal.SentinelCube(pyintertidal.sites.get("villaviciosa"), ("2023-01-01", "2025-12-31"), water="ndwi")`
  and `cube.ensure(pyintertidal.scenes.connect())` write one netCDF cube.
  The validation cubes (nine bands, one Zarr store per site and year) are
  downloaded with `python -X utf8 -m experiments.download_cube_zarr --login`,
  then `python -X utf8 -m experiments.download_cube_zarr villaviciosa --years 2023-2025`.
  The Zarr stores are read by the validation harness (`experiments/zarr_cubes.py`),
  not by `reconstruct`, which reads the netCDF cube. Every download uses the
  openEO processing credits of the account; a free account's monthly
  allowance limits how many site-years it can fetch.
* **The acquisition time of every scene**: `pyintertidal.overpass.get_overpass_times`
  reads it from the STAC catalog; the study sites' files are kept as
  `products_*/overpass_times.json`.
* **The EOT20 tide model**: download
  <https://www.seanoe.org/data/00683/79489/data/85762.zip> (2.3 GB), extract
  the inner `ocean_tides.zip` and place the files in
  `tide_models/EOT20/ocean_tides/`.
* **Optionally, tide gauges at the mouth**, from the IOC Sea Level Station
  Monitoring Facility: `boundary = pyintertidal.boundary.make_boundary(aoi, tide_model="EOT20", n_gauges=1)`
  downloads the nearest gauge into `data_v4/gauges/` (station list
  `data_v4/gauges/ioc_stations.json`) and averages it with EOT20; pass it as
  `reconstruct(..., boundary=boundary)`. (`python -X utf8 -m experiments.validation_v2 fetch ioc`
  fills the separate gauge cache of validation v2,
  `data_v4/gauges/ioc_2017-2025/`, which `reconstruct` does not read.)

Then `pyintertidal.marea.reconstruct("ndwi_cube_<site>.nc", "products_<site>", tide_dir="tide_models", overpass_times=...)`.
For validation v2, `python -X utf8 -m experiments.validation_v2 check-env`
lists every input each step needs and how to obtain it.

## Data sources and credits

* Contains modified Copernicus Sentinel data (2016-2025), processed with
  openEO on the Copernicus Data Space Ecosystem.
* Tide gauges: IOC Sea Level Station Monitoring Facility; Rijkswaterstaat;
  WSV (PEGELONLINE); Puertos del Estado (REDMAR).
* Bathymetry: Rijkswaterstaat (Vaklodingen). LiDAR: PNOA, © Instituto
  Geográfico Nacional de España (CC BY 4.0). Tide model: EOT20 (DGFI-TUM).
  Basemap of the poster maps: Natural Earth.
* Validation v2 also reads, where installed, other global tide models (FES2022,
  GOT, TPXO), the Copernicus Marine Service regional models (AMM15, IBI) and
  EMODnet Bathymetry; they will be credited with the results that use them.
  The satellite basemap of `pyintertidal.viz.plot_aoi`, seen in some archived
  notebooks, is Esri World Imagery.
* RTK-GNSS survey of the Ría de Villaviciosa: the authors' field survey
  (`data/`).
* Vendored code: `experiments/benchmark/vendor/spa.py`, the NREL Solar
  Position Algorithm as implemented in pvlib python 0.13.0 (BSD-3-Clause,
  [`LICENSE-pvlib.txt`](experiments/benchmark/vendor/LICENSE-pvlib.txt)).

References cited on the poster:

1. Bishop-Taylor, R. et al. (2019). Between the tides: modelling the
   elevation of Australia's exposed intertidal zone at continental scale.
   *Estuarine, Coastal and Shelf Science* 223, 115-128.
2. Granadeiro, J. P. et al. (2021). Using Sentinel-2 images to estimate
   topography, tidal-stage lags and exposure periods over large intertidal
   areas. *Remote Sensing* 13(2), 320.
3. Schramm, M. et al. (2021). The openEO API – harmonising the use of Earth
   observation cloud services using virtual data cube functionalities.
   *Remote Sensing* 13(6), 1125.
4. Hart-Davis, M. G. et al. (2021). EOT20: a global ocean tide model from
   multi-mission satellite altimetry. *Earth System Science Data* 13,
   3869-3884.

## How to cite

Until the paper is published, please cite the poster:

> Alías Álvarez, J., García-González, F., González, P. (2026). MAREA: a
> method for mapping intertidal topography using Sentinel-2 openEO data
> cubes. Poster, CARTOGRASS 2026, Simposio Ibérico de Cartografía y
> Seguimiento de Praderas Marinas, Cartagena, Spain, 24-26 November 2026.
> Code: https://github.com/coastal-AI/Intertidal_analysis

```bibtex
@misc{alias2026marea,
  author       = {Al{\'\i}as {\'A}lvarez, Jorge and Garc{\'\i}a-Gonz{\'a}lez, Fernando and Gonz{\'a}lez, Pablo},
  title        = {{MAREA}: a method for mapping intertidal topography using {Sentinel-2} {openEO} data cubes},
  howpublished = {Poster, CARTOGRASS 2026, Simposio Ib{\'e}rico de Cartograf{\'\i}a y Seguimiento de Praderas Marinas, Cartagena, Spain},
  month        = nov,
  year         = {2026},
  note         = {Code: \url{https://github.com/coastal-AI/Intertidal_analysis}}
}
```

[`CITATION.cff`](CITATION.cff) describes the software; it will carry a DOI
when one exists.

## License

No license has been chosen yet; the authors will add one. Until then, please
contact the authors before reusing the code or the data. Third-party code
keeps its own license (see Data sources and credits).

<!-- Acknowledgments: to be supplied by the authors (funding, projects, field team), as on the poster (\cgAckBody). -->

## Contact

Jorge Alías Álvarez, jorge.alias.alvarez@gmail.com
