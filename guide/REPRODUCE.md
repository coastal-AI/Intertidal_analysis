# Reproduce the Villaviciosa map

* **What this page is for:** rebuilding [`products_marea/`](../products_marea/README.md),
  the 10 m MAREA product of the Ría de Villaviciosa (2023–2025) that the website shows,
  and checking the rebuild array by array.
* **Who it is for:** reviewers, and anyone who wants to check the method end to end.
* **Time and needs:** Route A takes about a minute and about 0.6 GB of memory, with no
  account, no download and no tide model. Route B needs the authors' Sentinel-2 cube
  (3.09 GB, not public yet) and the EOT20 files (2.26 GB). Route C needs a free
  Copernicus account and openEO credits.

Here *reproduce* means that the same data and the same code give identical arrays, or
arrays within a stated tolerance where an optimiser is involved. Only the Villaviciosa
product can be rebuilt from this branch: the other maps of the viewer were made with the
same library outside it ([`DATA.md`](DATA.md#1-the-four-sites)). Mapping a new estuary
with the same procedure is a different task: see [`NEW_ESTUARY.md`](NEW_ESTUARY.md).

## 1. Environment

Python 3.13 (3.13.3 was used), in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or in a conda environment:

```bash
mamba env create -f environment.yml
mamba activate marea-cartograss26
```

With pip, the installation takes about 8 minutes and 1 GB of disk; its "Installing
collected packages" step stays silent for about 5 minutes.

On Windows, clone the repository into a short folder such as `C:\src\`: some installed
files have long paths, and pip fails when the clone's folder path is longer than about
110 characters (or enable long paths in Windows, which needs an administrator). Calling
the environment's Python directly avoids PowerShell's activation rules:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then write `.venv\Scripts\python` wherever this page writes `python`.

[`requirements.txt`](../requirements.txt) pins the versions with which the product was
made and reproduced (numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, xarray 2025.1.1,
pyTMD 2.2.9.1, eo-tides 0.10.4, …). pyTMD, eo-tides and openeo are needed only for
Routes B and C and for a new estuary; Route A, the notebooks and the tests run without
them, but the file installs them all.

The library is used from the checkout, not installed with pip. Run every command from
the repository root (the notebooks find it themselves), and keep
[`configs/`](../configs/) next to `pyintertidal/`: `marea.reconstruct` reads
`configs/m2.yaml` from there. The scripts write into `runs/`, which git ignores. On
Windows, run Python with `-X utf8` (or set `PYTHONUTF8=1`), as the commands on this page
do.

Check the environment before a run:

```bash
python -X utf8 scripts/reproduce_villaviciosa.py --check-env
```

It prints one row per package, with the columns package, required, installed and status,
and ends with `environment ok`. A version other than the pinned one is reported as
`other version` but does not stop anything: the arrays were checked identical with the
pins. A missing core package is reported as `MISSING`, and the command exits with code 1.
A missing pyTMD, eo-tides or openeo is reported with what it is needed for
(`missing (needed for tide models)`, `missing (needed for download)`) and does not fail
the check.

## 2. Route A: the offline kit

The kit in [`replication/`](../replication/README.md) holds what steps 1–5 of the method
produced from the authors' 10-year cube:

* the record: the NDWI and clear-sky flag of every look of the 97 screened scenes of
  2023–2025 at the 32,120 intertidal pixels;
* the EOT20 tide at the mouth at the 97 image instants, for every whole-minute delay from
  −45 to 180 min;
* the EOT20 tide every 10 minutes over 2023–2025, for the emersion time.

The script reads the tide from that table instead of the tide model, so no EOT20 files
are needed. The table answers only the instants it holds and fails loudly for any other.

```bash
python -X utf8 scripts/reproduce_villaviciosa.py --from-kit replication
```

The run goes into `runs/villaviciosa_repro/` (change it with `--out`); the script never
writes into `products_marea/`. It first checks every kit file against its sha256 in
[`replication/MANIFEST.json`](../replication/MANIFEST.json), then prints each step with
the count it must reach. On a Windows 10 laptop, on 27 October 2026, it printed (times
vary from run to run; *listo* is Spanish for done):

```
MAREA: reproduce products_marea/ (Ría de Villaviciosa); route A, offline kit
settings from scripts/sites/villaviciosa.yaml: min_scenes 40, quality_min_bracket 0, mouth_side any, band_km 3.0, n_boot 30; output runs/villaviciosa_repro
(1) Download the Sentinel-2 cube: B03, B08, SCL at 10 m: not needed (the kit holds the record); source cube sha256 5f3ccb2bf916...  (expected 5f3ccb2bf916...)  ok
(2) Get the acquisition time of every scene: 1,380 acquisition times in products_marea/overpass_times.json
(3) Screen the scenes: done when the kit was built (replication/MANIFEST.json); every kit file matches its sha256
(4) Extract the looks and the intertidal mask: 32,120 intertidal px on a 915 x 915 frame  (expected 32,120)  ok
(5) Build the record: 97 scenes, 2023-01-06 .. 2025-12-28  (expected 97)  ok
(6) Read the tide at the mouth: EOT20 tabulated at (43.509024, -5.416135), 97 instants x 226 shifts; level -2.172 .. +1.146 m  (expected -2.172 .. +1.146 m at (43.509024, -5.416135))  ok
steps 7-11: pyintertidal.marea.reconstruct (n_boot 30)
[Villaviciosa] 32,120 intertidal px
[Villaviciosa] bootstrap of 30 scene resamples [40 s]
[Villaviciosa] tau=[ 0.   5.  20.7] -> applied [ 0.   5.  20.7]
[Villaviciosa] per-pixel lag: median 6 min, max 16 min, 30,781 px shifted
[Villaviciosa] listo en 57 s -> runs/villaviciosa_repro
reconstruct took 57 s
(7) Find the mouth, the distance along the water and the 3-km sections: 3 sections centred at 1.49, 4.27, 7.57 km  (expected 1.49, 4.27, 7.57)  ok
(8) Fit one delay per section, with its interval: delays 0.00, 5.05, 20.74 min; 95 % interval [0.0, 0.0], [0.0, 17.9], [13.1, 28.7]  (expected 0.00, 5.05, 20.74)  ok
(9) Give every pixel its delay: per-pixel delay median 6 min, max 16 min, 30,781 px shifted  (expected 6, 16)  ok
(10) Fit each pixel's elevation and width on its delayed tide: 29,728 of 32,120 px with an elevation (quality_min_bracket 0)  (expected 29,728)  ok
(11) Make the products: hypsometry, GeoTIFFs, emersion time: hypsometry over 2.9728 km2; marea.npz, result.json, figure.png, record.json, provenance.json in runs/villaviciosa_repro
```

then the check table of [section 3](#3-the-check) and, last:

```
PASS: 41 of 41 checks passed; total 57 s
```

`--fast` skips the 30 bootstrap fits of the delay intervals, which are then not checked;
its last line reads `PASS: 37 of 37 checks passed, 4 skipped (--fast); total 20 s`. The
script exits with code 0 only if every check passes. It has been run on Windows and Linux
with Python 3.13.3.

Step 11 of Route A writes the product files only: no GeoTIFF and no emersion time, despite
the step's name. Open `runs/villaviciosa_repro/figure.png` first (its labels are in
Spanish; [section 8](#8-what-a-run-folder-holds)). For a GeoTIFF of the elevation, use the
snippet of [`DATA.md`](DATA.md#7-the-villaviciosa-product-files) on
`runs/villaviciosa_repro/marea.npz`: it gives a GeoTIFF in EPSG:32630 with 29,728 values.
Notebook 01 maps the emersion time.

**Three settings differ from the library defaults**, because the product predates them.
They are read from [`scripts/sites/villaviciosa.yaml`](../scripts/sites/villaviciosa.yaml):

* `quality_min_bracket=0`: the product was made before the quality rule existed;
  `marea.reconstruct` now defaults to 3.
* The tide point: EOT20 is read at the centroid of the registered site polygon,
  (43.509024 N, 5.416135 W). Without a boundary, `reconstruct` would read it at the
  centre of the cube's bounding box, a different point.
* `min_scenes=40`: the shortest record the production runs accepted (the default is 80).
  The record of 97 scenes passes either floor.

## 3. The check

The script compares the run with `products_marea/` and prints one row per check, with
the columns check, expected, found, tolerance and status. The status is PASS, FAIL or
SKIP (a check that `--fast` leaves out); the run passes when no row is FAIL.

| what is compared | expected | tolerance |
|---|---|---|
| `marea.npz`: `keep`, `shape`, `z`, `sigma`, `s_km`, `band`, `tau_px_min`, `hyp_z`, `hyp_area`, `hyp_lo`, `hyp_hi`, `band_km` | the arrays of `products_marea/marea.npz` | identical, same dtype and shape (NaN equal to NaN) |
| `marea.npz`: `tau_min`, `tau_usado_min` (the section delays) | 0, 5.05 and 20.74 min | 1e-6 min |
| `marea.npz`: `tau_boot_min`, `tau_lo_min`, `tau_hi_min` (the bootstrap fits and intervals) | the arrays of `products_marea/marea.npz` | 0.01 min; SKIP with `--fast` |
| `result.json`: status, scene and pixel counts, record dates | `ok`; 97 scenes; 32,120 pixels, 29,728 with an elevation | exact |
| `result.json`: section centres, pixel-delay median and maximum, tide range searched | 1.49, 4.27, 7.57 km; 6 and 16 min; −2.172 to 1.146 m | 1e-9 |
| `result.json`: the other settings and results (site, `min_scenes`, `band_km`, hypsometry, …) | the values of `products_marea/result.json` | exact (numbers within 1e-9) |
| `result.json`: `quality_min_bracket` | 0 (the only key the published file lacks) | exact |
| `result.json`: `inputs_sha.cube` | the sha256 of the kit's record file (Route A) or of the cube (Route B) | exact |

These checks compare a run with the published product. They say whether the code
behaves as it did when the product was made, not how close the map is to the ground.

## 4. Tests

```bash
python -X utf8 -m pytest
```

The tests take about a minute (49 s on a Windows 10 laptop). They hold unit tests of the
library, a check of the notebook sample in [`data/`](../data/README.md) against the stored
elevations, and the `--fast` run of Route A. The last test,
`tests/test_publication_guard.py`, is for the maintainers: it checks that no file a commit
would take holds personal paths or credentials, so it fails if you keep in the checkout an
untracked file that names a folder of your own. Two of its checks need a word list that
the maintainers keep outside the repository; without it they are reported as skipped.

## 5. Notebooks

The three notebooks in [`notebooks/`](../notebooks/) are stored with their outputs. On a
Windows 10 laptop, notebook 01 ran in 59 s, 02 in 14 s and 03 in 9 s. To run them
yourself, install JupyterLab in the same environment (`pip install jupyterlab`; it is not
in `requirements.txt`) and start `jupyter lab`, or run one without opening it:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_marea_villaviciosa.ipynb
```

It writes `notebooks/01_marea_villaviciosa.nbconvert.ipynb`, which git ignores. On
Windows, a pyzmq warning about the "Proactor event loop" and a numpy warning that
"ndarray size changed" may appear; both are harmless. The notebooks also open on
[Binder](https://mybinder.org/v2/gh/coastal-AI/Intertidal_analysis/CARTOGRASS26?labpath=notebooks%2F01_marea_villaviciosa.ipynb).

## 6. Route B: from the authors' cube

**Inputs.**

* The cube `ndwi_cube_villaviciosa_grande_10y.nc`: 3,092,569,908 bytes, sha256
  `5f3ccb2bf9162037798111b86cc295e77fe39159480e665a4911a18c53aaa337`; 1379 scenes from
  2016-01-01 to 2025-12-29; 915 × 915 pixels of 10 m in EPSG:32630; bands B03, B08 and
  SCL. Not public yet. Check it with `sha256sum` (Linux, macOS) or
  `certutil -hashfile <file> SHA256` (Windows).
* The EOT20 tide model in `tide_models/EOT20/ocean_tides/` (below).
* The acquisition times, in the repository:
  [`products_marea/overpass_times.json`](../products_marea/overpass_times.json).

**EOT20.** Download `85762.zip` (2.3 GB) from SEANOE,
<https://www.seanoe.org/data/00683/79489/data/85762.zip> (DOI
[10.17882/79489](https://doi.org/10.17882/79489), CC BY 4.0). Unzip it, unzip the
`ocean_tides.zip` it contains, and move the 17 files `<constituent>_ocean_eot20.nc`
(2N2, J1, K1, K2, M2, M4, MF, MM, N2, O1, P1, Q1, S1, S2, SA, SSA and T2; 2.26 GB in all)
into `tide_models/EOT20/ocean_tides/`. Git ignores `tide_models/`. To test it, read the
tide at the Villaviciosa tide point at the instant of the first image of the record:

```python
from pyintertidal import sites
from pyintertidal.boundary import PyTMDBoundary

lat, lon = sites.get("villaviciosa").centroid             # (43.5090, -5.4161)
eot20 = PyTMDBoundary("EOT20", lat, lon, directory="tide_models")
print(eot20.levels(["2023-01-06 11:13:49"]))              # UTC; expected [-0.72246313]
```

**Run.**

```bash
python -X utf8 scripts/reproduce_villaviciosa.py --from-cube ndwi_cube_villaviciosa_grande_10y.nc --tide-dir tide_models
```

It hashes the cube, checks that every scene has an acquisition time, then reruns steps
3–11: the screening keeps 305 scenes of 2016–2025, 97 of them in 2023–2025; the
extraction finds the 32,120 intertidal pixels; the record holds the 97 scenes. It ends
with the check table of [section 3](#3-the-check). On a Windows 10 laptop the screening
took 562 s and `reconstruct` 688 s with the bootstrap. `--fast` works here too.

## 7. Route C: download the cube again

The original cube came from one openEO request on the Copernicus Data Space Ecosystem:
collection `SENTINEL2_L2A`; box W −5.46, S 43.47, E −5.35, N 43.55; 2016-01-01 to
2025-12-31 (end date exclusive); bands B03, B08 and SCL; resampled to 10 m by nearest
neighbour; netCDF. The same request in Python (a browser login opens the first time):

```python
import pyintertidal as pit

aoi = pit.AOI.from_bbox(-5.46, 43.47, -5.35, 43.55, name="Villaviciosa")
pit.SentinelCube(aoi, ("2016-01-01", "2025-12-31"), water="ndwi",
                 cache_path="ndwi_cube_villaviciosa_redownload.nc",
                 resolution=10).ensure(pit.scenes.connect())
```

Use a new file name: an existing file of the same name is reused, never fetched again.
The job spends openEO processing credits of your account.

The bytes will differ from the authors' cube: the backend version is written into the
file, and Sentinel-2 scenes are reprocessed over time. The screening, the record and the
product can therefore differ too. Run Route B on the new file, into its own folder, then
compare:

```bash
python -X utf8 scripts/reproduce_villaviciosa.py --from-cube ndwi_cube_villaviciosa_redownload.nc --tide-dir tide_models --out runs/villaviciosa_redownload
python -X utf8 scripts/reproduce_villaviciosa.py --compare-only runs/villaviciosa_redownload
```

The first command will show FAIL rows, because the identity checks do not apply to a new
cube. The second reports, without a verdict, the pixels of each product and those in
common, the share of identical elevations, and the largest and median elevation
differences.

## 8. What a run folder holds

Every run folder holds:

* `marea.npz` and `result.json`: the product, described in
  [`DATA.md`](DATA.md#7-the-villaviciosa-product-files);
* `figure.png`: the elevation map, the delay against distance and the hypsometric curve
  (panel titles in Spanish);
* `record.json`: the record dates, the epoch and the tide source;
* `provenance.json`: the git commit, the Python and package versions, the path, size and
  sha256 of every input, the settings of the run, and its start and end times.

## Next steps

* How the method works, step by step: [`METHOD.md`](METHOD.md).
* What the maps and files hold: [`DATA.md`](DATA.md).
* A new estuary: [`NEW_ESTUARY.md`](NEW_ESTUARY.md), and notebook
  [`03_new_ria_openeo`](../notebooks/03_new_ria_openeo.ipynb) as a dry run.
