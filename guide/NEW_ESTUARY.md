# Map your own estuary

* **What this page is for:** running MAREA on an estuary of your choice: what you need, how
  to draw the box, how to check it before you fit anything, the chain of calls, what comes
  out and how to judge it.
* **Who it is for:** researchers who have read [`METHOD.md`](METHOD.md) and want to apply
  it elsewhere.
* **Time and needs:** a free Copernicus Data Space Ecosystem account and openEO processing
  credits for the download; a tide series; the environment of
  [`REPRODUCE.md`](REPRODUCE.md#1-environment). The box check of section 3 runs offline,
  on the Villaviciosa kit, in a few seconds, and the rehearsal of section 5 runs the whole
  chain offline, on the notebook sample, in about 20 seconds.

Notebook [`03_new_ria_openeo`](../notebooks/03_new_ria_openeo.ipynb) shows the same steps for
the Marismas de Santoña, as a dry run that downloads nothing.

## Short version

Four steps, from the repository root, shown for the Santoña box of notebook 03. Draw your
own box by the rules of [section 2](#2-draw-the-box) first.

1. **Download the Sentinel-2 cube** of the box: one openEO batch job, which spends credits
   of your account ([section 4](#4-size-and-cost-of-the-download)). The first connection
   opens a browser login.

   ```python
   from pathlib import Path

   import pyintertidal as pit

   aoi = pit.AOI.from_bbox(-3.56, 43.36, -3.40, 43.47, name="santona")   # west, south, east, north (degrees)
   cube = pit.SentinelCube(aoi, ("2023-01-01", "2026-01-01"),            # 2023-2025: the end date is exclusive
                           water="ndwi", cache_path="cubes/santona_2023-2025.nc", resolution=10)
   Path("cubes").mkdir(exist_ok=True)                                     # git ignores cubes/
   memory = {"executor-memory": "4G", "executor-memoryOverhead": "4G"}    # the job's memory, as in notebook 03
   cube.ensure(pit.scenes.connect(), job_options=memory)                  # an existing file is reused, not fetched again
   ```

2. **Check the box** on the cube before you fit anything: run the box check of
   [section 3](#3-check-the-box-before-fitting), with the two lines that load the kit
   replaced as it says. The mouth must lie only where the open sea leaves the box. A wrong
   mouth still gives a run that reports ok, and a quality table (section 7) whose every
   row looks fine.
3. **Get the tide model.** Put the 17 files of EOT20 into `tide_models/EOT20/ocean_tides/`
   ([`REPRODUCE.md`](REPRODUCE.md#6-route-b-from-the-authors-cube) says where to download
   them and how to check them).
4. **Make the maps.** With `<lat>` and `<lon>` a point in open water off the mouth, checked
   on a map (decimal degrees, west negative):

   ```bash
   python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona --lat <lat> --lon <lon> --mouth-side north
   ```

   With a tide series of your own, skip step 3 and give `--tide-csv <file>` in place of
   `--lat` and `--lon` ([section 1](#1-what-you-need)).

It writes the maps as GeoTIFFs into `runs/santona/` ([section 6](#6-what-comes-out)).
Judge the run with [section 7](#7-is-the-result-usable) before you use it. To try the chain
first, offline and in about 20 seconds, rehearse it on the notebook sample
([section 5](#5-run-the-chain)).

## 1. What you need

* **The Sentinel-2 cube.** Bands B03, B08 and SCL at 10 m over your box, as one netCDF file,
  downloaded through openEO on the Copernicus Data Space Ecosystem
  (`pyintertidal.SentinelCube.ensure`). The first connection opens a browser login
  (`pyintertidal.scenes.connect`). The download spends credits of your account. As a
  guide, a box of 1.6 million 10 m pixels over 3 years is about 2 GB on disk. Notebook 03
  gives a credit estimate.
* **A tide series at the mouth.** Any one of:
  * the global tide model **EOT20**: 17 files, 2.26 GB, free from SEANOE. How to download
    them and check them: [`REPRODUCE.md`](REPRODUCE.md#6-route-b-from-the-authors-cube);
  * **your own tide-gauge series**: a CSV file with a column `time` (ISO 8601 instants in
    UTC, such as `2023-01-06T11:10:00Z`, each instant once) and a column `level_m`
    (metres). From a shell, `scripts/quickstart.py --tide-csv path` reads it into a
    `TideSeries` (`scripts/marea_tools.py`); in Python, `read_tide_csv(path)` of
    `scripts/quickstart.py` reads it the same way. That reader converts a time-zone offset
    to UTC, stops with an error at any instant outside the series, and bridges a gap of
    any length with a straight line, which over a few hours can be far from the tide: fill
    the gaps first (from a tide model, for example). `GaugeBoundary.from_file(path,
    time_column="time", level_column="level_m")` of `pyintertidal.boundary` reads the same
    file differently: an instant more than one hour from the nearest sample, or outside
    the series, reads as missing, and an offset is dropped without being converted
    (below);
  * **IOC tide gauges** (`pyintertidal.boundary.make_boundary` with `n_gauges`). This
    route reads the list of IOC stations from `data_v4/gauges/ioc_stations.json`, which
    this branch does not ship: without it the call raises `FileNotFoundError`. Save there
    a JSON list of stations with the fields `Code`, `Lat`, `Lon` and `Location` first, or
    use a gauge record of your own, as above. IOC data are licensed for non-commercial use
    (CC BY-NC 4.0).

  Give every instant in UTC. `TideSeries` converts an instant with a time-zone offset to
  UTC, but `GaugeBoundary` (its `from_file` too) and `PyTMDBoundary` drop the offset
  without converting it: 13:00+01:00 is read as 13:00 UTC, an hour late.

  Whichever you use, it must cover every image instant, from three hours before it to
  45 minutes after it (the delays MAREA tabulates run from −45 to 180 min), and, for the
  emersion time, the whole of the record's years, from up to three hours before their
  first day: a pixel delayed by τ minutes reads the tide τ minutes earlier. The replication
  kit's 10-minute series starts at 21:00 UTC on 31 December 2022 for that reason. In
  `map_estuary` the record's years are the latest three calendar years of the cube, whole,
  even when the cube ends during the last of them: with a series of your own, download
  whole calendar years. `--tide-csv` checks the span before the run starts.
* **The acquisition time of every scene**, from the Sentinel-2 STAC catalogue
  (`pyintertidal.overpass.get_overpass_times`). This is a metadata query: no credits, but
  it needs the network. Saved in a file, the times let a run go offline
  ([section 5](#5-run-the-chain), `overpass_times`).
* **The environment** of [`REPRODUCE.md`](REPRODUCE.md#1-environment), with the commands
  run from the repository root.

## 2. Draw the box

MAREA finds the mouth and measures every distance from the images inside the box, so the
box decides what the method can see. Five rules follow from the code:

1. **Reach the open sea.** The permanent sea (pixels wet in at least 90 % of more than 30
   clear looks) must touch at least one edge of the box: that is where MAREA puts the
   mouth (`geometry.mouth_seeds`). If it touches no edge, MAREA finds no mouth, measures
   no distance and fits no delay. It does not stop, and `estado` (the status) stays ok.
   The signs are the printed lines `tau=[0.] -> applied [0.]` and `0 px shifted`, and
   `result.json`, where `centros_km` (the section centres) holds a single 0,
   `con_operador` (a delay was applied) is false and `s_max_km` (the farthest distance
   from the mouth) is null ([`DATA.md`](DATA.md#7-the-villaviciosa-product-files)
   translates every key).
2. **Keep the head of tide inside.** Draw the box past the last intertidal flat up the
   river. Flats cut off by the box are not mapped. A channel cut by an edge is permanent
   water on that edge, and with `mouth_side="any"` it becomes a second mouth.
3. **One exit to the sea.** The distance runs from the nearest mouth pixel through water
   and intertidal pixels. A second connection to the sea inside the box (another inlet, a
   canal, a river leaving the box) puts inner flats close to a "mouth". If the open sea
   lies on one side of the box, name that side: `mouth_side="north"` (or `"east"`,
   `"north+east"`, …), so that only it counts.
4. **One estuary, one axis.** The 3-km sections are cut by distance alone. A neighbouring
   estuary inside the box, or a second branch whose tide arrives at another time, falls
   into the same sections and blurs their delay. Leave neighbours out of the box, or fit
   one branch at a time with `region_mask` (an array of the cube's shape, true on the
   pixels to keep). The mouth still comes from the whole frame, but the distances run only
   through the permanent sea and the mask's own intertidal pixels. Keep in the mask the
   flats that join the branch to the mouth: without them the branch's distances grow, or
   vanish, and with 5,000 or fewer pixels left with a distance no delay is fitted, with no
   warning. After the run, check the row "pixels with a distance s" of the quality table
   (section 7), and `s_max_km`. In this version, a branch whose flats all fall into one
   3-km section raises an error ([section 8](#8-when-marea-stops-or-fits-no-delay)).
5. **Enough flat.** MAREA stops below 3,000 intertidal pixels (0.3 km²), fits no delay
   with 5,000 pixels or fewer that have a water path to the mouth, and merges sections of
   fewer than 500 pixels with their neighbour. Delays can reach 180 min at most.

## 3. Check the box before fitting

The masks of step 4 (`marea.extract`) are enough to see where MAREA will put the mouth and
how far the flats lie from it. This check runs on the Villaviciosa kit, offline, from the
repository root:

```python
import numpy as np

from pyintertidal import geometry, tide_estimators

d = np.load("replication/villaviciosa_record_2023-2025.npz")   # the masks of step 4: sea and inter
sea, inter = d["sea"], d["inter"]
seeds = geometry.mouth_seeds(sea, side="any")      # the mouth: permanent sea on the frame's edge
s_km = geometry.along_distance(sea | inter, seeds, 10.0) / 1000.0   # distance from the mouth through water
s_inter = s_km[inter]                              # the intertidal pixels, row by row
with np.errstate(invalid="ignore"):                # pixels without a water path have no distance
    edges, centres, band = tide_estimators.make_bands_fixed(s_inter, 3.0, 500)
print(f"{inter.sum():,} intertidal pixels (MAREA stops below 3,000)")
print(f"{np.isfinite(s_inter).sum():,} of them reach the mouth through water (a delay needs more than 5,000)")
print(f"the farthest is {np.nanmax(s_inter):.2f} km from the mouth: {len(centres)} sections, "
      f"centred at {', '.join(f'{c:.2f}' for c in centres)} km")
print("permanent-sea pixels on each edge:", {"north": int(sea[0].sum()), "south": int(sea[-1].sum()),
                                            "west": int(sea[:, 0].sum()), "east": int(sea[:, -1].sum())})
```

It prints:

```
32,120 intertidal pixels (MAREA stops below 3,000)
30,781 of them reach the mouth through water (a delay needs more than 5,000)
the farthest is 9.15 km from the mouth: 3 sections, centred at 1.49, 4.27, 7.57 km
permanent-sea pixels on each edge: {'north': 391, 'south': 0, 'west': 0, 'east': 199}
```

These distances and sections are those of the published product
(`products_marea/marea.npz`, keys `s_km` and `band`), value for value. The sea touches the
north and east edges, and both are open sea, so `mouth_side="any"` is right here. For your
own cube, add `from pyintertidal import marea` to the imports and replace the two lines
that load the kit (`d = np.load(...)` and `sea, inter = d["sea"], d["inter"]`) with
`ex = marea.extract("cubes/<your cube>.nc")` and `sea, inter = ex["sea"], ex["inter"]`;
then give `mouth_seeds` as `side` the `mouth_side` you intend to give `reconstruct`. Then
draw the two masks and the distance:

![Two maps of the Ría de Villaviciosa from the replication kit: on the left, the permanent sea, the intertidal pixels and the mouth pixels along the north and east edges of the frame; on the right, the intertidal pixels coloured by their distance from the mouth along the water, from 0 to 9 km, with the pixels that have no water path to the mouth marked.](figures/box_check.png)

*The box check at Villaviciosa (replication kit). (a) Permanent sea, intertidal pixels and
the mouth, the permanent sea on the frame's edge. (b) Distance from the mouth along the
water; the dark pixels have no water path to the mouth (1,339 here) and keep a delay of 0.*

What to look for:

* mouth pixels only where the open sea leaves the box: none on a river, a canal or a lake;
* distances that grow steadily up the estuary, with no inner flat close to 0;
* few intertidal pixels without a water path. A channel narrower than a pixel can cut the
  path, and the flats behind it then keep a delay of 0.

## 4. Size and cost of the download

Sections 1 and 2 of notebook 03 turn a box into the size of the 10 m frame, the number of
acquisition dates and a credit estimate. Prices change: start with a small job, check its
cost in your openEO account, then download the rest. An existing cube file is reused,
never downloaded again, so give every new download its own file name.

## 5. Run the chain

**In one call.** Once the cube is on disk (step 1, the download of the
[short version](#short-version)), `map_estuary` of
[`scripts/quickstart.py`](../scripts/quickstart.py) runs steps 2 to 11 with the settings of
the Villaviciosa product, except the quality rule, which is on (3, the library default).
From the repository root, in Python:

```python
import sys

sys.path.insert(0, "scripts")
from quickstart import map_estuary
from pyintertidal.boundary import PyTMDBoundary

# the tide point (lat, lon): (43.415, -3.48) is the centroid of the Santoña box, which lies
# inland. It is a placeholder only: replace it with a point in open water off the mouth,
# checked on a map.
tide = PyTMDBoundary("EOT20", 43.415, -3.48, directory="tide_models")
paths = map_estuary("cubes/santona_2023-2025.nc", tide, "runs/santona", mouth_side="north")
```

or from a shell, with the same placeholder point (replace it as above):

```bash
python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona --lat 43.415 --lon -3.48 --mouth-side north
```

It writes `marea.npz`, `result.json`, `figure.png` and the GeoTIFFs of section 6 into the
output folder, and raises an error when MAREA refuses the run (the reason is in
`result.json`). Its docstring lists every argument. `tide` can also be your own gauge
series (`GaugeBoundary` or `TideSeries`), as long as it covers the period (section 1). From
a shell, two options take the place of the tide model and of the network:

* `--tide-csv <file>`: your tide series, a CSV file with the columns `time` and `level_m`
  (section 1), read into a `TideSeries`. It replaces `--lat` and `--lon`, and cannot be
  combined with `--model` or `--tide-dir`. Before the run starts, the command checks that
  the series reaches every instant the run will read (section 1), and stops with the span
  it needs if not.
* `--overpass-times <file>`: the acquisition times, from a file in the format of
  [`products_marea/overpass_times.json`](../products_marea/overpass_times.json), instead of
  a query of the STAC catalogue (see `overpass_times` below for how to save one). Its
  instants are UTC without an offset: an offset is dropped, not converted.

**Rehearse it offline.** The notebook sample in [`data/`](../data/README.md) is a real
Sentinel-2 cube (270 × 270 pixels of the Ría de Villaviciosa, the 97 scenes of 2023–2025)
that also holds the EOT20 tide every 10 minutes. The whole chain runs on it with no
account, no download and no tide model. From the repository root, in Python:

```python
import sys
import time

import pandas as pd
import xarray as xr

sys.path.insert(0, "scripts")
import marea_tools as mt
from quickstart import map_estuary
from pyintertidal.boundary import GaugeBoundary

with xr.open_dataset("data/villaviciosa_sample.nc") as s:         # its EOT20 tide every 10 minutes
    utc = pd.DatetimeIndex(s["series_time"].values, tz="UTC")     # the instants are UTC: say so
    tide = GaugeBoundary(utc, s["tide_series"].values, name="EOT20 every 10 min")
times = mt.load_overpass("products_marea/overpass_times.json")    # UTC acquisition times: no catalogue query
t0 = time.time()
paths = map_estuary("data/villaviciosa_sample.nc", tide, "runs/rehearsal", overpass_times=times,
                    n_boot=0)                                     # n_boot=0: no delay intervals, faster
print(f"{len(paths)} GeoTIFFs in runs/rehearsal, {time.time() - t0:.0f} s")
```

On a Windows 10 laptop it printed (the folder in the fourth line shortened here; *listo
en 5 s* is "done in 5 s"):

```
[villaviciosa_sample] 13,294 intertidal px
[villaviciosa_sample] tau=[ 0.  27.5] -> applied [ 0.  27.5]
[villaviciosa_sample] per-pixel lag: median 7 min, max 15 min, 12,685 px shifted
[villaviciosa_sample] listo en 5 s -> ...\runs\rehearsal
6 GeoTIFFs in runs/rehearsal, 22 s
```

and wrote the files of section 6, named `marea_villaviciosa_sample_2023-2025_<layer>.tif`.
Its `result.json` gives `estado` ok, 92 scenes in the record (the screening of step 3, run
on the block alone, keeps 92 of the 97), and 10,491 of the 13,294 intertidal pixels with
an elevation. The intertidal mask comes from the 97 scenes of the block, not from the
1,379 of the authors' cube, hence 13,294 pixels rather than the product's 15,064 in the
same block.

The rehearsal checks the mechanics only: its delays and heights are not meaningful. The
block lies inside the ría (its pixels are 3.5 to 8.4 km from the mouth in the published
product) and reaches no open sea, so MAREA takes the main channel, where the block's edges
cut it, for the mouth. Nothing in the run warns of this, since `estado` is ok, but the box
check of section 3 shows it. Run on the sample (with `marea` imported, and the two lines
that load the kit replaced by `ex = marea.extract("data/villaviciosa_sample.nc")` and
`sea, inter = ex["sea"], ex["inter"]`), it prints:

```
13,294 intertidal pixels (MAREA stops below 3,000)
12,685 of them reach the mouth through water (a delay needs more than 5,000)
the farthest is 4.61 km from the mouth: 2 sections, centred at 0.82, 3.80 km
permanent-sea pixels on each edge: {'north': 7, 'south': 1, 'west': 0, 'east': 70}
```

The "permanent sea" on the north, south and east edges is the channel where it crosses
them, not the sea. The two delays it leads to, 0 and 27.5 min, have nothing to do with the
product's 5.0 and 20.7 min for the same flats.

The same rehearsal from a shell takes the tide as a CSV file. The sample's own series
starts at 00:00 UTC on 1 January 2023, and the emersion time of a pixel delayed by
τ minutes reads the tide τ minutes before that: `GaugeBoundary` reads those instants as
missing and the emersion time leaves them out, but `--tide-csv` refuses, before the run,
a series that does not start three hours before 1 January. So the shell uses the
replication kit's 10-minute series, which holds the same EOT20 levels with three hours
more at the start. Export it as a CSV file, from the repository root:

```python
import numpy as np, os, pandas as pd
os.makedirs("runs", exist_ok=True)                                    # git ignores runs/
k = np.load("replication/villaviciosa_eot20_10min_2023-2025.npz")     # EOT20 every 10 min, from 2022-12-31 21:00 UTC
pd.DataFrame({"time": pd.to_datetime(k["times_ns"]).strftime("%Y-%m-%dT%H:%M:%SZ"),   # ISO 8601; Z: UTC
              "level_m": k["levels_m"].astype(float)}).to_csv("runs/villaviciosa_tide.csv", index=False)
```

then run:

```bash
python -X utf8 scripts/quickstart.py data/villaviciosa_sample.nc runs/rehearsal_cli --tide-csv runs/villaviciosa_tide.csv --overpass-times products_marea/overpass_times.json --n-boot 0
```

In 24 s it printed the span of the series and the number of acquisition times it read, the
same lines from `reconstruct` (with its own run time and folder) and the paths of the six
GeoTIFFs in `runs/rehearsal_cli/`. They hold the same elevations, widths and delays as
those of the Python rehearsal.

**The functions you call.** The 11 steps of
[`METHOD.md`](METHOD.md#10-the-steps-in-the-code), the numbering of the scripts, the
guides and the code comments of notebook 03:

| step | what | function |
|---|---|---|
| 1 | download the cube (credits) | `pyintertidal.SentinelCube(...).ensure(pyintertidal.scenes.connect())` |
| 2 | the acquisition time of every scene | `pyintertidal.overpass.get_overpass_times` (network), or `scripts/marea_tools.load_overpass` for a saved file |
| 3 | screen the scenes | `scripts/marea_tools.screen` |
| 4 | the looks and the intertidal mask | `pyintertidal.marea.extract` |
| 5 | the record | `scripts/marea_tools.build_record` |
| 6 | the tide at the mouth | `pyintertidal.boundary.PyTMDBoundary` or `GaugeBoundary`, `scripts/marea_tools.TideSeries`, or `make_boundary` (its IOC gauges need the station list of section 1) |
| 7–10 | the mouth, the sections, their delays, every pixel's delay, elevation and width, and the hypsometry of step 11 (it reads the tide of step 6 itself only without a `boundary`) | `pyintertidal.marea.reconstruct` |
| 11 | emersion time, GeoTIFFs, quality table | `scripts/marea_tools.emersion_time`, `export_geotiffs`, `qa_table` |

**Step by step.** `map_estuary` is the chain written out: about thirty commented lines of
[`scripts/quickstart.py`](../scripts/quickstart.py) that call these functions. Step 4 comes
first, because the catalogue query of step 2 needs the cube's box and dates. To change one
step, for example to fit one branch with `region_mask`, copy that function and change the
call. Route B of [`REPRODUCE.md`](REPRODUCE.md#6-route-b-from-the-authors-cube)
(`scripts/reproduce_villaviciosa.py --from-cube`) runs the same chain, with the product's
settings, and reproduces the published Villaviciosa product array by array from the
authors' cube. There, on a laptop, the screening of 1,379 scenes took about 9 minutes and
`reconstruct` about 11 minutes with the 30 bootstrap fits.

After a run, print its quality table (section 7):

```python
import sys

sys.path.insert(0, "scripts")
import marea_tools as mt

print(mt.format_table(mt.qa_table("runs/santona"), ("check", "where", "found", "fine when")))
```

**Choices to make.**

* `mouth_side`: `"any"` when only open sea touches the edges; otherwise the side or sides
  of the open sea (section 2).
* The tide point. A point inside the estuary still returns a level: pyTMD extrapolates from
  the model's nearest ocean cells. A point in open water off the mouth says what you mean.
  Without a `boundary`, `reconstruct` reads EOT20 at the centre of the box.
* `overpass_times`: the acquisition time of every scene, `{"YYYY-MM-DD": UTC instant}`.
  Without them, `map_estuary` asks the STAC catalogue for them over the cube's own dates
  (network, no credits). To run offline later, save them once, with the query that
  `map_estuary` makes, in the format of `products_marea/overpass_times.json`:

  ```python
  import json
  import sys

  sys.path.insert(0, "scripts")
  import marea_tools as mt
  from pyintertidal import overpass

  info = mt.cube_info("cubes/santona_2023-2025.nc")                     # the cube's box and dates
  times = overpass.get_overpass_times(info["bbox"], (info["dates"][0], info["dates"][-1]))   # network
  with open("cubes/santona_overpass_times.json", "w", encoding="utf-8") as f:
      json.dump({d: str(t) for d, t in times.items()}, f)                # "YYYY-MM-DD HH:MM:SS", UTC
  ```

  and pass them back: `overpass_times=mt.load_overpass("cubes/santona_overpass_times.json")`
  in Python, or `--overpass-times cubes/santona_overpass_times.json` from a shell. Only
  `marea.reconstruct` called on its own without them is limited: it then asks the
  catalogue for 2016–2025 only, and scenes outside those years drop out of the record.
* `min_scenes`: the shortest record accepted: 40 in `map_estuary`, as for the Villaviciosa
  product, whose record holds 97 scenes; `marea.reconstruct` on its own defaults to 80.
* `quality_min_bracket`: 3 by default (at least 3 looks with the water below the fitted
  elevation and 3 above it). The Villaviciosa product was made with 0, before the rule
  existed.
* `region_mask`: one branch at a time (section 2, rule 4).

**Site files.** [`scripts/sites/villaviciosa.yaml`](../scripts/sites/villaviciosa.yaml) is
the record of the published Villaviciosa run. `scripts/reproduce_villaviciosa.py` and
notebook 01 read it, and act on some of its keys only: `key`, `overpass_times`, `out_dir`,
`tide.model`, `screening`, `record`, and `marea` except `region_mask`; the comments in the
file say which of the others are not read. A new estuary needs no site file:
`map_estuary` and `quickstart.py` take their settings as arguments, and no script reads a
site file of yours. To keep your settings in a file anyway, `scripts/marea_tools.load_site`
reads and checks one in the same format, and you pass its values on yourself.

## 6. What comes out

| file | what it holds |
|---|---|
| `marea.npz` | the per-pixel arrays (`z`, `sigma`, `tau_px_min`, `s_km`, `band`, `keep`, `shape`) and, per section, the fitted and applied delays with their interval; keys in [`DATA.md`](DATA.md#7-the-villaviciosa-product-files) |
| `result.json` | the run in numbers (keys partly in Spanish, translated in [`DATA.md`](DATA.md#7-the-villaviciosa-product-files)) |
| `figure.png` | the elevation map, the delay against distance and the hypsometric curve (panel titles in Spanish) |
| `marea_<site>_<period>_<layer>.tif` | the layers as GeoTIFFs on the cube's grid. `quickstart.py` writes six: `z`, `sigma`, `tau_px` (minutes), `s_km` and `emersion` (the share of time out of the water, from 0 to 1, not a percentage) in float32 with NaN as no data, and `band` in int16 with −1 as no data. `export_geotiffs` can also write `n_clear` (int16, −1 as no data) |

Heights are on the zero of the tide you used (section 9).

## 7. Is the result usable?

First, the mouth: run the box check of [section 3](#3-check-the-box-before-fitting) on
your cube. The quality table below cannot see a wrong mouth: on the rehearsal of
section 5, whose "mouth" is a channel, none of its rows points to the problem.

`mt.qa_table(run_folder)` lists facts about a run, each with the range where it is fine. On
the published Villaviciosa product (`mt.qa_table("products_marea")`) it gives:

| check | found at Villaviciosa | fine when |
|---|---|---|
| estado | ok | ok |
| pixels with a distance s | 30,781 | more than 5,000 (else no delay is fitted) |
| share of band −1 | 4.2 % | small (pixels water cannot reach) |
| sections | 3 at 1.49, 4.27, 7.57 km | one per 3 km of estuary |
| interval width per section (min) | 0.0, 17.9, 15.6 | narrow compared with the delay |
| section at the 180-min bound | False | False |
| veto_fisico (a negative section clipped to 0) | False | False |
| elevations | 29,728 of 32,120 (93 %) | most pixels |
| share of z at the edges of rango_marea_m (the tide range searched) | 1.2 % | small |
| share of sigma at 0.03 or 0.65 m (the ends of the width grid) | 26.2 % | small |

Read the delays with their intervals: at Villaviciosa the middle section's interval (0 to
17.9 min) is wider than its delay (5.0 min), so the images decide it only loosely.

Read the widths with their grid: at Villaviciosa 26.2 % of the widths sit at an end of the
grid (13.7 % at 0.03 m, 12.5 % at 0.65 m), more than "small". Those pixels turn from dry to
wet more sharply, or more gradually, than the grid can express, so their σ is a bound
rather than a value; read the uncertainty map there with that in mind
([`METHOD.md`](METHOD.md#8-uncertainty)).

`mt.coverage(levels, tide, start, end, out_dir=run_folder)`, with `levels` the tide at the
record's instants, writes `coverage.json`: how much of the tide range the images saw,
judged against fixed thresholds. On the Villaviciosa product, from the repository root:

```python
import json
import sys

sys.path.insert(0, "scripts")
import marea_tools as mt

res = json.load(open("products_marea/result.json", encoding="utf-8"))                # the record's dates
times = mt.load_overpass("products_marea/overpass_times.json")                        # their UTC instants
tide = mt.TideSeries.from_npz("replication/villaviciosa_eot20_10min_2023-2025.npz")   # the tide at the mouth
levels = tide.levels([times[d] for d in res["record_dates"]])                         # at the record's instants
v = mt.coverage(levels, tide, "2023-01-01", "2026-01-01")        # out_dir=... also writes coverage.json
print(v["suitable"], v["reasons"])
```

It prints:

```
False ['only 69% of the tidal range was imaged (0.22 m missing at the bottom, 1.26 m at the top)']
```

The record is judged not suitable because the images saw only 69 % of the tide range.
That is the limit of a fixed overpass time
([`METHOD.md`](METHOD.md#9-what-the-images-cannot-tell)), not a failure of the run: no
elevation is read outside the levels the images saw.

None of these checks compares the map with the ground. Only an independent survey can say
how close it is.

## 8. When MAREA stops or fits no delay

| condition | what happens | where you see it |
|---|---|---|
| fewer than 3,000 intertidal pixels | `reconstruct` stops and returns `None` | `result.json`: `estado` `sin_intermareal` |
| fewer than `min_scenes` record scenes with an acquisition time | `reconstruct` stops and returns `None` | `result.json`: `estado` `pocas_escenas` |
| no permanent sea on the chosen edges | no mouth, no distance, no delay; the elevations are fitted on the tide at the mouth | `centros_km` = [0.0], `con_operador` false, `s_max_km` null |
| 5,000 or fewer pixels with a water path to the mouth | no delay; the elevations are fitted on the tide at the mouth | `centros_km` = [0.0], `con_operador` false, `s_max_km` a number (null when no pixel has a path) |
| a `region_mask` that leaves out the flats joining the branch to the mouth | the distances run only through the permanent sea and the mask's own flats, so they grow or vanish; with 5,000 or fewer pixels left with a distance, no delay | `qa_table`: pixels with a distance s; `s_max_km` null when none is left |
| all pixels within one 3-km section (a short estuary, or a short branch passed as `region_mask`) | `reconstruct` raises `ValueError` in the delay fit, before it writes `marea.npz` or `result.json` (a known limitation of this version) | the error, `not enough values to unpack (expected 2, got 0)` |
| a section's delay reaches 180 min | it cannot grow further: the delay is pinned, not measured | `qa_table`: section at the 180-min bound |
| an intertidal pixel without a water path | it keeps a delay of 0 | `band` −1, `s_km` NaN |

## 9. Heights in a national datum

MAREA's heights are on the zero of the tide you used: the mean level of the tide model, or
of your gauge series. To read them in a national datum, add the height of that zero in the
datum. The library does not compute it. For the maps of the viewer it was read from Spanish
tide gauges, tied to the datum NMMA directly or through the coast's mean sea level of 2011
([`DATA.md`](DATA.md#3-heights-nmma-and-the-constant-c)).

## Next steps

* How the method works: [`METHOD.md`](METHOD.md).
* The same steps for the Marismas de Santoña, as a dry run:
  [`03_new_ria_openeo`](../notebooks/03_new_ria_openeo.ipynb).
* The files and their formats: [`DATA.md`](DATA.md).
