# Map your own estuary

* **What this page is for:** running MAREA on an estuary of your choice: what you need, how
  to draw the box, how to check it before you fit anything, the chain of calls, what comes
  out and how to judge it.
* **Who it is for:** researchers who have read [`METHOD.md`](METHOD.md) and want to apply
  it elsewhere.
* **Time and needs:** a free Copernicus Data Space Ecosystem account and openEO processing
  credits for the download; a tide record; the environment of
  [`REPRODUCE.md`](REPRODUCE.md#1-environment). The box check of section 3 runs offline,
  on the Villaviciosa kit, in a few seconds.

Notebook [`03_new_ria_openeo`](../notebooks/03_new_ria_openeo.ipynb) shows the same steps for
the Marismas de Santoña, as a dry run that downloads nothing.

## 1. What you need

* **The Sentinel-2 cube.** Bands B03, B08 and SCL at 10 m over your box, as one netCDF file,
  downloaded through openEO on the Copernicus Data Space Ecosystem
  (`pyintertidal.SentinelCube.ensure`). The first connection opens a browser login
  (`pyintertidal.scenes.connect`). The download spends credits of your account. As a
  guide, a box of 1.6 million 10 m pixels over 3 years is about 2 GB on disk. Notebook 03
  gives a credit estimate.
* **A tide record at the mouth.** Any one of:
  * the global tide model **EOT20**: 17 files, 2.26 GB, free from SEANOE. How to download
    them and check them: [`REPRODUCE.md`](REPRODUCE.md#6-route-b-from-the-authors-cube);
  * **your own tide-gauge record**: a CSV file with a column `time` (ISO 8601, UTC) and a
    column `level_m` (metres), read by
    `GaugeBoundary.from_file(path, time_column="time", level_column="level_m")` from
    `pyintertidal.boundary`. Instants more than one hour from the nearest sample read
    as missing;
  * **IOC tide gauges** (`pyintertidal.boundary.make_boundary` with `n_gauges`). This
    route reads the list of IOC stations from `data_v4/gauges/ioc_stations.json`, which
    this branch does not ship: save there a JSON list of stations with the fields `Code`,
    `Lat`, `Lon` and `Location` first. IOC records are licensed for non-commercial use
    (CC BY-NC 4.0).

  Whichever you use, it must cover every image instant, from three hours before it to
  45 minutes after it (the delays MAREA tabulates run from −45 to 180 min), and, for the
  emersion time, the whole period.
* **The acquisition time of every scene**, from the Sentinel-2 STAC catalogue
  (`pyintertidal.overpass.get_overpass_times`). This is a metadata query: no credits.
* **The environment** of [`REPRODUCE.md`](REPRODUCE.md#1-environment), with the commands
  run from the repository root.

## 2. Draw the box

MAREA finds the mouth and measures every distance from the images inside the box, so the
box decides what the method can see. Five rules follow from the code:

1. **Reach the open sea.** The permanent sea (pixels wet in at least 90 % of more than 30
   clear looks) must touch at least one edge of the box: that is where MAREA puts the
   mouth (`geometry.mouth_seeds`). If it touches no edge, MAREA finds no mouth, measures
   no distance and fits no delay. It does not stop: the only sign is in `result.json`,
   where `centros_km` holds a single 0 and `con_operador` is false.
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
   pixels to keep; the mouth and the distances still come from the whole frame).
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
own cube, replace the first two lines with `ex = marea.extract("cubes/<your cube>.nc")` and
`sea, inter = ex["sea"], ex["inter"]`, and use the `mouth_side` you intend to pass. Then
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

**In one call.** Once the cube is on disk (step 1, as in notebook 03),
[`scripts/quickstart.py`](../scripts/quickstart.py) runs steps 2 to 11 with the settings of
the Villaviciosa product. From the repository root, in Python:

```python
import sys

sys.path.insert(0, "scripts")
from quickstart import map_estuary
from pyintertidal.boundary import PyTMDBoundary

tide = PyTMDBoundary("EOT20", 43.415, -3.48, directory="tide_models")   # the tide point (lat, lon)
paths = map_estuary("cubes/santona_2023-2025.nc", tide, "runs/santona", mouth_side="north")
```

or from a shell:

```bash
python -X utf8 scripts/quickstart.py cubes/santona_2023-2025.nc runs/santona --lat 43.415 --lon -3.48 --mouth-side north
```

It writes `marea.npz`, `result.json`, `figure.png` and the GeoTIFFs of section 6 into the
output folder, and raises an error when MAREA refuses the run (the reason is in
`result.json`). Its docstring lists every argument. `tide` can also be your own record
(`GaugeBoundary`), as long as it covers the period (section 1).

**The functions you call.** The 11 steps of
[`METHOD.md`](METHOD.md#10-the-steps-in-the-code), the numbering of the scripts, the
notebooks and the guides:

| step | what | function |
|---|---|---|
| 1 | download the cube (credits) | `pyintertidal.SentinelCube(...).ensure(pyintertidal.scenes.connect())` |
| 2 | the acquisition time of every scene | `pyintertidal.overpass.get_overpass_times` |
| 3 | screen the scenes | `scripts/marea_tools.screen` |
| 4 | the looks and the intertidal mask | `pyintertidal.marea.extract` |
| 5 | the record | `scripts/marea_tools.build_record` |
| 6 | the tide at the mouth | `pyintertidal.boundary.PyTMDBoundary`, `GaugeBoundary` or `make_boundary` |
| 7–10 | the mouth, the sections, their delays, every pixel's delay, elevation and width | `pyintertidal.marea.reconstruct` |
| 11 | emersion time, GeoTIFFs, quality table | `scripts/marea_tools.emersion_time`, `export_geotiffs`, `qa_table` |

**Step by step.** The same chain written out, as in notebook 03, for a box and a period of
your choice (here the Santoña box of notebook 03):

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")                       # the helpers of scripts/marea_tools.py
import marea_tools as mt
import pyintertidal as pit
from pyintertidal import marea, overpass
from pyintertidal.boundary import PyTMDBoundary

SITE, PERIOD = "santona", ("2023-01-01", "2026-01-01")      # end exclusive, as openEO reads it
CUBE, OUT = "cubes/santona_2023-2025.nc", "runs/santona"    # cubes/ and runs/ are not versioned
aoi = pit.AOI.from_bbox(-3.56, 43.36, -3.40, 43.47, name=SITE)   # west, south, east, north (degrees)

# 1 the cube: one openEO batch job (spends credits)
Path("cubes").mkdir(exist_ok=True)
pit.SentinelCube(aoi, PERIOD, water="ndwi", cache_path=CUBE, resolution=10).ensure(pit.scenes.connect())
# 2 the acquisition times, one catalogue query per year (no credits)
times = {}
for year in (2023, 2024, 2025):
    times.update(overpass.get_overpass_times(aoi.bbox, (f"{year}-01-01", f"{year}-12-31"), verbose=False))
times = {d: pd.Timestamp(v) for d, v in sorted(times.items())}
# 3 the usable scenes (the defaults are the settings of the Villaviciosa product)
usable, reference_map, cloud = mt.screen(CUBE)
# 4 the looks and the intertidal mask (then the box check of section 3)
ex = marea.extract(CUBE)
# 6 the tide at the mouth: a point in open water off the mouth (lat, lon), here the box's centroid
lat, lon = aoi.centroid
tide = PyTMDBoundary("EOT20", lat, lon, directory="tide_models")
# 5 the record: the usable scenes of the latest three years with a time and a tide level
ex_rec, record = mt.build_record(ex, usable, times, tide, epoch_years=3, which="latest")
# 7-10 MAREA: the mouth, the sections, their delays, every pixel's delay and elevation
result = marea.reconstruct(CUBE, OUT, name=SITE, boundary=tide, extraction=ex_rec,
                           overpass_times=times, mouth_side="north", min_scenes=40)
# 11 the products: emersion time and GeoTIFFs, then the quality table
with np.load(f"{OUT}/marea.npz") as d:
    z, tau_px = d["z"], d["tau_px_min"]
emersion = mt.emersion_time(z, tau_px, tide, PERIOD[0], PERIOD[1], step_min=10)
mt.export_geotiffs(OUT, CUBE, OUT, SITE, "2023-2025",
                   layers=("z", "sigma", "tau_px_min", "emersion"), extra={"emersion": emersion})
print(mt.format_table(mt.qa_table(OUT), ("check", "where", "found", "fine when")))
```

This is the chain of Route B in [`REPRODUCE.md`](REPRODUCE.md#6-route-b-from-the-authors-cube)
(`scripts/reproduce_villaviciosa.py --from-cube`), which reproduces the published
Villaviciosa product array by array from the authors' cube. There, on a laptop, the
screening of 1,379 scenes took about 9 minutes and `reconstruct` about 11 minutes with the
30 bootstrap fits.

**Choices to make.**

* `mouth_side`: `"any"` when only open sea touches the edges; otherwise the side or sides
  of the open sea (section 2).
* The tide point. A point inside the estuary still returns a level: pyTMD extrapolates from
  the model's nearest ocean cells. A point in open water off the mouth says what you mean.
  Without a `boundary`, `reconstruct` reads EOT20 at the centre of the box.
* `overpass_times`: pass them, as above. Without them, `reconstruct` queries the catalogue
  for 2016–2025 only, and scenes outside those years drop out of the record.
* `min_scenes`: the shortest record accepted; 80 by default, 40 for the Villaviciosa
  product, whose record holds 97 scenes.
* `quality_min_bracket`: 3 by default (at least 3 looks with the water below the fitted
  elevation and 3 above it). The Villaviciosa product was made with 0, before the rule
  existed.
* `region_mask`: one branch at a time (section 2, rule 4).

To keep a site's settings in one file, use
[`scripts/sites/villaviciosa.yaml`](../scripts/sites/villaviciosa.yaml) as the model.
`scripts/marea_tools.load_site` reads and checks such a file; `site_aoi` and
`site_boundary` turn it into the box and the tide (model, gauges or your own CSV).

## 6. What comes out

| file | what it holds |
|---|---|
| `marea.npz` | the per-pixel arrays (`z`, `sigma`, `tau_px_min`, `s_km`, `band`, `keep`, `shape`) and, per section, the fitted and applied delays with their interval; keys in [`DATA.md`](DATA.md#7-the-villaviciosa-product-files) |
| `result.json` | the run in numbers (keys partly in Spanish, translated in [`DATA.md`](DATA.md#7-the-villaviciosa-product-files)) |
| `figure.png` | the elevation map, the delay against distance and the hypsometric curve (panel titles in Spanish) |
| `marea_<site>_<period>_<layer>.tif` | the layers as GeoTIFFs on the cube's grid: `z`, `sigma`, `s_km` and `emersion` in float32 with NaN as no data, `tau_px` in minutes (`quickstart.py` writes these six, `band` included); `band` and `n_clear` in int16 with −1 as no data |

Heights are on the zero of the tide you used (section 9).

## 7. Is the result usable?

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

`mt.coverage(levels, tide, start, end, out_dir=run_folder)`, with `levels` the tide at the
record's instants, writes `coverage.json`: how much of the tide range the images saw,
judged against fixed thresholds. On the Villaviciosa
record it says `suitable: false`, because only 69 % of the tide range was imaged (0.22 m
missing at the bottom, 1.26 m at the top). That is the limit of a fixed overpass time
([`METHOD.md`](METHOD.md#9-what-the-images-cannot-tell)), not a failure of the run: no
elevation is read outside the levels the images saw.

None of these checks compares the map with the ground. Only an independent survey can say
how close it is.

## 8. When MAREA stops or fits no delay

| condition | what happens | where you see it |
|---|---|---|
| fewer than 3,000 intertidal pixels | `reconstruct` stops and returns `None` | `result.json`: `estado` `sin_intermareal` |
| fewer than `min_scenes` record scenes with an acquisition time | `reconstruct` stops and returns `None` | `result.json`: `estado` `pocas_escenas` |
| no permanent sea on the chosen edges | no mouth, no distance, no delay; the elevations are fitted on the tide at the mouth | `centros_km` = [0.0], `con_operador` false |
| 5,000 or fewer pixels with a water path to the mouth | no delay; the elevations are fitted on the tide at the mouth | `centros_km` = [0.0], `con_operador` false |
| a section's delay reaches 180 min | it cannot grow further: the delay is pinned, not measured | `qa_table`: section at the 180-min bound |
| an intertidal pixel without a water path | it keeps a delay of 0 | `band` −1, `s_km` NaN |

## 9. Heights in a national datum

MAREA's heights are on the zero of the tide you used: the mean level of the tide model, or
of your gauge record. To read them in a national datum, add the height of that zero in the
datum. The library does not compute it. For the four maps of the viewer it was read from
tide gauges tied to the Spanish datum NMMA
([`DATA.md`](DATA.md#3-heights-nmma-and-the-constant-c)).

## Next steps

* How the method works: [`METHOD.md`](METHOD.md).
* The same steps for the Marismas de Santoña, as a dry run:
  [`03_new_ria_openeo`](../notebooks/03_new_ria_openeo.ipynb).
* The files and their formats: [`DATA.md`](DATA.md).
