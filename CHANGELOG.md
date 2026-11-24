# Changelog

## Unreleased (2026-11-24)

* Map viewer: ten more estuaries of northern Spain, shown as **preliminary** maps: the Ría de Corme e Laxe, the
  inner part of the Ría de Noia, the Ría de Betanzos, the Ría de Ferrol, the Ría de Ortigueira, the Ría de Tina
  Menor, the Ría de San Vicente de la Barquera, the Ría de Mogro, the Bahía de Santander and the Marismas de Santoña.
  Each is drawn only inside an intertidal mask made by a rule (a 3 × 3 median filter, then only the pieces connected
  to the estuary), not checked by hand, and none has an accuracy assessment. All fourteen maps give heights in NMMA;
  at the five Cantabrian sites the tie rests on the coast's REDNAP-tied mean sea level of 2011 and the datum sheet of
  the Santander 2 tide gauge, whose benchmark has only a 1998 tie. The viewer now lists fourteen sites; the maps of
  the four sites drawn inside masks cleaned by hand are unchanged, byte for byte, but for the wording of the Eo delay
  note.
* Viewer: the site switcher of wide screens lists the sites in two groups, "Hand-cleaned masks" and "Preliminary ·
  automatic mask", and a preliminary site gets one line on what the word means; on phones the chips of the chosen
  site and layer scroll into view. The delay legend says "MAREA finds no delay here" at the sites where it finds
  none, and at Noia that the delay is counted from the west edge of the map.
* The tide-line files of the preliminary sites carry `C_rule` in their metadata: how their C was tied to NMMA.
* `site/scripts/data_layers.py`: the fourteen sites in `SITE_ORDER`, and `LEFT_OUT`, which leaves a site out of the
  website's data in one line (the texts that count or list the sites must then be edited too). The front end's site
  list (`src/lib/sites.ts`) and its tests follow the data, and a link to a site left out opens the default site.
* Size caps: the data folder may hold 50 MB (about 42 MB now) and the built site 60 MB (about 44 MB now); a visitor
  downloads only the site they open.

## Documentation pass and an offline command line (2026-11-24)

The guides were checked against the code, and their offline examples were run again. The products and the
replication kit are unchanged.

* `scripts/quickstart.py`: two new options let the command line run offline. `--tide-csv` takes the tide at the
  mouth from a CSV file of your own (the columns `time` and `level_m`), read by the new function `read_tide_csv`
  into a `marea_tools.TideSeries`, in place of `--lat`, `--lon`, `--model` and `--tide-dir`; before the run starts,
  the command checks that the series reaches every instant the run reads (the new function `tide_span_needed`).
  `--overpass-times` reads the acquisition times from a saved `overpass_times.json` instead of the STAC catalogue.
  The call with `--lat` and `--lon` works as before, except that `--tide`, an abbreviation of `--tide-dir`, is now
  ambiguous.
* New test file `tests/test_quickstart_offline.py`: the CSV reader, the span check, the argument checks, and the
  command line run offline on the notebook sample (marked slow).
* `pyintertidal/`, `scripts/marea_tools.py` and `tests/conftest.py`: docstrings and comments only, except one
  printed label: `ElevationResult.summary` prints "transition width σ" instead of "sub-pixel relief σ". Corrected
  in them: the time zones of the tide providers (`GaugeBoundary` and `PyTMDBoundary` drop an offset, `TideSeries`
  converts it), the IOC gauge route (it needs a station list that this edition does not ship), the box without a
  mouth (the run goes on and fits no delay), the distances with `region_mask` (they run only through the permanent
  sea and the mask's own flats), the SCL classes (ESA's stop at 11), the frame of a study area, the format of
  `overpass_times.json`, and the parameters that have no effect in this edition.
* README: the six steps of the method now follow the opening paragraph. The section that rebuilds the published map
  is called "Reproduce the Villaviciosa map (offline, about a minute)", so that "quick start" now means only
  `scripts/quickstart.py`, the one-call route for a new estuary.
  New: how to use the library from a folder of your own (`PYTHONPATH`), and how to run a notebook with the pinned
  packages (`jupyter execute`, with `PYTHONUTF8=1` on Windows); JupyterLab is needed only to open the notebooks.
* `guide/NEW_ESTUARY.md`: the short version checks the box before the fit, and section 7 starts with the mouth,
  which the quality table cannot see. An offline rehearsal of `map_estuary` on the notebook sample, in Python and
  from a shell, with the box check that shows why its delays are not the product's and its maps are not to be used
  (the sample reaches no open sea). The two new options of `scripts/quickstart.py`, how to save the acquisition
  times, and a recipe for the tide coverage of a record. Corrected: `map_estuary` asks the catalogue for the times
  over the cube's own dates, and only `marea.reconstruct` called on its own is limited to 2016-2025; `map_estuary`
  runs step 4 before step 2; its `min_scenes` is 40; with `region_mask` the distances run only through the mask's
  own flats. Added: give every instant in UTC (`GaugeBoundary` drops a time-zone offset without converting it), a
  series of your own is bridged across its gaps, the IOC gauge route needs a station list this edition does not
  ship, and a new estuary needs no site file.
* The width sigma is described one way in the README, `guide/METHOD.md` and `guide/DATA.md`: it is not a standard
  error of the height, but a wider transition defines the level at which the pixel floods less sharply, which is
  why the maps show it as the uncertainty.
* `guide/REPRODUCE.md`: the publication guard is not the last test file (it runs third); the tests now include the
  offline command line; step 11 of Route A makes the hypsometry only, under the step's full name.
* `data/README.md`: the notebook sample reaches no open sea, so the new-estuary chain run on it checks the
  mechanics only; `data/SHA256SUMS.txt` holds the sha256 of the new text.
* Notebook 03: one comment in the chain cell marks its tide point as the inland placeholder to replace; its outputs
  are unchanged.

## cartograss26 (2026-11-23)

First version of this branch: the companion repository of the poster *MAREA: a method for mapping intertidal
topography using Sentinel-2 openEO data cubes*, presented at CARTOGRASS 2026 (Cartagena, 24-26 November 2026).

* `pyintertidal/`: the 22 modules of the library that MAREA, the notebooks and the map layers use, with a shorter
  `__init__.py`. The code is unchanged from the development library, except that the site descriptions of
  `sites.py` were edited and `seal.py` keeps only its SHA-256 helper; elsewhere only docstrings and comments were
  edited for this edition.
* `configs/`: the settings `marea.reconstruct` reads (`m2.yaml`) and the known corrections of overpass times.
* `replication/`, `products_marea/`, `products_villaviciosa/`, `scripts/`: the published 10 m product of the Ría de
  Villaviciosa (97 Sentinel-2 scenes, 2023-2025), its 10 m grid, the offline replication kit and the script that
  reproduces the product from the kit, array by array (41 checks).
* `tests/`: unit tests of the inversion and of the delay search, the overpass-time corrections, the offline
  reproduction, the check of the notebook sample and a publication guard.
* Example notebooks (`notebooks/`), a small real sample with its tide (`data/`), the guide (`guide/`) and the map
  viewer website (`site/`, built into `docs/`).
* `guide/NEW_ESTUARY.md`: how to map an estuary of your own: what you need, the rules for drawing the box, a box
  check that runs offline on the Villaviciosa kit, the chain of calls, what comes out and how to judge a run.
* `scripts/quickstart.py`: MAREA in one call (`map_estuary`), from a Sentinel-2 cube and a tide series to the maps,
  with the settings of the Villaviciosa product except the quality rule, which is on (3, the library default); it
  calls existing functions of the library and of `scripts/marea_tools.py` only.
* Licences: MIT for the code, CC BY 4.0 for text, figures and data.
