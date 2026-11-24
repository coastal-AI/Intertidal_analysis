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
