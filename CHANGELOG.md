# Changelog

## cartograss26 (2026-11-23)

First version of this branch: the companion repository of the poster *MAREA: a method for mapping intertidal
topography using Sentinel-2 openEO data cubes*, presented at CARTOGRASS 2026 (Cartagena, 24-26 November 2026).

* `pyintertidal/`: the 22 modules of the library that MAREA, the notebooks and the map layers use, with a shorter
  `__init__.py`. The code is unchanged from the development library; only docstrings and comments were edited for
  this edition.
* `configs/`: the settings `marea.reconstruct` reads (`m2.yaml`) and the known corrections of overpass times.
* `replication/`, `products_marea/`, `products_villaviciosa/`, `scripts/`: the published 10 m product of the Ría de
  Villaviciosa (97 Sentinel-2 scenes, 2023-2025), its 10 m grid, the offline replication kit and the script that
  reproduces the product from the kit, array by array (41 checks).
* `tests/`: unit tests of the inversion and of the delay search, the overpass-time corrections, the offline
  reproduction, the check of the notebook sample and a publication guard.
* Example notebooks (`notebooks/`), a small real sample with its tide (`data/`), the guide (`guide/`) and the map
  viewer website (`site/`, built into `docs/`).
* Licences: MIT for the code, CC BY 4.0 for text, figures and data.
