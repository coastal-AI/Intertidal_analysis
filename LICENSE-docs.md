# Licence of the text, figures and data

Everything in this repository that is not source code (the text of the documentation and the notebooks, the
figures, the data files and the website's data layers) is licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/legalcode

Copyright (c) 2026 Jorge Alías Álvarez, Fernando García-González, Pablo González.

The source code (the `pyintertidal` package, the scripts, the tests, the code cells of the notebooks and the website's
source in `site/`) is licensed under the MIT licence: see [`LICENSE`](LICENSE).

## Third-party data and services

These keep their own licences and must be attributed when you reuse what is derived from them:

* **Sentinel-2.** The products, the replication kit and the sample contain modified Copernicus Sentinel data
  (2015-2026), processed with openEO on the Copernicus Data Space Ecosystem. Copernicus Sentinel data are free, full
  and open (Copernicus Sentinel data legal notice).
* **EOT20 ocean tide model.** Hart-Davis, M., Piccioni, G., Dettmering, D., Schwatke, C., Passaro, M. and Seitz, F.
  (2021). EOT20: a global ocean tide model from multi-mission satellite altimetry. SEANOE,
  [doi:10.17882/79489](https://doi.org/10.17882/79489). CC BY 4.0. The tide levels in `replication/`, in the sample
  and in the website's data are values of this model.
* **Tide-gauge records.** Where a product's tide at the mouth comes from a tide gauge (the Ferrol and Bilbao
  gauges), the records are those of the IOC Sea Level Station Monitoring Facility (VLIZ/IOC),
  [doi:10.14284/482](https://doi.org/10.14284/482), licensed for non-commercial use (CC BY-NC 4.0); acknowledge
  the facility when you reuse them. The records themselves are not redistributed here.
* **Base maps.** The website loads the PNOA orthophoto and the IGN base map of the Instituto Geográfico Nacional
  (CC BY 4.0 scne.es) when it runs; they are not redistributed here.
* **Terrain.** The website's 3D view loads Mapterhorn terrain tiles (built from open elevation models) when it runs;
  they are not redistributed here, and their attribution is shown on the map.

## Third-party code bundled in the website

The built website in `docs/` bundles MapLibre GL JS (BSD-3-Clause), React and React DOM (MIT),
Tailwind CSS (MIT), lucide-react (ISC), clsx (MIT) and the Inter typeface (SIL Open Font License 1.1). Their licence
texts are in `docs/licenses/`.

## Not covered

The institutional and congress logos on the poster (`poster/`) belong to their owners and are not licensed under
CC BY 4.0.
