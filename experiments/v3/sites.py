# -*- coding: utf-8 -*-
"""The truths of validation v3 and the production configuration of their cubes, in one place.

One entry per truth (a site and a LiDAR coverage), keyed ``<site>_<truth year>``: Urdaibai has two truths, so two
cubes, each holding only the years around its truth (2015-2019 and 2022-2026, without 2020-2021, so no window spans
a gap; see the design's amendment of 2026-11-03). Villaviciosa keeps its production: the 10-year cube, its
extraction and its products (EOT20 at the AOI centroid, no gauge), as every Villaviciosa product so far.

The new sites' production (notebooks/validation/v3_production_site.ipynb) is Ferrol's
(tide_boundary_comparison_ferrol.ipynb): NDWI > 0, the transition-zone cloud screening, EOT20 + the nearest IOC
gauge (``n_gauges=1``; none held out: no gauge lies inside these estuaries), the sea on the north edge of the box,
and the box of experiments/download_cube_v3.py (BOXES) as the AOI.
"""
from __future__ import annotations

import glob
import os

from experiments.download_cube_v3 import BOXES, box as download_box, merged_path, overpass_path

TRUTH_ROOT = "data_v4/truth/pnoa_v3"


def _new(site, years, truth_year, sheets, gauge):
    y0, y1 = years
    key = f"{site}_{truth_year}"
    return key, {
        "key": key, "site": site, "years": years, "period": (f"{y0}-01-01", f"{y1}-12-31"),
        "cube": merged_path(site, y0, y1), "box": BOXES[site],
        "out": f"products_v3_{key}_10m", "overpass": overpass_path(site),
        "tide_model": "EOT20", "n_gauges": 1, "judges": [], "expected_gauge": gauge, "mouth_side": "north",
        "min_scenes": 40,
        "sheets_dir": f"{TRUTH_ROOT}/{sheets}", "truth_date": f"{truth_year}-07-15",
        "truth": f"{TRUTH_ROOT}/{key}_10m.tif",
    }


V3 = dict([
    _new("urdaibai", (2015, 2019), 2017, "mundaka_2017", "bil3"),
    _new("urdaibai", (2022, 2026), 2024, "mundaka_2024", "bil3"),
    _new("foz", (2015, 2019), 2016, "foz", "fer2"),
    _new("eo", (2017, 2023), 2020, "eo", "fer2"),
])
V3["villaviciosa_2020"] = {
    "key": "villaviciosa_2020", "site": "villaviciosa", "years": (2016, 2025),
    "period": ("2016-01-01", "2025-12-31"), "cube": "ndwi_cube_villaviciosa_grande_10y.nc",
    "out": "products_marea", "overpass": "products_marea/overpass_times.json", "tide_model": "EOT20",
    "n_gauges": None, "judges": [], "expected_gauge": None, "mouth_side": "any", "min_scenes": 80,
    "sheets_dir": f"{TRUTH_ROOT}/villaviciosa", "truth_date": "2020-07-15",
    "truth": f"{TRUTH_ROOT}/villaviciosa_cob2_2020_10m.tif", "existing_production": True,
}
#: maps only (the design's amendment of 2026-11-03: their LiDAR was flown at high water): the 2023-2025 cubes, the
#: new sites' production, the window of the 100 screened scenes nearest 2024-07-15, no truth and no score
for _site, _gauge, _side in (("santander", "san2", "north"), ("santona", "san2", "any")):
    _key, _c = _new(_site, (2023, 2025), 2024, None, _gauge)
    _key = f"{_site}_maps"
    _c.update(key=_key, out=f"products_v3_{_key}_10m", box=download_box(_site), mouth_side=_side, sheets_dir=None, truth=None,
              maps_only=True, ns=(100,))
    V3[_key] = _c
#: the keys whose production notebooks/validation/v3_production_site.ipynb makes
NEW = tuple(k for k, v in V3.items() if not v.get("existing_production"))
#: the keys with a LiDAR truth (the others are maps only)
TRUTHS = tuple(k for k, v in V3.items() if not v.get("maps_only"))


def sheets(key):
    """The truth's GeoTIFF sheets, sorted by name (their order where they overlap: the first one wins)."""
    found = sorted(glob.glob(os.path.join(V3[key]["sheets_dir"], "*.tif")))
    if not found:
        raise FileNotFoundError(f"{key}: no sheet in {V3[key]['sheets_dir']}")
    return found
