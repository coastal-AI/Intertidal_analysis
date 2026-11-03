# -*- coding: utf-8 -*-
"""The v3 production notebook (notebooks/validation/_build_v3_production.py) and its configuration (experiments/v3/sites.py).

* the built notebook is the template's code: every code cell but the parameters and the products is a template cell,
  changed only where the builder says (the AOI, the overpass file, the log's name), and every code cell parses;
* a template that no longer holds a replaced text stops the build;
* the configuration: one cube per truth, its years around the truth, the download box, the nearest IOC gauge as
  the expected boundary member.

Run:  python -m pytest -q tests/test_v3_production_builder.py
"""
import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebooks", "validation"))

import _build_v3_production as bp  # noqa: E402
from experiments.download_cube_v3 import BOXES  # noqa: E402
from experiments.v3 import sites  # noqa: E402

TEMPLATE_CELLS = (8, 10, 12, 13, 15, 19, 21, 22)       # taken unchanged


def built(tmp_path):
    out = tmp_path / "v3.ipynb"
    bp.build(bp.TEMPLATE, str(out))
    return json.load(open(out, encoding="utf-8"))


def test_the_notebook_is_the_templates_code(tmp_path):
    nb = built(tmp_path)
    tpl = json.load(open(bp.TEMPLATE, encoding="utf-8"))
    codes = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for src in codes:
        ast.parse(src)
    for i in TEMPLATE_CELLS:
        assert bp.source(tpl, i).strip("\n") in codes, f"template cell {i} is not in the notebook unchanged"
    joined = "\n".join(codes)
    assert "pit.sites.get(SITE)" not in joined and 'AOI.from_bbox(**V3C["box"]' in joined
    assert 'OVERPASS_CACHE = f"{OUT_DIR}/overpass_times.json"' not in joined
    assert "Granadeiro" not in joined and "lag_validation" not in joined, "sections 9-10b are left out"
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])


def test_no_later_cell_rebinds_the_parameters(tmp_path):
    # the template's cells bind C (the clear mask): the configuration's name must be one no other cell assigns
    nb = built(tmp_path)
    codes = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    params = codes[0]
    mine = {t.id for node in ast.walk(ast.parse(params)) if isinstance(node, ast.Assign) for t in node.targets
            if isinstance(t, ast.Name)} | {"V3", "NEW"}
    for src in codes[1:]:
        bound = {t.id for node in ast.walk(ast.parse(src)) if isinstance(node, (ast.Assign, ast.AugAssign))
                 for t in (node.targets if isinstance(node, ast.Assign) else [node.target]) if isinstance(t, ast.Name)}
        assert not ({"V3C", "KEY", "V3", "NEW"} & bound), sorted({"V3C", "KEY", "V3", "NEW"} & bound)
    assert "V3C" in mine


def test_a_changed_template_stops_the_build(tmp_path):
    tpl = json.load(open(bp.TEMPLATE, encoding="utf-8"))
    src = bp.source(tpl, 6).replace("aoi = pit.sites.get(SITE)", "aoi = pit.sites.get(SITE, 1)")
    tpl["cells"][6]["source"] = src
    bad = tmp_path / "template.ipynb"
    json.dump(tpl, open(bad, "w", encoding="utf-8"))
    with pytest.raises(SystemExit, match="template changed"):
        bp.build(str(bad), str(tmp_path / "out.ipynb"))


def test_the_configuration():
    assert set(sites.NEW) == {"urdaibai_2017", "urdaibai_2024", "foz_2016", "eo_2020", "santander_maps", "santona_maps"}
    assert set(sites.TRUTHS) == {"urdaibai_2017", "urdaibai_2024", "foz_2016", "eo_2020", "villaviciosa_2020"}
    for key in ("santander_maps", "santona_maps"):
        c = sites.V3[key]
        assert c["maps_only"] and c["truth"] is None and c["years"] == (2023, 2025) and c["ns"] == (100,)
        assert c["box"] == pytest.importorskip("pyintertidal").sites.get(c["site"]).bbox
    for key in set(sites.NEW) & set(sites.TRUTHS):
        c = sites.V3[key]
        y0, y1 = c["years"]
        truth_year = int(c["truth_date"][:4])
        assert y0 <= truth_year <= y1 and c["cube"] == f"ndwi_cube_{c['site']}_{y0}-{y1}_10m.nc"
        assert c["box"] == BOXES[c["site"]] and c["n_gauges"] == 1 and c["judges"] == []
        assert c["mouth_side"] == "north" and c["truth"].endswith(f"{key}_10m.tif")
    a, b = sites.V3["urdaibai_2017"]["years"], sites.V3["urdaibai_2024"]["years"]
    assert a[1] < 2020 and b[0] > 2021, "no Urdaibai cube spans the 2020-2021 gap"
    assert sites.V3["villaviciosa_2020"]["cube"] == "ndwi_cube_villaviciosa_grande_10y.nc"


def test_the_expected_gauges_are_the_nearest():
    pytest.importorskip("shapely")
    from pyintertidal.aoi import AOI
    from pyintertidal.boundary import nearest_gauges

    if not os.path.exists(os.path.join(ROOT, "data_v4", "gauges", "ioc_stations.json")):
        pytest.skip("no IOC station list on this machine")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for key in sites.NEW:
            c = sites.V3[key]
            lat, lon = AOI.from_bbox(**c["box"]).centroid
            assert nearest_gauges(lat, lon, 1)[0][0] == c["expected_gauge"], key
    finally:
        os.chdir(cwd)
