# -*- coding: utf-8 -*-
"""Validation v2: ``fetch stac-meta``, the scene metadata sidecar
(``experiments/benchmark/fetch/stac_meta.py``; spec §1.10). No network: the
STAC items cache is a recorded stand-in."""
import json
import os

import pytest

from experiments.benchmark import inputs, sensing_time
from experiments.benchmark.fetch import sensing as fs
from experiments.benchmark.fetch import stac_meta as sm


def _rec(id_, dt, orbit, platform="sentinel-2a", tile="31UET"):
    return {"id": id_, "tile": tile, "platform": platform, "relative_orbit": orbit,
            "absolute_orbit": 40000, "orbit_state": "descending", "datatake_id": "GS2A_x",
            "datetime": dt, "start_datetime": dt, "cloud_cover": 12.5,
            "processing_version": "05.10", "bbox": [3.0, 51.0, 4.0, 52.0],
            "sun_azimuth": 160.0, "sun_elevation": 30.0, "view_azimuth": 100.0,
            "view_incidence": 5.0}


def _cache(site, recs, years=(2023,)):
    doc = fs.new_cache(site, [3.0, 51.0, 4.0, 52.0])
    for r in recs:
        doc["items"][r["id"]] = r
    for y in years:
        doc["chunks"][f"{y}-01-01/{y + 1}-01-01"] = {"complete": True, "n_items": len(recs)}
    return doc


RECS = [_rec("S2A_MSIL2A_20230103_R051_T31UET", "2023-01-03T10:55:01.024Z", 51),
        _rec("S2A_MSIL2A_20230103_R051_T31UFT", "2023-01-03T10:55:01.024Z", 51, tile="31UFT"),
        _rec("S2B_MSIL2A_20230108_R008_T31UET", "2023-01-08T10:45:21.024Z", 8,
             platform="sentinel-2b"),
        _rec("S2A_MSIL2A_20240103_R051_T31UET", "2024-01-03T10:55:01.024Z", 51)]


def test_the_sidecar_summarises_the_cached_items_per_date():
    meta = sm.build_meta("escalda", _cache("escalda", RECS), [2023])
    assert meta["format"] == inputs.SCENE_META_FORMAT and meta["site"] == "escalda"
    assert sorted(meta["dates"]) == ["2023-01-03", "2023-01-08"]     # 2024 is outside
    d = meta["dates"]["2023-01-03"]
    assert d["relative_orbits"] == [51] and d["tiles"] == ["31UET", "31UFT"]
    assert d["platforms"] == ["sentinel-2a"] and d["processing_baselines"] == ["05.10"]
    assert not d["double_overpass"] and d["n_items"] == 2
    assert meta["dates"]["2023-01-08"]["relative_orbits"] == [8]
    assert meta["query"]["years"] == [2023] and meta["query"]["end_exclusive"] == "2024-01-01"


def test_run_writes_once_refuses_by_guard_and_never_rewrites_unchanged(tmp_path, monkeypatch):
    cubes = str(tmp_path / "cubes")
    calls = []

    def fetch_items(site, bbox, start, end, out_root=None, session=None, refresh=False):
        calls.append((site, start, end, refresh))
        return _cache(site, RECS, years=(2023,))

    monkeypatch.setattr(fs, "fetch_items", fetch_items)
    monkeypatch.setattr(sensing_time, "site_geometry",
                        lambda site, root=None, **kw: {"bbox": [3.0, 51.0, 4.0, 52.0]})
    assert sm.run(["escalda"], root=cubes, years=[2023]) == 0
    p = inputs.scene_meta_path("escalda", cubes)
    meta = inputs.read_scene_meta(p)
    assert meta["dates"]["2023-01-08"]["relative_orbits"] == [8]
    assert calls == [("escalda", "2023-01-01", "2024-01-01", False)]
    mtime = os.stat(p).st_mtime_ns
    assert sm.run(["escalda"], root=cubes, years=[2023]) == 0           # unchanged: kept
    assert os.stat(p).st_mtime_ns == mtime
    assert sm.run(["escalda"], root=cubes, years=[2023],
                  guard=lambda s: "the record has provisional offsets") == 1
    # a site without a store and no years given has nothing to describe
    assert sm.run(["ems"], root=cubes) == 1


def test_an_incomplete_cache_is_not_written(tmp_path, monkeypatch):
    cubes = str(tmp_path / "cubes")
    monkeypatch.setattr(fs, "fetch_items", lambda *a, **k: _cache("escalda", RECS, years=()))
    monkeypatch.setattr(sensing_time, "site_geometry",
                        lambda site, root=None, **kw: {"bbox": [3.0, 51.0, 4.0, 52.0]})
    assert sm.run(["escalda"], root=cubes, years=[2023]) == 1
    assert not inputs.scene_meta_path("escalda", cubes).exists()
    assert json.dumps(sm.records_to_features(RECS[:1]))       # JSON-ready features
