# -*- coding: utf-8 -*-
"""L16 (regional hydrodynamic models at the gauges) on synthetic CMEMS fields.

No credentials, no network, no CMEMS file is opened: the dataset folder
holds a manifest over a dummy file (so the sha256 checks of
``fetch.cmems.manifest_files`` run for real) and ``regional.read_fields``
returns synthetic field dicts ``{t, lon, lat, zos}`` with a planted lag.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark import judges, settings
from experiments.benchmark.fetch import cmems
from experiments.benchmark.lags import common, models, regional
from experiments.benchmark.product import NotComputable
from tests.test_benchmark_lags_models import CONS, GAUGE, LAG, MEMBER, NS_MIN, make_instance, tide

AMM15_15MIN = "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i"
IBI_MEAN = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m"


def field(wid, step="15min", lag_lon=3.75, land=True):
    """A synthetic subset over the window +- 2 h: zos = h_B(t - lag), lag 40 min
    east of ``lag_lon``, 0 west of it; a land block of NaN cells."""
    w = settings.window(wid)
    t = pd.date_range(pd.Timestamp(w.start) - pd.Timedelta(hours=2),
                      pd.Timestamp(w.end) + pd.Timedelta(hours=2), freq=step).as_unit("ns")
    lon = np.round(np.arange(3.30, 3.9001, 0.05), 6)
    lat = np.round(np.arange(51.30, 51.5001, 0.05), 6)
    lag = np.where(lon >= lag_lon, LAG, 0.0)
    z = np.empty((len(t), len(lat), len(lon)))
    for j, lg in enumerate(lag):
        z[:, :, j] = tide(t.asi8 - int(round(lg * NS_MIN)))[:, None]
    if land:
        z[:, -1, 4:6] = np.nan                  # a land block (never wet)
    return {"t": t, "lon": lon, "lat": lat, "zos": z}


def write_dataset(inst, dsid, years, status="ok", stamp=None):
    """A dataset folder: dummy yearly files and a manifest (sha256 of each)."""
    d = regional.dataset_dir(inst, dsid)
    os.makedirs(d, exist_ok=True)
    files = {}
    for y in years:
        name = f"{dsid}_{y}.nc"
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(f"synthetic {dsid} {y}".encode())
        files[str(y)] = {"name": name, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                         "size": os.path.getsize(p)}
    man = {"format": cmems.MANIFEST_FORMAT, "site": inst.site, "dataset": dsid,
           "status": status, "reason": "" if status == "ok" else f"synthetic {status}",
           "files": files if status == "ok" else {}}
    with open(os.path.join(d, cmems.MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(man, f)
    if stamp is not None:
        y0 = str(years[0])
        with open(os.path.join(d, cmems.STAMP_SHIFT_NAME), "w", encoding="utf-8") as f:
            json.dump({"shift_min": stamp, "files": {"mean": {"name": files[y0]["name"],
                                                              "sha256": files[y0]["sha256"]}}}, f)
    return d


@pytest.fixture
def fake_cmems(monkeypatch):
    state = {"fields": None, "cred": {"status": "ok", "reason": "", "path": "x",
                                      "verified_online": False}}
    monkeypatch.setattr(regional, "read_fields", lambda paths: state["fields"])
    monkeypatch.setattr(cmems, "check_credentials", lambda *a, **k: dict(state["cred"]))
    monkeypatch.setattr(models, "CONSTITUENTS", CONS)

    def no_gauge(*a, **k):
        raise AssertionError("L16 must never read a gauge series")
    monkeypatch.setattr(judges, "read_series", no_gauge)
    return state


def _inst(tmp_path, site="escalda", wid="W1", **kw):
    """The instance paths of instance.instance_paths: cmems_dir is the CMEMS
    root of the package (``<root>/common``)."""
    return make_instance(wid, site=site, paths={"root": str(tmp_path),
                                                "cmems_dir": f"{tmp_path}/common"}, **kw)


def test_l16_recovers_a_planted_lag(fake_cmems, tmp_path):
    inst = _inst(tmp_path)
    fake_cmems["fields"] = [field("W1")]
    write_dataset(inst, AMM15_15MIN, [2025])
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert r["product"] == "L16" and r["status"] == "ok" and r["kind"] == "point"
    e = r["point"]["jA"]
    for k in ("a", "ab", "hwlw", "internal_a"):
        assert abs(e[k] - LAG) <= 1.0, (k, e[k])
    for k in ("b", "internal_b"):
        assert abs(e[k] - LAG) <= 0.5, (k, e[k])
    assert e["dist_km"] < 0.01 and "unresolved" not in e["flags"]
    assert "degenerate" not in e["flags"] and e["model"].startswith("AMM15 1.5 km")
    assert r["diag"]["dataset"] == AMM15_15MIN and r["diag"]["stamp_shift_min"] == 0.0
    assert r["diag"]["n_wet_cells"] == 65 - 2
    # the dataset folder of fetch.cmems (site_cache_dir under the CMEMS root)
    want = cmems.site_cache_dir("escalda", f"{tmp_path}/common", AMM15_15MIN)
    assert regional.dataset_dir(inst, AMM15_15MIN) == want and r["diag"]["dir"] == want
    site_cmems = make_instance("W1", paths={"cmems_dir": f"{tmp_path}/x/escalda/cmems"})
    want = f"{tmp_path}/x/escalda/cmems/{AMM15_15MIN}".replace(chr(92), "/")
    assert regional.dataset_dir(site_cmems, AMM15_15MIN) == want


def test_nearest_wet_cell_beyond_1km_is_unresolved(fake_cmems, tmp_path):
    far = dict(GAUGE, code="jF", lat=51.40, lon=3.825)      # 0.025 deg east: ~1.7 km
    land = dict(GAUGE, code="jL", lat=51.50, lon=3.50)      # on the land block
    inst = _inst(tmp_path, judges_=[dict(GAUGE), far, land])
    fake_cmems["fields"] = [field("W1")]
    write_dataset(inst, AMM15_15MIN, [2025])
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    p = r["point"]
    assert "unresolved" not in p["jA"]["flags"]
    assert "unresolved" in p["jF"]["flags"] and 1.0 < p["jF"]["dist_km"] < 2.5
    assert "unresolved" in p["jL"]["flags"] and p["jL"]["dist_km"] > 3.0
    c = p["jL"]["cell"]
    assert not (c["lat"] == 51.5 and c["lon"] in (3.5, 3.55))   # never a land cell


def test_credentials_missing_is_not_computable_never_zero(fake_cmems, tmp_path):
    inst = _inst(tmp_path)
    fake_cmems["fields"] = [field("W1")]
    fake_cmems["cred"] = {"status": cmems.STATUS_CREDENTIALS, "reason": "no credentials file",
                          "path": "x", "verified_online": False}
    with pytest.raises(NotComputable) as ei:                    # never fetched
        regional.run_unit(inst, {"products": ("L16",)}, {})
    assert ei.value.stage == "credentials missing" and "missing_input" in ei.value.diag
    write_dataset(inst, AMM15_15MIN, [2025], status=cmems.STATUS_CREDENTIALS)
    with pytest.raises(NotComputable) as ei:
        regional.run_unit(inst, {"products": ("L16",)}, {})
    assert ei.value.stage == "credentials missing"
    # a fetched, sha-verified subset needs no account at run time
    write_dataset(inst, AMM15_15MIN, [2025])
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert r["status"] == "ok" and r["diag"]["credentials"]["status"] == cmems.STATUS_CREDENTIALS


def test_manifest_not_ok_or_changed_files_are_not_available(fake_cmems, tmp_path):
    inst = _inst(tmp_path)
    fake_cmems["fields"] = [field("W1")]
    write_dataset(inst, AMM15_15MIN, [2025], status=cmems.STATUS_NOT_AVAILABLE)
    with pytest.raises(NotComputable) as ei:
        regional.run_unit(inst, {"products": ("L16",)}, {})
    assert ei.value.stage == "not available"
    d = write_dataset(inst, AMM15_15MIN, [2025])
    with open(os.path.join(d, f"{AMM15_15MIN}_2025.nc"), "ab") as f:
        f.write(b"changed")
    with pytest.raises(NotComputable) as ei:
        regional.run_unit(inst, {"products": ("L16",)}, {})
    assert ei.value.stage == "not available" and "changed" in ei.value.reason
    # no dataset covers the window (AMM15 starts 2024-08-04): never substituted
    with pytest.raises(NotComputable) as ei:
        regional.run_unit(_inst(tmp_path, wid="Y2023"), {"products": ("L16",)}, {})
    assert ei.value.stage == "not available" and ei.value.diag["dataset"] is None


def test_unmeasured_hourly_mean_stamp_is_not_computable(fake_cmems, tmp_path):
    inst = _inst(tmp_path, site="ferrol", wid="Y2024")
    assert cmems.choose_dataset("ferrol", "2024-01-01", "2025-01-01")[0] == IBI_MEAN
    fake_cmems["fields"] = [field("Y2024", step="1h")]
    write_dataset(inst, IBI_MEAN, [2024])
    with pytest.raises(NotComputable) as ei:
        regional.run_unit(inst, {"products": ("L16",)}, {})
    assert ei.value.stage == "stamp shift not measured"
    # once measured, the hourly means are read with the shift
    write_dataset(inst, IBI_MEAN, [2024], stamp=0.0)
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert r["status"] == "ok" and r["diag"]["stamp_shift_min"] == 0.0
    assert abs(r["point"]["jA"]["a"] - LAG) <= 1 and abs(r["point"]["jA"]["b"] - LAG) <= 0.5


def test_villaviciosa_is_degenerate_and_adjacent_cells_too(fake_cmems, tmp_path):
    inst = _inst(tmp_path, site="villaviciosa", wid="Y2024")
    fake_cmems["fields"] = [field("Y2024", step="1h")]
    write_dataset(inst, IBI_MEAN, [2024], stamp=0.0)
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert "degenerate" in r["point"]["jA"]["flags"] and r["diag"]["degenerate_site"]
    # elsewhere: degenerate only when the gauge and a member cell are adjacent
    near = dict(MEMBER, lat=GAUGE["lat"], lon=GAUGE["lon"] - 0.05)
    inst = _inst(tmp_path / "f", site="ferrol", wid="Y2024", members=[near])
    write_dataset(inst, IBI_MEAN, [2024], stamp=0.0)
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert "degenerate" in r["point"]["jA"]["flags"]
    assert r["point"]["jA"]["adjacent_members"] == ["bnd"]


def test_empty_without_judges_and_card(fake_cmems, tmp_path):
    inst = _inst(tmp_path, judges_=[])
    (r,) = regional.run_unit(inst, {"products": ("L16",)}, {})
    assert r["status"] == "empty" and r["point"] == {}
    assert common.check_card(regional.CARDS["L16"]) == []
    json.dumps(regional.CONFIG)
    write_dataset(inst, AMM15_15MIN, [2025])
    ext = regional.external_inputs(inst, {"method": "L16"})
    assert len(ext["cmems_manifest"]) == 64 and ext["judges.json"] is None
