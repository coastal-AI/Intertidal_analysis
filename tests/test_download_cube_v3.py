# -*- coding: utf-8 -*-
"""experiments/download_cube_v3.py: the queue, the job ledger and the credit stop, on a fake backend.

* the queue keeps the order given, once per part;
* a part's job id is in its ledger BEFORE the job is started (a killed run leaves a job to reattach);
* a recorded job is reattached, not paid again; one that ended in error is replaced only with --retry-failed;
* a refusal for credits (HTTP 402) stops the queue; the parts not reached are listed;
* a finished part is skipped; the merge keeps the parts and refuses a grid that differs.

Run:  python -m pytest -q tests/test_download_cube_v3.py
"""
import json
import os
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import download_cube_v3 as dcv  # noqa: E402

openeo_rest = pytest.importorskip("openeo.rest")


def write_part(path, dates, x0=500000.0, y0=4800000.0, shape=(6, 5), seed=0):
    rng = np.random.default_rng(seed)
    H, W = shape
    t = pd.to_datetime(dates)
    data = {b: (("t", "y", "x"), rng.integers(1, 3000, (len(t), H, W)).astype("int16")) for b in ("B03", "B08")}
    data["SCL"] = (("t", "y", "x"), rng.integers(1, 11, (len(t), H, W)).astype("int16"))
    ds = xr.Dataset(data, coords={"t": t, "y": y0 - 10.0 * np.arange(H), "x": x0 + 10.0 * np.arange(W)})
    ds["crs"] = xr.DataArray(0, attrs={"crs_wkt": "EPSG:32630"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path, encoding={b: {"zlib": True, "_FillValue": -32768} for b in ("B03", "B08", "SCL")})
    return path


class FakeJob:
    def __init__(self, conn, job_id, statuses=("running", "finished"), costs=12.0, src=None, start_exc=None):
        self.conn, self.job_id, self.statuses = conn, job_id, list(statuses)
        self.costs, self.src, self.start_exc = costs, src, start_exc
        self.started = 0
        self.state = "created"

    def status(self):
        return self.state

    def start(self):
        self.conn.events.append(("start", self.job_id, os.path.isfile(self.conn.ledger_hint or "")))
        if self.start_exc is not None:
            raise self.start_exc
        self.started += 1
        self.state = "queued"

    def describe(self):
        if self.statuses:
            self.state = self.statuses.pop(0)
        return {"status": self.state, "costs": self.costs if self.state in ("finished", "error") else None}

    def get_results(self):
        src = self.src
        return types.SimpleNamespace(get_assets=lambda: [types.SimpleNamespace(
            name="openEO.nc", download=lambda p: shutil.copy(src, p))])

    def logs(self, level=None):
        return [{"message": "executor lost"}]


class FakeConn:
    def __init__(self, src, start_exc=None):
        self.src, self.start_exc = src, start_exc
        self.jobs, self.created, self.events = {}, [], []
        self.ledger_hint = None

    def job(self, job_id):
        if job_id not in self.jobs:
            raise openeo_rest.OpenEoApiError(http_status_code=404, code="JobNotFound", message="gone")
        return self.jobs[job_id]

    def load_collection(self, collection, **kw):
        self.events.append(("load", kw["spatial_extent"], tuple(kw["temporal_extent"]), tuple(kw["bands"])))
        conn = self

        class Cube:
            def resample_spatial(self, resolution, method):
                assert (resolution, method) == (10, "near")
                return self

            def save_result(self, format):
                assert format == "netCDF"
                return self

            def create_job(self, title, job_options):
                j = FakeJob(conn, f"j-{len(conn.created) + 1}", src=conn.src, start_exc=conn.start_exc)
                conn.jobs[j.job_id] = j
                conn.created.append((title, job_options))
                return j

        return Cube()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(dcv, "PART_ROOT", str(tmp_path / "cubes_v3"))
    monkeypatch.setattr(dcv, "POLL_S", 0)
    monkeypatch.setattr(dcv.time, "sleep", lambda s: None)
    seen = []
    monkeypatch.setattr(dcv, "update_overpass", lambda site, year, dates, log: seen.append((site, year)) or [])
    src = write_part(str(tmp_path / "delivered.nc"), ["2020-01-03", "2020-01-08", "2020-02-12"])
    args = types.SimpleNamespace(retry_failed=False, job_options=dict(dcv.JOB_OPTIONS), account="x@y",
                                 max_wait_h=1.0)
    return types.SimpleNamespace(tmp=tmp_path, src=src, args=args, overpass=seen)


def test_the_queue_keeps_the_order_given_once_per_part():
    q = dcv.parse_queue(["urdaibai:2020", "urdaibai:2023", "foz:2019-2021", "urdaibai:2020", "eo:2020"])
    assert q == [("urdaibai", 2020), ("urdaibai", 2023), ("foz", 2019), ("foz", 2020), ("foz", 2021), ("eo", 2020)]
    for bad in ("mars:2020", "foz", "foz:2014", "foz:2021-2019", "foz:20x0"):
        with pytest.raises(ValueError):
            dcv.parse_queue([bad])
    assert dcv.year_window(2020) == ("2020-01-01", "2021-01-01")


def test_the_boxes_hold_the_estuaries():
    # the OpenStreetMap bay polygons the boxes were drawn around (module comment)
    inside = {"foz": (-7.2667, 43.5435, -7.2361, 43.5751), "eo": (-7.0638, 43.4717, -7.0053, 43.5568),
              "urdaibai": (-2.6930, 43.3100, -2.6488, 43.3979)}
    for site, (w, s, e, n) in inside.items():
        b = dcv.BOXES[site]
        assert b["west"] < w and b["south"] < s and b["east"] > e and b["north"] > n, site
        rows, cols = dcv.box_px(b)
        assert 500 < rows < 2000 and 500 < cols < 1200, (site, rows, cols)


def test_a_new_part_records_its_job_before_starting_it(env):
    conn = FakeConn(env.src)
    conn.ledger_hint = dcv.ledger_path("foz", 2020)
    r = dcv.fetch(conn, "foz", 2020, env.args, lambda m: None)
    assert r == "downloaded"
    assert conn.events[0] == ("load", dcv.BOXES["foz"], ("2020-01-01", "2021-01-01"), ("B03", "B08", "SCL"))
    assert ("start", "j-1", True) in conn.events, "the ledger must exist when the job is started"
    led = json.load(open(dcv.ledger_path("foz", 2020)))
    assert [j["job_id"] for j in led["jobs"]] == ["j-1"]
    assert led["jobs"][0]["status"] == "finished" and led["jobs"][0]["costs"] == 12.0
    assert led["dates"] == 3 and led["first"] == "2020-01-03"
    assert dcv.part_info(dcv.part_path("foz", 2020))["dates"] == ["2020-01-03", "2020-01-08", "2020-02-12"]
    assert env.overpass == [("foz", 2020)]
    assert not os.path.exists(dcv.part_path("foz", 2020) + ".download")
    # a second run skips it: no new job
    assert dcv.fetch(conn, "foz", 2020, env.args, lambda m: None) == "cached"
    assert len(conn.created) == 1


def test_a_recorded_job_is_reattached_not_paid_again(env):
    conn = FakeConn(env.src)
    job = FakeJob(conn, "j-old", statuses=("running", "running", "finished"), src=env.src)
    job.state = "running"
    conn.jobs["j-old"] = job
    led = dcv.load_ledger("eo", 2020)
    led["jobs"].append({"job_id": "j-old", "status": "created"})
    dcv.write_json(led, dcv.ledger_path("eo", 2020))
    assert dcv.fetch(conn, "eo", 2020, env.args, lambda m: None) == "downloaded"
    assert conn.created == [] and job.started == 0, "a running job is waited for, neither recreated nor restarted"


def test_a_job_in_error_is_replaced_only_with_retry_failed(env):
    conn = FakeConn(env.src)
    job = FakeJob(conn, "j-bad", src=env.src)
    job.state = "error"
    conn.jobs["j-bad"] = job
    led = dcv.load_ledger("eo", 2021)
    led["jobs"].append({"job_id": "j-bad", "status": "running"})
    dcv.write_json(led, dcv.ledger_path("eo", 2021))
    assert dcv.fetch(conn, "eo", 2021, env.args, lambda m: None) == "failed"
    assert conn.created == []
    env.args.retry_failed = True
    assert dcv.fetch(conn, "eo", 2021, env.args, lambda m: None) == "downloaded"
    led = json.load(open(dcv.ledger_path("eo", 2021)))
    assert [j["job_id"] for j in led["jobs"]] == ["j-bad", "j-1"] and led["jobs"][0]["status"] == "error"


def test_a_job_gone_from_the_backend_is_made_again(env):
    conn = FakeConn(env.src)
    led = dcv.load_ledger("foz", 2019)
    led["jobs"].append({"job_id": "j-expired", "status": "running"})
    dcv.write_json(led, dcv.ledger_path("foz", 2019))
    assert dcv.fetch(conn, "foz", 2019, env.args, lambda m: None) == "downloaded"
    assert len(conn.created) == 1


def test_an_error_at_the_end_is_failed_with_its_costs(env):
    conn = FakeConn(env.src)
    orig = conn.load_collection

    def load(collection, **kw):
        cube = orig(collection, **kw)
        make = cube.create_job

        def create_job(title, job_options):
            j = make(title, job_options)
            j.statuses = ["running", "error"]
            return j

        cube.create_job = create_job
        return cube

    conn.load_collection = load
    assert dcv.fetch(conn, "foz", 2018, env.args, lambda m: None) == "failed"
    led = json.load(open(dcv.ledger_path("foz", 2018)))
    assert led["jobs"][0]["status"] == "error" and led["jobs"][0]["costs"] == 12.0
    assert dcv.part_info(dcv.part_path("foz", 2018)) is None


def test_a_refusal_for_credits_stops_the_queue(env, monkeypatch, capsys):
    import pyintertidal as pit

    exc = openeo_rest.OpenEoApiError(http_status_code=402, code="PaymentRequired", message="insufficient credits")
    conn = FakeConn(env.src, start_exc=exc)
    monkeypatch.setattr(pit.scenes, "connect", lambda **kw: conn)
    monkeypatch.setattr(pit.net, "use_system_certificates", lambda: None)
    rc = dcv.main(["foz:2020", "eo:2020", "urdaibai:2020"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "STOP, credits" in out and "not reached: eo:2020 urdaibai:2020" in out
    assert len(conn.created) == 1, "nothing submitted after the refusal"
    # the refused job's id is recorded: a later run reattaches it instead of creating another
    led = json.load(open(dcv.ledger_path("foz", 2020)))
    assert [j["job_id"] for j in led["jobs"]] == ["j-1"]


def test_the_account_is_checked_before_any_job(env, monkeypatch):
    import pyintertidal as pit

    conn = FakeConn(env.src)
    conn.describe_account = lambda: {"info": {"oidc_userinfo": {"email": "someone@else.org"}}}
    monkeypatch.setattr(pit.scenes, "connect", lambda **kw: conn)
    monkeypatch.setattr(pit.net, "use_system_certificates", lambda: None)
    with pytest.raises(dcv.AccountMismatch):
        dcv.main(["foz:2020", "--expect-account", "f.garcia@ieo.csic.es"])
    assert conn.events == []


def test_the_merge_keeps_the_parts_and_refuses_another_grid(env, monkeypatch):
    monkeypatch.chdir(env.tmp)
    a = write_part(dcv.part_path("foz", 2019), ["2019-03-01", "2019-12-30"], seed=1)
    b = write_part(dcv.part_path("foz", 2020), ["2020-01-03", "2020-05-05", "2020-06-06"], seed=2)
    out = dcv.merge("foz", [2019, 2020], log=lambda m: None)
    assert os.path.isfile(a) and os.path.isfile(b), "the parts are kept"
    with xr.open_dataset(out) as m, xr.open_dataset(a) as pa, xr.open_dataset(b) as pb:
        assert m.sizes["t"] == 5 and "crs" in m
        for band in ("B03", "B08", "SCL"):
            np.testing.assert_array_equal(m[band].values, np.concatenate([pa[band].values, pb[band].values]))
    write_part(dcv.part_path("eo", 2019), ["2019-03-01"], x0=500010.0)
    write_part(dcv.part_path("eo", 2020), ["2020-03-01"])
    with pytest.raises(AssertionError, match="x grid differs"):
        dcv.merge("eo", [2019, 2020], log=lambda m: None)


def test_status_and_dry_run_need_no_network(env, capsys):
    write_part(dcv.part_path("foz", 2020), ["2020-01-03"])
    dcv.status(dcv.parse_queue(["foz:2020", "foz:2021"]))
    dcv.dry_run(dcv.parse_queue(["urdaibai:2020", "foz:2020"]))
    out = capsys.readouterr().out
    assert "downloaded, 1 dates" in out and "not started" in out and "2 parts" in out
