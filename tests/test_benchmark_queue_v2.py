# -*- coding: utf-8 -*-
"""Validation v2, Stage L: the instance-major run queue
(``experiments/benchmark/queue_v2.py``; plan INTERFACES §5-§9, spec §13.2-§13.5).

Dummy adapters stand in for the lag methods (registered through
monkeypatch); the instances are real, on the small v2 package of
``tests/benchmark_zarr_site.py``.
"""
import glob
import json
import os
import socket
import sys
import time
import types

import numpy as np
import pytest

from experiments.benchmark import designs, lags, product, runner, settings
from experiments.benchmark import instance as I
from experiments.benchmark import queue_v2 as Q
from experiments.benchmark.lags import common
from experiments.benchmark.product import NotComputable

P0 = settings.P0.id


# ─────────────────────────────────────────────────────────────────────────────
#  the package and the dummy adapters
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def zsite(tmp_path_factory):
    from tests import benchmark_zarr_site as zs
    mp = pytest.MonkeyPatch()
    s = zs.make_site(tmp_path_factory.mktemp("v2_queue"), mp)
    yield s
    s.pkg.close()
    mp.undo()


CALLS = []


def _mod(name, tmp, run, config=None, products=("X",), output="band"):
    m = types.ModuleType(name)
    path = tmp / f"{name}.py"
    path.write_text(f"# dummy adapter {name}\n", encoding="utf-8")
    m.__file__ = str(path)
    m.CONFIG = dict(config or {"v": 1})
    m.CARDS = {p: {"id": p, "output": output, "lag_kind": "linear", "kind": "lag"}
               for p in products}
    m.CODE_DEPS = ()
    m.run = run
    return m


def run_band(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id))
    with inst.stage("fit"):
        K = inst.domain.K
        inst.progress(2)
    return [common.lag_result(p, kind="band", band={"tau": np.linspace(10, 40, K),
                                                    "n_px": np.full(K, 7),
                                                    "edge": np.zeros(K, bool)},
                              K=K, diag={"note": "dummy", "arr": np.arange(3)})
            for p in unit["products"]]


def run_field(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id))
    n = inst.domain.n_keep
    return [common.lag_result(unit["products"][0], kind="field", lag_kind="phase",
                              field=np.full(n, 33.0), n_keep=n)]


def run_point(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id))
    return [{"product": unit["products"][0], "status": "ok", "kind": "point",
             "point": {"st_flat": {"a": 0.0, "b": 0.0, "dist_km": 0.0}}}]


def run_nc(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id))
    raise NotComputable("min_scenes", "needs 600 scenes", {"n": len(inst.sel)})


FLAKY = {"fail": True}


def run_flaky(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id))
    if FLAKY["fail"]:
        raise RuntimeError("boom")
    return run_band(inst, unit, opts)


def run_e0(inst, unit, opts):
    CALLS.append((unit["method"], inst.instance_id, opts["n_boot"]))
    n = inst.domain.n_keep
    out = []
    for p in unit["products"]:
        if p == "L01":
            out.append(run_band(inst, dict(unit, products=("L01",)), opts)[0])
        else:
            out.append(common.elev_result(p, z=np.zeros(n), sigma=np.ones(n),
                                          clear_obs=np.full(n, 3), tau_px=np.zeros(n),
                                          n_keep=n))
    return out


def run_slow(inst, unit, opts):
    time.sleep(0.5)
    return run_band(inst, unit, opts)


@pytest.fixture
def reg(monkeypatch, tmp_path):
    """Dummy adapters for E0, L02, L03, L07, L12, L17 (and L04 for slow units)."""
    CALLS.clear()
    FLAKY["fail"] = True
    mods = {
        "E0": _mod("dq_e0", tmp_path, run_e0, products=("E0a", "E0b", "L01")),
        "L02": _mod("dq_band", tmp_path, run_band, products=("L02",)),
        "L03": _mod("dq_nc", tmp_path, run_nc, products=("L03",)),
        "L07": _mod("dq_flaky", tmp_path, run_flaky, products=("L07",)),
        "L12": _mod("dq_field", tmp_path, run_field, products=("L12",), output="field"),
        "L17": _mod("dq_point", tmp_path, run_point, products=("L17",), output="point"),
        "L04": _mod("dq_slow", tmp_path, run_slow, products=("L04w", "L04i")),
    }
    for meth, m in mods.items():
        monkeypatch.setitem(sys.modules, m.__name__, m)
        monkeypatch.setitem(lags.REGISTRY, meth, f"{m.__name__}:run")
    return mods


def _opts(zsite, root, rows=None, **kw):
    kw.setdefault("ram_total_gb", 1000.0)
    return Q.QueueOptions(root=root, rows=rows, site2_for=lambda s: zsite.site2,
                          provider_for=lambda s: zsite.provider, **kw)


def _rows(*specs):
    return [Q.make_row("lag", "escalda", w, P0, m, variant=v) for w, m, v in specs]


def _status(root, unit_row, prod):
    r = I.norm_row(unit_row)
    u = Q.unit_of(r)
    return Q.read_run(Q.product_dir(root, u, prod))


# ─────────────────────────────────────────────────────────────────────────────
#  instance-major running, publishing, resume
# ─────────────────────────────────────────────────────────────────────────────

def test_instance_major_one_build_per_instance(zsite, reg, tmp_path, monkeypatch):
    builds = []
    real = I.build_instance

    def counting(*a, **k):
        builds.append(a[1]["window"])
        return real(*a, **k)

    monkeypatch.setattr(I, "build_instance", counting)
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L12", ""), ("Y2024", "L02", ""),
                 ("W3", "L17", ""), ("Y2024", "L17", ""))
    c = Q.run(_opts(zsite, root, rows))
    assert builds == ["W3", "Y2024"]                       # one build per instance
    assert c["ran"] == 5 and c["status:ok"] == 5
    w3 = [x for x in CALLS if x[0] in ("L02", "L12", "L17")]
    assert len({x[1] for x in w3}) == 2
    inst_dirs = glob.glob(os.path.join(root, "runs", "escalda", "*", P0, "_instance", "*", "*"))
    assert len(inst_dirs) == 2
    run = _status(root, rows[0], "L02")
    assert run["status"] == "ok" and run["kind"] == "band" and run["designs"] == ["lag"]
    d = Q.product_dir(root, Q.unit_of(I.norm_row(rows[0])), "L02")
    p = Q.load_lag_product(d)
    K = p["instance"]["K"]
    band_of = p["instance"]["band_of"].astype(np.int64)
    # the queue broadcast the band lags to the field (the adapter left it out)
    assert np.array_equal(p["field"], common.broadcast_band(p["band"]["tau"], band_of))
    assert p["band"]["tau"].shape == (K,) and p["diag"]["note"] == "dummy"
    assert np.array_equal(p["diag"]["arr"], np.arange(3))
    assert "lag.L02.fit" in p["timings"] and p["unit_total"]["wall_s"] >= 0
    assert set(p["instance_timings"]) == {"domain.build", "write.instance"}
    assert p["timings"]["lag.L02.fit"]["progress"] == 2
    assert p["instance"]["json"]["format"] == I.INSTANCE_FORMAT
    assert p["keys"]["subset_key"] == p["instance"]["json"]["subset_key"]
    ph = Q.load_lag_product(Q.product_dir(root, Q.unit_of(I.norm_row(rows[1])), "L12"))
    assert ph["lag_kind"] == "phase" and ph["edge_share"] == 0.0
    pt = _status(root, rows[3], "L17")
    assert pt["point"]["st_flat"]["a"] == 0.0 and np.isnan(pt["point"]["st_flat"]["ab"])
    log = open(os.path.join(d, "unit.log"), encoding="utf-8").read()
    assert "=== unit escalda W3" in log
    found = {r["product"] for _, r in Q.iter_lag_products(root)}
    assert found == {"L02", "L12", "L17"}
    led = Q.read_ledger(root, I.norm_row(rows[0])["unit_id"])
    assert led["status"] == "ok" and led["products"] == {"L02": "ok"}


def test_resume_skips_done_and_a_config_change_reruns_only_that_method(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L12", ""))
    assert Q.run(_opts(zsite, root, rows))["ran"] == 2
    CALLS.clear()
    c = Q.run(_opts(zsite, root, rows))
    assert c == {"done": 2} and CALLS == []
    reg["L02"].CONFIG["v"] = 2                              # a changed CONFIG
    c = Q.run(_opts(zsite, root, rows))
    assert c.get("ran") == 1 and c.get("done") == 1 and [x[0] for x in CALLS] == ["L02"]
    d = Q.product_dir(root, Q.unit_of(I.norm_row(rows[0])), "L02")
    assert glob.glob(d + ".old-*")                          # the first result is kept
    c = Q.run(_opts(zsite, root, rows, smoke=True))         # smoke never satisfies a real run
    assert c.get("ran") == 2


def test_not_computable_marks_every_product_of_the_unit(zsite, reg, tmp_path, monkeypatch):
    root = str(tmp_path / "root")
    monkeypatch.setattr(reg["E0"], "run", run_nc)
    rows = _rows(("W3", "E0", "boot30"), ("W3", "L03", ""))
    c = Q.run(_opts(zsite, root, rows))
    assert c["status:not_computable"] == 2
    for p in ("E0a", "E0b", "L01"):
        r = _status(root, rows[0], p)
        assert r["status"] == "not_computable" and r["stage"] == "min_scenes"
        assert r["reason"] == "needs 600 scenes" and r["diag"]["n"] == 64
    assert _status(root, rows[0], "E0a")["kind"] == "elev"
    assert _status(root, rows[1], "L03")["kind"] == "band"
    assert Q.run(_opts(zsite, root, rows)) == {"done": 2}  # not_computable is final


def test_e0_elevation_products_and_options(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "E0", "boot30"))
    Q.run(_opts(zsite, root, rows))
    assert CALLS[0][2] == 30                                # n_boot 30 for boot30
    u = Q.unit_of(I.norm_row(rows[0]))
    e = Q.load_lag_product(Q.product_dir(root, u, "E0b"))
    assert e["kind"] == "elev" and e["z"].dtype == np.float32
    assert e["clear_obs"].dtype == np.int16 and e["opts"]["n_boot"] == 30
    assert Q.product_dir(root, u, "L01").endswith(f"{P0}/L01__boot30/rand-A/Nall_r00")
    assert Q.unit_opts(I.norm_row(_rows(("W3", "L08", "boot30"))[0]))["refits"] == 20
    assert Q.unit_opts(I.norm_row(_rows(("W3", "E0", "seed3"))[0]))["seed_offset"] == 3
    # GM20's seed rows (design base) rerun its only random stage, the refits
    o = Q.unit_opts(I.norm_row(_rows(("W3", "L08", "seed2"))[0]))
    assert o["refits"] == 20 and o["seed_offset"] == 2 and o["n_boot"] == 0
    assert Q.unit_opts(I.norm_row(_rows(("W3", "L08", ""))[0]))["refits"] == 0
    assert Q.unit_opts(I.norm_row(_rows(("W3", "E0", "seed3"))[0]))["refits"] == 0


def test_an_error_is_not_final_and_the_rerun_keeps_the_attempt(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L07", ""), ("W3", "L02", ""))
    c = Q.run(_opts(zsite, root, rows))
    assert c["status:error"] == 1 and c["status:ok"] == 1
    r = _status(root, rows[0], "L07")
    assert r["status"] == "error" and "RuntimeError: boom" in r["reason"]
    assert "Traceback" in r["diag"]["traceback"]
    FLAKY["fail"] = False
    c = Q.run(_opts(zsite, root, rows))
    assert c["ran"] == 1 and c["done"] == 1
    d = Q.product_dir(root, Q.unit_of(I.norm_row(rows[0])), "L07")
    assert _status(root, rows[0], "L07")["status"] == "ok"
    old = glob.glob(d + ".old-*")
    assert len(old) == 1 and Q.read_run(old[0])["status"] == "error"
    assert [x for x, _ in Q.iter_lag_products(root)] and \
        all(".old-" not in x for x, _ in Q.iter_lag_products(root))


def test_a_wrong_result_list_is_an_error(zsite, reg, tmp_path, monkeypatch):
    root = str(tmp_path / "root")
    monkeypatch.setattr(reg["L02"], "run", lambda inst, unit, opts: [])
    rows = _rows(("W3", "L02", ""))
    Q.run(_opts(zsite, root, rows))
    r = _status(root, rows[0], "L02")
    assert r["status"] == "error" and "expected ['L02']" in r["reason"]


def test_a_module_not_built_fails_only_its_own_units(zsite, reg, tmp_path, monkeypatch):
    root = str(tmp_path / "root")
    monkeypatch.setitem(lags.REGISTRY, "L11", "experiments.benchmark.lags.not_yet_built:run")
    rows = _rows(("W3", "L11", ""), ("W3", "L02", ""))
    c = Q.run(_opts(zsite, root, rows))
    assert c["not_built"] == 1 and c["status:ok"] == 1
    led = Q.read_ledger(root, I.norm_row(rows[0])["unit_id"])
    assert led["status"] == "error" and led["reason"].startswith("module not built")
    assert _status(root, rows[0], "L11") is None


# ─────────────────────────────────────────────────────────────────────────────
#  plans, not_applicable, unavailable, STOP
# ─────────────────────────────────────────────────────────────────────────────

def _write_plan(root, rows, name="lag"):
    os.makedirs(os.path.join(root, "plans"), exist_ok=True)
    with open(os.path.join(root, "plans", f"{name}.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(designs.plan_csv(rows))


def test_plans_not_applicable_and_shared_rows_never_run(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L17", ""))
    na = dict(Q.make_row("lag", "escalda", "W3", P0, "L12"), status="not_applicable",
              note="NOT AVAILABLE: no archive")
    shared = dict(rows[0], status="shared", note=f"shares unit {rows[0]['unit_id']}")
    _write_plan(root, rows + [na, shared])
    c = Q.run(_opts(zsite, root, plans=("lag",)))
    assert c["ran"] == 2 and not any(x[0] == "L12" for x in CALLS)
    st = Q.write_status(root)
    counts = st["lag"]["counts"]
    assert counts == {"done": 3, "not_applicable": 1}         # the shared row reads done
    text = open(st["lag"]["path"], encoding="utf-8").read()
    assert "NOT AVAILABLE: no archive" in text


def test_unavailable_is_rechecked_on_every_pass(zsite, reg, tmp_path, monkeypatch):
    from experiments.benchmark import package
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("Y2024", "L02", ""))

    def partial(self, wid):
        if settings.window(wid).id == "W3":
            raise package.PackageError("escalda: window W3 needs the shard years [2025]")
        return list(settings.window(wid).years)

    original = package.V2Package.window_years
    monkeypatch.setattr(package.V2Package, "window_years", partial)
    c = Q.run(_opts(zsite, root, rows))
    assert c["unavailable"] == 1 and c["ran"] == 1
    led = Q.read_ledger(root, I.norm_row(rows[0])["unit_id"])
    assert led["status"] == "unavailable" and "2025" in led["reason"]
    _write_plan(root, rows)
    assert Q.write_status(root)["lag"]["counts"] == {"unavailable": 1, "done": 1}
    monkeypatch.setattr(package.V2Package, "window_years", original)   # the year arrives
    c = Q.run(_opts(zsite, root, rows))
    assert c["ran"] == 1 and c["done"] == 1
    assert _status(root, rows[0], "L02")["status"] == "ok"


def test_methods_without_a_lag_adapter_are_unavailable(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = [Q.make_row("base", "escalda", "W3", P0, "E5")] + _rows(("W3", "L02", ""))
    c = Q.run(_opts(zsite, root, rows))
    assert c["unavailable"] == 1 and c["ran"] == 1
    led = Q.read_ledger(root, rows[0]["unit_id"])
    assert "no lag adapter" in led["reason"]


def test_stop_files(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    rows = _rows(("W3", "L02", ""))
    open(os.path.join(root, "STOP-7"), "w").close()
    c = Q.run(_opts(zsite, root, rows, worker_id="7"))
    assert c == {"stopped": 1} and CALLS == []
    c = Q.run(_opts(zsite, root, rows, worker_id="8"))       # another worker's STOP
    assert c["ran"] == 1
    open(os.path.join(root, "STOP"), "w").close()
    assert Q.run(_opts(zsite, root, _rows(("Y2024", "L02", "")), worker_id="8")) == \
        {"stopped": 1}


# ─────────────────────────────────────────────────────────────────────────────
#  RAM ledger and claims
# ─────────────────────────────────────────────────────────────────────────────

def test_ram_ledger_reserve_refuse_release(tmp_path):
    root = str(tmp_path)
    a = Q.ram_reserve(root, 3.0, "a", total_gb=10.0, margin_gb=6.0)
    assert a is not None
    assert Q.ram_reserve(root, 2.0, "b", total_gb=10.0, margin_gb=6.0) is None
    Q.ram_release(root, a)
    b = Q.ram_reserve(root, 2.0, "b", total_gb=10.0, margin_gb=6.0)
    assert b is not None and list(Q.ram_status(root)) == [b]
    # an entry of a dead process on this host is dropped
    path, _ = Q._ram_paths(root)
    doc = product.read_json(path)
    doc["entries"]["dead"] = {"gb": 50.0, "host": socket.gethostname(), "pid": 2 ** 22 + 12345}
    product.write_json(path, doc)
    assert "dead" not in Q.ram_status(root)
    c = Q.ram_reserve(root, 1.0, "c", total_gb=10.0, margin_gb=6.0)
    assert c is not None
    Q.ram_release(root, b)
    Q.ram_release(root, c)
    assert Q.ram_status(root) == {}


def test_the_queue_respects_the_ram_ledger(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""))
    c = Q.run(_opts(zsite, root, rows, ram_total_gb=6.5))      # 1 GB x 1.25 + 6 > 6.5
    assert c == {"ram_refused": 1} and CALLS == []
    assert Q.ram_status(root) == {}
    c = Q.run(_opts(zsite, root, rows, ram_total_gb=7.5))
    assert c["ran"] == 1 and Q.ram_status(root) == {}         # released after the instance


def test_the_queue_builds_the_packages_on_its_grid(tmp_path, monkeypatch):
    """QueueOptions.res reaches package.v2_site (20: the local legacy 20 m
    packages of ``run --res 20``; 10 by default)."""
    from experiments.benchmark import package
    seen = []

    def v2_site(site, root=None, cubes_root=None, res=10, **kw):
        seen.append((site, root, cubes_root, res))
        return types.SimpleNamespace(site=site)

    monkeypatch.setattr(package, "v2_site", v2_site)
    monkeypatch.setattr(package, "V2Package",
                        lambda s2: types.SimpleNamespace(site2=s2, close=lambda: None))
    root = str(tmp_path)
    s2, pkg, err = Q._State(Q.QueueOptions(root=root, res=20)).package("escalda")
    assert err is None and pkg.site2 is s2 and seen == [("escalda", root, None, 20)]
    Q._State(Q.QueueOptions(root=root, cubes_root="cubes")).package("wadden")
    assert seen[-1] == ("wadden", root, "cubes", 10)


def _claim_file(path, host, pid, token, age_s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": runner.CLAIM_FORMAT, "host": host, "pid": pid, "token": token,
                   "start_utc": "old", "create_time": None}, f)
    old = time.time() - age_s
    os.utime(path, (old, old))


def test_claim_stealing_respects_staleness(zsite, reg, tmp_path):
    """Another host's claim (liveness unknown) is stolen only with
    --steal-stale after 30 min without a heartbeat; an alive claimant on this
    host is never stolen, however old its heartbeat."""
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""))
    cpath = Q.claim_path(root, "escalda", I.instance_id(I.instance_key(I.norm_row(rows[0]))))
    _claim_file(cpath, "another-host", 4242, "x", 3600)
    c = Q.run(_opts(zsite, root, rows))
    assert c == {"claimed": 1} and CALLS == []                 # no --steal-stale
    c = Q.run(_opts(zsite, root, rows, steal_stale=True, stale_after_s=60))
    assert c["ran"] == 1 and not os.path.exists(cpath)
    rows2 = _rows(("Y2024", "L02", ""))
    cpath2 = Q.claim_path(root, "escalda", I.instance_id(I.instance_key(I.norm_row(rows2[0]))))
    _claim_file(cpath2, socket.gethostname(), os.getpid(), "y", 3600)
    c = Q.run(_opts(zsite, root, rows2, steal_stale=True, stale_after_s=60))
    assert c == {"claimed": 1}
    _write_plan(root, rows2)
    assert Q.write_status(root)["lag"]["counts"] == {"running": 1}
    os.remove(cpath2)


def test_a_claim_of_a_dead_process_on_this_host_is_taken_over_at_once(zsite, reg, tmp_path):
    """SIGKILL, the OOM killer or a reboot leave a claim whose pid is gone:
    it is taken over at once, without --steal-stale and without waiting for
    the 30-min heartbeat age (status reads it 'stale' meanwhile)."""
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""))
    cpath = Q.claim_path(root, "escalda", I.instance_id(I.instance_key(I.norm_row(rows[0]))))
    _claim_file(cpath, socket.gethostname(), 2 ** 22 + 54321, "x", 5)     # fresh heartbeat
    _write_plan(root, rows)
    st = Q.write_status(root)["lag"]
    assert st["counts"] == {"stale": 1}
    assert "process is gone" in open(st["path"], encoding="utf-8").read()
    c = Q.run(_opts(zsite, root, rows))
    assert c["ran"] == 1 and c["status:ok"] == 1 and not os.path.exists(cpath)
    from experiments.benchmark import timing
    _claim_file(cpath, socket.gethostname(), 2 ** 22 + 54321, "z", 5)
    assert timing.active_claims(root) == []                    # a dead claim is not 'active'
    os.remove(cpath)


def test_a_unit_claimed_by_another_live_worker_is_skipped(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L12", ""))
    iid = I.instance_id(I.instance_key(I.norm_row(rows[0])))
    upath = Q.unit_claim_path(root, "escalda", iid, I.norm_row(rows[0])["unit_id"])
    _claim_file(upath, socket.gethostname(), os.getpid(), "other-worker", 1)
    c = Q.run(_opts(zsite, root, rows))
    assert c["claimed"] == 1 and c["ran"] == 1 and [x[0] for x in CALLS] == ["L12"]
    os.remove(upath)
    c = Q.run(_opts(zsite, root, rows))
    assert c.get("ran") == 1 and c.get("done") == 1


def test_a_methods_worker_never_blocks_the_instance(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L12", ""))
    iid = I.instance_id(I.instance_key(I.norm_row(rows[0])))
    assert Q.claim_path(root, "escalda", iid, ("L12",)) != Q.claim_path(root, "escalda", iid)
    assert Q.claim_path(root, "escalda", iid, ("L12", "L02")) == \
        Q.claim_path(root, "escalda", iid, ("L02", "L12"))
    # a live --methods L12 worker holds its own instance claim: the plain worker runs
    mpath = Q.claim_path(root, "escalda", iid, ("L12",))
    _claim_file(mpath, socket.gethostname(), os.getpid(), "m-worker", 1)
    c = Q.run(_opts(zsite, root, rows))
    assert c["ran"] == 2
    assert len(Q.instance_claims(root, "escalda", iid)) == 1
    os.remove(mpath)


# ─────────────────────────────────────────────────────────────────────────────
#  keys, hangs and the wall cap
# ─────────────────────────────────────────────────────────────────────────────

def test_adding_a_dummy_adapter_leaves_every_other_inputs_key_unchanged(zsite, reg, tmp_path,
                                                                        monkeypatch):
    rows = [I.norm_row(r) for r in _rows(("W3", "L02", ""), ("W3", "L12", ""),
                                         ("W3", "E0", "boot30"))]
    inst = I.build_instance(zsite.site2, rows[0], pkg=zsite.pkg)
    before = [Q.unit_keys(inst, r, Q.unit_opts(r))["inputs_key"] for r in rows]
    new = _mod("dq_new", tmp_path, run_band, products=("L05",))
    monkeypatch.setitem(sys.modules, "dq_new", new)
    monkeypatch.setitem(lags.REGISTRY, "L05", "dq_new:run")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "L05", "L05")
    after = [Q.unit_keys(inst, r, Q.unit_opts(r))["inputs_key"] for r in rows]
    assert before == after
    # the keys separate what they must
    k02 = Q.unit_keys(inst, rows[0], Q.unit_opts(rows[0]))
    assert k02["inputs_key"] != Q.unit_keys(inst, rows[0], Q.unit_opts(rows[0], threads=1))[
        "inputs_key"]
    reg["L02"].__file__ = str(tmp_path / "other.py")
    (tmp_path / "other.py").write_text("# changed code\n", encoding="utf-8")
    assert Q.unit_keys(inst, rows[0], Q.unit_opts(rows[0]))["code_key"] != k02["code_key"]
    assert Q.unit_keys(inst, rows[1], Q.unit_opts(rows[1]))["inputs_key"] == before[1]


def test_external_inputs_enter_the_inputs_key(zsite, reg, tmp_path):
    r = I.norm_row(_rows(("W3", "L02", ""))[0])
    inst = I.build_instance(zsite.site2, r, pkg=zsite.pkg)
    k0 = Q.unit_keys(inst, r, Q.unit_opts(r))
    reg["L02"].external_inputs = lambda inst, unit: {"cmems": "abc"}
    k1 = Q.unit_keys(inst, r, Q.unit_opts(r))
    assert k1["inputs_key"] != k0["inputs_key"] and k1["external_inputs"] == {"cmems": "abc"}
    assert k1["config_key"] == k0["config_key"] and k1["code_key"] == k0["code_key"]


def test_a_flagged_unit_that_returns_keeps_its_result(zsite, reg, tmp_path):
    """Declared amendment of §13.5: hung / budget are flags of a unit that
    returns (its result is final with the adapter's status); the flags are in
    run.json 'monitor' and every product's diag.monitor, and status says so."""
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L04", ""))
    c = Q.run(_opts(zsite, root, rows, progress_timeout_s=0.15, poll_s=0.02))
    assert c["status:ok"] == 1 and "status:hung" not in c
    for p in ("L04w", "L04i"):
        r = _status(root, rows[0], p)
        assert r["status"] == "ok" and r["monitor"]["hung_detected"] is True
        assert r["diag"]["monitor"]["hung_detected"] is True
    led = Q.read_ledger(root, I.norm_row(rows[0])["unit_id"])
    assert led["status"] == "ok" and led["monitor"]["hung_detected"] is True
    _write_plan(root, rows)
    st = Q.write_status(root)["lag"]
    assert st["counts"] == {"done": 1}
    assert "hang detected while it ran" in open(st["path"], encoding="utf-8").read()
    assert Q.run(_opts(zsite, root, rows)) == {"done": 1}      # final: never recomputed
    root2 = str(tmp_path / "root2")
    c = Q.run(_opts(zsite, root2, rows, wall_cap_floor_s=0.1, poll_s=0.02))
    assert c["status:ok"] == 1
    r = _status(root2, rows[0], "L04w")
    assert r["status"] == "ok" and r["monitor"]["wall_capped"] is True


def test_run_json_records_the_code_files_key_and_the_anonymous_memory(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("Y2024", "L02", ""))
    Q.run(_opts(zsite, root, rows))
    a, b = _status(root, rows[0], "L02"), _status(root, rows[1], "L02")
    ck = Q.code_files_key(lags.module_attrs("L02"))
    # window- and cell-free: the control's products and the real rows share it
    assert a["keys"]["code_files_key"] == b["keys"]["code_files_key"] == ck
    assert "peak_anon_mb" in a["memory"] and a["memory"]["tiles_unmapped"] is True
    assert "peak_anon_mb" in a["unit_total"]


def test_unit_done_needs_a_readable_product_npz(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""))
    assert Q.run(_opts(zsite, root, rows))["ran"] == 1
    d = Q.product_dir(root, Q.unit_of(I.norm_row(rows[0])), "L02")
    with open(os.path.join(d, "product.npz"), "r+b") as f:     # a power loss
        f.truncate(10)
    CALLS.clear()
    c = Q.run(_opts(zsite, root, rows))
    assert c["ran"] == 1 and [x[0] for x in CALLS] == ["L02"]
    assert Q.npz_readable(os.path.join(d, "product.npz"), _status(root, rows[0], "L02")["arrays"])


def test_a_publish_failure_is_an_error_and_the_worker_goes_on(zsite, reg, tmp_path,
                                                               monkeypatch):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""), ("W3", "L12", ""))
    real = Q.publish

    def full_disk(out_dir, arrays, run, log_text=None):
        if run["product"] == "L02":
            raise OSError(28, "No space left on device")
        return real(out_dir, arrays, run, log_text)

    monkeypatch.setattr(Q, "publish", full_disk)
    c = Q.run(_opts(zsite, root, rows))
    assert c["status:error"] == 1 and c["status:ok"] == 1
    led = Q.read_ledger(root, I.norm_row(rows[0])["unit_id"])
    assert led["status"] == "error" and led["reason"].startswith("publish: OSError")
    assert _status(root, rows[1], "L12")["status"] == "ok"


def test_leftovers_of_dead_workers_are_moved_never_deleted(tmp_path):
    root = str(tmp_path)
    host = socket.gethostname()
    dead = os.path.join(root, "runs", "ems", "W3", "x", "L02", "rand-A",
                        f"Nall_r00.tmp-{host}-{2 ** 22 + 99}-123")
    mine = os.path.join(root, "runs", "ems", "W3", "x", "L03", "rand-A",
                        f"Nall_r00.tmp-{host}-{os.getpid()}-1")
    other = os.path.join(root, "runs", "ems", "W3", "x", "L04", "rand-A",
                         "Nall_r00.tmp-another-host-77")
    for d in (dead, mine, other):
        os.makedirs(d)
        open(os.path.join(d, "run.json"), "w").close()
    moved = Q.recover_tmp_folders(root)
    assert len(moved) == 1 and moved[0].startswith(dead.replace("\\", "/") + ".old-")
    assert os.path.exists(os.path.join(moved[0], "run.json")) and not os.path.exists(dead)
    assert os.path.isdir(mine) and os.path.isdir(other)


def test_one_blas_thread_count_per_queue_root(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", ""))
    _write_plan(root, rows)
    Q.run(_opts(zsite, root, plans=("lag",), threads=4))
    assert product.read_json(Q.queue_json_path(root))["threads"] == 4
    with pytest.raises(Q.QueueRefused, match="--threads 4"):
        Q.run(_opts(zsite, root, plans=("lag",), threads=2))
    Q.run(_opts(zsite, root, plans=("lag",), threads=2, threads_change=True))
    doc = product.read_json(Q.queue_json_path(root))
    assert doc["threads"] == 2 and doc["history"][0]["threads"] == 4
    Q.run(_opts(zsite, root, rows=rows, threads=1))             # in-memory queues: exempt


def test_a_changed_package_is_reopened_and_a_failed_one_retried(zsite, tmp_path):
    from experiments.benchmark import package
    opts = _opts(zsite, str(tmp_path / "root"))
    st = Q._State(opts)
    s2, pkg, err = st.package("escalda")
    assert pkg is not None and err is None
    assert st.package("escalda")[1] is pkg                     # cached
    rp = os.path.join(s2.dir, "record.json")
    t = os.stat(rp).st_mtime_ns
    os.utime(rp, ns=(t + 10 ** 9, t + 10 ** 9))
    try:
        assert st.package("escalda")[1] is not pkg             # re-opened
    finally:
        os.utime(rp, ns=(t, t))
    st.close()
    stamps = Q.package_stamps(s2)
    assert stamps["record.json"] is not None and "levels.npz" in stamps
    # a site whose package could not be opened is retried when its files change
    bad = types.SimpleNamespace(site="ems", dir=str(tmp_path / "ems_pkg"))
    calls = []

    def fake_pkg(s):
        calls.append(s)
        raise package.PackageError("no record yet")

    st2 = Q._State(Q.QueueOptions(root=str(tmp_path), site2_for=lambda s: bad))
    real = package.V2Package
    package.V2Package = fake_pkg
    try:
        assert st2.package("ems")[1] is None and st2.package("ems")[1] is None
        assert len(calls) == 1
        os.makedirs(bad.dir)
        open(os.path.join(bad.dir, "record.json"), "w").close()
        assert st2.package("ems")[1] is None and len(calls) == 2
    finally:
        package.V2Package = real


def test_status_file_keeps_every_site(zsite, reg, tmp_path):
    root = str(tmp_path / "root")
    rows = _rows(("W3", "L02", "")) + [Q.make_row("lag", "ems", "W3", P0, "L02")]
    _write_plan(root, rows)
    st = Q.write_status(root, sites=["escalda"])["lag"]
    text = open(st["path"], encoding="utf-8").read()
    assert ",escalda," in text and ",ems," in text
    assert sum(st["counts"].values()) == 1
