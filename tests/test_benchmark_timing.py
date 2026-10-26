# -*- coding: utf-8 -*-
"""Validation v2: the stage stopwatch and the dedicated timing runs
(``experiments/benchmark/timing.py``; spec §8, §13.5)."""
import json
import os
import sys
import time
import types

import numpy as np
import pytest

from experiments.benchmark import lags, runner, timing


def _burn(s):
    t0 = time.perf_counter()
    x = 0.0
    while time.perf_counter() - t0 < s:
        x += float(np.sum(np.arange(2000.0)))
    return x


def test_stage_fields_are_present_and_non_negative():
    sw = timing.Stopwatch("unit")
    with sw.running():
        with sw.stage("io.read"):
            _burn(0.05)
            sw.progress(3)
        with sw.stage("fit.m2a"):
            _burn(0.05)
    rec = sw.records()
    assert set(rec) == {"io.read", "fit.m2a"}
    for r in rec.values():
        for k in timing.FIELDS:
            assert k in r
        for k in ("wall_s", "cpu_self_s", "cpu_children_s", "peak_rss_mb"):
            assert r[k] >= 0
        assert np.isnan(r["read_mb"]) or r["read_mb"] >= 0
        assert r["calls"] == 1
    assert rec["io.read"]["progress"] == 3 and rec["fit.m2a"]["progress"] == 0
    assert rec["io.read"]["wall_s"] >= 0.05
    tot = sw.total()
    assert tot["wall_s"] >= rec["io.read"]["wall_s"] + rec["fit.m2a"]["wall_s"]
    assert tot["progress"] == 3 and tot["peak_rss_mb"] > 0


def test_nested_stages_sum_and_repeated_stages_accumulate():
    sw = timing.Stopwatch("unit")
    with sw.running():
        with sw.stage("lag.L01.outer"):
            for _ in range(2):
                with sw.stage("lag.L01.inner"):
                    _burn(0.03)
            _burn(0.02)
    rec = sw.records()
    assert rec["lag.L01.inner"]["calls"] == 2
    assert rec["lag.L01.outer"]["wall_s"] >= rec["lag.L01.inner"]["wall_s"]
    assert rec["lag.L01.inner"]["wall_s"] >= 0.06
    assert rec["lag.L01.outer"]["cpu_self_s"] >= 0


def test_the_monitor_flags_a_stalled_progress():
    seen = []
    sw = timing.Stopwatch("stalled", progress_timeout_s=0.2, poll_s=0.02,
                          on_hung=lambda s: seen.append("hung"))
    with sw.running():
        time.sleep(0.6)
    assert sw.hung and seen == ["hung"]
    ok = timing.Stopwatch("moving", progress_timeout_s=0.25, poll_s=0.02)
    with ok.running():
        for _ in range(15):
            time.sleep(0.04)
            ok.progress()
    assert not ok.hung and ok.n_progress == 15


def test_the_monitor_flags_the_wall_cap():
    seen = []
    sw = timing.Stopwatch("long", wall_cap_s=0.1, poll_s=0.02,
                          on_wall_cap=lambda s: seen.append("cap"))
    with sw.running():
        for _ in range(10):
            time.sleep(0.03)
            sw.progress()
    assert sw.wall_capped and not sw.hung and seen == ["cap"]
    assert sw.flags() == {"hung": False, "wall_capped": True, "progress": 10}


def test_stage_entry_counts_as_progress():
    sw = timing.Stopwatch("stages", progress_timeout_s=0.2, poll_s=0.02)
    with sw.running():
        for _ in range(8):
            with sw.stage("io.x"):
                time.sleep(0.05)
    assert not sw.hung


def test_quiet_check_refuses_with_an_active_claim(tmp_path):
    root = str(tmp_path)
    ok, why, ctx = timing.check_quiet(root, max_load=1e9)
    assert ok and ctx["active_claims"] == 0
    p = os.path.join(root, "claims", "ems", "abc.claim")
    assert runner.claim(p, {"x": 1})
    try:
        ok, why, ctx = timing.check_quiet(root, max_load=1e9)
        assert not ok and "active claim" in why
    finally:
        runner.release(p)


class _Inst(types.SimpleNamespace):
    pass


def test_workload_rows_name_their_curve_set():
    """The N workloads are the K1 curve subsets (rep 0) at W3: naming the
    design fixes R (several W3 curve sets hold N = 45 and 180 with different
    R, and a design-less row is refused by instance.curve_R)."""
    from experiments.benchmark import instance as inst_mod
    for N in (45, 180, "all"):
        row = timing._workload_row("escalda", N)
        assert row["design"] == timing.WORKLOAD_DESIGN == "curves-k1"
        assert row["window"] == timing.WORKLOAD_WINDOW == "W3" and row["rep"] == 0
        if N != "all":
            assert inst_mod.curve_R(row["design"], "W3", N, "A") in (2, 5)
    with pytest.raises(ValueError):                 # why the design must be named
        inst_mod.curve_R("timing", "W3", 45, "A")


def test_run_dedicated_needs_the_w3_years_and_records_unbuilt_workloads(tmp_path,
                                                                        monkeypatch):
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark import package
    root = str(tmp_path)
    site2 = types.SimpleNamespace(site="escalda", root=root)

    def partial(s2):                                # a site holding only 2023
        def window_years(w):
            raise package.PackageError(f"escalda: window {w} needs the shard years [2024, 2025]")
        return types.SimpleNamespace(years=[2023], window_years=window_years)

    monkeypatch.setattr(package, "V2Package", partial)
    with pytest.raises(timing.TimingUnavailable, match=r"\[2023\]"):
        timing.run_dedicated(root, "escalda", ["L17"], site2=site2, max_load=1e9)

    def build(site2, row, pkg=None, provider=None, stopwatch=None):
        if row["N"] == 100:
            raise ValueError("no rand curve set holds N = 100 at W3 (design 'curves-k1')")
        return _Inst(domain=types.SimpleNamespace(n_keep=10), sel=np.arange(5), cache={},
                     stage=None, progress=None)

    mod = types.ModuleType("fake_timing_adapter2")
    mod.run = lambda inst, unit, opts: []
    monkeypatch.setitem(sys.modules, "fake_timing_adapter2", mod)
    monkeypatch.setitem(lags.REGISTRY, "LT2", "fake_timing_adapter2:run")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LT2x", "LT2")
    monkeypatch.setattr(inst_mod, "build_instance", build)
    monkeypatch.setattr(package, "V2Package", lambda s2: types.SimpleNamespace(
        years=[2023, 2024, 2025], window_years=lambda w: [2023, 2024, 2025]))
    out = timing.run_dedicated(root, "escalda", ["LT2x"], workloads=(100, "all"), repeats=1,
                               site2=site2, max_load=1e9, write=False)
    w = out["LT2x"]["workloads"]
    assert "N = 100" in w["100"]["error"] and "configs" in w["all"]


def test_run_dedicated_writes_the_protocol(tmp_path, monkeypatch):
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark import package
    root = str(tmp_path)
    calls = []

    def fake_build(site2, row, pkg=None, provider=None, stopwatch=None):
        n = {45: 10, 180: 20, "all": 30}[row["N"]]
        return _Inst(domain=types.SimpleNamespace(n_keep=n * 1000), sel=np.arange(n),
                     cache={}, stage=None, progress=None)

    mod = types.ModuleType("fake_timing_adapter")

    def run(inst, unit, opts):
        calls.append((unit["N"], opts["threads"]))
        with inst.stage("fit"):
            inst.progress(1)
            _burn(0.005)
        return []

    mod.run = run
    monkeypatch.setitem(sys.modules, "fake_timing_adapter", mod)
    monkeypatch.setitem(lags.REGISTRY, "LT1", "fake_timing_adapter:run")
    monkeypatch.setitem(lags.PRODUCT_METHOD, "LT1x", "LT1")
    monkeypatch.setattr(inst_mod, "build_instance", fake_build)
    monkeypatch.setattr(package, "V2Package", lambda s2: object())
    site2 = types.SimpleNamespace(site="escalda", root=root)
    out = timing.run_dedicated(root, "escalda", ["LT1x"], workloads=(45, "all"), repeats=2,
                               threads=3, site2=site2, max_load=1e9)
    doc = json.load(open(out["LT1x"], encoding="utf-8"))
    assert doc["label"] == "dedicated" and doc["format"] == timing.TIMING_FORMAT
    assert set(doc["workloads"]) == {"45", "all"}
    w = doc["workloads"]["45"]
    assert set(w["configs"]) == {"single", "production"}
    runs = w["configs"]["single"]["runs"]
    assert len(runs) == 3 and runs[0]["warmup"] and not runs[1]["warmup"]
    assert w["configs"]["production"]["threads"] == 3 and w["configs"]["single"]["threads"] == 1
    assert w["configs"]["single"]["summary"]["wall_s"]["median"] is not None
    assert "lag.LT1.fit" in runs[1]["stages"]
    assert doc["workloads"]["all"]["n_keep"] == 30000
    assert len(calls) == 2 * 2 * 3
    p = os.path.join(root, "claims", "ems", "busy.claim")
    assert runner.claim(p)
    try:
        with pytest.raises(timing.TimingRefused):
            timing.run_dedicated(root, "escalda", ["LT1x"], site2=site2, max_load=1e9)
    finally:
        runner.release(p)


def test_run_dedicated_records_a_failing_method_and_keeps_the_others(tmp_path, monkeypatch):
    """A method that raises is recorded as 'error (...)' and its JSON is
    still written; the other methods are timed (the site is not aborted)."""
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark import package
    root = str(tmp_path)

    def fake_build(site2, row, pkg=None, provider=None, stopwatch=None):
        return _Inst(domain=types.SimpleNamespace(n_keep=10), sel=np.arange(5), cache={},
                     stage=None, progress=None)

    good = types.ModuleType("fake_timing_good")
    good.run = lambda inst, unit, opts: []
    bad = types.ModuleType("fake_timing_bad")

    def boom(inst, unit, opts):
        raise RuntimeError("adapter bug")

    bad.run = boom
    for name, mod, meth, pid in (("fake_timing_good", good, "LT5", "LT5x"),
                                 ("fake_timing_bad", bad, "LT6", "LT6x")):
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setitem(lags.REGISTRY, meth, f"{name}:run")
        monkeypatch.setitem(lags.PRODUCT_METHOD, pid, meth)
    monkeypatch.setattr(inst_mod, "build_instance", fake_build)
    monkeypatch.setattr(package, "V2Package", lambda s2: object())
    site2 = types.SimpleNamespace(site="escalda", root=root)
    out = timing.run_dedicated(root, "escalda", ["LT6x", "LT5x"], workloads=(45,), repeats=2,
                               site2=site2, max_load=1e9)
    d6 = json.load(open(out["LT6x"], encoding="utf-8"))
    runs = d6["workloads"]["45"]["configs"]["single"]["runs"]
    assert runs[0]["status"].startswith("error (RuntimeError: adapter bug")
    assert len(runs) == 1                                   # the same error is not repeated
    d5 = json.load(open(out["LT5x"], encoding="utf-8"))
    assert d5["complete"] and d5["workloads"]["45"]["configs"]["single"]["runs"][1]["status"] == "ok"
    assert "peak_anon_mb" in d5["workloads"]["45"]["configs"]["single"]["summary"]


def test_dead_claims_do_not_make_the_machine_busy(tmp_path):
    import socket
    root = str(tmp_path)
    p = os.path.join(root, "claims", "ems", "abc", "unit.claim")
    os.makedirs(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"format": runner.CLAIM_FORMAT, "host": socket.gethostname(),
                   "pid": 2 ** 22 + 4321, "token": "t"}, f)
    ok, why, ctx = timing.check_quiet(root, max_load=1e9)
    assert ok and ctx["active_claims"] == 0
    q = os.path.join(root, "claims", "ems", "live", "unit.claim")
    assert runner.claim(q, {"x": 1})                        # unit claims count (recursive)
    try:
        assert not timing.check_quiet(root, max_load=1e9)[0]
    finally:
        runner.release(q)


def test_anonymous_memory_probe():
    v = timing.current_anon_mb()
    assert np.isnan(v) or v > 0
