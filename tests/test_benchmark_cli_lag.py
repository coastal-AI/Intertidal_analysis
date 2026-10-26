# -*- coding: utf-8 -*-
"""Validation v2, milestone V2: the CLI commands of the lag benchmark
(``experiments/validation_v2.py``: run, status, score, gate, control, pilot,
timing, lock). Every command is dispatched to a monkeypatched function; the
other packages' modules are replaced by stand-ins in ``sys.modules``."""
import json
import os
import sys
import types

import numpy as np
import pytest

from experiments import validation_v2 as cli
from experiments.benchmark import lock as lock_mod
from experiments.benchmark import pilots, queue_v2, timing


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = str(tmp_path / "v2").replace("\\", "/")
    monkeypatch.setenv("VALIDATION_V2_ROOT", r)
    return r


def _fake(monkeypatch, name, **attrs):
    m = types.ModuleType(f"experiments.benchmark.{name}")
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, f"experiments.benchmark.{name}", m)
    return m


def _log(root):
    p = os.path.join(root, "logs", "commands.jsonl")
    return [json.loads(x) for x in open(p, encoding="utf-8").read().splitlines()]


# ─────────────────────────────────────────────────────────────────────────────
#  run and status
# ─────────────────────────────────────────────────────────────────────────────

def test_run_dispatches_the_queue_options(root, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(queue_v2, "run", lambda opts: seen.setdefault("opts", opts) and {
        "ran": 2, "status:ok": 2})
    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    assert cli.main(["run", "--plans", "lag", "--worker-id", "3"]) == 2   # no plan file
    assert "plan --design lag" in capsys.readouterr().out
    os.makedirs(os.path.join(root, "plans"))
    for d in ("lag", "base"):
        open(os.path.join(root, "plans", f"{d}.csv"), "w").close()
    rc = cli.main(["run", "--plans", "lag,base", "--worker-id", "3", "--threads", "1",
                   "--steal-stale", "--sites", "ems,wadden", "--windows", "W3,Y2023",
                   "--smoke", "--methods", "E0,L08"])
    assert rc == 0
    o = seen["opts"]
    assert o.plans == ("lag", "base") and o.worker_id == "3" and o.threads == 1
    assert o.steal_stale and o.smoke and o.sites == ("ems", "wadden")
    assert o.windows == ("W3", "Y2023") and o.methods == ("E0", "L08") and o.root == root
    assert o.wall_cap is True and o.passes == 1
    assert _log(root)[-1]["command"] == "run" and _log(root)[-1]["exit"] == 0


def test_run_exit_code_reports_unit_errors(root, monkeypatch):
    monkeypatch.setattr(queue_v2, "run", lambda opts: {"ran": 1, "status:error": 1})
    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    os.makedirs(os.path.join(root, "plans"))
    open(os.path.join(root, "plans", "lag.csv"), "w").close()
    assert cli.main(["run", "--plans", "lag", "--no-wall-cap"]) == 1
    # an instance-level error (domain, keys) fails every unit before it runs
    monkeypatch.setattr(queue_v2, "run", lambda opts: {"error": 3, "unavailable": 2})
    assert cli.main(["run", "--plans", "lag"]) == 1
    # unavailable units (a partial-year site) are not errors
    monkeypatch.setattr(queue_v2, "run", lambda opts: {"unavailable": 5, "done": 1})
    assert cli.main(["run", "--plans", "lag"]) == 0


def test_run_score_and_plan_take_the_grid_resolution(root, monkeypatch, capsys):
    """--res 20 (the local legacy 20 m packages): the queue options and the
    scorer get it; a site without a 20 m cube is refused (exit 2); 10 is the
    default and is not passed to the scorer."""
    seen = {}
    monkeypatch.setattr(queue_v2, "run", lambda opts: seen.setdefault("opts", opts) and {})
    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    os.makedirs(os.path.join(root, "plans"))
    open(os.path.join(root, "plans", "lag.csv"), "w").close()
    assert cli.main(["run", "--plans", "lag", "--res", "20", "--sites", "escalda"]) == 0
    assert seen["opts"].res == 20 and seen["opts"].sites == ("escalda",)
    seen.clear()
    assert cli.main(["run", "--plans", "lag"]) == 0 and seen["opts"].res == 10
    assert cli.main(["run", "--plans", "lag", "--res", "20", "--sites", "ferrol"]) == 2
    assert "no 20 m input" in capsys.readouterr().out
    calls = []
    _fake(monkeypatch, "scoring", score_lag=lambda **kw: calls.append(kw) or {"warnings": []})
    assert cli.main(["score", "--part", "lag", "--pre-lock", "--res", "20"]) == 0
    assert calls[-1]["res"] == 20 and calls[-1]["root"] == root
    assert cli.main(["score", "--part", "lag", "--pre-lock"]) == 0 and "res" not in calls[-1]
    assert cli.build_parser().parse_args(["plan", "--design", "lag", "--res", "20"]).res == 20


def test_status_prints_the_run_state_of_every_plan(root, monkeypatch, capsys):
    monkeypatch.setattr(queue_v2, "write_status", lambda r, designs=None, sites=None: {
        "lag": {"path": f"{r}/status/lag.csv", "counts": {"done": 3, "pending": 2},
                "remaining_est_hours": 0.5}})
    assert cli.main(["status", "--sites", "ems"]) == 0
    out = capsys.readouterr().out
    assert "steps done" in out and "[status] lag: done 3, pending 2" in out
    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    assert cli.main(["status", "--sites", "ems"]) == 0
    assert "no plans under" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
#  score
# ─────────────────────────────────────────────────────────────────────────────

def test_score_refuses_without_lock_unless_pre_lock(root, monkeypatch, capsys):
    calls = []
    _fake(monkeypatch, "scoring",
          score_lag=lambda **kw: calls.append(kw) or {"table_md": "t.md", "warnings": ["w1"]})
    assert cli.main(["score", "--part", "lag"]) == 2
    assert "refused" in capsys.readouterr().out and calls == []
    assert cli.main(["score", "--part", "lag", "--pre-lock", "--sites", "ems",
                     "--windows", "Y2023"]) == 0
    assert calls[-1]["pre_lock"] is True and calls[-1]["sites"] == ["ems"]
    assert calls[-1]["windows"] == ["Y2023"] and calls[-1]["root"] == root
    out = capsys.readouterr().out
    assert "[score] table_md: t.md" in out and "WARNING: w1" in out
    product_doc = {"format": lock_mod.LOCK_FORMAT, "phases": {"A": {"utc": "x"}}}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "LOCK.json"), "w", encoding="utf-8") as f:
        json.dump(product_doc, f)
    assert cli.main(["score", "--part", "lag"]) == 0
    assert calls[-1]["pre_lock"] is False
    assert "show_unvalidated" not in calls[-1]


def test_score_show_unvalidated_is_passed_to_the_scorer(root, monkeypatch):
    calls = []
    _fake(monkeypatch, "scoring",
          score_lag=lambda **kw: calls.append(kw) or {"table_md": "t.md", "warnings": []})
    assert cli.main(["score", "--part", "lag", "--pre-lock", "--show-unvalidated"]) == 0
    assert calls[-1]["show_unvalidated"] is True and calls[-1]["pre_lock"] is True


def test_site2_defaults_to_the_years_of_the_existing_record(root, monkeypatch):
    """A partial-year site (prepare ems --years 2023): every later command
    (truth, status, pilot, prepare without --years) gets the record's years,
    so package.prepare_v2 never reads the default 2023-2025 as a request for
    years the record does not hold."""
    p = cli.build_parser()
    s2 = cli._site2("ems", p.parse_args(["truth", "ems"]))
    assert s2.years == (2023, 2024, 2025)                       # no record yet
    rec = {"years": [2023], "years_requested": [2023]}
    os.makedirs(os.path.join(root, "common", "ems"))
    with open(os.path.join(root, "common", "ems", "record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert cli._site2("ems", p.parse_args(["truth", "ems"])).years == (2023,)
    assert cli._site2("ems", p.parse_args(["status", "--sites", "ems"])).years == (2023,)
    assert cli._site2("ems", p.parse_args(["prepare", "ems"])).years == (2023,)
    # an explicit --years always wins (extending a record: --force --step record)
    assert cli._site2("ems", p.parse_args(["prepare", "ems", "--years", "2023-2025"])
                      ).years == (2023, 2024, 2025)
    # a record built on the default years at a site whose stores lacked some
    rec = {"years": [2023], "years_requested": [2023, 2024, 2025]}
    with open(os.path.join(root, "common", "ems", "record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert cli._site2("ems", p.parse_args(["truth", "ems"])).years == (2023, 2024, 2025)


@pytest.mark.parametrize("argv", [["score", "--part", "all"], ["score", "--part", "elev"],
                                  ["score", "--part", "lag", "--blocks"]])
def test_score_parts_of_later_milestones(root, argv, capsys):
    assert cli.main(argv) == 2
    assert "available from milestone V3" in capsys.readouterr().out


def test_score_defaults_to_the_lag_part(root, monkeypatch, capsys):
    calls = []
    _fake(monkeypatch, "scoring", PRELOCK_BLIND_WINDOWS=("W3",),
          score_lag=lambda **kw: calls.append(kw) or {"warnings": []})
    assert cli.main(["score"]) == 2                        # the lag part, refused unlocked
    assert "refused: no LOCK.json phase A" in capsys.readouterr().out and calls == []
    assert cli.main(["score", "--pre-lock"]) == 0 and calls[-1]["pre_lock"] is True


def test_score_pre_lock_keeps_w3_blinded_unless_the_user_unblinds_it(root, monkeypatch,
                                                                     capsys):
    calls = []
    _fake(monkeypatch, "scoring", PRELOCK_BLIND_WINDOWS=("W3",),
          score_lag=lambda **kw: calls.append(kw) or {"warnings": [], "ledger": "L.json"})
    assert cli.main(["score", "--pre-lock", "--windows", "W3"]) == 2
    out = capsys.readouterr().out
    assert "confirmatory window" in out and "--unblind W3" in out and calls == []
    assert cli.main(["score", "--pre-lock", "--sites", "wadden"]) == 0
    assert calls[-1]["unblind"] == () and "score --pre-lock" in calls[-1]["invocation"]
    assert "[score] ledger: L.json" in capsys.readouterr().out
    assert cli.main(["score", "--pre-lock", "--unblind", "W3", "--windows", "W3"]) == 0
    assert calls[-1]["unblind"] == ("W3",) and calls[-1]["windows"] == ["W3"]
    assert "UNBLINDED before lock-A: W3" in capsys.readouterr().out
    # the 20 m local checks are exempt: no blinding, no ledger keywords
    assert cli.main(["score", "--pre-lock", "--res", "20", "--windows", "W3"]) == 0
    assert "unblind" not in calls[-1] and calls[-1]["res"] == 20


def test_run_options_and_exit_codes_of_the_server_review(root, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    os.makedirs(os.path.join(root, "plans"))
    open(os.path.join(root, "plans", "lag.csv"), "w").close()
    monkeypatch.setattr(queue_v2, "run", lambda opts: seen.setdefault("opts", opts) and {})
    assert cli.main(["run", "--plans", "lag", "--ram-margin-gb", "16", "--threads-change"]) == 0
    o = seen["opts"]
    assert o.ram_margin_gb == 16.0 and o.threads_change and o.oom_score_adj == 1000
    seen.clear()
    assert cli.main(["run", "--plans", "lag"]) == 0
    assert seen["opts"].ram_margin_gb == queue_v2.RAM_MARGIN_GB and not seen["opts"].threads_change
    for bad in ({"status:hung": 1}, {"status:budget": 1}, {"claim_lost": 1}):
        monkeypatch.setattr(queue_v2, "run", lambda opts, _b=bad: dict(_b, ran=1))
        assert cli.main(["run", "--plans", "lag"]) == 1, bad

    def refuse(opts):
        raise queue_v2.QueueRefused("this queue root runs with --threads 4")

    monkeypatch.setattr(queue_v2, "run", refuse)
    assert cli.main(["run", "--plans", "lag", "--threads", "2"]) == 2
    assert "refused: this queue root runs with --threads 4" in capsys.readouterr().out


def test_plan_sites_merge_into_the_existing_plan(root, monkeypatch):
    from experiments.benchmark import designs
    assert cli.main(["plan", "--design", "lag", "--sites", "escalda"]) == 0
    assert cli.main(["plan", "--design", "lag", "--sites", "ems"]) == 0
    rows = queue_v2.read_plan(root, "lag")
    assert {r["site"] for r in rows} == {"escalda", "ems"}
    full = [r for r in designs.plan_rows("lag", ("escalda", "ems"))]
    assert [r["unit_id"] for r in rows] == [r["unit_id"] for r in full]   # design order
    assert cli.main(["plan", "--design", "lag", "--sites", "ems", "--replace"]) == 0
    assert {r["site"] for r in queue_v2.read_plan(root, "lag")} == {"ems"}


def test_fetch_stac_meta_refuses_a_site_with_provisional_offsets(root, monkeypatch):
    from experiments.benchmark import package
    seen = {}

    def run(sites, root=None, refresh=False, years=None, guard=None):
        seen.update(sites=sites, years=years, why={s: guard(s) for s in sites})
        return 0

    _fake(monkeypatch, "fetch.stac_meta", run=run)
    os.makedirs(os.path.join(root, "common", "ems"))
    with open(os.path.join(root, "common", "ems", "record.json"), "w", encoding="utf-8") as f:
        json.dump({"years": [2023], "years_requested": [2023]}, f)
    monkeypatch.setattr(package, "_frozen_offsets_for",
                        lambda s2: (None, "provisional (no inputs report)"))
    assert cli.main(["fetch", "stac-meta", "--sites", "ems,wadden", "--years", "2023"]) == 0
    assert seen["years"] == [2023] and seen["why"]["wadden"] is None       # no record yet
    assert "provisional per-scene offsets" in seen["why"]["ems"]
    monkeypatch.setattr(package, "_frozen_offsets_for", lambda s2: ({"2023-01-02": 0}, "rep"))
    assert cli.main(["fetch", "stac-meta", "--sites", "ems"]) == 0
    assert seen["why"]["ems"] is None


def test_lock_passes_the_unblind_amendment(root, monkeypatch):
    seen = []

    def wl(phase, r, extra_dev_judges=(), allow_missing=(), **kw):
        seen.append(kw)
        return {"phases": {phase: {}}, "amendments": [{}]}

    monkeypatch.setattr(lock_mod, "write_lock", wl)
    assert cli.main(["lock", "--phase", "A", "--amend-unblind", "user saw W3"]) == 0
    assert seen[-1] == {"unblind_amendment": "user saw W3"}
    assert cli.main(["lock", "--phase", "A"]) == 0 and seen[-1] == {}


def test_timing_defaults_to_a_full_record_site():
    assert cli.build_parser().parse_args(["timing"]).sites == "wadden"


def test_a_closed_pipe_never_kills_a_command(root, monkeypatch):
    class Broken:
        def write(self, s):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(queue_v2, "write_status", lambda *a, **k: {})
    os.makedirs(os.path.join(root, "plans"))
    open(os.path.join(root, "plans", "lag.csv"), "w").close()
    monkeypatch.setattr(queue_v2, "run", lambda opts: (queue_v2.log("a unit line"),
                                                       {"ran": 1, "status:ok": 1})[1])
    monkeypatch.setattr(sys, "stdout", Broken())
    assert cli.main(["run", "--plans", "lag"]) == 0
    s = cli._PipeSafe(Broken())
    s.write("x")
    s.flush()
    assert s.broken


# ─────────────────────────────────────────────────────────────────────────────
#  gate, pilot, timing, lock
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_dispatch(root, monkeypatch, capsys):
    calls = []

    def run_gate(name, site, root=None, cubes_root=None, **kw):
        calls.append((name, site, root, cubes_root))
        return {"verdict": "FAIL" if (name, site) == ("keep", "wadden") else "PASS",
                "reason": "x"}

    _fake(monkeypatch, "gates", GATES={"record": None, "keep": None}, run_gate=run_gate)
    assert cli.main(["gate", "record", "--sites", "ems", "--cubes-root", "C"]) == 0
    assert calls == [("record", "ems", root, "C")]
    assert cli.main(["gate", "all", "--sites", "ems,wadden"]) == 1      # keep FAILs at wadden
    assert [(n, s) for n, s, _, _ in calls[1:]] == [("record", "ems"), ("keep", "ems"),
                                                    ("record", "wadden"), ("keep", "wadden")]
    assert cli.main(["gate", "nidem", "--sites", "ems"]) == 2
    assert "available from milestone V3" in capsys.readouterr().out


def test_gate_no_build_reaches_only_the_gates_that_build(root, monkeypatch):
    """--no-build: G-record/keep/marea never build the legacy package; the
    thin gates (no 'build' keyword) are called as before."""
    calls = []

    def g_build(site, root=None, cubes_root=None, *, build=True):
        return {"verdict": "NOT RUN"}

    def g_thin(site, root=None, cubes_root=None, *, years=None):
        return {"verdict": "PASS"}

    def run_gate(name, site, root=None, cubes_root=None, **kw):
        calls.append((name, kw))
        return {"verdict": "PASS", "reason": "x"}

    fns = {"record": g_build, "zarr": g_thin}
    _fake(monkeypatch, "gates", GATES=dict(fns), run_gate=run_gate,
          _resolve=lambda n: fns[n])
    assert cli.main(["gate", "all", "--sites", "ems", "--no-build"]) == 0
    assert calls == [("record", {"build": False}), ("zarr", {})]
    calls.clear()
    assert cli.main(["gate", "all", "--sites", "ems"]) == 0
    assert calls == [("record", {}), ("zarr", {})]


def test_fetch_emodnet_forwards_the_root_and_cubes_root(root, monkeypatch):
    seen = []
    _fake(monkeypatch, "fetch.emodnet",
          run=lambda sites, root=None, cubes_root=None: seen.append((sites, root, cubes_root)))
    assert cli.main(["fetch", "emodnet", "--sites", "ems,wadden", "--cubes-root", "C"]) == 0
    assert seen == [(["ems", "wadden"], root, "C")]


def test_pilot_reps_attain_and_units(root, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(pilots, "pilot_reps", lambda s2s, design, root: seen.update(
        sites=[s.site for s in s2s], design=design, root=root) or {})
    assert cli.main(["pilot", "reps", "--sites", "ems,wadden"]) == 0
    assert seen == {"sites": ["ems", "wadden"], "design": "curves-k1", "root": root}
    att = []
    _fake(monkeypatch, "inference", attain_report=lambda r, sites=None: att.append(
        (r, sites)) or {"a": 1})
    assert cli.main(["pilot", "attain", "--sites", "ems"]) == 0
    assert att == [(root, ["ems"])]
    assert cli.main(["pilot", "units"]) == 2
    assert "available from milestone V3" in capsys.readouterr().out


def test_timing_dispatch_and_refusal(root, monkeypatch, capsys):
    seen = []

    def rd(r, site, products, workloads, repeats, threads):
        seen.append((r, site, list(products), workloads, repeats, threads))
        return {p: f"{r}/timing/{site}/{p}.json" for p in products}

    monkeypatch.setattr(timing, "run_dedicated", rd)
    assert cli.main(["timing", "--site", "ems", "--products", "L01,L08", "--workloads",
                     "45,all", "--repeats", "3", "--threads", "2"]) == 0
    assert seen == [(root, "ems", ["L01", "L08"], (45, "all"), 3, 2)]

    def refuse(*a, **k):
        raise timing.TimingRefused("timing refused: 2 active claim(s)")

    monkeypatch.setattr(timing, "run_dedicated", refuse)
    assert cli.main(["timing", "--sites", "ems", "--products", "L01"]) == 2
    assert "active claim" in capsys.readouterr().out

    def unavailable(*a, **k):
        raise timing.TimingUnavailable("escalda: the timing workloads are W3 subsets")

    monkeypatch.setattr(timing, "run_dedicated", unavailable)
    assert cli.main(["timing", "--sites", "escalda", "--products", "L01"]) == 1
    assert "escalda: not timed" in capsys.readouterr().out


def test_lock_dispatch_and_refusal(root, monkeypatch, capsys):
    seen = []

    def wl(phase, r, extra_dev_judges=(), allow_missing=()):
        seen.append((phase, r, list(extra_dev_judges), list(allow_missing)))
        return {"phases": {phase: {}}, "amendments": []}

    monkeypatch.setattr(lock_mod, "write_lock", wl)
    assert cli.main(["lock", "--phase", "A", "--dev-judges", "delfzijl,fer2",
                     "--allow-missing", "ems:ocm"]) == 0
    assert seen == [("A", root, ["delfzijl", "fer2"], ["ems:ocm"])]

    def refuse(*a, **k):
        raise lock_mod.LockRefused(["the tree is dirty", "attain.json is missing"])

    monkeypatch.setattr(lock_mod, "write_lock", refuse)
    assert cli.main(["lock", "--phase", "B"]) == 2
    out = capsys.readouterr().out
    assert "[lock] refused: the tree is dirty" in out and "attain.json" in out


# ─────────────────────────────────────────────────────────────────────────────
#  control
# ─────────────────────────────────────────────────────────────────────────────

def test_control_runs_world_package_queue_and_verdicts(root, monkeypatch, capsys):
    from experiments.benchmark import package
    calls = {"prepare": [], "rows": [], "worlds": []}

    def write_world(out_root, world="gating", scenario="profile"):
        calls["worlds"].append((world, scenario))
        return {"cubes_root": f"{out_root}/cubes", "planted_path": f"{out_root}/planted.npz",
                "world_doc": {"world": world, "scenario": scenario}}

    _fake(monkeypatch, "control", write_world=write_world,
          control_site2=lambda out_root, doc: types.SimpleNamespace(site="pc", root=out_root),
          planted_provider=lambda doc: "PROVIDER",
          planted_band_lags=lambda path, dom: {"bands": [0], "tau_mean_min": [60.0],
                                               "n_px": [int(len(dom.keep))]})
    monkeypatch.setattr(package, "prepare_v2", lambda s2, steps=None, provider=None:
                        calls["prepare"].append((steps, provider)) or {})

    def qrun(opts):
        calls["rows"].append([r["method"] for r in opts.rows])
        assert opts.site2_for("pc").site == "pc" and opts.provider_for("pc") == "PROVIDER"
        return {"ran": len(opts.rows)}

    monkeypatch.setattr(queue_v2, "run", qrun)
    inst = {"keep": np.arange(5), "band_of": np.zeros(5, np.int16), "json": {"n_keep": 5}}
    monkeypatch.setattr(queue_v2, "iter_lag_products", lambda r, site=None: iter(
        [(f"{r}/a", {"product": "L01"}), (f"{r}/b", {"product": "L08"})]))
    monkeypatch.setattr(queue_v2, "load_lag_product", lambda d: {
        "product": d[-1], "status": "ok", "dir": d, "instance": inst,
        "keys": {"code_files_key": f"K-{d[-1]}"}})
    seen = []

    def verdicts(prods, planted, tol_min=None, domain=None):
        seen.append((sorted(prods), planted, tol_min, list(domain.keep)))
        return {"L01": {"verdict": "PASS", "median_abs_err_min": 2.0, "n_bands": 1},
                "L08": {"verdict": "control mismatch", "median_abs_err_min": 30.0,
                        "n_bands": 1}}

    _fake(monkeypatch, "scoring", control_verdicts=verdicts)
    assert cli.main(["control", "--kind", "lag", "--method", "L01,L08"]) == 0
    assert calls["worlds"] == [("gating", "zero"), ("gating", "profile")]
    assert calls["rows"] == [["E0", "L08"], ["E0", "L08"]]
    assert calls["prepare"] == [(None, "PROVIDER"), (("series",), "PROVIDER")] * 2
    assert seen[0][2] == 10.0 and seen[0][3] == [0, 1, 2, 3, 4]
    w = json.load(open(os.path.join(root, "control", "gating_profile.json"), encoding="utf-8"))
    assert w["verdicts"]["L01"]["verdict"] == "PASS" and w["world"] == "gating"
    p1 = json.load(open(os.path.join(root, "control", "L01.json"), encoding="utf-8"))
    assert p1["verdict"] == "PASS" and set(p1["worlds"]) == {"gating_zero", "gating_profile"}
    assert p1["scenarios"] == {"zero": "PASS", "profile": "PASS"}
    # the adapter code of the verdict: the printing rule compares it with the row's
    assert p1["code_files_key"] == "K-a" and p1["worlds"]["gating_zero"]["code_files_key"] == "K-a"
    p8 = json.load(open(os.path.join(root, "control", "L08.json"), encoding="utf-8"))
    assert p8["verdict"] == p8["overall"]["verdict"] == "control mismatch"
    assert cli._control_overall(["PASS", None]) == "pending"
    assert cli._control_overall(["PASS", "FAIL (fault)"]) == "FAIL (fault)"
    assert cli.main(["control", "--kind", "elev"]) == 2
    assert "available from milestone V3" in capsys.readouterr().out
    assert cli.main(["control", "--method", "L14"]) == 2          # not an imagery unit


def test_control_exit_code_reports_queue_errors_and_missing_products(root, monkeypatch,
                                                                     capsys):
    """A control whose queue fails before any unit runs (e.g. the domain
    step) has no product to judge: exit 1, never a silent 0."""
    from experiments.benchmark import package
    _fake(monkeypatch, "control",
          write_world=lambda out_root, world="gating", scenario="profile": {
              "cubes_root": f"{out_root}/cubes", "planted_path": f"{out_root}/planted.npz",
              "world_doc": {}},
          control_site2=lambda out_root, doc: types.SimpleNamespace(site="pc", root=out_root),
          planted_provider=lambda doc: "PROVIDER",
          planted_band_lags=lambda path, dom: {"bands": [], "tau_mean_min": [], "n_px": []})
    monkeypatch.setattr(package, "prepare_v2", lambda s2, steps=None, provider=None: {})
    monkeypatch.setattr(queue_v2, "run", lambda opts: {"error": len(opts.rows)})
    monkeypatch.setattr(queue_v2, "iter_lag_products", lambda r, site=None: iter([]))
    _fake(monkeypatch, "scoring", control_verdicts=lambda *a, **k: {})
    assert cli.main(["control", "--kind", "lag", "--method", "L02", "--scenario", "zero"]) == 1
    assert "no product" in capsys.readouterr().out


def test_later_milestones_say_so(root, capsys):
    assert cli.main(["report"]) == 2
    assert "available from milestone V6" in capsys.readouterr().out
