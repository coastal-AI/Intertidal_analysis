"""Tests of experiments.benchmark.runner and the v13 CLI (milestone M1 parts).

They pin (spec §7.4, §8): exclusive claim files (one winner among racing
threads), the heartbeat and its loss, stale claims (dead process or foreign
host, older than 30 min) taken over only with ``steal_stale`` and by exactly
one of several racing stealers; atomic publication over a failed unit (the
failed attempt kept as ``.old-<utc>``, never listed); resume by
``inputs_key`` (final statuses skipped, error / r_failed / budget rerun, a
config edit reruns, a smoke product never satisfies a real run); the status
mapping of NotComputable, R failures, BudgetExceeded, Deferred and wrong
adapter outputs; the per-unit log with stage timings and peak RSS; the BLAS
thread limits; the gate and positive-control checks; the planning rules;
the argument parsers; and the CLI's milestone messages, thread setting and
check-env cube sampler on a tiny production-layout cube.

Run:  python -X utf8 -m pytest tests/test_benchmark_runner.py -q
"""
import gc
import json
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, product, runner  # noqa: E402
from experiments.benchmark.product import NotComputable, Product, Unit  # noqa: E402

FILL = -32768


@pytest.fixture(autouse=True)
def _existing_core_files(monkeypatch):
    # the code key hashes obs.py, domain.py and levels.py; domain.py comes
    # from another M1 module and may not be in a partial checkout
    keep = tuple(f for f in product.CORE_CODE_FILES if os.path.exists(os.path.join(ROOT, f)))
    monkeypatch.setattr(product, "CORE_CODE_FILES", keep)


# ─────────────────────────────────────────────────────────────────────────────
#  fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakePkg:
    """What the runner reads from a CommonPackage: T, keys and subset()."""

    def __init__(self, T=30, key="p" * 64):
        self.T = T
        self.keys = {"package_key": key}

    def subset(self, family, N, rep):
        if N == config.N_ALL:
            return np.arange(self.T)
        rng = np.random.default_rng([rep, int(N), len(family)])
        return np.sort(rng.permutation(self.T)[:int(N)])


def _adapter(tmp_path, group="gm20", methods=("gm20",), variants=("rel",), cfg=None,
             run=None, name="fake", **extra):
    src = tmp_path / f"adapter_{name}_{group}.py"
    if not src.exists():
        src.write_text(f"# fake adapter {name} of {group}\n", encoding="utf-8")
    calls = []

    def default_run(pkg, unit, out_root, **opts):
        calls.append(dict(opts))
        return [Product(method=m, variant=unit.variant, unit=unit, status="ok",
                        z=np.arange(7, dtype=np.float32), lag=np.full(7, 30.0, np.float32),
                        lag_kind="phase") for m in methods]

    ad = SimpleNamespace(GROUP=group, METHODS=tuple(methods), VARIANTS=tuple(variants),
                         CONFIG=dict(cfg or {"k": 1}), gate=lambda pkg, root, **o: {},
                         __file__=str(src), calls=calls, **extra)
    ad.run = run or default_run
    return ad


def _unit(group="gm20", variant="rel", N=10, rep=0, sens=None, site="escalda",
          mode="A", family="rand", T=30):
    sel = np.arange(T) if N == config.N_ALL else FakePkg(T).subset(family, N, rep)
    return Unit(site=site, res_m=20, group=group, variant=variant, family=family,
                mode=mode, N=N, rep=rep, sens=sens or {}, sel=sel)


def _opts(tmp_path, reg, pkg=None, **kw):
    pkg = pkg or FakePkg()
    kw.setdefault("check_gates", False)
    return runner.RunOptions(res_m=20, root=str(tmp_path / "out"), registry=reg,
                             load=lambda site: pkg, **kw)


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    pid = p.pid
    del p
    gc.collect()                            # close the handle (Windows)
    return pid


def _foreign_claim(path, host=None, pid=None, create_time=None, age_s=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": runner.CLAIM_FORMAT, "host": host or socket.gethostname(),
                   "pid": os.getpid() if pid is None else pid, "create_time": create_time,
                   "token": "someone-else", "start_utc": "20261008T000000Z"}, f)
    t = time.time() - age_s
    os.utime(path, (t, t))


# ─────────────────────────────────────────────────────────────────────────────
#  claims
# ─────────────────────────────────────────────────────────────────────────────

def test_claim_is_exclusive_and_released(tmp_path):
    p = str(tmp_path / "a" / "u.claim")
    assert runner.claim(p, {"unit": "u1", "worker_id": "w1"})
    d = json.load(open(p, encoding="utf-8"))
    assert d["host"] == socket.gethostname() and d["pid"] == os.getpid()
    assert d["unit"] == "u1" and d["worker_id"] == "w1" and d["token"]
    assert not runner.claim(p)                       # held: no second claim
    assert not runner.claim(p, steal_stale=True)     # held and fresh: not stale
    runner.release(p)
    assert not os.path.exists(p)
    assert runner.claim(p)
    runner.release(p)


def test_claim_race_has_one_winner(tmp_path):
    p = str(tmp_path / "race.claim")
    n = 16
    barrier = threading.Barrier(n)
    wins = []

    def worker():
        barrier.wait()
        wins.append(runner.claim(p))

    ts = [threading.Thread(target=worker) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(wins) == 1
    runner.release(p)


def test_heartbeat_touches_and_detects_a_lost_claim(tmp_path):
    p = str(tmp_path / "hb.claim")
    assert runner.claim(p)
    old = time.time() - 3600
    os.utime(p, (old, old))
    runner.heartbeat(p)
    assert time.time() - os.stat(p).st_mtime < 60
    _foreign_claim(p)                                # taken over by another worker
    with pytest.raises(runner.ClaimLost):
        runner.heartbeat(p)
    runner.release(p)                                # not ours: left in place
    assert os.path.exists(p)
    os.remove(p)
    with pytest.raises(runner.ClaimLost):
        runner.heartbeat(p)


def test_stale_claims(tmp_path):
    p = str(tmp_path / "s.claim")
    old = runner.STALE_AFTER_S + 60
    # a dead process on this host, last heartbeat 31 min ago: stale
    _foreign_claim(p, pid=_dead_pid(), age_s=old)
    st = runner.claim_state(p)
    assert st["alive"] is False and st["stale"]
    assert not runner.claim(p)                       # never taken without --steal-stale
    assert runner.claim(p, steal_stale=True)
    assert json.load(open(p, encoding="utf-8"))["pid"] == os.getpid()
    runner.release(p)
    # a dead process with a recent heartbeat: not stale
    _foreign_claim(p, pid=_dead_pid(), age_s=60)
    assert not runner.claim_state(p)["stale"]
    assert not runner.claim(p, steal_stale=True)
    # an old heartbeat of a live process: not stale (it may be hung, not gone)
    _foreign_claim(p, age_s=old, create_time=runner._create_time())
    assert runner.claim_state(p)["alive"] is True
    assert not runner.claim(p, steal_stale=True)
    # the pid was reused by another process (create time differs): gone
    if runner._create_time() is not None:
        _foreign_claim(p, age_s=old, create_time=12345.0)
        assert runner.claim_state(p)["alive"] is False
        assert runner.claim(p, steal_stale=True)
        runner.release(p)
    # another host: judged by the heartbeat age alone
    _foreign_claim(p, host="another-host", pid=1, age_s=old)
    st = runner.claim_state(p)
    assert st["alive"] is None and st["stale"]
    assert runner.claim(p, steal_stale=True)
    runner.release(p)
    _foreign_claim(p, host="another-host", pid=1, age_s=60)
    assert not runner.claim(p, steal_stale=True)


def test_racing_stealers_take_a_stale_claim_once(tmp_path):
    p = str(tmp_path / "steal.claim")
    _foreign_claim(p, host="another-host", pid=1, age_s=runner.STALE_AFTER_S + 60)
    n = 12
    barrier = threading.Barrier(n)
    wins = []

    def worker():
        barrier.wait()
        wins.append(runner.claim(p, steal_stale=True))

    ts = [threading.Thread(target=worker) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(wins) == 1
    assert json.load(open(p, encoding="utf-8"))["host"] == socket.gethostname()
    assert not os.path.exists(p + ".steal")
    runner.release(p)


def test_a_crashed_stealer_lock_expires(tmp_path):
    p = str(tmp_path / "lock.claim")
    _foreign_claim(p, host="another-host", pid=1, age_s=runner.STALE_AFTER_S + 60)
    lock = p + ".steal"
    open(lock, "w").close()
    assert not runner.claim(p, steal_stale=True)    # someone is stealing it
    old = time.time() - runner.STEAL_LOCK_STALE_S - 60
    os.utime(lock, (old, old))
    assert not runner.claim(p, steal_stale=True)    # the old lock is cleared ...
    assert not os.path.exists(lock)
    assert runner.claim(p, steal_stale=True)        # ... and the next pass steals
    runner.release(p)


def test_claim_path_is_outside_the_product_tree(tmp_path):
    u = _unit(sens={"mask": "blocklist"})
    cp = runner.claim_path(str(tmp_path), u)
    assert "/runs/escalda/_claims/gm20__rel/rand-A/N010_r00__mask-blocklist.claim" in cp
    d = product.unit_dir(str(tmp_path), u, "gm20", "rel")
    assert os.path.basename(cp)[:-len(".claim")] == os.path.basename(d)


# ─────────────────────────────────────────────────────────────────────────────
#  publication, resume, statuses
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_publish_over_a_failed_unit(tmp_path):
    state = {"fail": True}

    def run(pkg, unit, out_root, **opts):
        print("adapter says hello")
        if state["fail"]:
            raise RuntimeError("boom")
        return [Product(method="gm20", variant=unit.variant, unit=unit, status="ok",
                        lag=np.zeros(5, np.float32), lag_kind="phase")]

    ad = _adapter(tmp_path, run=run)
    opts = _opts(tmp_path, {"gm20": ad})
    u = _unit()
    d = product.unit_dir(opts.root, u, "gm20", "rel")
    out = runner.process_unit(u, opts)
    assert out.outcome == "ran" and out.products[0].status == "error"
    r = product.read_run(d)
    assert r["status"] == "error" and "RuntimeError: boom" in r["reason"]
    assert "boom" in r["diag"]["traceback"]
    assert "adapter says hello" in open(os.path.join(d, "log.txt"), encoding="utf-8").read()
    state["fail"] = False
    out = runner.process_unit(u, opts)                # error is not final: rerun
    assert out.outcome == "ran" and out.products[0].status == "ok"
    assert product.read_run(d)["status"] == "ok"
    olds = [x for x in os.listdir(os.path.dirname(d)) if ".old-" in x]
    assert len(olds) == 1
    assert product.read_run(os.path.join(os.path.dirname(d), olds[0]))["status"] == "error"
    assert not [x for x in os.listdir(os.path.dirname(d)) if ".tmp-" in x]
    rows = runner.status_rows(opts.root)
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert not os.path.exists(runner.claim_path(opts.root, u))      # released
    work = os.path.join(opts.root, "runs", "escalda", runner.WORK_DIR)
    assert not os.path.exists(work) or not any(
        fs for _, _, fs in os.walk(work))                            # scratch removed


def test_resume_by_inputs_key(tmp_path):
    ad = _adapter(tmp_path)
    opts = _opts(tmp_path, {"gm20": ad})
    u = _unit()
    assert runner.process_unit(u, opts).outcome == "ran"
    assert runner.process_unit(u, opts).outcome == "done"
    assert len(ad.calls) == 1
    run = product.read_run(product.unit_dir(opts.root, u, "gm20", "rel"))
    keys = run["keys"]
    pkg = FakePkg()
    code = product.code_key(ad.__file__)
    ck = product.config_key(ad.CONFIG, "gm20", "A", {}, "rel", group="gm20")
    assert keys["config_key"] == ck and keys["code_key"] == code
    assert keys["inputs_key"] == product.inputs_key(pkg, u, "gm20", "rel", ck, code)
    assert keys["frozen"]["gm20"] == config.frozen_sha("gm20")
    # another subset, sensitivity or config is another unit / another key
    assert runner.process_unit(_unit(rep=1), opts).outcome == "ran"
    assert runner.process_unit(_unit(sens={"mask": "blocklist"}), opts).outcome == "ran"
    ad.CONFIG = {"k": 2}
    assert runner.process_unit(u, opts).outcome == "ran"
    assert len(ad.calls) == 4
    # --force reruns a done unit
    opts.force = True
    assert runner.process_unit(u, opts).outcome == "ran"
    assert len(ad.calls) == 5


@pytest.mark.parametrize("exc, status, final", [
    (NotComputable("record_floor", "12 scenes < 40", {"n": 12}), "not_computable", True),
    (RuntimeError("R job lag#3 failed (rc=1):\nError in nplr: bad"), "r_failed", False),
    (runner.RFailed("prelim R crashed", log_tail="tail text"), "r_failed", False),
    (runner.BudgetExceeded("projected 31 h > 24 h", projected_r_hours=31.0), "budget", False),
    (MemoryError("out of memory"), "error", False),
])
def test_status_mapping(tmp_path, exc, status, final):
    calls = []

    def run(pkg, unit, out_root, **opts):
        calls.append(1)
        raise exc

    ad = _adapter(tmp_path, group="marea_pair", methods=("nodelay", "marea"), variants=("",),
                  run=run)
    opts = _opts(tmp_path, {"marea_pair": ad})
    u = _unit(group="marea_pair", variant="")
    out = runner.process_unit(u, opts)
    assert [p.status for p in out.products] == [status, status]
    for m in ("nodelay", "marea"):
        r = product.read_run(product.unit_dir(opts.root, u, m, ""))
        assert r["status"] == status
        if status == "not_computable":
            assert r["stage"] == "record_floor" and r["reason"] == "12 scenes < 40"
            assert r["diag"] == {"n": 12}
        if status == "r_failed":
            assert r["diag"]["r_log_tail"]
        if status == "budget":
            assert r["diag"]["projected_r_hours"] == 31.0
        if status == "error":
            assert "MemoryError" in r["diag"]["traceback"]
    again = runner.process_unit(u, opts)
    assert again.outcome == ("done" if final else "ran")
    assert len(calls) == (1 if final else 2)


def test_deferred_and_wrong_outputs(tmp_path):
    def deferred(pkg, unit, out_root, **opts):
        raise runner.Deferred("the N = all unit is not done")

    ad = _adapter(tmp_path, run=deferred)
    opts = _opts(tmp_path, {"gm20": ad})
    u = _unit()
    out = runner.process_unit(u, opts)
    assert out.outcome == "deferred" and out.products == []
    assert not os.path.exists(product.unit_dir(opts.root, u, "gm20", "rel"))
    assert not os.path.exists(runner.claim_path(opts.root, u))

    def wrong(pkg, unit, out_root, **opts):
        return [Product(method="other", variant=unit.variant, unit=unit, status="ok")]

    ad.run = wrong
    out = runner.process_unit(u, opts)
    assert [p.status for p in out.products] == ["error"]
    assert "expected [('gm20', 'rel')]" in out.products[0].reason


def test_log_timings_memory_and_threads(tmp_path):
    def run(pkg, unit, out_root, **opts):
        with runner.stage("fit"):
            print("fitting the sinusoid")
            time.sleep(0.05)
        with runner.stage("qc"):
            pass
        assert os.environ["OPENBLAS_NUM_THREADS"] == "4"   # what child processes inherit
        assert os.path.isdir(opts["work_dir"])
        return [Product(method="gm20", variant=unit.variant, unit=unit, status="ok",
                        lag=np.zeros(3, np.float32), lag_kind="phase")]

    ad = _adapter(tmp_path, run=run)
    opts = _opts(tmp_path, {"gm20": ad}, worker_id="w7")
    u = _unit()
    p = runner.process_unit(u, opts).products[0]
    d = product.unit_dir(opts.root, u, "gm20", "rel")
    text = open(os.path.join(d, "log.txt"), encoding="utf-8").read()
    for s in ("=== unit escalda 20m gm20[rel]", "fitting the sinusoid", "[stage] fit: done",
              "[stage] qc: done", "peak RSS", "BLAS threads 4", "worker w7"):
        assert s in text, s
    r = product.read_run(d)
    assert r["timings"]["fit"] >= 0.04 and "qc" in r["timings"] and r["timings"]["total"] > 0
    assert np.isfinite(r["peak_rss_mb"]) and r["peak_rss_mb"] > 10
    mem = r["provenance"]["memory"]
    assert mem["sampled_unit_peak_mb"] > 10 and mem["peak_scope"] in ("unit", "process")
    assert r["provenance"]["threads"]["requested"] == config.FROZEN["threads"]["blas"]["gm20"]
    assert r["provenance"]["worker"]["worker_id"] == "w7"
    assert r["smoke"] is False and p.smoke is False


def test_heartbeat_runs_during_a_unit(tmp_path):
    def run(pkg, unit, out_root, **opts):
        time.sleep(0.4)
        return [Product(method="gm20", variant=unit.variant, unit=unit, status="empty")]

    ad = _adapter(tmp_path, run=run)
    opts = _opts(tmp_path, {"gm20": ad}, heartbeat_s=0.05)
    p = runner.process_unit(_unit(), opts).products[0]
    assert p.provenance["claim"]["heartbeats"] >= 2
    assert p.provenance["claim"]["lost"] is None


def test_r_runner_peak_enters_the_unit_peak(tmp_path):
    def run(pkg, unit, out_root, **opts):
        return [Product(method="granadeiro", variant=unit.variant, unit=unit, status="ok",
                        diag={"r_runner": {"python_plus_r_peak_mb": 1.0e6, "jobs": 3}})]

    ad = _adapter(tmp_path, group="granadeiro", methods=("granadeiro",),
                  variants=("ctl",), run=run)
    opts = _opts(tmp_path, {"granadeiro": ad})
    u = _unit(group="granadeiro", variant="ctl")
    p = runner.process_unit(u, opts).products[0]
    assert p.peak_rss_mb == 1.0e6
    r = product.read_run(product.unit_dir(opts.root, u, "granadeiro", "ctl"))
    assert r["provenance"]["threads"]["requested"] == 1        # FROZEN: granadeiro = 1


def test_smoke_products_never_satisfy_a_real_run(tmp_path):
    ad = _adapter(tmp_path)
    smoke = _opts(tmp_path, {"gm20": ad}, smoke=True)
    real = _opts(tmp_path, {"gm20": ad})
    u = _unit()
    p = runner.process_unit(u, smoke).products[0]
    assert p.smoke is True
    k_smoke = p.keys["inputs_key"]
    assert runner.process_unit(u, smoke).outcome == "done"
    out = runner.process_unit(u, real)
    assert out.outcome == "ran" and out.products[0].smoke is False
    assert out.products[0].keys["inputs_key"] != k_smoke
    # a BLAS override is a smoke setting too
    bt = _opts(tmp_path, {"gm20": ad}, blas_threads=1)
    assert runner.smoke_settings(bt, "gm20") == {"blas_threads": 1}
    assert runner.smoke_settings(_opts(tmp_path, {}, blas_threads=4), "gm20") == {}


def test_queue_skips_claimed_units_and_steals_stale_ones(tmp_path):
    ad = _adapter(tmp_path)
    opts = _opts(tmp_path, {"gm20": ad})
    u1, u2 = _unit(rep=0), _unit(rep=1)
    cp = runner.claim_path(opts.root, u1)
    _foreign_claim(cp)                               # a live worker (this pid, other token)
    c = runner.run_queue([u1, u2], opts)
    assert c["claimed"] == 1 and c["ran"] == 1 and c["status:ok"] == 1
    assert product.read_run(product.unit_dir(opts.root, u1, "gm20", "rel")) is None
    _foreign_claim(cp, host="another-host", pid=1, age_s=runner.STALE_AFTER_S + 60)
    c = runner.run_queue([u1, u2], opts)
    assert c["stale_claim"] == 1 and c["done"] == 1
    opts.steal_stale = True
    c = runner.run_queue([u1, u2], opts)
    assert c["ran"] == 1 and c["done"] == 1
    assert not os.path.exists(cp)


def test_queue_stop_file_and_unit_state(tmp_path):
    ad = _adapter(tmp_path)
    opts = _opts(tmp_path, {"gm20": ad}, worker_id="w2")
    units = [_unit(rep=r) for r in range(3)]
    assert [runner.unit_state(u, opts) for u in units] == ["pending"] * 3
    os.makedirs(opts.root, exist_ok=True)
    open(os.path.join(opts.root, "STOP-w2"), "w").close()
    assert runner.run_queue(units, opts).get("stopped") == 1
    os.remove(os.path.join(opts.root, "STOP-w2"))
    opts.max_units = 1
    assert runner.run_queue(units, opts)["ran"] == 1
    assert [runner.unit_state(u, opts) for u in units] == ["done", "pending", "pending"]

    def boom(pkg, unit, out_root, **o):
        raise ValueError("x")

    ad.run = boom
    runner.process_unit(units[1], opts)
    assert runner.unit_state(units[1], opts) == "failed:error"


# ─────────────────────────────────────────────────────────────────────────────
#  gates and the positive control
# ─────────────────────────────────────────────────────────────────────────────

def _gate(root, site, gate, verdict="PASS", package_key="p" * 64):
    product.write_json(runner.gate_path(site, 20, gate, root),
                       {"gate": gate, "verdict": verdict, "package_key": package_key})


def test_gates_are_checked_before_a_group_runs(tmp_path):
    ad = _adapter(tmp_path, group="marea_pair", methods=("nodelay", "marea"), variants=("",))
    opts = _opts(tmp_path, {"marea_pair": ad}, check_gates=True)
    u = _unit(group="marea_pair", variant="")
    out = runner.process_unit(u, opts)
    assert out.outcome == "refused" and "missing" in out.message and not ad.calls
    for g in ("record", "levels", "keep"):
        _gate(opts.root, "escalda", g)
    _gate(opts.root, "escalda", "marea", verdict="FAIL")
    g = runner.check_gates("escalda", 20, "marea_pair", "p" * 64, opts.root)
    assert not g["ok"] and g["failed"] == ["marea"]
    _gate(opts.root, "escalda", "marea", package_key="q" * 64)        # another package
    g = runner.check_gates("escalda", 20, "marea_pair", "p" * 64, opts.root)
    assert not g["ok"] and g["stale"] == ["marea"]
    assert runner.process_unit(u, opts).outcome == "refused"
    smoke = _opts(tmp_path, {"marea_pair": ad}, check_gates=True, smoke=True)
    assert runner.process_unit(u, smoke).outcome == "ran"             # smoke: warning only
    _gate(opts.root, "escalda", "marea")
    assert runner.check_gates("escalda", 20, "marea_pair", "p" * 64, opts.root)["ok"]
    assert runner.process_unit(u, opts).outcome == "ran"
    # the positive control is exempt (it is part of the validation)
    pc = _unit(group="marea_pair", variant="", site="pc")
    assert runner.process_unit(pc, opts).outcome == "ran"
    # Granadeiro's gate runs at Villaviciosa 10 m and counts everywhere
    req = runner.required_gates("escalda", 20, "granadeiro")
    assert ("granadeiro", "villaviciosa", 10) in req and ("record", "escalda", 20) in req
    assert runner.required_gates("escalda", 20, "published") == []


def test_validated_needs_gates_and_the_positive_control(tmp_path):
    root, croot = str(tmp_path / "out"), str(tmp_path / "control")
    for g in ("record", "levels", "keep", "marea"):
        _gate(root, "escalda", g)
    assert runner.control_status("marea", croot) == {"zero": "missing", "profile": "missing"}
    assert not runner.validated("escalda", 20, "marea", root=root, control_root=croot)
    product.write_json(f"{croot}/zero/verdict.json",
                       {"methods": {"marea": {"verdict": "PASS"},
                                    "nodelay": {"verdict": "PASS"}}})
    product.write_json(f"{croot}/profile/verdict.json",
                       {"methods": {"marea": {"verdict": "PASS"},
                                    "nodelay": {"verdict": "REPORTED"}}})
    assert runner.validated("escalda", 20, "marea", root=root, control_root=croot)
    # no delay must pass only the zero scenario (profile is reported)
    assert runner.control_required("nodelay", "zero")
    assert not runner.control_required("nodelay", "profile")
    assert runner.validated("escalda", 20, "nodelay", root=root, control_root=croot)
    # every lag method must pass zero; GM20 has no elevation
    assert runner.control_required("gm20", "zero") and runner.control_required("gm20", "profile")
    assert not runner.validated("escalda", 20, "gm20", root=root, control_root=croot)
    product.write_json(f"{croot}/profile/verdict.json",
                       {"methods": {"marea": {"verdict": "FAIL"}}})
    assert not runner.validated("escalda", 20, "marea", root=root, control_root=croot)


# ─────────────────────────────────────────────────────────────────────────────
#  planning and parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_plan_rules(tmp_path):
    reg = {"gm20": _adapter(tmp_path),
           "nidem": _adapter(tmp_path, group="nidem", methods=("nidem",), variants=("",),
                             MODES=("A",), SITES=("escalda", "wadden", "ems")),
           "marea_pair": _adapter(tmp_path, group="marea_pair", methods=("nodelay", "marea"),
                                  variants=("", "tau120"), SENS_KEYS=("mask", "index", "floor"))}
    pkg = FakePkg(T=30)
    units = runner.plan(["escalda"], ["gm20", "nidem"], ["rand", "win", "rule-tz"],
                        ["A", "F"], [10, 40, "all"], [0, 1], None, [{}], 20,
                        load=lambda s: pkg, registry_=reg)
    labels = [(u.group, u.family, u.mode, u.N, u.rep) for u in units]
    assert len(labels) == len(set(labels))
    assert labels[0] == ("gm20", "rand", "A", "all", 0)              # N = all first
    assert ("gm20", "rand", "A", "all", 1) not in labels            # all: rep 0 only
    assert not any(f == "win" and n == "all" for _, f, _, n, _ in labels)
    assert not any(n == 40 for _, _, _, n, _ in labels)             # N > T
    assert [x for x in labels if x[1] == "rule-tz" and x[0] == "gm20"] == [
        ("gm20", "rule-tz", "A", "all", 0), ("gm20", "rule-tz", "F", "all", 0)]
    assert not any(g == "nidem" and m == "F" for g, _, m, _, _ in labels)   # A serves F
    assert ("nidem", "rand", "A", 10, 1) in labels
    for u in units:
        assert np.array_equal(u.sel, pkg.subset(u.family, u.N, u.rep))
        assert u.variant == ("rel" if u.group == "gm20" else "")
    assert not runner.plan(["ferrol"], ["nidem"], ["rand"], ["A"], ["all"], [0], None,
                           [{}], 10, load=lambda s: pkg, registry_=reg)
    us = runner.plan(["escalda"], ["marea_pair", "gm20"], ["rand"], ["A"], ["all"], [0],
                     ["tau120"], [{"floor": 80}], 20, load=lambda s: pkg, registry_=reg)
    assert [(u.group, u.variant, u.sens) for u in us] == [("marea_pair", "tau120", {"floor": 80})]
    with pytest.raises(KeyError, match="milestone M3"):
        runner.plan(["escalda"], ["granadeiro"], ["rand"], ["A"], ["all"], [0], None, [{}],
                    20, load=lambda s: pkg, registry_=reg)


def test_parsers():
    assert runner.parse_N("all") == ["all"]
    assert runner.parse_N("grid") == list(config.N_GRID) + ["all"]
    assert runner.parse_N("gran,10") == list(config.GRAN_N_GRID) + ["all", 10]
    with pytest.raises(ValueError):
        runner.parse_N("lots")
    assert runner.parse_reps("0-9") == list(range(10))
    assert runner.parse_reps("4,0,2-3") == [0, 2, 3, 4]
    with pytest.raises(ValueError):
        runner.parse_reps("3-1")
    assert runner.parse_sens(None) == [{}]
    assert runner.parse_sens(["mask=blocklist,floor=80", "index=raw", "band_share=0.5"]) == [
        {"mask": "blocklist", "floor": 80}, {"index": "raw"}, {"band_share": 0.5}]
    for bad in ("mask=marea_clear", "index=floor0", "floor=40", "mask=cloudy", "floor=-1",
                "nonsense"):
        with pytest.raises(ValueError):
            runner.parse_sens([bad])
    # numbers typed differently give the same unit directory and config key
    a = runner.parse_sens(["floor=80"])[0]
    b = runner.parse_sens(["floor=+80"])[0]
    assert product.senstag(a) == product.senstag(b) == "floor-80"


def test_status_rows_and_csv(tmp_path):
    ad = _adapter(tmp_path, group="marea_pair", methods=("nodelay", "marea"), variants=("",))
    opts = _opts(tmp_path, {"marea_pair": ad})
    runner.process_unit(_unit(group="marea_pair", variant=""), opts)
    rows = runner.status_rows(opts.root, "escalda")
    assert sorted(r["method"] for r in rows) == ["marea", "nodelay"]
    assert all(r["status"] == "ok" and r["n_scenes"] == 10 and r["seconds"] >= 0
               for r in rows)
    path = runner.write_status_csv(rows, str(tmp_path / "scores" / "status.csv"))
    head = open(path, encoding="utf-8").readline().strip().split(",")
    assert head == list(runner.STATUS_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
#  threads and memory
# ─────────────────────────────────────────────────────────────────────────────

def test_blas_limits():
    tpc = pytest.importorskip("threadpoolctl")
    before = os.environ.get("OPENBLAS_NUM_THREADS")
    with runner.blas_limits(1) as info:
        assert info["api"] == "threadpoolctl" and info["requested"] == 1
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1" and os.environ["OMP_NUM_THREADS"] == "1"
        blas = [p for p in tpc.threadpool_info() if p["user_api"] == "blas"]
        assert blas and all(p["num_threads"] == 1 for p in blas)
        assert info["effective"] == [1]
    assert os.environ.get("OPENBLAS_NUM_THREADS") == before
    with pytest.raises(ValueError):
        with runner.blas_limits(0):
            pass


def test_memory_helpers():
    a = np.ones(20_000_000)                          # 160 MB: the peak must see it
    rss = runner.current_rss_mb()
    assert rss > 150
    assert runner.process_peak_rss_mb() >= rss * 0.9
    del a
    p = runner.peak_rss_mb()
    assert np.isfinite(p) and p > 150
    assert runner.peak_rss_mb({"python_plus_r_peak_mb": 1e7}) == 1e7
    assert runner.peak_rss_mb({"python_plus_r_peak_mb": None}) == pytest.approx(p, rel=0.5)


@pytest.mark.skipif(os.name != "nt", reason="Windows memory counters")
def test_windows_counters_without_psutil():
    psutil = pytest.importorskip("psutil")
    ws, peak = runner._win_memory_counters()
    mi = psutil.Process().memory_info()
    assert ws / mi.rss == pytest.approx(1.0, rel=0.2)
    assert peak / mi.peak_wset == pytest.approx(1.0, rel=0.2)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux high-water mark")
def test_linux_peak_reset():
    a = np.ones(25_000_000)
    del a
    high = runner.process_peak_rss_mb()
    assert runner.reset_peak_rss()
    assert runner.process_peak_rss_mb() < high


def test_r_failure_detection():
    assert runner.is_r_failure(RuntimeError("R job prelim#12 failed (rc=1):\nError"))
    assert runner.is_r_failure(runner.RFailed("x"))
    assert not runner.is_r_failure(RuntimeError("something else"))
    assert not runner.is_r_failure(ValueError("R job a#1 failed (rc=1)"))


# ─────────────────────────────────────────────────────────────────────────────
#  the CLI
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli(monkeypatch):
    from experiments import v13_benchmark
    monkeypatch.chdir(ROOT)                          # main() chdirs; restored afterwards
    for k in v13_benchmark.THREAD_ENV_VARS:
        monkeypatch.setenv(k, os.environ.get(k, "4"))
    return v13_benchmark


@pytest.mark.parametrize("argv, ms", [
    (["lock", "--res", "10"], "M2"),
    (["score", "all", "--res", "10"], "M2"),
    (["report", "--res", "10", "--paper"], "M2"),
    (["gate", "escalda", "--res", "20", "--gate", "record,keep"], "M2"),
    (["gate", "villaviciosa", "--res", "10", "--gate", "granadeiro"], "M3"),
    (["control", "--scenario", "zero", "--groups", "granadeiro"], "M3"),
])
def test_cli_later_commands(cli, capsys, argv, ms):
    assert cli.main(argv) == 2
    out = capsys.readouterr().out
    assert f"available from milestone {ms}" in out


def test_cli_run_waits_for_the_adapters(cli, capsys, monkeypatch):
    def no_adapters():
        raise ImportError("no adapters")

    monkeypatch.setattr(runner, "registry", no_adapters)
    assert cli.main(["run", "all", "--res", "10", "--groups", "marea_pair,gm20"]) == 2
    assert "available from milestone M2" in capsys.readouterr().out
    assert cli.main(["run", "escalda", "--res", "20", "--groups", "granadeiro"]) == 2
    assert "available from milestone M3" in capsys.readouterr().out


def test_cli_thread_count(cli):
    p = cli.build_parser()
    n, _ = cli.thread_count(p.parse_args(["run", "all", "--res", "10", "--groups", "granadeiro"]))
    assert n == 1
    n, _ = cli.thread_count(p.parse_args(["run", "all", "--res", "10",
                                          "--groups", "granadeiro,marea_pair"]))
    assert n == 4
    n, why = cli.thread_count(p.parse_args(["prepare", "escalda", "--res", "20",
                                            "--blas-threads", "2"]))
    assert (n, why) == (2, "--blas-threads")
    assert cli.thread_count(p.parse_args(["status", "--res", "20"]))[0] == cli.DEFAULT_THREADS


def test_cli_call_refuses_an_unknown_keyword(cli):
    def prepare(site, res_m, steps=None, force=False):
        return (site, res_m, steps, force)

    assert cli._call(prepare, "escalda", 20, steps=None, force=True, scenario=None) == (
        "escalda", 20, None, True)
    with pytest.raises(TypeError, match="scenario"):
        cli._call(prepare, "pc", 20, scenario="zero")


def test_cli_sites(cli):
    assert cli._sites("all", 20) == [("escalda", None), ("wadden", None), ("ems", None)]
    assert cli._sites("pc", 20, "zero") == [("pc", "zero")]
    assert cli._sites("pc", 20) == [("pc", "zero"), ("pc", "profile")]
    with pytest.raises(ValueError):
        cli._sites("ferrol", 20)


def test_write_frozen_keeps_a_changed_plan(cli, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = cli.write_frozen(20)
    first = os.stat(p).st_mtime_ns
    assert json.load(open(p, encoding="utf-8"))["sha256"] == config.frozen_shas()
    cli.write_frozen(20)                             # unchanged: not rewritten
    assert os.stat(p).st_mtime_ns == first
    doc = config.frozen_document()
    doc["sha256"]["marea"] = "changed"
    monkeypatch.setattr(config, "frozen_document", lambda: doc)
    cli.write_frozen(20)
    kept = [f for f in os.listdir(os.path.dirname(p)) if f.startswith("FROZEN.2")]
    assert len(kept) == 1
    assert json.load(open(p, encoding="utf-8"))["sha256"]["marea"] == "changed"


def _cube(path, b03, b08, scl):
    xr = pytest.importorskip("xarray")
    pyproj = pytest.importorskip("pyproj")
    import pandas as pd

    T, H, W = b03.shape
    ds = xr.Dataset(
        {v: (("t", "y", "x"), a) for v, a in (("B03", b03), ("B08", b08), ("SCL", scl))},
        coords={"t": pd.date_range("2022-12-27", periods=T, freq="7D"),
                "y": 5_700_000.0 - 10.0 * np.arange(H), "x": 500_000.0 + 10.0 * np.arange(W)})
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"crs_wkt": pyproj.CRS.from_epsg(32631).to_wkt()})
    enc = {v: {"dtype": "int16", "_FillValue": FILL, "chunksizes": (1, 8, 8), "zlib": True}
           for v in ("B03", "B08", "SCL")}
    ds.to_netcdf(path, encoding=enc)
    return path


def test_check_env_cube_sampler(cli, tmp_path):
    rng = np.random.default_rng(5)
    T, H, W = 12, 9, 11
    b03 = rng.integers(-999, 2500, (T, H, W)).astype(np.int16)
    b08 = rng.integers(-999, 2500, (T, H, W)).astype(np.int16)
    scl = rng.choice([1, 2, 4, 5, 6, 8, 11], (T, H, W)).astype(np.int16)
    b03[3, 0, 0] = FILL
    scl[4, 1, 1] = FILL
    good = cli.inspect_cube(_cube(str(tmp_path / "good.nc"), b03, b08, scl), n_sample=5)
    assert good["problems"] == []
    assert good["shape"] == (T, H, W) and good["chunks"]["B03"] == (1, 8, 8)
    # the first scene (2022-12-27) is outside the period: never sampled
    assert good["n_in_period"] == T - 1 and len(good["samples"]) == 5
    assert all(s["date"] >= "2023-01-01" for s in good["samples"])
    picked = [s["index"] for s in good["samples"]]
    expect = np.r_[1:T][np.round(np.linspace(0, T - 2, 5)).astype(int)]
    assert picked == expect.tolist()
    assert good["scl_counts"][1] == sum(int((scl[i] == 1).sum()) for i in picked)
    assert sum(good["scl_counts"]) == 5 * H * W
    # a value below -1000 (not DN - 1000 data) and an SCL code outside 0..11
    b08[picked[2], 4, 4] = -1500
    scl[picked[3], 2, 2] = 13
    bad = cli.inspect_cube(_cube(str(tmp_path / "bad.nc"), b03, b08, scl), n_sample=5)
    assert any("below -1000" in p for p in bad["problems"])
    assert any("SCL" in p and "13" in p for p in bad["problems"])
    # a scaled band is refused before any scene is read
    xr = pytest.importorskip("xarray")
    ds = xr.open_dataset(str(tmp_path / "good.nc"), mask_and_scale=False)
    ds = ds.load()
    ds.close()
    ds["B08"].attrs["scale_factor"] = 0.0001
    enc = {v: {"dtype": "int16", "chunksizes": (1, 8, 8)} for v in ("B03", "B08", "SCL")}
    ds.to_netcdf(str(tmp_path / "scaled.nc"), encoding=enc)
    sc = cli.inspect_cube(str(tmp_path / "scaled.nc"))
    assert any("scale_factor" in p for p in sc["problems"]) and sc["samples"] == []
