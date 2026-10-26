# -*- coding: utf-8 -*-
"""Validation v2 — the instance-major run queue of the lag benchmark (spec §5.1, §13.2-§13.5).

Question
--------
How do several workers (tmux panes on the server, one process on the
laptop) run the units of the plans in priority order, each instance's
methods one after the other on one shared instance, so that any worker can
be killed and restarted at any moment, no unit runs twice at the same time,
a finished unit is never recomputed while its inputs are unchanged, a failed
attempt is kept, and every unit leaves its timings, log and keys?

Pre-registered rules (spec §13.2-§13.5; plan INTERFACES §5-§9)
--------------------------------------------------------------
* Plans: ``plans/<design>.csv`` (``designs.CSV_COLUMNS``) in the order given
  (``--plans lag,base``), rows in file order; only ``planned`` rows run
  (``not_applicable``, ``shared`` and ``reuse`` rows never do; a shared row
  has the unit id of the unit it shares). A unit id met again in a later
  plan is the same unit (its run.json lists every design).
* Instance-major (§5.1, §13.4): rows are grouped by instance key (site,
  window, setting without its level, family, mode, N, rep, instance
  variant ``bnd-*``); a worker claims one instance
  (``<root>/claims/<site>/<instance_id>.claim``; a worker restricted with
  ``--methods`` claims ``<instance_id>__m<sha8 of the methods>.claim`` so it
  never blocks the instance for the others; ``runner.claim`` with
  O_CREAT|O_EXCL, a heartbeat every 60 s, ``--steal-stale`` after 30 min
  without a heartbeat when the claimant's liveness is unknown, i.e. another
  host; a claim whose process is GONE on this host (SIGKILL, the OOM killer,
  a reboot: ``runner.claim_state`` alive False) is taken over at once, with
  or without ``--steal-stale``, under the claim's steal lock), takes a unit
  claim (``<instance_id>/<unit_id>.claim``, the same rules) for every unit it
  runs, so two workers never run one unit at the same time, and runs every
  not-done unit of it in plan order on ONE
  :class:`instance.LagInstance` (built once; its lazy domain, observations
  and level rows are shared, and ``inst.cache`` holds what the methods share,
  e.g. E0's lag stage for the MAREA family).
* Keys (§13.2; plan INTERFACES §7): ``package_keys`` (the window's year-shard
  keys and the package key), ``subset_key`` (the instance's scenes),
  ``effective_key`` (the plan row's), ``config_key`` (the adapter module's
  CONFIG, the cards of the unit's products, the variant and the options that
  change results: n_boot, refits, seed_offset, BLAS threads, smoke),
  ``code_key`` (the adapter module, lags/common.py, instance.py, obs.py,
  domain.py, levels.py, the module's CODE_DEPS, and the source hashes of the
  registry entries the effective key uses, ``settings.code_shas``),
  ``code_files_key`` (the same source files without the registry entries:
  window- and cell-free, so the positive control's products carry the key
  of the real rows' code and the printing rule can tell a stale control)
  and ``inputs_key`` (all of them, the method, the variant and the module's
  ``external_inputs``). A unit is done when every product folder holds a
  final status (ok, empty, not_computable), the same inputs_key and the same
  smoke flag; the check builds the instance's scenes, never its domain.
  Adding a method never changes another unit's key.
* RAM ledger (§13.5): ``claims/ram_ledger.json`` under ``ram_ledger.lock``;
  an instance reserves max(est_ram_gb of its pending units) x 1.25 and is
  skipped while the reservations plus a 6 GB margin (``--ram-margin-gb``)
  exceed the physical RAM; entries of dead processes on this host are
  dropped. The CLI's workers raise their own ``oom_score_adj`` to 1000
  (Linux; an unprivileged process may), so the kernel kills a cheap worker
  before a prepare or Granadeiro.
* Memory figures: before each unit the package's cached tile memmaps are
  closed (``pkg._mm``, re-opened lazily), so the unit's peak RSS no longer
  counts the file pages of tiles read by earlier units; run.json
  ``memory`` also records ``peak_anon_mb``, the peak anonymous memory
  (Linux RssAnon; the private bytes elsewhere), which the cost table uses.
* One BLAS thread count per queue root: the first run writes
  ``claims/queue.json`` {threads}; a worker with another ``--threads`` is
  refused (:class:`QueueRefused`) unless ``threads_change`` (CLI
  ``--threads-change``), because the thread count enters every config key
  and a mixed queue would recompute every finished unit.
* Packages: each worker opens a site's package once, and re-opens it (or
  retries a package that could not be opened) when one of its key files
  changed (record, levels, series, marks, geometry, dom/current.json, the
  shard manifests), so a worker started while V1 is still preparing a site
  picks the site up when V1 finishes.
* ``STOP`` / ``STOP-<worker>`` in the root stop the queue before the next
  instance and between units.
* A unit runs under :class:`timing.Stopwatch` (stage names
  ``lag.<method>.<name>``), ``runner.blas_limits`` and ``runner.UnitLog``;
  every returned dict is validated by ``common.lag_result`` /
  ``common.elev_result`` (a band product without its broadcast field gets it
  from ``domain.band_of``). ``product.NotComputable`` makes every product of
  the unit not_computable; :class:`instance.InstanceUnavailable` makes the
  unit 'unavailable' (status ledger, re-checked on every pass); any other
  exception gives 'error' with the traceback (not final). A unit whose
  progress stalls for 30 min is flagged hung, one running longer than 3 x
  est_hours (floor :data:`WALL_CAP_FLOOR_S` = 6 h, declared: the V2
  est_hours are the cost model's, not the pilot's) is flagged budget: the
  status ledger reads 'hung' / 'budget' while the unit runs, and a hung unit
  pauses the claim's heartbeat so another worker can take the instance over
  (resumed when the unit returns; a worker whose claim was taken over
  publishes nothing more and leaves the instance). DECLARED amendment of
  spec §13.5 (2026-10-20, implementation only, before Stage L): a unit that
  RETURNS is final with the adapter's own status; the flags are recorded in
  run.json ``monitor`` and in every product's ``diag.monitor`` (a Python
  thread cannot be interrupted, so the flags only ever mark slow units, and
  discarding a deterministic result would make the same unit hang again on
  every rerun). Only a unit that never returns stays 'hung' / 'budget' in the
  ledger, which the next run of the instance redoes.
* Publishing is atomic (§13.5): ``<dir>.tmp-<host>-<pid>-<thread>`` is
  written and fsynced (files and folder), an existing ``<dir>`` is renamed
  ``<dir>.old-<utc>``, then ``os.replace`` and an fsync of the parent; a
  unit is done only when its run.json AND its product.npz are readable. A
  failure while publishing (a full disk, a permission error) records the
  unit as 'error' (``publish: ...``) and the worker goes on. At the start
  of a run, this host's ``runs/**/*.tmp-<host>-<pid>*`` folders whose
  process is gone are moved (never deleted) to ``<folder>.old-<utc>``.
  Nothing is deleted. A unit whose adapter module is not built yet gets
  status 'error' (reason 'module not built: ...') in the status ledger
  ``claims/status/<unit_id>.json`` only; units of methods without a lag
  adapter (elevation units of later milestones) read 'unavailable'.

Product layout (plan INTERFACES §6): ``<root>/runs/<site>/<window>/
<setting_id>/<product>[__<variant>]/<family>-<mode>/<Nall|N%03d>_r%02d/``
(``NruleC2_r01`` for a matched-N row) with ``product.npz`` (field, band_tau,
band_n_px, band_edge, band_lo, band_hi, iv_lo, iv_hi, z, sigma, clear_obs,
tau_px when present, and the ``diag.<i>`` arrays), ``run.json`` and
``unit.log``; the instance record in ``.../<setting>/_instance/<family>-
<mode>/<leaf>[__<instance_variant>]/``. run.json is flat: the identity
(format 'v2-lag-product-1', product, method, variant, designs, unit_id,
site, window, setting, family, mode, N, rep), the outcome (status, stage,
reason, kind, lag_kind, edge_share, n_defined, point, groups with tau lists,
interval95 {level, nominal}), ``instance`` {path relative to the product
folder, instance_sha256, instance_id}, ``keys``, ``external_inputs``,
``timings`` (the unit's stages), ``unit_total``, ``instance_timings`` (the
harness domain build and the instance record, spec §8.1: charged to the
methods that consume the domain), ``memory``, ``card_sha256``,
``provenance``, ``smoke``, ``opts``, ``diag`` (arrays as ``{"__npz__":
member}``), ``arrays`` and ``saved_utc``.
"""
from __future__ import annotations

import contextlib
import csv
import errno
import gc
import glob
import hashlib
import io
import json
import math
import os
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from experiments.benchmark import config, product, runner, settings, timing
from experiments.benchmark import instance as inst_mod
from experiments.benchmark.instance import InstanceUnavailable, leaf_name, norm_row

PRODUCT_FORMAT = "v2-lag-product-1"
LEDGER_FORMAT = "v2-unit-status-1"
RAM_FORMAT = "v2-ram-ledger-1"
DONE_STATUSES = ("ok", "empty", "not_computable")
NOT_FINAL = ("error", "budget", "hung")
#: elevation products of E0 (canonical ElevResult)
ELEV_PRODUCTS = ("E0a", "E0b", "E0g")
#: options that change a unit's results (they enter config_key, §13.2)
RESULT_OPTS = ("n_boot", "refits", "seed_offset", "threads", "smoke")
#: RAM ledger (§13.5)
RAM_FACTOR = 1.25
RAM_MARGIN_GB = 6.0
RAM_LOCK_STALE_S = 120.0
#: wall cap (§13.5): 3 x est_hours, with a floor (declared): the V2 est_hours
#: are the cost model's guesses (designs.py), not the pilot's measurements
WALL_CAP_FACTOR = 3.0
WALL_CAP_FLOOR_S = 6 * 3600.0
#: the states of status.csv (spec §13.3 ``status``)
STATES = ("done", "running", "stale", "hung", "failed", "not_computable", "not_applicable",
          "unavailable", "pending", "shared", "reuse")
STATUS_COLUMNS = ("design", "unit_id", "site", "window", "setting_id", "method", "variant",
                  "family", "mode", "N", "rep", "plan_status", "state", "reason",
                  "statuses", "instance_id", "est_hours", "note")
#: the code every unit's code key holds besides its adapter (plan INTERFACES §7)
CORE_CODE = ("experiments/benchmark/lags/common.py", "experiments/benchmark/instance.py",
             "experiments/benchmark/obs.py", "experiments/benchmark/domain.py",
             "experiments/benchmark/levels.py")
#: the queue root's thread record (one BLAS thread count per queue)
QUEUE_JSON = "queue.json"
#: the package files whose change makes a worker re-open the site's package
PACKAGE_STAMP_FILES = ("record.json", "levels.npz", "geometry.npz", "dom/current.json")


class QueueRefused(RuntimeError):
    """The queue refuses to start (e.g. another BLAS thread count than the
    queue root's)."""


def log(msg):
    """Print a line; a closed pipe (tee killed, pane closed) never kills the
    worker (the line is lost, the unit goes on)."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, ValueError):
        pass
    except OSError as e:                         # EPIPE / EINVAL on a dead pipe
        if e.errno not in (errno.EPIPE, errno.EINVAL):
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  plans and rows
# ─────────────────────────────────────────────────────────────────────────────

def plan_path(root, design):
    return os.path.join(root, "plans", f"{design}.csv")


def read_plan(root, design):
    """The rows of ``plans/<design>.csv`` as dicts (every status)."""
    p = plan_path(root, design)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing: run 'plan --design {design}' first")
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_row(design, site, window, setting, method, variant="", family="rand", mode="A",
             N="all", rep=0, status="planned", est_hours=0.0, est_ram_gb=1.0, note="",
             products=None):
    """An in-memory plan row (the positive control's queue, tests): the unit
    id and effective key exactly as ``designs.Unit`` computes them."""
    import hashlib

    from experiments.benchmark import designs
    m = designs.METHODS[method]
    ek = settings.effective_key(setting, window, m.signature)
    dedupe = (site, ek, method, variant, family, mode, str(N), int(rep))
    uid = hashlib.sha256(json.dumps(list(dedupe)).encode("utf-8")).hexdigest()[:16]
    if products is None:
        parts = variant.split("+")
        if method == "E0" and "gaugelag" in parts:
            products = ("E0g",)
        elif method == "E0" and "nodelay" in parts:
            products = ("E0a",)
        else:
            products = tuple(m.products)
    return {"design": design, "tier": "", "priority": 0, "unit_id": uid, "site": site,
            "window": window, "setting_id": setting, "method": method, "variant": variant,
            "family": family, "mode": mode, "N": str(N), "rep": int(rep), "kind": "all",
            "effective_key": ek, "status": status, "est_hours": float(est_hours),
            "est_ram_gb": float(est_ram_gb), "est_read_gb": 0.0, "cut_step": 0,
            "products": " ".join(products), "note": note}


def load_rows(root, plans, sites=None, windows=None, rows=None):
    """``(units, designs_of)``: the planned rows of ``plans`` (in order, each
    unit id once) and ``{unit_id: [designs]}``; ``rows`` replaces the plan
    files (in-memory queue)."""
    raw = []
    if rows is not None:
        raw = [dict(r) for r in rows]
    else:
        for d in plans:
            raw += [dict(r, design=r.get("design") or d) for r in read_plan(root, d)]
    designs_of = {}
    units = OrderedDict()
    for r in raw:
        n = norm_row(r)
        if sites and n["site"] not in sites:
            continue
        if windows and n["window"] not in windows:
            continue
        if n["status"] in ("planned", "shared"):
            designs_of.setdefault(n["unit_id"], [])
            if n["design"] not in designs_of[n["unit_id"]]:
                designs_of[n["unit_id"]].append(n["design"])
        if n["status"] != "planned" or n["unit_id"] in units:
            continue
        units[n["unit_id"]] = n
    return list(units.values()), designs_of


def group_instances(rows):
    """``OrderedDict{instance key: [rows]}`` in the order of first appearance."""
    out = OrderedDict()
    for r in rows:
        out.setdefault(inst_mod.instance_key(r), []).append(r)
    return out


def unit_of(row, designs_of=None):
    """The unit dict an adapter receives (plan INTERFACES §5); ``design`` is
    the first plan (in queue order) that holds the unit."""
    design = (designs_of or {}).get(row["unit_id"]) or [row["design"]]
    return {"method": row["method"], "variant": row["variant"],
            "products": tuple(row["products"]), "design": design[0],
            "unit_id": row["unit_id"], "site": row["site"], "window": row["window"],
            "setting": row["setting_id"], "family": row["family"], "mode": row["mode"],
            "N": row["N"], "rep": int(row["rep"])}


def unit_opts(row, threads=4, boot_threads=1, smoke=False):
    """``opts`` of a unit (plan INTERFACES §5): n_boot 30 for E0 'boot30',
    refits 20 for L08 'boot30' and for L08 'seed<k>' (GM20's only random
    stage is its refits: a seed row without them would duplicate the base
    row, spec §9.1 seed-variance floor), seed_offset k for 'seed<k>'."""
    parts = [p for p in str(row["variant"]).split("+") if p]
    seed = 0
    for p in parts:
        if p.startswith("seed") and p[4:].isdigit():
            seed = int(p[4:])
    return {"n_boot": 30 if (row["method"] == "E0" and "boot30" in parts) else 0,
            "refits": 20 if (row["method"] == "L08" and ("boot30" in parts or seed)) else 0,
            "seed_offset": seed, "threads": int(threads), "boot_threads": int(boot_threads),
            "smoke": bool(smoke)}


# ─────────────────────────────────────────────────────────────────────────────
#  keys
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_jsonable(v) for v in o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return repr(o)


def sha_json(obj):
    return config.sha256_json(_jsonable(obj))


def _rel(path):
    p = os.path.abspath(path)
    try:
        r = os.path.relpath(p, config.ROOT)
    except ValueError:                       # another drive
        return p.replace("\\", "/")
    return r.replace("\\", "/") if not r.startswith("..") else p.replace("\\", "/")


def _src_sha(path):
    p = path if os.path.isabs(path) else os.path.join(config.ROOT, path)
    try:
        return config.sha256_source(p)
    except OSError:
        return "missing"


def code_files_of(attrs):
    """``{repository path: sha256 of the source}`` of the adapter module, the
    core files and the module's CODE_DEPS (sorted)."""
    files = {}
    if attrs.get("file"):
        files[_rel(attrs["file"])] = _src_sha(attrs["file"])
    for p in CORE_CODE + tuple(attrs.get("CODE_DEPS") or ()):
        files[_rel(p if os.path.isabs(p) else os.path.join(config.ROOT, p))] = _src_sha(p)
    return dict(sorted(files.items()))


def code_files_key(attrs):
    """sha256 of :func:`code_files_of`: the code of a unit without the
    registry entries of its cell (window- and setting-free: the positive
    control's products and the real rows of one adapter share it)."""
    return sha_json(code_files_of(attrs))


def code_key_of(method, attrs, row):
    """sha256 of the adapter module, the core files, CODE_DEPS and the
    registry entries of the unit's effective key."""
    from experiments.benchmark import designs
    files = code_files_of(attrs)
    entries = {}
    if method in designs.METHODS:
        try:
            entries = settings.code_shas(row["setting_id"], row["window"],
                                         designs.METHODS[method].signature)
        except Exception as e:               # an unknown level: recorded, never fatal
            entries = {"error": f"{type(e).__name__}: {e}"}
    return sha_json({"files": files, "entries": entries})


def unit_keys(inst, row, uopts, attrs=None):
    """The keys of one unit (plan INTERFACES §7). ``attrs`` default
    ``lags.module_attrs(method)`` (raises ``lags.ModuleNotBuilt``)."""
    from experiments.benchmark import designs, lags
    attrs = attrs or lags.module_attrs(row["method"])
    cards = {p: (attrs["CARDS"] or {}).get(p) for p in row["products"]}
    cfg = {"CONFIG": attrs["CONFIG"], "CARDS": cards, "variant": row["variant"],
           "opts": {k: uopts[k] for k in RESULT_OPTS}}
    config_key = sha_json(cfg)
    code_key = code_key_of(row["method"], attrs, row)
    ext = {}
    fn = attrs.get("external_inputs")
    if callable(fn):
        try:
            ext = dict(fn(inst, unit_of(row)) or {})
        except InstanceUnavailable:
            raise
        except Exception as e:                 # a missing input: the adapter decides
            ext = {"__error__": f"{type(e).__name__}: {e}"}
    ek = row.get("effective_key") or ""
    if not ek and row["method"] in designs.METHODS:
        ek = settings.effective_key(row["setting_id"], row["window"],
                                    designs.METHODS[row["method"]].signature)
    keys = {"package_keys": inst.package_keys, "subset_key": inst.subset_key,
            "effective_key": ek, "config_key": config_key, "code_key": code_key}
    keys["inputs_key"] = sha_json({**keys, "method": row["method"], "variant": row["variant"],
                                   "external_inputs": ext})
    # recorded beside the keys (the code key already covers it): the printing
    # rule compares the scored product's with the positive control's
    keys["code_files_key"] = code_files_key(attrs)
    keys["external_inputs"] = ext
    return keys


# ─────────────────────────────────────────────────────────────────────────────
#  folders, done check, status ledger
# ─────────────────────────────────────────────────────────────────────────────

def product_dir(root, unit, prod):
    """``<root>/runs/<site>/<window>/<setting_id>/<product>[__<variant>]/<family>-<mode>/<leaf>``."""
    p = f"{prod}__{unit['variant']}" if unit["variant"] else str(prod)
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "runs", unit["site"],
                     unit["window"], unit["setting"], p, f"{unit['family']}-{unit['mode']}",
                     leaf_name(unit["N"], unit["rep"])])


def read_run(d):
    p = os.path.join(d, "run.json")
    if not os.path.exists(p):
        return None
    try:
        return product.read_json(p)
    except (OSError, ValueError):
        return None


def npz_readable(path, members=None):
    """True when ``path`` is a readable zip (``np.savez``) holding
    ``members`` (a product.npz truncated by a power loss reads False)."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except (OSError, zipfile.BadZipFile, ValueError):
        return False
    if members:
        want = {f"{m}.npy" for m in members}
        return want <= names
    return True


def unit_done(root, unit, keys, smoke=False):
    """Every product folder final, with this inputs_key and smoke flag, and
    its product.npz readable with the arrays run.json lists."""
    for p in unit["products"]:
        d = product_dir(root, unit, p)
        run = read_run(d)
        if (run is None or run.get("status") not in DONE_STATUSES
                or (run.get("keys") or {}).get("inputs_key") != keys["inputs_key"]
                or bool(run.get("smoke", False)) != bool(smoke)):
            return False
        if not npz_readable(os.path.join(d, "product.npz"), run.get("arrays")):
            return False
    return True


def ledger_path(root, unit_id):
    return os.path.join(root, "claims", "status", f"{unit_id}.json")


#: the monitor thread (hung / budget flags) and the main thread write the
#: same ledger file: one writer at a time in this process
_LEDGER_LOCK = threading.Lock()


def write_ledger(root, unit_id, status, reason=None, worker=None, **extra):
    """``claims/status/<unit_id>.json`` = {status, reason, utc, worker, ...}."""
    rec = {"format": LEDGER_FORMAT, "status": status, "reason": reason,
           "utc": product.utc_stamp(), "worker": worker or _worker()}
    rec.update(_jsonable(extra))
    try:
        with _LEDGER_LOCK:
            product.write_json(ledger_path(root, unit_id), rec)
    except OSError as e:                     # the ledger must never stop the queue
        log(f"[queue] could not write the status ledger of {unit_id}: {e}")
    return rec


def read_ledger(root, unit_id):
    p = ledger_path(root, unit_id)
    if not os.path.exists(p):
        return None
    try:
        return product.read_json(p)
    except (OSError, ValueError):
        return None


_WORKER_ID = {"id": None}


def _worker():
    return {"host": socket.gethostname(), "pid": os.getpid(), "worker_id": _WORKER_ID["id"]}


def methods_tag(methods):
    """``__m<sha8>`` of a ``--methods`` restriction ('' without one)."""
    if not methods:
        return ""
    h = hashlib.sha256(",".join(sorted(str(m) for m in methods)).encode("utf-8"))
    return f"__m{h.hexdigest()[:8]}"


def claim_path(root, site, iid, methods=None):
    """The instance claim (a ``--methods`` worker's carries its tag, so it
    never blocks the instance for the other workers; the unit claims keep two
    workers off the same unit)."""
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "claims", site,
                     f"{iid}{methods_tag(methods)}.claim"])


def unit_claim_path(root, site, iid, unit_id):
    """``claims/<site>/<instance_id>/<unit_id>.claim``: held while the unit runs."""
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "claims", site, str(iid),
                     f"{unit_id}.claim"])


def instance_claims(root, site, iid, stale_after_s=runner.STALE_AFTER_S):
    """``runner.claim_state`` of every instance claim of ``iid`` (the plain
    one and those of ``--methods`` workers)."""
    base = claim_path(root, site, iid)[:-len(".claim")]
    out = []
    for p in sorted(set(glob.glob(base + ".claim")) | set(glob.glob(base + "__m*.claim"))):
        st = runner.claim_state(p, stale_after_s)
        if st is not None:
            out.append(st)
    return out


def take_dead_claim(path, info):
    """Take over a claim whose process is GONE on this host (killed by
    SIGKILL or the OOM killer, a reboot): under the claim's steal lock, the
    state is read again, the dead claim file (a lock file, not data) removed
    and the claim created exclusively. False when the lock is busy or the
    claimant turned out alive or unknown."""
    lock = f"{path}.steal"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            if time.time() - os.stat(lock).st_mtime > runner.STEAL_LOCK_STALE_S:
                os.remove(lock)               # left by a crashed stealer: next pass
        except OSError:
            pass
        return False
    os.close(fd)
    try:
        st = runner.claim_state(path)
        if st is None:                        # released meanwhile: an ordinary claim
            return runner.claim(path, info)
        if st.get("alive") is not False:
            return False
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return runner.claim(path, info)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def acquire(path, info, opts, label=""):
    """``(held, state)``: the claim ``path`` taken (``runner.claim`` with the
    queue's stale rule), or taken over at once when its process is gone on
    this host; ``state`` is the other claimant's ``runner.claim_state``."""
    if runner.claim(path, info, steal_stale=opts.steal_stale, stale_after_s=opts.stale_after_s):
        return True, None
    st = runner.claim_state(path, opts.stale_after_s)
    if st is not None and st.get("alive") is False:
        log(f"[run] {label}: the claim of {st.get('host')}:{st.get('pid')} belongs to a process "
            f"that is gone (last heartbeat {st['age_s'] / 60:.0f} min ago): taken over")
        if take_dead_claim(path, info):
            return True, st
        st = runner.claim_state(path, opts.stale_after_s)
    return False, st


def stop_file(root, worker_id=None):
    """The STOP file that applies to this worker, or None."""
    names = ["STOP"] + ([f"STOP-{worker_id}"] if worker_id else [])
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  the RAM ledger (§13.5)
# ─────────────────────────────────────────────────────────────────────────────

def _ram_paths(root):
    d = os.path.join(root, "claims")
    return os.path.join(d, "ram_ledger.json"), os.path.join(d, "ram_ledger.lock")


@contextlib.contextmanager
def _ram_lock(root, timeout_s=120.0):
    _, lock = _ram_paths(root)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, json.dumps(_worker()).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.stat(lock).st_mtime > RAM_LOCK_STALE_S:
                    os.remove(lock)                 # left by a crashed worker
                    continue
            except OSError:
                continue
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"{lock} held for more than {timeout_s:.0f} s")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _read_ram(path):
    if not os.path.exists(path):
        return {"format": RAM_FORMAT, "entries": {}}
    try:
        doc = product.read_json(path)
        doc.setdefault("entries", {})
        return doc
    except (OSError, ValueError):
        return {"format": RAM_FORMAT, "entries": {}}


def _alive(entry):
    if entry.get("host") != socket.gethostname():
        return True                                 # another host: kept
    pid = entry.get("pid")
    if pid is None:
        return True
    try:
        import psutil
        if not psutil.pid_exists(int(pid)):
            return False
        ct = entry.get("create_time")
        if ct is not None and abs(psutil.Process(int(pid)).create_time() - float(ct)) > 1.0:
            return False
        return True
    except Exception:
        return True


def physical_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2 ** 30
    except Exception:
        return float("nan")


def ram_status(root):
    """The current reservations (dead processes dropped)."""
    path, _ = _ram_paths(root)
    doc = _read_ram(path)
    return {k: v for k, v in doc["entries"].items() if _alive(v)}


def ram_reserve(root, gb, label="", total_gb=None, margin_gb=RAM_MARGIN_GB, worker_id=None):
    """Reserve ``gb``; returns a token, or None when the reservations plus the
    margin would exceed the physical RAM (``total_gb`` overrides it)."""
    path, _ = _ram_paths(root)
    total = physical_ram_gb() if total_gb is None else float(total_gb)
    with _ram_lock(root):
        doc = _read_ram(path)
        live = {k: v for k, v in doc["entries"].items() if _alive(v)}
        used = sum(float(v.get("gb", 0.0)) for v in live.values())
        if math.isfinite(total) and used + float(gb) + float(margin_gb) > total:
            return None
        tok = uuid.uuid4().hex
        try:
            import psutil
            ct = psutil.Process().create_time()
        except Exception:
            ct = None
        live[tok] = {"gb": round(float(gb), 3), "label": label, "host": socket.gethostname(),
                     "pid": os.getpid(), "create_time": ct, "worker_id": worker_id,
                     "utc": product.utc_stamp()}
        product.write_json(path, {"format": RAM_FORMAT, "entries": live,
                                  "total_gb": total, "margin_gb": margin_gb})
        return tok


def ram_release(root, token):
    if token is None:
        return
    path, _ = _ram_paths(root)
    with _ram_lock(root):
        doc = _read_ram(path)
        if token in doc["entries"]:
            del doc["entries"][token]
            product.write_json(path, doc)


# ─────────────────────────────────────────────────────────────────────────────
#  heartbeat
# ─────────────────────────────────────────────────────────────────────────────

class _Heartbeat:
    """Touches the instance claim every ``every_s``; :meth:`pause` stops it
    (a hung unit: the claim goes stale and can be stolen)."""

    def __init__(self, path, every_s=runner.HEARTBEAT_S):
        self.path = path
        self.every_s = float(every_s)
        self.paused = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.beats = 0
        self.lost = None
        #: the claim of the unit running now (touched with the instance claim)
        self.unit_path = None

    def _loop(self):
        while not self._stop.wait(self.every_s):
            if self.paused.is_set():
                continue
            try:
                runner.heartbeat(self.path)
                self.beats += 1
            except runner.ClaimLost as e:
                if self.lost is None:
                    self.lost = str(e)
                    log(f"[queue] WARNING: {e}")
            except OSError as e:
                log(f"[queue] heartbeat failed: {e}")
            up = self.unit_path
            if up:
                try:
                    runner.heartbeat(up)
                except (runner.ClaimLost, OSError) as e:
                    log(f"[queue] unit claim heartbeat: {e}")

    def pause(self):
        self.paused.set()

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="v2-heartbeat")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  results: validation and publishing
# ─────────────────────────────────────────────────────────────────────────────

def _card(prod):
    from experiments.benchmark import lags
    try:
        return lags.card(prod)
    except Exception:
        return None


def _status_result(prod, status, stage=None, reason=None, diag=None):
    """A result without payload (not_computable, error, budget, hung)."""
    c = _card(prod) or {}
    elev = prod in ELEV_PRODUCTS or c.get("kind") == "elev"
    out = {"product": prod, "status": status, "stage": stage, "reason": reason,
           "kind": "elev" if elev else c.get("output"),
           "lag_kind": None if elev else (c.get("lag_kind") or "linear"),
           "field": None, "band": None, "point": None, "groups": None, "interval95": None,
           "edge_share": float("nan"), "n_defined": 0, "diag": dict(diag or {})}
    if out["kind"] not in (None, "field", "band", "point", "groups", "elev"):
        out["kind"] = None
    return out


def validate_results(out, unit, inst):
    """Check an adapter's list against the unit's products and canonicalise
    every dict (``common.lag_result`` / ``common.elev_result``)."""
    from experiments.benchmark.lags import common
    prods = list(unit["products"])
    if not isinstance(out, (list, tuple)):
        raise TypeError(f"{unit['method']} returned {type(out).__name__}, not a list of dicts")
    by = {}
    for d in out:
        if not isinstance(d, dict) or "product" not in d:
            raise TypeError(f"{unit['method']} returned an item that is not a result dict")
        if d["product"] in by:
            raise ValueError(f"{unit['method']} returned product {d['product']} twice")
        by[d["product"]] = d
    if sorted(by) != sorted(prods):
        raise ValueError(f"{unit['method']} returned products {sorted(by)}, expected {prods}")
    res = []
    for p in prods:
        d = dict(by[p])
        if d.get("kind") == "elev" or p in ELEV_PRODUCTS:
            d.pop("lag_kind", None)
            n_keep = inst.domain.n_keep if any(d.get(k) is not None for k in
                                               ("z", "sigma", "clear_obs", "tau_px")) else None
            d.pop("kind", None)
            res.append(common.elev_result(**d, n_keep=n_keep))
            continue
        need = any(d.get(k) is not None for k in ("field", "band", "groups", "interval95"))
        n_keep = K = None
        if need:
            dom = inst.domain
            n_keep, K = dom.n_keep, dom.K
        if (d.get("status", "ok") == "ok" and d.get("kind") == "band"
                and d.get("field") is None and isinstance(d.get("band"), dict)
                and d["band"].get("tau") is not None):
            d["field"] = common.broadcast_band(d["band"]["tau"], inst.domain.band_of)
        res.append(common.lag_result(**d, n_keep=n_keep, K=K))
    return res


def _split(obj, prefix, arrays):
    """JSON-ready copy with ndarrays moved to ``arrays`` as ``<prefix>.<i>``."""
    if isinstance(obj, np.ndarray):
        name = f"{prefix}.{sum(1 for k in arrays if k.startswith(prefix + '.'))}"
        arrays[name] = obj
        return {"__npz__": name}
    if isinstance(obj, dict):
        return {str(k): _split(v, prefix, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split(v, prefix, arrays) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _join(obj, arrays):
    if isinstance(obj, dict):
        if set(obj) == {"__npz__"}:
            return arrays[obj["__npz__"]]
        return {k: _join(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_join(v, arrays) for v in obj]
    return obj


def result_arrays(res):
    """``(arrays for product.npz, run.json outcome fields)`` of one result."""
    arrays = {}
    if res.get("kind") == "elev":
        for k in ("z", "sigma", "clear_obs", "tau_px"):
            if res.get(k) is not None:
                arrays[k] = np.asarray(res[k])
    if res.get("field") is not None:
        arrays["field"] = np.asarray(res["field"], np.float32)
    b = res.get("band")
    if b is not None:
        arrays["band_tau"] = np.asarray(b["tau"], np.float64)
        arrays["band_n_px"] = np.asarray(b["n_px"], np.int64)
        arrays["band_edge"] = np.asarray(b["edge"], bool)
        if b.get("lo") is not None:
            arrays["band_lo"] = np.asarray(b["lo"], np.float64)
        if b.get("hi") is not None:
            arrays["band_hi"] = np.asarray(b["hi"], np.float64)
    iv = res.get("interval95")
    if iv is not None:
        arrays["iv_lo"] = np.asarray(iv["lo"], np.float64)
        arrays["iv_hi"] = np.asarray(iv["hi"], np.float64)
    groups = None
    if res.get("groups") is not None:
        groups = {}
        for g, e in res["groups"].items():
            groups[g] = _jsonable({k: v for k, v in e.items()})
    outcome = {"status": res["status"], "stage": res.get("stage"), "reason": res.get("reason"),
               "kind": res.get("kind"), "lag_kind": res.get("lag_kind"),
               "edge_share": float(res.get("edge_share", float("nan"))),
               "n_defined": int(res.get("n_defined", 0) or 0),
               "point": None if res.get("point") is None else _jsonable(res["point"]),
               "groups": groups,
               "interval95": None if iv is None else {"level": iv["level"],
                                                      "nominal": iv["nominal"]}}
    diag = _split(res.get("diag") or {}, "diag", arrays)
    return arrays, outcome, diag


def _fsync_path(path, directory=False):
    """fsync a file (or, on POSIX, a folder); never fatal."""
    if directory and os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish(out_dir, arrays, run, log_text=None):
    """Atomic publish of one product folder (§13.5): written in
    ``<dir>.tmp-<host>-<pid>-<thread>``, its files and the folder fsynced,
    a previous folder kept as ``<dir>.old-<utc>``, ``os.replace``, then the
    parent fsynced (a power loss never leaves a run.json beside a truncated
    product.npz)."""
    out_dir = os.path.normpath(os.fspath(out_dir))
    parent = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{out_dir}.tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)                           # our own leftover only
    os.makedirs(tmp)
    names = ["product.npz", "run.json"]
    product.save_npz(os.path.join(tmp, "product.npz"), arrays, compress=True)
    product.write_json(os.path.join(tmp, "run.json"), run)
    if log_text is not None:
        product.write_text(os.path.join(tmp, "unit.log"), log_text)
        names.append("unit.log")
    for n in names:
        _fsync_path(os.path.join(tmp, n))
    _fsync_path(tmp, directory=True)
    if os.path.exists(out_dir):
        old = f"{out_dir}.old-{product.utc_stamp()}"
        os.replace(out_dir, old)
        log(f"[queue] previous attempt kept as {old}")
    os.replace(tmp, out_dir)
    _fsync_path(parent, directory=True)
    return out_dir.replace("\\", "/")


def recover_tmp_folders(root, host=None):
    """Move (never delete) this host's ``runs/**/*.tmp-<host>-<pid>[-<thread>]``
    folders whose process is gone to ``<folder>.old-<utc>`` (a worker killed
    while publishing); returns the new paths."""
    host = host or socket.gethostname()
    out = []
    base = os.path.join(root, "runs")
    if not os.path.isdir(base):
        return out
    marker = f".tmp-{host}-"
    for d in sorted(glob.glob(os.path.join(base, "**", f"*{marker}*"), recursive=True)):
        name = os.path.basename(d)
        if not os.path.isdir(d) or ".old-" in name:
            continue
        tail = name.rsplit(marker, 1)[-1]
        pid = tail.split("-", 1)[0]
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        if runner._pid_alive(int(pid)) is not False:
            continue
        new = f"{d}.old-{product.utc_stamp()}"
        try:
            os.replace(d, new)
        except OSError as e:
            log(f"[run] could not move the leftover {d}: {e}")
            continue
        log(f"[run] leftover of a dead worker moved: {d} -> {new}")
        out.append(new.replace("\\", "/"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the queue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueueOptions:
    """Options of :func:`run` (CLI ``run``; the positive control's queue)."""

    root: str = "results/validation_v2"
    plans: tuple = ("lag",)
    worker_id: str = "1"
    threads: int = 4
    boot_threads: int = 1
    steal_stale: bool = False
    sites: Optional[tuple] = None
    windows: Optional[tuple] = None
    smoke: bool = False
    stale_after_s: float = runner.STALE_AFTER_S
    heartbeat_s: float = runner.HEARTBEAT_S
    progress_timeout_s: float = timing.PROGRESS_TIMEOUT_S
    poll_s: float = timing.POLL_S
    wall_cap: bool = True
    wall_cap_floor_s: float = WALL_CAP_FLOOR_S
    ram_total_gb: Optional[float] = None
    ram_margin_gb: float = RAM_MARGIN_GB
    passes: int = 1
    cubes_root: Optional[str] = None
    rows: Optional[list] = None
    site2_for: Optional[Callable] = None
    provider_for: Optional[Callable] = None
    methods: Optional[tuple] = None
    #: the grid of the packages (10: the Zarr packages; 20: the local legacy
    #: 20 m netCDF packages, ``package.v2_site(res=20)``)
    res: int = 10
    #: accept another BLAS thread count than the queue root's (claims/queue.json)
    threads_change: bool = False
    #: Linux: write this to /proc/self/oom_score_adj at start (None: untouched;
    #: the CLI's workers pass 1000, spec §13.5)
    oom_score_adj: Optional[int] = None


def package_stamps(site2):
    """``{relative path: mtime_ns or None}`` of the package files whose
    change makes a worker re-open the package (:data:`PACKAGE_STAMP_FILES`,
    the series and marks, every shard manifest)."""
    d = getattr(site2, "dir", None)
    if not d:
        return {}
    names = list(PACKAGE_STAMP_FILES)
    try:
        from experiments.benchmark import package
        names += [package.SERIES_V2, package.MARKS_V2]
    except (ImportError, AttributeError):
        pass
    out = {}
    for n in names:
        p = os.path.join(d, n)
        try:
            out[n] = os.stat(p).st_mtime_ns
        except OSError:
            out[n] = None
    for p in sorted(glob.glob(os.path.join(d, "**", "manifest*.json"), recursive=True)):
        try:
            out[os.path.relpath(p, d).replace("\\", "/")] = os.stat(p).st_mtime_ns
        except OSError:
            continue
    return out


class _State:
    """Per-run caches: the packages, provenance. A package is re-opened (and
    a failed open retried) when :func:`package_stamps` changed."""

    def __init__(self, opts):
        self.opts = opts
        self._pkgs = {}
        self._stamps = {}
        self._git = None
        self._runtime = None

    def _site2(self, site):
        from experiments.benchmark import package
        o = self.opts
        return (o.site2_for(site) if o.site2_for is not None
                else package.v2_site(site, root=o.root, cubes_root=o.cubes_root,
                                     res=int(o.res or 10)))

    def package(self, site):
        if site in self._pkgs:
            site2, pkg, _ = self._pkgs[site]
            if package_stamps(site2) != self._stamps.get(site):
                log(f"[run] {site}: the package changed on disk "
                    f"({'not readable before' if pkg is None else 're-opened'})")
                if pkg is not None:
                    try:
                        pkg.close()
                    except Exception:
                        pass
                del self._pkgs[site]
        if site not in self._pkgs:
            from experiments.benchmark import package
            o = self.opts
            site2 = self._site2(site)
            self._stamps[site] = package_stamps(site2)
            try:
                pkg = package.V2Package(site2)
            except Exception as e:
                self._pkgs[site] = (site2, None, f"{type(e).__name__}: {e}")
            else:
                prov = o.provider_for(site) if o.provider_for is not None else None
                if prov is not None:
                    pkg._provider = prov
                self._pkgs[site] = (site2, pkg, None)
        return self._pkgs[site]

    def provider(self, site):
        o = self.opts
        return o.provider_for(site) if o.provider_for is not None else None

    def git(self):
        if self._git is None:
            self._git = product.git_state()
        return self._git

    def runtime(self):
        if self._runtime is None:
            self._runtime = product.runtime_provenance()
        return self._runtime

    def close(self):
        for _, pkg, _ in self._pkgs.values():
            if pkg is not None:
                try:
                    pkg.close()
                except Exception:
                    pass
        self._pkgs.clear()


def _wall_cap_s(row, opts):
    if not opts.wall_cap:
        return None
    return max(WALL_CAP_FACTOR * float(row.get("est_hours") or 0.0) * 3600.0,
               float(opts.wall_cap_floor_s))


def queue_json_path(root):
    return os.path.join(root, "claims", QUEUE_JSON)


def check_threads(root, threads, allow_change=False):
    """One BLAS thread count per queue root (it enters every config key):
    the first run writes ``claims/queue.json``; another count is refused
    (:class:`QueueRefused`) unless ``allow_change`` (then recorded)."""
    p = queue_json_path(root)
    doc = None
    if os.path.exists(p):
        try:
            doc = product.read_json(p)
        except (OSError, ValueError):
            doc = None
    have = None if not isinstance(doc, dict) else doc.get("threads")
    if have is not None and int(have) != int(threads) and not allow_change:
        raise QueueRefused(
            f"this queue root runs with --threads {have} ({p}); --threads {threads} would "
            f"enter every config key and recompute the finished units: pass --threads {have}, "
            "or --threads-change to switch the whole queue deliberately")
    if have is None or int(have) != int(threads):
        hist = list((doc or {}).get("history") or [])
        if have is not None:
            hist.append({"threads": int(have), "until_utc": product.utc_stamp()})
        product.write_json(p, {"format": "v2-queue-1", "threads": int(threads),
                               "utc": product.utc_stamp(), "worker": _worker(),
                               "history": hist})
        if have is not None:
            log(f"[run] the queue's BLAS threads change from {have} to {threads} "
                "(--threads-change): finished units rerun")
    return int(threads)


def set_oom_score_adj(value):
    """Linux: raise this process's ``oom_score_adj`` (an unprivileged process
    may raise it); returns the value written or None."""
    if value is None or not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as f:
            f.write(str(int(value)))
        return int(value)
    except OSError:
        return None


def run(opts):
    """Run the queue: ``opts.passes`` passes over the planned units.

    Returns
    -------
    dict
        Outcome counts (``ran``, ``done``, ``claimed``, ``unavailable``,
        ``not_built``, ``ram_refused``, ``stopped``, ``status:<s>``, ...).

    Raises
    ------
    QueueRefused
        Another BLAS thread count than the queue root's (plans only; the
        in-memory queues of the positive control and the tests are exempt).
    """
    _WORKER_ID["id"] = opts.worker_id
    if opts.rows is None:
        check_threads(opts.root, opts.threads, opts.threads_change)
    adj = set_oom_score_adj(opts.oom_score_adj)
    if adj is not None:
        log(f"[run] oom_score_adj = {adj} (a cheap worker is killed first, spec §13.5)")
    recover_tmp_folders(opts.root)
    rows, designs_of = load_rows(opts.root, opts.plans, opts.sites, opts.windows, opts.rows)
    if opts.methods:
        rows = [r for r in rows if r["method"] in opts.methods]
    groups = group_instances(rows)
    log(f"[run] {len(rows)} planned unit(s) in {len(groups)} instance(s) of "
        f"{','.join(opts.plans) if opts.rows is None else 'in-memory rows'}; worker "
        f"{opts.worker_id} on {socket.gethostname()} (pid {os.getpid()})")
    state = _State(opts)
    total = Counter()
    try:
        for k in range(max(1, int(opts.passes))):
            c = run_pass(groups, designs_of, opts, state)
            total.update(c)
            if c.get("stopped") or not c.get("ran"):
                break
    finally:
        state.close()
    log("[run] finished: " + ", ".join(f"{k} {v}" for k, v in sorted(total.items())))
    return dict(total)


def run_pass(groups, designs_of, opts, state):
    """One pass over the instances (in priority order)."""
    from experiments.benchmark import lags
    counts = Counter()
    told = set()
    for key, rows in groups.items():
        stop = stop_file(opts.root, opts.worker_id)
        if stop:
            log(f"[run] {stop} exists: stopping")
            counts["stopped"] += 1
            break
        site = key[0]
        site2, pkg, err = state.package(site)
        if pkg is None:
            for r in rows:
                write_ledger(opts.root, r["unit_id"], "unavailable",
                             f"package not readable: {err}")
            counts["unavailable"] += len(rows)
            if site not in told:
                told.add(site)
                log(f"[run] {site}: package not readable ({err}): its units are unavailable")
            continue
        label = f"{site} {key[1]} {key[2]} {key[3]}-{key[4]} {leaf_name(inst_mod.parse_N(key[5]), key[6])}"
        try:
            inst = inst_mod.build_instance(site2, rows[0], pkg=pkg,
                                           provider=state.provider(site))
        except InstanceUnavailable as e:
            for r in rows:
                write_ledger(opts.root, r["unit_id"], "unavailable", e.reason)
            counts["unavailable"] += len(rows)
            log(f"[run] {label}: UNAVAILABLE ({e.reason})")
            continue
        except Exception as e:                       # noqa: BLE001 -- recorded per unit
            tb = traceback.format_exc()
            for r in rows:
                write_ledger(opts.root, r["unit_id"], "error",
                             f"instance: {type(e).__name__}: {e}"[:2000], traceback=tb)
            counts["error"] += len(rows)
            log(f"[run] {label}: instance ERROR {type(e).__name__}: {e}")
            continue
        todo = []
        for r in rows:
            if r["method"] not in lags.REGISTRY:
                write_ledger(opts.root, r["unit_id"], "unavailable",
                             f"method {r['method']}: no lag adapter in V2 (later milestone)")
                counts["unavailable"] += 1
                continue
            uo = unit_opts(r, opts.threads, opts.boot_threads, opts.smoke)
            try:
                keys = unit_keys(inst, r, uo)
            except lags.ModuleNotBuilt as e:
                write_ledger(opts.root, r["unit_id"], "error", str(e))
                counts["not_built"] += 1
                if (r["method"], "nb") not in told:
                    told.add((r["method"], "nb"))
                    log(f"[run] {e}: its units fail until it lands")
                continue
            except InstanceUnavailable as e:
                write_ledger(opts.root, r["unit_id"], "unavailable", e.reason)
                counts["unavailable"] += 1
                continue
            except Exception as e:                   # noqa: BLE001
                write_ledger(opts.root, r["unit_id"], "error",
                             f"keys: {type(e).__name__}: {e}"[:2000],
                             traceback=traceback.format_exc())
                counts["error"] += 1
                continue
            unit = unit_of(r, designs_of)
            if unit_done(opts.root, unit, keys, opts.smoke):
                counts["done"] += 1
                continue
            todo.append((r, unit, uo, keys, list(designs_of.get(r["unit_id"]) or
                                                 [r["design"]])))
        if not todo:
            continue
        cpath = claim_path(opts.root, site, inst.instance_id, opts.methods)
        info = {"instance": label, "instance_id": inst.instance_id,
                "worker_id": opts.worker_id,
                "methods": list(opts.methods) if opts.methods else None}
        held, st = acquire(cpath, info, opts, label)
        if not held:
            counts["claimed"] += len(todo)
            if st is not None and st.get("stale"):
                log(f"[run] {label}: stale claim of {st.get('host')}:{st.get('pid')} "
                    f"({st['age_s'] / 60:.0f} min); use --steal-stale")
            continue
        tok = None
        try:
            need = max(float(r.get("est_ram_gb") or 0.0) for r, *_ in todo) * RAM_FACTOR
            tok = ram_reserve(opts.root, need, label=label, total_gb=opts.ram_total_gb,
                              margin_gb=opts.ram_margin_gb, worker_id=opts.worker_id)
            if tok is None:
                counts["ram_refused"] += len(todo)
                log(f"[run] {label}: RAM ledger refuses {need:.2f} GB (reservations "
                    f"{sum(v['gb'] for v in ram_status(opts.root).values()):.2f} GB + margin "
                    f"{opts.ram_margin_gb} GB > physical RAM): later")
                continue
            with _Heartbeat(cpath, opts.heartbeat_s) as hb:
                c = _run_instance(inst, todo, opts, state, hb, label)
            counts.update(c)
        finally:
            ram_release(opts.root, tok)
            runner.release(cpath)
        inst.cache.clear()
        inst._lazy.clear()
        pkg.__dict__.get("_domain_cache", {}).clear()     # the on-disk cache stays
    return counts


def _run_instance(inst, todo, opts, state, hb, label):
    counts = Counter()
    log(f"[run] === instance {label} ({inst.instance_id}): {len(inst.sel)} scenes, "
        f"{len(todo)} unit(s) to run")
    isw = timing.Stopwatch(f"instance {label}", progress_timeout_s=float("inf"))
    try:
        with isw.running():
            with isw.stage("domain.build"):
                inst.domain
            with isw.stage("write.instance"):
                inst_dir, inst_sha = inst_mod.write_instance_npz(opts.root, inst)
    except InstanceUnavailable as e:
        for r, *_ in todo:
            write_ledger(opts.root, r["unit_id"], "unavailable", e.reason)
        counts["unavailable"] += len(todo)
        log(f"[run] {label}: UNAVAILABLE ({e.reason})")
        return counts
    except Exception as e:                           # noqa: BLE001
        from experiments.benchmark import package
        if isinstance(e, package.PackageError):
            for r, *_ in todo:
                write_ledger(opts.root, r["unit_id"], "unavailable", f"domain: {e}")
            counts["unavailable"] += len(todo)
            log(f"[run] {label}: UNAVAILABLE (domain: {e})")
            return counts
        tb = traceback.format_exc()
        for r, *_ in todo:
            write_ledger(opts.root, r["unit_id"], "error",
                         f"domain: {type(e).__name__}: {e}"[:2000], traceback=tb)
        counts["error"] += len(todo)
        log(f"[run] {label}: domain ERROR {type(e).__name__}: {e}")
        return counts
    d = inst.domain
    # the harness components of the instance (spec §8.1: charged to the methods
    # that consume the domain); recorded in every product's run.json
    inst.cache["queue_v2.instance_timings"] = isw.records()
    log(f"[run] domain {d.status}: {d.n_keep:,} keep px, K = {d.K}; instance {inst_dir}")
    for r, unit, uo, keys, designs in todo:
        stop = stop_file(opts.root, opts.worker_id)
        if stop:
            log(f"[run] {stop} exists: stopping after the current unit")
            counts["stopped"] += 1
            break
        if unit_done(opts.root, unit, keys, opts.smoke):     # another worker meanwhile
            counts["done"] += 1
            continue
        upath = unit_claim_path(opts.root, unit["site"], inst.instance_id, r["unit_id"])
        held, ust = acquire(upath, {"unit_id": r["unit_id"], "instance_id": inst.instance_id,
                                    "worker_id": opts.worker_id}, opts, label)
        if not held:
            counts["claimed"] += 1
            log(f"[run] {label} {r['method']}: the unit is claimed by "
                f"{(ust or {}).get('host')}:{(ust or {}).get('pid')}; skipped")
            continue
        try:
            if hb is not None:
                hb.unit_path = upath
            if unit_done(opts.root, unit, keys, opts.smoke):  # finished before our claim
                counts["done"] += 1
                continue
            st = run_one(inst, r, unit, uo, keys, opts, state, hb, inst_dir, inst_sha,
                         designs=designs)
        finally:
            if hb is not None:
                hb.unit_path = None
            runner.release(upath)
        counts["ran"] += 1
        counts[f"status:{st}"] += 1
        if hb is not None and hb.lost:
            log(f"[run] {label}: the claim was taken over ({hb.lost}): leaving the instance")
            counts["claim_lost"] += 1
            break
    return counts


def _flag_ledger(root, unit_id, status, reason):
    def cb(sw):
        write_ledger(root, unit_id, status, reason)
    return cb


def run_one(inst, row, unit, uopts, keys, opts, state, hb, inst_dir, inst_sha, designs=None):
    """Run, validate and publish one unit; returns its overall status."""
    from experiments.benchmark import lags
    from experiments.benchmark.product import NotComputable
    method = row["method"]
    prods = list(unit["products"])
    label = (f"{unit['site']} {unit['window']} {unit['setting']} {method}"
             f"{'__' + unit['variant'] if unit['variant'] else ''} {unit['family']}-"
             f"{unit['mode']} {leaf_name(unit['N'], unit['rep'])}")
    try:
        fn = lags.load(method)
    except ImportError as e:
        write_ledger(opts.root, row["unit_id"], "error", str(e))
        log(f"[unit] {label}: {e}")
        return "error"

    def on_hung(sw):
        if hb is not None:
            hb.pause()
        write_ledger(opts.root, row["unit_id"], "hung",
                     f"no progress for {opts.progress_timeout_s:.0f} s")
        log(f"[unit] {label}: HUNG (no progress for {opts.progress_timeout_s:.0f} s); the "
            "claim heartbeat stops")

    sw = timing.Stopwatch(label, progress_timeout_s=opts.progress_timeout_s,
                          wall_cap_s=_wall_cap_s(row, opts), poll_s=opts.poll_s,
                          on_hung=on_hung,
                          on_wall_cap=_flag_ledger(opts.root, row["unit_id"], "budget",
                                                   "wall cap (3 x est_hours) exceeded"))
    inst.stage = lambda name, _sw=sw, _m=method: _sw.stage(f"lag.{_m}.{name}")
    inst.progress = sw.progress
    ulog = runner.UnitLog(label)
    results, unavailable, blas = None, None, {}
    t_start = product.utc_stamp()
    # unmap the package's cached tiles (re-opened lazily): their file pages
    # would otherwise count in this unit's RSS peak (VmHWM includes RssFile)
    mm = getattr(getattr(inst, "pkg", None), "_mm", None)
    if isinstance(mm, dict) and mm:
        mm.clear()
        gc.collect()
    with ulog.capture():
        reset_ok = runner.reset_peak_rss()
        with runner.blas_limits(int(uopts["threads"])) as blas, sw.running():
            log(f"=== unit {label} ({row['unit_id']})")
            log(f"    {len(inst.sel)} scenes, subset {inst.subset_key[:16]}, products "
                f"{', '.join(prods)}; inputs {keys['inputs_key'][:16]}, code "
                f"{keys['code_key'][:16]}, config {keys['config_key'][:16]}; opts {uopts}")
            try:
                out = fn(inst, unit, dict(uopts))
                results = validate_results(out, unit, inst)
            except NotComputable as e:
                log(f"[unit] NOT COMPUTABLE at {e.stage}: {e.reason or ''}")
                results = [_status_result(p, "not_computable", e.stage, e.reason or e.stage,
                                          e.diag) for p in prods]
            except InstanceUnavailable as e:
                unavailable = e.reason
                log(f"[unit] UNAVAILABLE: {e.reason}")
            except Exception as e:                   # noqa: BLE001 -- recorded
                tb = traceback.format_exc()
                log(tb.rstrip())
                results = [_status_result(p, "error", getattr(e, "stage", None) or "run",
                                          f"{type(e).__name__}: {e}"[:2000],
                                          {"traceback": tb, "exception": type(e).__name__})
                           for p in prods]
                log(f"[unit] ERROR {type(e).__name__}: {str(e)[:300]}")
        peak = float(runner.peak_rss_mb())
    if unavailable is not None:
        write_ledger(opts.root, row["unit_id"], "unavailable", unavailable)
        return "unavailable"
    # a unit that returns is final with its own status; the monitor's flags
    # are recorded (declared amendment of spec §13.5, module docstring)
    mon = {"hung_detected": bool(sw.hung), "wall_capped": bool(sw.wall_capped),
           "progress_timeout_s": float(opts.progress_timeout_s),
           "wall_cap_s": _wall_cap_s(row, opts)}
    for res in results:
        res["diag"] = dict(res.get("diag") or {}, monitor=mon)
    if mon["hung_detected"] or mon["wall_capped"]:
        log(f"[unit] {label}: returned after the monitor flagged it "
            f"({'no progress for the hang timeout' if mon['hung_detected'] else ''}"
            f"{'; ' if mon['hung_detected'] and mon['wall_capped'] else ''}"
            f"{'wall cap exceeded' if mon['wall_capped'] else ''}): its result is kept")
    total = sw.total()
    timings = sw.records()
    text = ulog.text()
    designs = list(designs or [unit["design"]])
    if hb is not None:
        if not _still_ours(hb):
            write_ledger(opts.root, row["unit_id"], "error",
                         "the instance claim was taken over while the unit ran: its result "
                         "was not published")
            log(f"[unit] {label}: the claim was taken over; result NOT published")
            return "error"
        hb.paused.clear()                          # alive again: resume the heartbeat
    statuses = []
    try:
        for res in results:
            p = res["product"]
            arrays, outcome, diag = result_arrays(res)
            d = product_dir(opts.root, unit, p)
            card = _card(p)
            run_doc = {
                "format": PRODUCT_FORMAT, "product": p, "method": method,
                "variant": unit["variant"], "designs": designs,
                "unit_id": row["unit_id"], "site": unit["site"], "window": unit["window"],
                "setting": unit["setting"], "family": unit["family"], "mode": unit["mode"],
                "N": unit["N"], "rep": int(unit["rep"]),
                **outcome,
                "instance": {"path": os.path.relpath(inst_dir, d).replace("\\", "/"),
                             "instance_sha256": inst_sha, "instance_id": inst.instance_id},
                "keys": {k: v for k, v in keys.items() if k != "external_inputs"},
                "external_inputs": keys.get("external_inputs") or {},
                "timings": timings, "unit_total": total,
                "instance_timings": inst.cache.get("queue_v2.instance_timings") or {},
                "memory": {"peak_rss_mb": peak,
                           "peak_scope": "unit" if reset_ok else "process",
                           "peak_anon_mb": total.get("peak_anon_mb"),
                           "tiles_unmapped": True},
                "monitor": mon,
                "card_sha256": None if card is None else sha_json(card),
                "provenance": {"git": state.git(), "runtime": state.runtime(),
                               "host": socket.gethostname(),
                               "worker": {"host": socket.gethostname(), "pid": os.getpid(),
                                          "worker_id": opts.worker_id},
                               "threads": _jsonable(blas), "flags": sw.flags(),
                               "started_utc": t_start,
                               "claim": None if hb is None else {"beats": hb.beats,
                                                                 "lost": hb.lost}},
                "smoke": bool(opts.smoke), "opts": uopts, "diag": diag,
                "arrays": sorted(arrays), "saved_utc": product.utc_stamp()}
            publish(d, arrays, run_doc, text)
            statuses.append(res["status"])
    except Exception as e:                           # noqa: BLE001 -- a full disk, ...
        tb = traceback.format_exc()
        write_ledger(opts.root, row["unit_id"], "error",
                     f"publish: {type(e).__name__}: {e}"[:2000], traceback=tb)
        log(tb.rstrip())
        log(f"[unit] {label}: PUBLISH FAILED ({type(e).__name__}: {e}); the worker goes on")
        return "error"
    overall = _overall(statuses)
    write_ledger(opts.root, row["unit_id"], overall,
                 None if overall in DONE_STATUSES else results[0].get("reason"),
                 products={r["product"]: r["status"] for r in results},
                 inputs_key=keys["inputs_key"], monitor=mon)
    log(f"[unit] {label}: " + ", ".join(f"{r['product']}={r['status']}" for r in results)
        + f" in {total['wall_s']:.1f} s (cpu {total['cpu_self_s']:.1f} s, peak {peak:.0f} MB)")
    return overall


def _still_ours(hb):
    """True while the instance claim is still this worker's (a heartbeat)."""
    try:
        runner.heartbeat(hb.path)
        return True
    except runner.ClaimLost:
        hb.lost = hb.lost or "claim taken over"
        return False
    except OSError:
        return True


def _overall(statuses):
    for s in ("error", "hung", "budget"):
        if s in statuses:
            return s
    if statuses and all(s == "not_computable" for s in statuses):
        return "not_computable"
    if statuses and all(s == "empty" for s in statuses):
        return "empty"
    return "ok"


# ─────────────────────────────────────────────────────────────────────────────
#  reading products (D reads these)
# ─────────────────────────────────────────────────────────────────────────────

def load_lag_product(d):
    """run.json plus the arrays of one product folder.

    Returns
    -------
    dict
        Every run.json field (identity, outcome, keys, timings, provenance)
        plus ``dir``; ``field`` (n_keep,) float32 or None; ``band`` {tau,
        n_px, edge, lo, hi} or None; ``interval95`` {level, nominal, lo, hi}
        or None; ``z``, ``sigma``, ``clear_obs``, ``tau_px`` (E0 elevation
        products) or None; ``groups`` with each ``tau`` as float64 arrays;
        ``diag`` with its arrays joined; ``instance`` (the instance record,
        :func:`instance.read_instance`) or None.
    """
    run_doc = product.read_json(os.path.join(d, "run.json"))
    if run_doc.get("format") != PRODUCT_FORMAT:
        raise ValueError(f"{d}: not a {PRODUCT_FORMAT} folder ({run_doc.get('format')!r})")
    z = product.load_npz(os.path.join(d, "product.npz"))
    out = dict(run_doc)
    out["dir"] = str(d).replace("\\", "/")
    out["field"] = z.get("field")
    if "band_tau" in z:
        out["band"] = {"tau": z["band_tau"], "n_px": z.get("band_n_px"),
                       "edge": z.get("band_edge"), "lo": z.get("band_lo"),
                       "hi": z.get("band_hi")}
    else:
        out["band"] = None
    iv = run_doc.get("interval95")
    out["interval95"] = (None if iv is None else
                         dict(iv, lo=z.get("iv_lo"), hi=z.get("iv_hi")))
    for k in ("z", "sigma", "clear_obs", "tau_px"):
        out[k] = z.get(k)
    if run_doc.get("groups") is not None:
        out["groups"] = {g: dict(e, tau=np.asarray(e.get("tau"), np.float64))
                         for g, e in run_doc["groups"].items()}
    out["diag"] = _join(run_doc.get("diag") or {}, z)
    ip = (run_doc.get("instance") or {}).get("path")
    out["instance"] = None
    if ip:
        idir = os.path.normpath(os.path.join(d, ip))
        if os.path.exists(os.path.join(idir, "instance.npz")):
            out["instance"] = inst_mod.read_instance(idir)
    return out


def iter_lag_products(root, site=None):
    """Yield ``(product folder, run.json dict)`` of every published lag
    product under ``<root>/runs[/<site>]`` (``.tmp-``/``.old-`` attempts and
    instance folders skipped)."""
    base = os.path.join(root, "runs", site) if site else os.path.join(root, "runs")
    for p in sorted(glob.glob(os.path.join(base, "**", "run.json"), recursive=True)):
        d = os.path.dirname(p)
        parts = d.replace("\\", "/").split("/")
        if any(".tmp-" in s or ".old-" in s for s in parts) or inst_mod.INSTANCE_DIR in parts:
            continue
        run_doc = read_run(d)
        if run_doc is not None and run_doc.get("format") == PRODUCT_FORMAT:
            yield d.replace("\\", "/"), run_doc


# ─────────────────────────────────────────────────────────────────────────────
#  status (spec §13.3 ``status``)
# ─────────────────────────────────────────────────────────────────────────────

def _plan_names(root):
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(root, "plans", "*.csv")))


def unit_state(root, row, stale_after_s=runner.STALE_AFTER_S):
    """``(state, reason, statuses)`` of one plan row (no key is computed:
    'done' means every product folder holds a final status)."""
    st = row["status"]
    if st == "not_applicable":
        return "not_applicable", row.get("note"), ""
    if st == "reuse":
        return "reuse", row.get("note"), ""
    if not row.get("products"):
        return "pending", "no products listed", ""
    unit = {"site": row["site"], "window": row["window"], "setting": row["setting_id"],
            "variant": row["variant"], "family": row["family"], "mode": row["mode"],
            "N": row["N"], "rep": row["rep"], "products": row["products"]}
    runs = [read_run(product_dir(root, unit, p)) for p in row["products"]]
    stats = [r.get("status") if r else None for r in runs]
    led = read_ledger(root, row["unit_id"])
    if runs and all(r is not None and r.get("status") in DONE_STATUSES for r in runs):
        if all(s == "not_computable" for s in stats):
            return "not_computable", runs[0].get("reason"), " ".join(map(str, stats))
        mon = runs[0].get("monitor") or {}
        why = "; ".join(w for w, f in (("hang detected while it ran", mon.get("hung_detected")),
                                        ("wall cap exceeded while it ran",
                                         mon.get("wall_capped"))) if f) or None
        return "done", why, " ".join(map(str, stats))
    try:
        key = inst_mod.instance_key(row)
    except Exception:
        key = None
    claims = []
    if key is not None:
        claims = instance_claims(root, row["site"], inst_mod.instance_id(key), stale_after_s)
    lst = (led or {}).get("status")
    if lst == "hung":
        return "hung", led.get("reason"), " ".join(map(str, stats))
    if claims:
        live = [c for c in claims if not c["stale"] and c.get("alive") is not False]
        if live:
            return "running", None, " ".join(map(str, stats))
        dead = any(c.get("alive") is False for c in claims)
        return "stale", ("the claimant's process is gone" if dead else None), \
            " ".join(map(str, stats))
    if lst == "unavailable":
        return "unavailable", led.get("reason"), " ".join(map(str, stats))
    if any(s in NOT_FINAL for s in stats if s) or lst in NOT_FINAL:
        why = (led or {}).get("reason") or next((r.get("reason") for r in runs
                                                 if r and r.get("status") in NOT_FINAL), None)
        return "failed", why, " ".join(map(str, stats))
    return "pending", None, " ".join(map(str, stats))


def status_rows(root, designs=None, sites=None, stale_after_s=runner.STALE_AFTER_S):
    """``{design: [rows of status.csv]}`` for the plans under ``<root>/plans``.

    A ``shared`` row reads the state of the planned row of the same unit id
    (its products live in that row's cell)."""
    out = {}
    names = list(designs or _plan_names(root))
    raws = {}
    planned = {}
    for d in names:
        try:
            raws[d] = [norm_row(dict(r0, design=r0.get("design") or d))
                       for r0 in read_plan(root, d)]
        except FileNotFoundError:
            continue
        for r in raws[d]:
            if r["status"] == "planned":
                planned.setdefault(r["unit_id"], r)
    for d, raw in raws.items():
        rows = []
        for r in raw:
            if sites and r["site"] not in sites:
                continue
            ref = planned.get(r["unit_id"], r) if r["status"] == "shared" else r
            if r["status"] == "shared" and ref is r:
                state, why, stats = "shared", r["note"], ""
            else:
                state, why, stats = unit_state(root, ref, stale_after_s)
            try:
                iid = inst_mod.instance_id(inst_mod.instance_key(r))
            except Exception:
                iid = ""
            rows.append({"design": d, "unit_id": r["unit_id"], "site": r["site"],
                         "window": r["window"], "setting_id": r["setting_id"],
                         "method": r["method"], "variant": r["variant"],
                         "family": r["family"], "mode": r["mode"], "N": r["N"],
                         "rep": r["rep"], "plan_status": r["status"], "state": state,
                         "reason": why, "statuses": stats, "instance_id": iid,
                         "est_hours": r["est_hours"], "note": r["note"]})
        out[d] = rows
    return out


def write_status(root, designs=None, sites=None):
    """``<root>/status/<design>.csv`` per plan (always every site of the
    plan: a call for a few sites never shrinks the file) and the counts per
    state of ``sites`` (default every site).

    Returns
    -------
    dict
        ``{design: {"path", "counts", "remaining_est_hours"}}``.
    """
    res = {}
    for d, rows in status_rows(root, designs, None).items():
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(STATUS_COLUMNS), lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        path = os.path.join(root, "status", f"{d}.csv")
        product.write_text(path, buf.getvalue())
        mine = [r for r in rows if not sites or r["site"] in sites]
        counts = Counter(r["state"] for r in mine)
        rem = sum(float(r["est_hours"] or 0.0) for r in mine
                  if r["state"] in ("pending", "failed", "unavailable", "stale", "hung"))
        res[d] = {"path": path.replace("\\", "/"), "counts": dict(counts),
                  "remaining_est_hours": round(rem, 3)}
    return res
