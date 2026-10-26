# -*- coding: utf-8 -*-
"""Validation v2 — stage stopwatch and dedicated timing runs (spec §8, §13.5).

Question
--------
What does each unit of the lag benchmark cost (wall, CPU of this process and
of its children, bytes read, peak memory) per named stage, when has a unit
stopped making progress, and what does each method cost on a quiet machine?

Pre-registered rules (spec §8.1, §8.2, §13.5; Appendix A ``timing``)
-------------------------------------------------------------------
* :class:`Stopwatch` records, per named stage (``io.*``, ``obs.*``,
  ``domain.*``, ``fit.*``, ``lag.*``, ``write.*``, so I/O separates from
  estimator time): ``wall_s`` (``time.perf_counter``), ``cpu_self_s``
  (``time.process_time``: user + system of every thread of this process),
  ``cpu_children_s`` (POSIX ``RUSAGE_CHILDREN`` delta, i.e. the children that
  exited and were waited for, plus the CPU of live children sampled with
  psutil by the monitor thread; on Windows only the psutil sampling, which
  is 0 for the numpy methods), ``read_mb`` (psutil ``io_counters`` read
  bytes delta, MiB; NaN where psutil cannot tell), ``peak_rss_mb`` (the
  largest RSS the monitor sampled during the stage, and at its start and
  end; the unit's own peak comes from ``runner.peak_rss_mb`` after
  ``runner.reset_peak_rss``), ``peak_anon_mb`` (the same for the anonymous
  memory, :func:`current_anon_mb`: Linux RssAnon, which leaves out the file
  pages of memory-mapped package tiles; the private bytes on Windows; the RSS
  elsewhere) and ``progress`` (the counter the method
  advances inside the stage). A repeated stage name accumulates (wall, CPU,
  read and progress summed, peak maximised, ``calls`` counted); nested stages
  are recorded separately, so an outer stage's wall time is at least the sum
  of its inner stages'.
* :meth:`Stopwatch.total` gives the whole-unit figures (``unit_total`` of
  run.json, spec §8.1: the runner no longer copies the unit total into every
  stage).
* Hang detection and wall cap (§13.5): a monitor thread sets ``hung`` when
  the progress counter (entering or leaving a stage also counts) has not
  moved for ``progress_timeout_s`` (30 min), and ``wall_capped`` when the
  unit has run longer than ``wall_cap_s`` (3 x est_hours, set by the queue).
  It never interrupts the computation (a Python thread cannot be killed): the
  queue's ledger reads 'hung' or 'budget' while the unit runs, and it stops
  the claim's heartbeat for a hung unit, so another worker can take the
  instance over; a unit that returns keeps its result with the flags
  recorded (queue_v2, declared amendment of §13.5).
* Dedicated timing runs (§8.2, Stage T): :func:`run_dedicated` refuses while
  any claim of the queue is active (a claim whose process is gone on this
  host does not count) or the POSIX one-minute load average is >= 1.5,
  writes each product's JSON as soon as its method has run every workload,
  records a failing method as 'error (...)' instead of aborting the site, and
  does not run at a site whose package lacks the W3 years
  (:class:`TimingUnavailable`, e.g. a partial-year site); per workload (N in
  {45, 180, all} at P0, W3, A-mode, rep 0: the subsets of the K1 curve set,
  ``WORKLOAD_DESIGN``; a workload that cannot be built is recorded with its
  error) and per
  configuration ("single": BLAS 1, only up to 10^5 domain pixels;
  "production": the declared threads) it runs one discarded warm-up and
  ``repeats`` (2) measured runs and writes
  ``<root>/timing/<site>/<product>.json`` labelled 'dedicated' with the
  median and the min-max. The page cache cannot be dropped without root:
  the warm-up is the cold proxy, the repeats are warm (§8.2).

This module imports only the standard library and numpy at import time.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import threading
import time

import numpy as np

TIMING_FORMAT = "v2-timing-1"
#: hang detection (spec §13.5): progress unchanged for 30 min
PROGRESS_TIMEOUT_S = 30 * 60.0
#: the monitor thread's period (RSS sampling and the flags)
POLL_S = 1.0
#: dedicated runs refuse above this POSIX one-minute load average (§8.2)
MAX_LOAD = 1.5
#: the single-thread configuration only up to this many domain pixels (§8.2)
SINGLE_THREAD_MAX_PX = 100_000
#: the standard workloads and protocol of §8.2 (Appendix A ``timing``)
WORKLOADS = (45, 180, "all")
REPEATS = 2
STAGE_PREFIXES = ("io.", "obs.", "domain.", "fit.", "lag.", "write.")
FIELDS = ("wall_s", "cpu_self_s", "cpu_children_s", "read_mb", "peak_rss_mb", "peak_anon_mb",
          "progress")


def log(msg):
    """Print a line; a closed pipe never kills the caller."""
    try:
        print(msg, flush=True)
    except (BrokenPipeError, ValueError):
        pass
    except OSError as e:
        import errno
        if e.errno not in (errno.EPIPE, errno.EINVAL):
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  process probes
# ─────────────────────────────────────────────────────────────────────────────

def _psutil_process():
    try:
        import psutil
        return psutil.Process()
    except Exception:                       # absent or unsupported
        return None


def _rss_mb(proc):
    if proc is not None:
        try:
            return proc.memory_info().rss / 2 ** 20
        except Exception:
            pass
    from experiments.benchmark import runner
    return float(runner.current_rss_mb())


def _proc_status_kb(field_name):
    with open("/proc/self/status", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith(field_name + ":"):
                return float(line.split()[1])
    raise OSError(f"{field_name} not in /proc/self/status")


def current_anon_mb(proc=None):
    """Anonymous memory of this process now (MB): Linux ``RssAnon`` (no file
    pages: the memory-mapped package tiles do not count), psutil's private
    bytes on Windows, the RSS elsewhere; NaN when unknown."""
    if sys.platform.startswith("linux"):
        try:
            return _proc_status_kb("RssAnon") / 1024.0
        except (OSError, ValueError):
            pass
    proc = proc if proc is not None else _psutil_process()
    if proc is not None:
        try:
            mi = proc.memory_info()
            priv = getattr(mi, "private", None)
            return float(priv if priv is not None else mi.rss) / 2 ** 20
        except Exception:
            pass
    return float("nan")


def _read_bytes(proc):
    """Bytes read by this process so far (psutil io_counters), or None."""
    if proc is None:
        return None
    try:
        return float(proc.io_counters().read_bytes)
    except Exception:                       # macOS, restricted /proc ...
        return None


def _rusage_children_s():
    """User + system CPU of the waited-for children (POSIX), else 0."""
    if os.name != "posix":
        return 0.0
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return float(r.ru_utime + r.ru_stime)
    except Exception:
        return 0.0


def _children_cpu(proc):
    """``{pid: user + system}`` of the live children (recursive), or {}."""
    if proc is None:
        return {}
    out = {}
    try:
        for c in proc.children(recursive=True):
            try:
                t = c.cpu_times()
                out[c.pid] = float(t.user + t.system)
            except Exception:
                continue
    except Exception:
        return {}
    return out


def load_average():
    """The POSIX one-minute load average, or None (Windows)."""
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  the stopwatch
# ─────────────────────────────────────────────────────────────────────────────

class _Open:
    """One running stage (or the unit total)."""

    __slots__ = ("name", "t0", "cpu0", "rch0", "kids0", "io0", "peak", "apeak", "prog0")

    def __init__(self, name, sw):
        self.name = name
        self.t0 = time.perf_counter()
        self.cpu0 = time.process_time()
        self.rch0 = _rusage_children_s()
        self.kids0 = dict(sw._kids_last)
        self.io0 = _read_bytes(sw._proc)
        self.peak = _rss_mb(sw._proc)
        self.apeak = current_anon_mb(sw._proc)
        self.prog0 = sw._progress


def _raise_peak(cur, v):
    """max(cur, v) where either may be NaN."""
    if np.isfinite(v) and not (np.isfinite(cur) and cur >= v):
        return v
    return cur


class Stopwatch:
    """Per-stage wall, CPU, children CPU, read volume, peak RSS and progress
    of one unit, with a monitor thread for hang detection and the wall cap.

    Parameters
    ----------
    label : str
        The unit (for the log lines).
    progress_timeout_s : float
        Seconds without progress after which ``hung`` is set (spec: 30 min).
    wall_cap_s : float, optional
        Seconds after which ``wall_capped`` is set (3 x est_hours); None: no cap.
    poll_s : float
        Monitor period.
    on_hung, on_wall_cap : callable, optional
        Called once (from the monitor thread) when the flag is set.
    """

    def __init__(self, label, progress_timeout_s=PROGRESS_TIMEOUT_S, wall_cap_s=None,
                 poll_s=POLL_S, on_hung=None, on_wall_cap=None):
        self.label = str(label)
        self.progress_timeout_s = float(progress_timeout_s)
        self.wall_cap_s = None if wall_cap_s is None else float(wall_cap_s)
        self.poll_s = float(poll_s)
        self.on_hung, self.on_wall_cap = on_hung, on_wall_cap
        self.hung = False
        self.wall_capped = False
        self._proc = _psutil_process()
        self._lock = threading.Lock()
        self._records = {}
        self._open = []
        self._progress = 0
        self._last_move = time.monotonic()
        self._kids_last = {}
        self._kids_ended = 0.0
        self._total = None
        self._total_rec = None
        self._t_start_mono = None
        self._stop = threading.Event()
        self._thread = None

    # -- progress --------------------------------------------------------------
    def progress(self, n=1):
        """Advance the progress counter (pixels, scenes, jobs done)."""
        with self._lock:
            self._progress += int(n)
            self._last_move = time.monotonic()

    @property
    def n_progress(self):
        return int(self._progress)

    def _touch(self):
        with self._lock:
            self._last_move = time.monotonic()

    # -- sampling --------------------------------------------------------------
    def sample(self):
        """One monitor step: RSS into every open stage, live children CPU,
        the hang and wall-cap flags."""
        r = _rss_mb(self._proc)
        a = current_anon_mb(self._proc)
        kids = _children_cpu(self._proc)
        now = time.monotonic()
        fire = []
        with self._lock:
            for o in self._open:
                o.peak = _raise_peak(o.peak, r)
                o.apeak = _raise_peak(o.apeak, a)
            if self._total is not None:
                self._total.peak = _raise_peak(self._total.peak, r)
                self._total.apeak = _raise_peak(self._total.apeak, a)
            gone = set(self._kids_last) - set(kids)
            for pid in gone:                     # a child that ended: keep its last CPU
                self._kids_ended += self._kids_last[pid]
            self._kids_last = kids
            if not self.hung and now - self._last_move > self.progress_timeout_s:
                self.hung = True
                fire.append(self.on_hung)
            if (not self.wall_capped and self.wall_cap_s is not None
                    and self._t_start_mono is not None
                    and now - self._t_start_mono > self.wall_cap_s):
                self.wall_capped = True
                fire.append(self.on_wall_cap)
        for f in fire:
            if f is not None:
                try:
                    f(self)
                except Exception as e:           # a callback must never kill the monitor
                    log(f"[timing] monitor callback failed: {type(e).__name__}: {e}")

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            self.sample()

    def _children_delta(self, o):
        live = 0.0
        for pid, c in self._kids_last.items():
            live += c - o.kids0.get(pid, 0.0)
        return max(0.0, (_rusage_children_s() - o.rch0) + live)

    def _close(self, o):
        r = _rss_mb(self._proc)
        a = current_anon_mb(self._proc)
        io1 = _read_bytes(self._proc)
        peak = _raise_peak(o.peak, r)
        apeak = _raise_peak(o.apeak, a)
        read = (float("nan") if (o.io0 is None or io1 is None)
                else max(0.0, (io1 - o.io0) / 2 ** 20))
        return {"wall_s": max(0.0, time.perf_counter() - o.t0),
                "cpu_self_s": max(0.0, time.process_time() - o.cpu0),
                "cpu_children_s": self._children_delta(o),
                "read_mb": read, "peak_rss_mb": float(peak), "peak_anon_mb": float(apeak),
                "progress": int(self._progress - o.prog0)}

    # -- the unit --------------------------------------------------------------
    def start(self):
        """Start the unit total and the monitor thread (idempotent)."""
        if self._total is None:
            self._total = _Open("unit_total", self)
            self._t_start_mono = time.monotonic()
            self._touch()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name=f"v2-stopwatch-{self.label[:20]}")
            self._thread.start()
        return self

    def stop(self):
        """Stop the monitor and freeze the unit total."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            self._thread = None
        self.sample()
        if self._total is not None and self._total_rec is None:
            self._total_rec = self._close(self._total)
        return self

    @contextlib.contextmanager
    def running(self):
        """``with sw.running():`` start ... stop."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @contextlib.contextmanager
    def stage(self, name):
        """Time one named stage (entering and leaving count as progress)."""
        name = str(name)
        o = _Open(name, self)
        with self._lock:
            self._open.append(o)
        self._touch()
        try:
            yield o
        finally:
            rec = self._close(o)
            with self._lock:
                if o in self._open:
                    self._open.remove(o)
                prev = self._records.get(name)
                if prev is None:
                    self._records[name] = dict(rec, calls=1)
                else:
                    for k in ("wall_s", "cpu_self_s", "cpu_children_s", "progress"):
                        prev[k] = prev[k] + rec[k]
                    prev["read_mb"] = (float("nan") if not (np.isfinite(prev["read_mb"])
                                                            and np.isfinite(rec["read_mb"]))
                                       else prev["read_mb"] + rec["read_mb"])
                    for k in ("peak_rss_mb", "peak_anon_mb"):
                        prev[k] = float(_raise_peak(prev.get(k, float("nan")), rec[k]))
                    prev["calls"] += 1
            self._touch()

    def records(self):
        """``{stage: {wall_s, cpu_self_s, cpu_children_s, read_mb, peak_rss_mb,
        progress, calls}}`` (rounded copies)."""
        with self._lock:
            return {k: _rounded(v) for k, v in self._records.items()}

    def total(self):
        """The whole-unit figures (since :meth:`start`; live until :meth:`stop`)."""
        if self._total_rec is not None:
            return _rounded(self._total_rec)
        if self._total is None:
            return _rounded({k: 0.0 for k in FIELDS})
        return _rounded(self._close(self._total))

    def flags(self):
        return {"hung": bool(self.hung), "wall_capped": bool(self.wall_capped),
                "progress": int(self._progress)}


def _rounded(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, float):
            out[k] = v if not math.isfinite(v) else round(v, 6)
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  dedicated timing runs (§8.2)
# ─────────────────────────────────────────────────────────────────────────────

class TimingRefused(RuntimeError):
    """The machine is not quiet (claims active, load too high)."""


class TimingUnavailable(RuntimeError):
    """The site cannot hold the protocol's workloads (its package lacks the
    W3 years, e.g. a partial-year site): nothing is timed there."""


def active_claims(root):
    """The live claims of the queue under ``<root>/claims`` (instance and
    unit claims of every site): neither stale nor held by a process that is
    gone on this host."""
    import glob

    from experiments.benchmark import runner
    out = []
    for p in sorted(glob.glob(os.path.join(root, "claims", "**", "*.claim"), recursive=True)):
        st = runner.claim_state(p)
        if st is not None and not st["stale"] and st.get("alive") is not False:
            out.append(st)
    return out


def check_quiet(root, max_load=MAX_LOAD):
    """``(ok, why, context)``: no active claim and a load average below
    ``max_load`` (POSIX; not measurable on Windows, recorded as None)."""
    claims = active_claims(root)
    load = load_average()
    ctx = {"active_claims": len(claims), "load_1min": load, "max_load": max_load}
    if claims:
        return False, f"{len(claims)} active claim(s) under {root}/claims", ctx
    if load is not None and load >= max_load:
        return False, f"load average {load:.2f} >= {max_load}", ctx
    return True, "", ctx


#: the curve design whose subsets the N workloads are (K1: the A-mode W3 curve
#: set of P0, N in {20, 45, 90, 180, 330, all}); naming it fixes the replicate
#: count R of the subset draw (``instance.curve_R``: several W3 curve sets hold
#: N = 45 and 180 with different R, so a design-less row is ambiguous)
WORKLOAD_DESIGN = "curves-k1"
WORKLOAD_WINDOW = "W3"


def _workload_row(site, N):
    from experiments.benchmark import settings
    return {"design": WORKLOAD_DESIGN, "site": site, "window": WORKLOAD_WINDOW,
            "setting_id": settings.P0.id, "family": "rand", "mode": "A", "N": N, "rep": 0,
            "variant": ""}


def _summary(runs):
    out = {}
    for k in ("wall_s", "cpu_self_s", "cpu_children_s", "read_mb", "peak_rss_mb",
              "peak_anon_mb"):
        v = np.array([r["total"].get(k, np.nan) for r in runs], float)
        f = v[np.isfinite(v)]
        out[k] = ({"median": float(np.median(f)), "min": float(f.min()), "max": float(f.max())}
                  if f.size else {"median": None, "min": None, "max": None})
    return out


def run_dedicated(root, site, products, workloads=WORKLOADS, repeats=REPEATS, threads=4,
                  site2=None, provider=None, max_load=MAX_LOAD, write=True):
    """Dedicated timing runs of ``products`` at ``site`` (spec §8.2).

    Parameters
    ----------
    root : str
        The validation v2 root (claims, packages, output ``timing/``).
    site : str
    products : sequence of str
        Product ids (``lags.PRODUCT_METHOD``); each method's unit runs once
        per measured run and its timing is written for every requested
        product it writes.
    workloads : sequence
        N values at P0, W3, A-mode, rep 0 ("all" = the whole window).
    repeats : int
        Measured runs after one discarded warm-up.
    threads : int
        The production configuration's BLAS threads.
    site2, provider : optional
        The package site (default ``package.v2_site(site, root)``) and the
        live boundary provider (default the package's).

    Returns
    -------
    dict
        ``{product: path of the JSON written}``.
    """
    from experiments.benchmark import designs, lags, package, product, runner
    from experiments.benchmark import instance as inst_mod
    from experiments.benchmark.product import NotComputable

    ok, why, ctx = check_quiet(root, max_load)
    if not ok:
        raise TimingRefused(f"timing refused: {why} (dedicated runs need a quiet machine, "
                            "spec §8.2)")
    site2 = site2 or package.v2_site(site, root=root)
    pkg = package.V2Package(site2)
    wy = getattr(pkg, "window_years", None)
    if callable(wy):
        try:
            wy(WORKLOAD_WINDOW)
        except Exception as e:                   # a partial-year site (package.PackageError)
            raise TimingUnavailable(
                f"{site}: the timing workloads are {WORKLOAD_WINDOW} subsets (spec §8.2) and "
                f"the package holds the years {list(getattr(pkg, 'years', []) or [])} ({e}); "
                "time a site whose stores cover 2023-2025 (e.g. --sites wadden)") from None
    methods = {}
    for p in products:
        methods.setdefault(lags.method_of(p), []).append(p)
    docs = {p: {"format": TIMING_FORMAT, "label": "dedicated", "site": site, "product": p,
                "method": lags.method_of(p), "protocol": {"warmup": 1, "repeats": int(repeats),
                                                          "single_max_px": SINGLE_THREAD_MAX_PX,
                                                          "production_threads": int(threads)},
                "machine": {"cpu_count": os.cpu_count(), "platform": sys.platform,
                            "context_start": ctx},
                "workloads": {}} for p in products}
    for N in workloads:
        row = _workload_row(site, N)
        try:
            inst = inst_mod.build_instance(site2, row, pkg=pkg, provider=provider)
        except (inst_mod.InstanceUnavailable, ValueError) as e:
            # e.g. an N outside the K1 curve set: that workload only, recorded
            why = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
            log(f"[timing] {site} N={N}: workload not built ({why})")
            for p in products:
                docs[p]["workloads"][str(N)] = {"N": N, "error": why}
            continue
        n_keep = int(inst.domain.n_keep)
        configs = [("production", int(threads))]
        if n_keep <= SINGLE_THREAD_MAX_PX:
            configs.insert(0, ("single", 1))
        for meth, prods in methods.items():
            try:
                fn = lags.load(meth)
            except Exception as e:               # recorded for its products, never fatal
                why = f"error ({type(e).__name__}: {str(e)[:300]})"
                for p in prods:
                    docs[p]["workloads"][str(N)] = {"N": N, "error": why}
                _save(docs, prods, root, site, write)
                continue
            unit = {"method": meth, "variant": "", "design": "timing",
                    "products": tuple(designs.METHODS[meth].products) if meth in designs.METHODS
                    else tuple(prods), "unit_id": f"timing-{meth}", "site": site,
                    "window": "W3", "setting": row["setting_id"], "family": "rand",
                    "mode": "A", "N": N, "rep": 0}
            per_cfg = {}
            for cname, nthr in configs:
                runs = []
                for k in range(1 + int(repeats)):
                    sw = Stopwatch(f"timing {site} {meth} N={N} {cname} #{k}")
                    inst.stage = lambda name, _sw=sw, _m=meth: _sw.stage(f"lag.{_m}.{name}")
                    inst.progress = sw.progress
                    inst.cache.clear()
                    status = "ok"
                    runner.reset_peak_rss()
                    with runner.blas_limits(nthr), sw.running():
                        try:
                            fn(inst, unit, {"n_boot": 0, "refits": 0, "seed_offset": 0,
                                            "threads": nthr, "boot_threads": 1,
                                            "smoke": False})
                        except NotComputable as e:
                            status = f"not_computable ({e.stage})"
                        except Exception as e:   # noqa: BLE001 -- one method, recorded
                            status = f"error ({type(e).__name__}: {str(e)[:300]})"
                    rec = {"run": k, "warmup": k == 0, "status": status,
                           "total": sw.total(), "stages": sw.records(),
                           "peak_rss_process_mb": float(runner.peak_rss_mb())}
                    runs.append(rec)
                    log(f"[timing] {site} {meth} N={N} {cname} run {k}"
                        f"{' (warm-up)' if k == 0 else ''}: {rec['total']['wall_s']:.2f} s "
                        f"wall, {rec['total']['cpu_self_s']:.2f} s cpu, {status}")
                    if status.startswith("error"):
                        break                    # the same error on every repeat
                measured = [r for r in runs if not r["warmup"]]
                per_cfg[cname] = {"threads": nthr, "runs": runs,
                                  "summary": _summary(measured)}
            for p in prods:
                docs[p]["workloads"][str(N)] = {"N": N, "T": int(len(inst.sel)),
                                                "n_keep": n_keep, "unit_products":
                                                    list(unit["products"]),
                                                "configs": per_cfg}
            _save(docs, prods, root, site, write)        # progress survives a later failure
        inst.cache.clear()
    _, _, ctx_end = check_quiet(root, max_load)
    for doc in docs.values():
        doc["machine"]["context_end"] = ctx_end
        doc["complete"] = True
    return _save(docs, list(docs), root, site, write)


def _save(docs, prods, root, site, write):
    """Write the timing JSON of ``prods`` (``{product: path}``), or return the
    documents themselves when ``write`` is False."""
    from experiments.benchmark import product
    out = {}
    for p in prods:
        doc = docs[p]
        doc["saved_utc"] = product.utc_stamp()
        if write:
            path = os.path.join(root, "timing", site, f"{p}.json")
            product.write_json(path, doc)
            out[p] = path
        else:
            out[p] = doc
    return out


def summary_line(rec):
    """One log line of a stage record."""
    return (f"{rec.get('wall_s', 0):.2f} s wall, {rec.get('cpu_self_s', 0):.2f} s cpu, "
            f"{rec.get('cpu_children_s', 0):.2f} s children, read "
            f"{rec.get('read_mb', float('nan')):.1f} MB, peak {rec.get('peak_rss_mb', 0):.0f} MB")


def as_json(records):
    """JSON text of stage records (NaN kept as NaN)."""
    return json.dumps(records, sort_keys=True)
