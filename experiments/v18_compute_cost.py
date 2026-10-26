# -*- coding: utf-8 -*-
"""V18 - computing cost and inputs of MAREA vs NIDEM on equal inputs, per site.

Question (first author, 2026-10-20)
-----------------------------------
Is MAREA publishable as a method in its own right? One part of the answer is
practical: what does it cost to run (wall time, CPU time, memory) next to the
established waterline-contour method (ITEM/NIDEM, :mod:`pyintertidal.
waterline_contour`) when both get the SAME inputs, and what does each method
need as input (tide information, cloud mask, scenes, pixels), and can it fail
outright (NOT COMPUTABLE)? Three subcommands answer it:

``harvest``  the costs already logged, read from existing files only:
             v14 per-job seconds and gate timings (settings JSONs and CSVs of
             results/scene_count_curves/server_2026-10-20), the NIDEM logs and
             diagnostics (results/server_logs_2026-10-19/same_scenes,
             results/server_10m/logs, nidem_diagnostics.json), the per-cell
             timings of the production notebooks (results/server_10m/*.ipynb),
             the reconstruct duration of each production result.json, and the
             Granadeiro 2021 totals (context only). One long CSV, every value
             with its source file and the conditions it ran under (threads,
             concurrent runs).
``time``     a controlled timing of ONE site (run the four sites as four
             processes): single thread (every BLAS/OpenMP pool pinned to 1 and
             checked with threadpoolctl), one timed run of every stage after a
             warm-up, wall time (perf_counter), CPU time (process_time) and
             peak memory (Linux: VmHWM after resetting it through
             /proc/self/clear_refs; elsewhere RSS sampled every 5 ms) per stage.
``report``   one summary CSV, the input-requirements table, a markdown report
             in English and Spanish, and one small figure in both languages.

Equal inputs
------------
Both methods get the same scenes (MAREA's screened record, ``screened``, and
every scene with an overpass time and a boundary level, ``all``: NIDEM's own
ITEM record), the same NDWI values and the same per-pixel cloud mask (the
extraction's C, SCL 4-7), on the same pixels (the extraction's keep mask,
where MAREA inverts), with the same boundary levels. Each method keeps its own
handling of an undefined NDWI: MAREA, as production, reads it as 0 with its
clear flag; NIDEM never counts it as an observation. Setup reuses
:func:`experiments.v14_scene_count_curves.load_site` (record, pools,
production lags, the production lag-fit pixels; its scoring set is loaded and
ignored) and :func:`~experiments.v14_scene_count_curves.stream_columns`; the
inputs of the timed stages are then held in memory, so the timing measures
the methods, not our storage. Reading the cube (each method as published) is
the optional ``nidem_frame`` / ``marea_extract`` stages and the as-run logs.

Stages (per record; ``core`` unless noted)
------------------------------------------
shared  load_inputs       the extraction's Y and C of the record scenes at all
                          keep pixels, streamed by rows (input preparation,
                          not a method cost)
shared  scene_levels      one boundary call at the scene times (both methods)
MAREA   lagged_levels     one call for every lag MAREA can need (the shift
                          bank -45..180 min and every whole minute 1..180,
                          v14.load_site's batching; production made one call
                          per shift and per per-pixel lag)
MAREA   lag_fit           v14.refit_lags: m2a_rasch on production's seeded
                          <= 1200 px per band, the per-pixel lag profile of
                          every keep pixel
MAREA   bootstrap_resample  K (``--boot``) single-thread resamples of the
                          production scene bootstrap (its first K draws,
                          600 px per band); production runs 30 in a thread
                          pool. Timing only (reported apart, not in totals)
MAREA   inversion         v14.invert_subset over EVERY keep pixel, rules
                          min_bracket 0 and 3 (the production inversion)
NIDEM   preflight         interval assignment (9 ITEM intervals) and the
                          per-interval observation counts at keep; under 1 %
                          of the pixels valid (or an empty interval) the
                          record is NOT COMPUTABLE, as v7 decides
NIDEM   composites        per-interval median composites at keep
                          (waterline_contour._composite_rows, the function of
                          stream_composites), placed on the full frame
NIDEM   contours_tin      waterline_contour.nidem: relative extents, gap fill,
                          contours, the linear TIN over the frame, filters
nidem_frame (optional)    NIDEM as published, from the cube: ITEM pixel mask,
                          pre-pass, pre-flight, streamed composites, contours
                          and TIN (I/O included; the v7 path without its
                          verification checks)
marea_extract (optional)  marea.extract from the cube (MAREA's input
                          preparation: two passes over every cube scene)

Warm-up (``--warmup``): ``small`` (default) runs each computing stage once
on a slice (20,000 pixels, 60 px per band, a 64 x 64 synthetic frame for the
TIN) before the timed run; a tide call keeps no state between calls (pyTMD
reads the model files each time, about the same cost whatever the size), so
the setup's own boundary calls are its warm-up; ``full`` repeats every
whole stage first; ``none`` skips it. The I/O stages are never warmed (the
page cache cannot be dropped without root; their figures are as-run).

Totals: the report gives each method's computing time (its own stages) and,
apart, its tide-model calls (MAREA 2, NIDEM 1) and their cost; the working
memory of the computing stages and of the tide calls are also kept apart.

Gates
-----
Before any timing (a failure stops the run, exit 3): every thread pool
reports 1 thread (threadpoolctl); the 1-minute load average is at most
``--max-load`` (POSIX). After each stage (recorded, never a stop; any failure
gives exit 2 at the end): the timed scene levels equal build_record's (1e-9
m); on MAREA's record the refitted band lags equal result.json tau_min
(1e-6 min) and the per-pixel lags production's tau_px_min (0 differing), the
inversion reproduces the stored product on keep (share of pixels with a
validity flip or |dz| > 1e-6 m <= 1e-5); the min_bracket post-filter equals
invert_series(min_bracket=3) on the check slice; NIDEM's |S_j|, Z_j and U_j
equal the stored NIDEM run on the same record (sensitivity_marea_record for
``screened``, the primary for ``all``) or both are NOT COMPUTABLE; the
pre-flight counts equal the composites' counts; ``nidem_frame`` reproduces
the stored filtered DEM bitwise; ``marea_extract`` reproduces the stored
extraction (keep, Y, C on the record rows).

Outputs (new files only, under ``--out``, default results/compute_cost)
-----------------------------------------------------------------------
harvest.csv, harvest_sources.json         ``harvest``
timing_<site>[_<tag>].csv / .json         ``time`` (one row per stage)
compute_cost.csv, inputs_table.csv, site_facts.csv,
compute_cost.md, compute_cost.es.md,
compute_cost_en.{png,pdf}, compute_cost_es.{png,pdf}   ``report``

Run from the repository root:
    python -X utf8 -m experiments.v18_compute_cost harvest --out results/compute_cost/<run>
    python -X utf8 -m experiments.v18_compute_cost time escalda --out results/compute_cost/<run>
    python -X utf8 -m experiments.v18_compute_cost report --out results/compute_cost/<run>
"""
from __future__ import annotations

import os
import sys

#: every thread-count variable a BLAS or OpenMP runtime reads
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
if __name__ == "__main__":
    # single thread BEFORE numpy loads its BLAS (the controlled timing is
    # single-thread by design; harvest and report do not care)
    for _v in THREAD_VARS:
        os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import v14_scene_count_curves as v14  # noqa: E402

OUT = "results/compute_cost"
#: a folder this script created (it may write there again)
MARKER = ".v18_compute_cost"
SITE_ORDER = v14.SITE_ORDER
SITE_NAMES = v14.SITE_NAMES
RECORDS = ("screened", "all")
WARMUPS = ("small", "full", "none")
STAGE_GROUPS = ("core", "nidem_frame", "marea_extract")
#: pixels of the warm-up / check slice
WARM_PX = 20_000
#: lag-fit pixels per band of the warm-up fit (m2a_rasch skips bands < 50)
WARM_PX_BAND = 60
#: pixels per block of the NIDEM composites at keep (a 256-row block of the
#: Ems frame is 874k px; this keeps the per-interval stack under ~100 MB)
NIDEM_BLOCK = 1 << 18
#: lag-fit pixels per band of each production bootstrap resample
#: (marea.reconstruct: bootstrap_lags(..., max_px_band=600))
BOOT_PX_BAND = 600
#: production bootstrap size when result.json does not say
N_BOOT_DEFAULT = 30
TOL_LEVEL = 1e-9
TOL_TAU = 1e-6
TOL_Z = 1e-6
#: share of keep allowed to differ from the stored product (a validity flip
#: or |dz| > TOL_Z): BLAS rounding can flip a near-tied grid choice
TOL_DIFF_SHARE = 1e-5
DEFAULT_MAX_LOAD = 6.0
#: status values of a timing row
STATUS = ("ok", "not_computable", "failed", "setup")

TIMING_COLUMNS = ("site", "tag", "record", "n_scenes", "n_px", "frame_px", "method", "stage",
                  "status", "wall_s", "cpu_s", "cpu_util", "rss_start_mb", "peak_rss_mb",
                  "work_mb", "mem_method", "inputs_mb", "warmup", "warmup_wall_s", "threads",
                  "load1_start", "load1_end", "read_mb", "n_out", "note")
HARVEST_COLUMNS = ("site", "method", "source", "condition", "record", "stage", "quantity",
                   "value", "unit", "n_scenes", "n_px", "threads", "concurrent_runs", "note",
                   "path")

#: the computing stages of each method (its own work on the in-memory inputs)
METHOD_STAGES = {
    "marea": ("lag_fit", "inversion"),
    "nidem": ("preflight", "composites", "contours_tin"),
    "nidem_frame": ("prepass", "preflight", "composites", "contours_tin"),
    "marea_extract": ("extract",),
}
#: the tide-model calls of each method, reported apart (one pyTMD call costs
#: about the same whatever its size); ``shared`` scene_levels counts for both
TIDE_STAGES = {"marea": ("lagged_levels",), "nidem": ()}
WITH_SHARED = ("marea", "nidem")

#: default sources of ``harvest`` (paths relative to the repository root;
#: laptop mirrors of the server first, then the server's own layout)
V14_DIRS = ("results/scene_count_curves/server_2026-10-20",)
NIDEM_LOG_DIRS = ("results/server_logs_2026-10-19/same_scenes", "results/server_10m/logs")
NOTEBOOK_DIRS = ("results/server_10m",)
NIDEM_DIAG_GLOBS = (
    "results/server_10m/products_*/nidem_*/nidem_diagnostics.json",
    "products_*_10m/nidem_*/nidem_diagnostics.json",
    "products_*_10m/nidem_*/sensitivity_marea_record/nidem_diagnostics.json",
    "products_marea/nidem/nidem_diagnostics.json",
    "products_marea/nidem/sensitivity_marea_record/nidem_diagnostics.json",
)
RESULT_JSON_GLOBS = (
    "results/server_10m/products_*/marea_*/result.json",
    "products_*_10m/marea_eot20_*/result.json",
    "products_marea/marea_eot20_g0_2023-2025/result.json",
)
GRANADEIRO_LOG_GLOBS = ("results/server_10m/logs/granadeiro_*.log",)
GAUGES_CSV = "results/lag_vs_gauges_2026-10-20/marea_lag_vs_gauges.csv"
SAME_SCENES_SUMMARY = "results/method_comparison_same_scenes/same_scenes_summary.csv"
#: results folders this script never writes into
PROTECTED_RESULTS = ("scene_count_curves", "server_10m", "server_logs", "method_comparison",
                     "lag_vs_gauges", "validation_v2", "backup_all_scenes", "v13_benchmark")

PAL = v14.PAL
_T0 = time.time()


def log(msg):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{(time.time() - _T0) / 60:6.1f} min] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  measurement
# ─────────────────────────────────────────────────────────────────────────────

def _proc_status_kb(key):
    """A ``kB`` field of /proc/self/status (Linux), or None."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _psutil_process():
    try:
        import psutil
        return psutil.Process()
    except Exception:  # noqa: BLE001
        return None


def rss_mb():
    """Resident set size of this process (MiB)."""
    v = _proc_status_kb("VmRSS")
    if v is not None:
        return v / 1024.0
    p = _psutil_process()
    if p is not None:
        return p.memory_info().rss / 2 ** 20
    return float("nan")


def _reset_hwm():
    """Reset this process's peak RSS (VmHWM) to its current RSS (Linux >= 4.0)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
        return True
    except OSError:
        return False


def read_mb():
    """Bytes this process has read from storage so far (MiB), or None."""
    p = _psutil_process()
    try:
        return p.io_counters().read_bytes / 2 ** 20 if p is not None else None
    except Exception:  # noqa: BLE001
        return None


def load1():
    """1-minute load average (POSIX), or None."""
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


class MemProbe:
    """Peak RSS while a stage runs.

    Linux: VmHWM after resetting it through /proc/self/clear_refs (exact,
    no thread). Elsewhere (or if the reset is refused): a daemon thread
    samples the RSS every ``interval_s`` seconds (psutil); a peak shorter than
    the interval can be missed. ``work_mb`` = peak - RSS at the stage start:
    the stage's working memory on top of what was already resident."""

    def __init__(self, interval_s=0.005, force_sampling=False):
        self.interval = float(interval_s)
        self.force_sampling = bool(force_sampling)
        self.kind = None

    def start(self):
        self.rss0 = rss_mb()
        self._hwm = (not self.force_sampling and sys.platform.startswith("linux")
                     and _proc_status_kb("VmHWM") is not None and _reset_hwm())
        self._thread = None
        if self._hwm:
            self.kind = "VmHWM after reset (/proc/self/clear_refs)"
            return self
        self._max = self.rss0
        self._stop = threading.Event()
        if _psutil_process() is None:
            self.kind = "unavailable (no psutil)"
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.kind = f"RSS sampled every {self.interval * 1000:.0f} ms (psutil)"
        return self

    def _loop(self):
        p = _psutil_process()
        while not self._stop.is_set():
            try:
                r = p.memory_info().rss / 2 ** 20
            except Exception:  # noqa: BLE001
                break
            if r > self._max:
                self._max = r
            self._stop.wait(self.interval)

    def stop(self):
        rss1 = rss_mb()
        if self._hwm:
            v = _proc_status_kb("VmHWM")
            peak = v / 1024.0 if v is not None else float("nan")
        elif self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            peak = max(self._max, rss1)
        else:
            peak = float("nan")
        return {"rss_start_mb": self.rss0, "peak_rss_mb": peak, "rss_end_mb": rss1,
                "work_mb": peak - self.rss0, "mem_method": self.kind}


def preload_libraries():
    """Import every library whose BLAS/OpenMP pool a timed stage uses, so the
    thread limit and its check see all of them."""
    import scipy.interpolate  # noqa: F401
    import scipy.linalg  # noqa: F401
    import scipy.ndimage  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.spatial  # noqa: F401
    import scipy.special  # noqa: F401
    import skimage.measure  # noqa: F401
    from pyintertidal import marea, tide_estimators, waterline_contour  # noqa: F401


def pin_threads(n=1):
    """Every loaded BLAS/OpenMP pool limited to ``n`` threads (threadpoolctl),
    on top of the environment set at start-up. Returns the limiter (keep a
    reference) or None if threadpoolctl is missing."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=int(n))
    except Exception:  # noqa: BLE001
        return None


def thread_report():
    """The thread-count environment and every pool threadpoolctl sees."""
    out = {"env": {v: os.environ.get(v) for v in THREAD_VARS}}
    try:
        from threadpoolctl import threadpool_info
        pools = [{"api": d.get("internal_api"), "lib": os.path.basename(str(d.get("filepath", ""))),
                  "num_threads": d.get("num_threads")} for d in threadpool_info()]
        out["pools"] = pools
        out["max_threads"] = max([int(p["num_threads"] or 0) for p in pools], default=None)
    except Exception as e:  # noqa: BLE001
        out["pools"] = None
        out["max_threads"] = None
        out["error"] = repr(e)
    return out


def threads_text(rep):
    pools = rep.get("pools")
    if not pools:
        return "env " + ",".join(f"{k}={v}" for k, v in rep["env"].items() if v)
    return ";".join(f"{p['api']}:{p['num_threads']}" for p in pools)


def threads_gate(rep):
    """OK iff every pool threadpoolctl sees runs 1 thread; without
    threadpoolctl the environment is recorded and the gate is 'unverified'."""
    m = rep.get("max_threads")
    if rep.get("pools") is None:
        return {"ok": True, "verified": False, "reason": "threadpoolctl unavailable: environment only",
                "env": rep["env"]}
    return {"ok": m == 1, "verified": True, "max_threads": m, "pools": rep["pools"]}


def machine_info():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cpu_count": os.cpu_count(), "node": platform.node()}
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["cpu_model"] = platform.processor() or None
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        pass
    for mod in ("scipy", "skimage", "pandas", "xarray"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    return info


# ─────────────────────────────────────────────────────────────────────────────
#  timing rows
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else repr(round(float(v), 6))
    return str(v).replace("\n", " ")


class Recorder:
    """Times stages and appends one CSV row per stage (written and fsynced as
    each stage ends, so a crash keeps what finished)."""

    def __init__(self, csv_path, site, tag="", warmup="small", threads=""):
        self.csv_path = csv_path
        self.site, self.tag, self.warmup, self.threads = site, tag, warmup, threads
        self.rows = []
        if csv_path and (not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(TIMING_COLUMNS)

    def base(self, record, method, stage, n_scenes=None, n_px=None, frame_px=None, inputs_mb=None,
             note=""):
        r = {c: None for c in TIMING_COLUMNS}
        r.update(site=self.site, tag=self.tag, record=record, method=method, stage=stage,
                 n_scenes=n_scenes, n_px=n_px, frame_px=frame_px, inputs_mb=inputs_mb,
                 warmup=self.warmup, threads=self.threads, note=note, status="ok")
        return r

    def stage(self, record, method, stage, fn, warm_small=None, warm_full=None, **base_kw):
        """Warm-up (``self.warmup``), then ONE timed run of ``fn``. Returns
        (value, row); the caller completes the row and emits it."""
        warm = warm_full if self.warmup == "full" else warm_small if self.warmup == "small" else None
        warm_s = None
        if warm is not None:
            t0 = time.perf_counter()
            warm()
            warm_s = time.perf_counter() - t0
        gc.collect()
        l0, io0 = load1(), read_mb()
        probe = MemProbe().start()
        t0, c0 = time.perf_counter(), time.process_time()
        value = fn()
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        mem = probe.stop()
        io1 = read_mb()
        row = self.base(record, method, stage, **base_kw)
        row.update(wall_s=wall, cpu_s=cpu, cpu_util=(cpu / wall if wall > 0 else float("nan")),
                   rss_start_mb=mem["rss_start_mb"], peak_rss_mb=mem["peak_rss_mb"],
                   work_mb=mem["work_mb"], mem_method=mem["mem_method"],
                   warmup=(self.warmup if warm is not None else "none"), warmup_wall_s=warm_s,
                   load1_start=l0, load1_end=load1(),
                   read_mb=(io1 - io0 if io0 is not None and io1 is not None else None))
        return value, row

    def skipped(self, record, method, stage, status, note="", **base_kw):
        row = self.base(record, method, stage, note=note, **base_kw)
        row.update(status=status, wall_s=float("nan"), cpu_s=float("nan"), warmup="none")
        return row

    def emit(self, row):
        self.rows.append(row)
        if self.csv_path:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([_fmt(row.get(c)) for c in TIMING_COLUMNS])
                f.flush()
                os.fsync(f.fileno())
        if row.get("status") == "ok" and row.get("wall_s") is not None:
            log(f"[{self.site}] {row['record'] or '-':8s} {row['method']:13s} {row['stage']:18s} "
                f"wall {row['wall_s']:9.2f} s  cpu {row['cpu_s'] or 0:9.2f} s  work "
                f"{(row.get('work_mb') or float('nan')):8.1f} MB  {row.get('note') or ''}")
        else:
            log(f"[{self.site}] {row['record'] or '-':8s} {row['method']:13s} {row['stage']:18s} "
                f"{row.get('status')}  {row.get('note') or ''}")


# ─────────────────────────────────────────────────────────────────────────────
#  equal-input helpers
# ─────────────────────────────────────────────────────────────────────────────

class RecordLevels:
    """Boundary levels computed at the record's own instants only (a
    :class:`v14.LevelTable` over them), presented in the row space of the
    loaded arrays (NaN outside the record), the indexing that
    v14.invert_subset and v14.refit_lags expect (``levels[tau][rows]``)."""

    def __init__(self, table, rows, n):
        self.table = table
        self.rows = np.asarray(rows, np.int64)
        self.n = int(n)
        if len(self.rows) != len(table.times):
            raise ValueError("one record row per table instant")

    def ensure(self, taus):
        self.table.ensure(taus)

    def __getitem__(self, tau):
        out = np.full(self.n, np.nan)
        out[self.rows] = self.table[tau]
        return out


def clear_nan_exceptions(Y, C):
    """{row: columns} where the clear flag is set but the NDWI is undefined
    (NaN): MAREA reads these as NDWI 0 (production nan_to_num), NIDEM never
    counts them. Computed before Y's NaN are replaced."""
    bad = {}
    for q in range(Y.shape[0]):
        e = np.flatnonzero(C[q] & np.isnan(Y[q]))
        if len(e):
            bad[int(q)] = e
    return bad


def _exceptions_in(bad, q, c0, c1):
    e = bad.get(int(q))
    if e is None:
        return None
    e = e[(e >= c0) & (e < c1)]
    return e - c0 if len(e) else None


class ObsRows:
    """NIDEM's observation mask on the record (the extraction's C AND a
    finite NDWI) at the columns ``c0:c1``, one row at a time, for
    waterline_contour.clear_counts_by_interval."""

    def __init__(self, C0, bad, rows, c0=0, c1=None):
        self.C0, self.bad = C0, bad
        self.rows = np.asarray(rows, np.int64)
        self.c0 = int(c0)
        self.c1 = int(C0.shape[1] if c1 is None else c1)
        self.shape = (len(self.rows), self.c1 - self.c0)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, q):
        r = int(self.rows[q])
        c = np.array(self.C0[r, self.c0:self.c1], dtype=bool)
        e = _exceptions_in(self.bad, r, self.c0, self.c1)
        if e is not None:
            c[e] = False
        return c


def composites_keep(Y0, C0, bad, rows, j, c0=0, c1=None, block=NIDEM_BLOCK):
    """ITEM's per-interval composites at the keep columns ``c0:c1`` from the
    extraction arrays: for every pixel block, exactly
    ``waterline_contour._composite_rows`` (the function stream_composites
    runs per row block) with the masked NDWI of each record scene (NaN where
    the clear flag is unset or the NDWI is undefined). Returns (9, P) median
    and count, (P,) sd_mean, sd_mean_bounded, n_ndwi_above_bound,
    n_ndwi_below_minus_bound, and max_abs_ndwi."""
    from pyintertidal import waterline_contour as wc

    rows = np.asarray(rows, np.int64)
    c1 = int(Y0.shape[1] if c1 is None else c1)
    P = c1 - int(c0)
    out = {"median": np.empty((wc.N_INTERVALS, P), np.float32),
           "count": np.empty((wc.N_INTERVALS, P), np.uint16),
           "sd_mean": np.empty(P, np.float32), "sd_mean_bounded": np.empty(P, np.float32),
           "n_ndwi_above_bound": np.empty(P, np.uint16),
           "n_ndwi_below_minus_bound": np.empty(P, np.uint16), "max_abs_ndwi": float("nan")}
    for b0 in range(int(c0), c1, int(block)):
        b1 = min(c1, b0 + int(block))

        def read(i, b0=b0, b1=b1):
            q = int(rows[i])
            v = np.where(C0[q, b0:b1], Y0[q, b0:b1], np.nan).astype(np.float32)
            e = _exceptions_in(bad, q, b0, b1)
            if e is not None:
                v[e] = np.nan
            return v[None, :]

        m_, c_, s_, x_ = wc._composite_rows(read, j, 1, b1 - b0)  # noqa: SLF001
        o0, o1 = b0 - int(c0), b1 - int(c0)
        out["median"][:, o0:o1] = m_[:, 0]
        out["count"][:, o0:o1] = c_[:, 0]
        out["sd_mean"][o0:o1] = s_[0]
        out["sd_mean_bounded"][o0:o1] = x_["sd_mean_bounded"][0]
        out["n_ndwi_above_bound"][o0:o1] = x_["n_ndwi_above_bound"][0]
        out["n_ndwi_below_minus_bound"][o0:o1] = x_["n_ndwi_below_minus_bound"][0]
        out["max_abs_ndwi"] = wc._merge_max(out["max_abs_ndwi"], x_["max_abs_ndwi"])  # noqa: SLF001
    return out


def to_frame(ck, keep, shape):
    """The keep composites on the full H x W frame (NaN / 0 elsewhere): the
    dict :func:`pyintertidal.waterline_contour.nidem` takes."""
    H, W = (int(v) for v in shape)
    n = H * W
    keep = np.asarray(keep, np.int64)
    nI = ck["median"].shape[0]
    med = np.full((nI, n), np.nan, np.float32)
    med[:, keep] = ck["median"]
    cnt = np.zeros((nI, n), np.uint16)
    cnt[:, keep] = ck["count"]
    fr = {"median": med.reshape(nI, H, W), "count": cnt.reshape(nI, H, W)}
    for k, fill, dt in (("sd_mean", np.nan, np.float32), ("sd_mean_bounded", np.nan, np.float32),
                        ("n_ndwi_above_bound", 0, np.uint16), ("n_ndwi_below_minus_bound", 0, np.uint16)):
        a = np.full(n, fill, dt)
        a[keep] = ck[k]
        fr[k] = a.reshape(H, W)
    fr["max_abs_ndwi"] = ck["max_abs_ndwi"]
    fr["shape"] = (H, W)
    return fr


def synthetic_composites(n=64):
    """A small, valid 9-interval composite (a sloping flat): the TIN warm-up."""
    from pyintertidal import waterline_contour as wc

    yy, xx = np.mgrid[0:n, 0:n]
    zsurf = (xx / (n - 1.0)) * 2.0 - 1.0 + 0.1 * np.sin(yy / 5.0)
    Z = np.linspace(-0.9, 0.9, wc.N_INTERVALS)
    med = np.stack([np.where(z > zsurf, 0.4, -0.3) for z in Z]).astype(np.float32)
    cnt = np.full(med.shape, 3, np.uint16)
    sd = np.full((n, n), 0.05, np.float32)
    comp = {"median": med, "count": cnt, "sd_mean": sd, "sd_mean_bounded": sd.copy(),
            "n_ndwi_above_bound": np.zeros((n, n), np.uint16),
            "n_ndwi_below_minus_bound": np.zeros((n, n), np.uint16), "max_abs_ndwi": 0.4,
            "shape": (n, n)}
    return comp, Z, np.full(wc.N_INTERVALS, 0.05)


def bootstrap_draws(T, n_boot, seed):
    """The scene resamples of :func:`pyintertidal.tide_estimators.
    bootstrap_lags` (same generator, same order)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, T, T) for _ in range(int(n_boot))]


def bootstrap_resample(m2a, levels, rows, draw, b, seed):
    """One production bootstrap resample, single thread: ``bootstrap_lags``'s
    ``one(b)`` on the lag-fit arrays (the shifted levels of the drawn scenes,
    m2a_rasch with max_px_band 600 and rng seed + 100 + b). Returns the band
    lags."""
    from pyintertidal import marea
    from pyintertidal import tide_estimators as te

    idx = np.asarray(rows, np.int64)[np.asarray(draw, np.int64)]
    Cm = m2a["C"][idx]
    wet = Cm & (m2a["Y"][idx] > 0)
    bank = te.ShiftBank(marea.BANK_TAUS, [levels[t][idx] for t in marea.BANK_TAUS])
    r = te.m2a_rasch(wet, Cm, bank, m2a["band"], m2a["centers"], m2a["nb"],
                     rng=np.random.default_rng(int(seed) + 100 + int(b)),
                     **{**m2a["fit_kw"], "max_px_band": BOOT_PX_BAND})
    return np.asarray(r["tau"], float)


def small_m2a(m2a, per_band=WARM_PX_BAND):
    """The lag-fit arrays restricted to the first ``per_band`` pixels of each
    band (the warm-up fit)."""
    band = np.asarray(m2a["band"])
    sel = np.concatenate([np.flatnonzero(band == k)[:per_band] for k in range(int(m2a["nb"]))])
    return dict(m2a, Y=m2a["Y"][:, sel], C=m2a["C"][:, sel], band=band[sel])


# ─────────────────────────────────────────────────────────────────────────────
#  gates
# ─────────────────────────────────────────────────────────────────────────────

def compare_z(z_new, z_ref, tol=TOL_Z, share_max=TOL_DIFF_SHARE):
    """Reproduction of a stored per-pixel product: validity flips, values
    differing by more than ``tol``, and the share of pixels with either."""
    z_new, z_ref = np.asarray(z_new, float), np.asarray(z_ref, float)
    fa, fb = np.isfinite(z_new), np.isfinite(z_ref)
    flips = int(np.sum(fa != fb))
    both = fa & fb
    dz = np.abs(z_new[both] - z_ref[both])
    n_dz = int(np.sum(dz > tol))
    share = (flips + n_dz) / max(len(z_ref), 1)
    return {"pixels": int(len(z_ref)), "validity_flips": flips, "n_abs_dz_gt_tol": n_dz,
            "max_abs_dz_m": float(dz.max()) if dz.size else float("nan"), "tol_m": tol,
            "share_differing": share, "share_max": share_max, "ok": bool(share <= share_max)}


def stored_nidem_state(paths):
    """('npz', arrays) for a stored NIDEM run, ('not_computable', reason),
    or (None, why)."""
    npz, diag = paths.get("npz"), paths.get("diag")
    if npz and os.path.exists(npz):
        with np.load(npz, allow_pickle=False) as z:
            keys = [k for k in ("sizes", "Z", "U", "h", "filtered_keep") if k in z.files]
            return "npz", {k: np.asarray(z[k]) for k in keys}
    if diag and os.path.exists(diag):
        d = json.load(open(diag, encoding="utf-8"))
        if str(d.get("status", "")).upper().startswith("NOT COMPUTABLE"):
            return "not_computable", d.get("reason")
        return None, f"{diag}: status {d.get('status')!r} without a nidem.npz"
    return None, "no stored NIDEM run for this record"


def nidem_intervals_gate(paths, computable, sizes=None, Z=None, U=None, h=None):
    """The record's NIDEM intervals equal the stored run on the same record
    (|S_j|, Z_j, U_j depend only on the levels, not on the pixel mask), or
    both runs are NOT COMPUTABLE."""
    kind, s = stored_nidem_state(paths)
    if kind is None:
        return {"checked": False, "reason": s, "ok": True}
    if kind == "not_computable":
        return {"checked": True, "stored": "NOT COMPUTABLE", "stored_reason": s,
                "mine": "computable" if computable else "NOT COMPUTABLE", "ok": not computable}
    if not computable:
        return {"checked": True, "stored": "computed", "mine": "NOT COMPUTABLE", "ok": False}
    g = {"checked": True, "stored": "computed", "mine": "computed",
         "sizes_mine": np.asarray(sizes).tolist(), "sizes_stored": s["sizes"].tolist()}
    ok = np.array_equal(np.asarray(sizes), s["sizes"])
    ok &= np.allclose(np.asarray(Z, float), s["Z"].astype(float), rtol=0, atol=1e-12, equal_nan=True)
    ok &= np.allclose(np.asarray(U, float), s["U"].astype(float), rtol=0, atol=1e-12, equal_nan=True)
    if h is not None and "h" in s:
        hs = s["h"].astype(float)
        g["h_max_abs_diff_m"] = (float(np.max(np.abs(np.asarray(h, float) - hs)))
                                 if len(hs) == len(h) else None)
        ok &= g["h_max_abs_diff_m"] is not None and g["h_max_abs_diff_m"] <= TOL_LEVEL
    g["ok"] = bool(ok)
    return g


def agreement(ours, stored):
    """How the equal-input NIDEM at keep compares with the stored NIDEM at
    keep (information: the masks and the frame differ by design)."""
    a, b = np.asarray(ours, float), np.asarray(stored, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.abs(a[both] - b[both])
    return {"n_ours": int(fa.sum()), "n_stored": int(fb.sum()), "n_both": int(both.sum()),
            "median_abs_dz_m": float(np.median(d)) if d.size else None,
            "p95_abs_dz_m": float(np.percentile(d, 95)) if d.size else None,
            "share_identical": float(np.mean(d == 0)) if d.size else None}


def failed_gates(gates):
    """Names of failed gates in a nested {name: {..., 'ok': bool}} dict."""
    out = []
    for k, v in gates.items():
        if isinstance(v, dict) and "ok" in v:
            if v.get("ok") is False:
                out.append(k)
        elif isinstance(v, dict):
            out += [f"{k}.{x}" for x in failed_gates(v)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  output folder
# ─────────────────────────────────────────────────────────────────────────────

def guard_out(out):
    """Outputs go to a NEW folder under results/ (or outside the repository,
    e.g. a scratch area). Refuses v14's forbidden paths, the protected
    results folders, and an existing folder this script did not create
    unless it holds only logs (a ``logs/`` folder, ``*.log`` files). Creates
    the folder and its marker."""
    v14.guard_out(out)
    try:
        rel = os.path.relpath(os.path.abspath(out), ROOT)
    except ValueError:                       # another drive (Windows): outside
        rel = None
    inside = rel is not None and rel != ".." and not rel.startswith(".." + os.sep)
    if inside:
        parts = rel.replace("\\", "/").split("/")
        if parts[0] != "results" or len(parts) < 2:
            raise SystemExit(f"refusing to write into {out!r}: outputs go to a new folder under results/")
        bad = [p for p in parts[1:] if any(p.lower().startswith(x) for x in PROTECTED_RESULTS)]
        if bad:
            raise SystemExit(f"refusing to write into {out!r}: {bad[0]!r} holds other experiments' results")
    if os.path.isdir(out) and not os.path.exists(os.path.join(out, MARKER)):
        other = [e for e in os.listdir(out) if e != "logs" and not e.endswith(".log")]
        if other:
            raise SystemExit(f"refusing to write into {out!r}: an existing folder this script did not create "
                             f"(holds {other[:3]})")
    os.makedirs(out, exist_ok=True)
    mk = os.path.join(out, MARKER)
    if not os.path.exists(mk):
        with open(mk, "w", encoding="utf-8") as f:
            f.write("created by experiments/v18_compute_cost.py\n")
    return out


def keep_previous(path):
    """An existing output is renamed <path>.prev-<timestamp>, never overwritten."""
    if os.path.exists(path):
        os.replace(path, f"{path}.prev-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")


# ─────────────────────────────────────────────────────────────────────────────
#  the controlled timing of one site
# ─────────────────────────────────────────────────────────────────────────────

def site_inputs(args):
    """The site's input configuration (v14.site_config with overrides) and
    the cube path."""
    from experiments import v7_waterline_contour as v7

    cfg = v14.site_config(args.site)
    for k in ("extract", "product", "comparison", "nidem"):
        v = getattr(args, f"{k}_override", None)
        if v:
            cfg[k] = v
    cube = args.cube or v7.SITES[args.site]["cube"]
    return cfg, cube


def setup(args, recorder):
    """Everything the timed stages need, in memory (untimed except
    ``shared/load_inputs``). Returns a namespace."""
    site = args.site
    cfg, cube = site_inputs(args)
    ctx = types.SimpleNamespace(site=site, cfg=cfg, cube=cube, gates={}, outputs={}, failures=[],
                                recorder=recorder, warmup=args.warmup)
    t0 = time.perf_counter()
    log(f"[{site}] setup: v14.load_site (record, pools, production lags, lag-fit pixels; "
        "its scoring set is not used)")
    sd, meta, extras = v14.load_site(site, cfg, "refit", 1, v14.DEFAULT_SEED, ("screened", "all"))
    rec, res, keep = extras["rec"], extras["res"], np.asarray(extras["keep"], np.int64)
    n_all = len(rec["time_index"])
    pools = {"screened": np.asarray(extras["idx_scr"], np.int64), "all": np.arange(n_all)}
    want = [r for r in RECORDS if r in args.records]
    row_space = np.unique(np.concatenate([pools[r] for r in want]))
    ctx.rows = {r: np.searchsorted(row_space, pools[r]) for r in want}
    ctx.records = want
    ctx.n_space = len(row_space)
    ctx.t_space = rec["t_real"][row_space]
    ctx.h_space = np.asarray(rec["h"], float)[row_space]
    ctx.ti_space = np.asarray(rec["time_index"], np.int64)[row_space]
    ctx.dates_space = np.asarray(rec["dates"]).astype(str)[row_space]
    ctx.boundary, ctx.boundary_name = rec["boundary_obj"], rec["boundary"]
    ctx.res, ctx.keep, ctx.P = res, keep, len(keep)
    H, W = (int(v) for v in meta["inputs"]["extract"]["grid"])
    ctx.shape, ctx.frame_px = (H, W), H * W
    ctx.z_stored = np.asarray(extras["z_stored"], float)
    ctx.tau_px_all = np.asarray(extras["tau_px_all"], float)
    m2a = sd.m2a
    if ctx.n_space != n_all:
        m2a = dict(m2a, Y=m2a["Y"][row_space], C=m2a["C"][row_space])
    ctx.m2a = m2a
    # the mouth distance of every keep pixel, as v14.load_site takes it
    s_km = None
    cp = f"{cfg['comparison']}/products.npz" if cfg.get("comparison") else None
    if cp and os.path.exists(cp):
        with np.load(cp, allow_pickle=False) as z:
            if "s_km" in z.files:
                s_km, s_src = z["s_km"].astype(float), f"{cp} s_km"
    if s_km is None:
        mp = f"{cfg['product']}/marea.npz"
        with np.load(mp, allow_pickle=False) as z:
            s_km, s_src = z["s_km"].astype(float), f"{mp} s_km (stored float32)"
    if len(s_km) != ctx.P:
        raise RuntimeError(f"s_km has {len(s_km)} values for {ctx.P} keep pixels")
    ctx.s_km = s_km
    nd = cfg["nidem"]
    ctx.stored = {"screened": {"npz": os.path.join(nd, "sensitivity_marea_record", "nidem.npz"),
                               "diag": os.path.join(nd, "sensitivity_marea_record", "nidem_diagnostics.json")},
                  "all": {"npz": os.path.join(nd, "nidem.npz"),
                          "diag": os.path.join(nd, "nidem_diagnostics.json")}}
    ctx.meta = {k: meta.get(k) for k in ("inputs", "notes", "boundary", "lags_production", "lag_fit")}
    ctx.meta["pools"] = {k: {kk: vv for kk, vv in v.items() if kk != "dates"}
                         for k, v in meta.get("pools", {}).items()}
    ctx.meta["s_km"] = s_src
    ctx.meta["row_space"] = {"scenes": int(ctx.n_space), "records": {r: int(len(v)) for r, v in ctx.rows.items()}}
    setup_s = time.perf_counter() - t0
    row = recorder.base(None, "shared", "setup", n_px=ctx.P, frame_px=ctx.frame_px,
                        note="v14.load_site and the mouth distance; untimed setup, not a method cost")
    row.update(status="setup", wall_s=setup_s, warmup="none")
    recorder.emit(row)

    # the extraction's Y and C of the row space at every keep pixel
    P = ctx.P

    def load():
        Ys = v14.stream_columns(cfg["extract"], "Y", ctx.ti_space, [np.arange(P)])[0]
        Cs = v14.stream_columns(cfg["extract"], "C", ctx.ti_space, [np.arange(P)])[0]
        return Ys, Cs

    (Y, C), row = recorder.stage(None, "shared", "load_inputs", load, n_scenes=ctx.n_space, n_px=P,
                                 frame_px=ctx.frame_px)
    ctx.bad = clear_nan_exceptions(Y, C)
    np.nan_to_num(Y, copy=False, nan=0.0)
    ctx.Y0, ctx.C0 = Y, C.astype(bool, copy=False)
    n_bad = int(sum(len(v) for v in ctx.bad.values()))
    row.update(inputs_mb=(Y.nbytes + C.nbytes) / 2 ** 20,
               note=(f"extraction {cfg['extract']}: {ctx.n_space} scenes x {P:,} px streamed by rows "
                     f"(input preparation, not a method cost); {n_bad} clear observations with an "
                     "undefined NDWI (MAREA: NDWI 0, as production; NIDEM: not an observation)"))
    recorder.emit(row)
    ctx.outputs["clear_but_nan_ndwi"] = n_bad
    return ctx


def _rec_kw(ctx, record):
    return dict(n_scenes=int(len(ctx.rows[record])), n_px=ctx.P, frame_px=ctx.frame_px)


def _io_mb(T, P):
    """Logical size of a record's NDWI (float32) and clear flags (bool)."""
    return T * P * 5 / 2 ** 20


def scene_levels_group(ctx, record):
    """shared/scene_levels: the record's level table (one call at lag 0)."""
    rows = ctx.rows[record]
    t_rec = ctx.t_space[rows]
    table = v14.LevelTable(ctx.boundary, t_rec)
    if not hasattr(ctx, "tables"):
        ctx.tables = {}

    def fn():
        table.ensure([0])
        return table[0]

    def warm_full():
        v14.LevelTable(ctx.boundary, t_rec).ensure([0])

    # a tide call keeps no state between calls (pyTMD reads the model each
    # time): the setup's own boundary calls are its warm-up
    h0, row = ctx.recorder.stage(record, "shared", "scene_levels", fn, None, warm_full,
                                 **_rec_kw(ctx, record))
    d = float(np.max(np.abs(np.asarray(h0, float) - ctx.h_space[rows])))
    ctx.gates.setdefault(record, {})["scene_levels_vs_build_record"] = {"max_abs_diff_m": d,
                                                                         "ok": d <= TOL_LEVEL}
    row.update(n_out=len(h0), warmup=("setup's boundary calls" if ctx.recorder.warmup == "small" else row["warmup"]),
               note=f"one boundary call ({ctx.boundary_name}) at {len(h0)} instants")
    ctx.recorder.emit(row)
    ctx.tables[record] = (table, RecordLevels(table, rows, ctx.n_space))


def marea_group(ctx, record, boot=0, boot_here=False):
    """MAREA on the record: the lagged levels (one call), the lag fit,
    (bootstrap resamples), the inversion of every keep pixel."""
    from pyintertidal import marea
    from experiments import v14_scene_count_curves as v14_

    rows = ctx.rows[record]
    T, P = len(rows), ctx.P
    t_rec = ctx.t_space[rows]
    table, lev = ctx.tables[record]
    kw = _rec_kw(ctx, record)
    g = ctx.gates.setdefault(record, {})
    o = ctx.outputs.setdefault(record, {}).setdefault("marea", {})
    R = ctx.recorder

    # every lag MAREA can need, in one call (v14.load_site's batching): the
    # shift bank of the lag fit and every whole-minute per-pixel lag (the
    # profile is clipped to >= 0 and the fit bounded at TAU_BOUNDS[1])
    lags_all = sorted({float(v) for v in marea.BANK_TAUS}
                      | {float(v) for v in range(1, int(marea.TAU_BOUNDS[1]) + 1)})
    new0 = [v for v in lags_all if int(v) not in table.L]
    _, row = R.stage(record, "marea", "lagged_levels", lambda: table.ensure(lags_all), None,
                     lambda: v14_.LevelTable(ctx.boundary, t_rec).ensure(lags_all), **kw)
    row.update(n_out=len(new0), warmup=("setup's boundary calls" if R.warmup == "small" else row["warmup"]),
               note=(f"{len(new0)} lags ({lags_all[0]:g}..{lags_all[-1]:g} min: the shift bank and every "
                     f"whole-minute per-pixel lag) in one boundary call over {len(new0) * T:,} instants"))
    R.emit(row)

    # the lag fit (v14.refit_lags = reconstruct L312-371 without the bootstrap)
    sdn = types.SimpleNamespace(m2a=ctx.m2a, levels=lev, s_km=ctx.s_km)
    sdw = types.SimpleNamespace(m2a=small_m2a(ctx.m2a), levels=lev, s_km=ctx.s_km[:1000])
    n_fit = int(ctx.m2a["Y"].shape[1])
    (tau, tau_used, tp), row = R.stage(record, "marea", "lag_fit", lambda: v14_.refit_lags(sdn, rows),
                                       lambda: v14_.refit_lags(sdw, rows),
                                       lambda: v14_.refit_lags(sdn, rows),
                                       inputs_mb=_io_mb(T, n_fit), **kw)
    row.update(n_out=len(tau), note=(f"m2a_rasch on {n_fit:,} lag-fit px ({int(ctx.m2a['nb'])} bands), "
                                     f"per-pixel profile of {P:,} px; band lags "
                                     + ";".join(f"{v:.1f}" for v in tau) + " min"))
    R.emit(row)
    o.update(tau_bands_min=np.asarray(tau).tolist(), tau_used_min=np.asarray(tau_used).tolist(),
             tau_px_median_min=float(np.median(tp)), tau_px_max_min=float(np.max(tp)),
             distinct_pixel_lags=int(len(np.unique(tp))), lag_fit_pixels=n_fit)
    if record == "screened":
        stored = np.asarray(ctx.res.get("tau_min", []), float)
        dt = float(np.max(np.abs(np.asarray(tau) - stored))) if len(stored) == len(tau) else None
        g["lag_fit_vs_result_json_tau_min"] = {"mine_min": np.asarray(tau).tolist(),
                                               "stored_min": stored.tolist(), "max_abs_diff_min": dt,
                                               "ok": dt is not None and dt <= TOL_TAU}
        nd = int(np.sum(tp != ctx.tau_px_all))
        g["pixel_lags_vs_production_tau_px_min"] = {"differing": nd, "of": int(P), "ok": nd == 0}

    # bootstrap resamples (timing only)
    if boot and boot_here:
        n_boot = int(ctx.res.get("n_boot") or N_BOOT_DEFAULT)
        seed = int(ctx.m2a["seed"])
        draws = bootstrap_draws(T, n_boot, seed)
        K = int(min(boot, n_boot))
        taus_b, row = R.stage(record, "marea", "bootstrap_resample",
                              lambda: [bootstrap_resample(ctx.m2a, lev, rows, draws[b], b, seed)
                                       for b in range(K)], inputs_mb=_io_mb(T, n_fit), **kw)
        row.update(n_out=K, note=(f"{K} of the {n_boot} production resamples, single thread, "
                                  f"{BOOT_PX_BAND} px per band; {row['wall_s'] / K:.2f} s per resample; "
                                  "timing only, not in the method total"))
        R.emit(row)
        o.update(bootstrap_resamples=K, bootstrap_n_production=n_boot,
                 bootstrap_s_per_resample=row["wall_s"] / K,
                 bootstrap_taus_min=[np.asarray(t).tolist() for t in taus_b])

    # the per-pixel lags are all in the table (a lag outside it would cost one
    # more call inside the inversion stage; counted in its note)
    lags_px = np.unique(tp)
    missing = [float(v) for v in lags_px if int(v) not in table.L]

    # the inversion of every keep pixel (rules 0 and 3 from one fit)
    nw = int(min(P, WARM_PX))
    chk = {}

    def check_slice():
        _, _, _, _, mism = v14_.invert_subset(ctx.Y0[:, :nw], ctx.C0[:, :nw], lev, rows,
                                              v14_.lag_groups(tp[:nw]), [0, 3], check_mb=True)
        chk["mismatch"] = int(mism)

    def inv():
        if missing:
            table.ensure(missing)
        return v14_.invert_subset(ctx.Y0, ctx.C0, lev, rows, v14_.lag_groups(tp), [0, 3])

    if missing and R.warmup != "none":
        table.ensure(missing)            # the warm-up slice needs them too (then not timed)
    (z, nclear, (lo, hi), t_inv, _), row = R.stage(record, "marea", "inversion", inv, check_slice, inv,
                                                  inputs_mb=_io_mb(T, P), **kw)
    if "mismatch" not in chk:
        check_slice()
    n0, n3 = int(np.isfinite(z[0]).sum()), int(np.isfinite(z[3]).sum())
    row.update(n_out=n3, note=(f"invert_series on {P:,} px x {T} scenes, {len(lags_px)} lag groups; "
                               f"estimates mb0 {n0:,}, mb3 {n3:,}; level range [{lo:+.3f}, {hi:+.3f}] m; "
                               f"{t_inv:.1f} s inside invert_series"
                               + (f"; {len(missing)} lag(s) outside the batched table" if missing else "")))
    R.emit(row)
    g["quality_rule_post_filter_vs_invert_series"] = {"values_differing": chk["mismatch"],
                                                      "slice_px": nw, "ok": chk["mismatch"] == 0}
    if record == "screened":
        g["inversion_vs_stored_product"] = compare_z(z[0], ctx.z_stored)
    o.update(estimates_mb0=n0, estimates_mb3=n3, coverage_mb0=n0 / P, coverage_mb3=n3 / P,
             level_range_m=[lo, hi], seconds_inside_invert_series=t_inv,
             median_clear_obs=float(np.median(nclear)))
    del z, nclear
    gc.collect()


def nidem_group(ctx, record):
    """NIDEM on the same record, pixels, NDWI and mask: pre-flight,
    composites at keep, contours and TIN on the frame."""
    from pyintertidal import waterline_contour as wc

    rows = ctx.rows[record]
    T, P = len(rows), ctx.P
    h = ctx.h_space[rows]
    kw = _rec_kw(ctx, record)
    g = ctx.gates.setdefault(record, {})
    o = ctx.outputs.setdefault(record, {}).setdefault("nidem", {})
    R = ctx.recorder
    nw = int(min(P, WARM_PX))

    def pre(c1=None):
        try:
            j, edges, Z, U, sizes = wc.interval_assignment(h)
        except wc.EmptyIntervalError as e:
            return {"computable": False, "reason": f"empty tidal interval: {e}"}
        counts, _ = wc.clear_counts_by_interval(ObsRows(ctx.C0, ctx.bad, rows, 0, c1), None, j)
        cov = wc.interval_coverage(counts, raise_error=False)
        out = {"computable": bool(cov["passed"]), "j": j, "edges": edges, "Z": Z, "U": U,
               "sizes": sizes, "counts": counts, "coverage": cov}
        if not cov["passed"]:
            out["reason"] = (f"expected valid share {cov['expected_valid_share']:.3g} "
                             f"({cov['expected_valid_px']:,} of {cov['px']:,} px) < "
                             f"{cov['min_valid_share']} (intervals {cov['intervals_below_floor']}); "
                             f"|S_j| = {np.asarray(sizes).tolist()}")
        return out

    pf, row = R.stage(record, "nidem", "preflight", pre, lambda: pre(nw), pre,
                      inputs_mb=T * P / 2 ** 20, **kw)
    computable = pf["computable"]
    cov = pf.get("coverage") or {}
    row.update(n_out=cov.get("expected_valid_px"),
               note=("computable" if computable else "NOT COMPUTABLE: " + pf["reason"])
               + (f"; |S_j| = {np.asarray(pf['sizes']).tolist()}" if "sizes" in pf and computable else ""))
    R.emit(row)
    g["nidem_intervals_vs_stored_run"] = nidem_intervals_gate(
        ctx.stored[record], computable, pf.get("sizes"), pf.get("Z"), pf.get("U"), h)
    o.update(computable=computable, reason=pf.get("reason"),
             sizes=(np.asarray(pf["sizes"]).tolist() if "sizes" in pf else None),
             expected_valid_share=cov.get("expected_valid_share"),
             share_px_with_obs_per_interval=cov.get("share_px_with_ge1_clear_obs"))
    if not computable:
        for st in ("composites", "contours_tin"):
            R.emit(R.skipped(record, "nidem", st, "not_computable", note=pf["reason"], **kw))
        return

    def comp_fn():
        ck = composites_keep(ctx.Y0, ctx.C0, ctx.bad, rows, pf["j"])
        return to_frame(ck, ctx.keep, ctx.shape), ck["count"]

    (comp, cnt_keep), row = R.stage(
        record, "nidem", "composites", comp_fn,
        lambda: composites_keep(ctx.Y0, ctx.C0, ctx.bad, rows, pf["j"], 0, nw), comp_fn,
        inputs_mb=_io_mb(T, P), **kw)
    row.update(n_out=int(np.all(cnt_keep >= 1, axis=0).sum()),
               note=f"9 per-interval median composites at {P:,} keep px, on the {ctx.frame_px:,} px frame")
    R.emit(row)
    g["nidem_preflight_counts_vs_composites"] = {
        "ok": bool(np.array_equal(cnt_keep.astype(np.int64), pf["counts"].astype(np.int64)))}
    del cnt_keep
    syn = synthetic_composites()

    def tin():
        try:
            return wc.nidem(comp, pf["Z"], pf["U"])
        except wc.NoClearObservationsInIntervalError as e:
            return {"not_computable": str(e)}

    res_n, row = R.stage(record, "nidem", "contours_tin", tin, lambda: wc.nidem(*syn), tin,
                         inputs_mb=sum(v.nbytes for v in comp.values() if isinstance(v, np.ndarray)) / 2 ** 20,
                         **kw)
    if "not_computable" in res_n:
        row.update(status="not_computable", note=res_n["not_computable"])
        R.emit(row)
        o.update(computable=False, reason=res_n["not_computable"])
        return
    filt_keep = res_n["filtered"].ravel()[ctx.keep]
    nf = int(np.isfinite(filt_keep).sum())
    row.update(n_out=nf, note=(f"{res_n['n_vertices']:,} contour vertices, empty levels "
                               f"{res_n['empty_levels']}; TIN over the {ctx.frame_px:,} px frame; "
                               f"filtered at keep {nf:,} px"))
    R.emit(row)
    o.update(n_vertices=int(res_n["n_vertices"]), filtered_keep_px=nf, coverage_filtered=nf / P,
             empty_levels=res_n["empty_levels"])
    kind, s = stored_nidem_state(ctx.stored[record])
    if kind == "npz" and "filtered_keep" in s and len(s["filtered_keep"]) == P:
        o["agreement_with_stored_filtered_keep"] = agreement(filt_keep, s["filtered_keep"])
    del comp, res_n, filt_keep
    gc.collect()


def nidem_frame_group(ctx, record):
    """NIDEM as published (v7's path without its verification checks): the
    cube streamed with the ITEM pixel mask, pre-pass, pre-flight,
    composites, contours and TIN. I/O included."""
    from pyintertidal import waterline_contour as wc

    rows = ctx.rows[record]
    T, P = len(rows), ctx.P
    h = ctx.h_space[rows]
    ti = ctx.ti_space[rows]
    kw = _rec_kw(ctx, record)
    g = ctx.gates.setdefault(record, {})
    o = ctx.outputs.setdefault(record, {}).setdefault("nidem_frame", {})
    R = ctx.recorder
    q, row = R.stage(record, "nidem_frame", "prepass",
                     lambda: wc.stream_scene_quality(ctx.cube, ti, ctx.keep, rule="item", verbose=False), **kw)
    row.update(n_out=int(q["n_obs_keep"].sum()),
               note=f"{ctx.cube}: {T} scenes, whole frames (SCL, B03, B08), ITEM mask at keep")
    R.emit(row)
    try:
        j, edges, Z, U, sizes = wc.interval_assignment(h)
        reason = None
    except wc.EmptyIntervalError as e:
        j = Z = U = sizes = None
        reason = f"empty tidal interval: {e}"
    cov = None
    if reason is None:
        def pre():
            counts, _ = wc.clear_counts_by_interval(q["obs_keep"], None, j)
            return wc.interval_coverage(counts, raise_error=False)

        cov, row = R.stage(record, "nidem_frame", "preflight", pre, **kw)
        if not cov["passed"]:
            reason = (f"expected valid share {cov['expected_valid_share']:.3g} < {cov['min_valid_share']} "
                      f"(intervals {cov['intervals_below_floor']}); |S_j| = {np.asarray(sizes).tolist()}")
        row.update(n_out=cov["expected_valid_px"], note="computable" if reason is None else
                   "NOT COMPUTABLE: " + reason)
        R.emit(row)
    computable = reason is None
    o.update(computable=computable, reason=reason)
    kind, s = stored_nidem_state(ctx.stored[record])
    if not computable:
        for st in ("composites", "contours_tin"):
            R.emit(R.skipped(record, "nidem_frame", st, "not_computable", note=reason, **kw))
        g["nidem_frame_status_vs_stored_run"] = {"stored": kind, "ok": kind in (None, "not_computable"),
                                                 "checked": kind is not None}
        return
    comp, row = R.stage(record, "nidem_frame", "composites",
                        lambda: wc.stream_composites(ctx.cube, ti, j, rule="item", verbose=False), **kw)
    row.update(note=f"streamed from the cube in row blocks of {comp['row_block']}")
    R.emit(row)
    res_n, row = R.stage(record, "nidem_frame", "contours_tin", lambda: wc.nidem(comp, Z, U), **kw)
    nf = int(np.isfinite(res_n["filtered"].ravel()[ctx.keep]).sum())
    row.update(n_out=nf, note=f"{res_n['n_vertices']:,} contour vertices")
    R.emit(row)
    npz = ctx.stored[record]["npz"]
    if os.path.exists(npz):
        with np.load(npz, allow_pickle=False) as z:
            same = bool(np.array_equal(res_n["filtered"], z["filtered"], equal_nan=True))
        g["nidem_frame_filtered_vs_stored_bitwise"] = {"stored": npz, "ok": same}
    else:
        g["nidem_frame_filtered_vs_stored_bitwise"] = {"checked": False, "reason": f"no {npz}", "ok": True}
    o.update(n_vertices=int(res_n["n_vertices"]), filtered_keep_px=nf)
    del comp, res_n, q
    gc.collect()


def marea_extract_group(ctx):
    """MAREA's input preparation from the cube (marea.extract: two passes
    over every cube scene), checked against the stored extraction."""
    from pyintertidal import marea

    R = ctx.recorder
    ex, row = R.stage(None, "marea_extract", "extract", lambda: marea.extract(ctx.cube),
                      n_px=ctx.P, frame_px=ctx.frame_px)
    T_cube = int(ex["Y"].shape[0])
    row.update(n_scenes=T_cube, n_out=int(len(ex["keep"])),
               note=f"{ctx.cube}: every one of {T_cube} cube scenes, twice (water frequency, then Y and C)")
    R.emit(row)
    same_keep = bool(np.array_equal(ex["keep"], ctx.keep))
    ok_y = ok_c = False
    if same_keep:
        Ye = np.nan_to_num(ex["Y"][ctx.ti_space], nan=0.0)
        ok_y = bool(np.array_equal(Ye, ctx.Y0))
        ok_c = bool(np.array_equal(ex["C"][ctx.ti_space], ctx.C0))
    ctx.gates["marea_extract_vs_stored_extraction"] = {"keep_equal": same_keep, "Y_equal_on_rows": ok_y,
                                                       "C_equal_on_rows": ok_c,
                                                       "ok": same_keep and ok_y and ok_c}
    del ex
    gc.collect()


def run_group(ctx, name, fn, *a, **kw):
    """Run a stage group; an exception is logged and recorded (a 'failed'
    row) and the run goes on with the next group."""
    try:
        return fn(ctx, *a, **kw)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"[{ctx.site}] GROUP FAILED {name}: {e!r}\n{tb}")
        ctx.failures.append({"group": name, "error": repr(e), "traceback": tb})
        ctx.recorder.emit(ctx.recorder.skipped(a[0] if a and isinstance(a[0], str) else None, name, "-",
                                               "failed", note=repr(e)[:300]))
        gc.collect()
        return None


def git_head():
    return v14.git_head()


def cmd_time(args):
    """The controlled timing of one site; the thread limits it sets on this
    process are restored when it ends (a programmatic caller keeps its own)."""
    holder = {}
    try:
        return _cmd_time(args, holder)
    finally:
        lim = holder.get("limiter")
        if lim is not None:
            try:
                lim.restore_original_limits()
            except Exception:  # noqa: BLE001
                pass


def _cmd_time(args, holder):
    site = args.site
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGE_GROUPS]
    if bad:
        raise SystemExit(f"--stages: unknown {bad}; choose from {STAGE_GROUPS}")
    args.records = [r.strip() for r in args.records.split(",") if r.strip()]
    if not args.records or any(r not in RECORDS for r in args.records):
        raise SystemExit(f"--records: choose from {RECORDS}")
    out = guard_out(args.out)
    tag = args.tag or ""
    stem = os.path.join(out, f"timing_{site}" + (f"_{tag}" if tag else ""))
    csv_path, json_path = stem + ".csv", stem + ".json"
    for p in (csv_path, json_path):
        keep_previous(p)
    preload_libraries()
    holder["limiter"] = pin_threads(1)        # kept alive for the whole run
    trep = thread_report()
    settings = {"experiment": "v18_compute_cost", "command": "time", "site": site,
                "site_name": SITE_NAMES.get(site, site), "argv": sys.argv[1:], "status": "running",
                "started": _dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
                "script": v14.file_info(os.path.relpath(os.path.abspath(__file__), ROOT)),
                "machine": machine_info(), "threads": trep, "records": args.records, "stages": stages,
                "warmup": args.warmup, "boot": int(args.boot), "boot_records": args.boot_records,
                "tag": tag, "max_load": args.max_load}
    gates = {"threads_pinned_to_1": threads_gate(trep)}
    l1 = load1()
    gates["machine_quiet"] = {"load1": l1, "max_load": args.max_load,
                              "ok": (l1 is None or args.max_load < 0 or l1 <= args.max_load),
                              "checked": l1 is not None and args.max_load >= 0}
    settings["gates"] = gates
    refuse = [k for k in ("threads_pinned_to_1", "machine_quiet") if not gates[k]["ok"]]
    if refuse == ["threads_pinned_to_1"] and args.allow_threads:
        refuse = []
        settings["threads_gate_overridden"] = True
    if refuse:
        settings["status"] = "refused"
        v14.write_json(json_path, settings)
        raise SystemExit(f"[{site}] refused before any timing: {refuse} (see {json_path})")
    log(f"[{site}] v18 controlled timing: records {args.records}, stages {stages}, warm-up {args.warmup}, "
        f"threads {threads_text(trep)}, load1 {l1}")
    rec = Recorder(csv_path, site, tag, args.warmup, threads_text(trep))
    ctx = setup(args, rec)
    settings["inputs"] = ctx.meta
    settings["cube"] = ctx.cube
    settings["stored_nidem"] = ctx.stored
    v14.write_json(json_path, settings)
    boot_records = [r.strip() for r in args.boot_records.split(",") if r.strip()]
    for record in ctx.records:
        if "core" in stages:
            run_group(ctx, "scene_levels", scene_levels_group, record)
            if record in getattr(ctx, "tables", {}):
                run_group(ctx, "marea", marea_group, record, boot=int(args.boot),
                          boot_here=record in boot_records)
            run_group(ctx, "nidem", nidem_group, record)
        if "nidem_frame" in stages:
            run_group(ctx, "nidem_frame", nidem_frame_group, record)
        settings.update(gates={**gates, **ctx.gates}, outputs=ctx.outputs, failures=ctx.failures,
                        progress={"records_done": ctx.records[:ctx.records.index(record) + 1]})
        v14.write_json(json_path, settings)
    if "marea_extract" in stages:
        run_group(ctx, "marea_extract", marea_extract_group)
    all_gates = {**gates, **ctx.gates}
    fails = failed_gates(all_gates)
    peaks = [r.get("peak_rss_mb") for r in rec.rows if r.get("peak_rss_mb") is not None
             and np.isfinite(r.get("peak_rss_mb"))]
    settings.update(gates=all_gates, outputs=ctx.outputs, failures=ctx.failures, failed_gates=fails,
                    finished=_dt.datetime.now().isoformat(timespec="seconds"),
                    elapsed_minutes=round((time.time() - _T0) / 60, 2),
                    peak_rss_timed_stages_mb=max(peaks) if peaks else None,
                    status=("complete" if not ctx.failures and not fails else
                            "complete_with_failed_gates" if not ctx.failures else "group_failed"))
    v14.write_json(json_path, settings)
    log(f"[{site}] {settings['status']}: {len(rec.rows)} rows -> {csv_path}, {json_path}"
        + (f"; FAILED GATES {fails}" if fails else "") + (f"; FAILED GROUPS {[f['group'] for f in ctx.failures]}"
                                                           if ctx.failures else ""))
    if ctx.failures:
        raise SystemExit(4)
    if fails:
        raise SystemExit(2)


# ─────────────────────────────────────────────────────────────────────────────
#  harvest: costs already logged
# ─────────────────────────────────────────────────────────────────────────────

def _iso(s):
    """datetime from an ISO text ('Z' accepted), or None."""
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def overlap_counts(intervals):
    """For {key: (start, end)}, how many intervals (itself included) overlap
    each one; None where an end is unknown."""
    out = {}
    for k, (a, b) in intervals.items():
        if a is None or b is None:
            out[k] = None
            continue
        out[k] = sum(1 for (c, d) in intervals.values()
                     if c is not None and d is not None and c < b and d > a)
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hrow(**kw):
    r = {c: None for c in HARVEST_COLUMNS}
    r.update(kw)
    return r


def harvest_v14(v14_dir):
    """Rows from the v14 settings JSONs (gate timings: the full screened
    record inverted on the scored pixels, the refit on the full record, the
    boundary calls, the setup) and the CSVs (per-job seconds against N)."""
    rows, files = [], []
    data, intervals = {}, {}
    for p in sorted(glob.glob(os.path.join(v14_dir, "*_settings.json"))):
        d = json.load(open(p, encoding="utf-8"))
        data[p] = d
        r = (d.get("runs") or [{}])[-1]
        intervals[p] = (_iso(r.get("started")), _iso(r.get("finished")))
        files.append(p)
    conc = overlap_counts(intervals)
    for p, d in data.items():
        site, lags = d.get("site"), d.get("lags")
        tim = d.get("timing") or {}
        thr = (d.get("blas") or {}).get("OMP_NUM_THREADS")
        cond = f"v14 curves ({lags} lags): {thr} BLAS threads, {conc[p]} v14 processes running at once"
        common = dict(site=site, method="MAREA", source=f"v14 settings ({lags})", condition=cond,
                      threads=thr, concurrent_runs=conc[p], path=p)
        N, Px = tim.get("gate_N"), tim.get("inverted_px")
        w = tim.get("gate_inversion_seconds_mb0")
        if w is not None and N and Px:
            rows.append(_hrow(**common, record="screened", stage="inversion", quantity="wall_s", value=w,
                              unit="s", n_scenes=N, n_px=Px,
                              note="full screened record on the scored pixels, time inside invert_series (mb 0)"))
            rows.append(_hrow(**common, record="screened", stage="inversion", quantity="us_per_px_scene",
                              value=1e6 * w / (N * Px), unit="us", n_scenes=N, n_px=Px,
                              note="wall / (pixels x scenes)"))
        if tim.get("refit_seconds_full_screened") is not None:
            rows.append(_hrow(**common, record="screened", stage="lag_fit", quantity="wall_s",
                              value=tim["refit_seconds_full_screened"], unit="s", n_scenes=N,
                              n_px=(d.get("lag_fit") or {}).get("n_pixels"),
                              note="v14.refit_lags on the full screened record (production lag-fit pixels)"))
        calls = tim.get("level_calls") or []
        if calls:
            rows.append(_hrow(**common, record="all", stage="boundary_levels", quantity="wall_s",
                              value=sum(float(c.get("seconds", 0)) for c in calls), unit="s",
                              note=(f"{len(calls)} batched boundary call(s): "
                                    + "; ".join(f"{c.get('n_lags')} lags x {c.get('instants')} instants"
                                                for c in calls))))
        if tim.get("load_seconds") is not None:
            rows.append(_hrow(**{**common, "method": "shared"}, record=None, stage="v14_setup",
                              quantity="wall_s", value=tim["load_seconds"], unit="s",
                              note="v14.load_site: record, truth, streaming of the scored columns"))
    # per-job seconds against N, one linear fit per (site, lags, pool)
    try:
        import pandas as pd
    except ImportError:                      # pragma: no cover
        return rows, files
    for p in sorted(glob.glob(os.path.join(v14_dir, "*_fixed.csv")) + glob.glob(os.path.join(v14_dir, "*_refit.csv"))):
        files.append(p)
        df = pd.read_csv(p)
        if df.empty:
            continue
        df = df[df["mb"] == df["mb"].min()]
        site, lags = str(df["site"].iloc[0]), str(df["lags"].iloc[0])
        sp = os.path.join(v14_dir, f"{site}_{lags}_settings.json")
        thr = (data.get(sp, {}).get("blas") or {}).get("OMP_NUM_THREADS")
        # the inverted pixels (an RTK set has more points than pixels)
        px_inv = (data.get(sp, {}).get("timing") or {}).get("inverted_px")
        for pool, gq in df.groupby("pool"):
            if lags == "refit":
                # the full screened record's refit ran in the gate (cached): excluded
                gq = gq[~((gq["pool"] == "screened") & (gq["N"] == gq["pool_size"]))]
            if len(gq) < 3 or gq["N"].nunique() < 2:
                continue
            A = np.vstack([np.ones(len(gq)), gq["N"].to_numpy(float)]).T
            (a, b), *_ = np.linalg.lstsq(A, gq["seconds"].to_numpy(float), rcond=None)
            n_set = int(px_inv or gq["n_set"].iloc[0])
            what = ("inversion of the scored pixels + scoring" if lags == "fixed"
                    else "lag refit + inversion of the scored pixels + scoring")
            common = dict(site=site, method="MAREA", source=f"v14 jobs ({lags})",
                          condition=f"v14 curves: {thr} BLAS threads, several v14 processes at once",
                          record=pool, threads=thr, n_px=n_set, path=p)
            rows.append(_hrow(**common, stage="job_vs_N", quantity="job_s_per_scene", value=float(b),
                              unit="s", note=f"seconds ~ a + b N over {len(gq)} jobs (N {int(gq['N'].min())}-"
                                             f"{int(gq['N'].max())}): {what}"))
            rows.append(_hrow(**common, stage="job_vs_N", quantity="job_s_intercept", value=float(a), unit="s",
                              note=f"{len(gq)} jobs"))
            rows.append(_hrow(**common, stage="job_vs_N", quantity="job_s_per_scene_per_Mpx",
                              value=float(b) / n_set * 1e6, unit="s", note="b per million pixels"))
    return rows, files


_RE_V7_HEAD = re.compile(r"^\[v7\] (\w+) \((.+?)\) -> (\S+)", re.M)
_RE_V7_GRID = re.compile(r"grid (\d+) x (\d+), (\d+) cube scenes, keep ([\d,]+) px")
_RE_V7_REC = re.compile(r"^\s*record: (\d+) scenes", re.M)
_RE_V7_PRE = re.compile(r"pre-pass \((\d+(?:\.\d+)?) s\)")
_RE_V7_COMP = re.compile(r"\[composites\] rows \d+-(\d+) of (\d+) \((\d+(?:\.\d+)?) s\)")
_RE_V7_NIDEM = re.compile(r"NIDEM: ([\d,]+) contour vertices.*?\((\d+(?:\.\d+)?) s\)")
_RE_V7_TOTAL = re.compile(r"total (\d+(?:\.\d+)?) s, peak (\d+(?:\.\d+)?) MB")
_RE_STATUS = re.compile(r"^(\S+Z) (START|OK|FAIL\w*|DONE\w*|END\w*) nidem (\w+)", re.M)

#: v7's "peak" on Linux: psutil memory_info().rss at the end of the run (the
#: peak working set exists only on Windows)
V7_PEAK_NOTE = "v7 'peak' = psutil RSS at the end of the run on Linux (peak_wset is Windows-only): a lower bound"


def _v7_record(label):
    lab = label.lower()
    if "sensitivity" in lab:
        return "screened"
    if "variant" in lab or "scene rule" in lab:
        return "variant"
    return "all"


def harvest_nidem_logs(log_dir):
    """Rows from v7's console logs (nidem_<site>.log) and, when present, the
    STATUS file of the same run (concurrency)."""
    rows, files = [], []
    st_path = os.path.join(log_dir, "STATUS")
    inter = {}
    if os.path.exists(st_path):
        files.append(st_path)
        text = open(st_path, encoding="utf-8", errors="replace").read()
        for ts, ev, site in _RE_STATUS.findall(text):
            a, b = inter.get(site, (None, None))
            if ev == "START":
                a = _iso(ts)
            elif ev == "OK" or ev.startswith("FAIL"):
                b = _iso(ts)
            inter[site] = (a, b)
    conc = overlap_counts(inter) if inter else {}
    for p in sorted(glob.glob(os.path.join(log_dir, "nidem_*.log"))):
        files.append(p)
        text = open(p, encoding="utf-8", errors="replace").read()
        m = _RE_V7_HEAD.search(text)
        if not m:
            continue
        site, label = m.group(1), m.group(2)
        record = _v7_record(label)
        g = _RE_V7_GRID.search(text)
        r = _RE_V7_REC.search(text)
        n_sc = int(r.group(1)) if r else None
        n_px = int(g.group(4).replace(",", "")) if g else None
        c = conc.get(site)
        cond = ("v7 NIDEM run" + (f", {c} NIDEM runs at once" if c else "")
                + ", threads not pinned (production default)")
        common = dict(site=site, method="NIDEM", source="v7 log", condition=cond, record=record,
                      n_scenes=n_sc, n_px=n_px, concurrent_runs=c, path=p)
        x = _RE_V7_PRE.search(text)
        if x:
            rows.append(_hrow(**common, stage="prepass", quantity="wall_s", value=float(x.group(1)), unit="s",
                              note="truth-free pre-pass over the record scenes (whole frames)"))
        comp = [mm for mm in _RE_V7_COMP.finditer(text) if mm.group(1) == mm.group(2)]
        if comp:
            rows.append(_hrow(**common, stage="composites", quantity="wall_s", value=float(comp[-1].group(3)),
                              unit="s", note="composites streamed from the cube (I/O included)"))
        x = _RE_V7_NIDEM.search(text)
        if x:
            rows.append(_hrow(**common, stage="contours_tin", quantity="wall_s", value=float(x.group(2)),
                              unit="s", note=f"{x.group(1)} contour vertices; relative extents, contours, TIN, filters"))
        x = _RE_V7_TOTAL.search(text)
        if x:
            rows.append(_hrow(**common, stage="total", quantity="wall_s", value=float(x.group(1)), unit="s",
                              note="whole v7 run (record, pre-pass, checks, composites, NIDEM, scoring, writing)"))
            rows.append(_hrow(**common, stage="total", quantity="rss_at_end_mb", value=float(x.group(2)),
                              unit="MB", note=V7_PEAK_NOTE))
        if "NOT COMPUTABLE" in text:
            rows.append(_hrow(**common, stage="status", quantity="not_computable", value=1, unit="flag",
                              note="pre-flight stop: NIDEM not computable on this record"))
    return rows, files


def harvest_nidem_diagnostics(patterns, seen):
    """Rows from nidem_diagnostics.json (runtime_s per stage, the peak)."""
    rows, files = [], []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            sha = _sha(p)
            if sha in seen:
                continue
            seen.add(sha)
            files.append(p)
            d = json.load(open(p, encoding="utf-8"))
            site, mode = d.get("site"), d.get("mode", "primary")
            record = {"primary": "all", "sensitivity": "screened"}.get(mode, mode)
            common = dict(site=site, method="NIDEM", source="v7 nidem_diagnostics.json",
                          condition="v7 NIDEM run, threads not pinned (production default); machine not recorded",
                          record=record, n_scenes=d.get("n_scenes"), path=p)
            if str(d.get("status", "")).upper().startswith("NOT COMPUTABLE"):
                rows.append(_hrow(**common, stage="status", quantity="not_computable", value=1, unit="flag",
                                  note=str(d.get("reason", ""))[:300]))
            for k, v in (d.get("runtime_s") or {}).items():
                rows.append(_hrow(**common, stage=k[:-2] if k.endswith("_s") else k, quantity="wall_s",
                                  value=float(v), unit="s"))
            pk = d.get("peak_memory_mb (process peak working set)")
            if pk is not None:
                rows.append(_hrow(**common, stage="total", quantity="rss_at_end_mb", value=float(pk), unit="MB",
                                  note=V7_PEAK_NOTE + " (on Windows: the peak working set)"))
    return rows, files


_RE_NB_SITE = re.compile(r'SITE\s*=\s*"(\w+)"')
_RE_NB_BOOT = re.compile(r"bootstrap of (\d+) scene resamples \[(\d+(?:\.\d+)?) s\]")
_RE_NB_DONE = re.compile(r"listo en (\d+(?:\.\d+)?) s")
_RE_NB_PX = re.compile(r"([\d,]+) intertidal px")
_RE_NB_REC = re.compile(r"MAREA record: (\d+) of (\d+) cube scenes")
_RE_NB_REC_VVA = re.compile(r"scenes in the MAREA record[^:]*: (\d+)")
_RE_NB_PLAIN = re.compile(r"plain inversion: ([\d,]+) px\s+\[(\d+(?:\.\d+)?) s\]")
_RE_NB_MAREA = re.compile(r"MAREA inversion: ([\d,]+) px, (\d+) distinct lags\s+\[(\d+(?:\.\d+)?) s\]")
#: notebook cells by the call their source makes (a regex: 'pit.water_frequency(' but not
#: 'viz.plot_water_frequency(')
NB_CELLS = ((r"\breference_and_clouds\(", "screening",
             "reference map and per-scene clouds over the transition zone (MAREA's scene rule)"),
            (r"(?<![\w.])(?:pit\.)?water_frequency\(", "water_frequency", "water frequency and intertidal mask"),
            (r"\bmake_boundary\(", "boundary", "boundary: tide model + gauges (cached)"),
            (r"\btides\.series\(", "boundary", "tide heights at the scenes and the reference series"),
            (r"\bmarea\.reconstruct\(", "reconstruct_cell",
             "the MAREA product cell (extraction if not cached + reconstruct)"))


def _cell_text(cell):
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o.get("text", "")))
        elif "text/plain" in (o.get("data") or {}):
            out.append("".join(o["data"]["text/plain"]))
    return "\n".join(out)


def _cell_wall(cell):
    ex = (cell.get("metadata") or {}).get("execution") or {}
    a, b = _iso(ex.get("iopub.execute_input")), _iso(ex.get("shell.execute_reply"))
    return (a, b, (b - a).total_seconds() if a and b else None)


def harvest_notebooks(nb_dir):
    """Rows from the executed production notebooks: per-cell wall times (the
    execution metadata nbconvert records) and the timings they print."""
    rows, files = [], []
    nbs = sorted(glob.glob(os.path.join(nb_dir, "tide_boundary_comparison_*.ipynb"))
                 + glob.glob(os.path.join(nb_dir, "intertidal_topography_villaviciosa.ipynb")))
    parsed, inter = {}, {}
    for p in nbs:
        try:
            nb = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        site = None
        for c in cells[:3]:
            m = _RE_NB_SITE.search("".join(c.get("source", "")))
            if m:
                site = m.group(1)
                break
        walls = [_cell_wall(c) for c in cells]
        starts = [a for a, _, _ in walls if a]
        ends = [b for _, b, _ in walls if b]
        inter[p] = (min(starts) if starts else None, max(ends) if ends else None)
        parsed[p] = (site, cells, walls)
        files.append(p)
    conc = overlap_counts(inter)
    for p, (site, cells, walls) in parsed.items():
        if site is None:
            continue
        cond = (f"production notebook (nbconvert), BLAS threads not pinned, {conc[p]} notebook(s) running at once")
        common = dict(site=site, method="MAREA", source="production notebook", condition=cond,
                      concurrent_runs=conc[p], path=p, record="screened")
        n_sc = n_px = None
        for c, (_, _, w) in zip(cells, walls):
            src = "".join(c.get("source", ""))
            txt = _cell_text(c)
            for key, stage, note in NB_CELLS:
                if re.search(key, src) and w is not None:
                    method = "MAREA" if stage in ("screening", "water_frequency", "reconstruct_cell") else "shared"
                    rows.append(_hrow(**{**common, "method": method}, stage=stage, quantity="wall_s", value=w,
                                      unit="s", note=f"cell wall time: {note}"))
                    break
            m = _RE_NB_REC.search(txt) or _RE_NB_REC_VVA.search(txt)
            if m:
                n_sc = int(m.group(1))
            m = _RE_NB_PX.search(txt)
            if m and n_px is None:
                n_px = int(m.group(1).replace(",", ""))
            if re.search(r"\bmarea\.reconstruct\(", src):
                mb, md = _RE_NB_BOOT.search(txt), _RE_NB_DONE.search(txt)
                if mb:
                    rows.append(_hrow(**common, stage="bootstrap", quantity="wall_s", value=float(mb.group(2)),
                                      unit="s", note=f"{mb.group(1)} scene resamples (joblib thread pool)"))
                if md:
                    rows.append(_hrow(**common, stage="reconstruct", quantity="wall_s", value=float(md.group(1)),
                                      unit="s", note="marea.reconstruct: geometry, bank (one call per shift), "
                                                     "lag fit, bootstrap, inversion (one call per lag), outputs"))
                    if "extracted and saved" in txt and w is not None:
                        rows.append(_hrow(**common, stage="extraction", quantity="wall_s",
                                          value=w - float(md.group(1)), unit="s",
                                          note="cell wall minus reconstruct: marea.extract from the cube, the "
                                               "mouth distance and the compressed npz write"))
            mp_, mm_ = _RE_NB_PLAIN.search(txt), _RE_NB_MAREA.search(txt)
            if mp_:
                rows.append(_hrow(**common, stage="plain_inversion", quantity="wall_s", value=float(mp_.group(2)),
                                  unit="s", n_px=int(mp_.group(1).replace(",", "")),
                                  note="invert_series of every keep pixel, no lag (comparison cell)"))
            if mm_ and mp_:
                rows.append(_hrow(**common, stage="marea_inversion_with_per_lag_calls", quantity="wall_s",
                                  value=float(mm_.group(3)) - float(mp_.group(2)), unit="s",
                                  n_px=int(mm_.group(1).replace(",", "")),
                                  note=f"{mm_.group(2)} lags: one boundary call per lag + the inversion"))
        for r in rows:
            if r["path"] == p:
                r["n_scenes"] = r["n_scenes"] or n_sc
                r["n_px"] = r["n_px"] or n_px
    return rows, files


def harvest_result_jsons(patterns, seen):
    rows, files = [], []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            sha = _sha(p)
            if sha in seen:
                continue
            seen.add(sha)
            d = json.load(open(p, encoding="utf-8"))
            if "duracion_s" not in d:
                continue
            files.append(p)
            site = _site_from_path(p)
            rows.append(_hrow(site=site, method="MAREA", source="production result.json",
                              condition="marea.reconstruct as run in the production notebook", record="screened",
                              stage="reconstruct", quantity="wall_s", value=float(d["duracion_s"]), unit="s",
                              n_scenes=d.get("n_escenas"), n_px=d.get("n_px"),
                              note=f"duracion_s; {d.get('n_boot')} bootstrap resamples included", path=p))
    return rows, files


def _site_from_path(p):
    q = p.replace("\\", "/").lower()
    for s in ("villaviciosa", "escalda", "wadden", "ems", "ferrol"):
        if f"products_{s}" in q:
            return s
    if "products_marea" in q:
        return "villaviciosa"
    return None


_RE_G21_TOTAL = re.compile(r"total (\d+(?:\.\d+)?) s; python peak (\d+(?:\.\d+)?) MB, python \+ R peak "
                           r"(\d+(?:\.\d+)?) MB")
_RE_G21_HEAD = re.compile(r"at (\w+) -> \S+; (\d+) R workers")
_RE_G21_RULE = re.compile(r"scene rule: (\d+) of (\d+)")


def harvest_granadeiro(patterns):
    """Context only: the Granadeiro 2021 totals (its accuracy is withheld
    under review; cost as logged)."""
    rows, files = [], []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            files.append(p)
            text = open(p, encoding="utf-8", errors="replace").read()
            h = _RE_G21_HEAD.search(text)
            t = _RE_G21_TOTAL.search(text)
            if not h:
                continue
            site, workers = h.group(1), int(h.group(2))
            r = _RE_G21_RULE.search(text)
            common = dict(site=site, method="Granadeiro 2021", source="v8 log",
                          condition=f"v8 run, {workers} R worker processes", record="own scene rule",
                          n_scenes=int(r.group(1)) if r else None, path=p,
                          note="context only: accuracy withheld under review")
            if not t:
                rows.append(_hrow(**common, stage="status", quantity="incomplete", value=1, unit="flag"))
                continue
            rows.append(_hrow(**common, stage="total", quantity="wall_s", value=float(t.group(1)), unit="s"))
            rows.append(_hrow(**common, stage="total", quantity="python_peak_mb", value=float(t.group(2)), unit="MB"))
            rows.append(_hrow(**common, stage="total", quantity="python_plus_r_peak_mb", value=float(t.group(3)),
                              unit="MB"))
    return rows, files


def cmd_harvest(args):
    out = guard_out(args.out)
    rows, sources, seen = [], {}, set()
    for d in (args.v14_dir or V14_DIRS):
        r, f = harvest_v14(d)
        rows += r
        sources[f"v14:{d}"] = f
    for d in (args.nidem_log_dir or NIDEM_LOG_DIRS):
        r, f = harvest_nidem_logs(d)
        rows += r
        sources[f"nidem_logs:{d}"] = f
    r, f = harvest_nidem_diagnostics(NIDEM_DIAG_GLOBS, seen)
    rows += r
    sources["nidem_diagnostics"] = f
    for d in (args.notebook_dir or NOTEBOOK_DIRS):
        r, f = harvest_notebooks(d)
        rows += r
        sources[f"notebooks:{d}"] = f
    r, f = harvest_result_jsons(RESULT_JSON_GLOBS, seen)
    rows += r
    sources["result_json"] = f
    r, f = harvest_granadeiro(GRANADEIRO_LOG_GLOBS)
    rows += r
    sources["granadeiro"] = f
    path = os.path.join(out, "harvest.csv")
    keep_previous(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HARVEST_COLUMNS)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in HARVEST_COLUMNS])
    info = {k: [v14.file_info(p) for p in v] for k, v in sources.items()}
    v14.write_json(os.path.join(out, "harvest_sources.json"),
                   {"experiment": "v18_compute_cost", "command": "harvest", "rows": len(rows),
                    "when": _dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
                    "sources": info, "caveats": HARVEST_CAVEATS})
    log(f"harvest: {len(rows)} rows from {sum(len(v) for v in sources.values())} files -> {path}")
    return rows


HARVEST_CAVEATS = [
    "Logged runs were not controlled: the production notebooks ran with unpinned BLAS threads, some of them "
    "at the same time; v14 ran 2 BLAS threads per process beside other v14 processes; the NIDEM same-scenes "
    "runs ran side by side (the concurrent_runs column of harvest.csv says how many). Compare them with care.",
    V7_PEAK_NOTE + ".",
    "The production MAREA reconstruct makes one boundary call per bank shift and per per-pixel lag; "
    "v14 and the controlled timing batch them into one or two calls.",
    "The production bootstrap runs 30 resamples in a joblib thread pool (cpu_count - 2 threads), each "
    "with unpinned BLAS threads.",
]


# ─────────────────────────────────────────────────────────────────────────────
#  report
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path) if path and os.path.exists(path) else None


def load_timing(paths):
    """Every timing CSV, the latest file winning for a repeated
    (site, record, method, stage)."""
    import pandas as pd

    frames = []
    for p in sorted(paths, key=lambda q: os.path.getmtime(q)):
        d = pd.read_csv(p)
        d["file"] = p
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["record"] = df["record"].fillna("")
    df["tag"] = df["tag"].fillna("").astype(str)
    return df.drop_duplicates(subset=["site", "tag", "record", "method", "stage"], keep="last")


def summarize_timing(df):
    """One row per (site, record, method): the computing time of its stages
    (``compute_s``), its tide-model calls (``tide_s``: shared scene_levels
    and, for MAREA, lagged_levels) and their sum (``wall_s``), CPU, the
    largest working memory of its computing stages (``work_mb``) and of its
    tide calls (``tide_work_mb``), the largest peak RSS, and its status."""
    import pandas as pd

    rows = []
    if df is None or df.empty:
        return rows

    def _max(d, k):
        v = d[k].to_numpy(float)
        v = v[np.isfinite(v)]
        return float(v.max()) if v.size else float("nan")

    for (site, record), g in df.groupby(["site", "record"]):
        if record == "":
            continue
        sh = g[(g["method"] == "shared") & (g["stage"] == "scene_levels")]
        for method, stages in METHOD_STAGES.items():
            gm = g[(g["method"] == method) & (g["stage"].isin(stages))]
            if gm.empty:
                continue
            gt = g[(g["method"] == method) & (g["stage"].isin(TIDE_STAGES.get(method, ())))]
            if method in WITH_SHARED:
                gt = pd.concat([gt, sh])
            ok, okt = gm[gm["status"] == "ok"], gt[gt["status"] == "ok"]
            nc = bool((gm["status"] == "not_computable").any())
            missing = len(set(stages) - set(gm["stage"])) > 0
            failed = bool((gm["status"] == "failed").any()) or (missing and not nc)
            status = "not_computable" if nc else "failed" if failed else "ok"
            first = gm.iloc[0]
            comp = float(ok["wall_s"].sum())
            tide = float(okt["wall_s"].sum())
            rows.append(dict(
                site=site, record=record, method=method, status=status,
                n_scenes=int(first["n_scenes"]) if np.isfinite(first["n_scenes"]) else None,
                n_px=int(first["n_px"]) if np.isfinite(first["n_px"]) else None,
                frame_px=int(first["frame_px"]) if np.isfinite(first["frame_px"]) else None,
                compute_s=comp if status == "ok" else float("nan"),
                tide_s=tide, n_tide_calls=int(len(okt)),
                wall_s=comp + tide if status == "ok" else float("nan"),
                cpu_s=float(pd.concat([ok, okt])["cpu_s"].sum()) if status == "ok" else float("nan"),
                detect_s=comp + tide if status == "not_computable" else float("nan"),
                work_mb=_max(ok, "work_mb"), tide_work_mb=_max(okt, "work_mb"),
                peak_rss_mb=_max(pd.concat([ok, okt]), "peak_rss_mb"),
                stages="; ".join(f"{r.stage} {r.wall_s:.1f} s" for r in pd.concat([okt, ok]).itertuples()),
                note="; ".join(str(x) for x in gm.loc[gm["status"] != "ok", "note"].dropna().unique())[:400]))
        b = g[(g["method"] == "marea") & (g["stage"] == "bootstrap_resample") & (g["status"] == "ok")]
        for r in b.itertuples():
            k = max(int(r.n_out), 1) if np.isfinite(r.n_out) else 1
            rows.append(dict(site=site, record=record, method="marea_bootstrap", status="ok",
                             n_scenes=int(r.n_scenes), n_px=int(r.n_px), frame_px=int(r.frame_px),
                             compute_s=float(r.wall_s) / k, tide_s=float("nan"), n_tide_calls=0,
                             wall_s=float(r.wall_s) / k, cpu_s=float(r.cpu_s) / k, detect_s=float("nan"),
                             work_mb=float(r.work_mb), tide_work_mb=float("nan"), peak_rss_mb=float(r.peak_rss_mb),
                             stages=f"{k} resample(s), {float(r.wall_s) / k:.1f} s each", note="per resample"))
    return rows


def _h(harv, **kw):
    """Harvest rows matching every given column value."""
    if harv is None:
        return harv
    m = np.ones(len(harv), bool)
    for k, v in kw.items():
        m &= (harv[k].astype(str) == str(v)).to_numpy()
    return harv[m]


def _hval(harv, **kw):
    d = _h(harv, **kw)
    if d is None or d.empty:
        return None
    v = pd_float(d["value"].iloc[0])
    return v


def pd_float(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def as_run_rows(harv):
    """Per site, the as-run (logged) pipeline of each method, one row each."""
    out = []
    if harv is None or harv.empty:
        return out
    for site in SITE_ORDER:
        hs = harv[harv["site"].astype(str) == site]
        if hs.empty:
            continue
        rec_ = _hval(hs, source="production result.json", stage="reconstruct", quantity="wall_s")
        rec_nb = _hval(hs, source="production notebook", stage="reconstruct", quantity="wall_s")
        boot = _hval(hs, source="production notebook", stage="bootstrap", quantity="wall_s")
        scr = _hval(hs, source="production notebook", stage="screening", quantity="wall_s")
        wf = _hval(hs, source="production notebook", stage="water_frequency", quantity="wall_s")
        ext = _hval(hs, source="production notebook", stage="extraction", quantity="wall_s")
        nb = _h(hs, source="production notebook")
        conc = (int(nb["concurrent_runs"].dropna().astype(float).max())
                if nb is not None and not nb.empty and nb["concurrent_runs"].notna().any() else None)
        n_sc = _h(hs, source="production result.json")
        if rec_ is not None or rec_nb is not None:
            parts = [f"reconstruct {(rec_ or rec_nb):.0f} s" + (f" (bootstrap {boot:.0f} s)" if boot else "")]
            if scr is not None:
                parts.append(f"scene screening {scr:.0f} s")
            if wf is not None:
                parts.append(f"water frequency {wf:.0f} s")
            if ext is not None:
                parts.append(f"extraction {ext:.0f} s")
            out.append(dict(site=site, method="MAREA (production notebook)", record="screened",
                            n_scenes=(int(float(n_sc["n_scenes"].iloc[0])) if n_sc is not None and not n_sc.empty
                                      and pd_float(n_sc["n_scenes"].iloc[0]) else None),
                            total_s=rec_ or rec_nb, breakdown="; ".join(parts), memory="not logged",
                            condition=(f"BLAS threads not pinned; {conc} notebook(s) at once" if conc else
                                       "BLAS threads not pinned")))
        for record, lab in (("all", "NIDEM (v7, all scenes)"), ("screened", "NIDEM (v7, MAREA's record)")):
            hn = hs[(hs["method"] == "NIDEM") & (hs["record"].astype(str) == record)]
            if hn.empty:
                continue
            # the diagnostics JSON (exact stage seconds) first, the console log second
            srcs = (hn[hn["source"] == "v7 nidem_diagnostics.json"], hn[hn["source"] == "v7 log"])
            nc = not hn[(hn["quantity"] == "not_computable")].empty

            def g_(stages, quantity="wall_s", srcs=srcs):
                for src in srcs:
                    for st in stages:
                        d = src[(src["stage"] == st) & (src["quantity"] == quantity)]
                        if not d.empty:
                            return pd_float(d["value"].iloc[0])
                return None
            pre = g_(("scene_quality_prepass", "prepass"))
            comp = g_(("composites",))
            tin = g_(("nidem", "contours_tin"))
            tot = g_(("total",))
            mem_v = g_(("total",), "rss_at_end_mb")
            c = hn["concurrent_runs"].dropna()
            parts = [f"pre-pass {pre:.0f} s" if pre is not None else None,
                     f"composites (streamed) {comp:.0f} s" if comp is not None else None,
                     f"contours + TIN {tin:.0f} s" if tin is not None else None]
            n_scn = pd_float(hn["n_scenes"].dropna().iloc[0]) if hn["n_scenes"].notna().any() else None
            out.append(dict(site=site, method=lab, record=record, n_scenes=int(n_scn) if n_scn else None,
                            total_s=tot, breakdown=("NOT COMPUTABLE (pre-flight stop); " if nc else "")
                            + "; ".join(p for p in parts if p),
                            memory=(f"RSS at end {mem_v:.0f} MB" if mem_v else "not logged"),
                            condition=("threads not pinned" + (f"; {int(float(c.iloc[0]))} NIDEM runs at once"
                                                               if not c.empty else ""))))
        g21 = hs[hs["method"] == "Granadeiro 2021"]
        if not g21.empty:
            tot = _hval(g21, quantity="wall_s")
            pk = _hval(g21, quantity="python_plus_r_peak_mb")
            if tot is not None:
                out.append(dict(site=site, method="Granadeiro 2021 (context)", record="own scene rule",
                                n_scenes=(int(float(g21["n_scenes"].iloc[0])) if pd_float(g21["n_scenes"].iloc[0])
                                          else None), total_s=tot, breakdown="whole v8 run",
                                memory=f"python + R peak {pk:.0f} MB" if pk else "",
                                condition=str(g21["condition"].iloc[0]) + "; accuracy withheld under review"))
    return out


def v14_min_scenes(v14_dir, thresholds=(0.5, 0.9), mb=3):
    """Smallest N whose median coverage over the v14 subsets reaches each
    threshold (fixed lags, rule ``mb``), per site and pool."""
    import pandas as pd

    out = {}
    for site in SITE_ORDER:
        p = os.path.join(v14_dir, f"{site}_fixed.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["mb"] == mb]
        for pool, g in d.groupby("pool"):
            med = g.groupby("N")["coverage"].median().sort_index()
            out[(site, pool)] = {t: (int(med.index[med >= t][0]) if (med >= t).any() else None)
                                 for t in thresholds}
            out[(site, pool)]["pool_size"] = int(g["pool_size"].iloc[0])
    return out


def gauges_per_site(path=GAUGES_CSV):
    d = _read_csv(path)
    if d is None or "site" not in d or "judge" not in d:
        return {}
    return {str(k): int(v) for k, v in d.groupby("site")["judge"].nunique().items()}


def nidem_status_per_site(path=SAME_SCENES_SUMMARY):
    d = _read_csv(path)
    out = {}
    if d is None:
        return out
    for r in d.itertuples():
        rec = "all" if "all observations" in str(r.method) else "screened" if "MAREA's record" in str(r.method) else None
        if rec and str(r.method).startswith("NIDEM"):
            out[(str(r.site), rec)] = ("NOT COMPUTABLE" if "NOT COMPUTABLE" in str(r.scenes) else
                                       f"computable ({int(float(r.scenes))} scenes)")
    return out


def site_facts(timing_json, v14_dir, gauges, nidem_status):
    """Per site: records, pixels, frame, in-estuary gauges, NIDEM status, the
    v14 minimum scenes for coverage."""
    mins = v14_min_scenes(v14_dir) if os.path.isdir(v14_dir) else {}
    rows = []
    for site in SITE_ORDER:
        tj = timing_json.get(site) or {}
        st_path = os.path.join(v14_dir, f"{site}_fixed_settings.json")
        st = json.load(open(st_path, encoding="utf-8")) if os.path.exists(st_path) else {}
        pools = (tj.get("inputs") or {}).get("pools") or st.get("pools") or {}
        ext = ((tj.get("inputs") or {}).get("inputs") or st.get("inputs") or {}).get("extract") or {}
        grid = ext.get("grid")
        r = dict(site=site, site_name=SITE_NAMES.get(site, site),
                 scenes_screened=(pools.get("screened") or {}).get("size"),
                 scenes_all=(pools.get("all") or {}).get("size"),
                 keep_px=ext.get("n_keep"), frame_px=(int(grid[0]) * int(grid[1]) if grid else None),
                 in_estuary_gauges=gauges.get(site, 0),
                 nidem_all=nidem_status.get((site, "all")), nidem_screened=nidem_status.get((site, "screened")))
        for pool in ("screened", "all"):
            m = mins.get((site, pool)) or {}
            r[f"marea_min_N_cov50_{pool}"] = m.get(0.5)
            r[f"marea_min_N_cov90_{pool}"] = m.get(0.9)
        rows.append(r)
    return rows


# ── the bilingual text ───────────────────────────────────────────────────────

T = {
    "en": {
        "title": "Computing cost and inputs: MAREA vs NIDEM",
        "rec_screened": "MAREA's record", "rec_all": "all scenes",
        "fig_title": "Computing cost on equal inputs (one thread)",
        "fig_sub": ("Same scenes, NDWI, cloud mask and pixels. MAREA: lag fit + inversion of every pixel. "
                    "NIDEM: pre-flight + composites + contours/TIN."),
        "ax_time": "Computing time (s, log scale)", "ax_mem": "Working memory (MB, log scale)",
        "p_time": "Time", "p_mem": "Memory",
        "nc": "NIDEM not computable", "not_timed": "not timed yet",
        "foot": ("×n = MAREA's computing time over NIDEM's. Single thread (BLAS/OpenMP pinned to 1), one timed "
                 "run after a warm-up. Tide-model calls (MAREA 2, NIDEM 1) and the bootstrap are not included: see "
                 "the table. Working memory = peak RSS during a stage minus RSS at its start (shared inputs "
                 "excluded)."),
        "h_ctrl": "Controlled timing on equal inputs (single thread)",
        "h_stage": "Stage breakdown (controlled timing)",
        "h_boot": "MAREA bootstrap (uncertainty of the lags; not in the totals)",
        "h_asrun": "As run (production logs, not controlled)",
        "h_scaling": "MAREA cost against the number of scenes (v14 jobs)",
        "h_inputs": "What each method needs",
        "h_facts": "Per-site facts",
        "h_gates": "Gates of the controlled runs",
        "h_caveats": "Caveats",
        "no_timing": ("No controlled timing yet: run the `time` command per site (see the server commands); "
                      "the tables below come from the logs only."),
        "cols_ctrl": ["Site", "Record", "Scenes", "Keep px", "MAREA compute (s)", "NIDEM compute (s)",
                      "MAREA / NIDEM", "MAREA with tide calls (s)", "NIDEM with tide calls (s)",
                      "Tide calls / s (MAREA · NIDEM)", "MAREA work (MB)", "NIDEM work (MB)", "Notes"],
        "cols_stage": ["Site", "Record", "Method", "Stage", "Wall (s)", "CPU (s)", "Work (MB)", "Note"],
        "cols_boot": ["Site", "Scenes", "s per resample (1 thread)", "x production resamples (s)",
                      "Production bootstrap as logged (s)"],
        "cols_asrun": ["Site", "Method", "Record", "Scenes", "Total (s)", "Breakdown", "Memory", "Conditions"],
        "cols_scaling": ["Site", "Lags", "Pool", "s per scene (job)", "s per scene per inverted Mpx (fixed lags)",
                         "Intercept (s)", "Conditions"],
        "cols_facts": ["Site", "Scenes (MAREA's record / all)", "Keep px", "Frame px", "In-estuary gauges",
                       "NIDEM, all scenes", "NIDEM, MAREA's record",
                       "MAREA min N, coverage >= 90 % (record / all)"],
        "cols_inputs": ["Requirement", "MAREA", "No delay (MAREA without lag)", "NIDEM (ITEM rules)"],
        "cols_gates": ["Site", "Gate", "OK", "Detail"],
        "yes": "yes", "no": "no", "not_run": "not run", "nc_short": "NOT COMPUTABLE",
    },
    "es": {
        "title": "Coste de cálculo y entradas: MAREA frente a NIDEM",
        "rec_screened": "registro de MAREA", "rec_all": "todas las escenas",
        "fig_title": "Coste de cálculo con las mismas entradas (un hilo)",
        "fig_sub": ("Mismas escenas, NDWI, máscara de nubes y píxeles. MAREA: ajuste del retardo + inversión de "
                    "cada píxel. NIDEM: comprobación previa + compuestos + contornos/TIN."),
        "ax_time": "Tiempo de cálculo (s, escala log)", "ax_mem": "Memoria de trabajo (MB, escala log)",
        "p_time": "Tiempo", "p_mem": "Memoria",
        "nc": "NIDEM no calculable", "not_timed": "aún sin medir",
        "foot": ("×n = tiempo de cálculo de MAREA entre el de NIDEM. Un hilo (BLAS/OpenMP fijados a 1), una "
                 "ejecución medida tras un calentamiento. No incluye las llamadas al modelo de marea (MAREA 2, "
                 "NIDEM 1) ni el bootstrap: ver la tabla. Memoria de trabajo = pico de RSS en la etapa menos la RSS "
                 "al empezarla (sin las entradas comunes)."),
        "h_ctrl": "Medición controlada con las mismas entradas (un hilo)",
        "h_stage": "Desglose por etapas (medición controlada)",
        "h_boot": "Bootstrap de MAREA (incertidumbre de los retardos; fuera de los totales)",
        "h_asrun": "Tal como se ejecutó (registros de producción, sin controlar)",
        "h_scaling": "Coste de MAREA según el número de escenas (trabajos de v14)",
        "h_inputs": "Qué necesita cada método",
        "h_facts": "Datos por sitio",
        "h_gates": "Comprobaciones de las mediciones controladas",
        "h_caveats": "Advertencias",
        "no_timing": ("Aún no hay medición controlada: ejecutar `time` por sitio (ver los comandos del servidor); "
                      "las tablas de abajo salen solo de los registros."),
        "cols_ctrl": ["Sitio", "Registro", "Escenas", "Píxeles", "Cálculo MAREA (s)", "Cálculo NIDEM (s)",
                      "MAREA / NIDEM", "MAREA con llamadas de marea (s)", "NIDEM con llamadas de marea (s)",
                      "Llamadas de marea / s (MAREA · NIDEM)", "Memoria MAREA (MB)", "Memoria NIDEM (MB)", "Notas"],
        "cols_stage": ["Sitio", "Registro", "Método", "Etapa", "Reloj (s)", "CPU (s)", "Memoria (MB)", "Nota"],
        "cols_boot": ["Sitio", "Escenas", "s por remuestreo (1 hilo)", "x remuestreos de producción (s)",
                      "Bootstrap de producción registrado (s)"],
        "cols_asrun": ["Sitio", "Método", "Registro", "Escenas", "Total (s)", "Desglose", "Memoria", "Condiciones"],
        "cols_scaling": ["Sitio", "Retardos", "Conjunto", "s por escena (trabajo)",
                         "s por escena y Mpx invertido (retardos fijos)", "Constante (s)", "Condiciones"],
        "cols_facts": ["Sitio", "Escenas (registro de MAREA / todas)", "Píxeles", "Píxeles del marco",
                       "Mareógrafos en el estuario", "NIDEM, todas las escenas", "NIDEM, registro de MAREA",
                       "N mínimo de MAREA, cobertura >= 90 % (registro / todas)"],
        "cols_inputs": ["Requisito", "MAREA", "Sin retardo (MAREA sin retardo)", "NIDEM (reglas ITEM)"],
        "cols_gates": ["Sitio", "Comprobación", "OK", "Detalle"],
        "yes": "sí", "no": "no", "not_run": "no ejecutado", "nc_short": "NO CALCULABLE",
    },
}

#: what each method needs, as implemented here (rows: EN / ES; columns MAREA,
#: no delay, NIDEM). Numbers in {braces} are filled from the data.
INPUT_ROWS = [
    {"en": ("Tide information",
            "One water-level series at the mouth (the boundary: a tide model at the site, or its consensus with "
            "the nearest coastal gauges outside the estuary), read at t - tau(s) per pixel; tau(s) is estimated "
            "from the images themselves",
            "The same series, read at the overpass time t (no lag)",
            "The same series at the overpass time t, one level per scene for the whole frame (published NIDEM: "
            "a tide model at the tidal-polygon centroid)"),
     "es": ("Información de marea",
            "Una serie de nivel en la boca (el contorno: un modelo de marea en el sitio, o su consenso con los "
            "mareógrafos costeros más cercanos fuera del estuario), leída en t - tau(s) por píxel; tau(s) se estima "
            "de las propias imágenes",
            "La misma serie, leída en la hora de paso t (sin retardo)",
            "La misma serie en la hora de paso t, un nivel por escena para todo el marco (NIDEM publicado: un "
            "modelo de marea en el centroide del polígono de marea)")},
    {"en": ("In-estuary gauges",
            "Not needed; used only to validate the lag ({gauges})", "Not needed", "Not needed"),
     "es": ("Mareógrafos dentro del estuario",
            "No hacen falta; solo validan el retardo ({gauges})", "No hacen falta", "No hacen falta")},
    {"en": ("Per-pixel tide model",
            "Not needed. The inversion takes any level series per group of pixels, so a per-pixel model or "
            "gauge-derived lags can replace the fitted lag without changing it",
            "Not needed",
            "Not needed. Scenes are pooled by one level per scene, so a spatially varying tide is outside the "
            "method"),
     "es": ("Modelo de marea por píxel",
            "No hace falta. La inversión acepta cualquier serie de niveles por grupo de píxeles, así que un modelo "
            "por píxel o retardos de mareógrafos pueden sustituir al retardo ajustado sin cambiarla",
            "No hace falta",
            "No hace falta. Las escenas se agrupan por un nivel por escena, así que una marea que varía en el "
            "espacio queda fuera del método")},
    {"en": ("Geometry",
            "Distance from the mouth along the water, from the archive's own sea mask (which image edge is sea "
            "is a site parameter)", "None", "None (spatial interpolation of waterlines)"),
     "es": ("Geometría",
            "Distancia a la boca por el agua, a partir de la máscara de mar del propio archivo (qué borde de la "
            "imagen es mar es un parámetro del sitio)", "Ninguna", "Ninguna (interpolación espacial de líneas de agua)")},
    {"en": ("Cloud mask per pixel",
            "Sentinel-2 SCL classes 4-7 (vegetation, bare soil, water, unclassified)", "The same",
            "ITEM pixel-quality mask mapped to SCL (masked 0, 1, 3, 8, 9, 10); the controlled timing gives it "
            "MAREA's mask (equal inputs)"),
     "es": ("Máscara de nubes por píxel",
            "Clases SCL de Sentinel-2 4-7 (vegetación, suelo desnudo, agua, sin clasificar)", "La misma",
            "Máscara de calidad de ITEM traducida a SCL (enmascara 0, 1, 3, 8, 9, 10); en la medición controlada "
            "usa la de MAREA (mismas entradas)")},
    {"en": ("Scene selection",
            "Transition-zone cloud screening (<= 10 % cloud over the transition zone): {scr} scenes",
            "The same", "All observations: {all} scenes"),
     "es": ("Selección de escenas",
            "Cribado de nubes en la zona de transición (<= 10 % de nube): {scr} escenas", "La misma",
            "Todas las observaciones: {all} escenas")},
    {"en": ("Pixels processed",
            "Per pixel, on the intertidal mask (water frequency 0.10-0.90): {keep} px", "The same",
            "The whole frame (contours and the TIN need the 2-D image): {frame} px"),
     "es": ("Píxeles procesados",
            "Por píxel, en la máscara intermareal (frecuencia de agua 0,10-0,90): {keep} px", "Los mismos",
            "Todo el marco (los contornos y la TIN necesitan la imagen 2-D): {frame} px")},
    {"en": ("Minimum data per pixel",
            ">= 8 clear observations; quality rule: >= 3 clear observations with the level below the estimate "
            "and >= 3 above", "The same",
            ">= 1 valid observation in each of the 9 tide intervals"),
     "es": ("Datos mínimos por píxel",
            ">= 8 observaciones despejadas; regla de calidad: >= 3 con el nivel por debajo de la cota y >= 3 por "
            "encima", "Los mismos", ">= 1 observación válida en cada uno de los 9 intervalos de marea")},
    {"en": ("Minimum record",
            "Production floor of 40 scenes (reconstruct default 80); the lag fit needs > 5,000 pixels with a mouth "
            "distance and bands of >= 500 px, otherwise the product falls back to no delay. Coverage >= 90 % "
            "from {min90} scenes (v14 subsets)",
            "Production floor of 40 scenes",
            ">= 1 scene in each of the 9 intervals of the observed tidal range; our pre-flight guard (not a "
            "published rule) also asks >= 1 % of the pixels to be valid"),
     "es": ("Registro mínimo",
            "Mínimo de producción de 40 escenas (80 por defecto en reconstruct); el ajuste del retardo necesita "
            "> 5.000 píxeles con distancia a la boca y bandas de >= 500 px, si no el producto queda sin retardo. "
            "Cobertura >= 90 % desde {min90} escenas (subconjuntos de v14)",
            "Mínimo de producción de 40 escenas",
            ">= 1 escena en cada uno de los 9 intervalos del rango de marea observado; nuestra comprobación previa "
            "(no es una regla publicada) pide además >= 1 % de píxeles válidos")},
    {"en": ("Can it be NOT COMPUTABLE?",
            "Only below 3,000 intertidal pixels or below the scene floor; otherwise pixels without enough data "
            "stay empty (coverage < 100 %)", "The same",
            "Yes: {nc_text}"),
     "es": ("¿Puede quedar NO CALCULABLE?",
            "Solo por debajo de 3.000 píxeles intermareales o del mínimo de escenas; si no, los píxeles sin datos "
            "suficientes quedan vacíos (cobertura < 100 %)", "Lo mismo", "Sí: {nc_text}")},
    {"en": ("Estimated from the data",
            "Band lags (L-BFGS-B over the band increments) and per pixel the elevation and transition width "
            "(grid sweep)", "Per pixel the elevation and width", "Nothing (fixed published constants)"),
     "es": ("Estimado de los datos",
            "Retardos por banda (L-BFGS-B sobre los incrementos) y por píxel la cota y la anchura de la transición "
            "(barrido de rejilla)", "Por píxel la cota y la anchura", "Nada (constantes publicadas fijas)")},
    {"en": ("How the computation scales",
            "Independent per pixel, in chunks of 4,000 (memory per chunk is constant; tiles or parallel workers "
            "need no change); the lag fit uses <= 1,200 px per band",
            "Independent per pixel",
            "Per-interval composites per pixel, then one triangulation of all the frame's contour vertices"),
     "es": ("Cómo escala el cálculo",
            "Independiente por píxel, en bloques de 4.000 (memoria constante por bloque; teselas o procesos en "
            "paralelo sin cambios); el ajuste del retardo usa <= 1.200 px por banda",
            "Independiente por píxel",
            "Compuestos por intervalo y píxel, y una triangulación de todos los vértices de contorno del marco")},
    {"en": ("Output",
            "Per-pixel elevation and its spread; the lag profile tau(s) with a bootstrap interval",
            "Per-pixel elevation and its spread",
            "DEM interpolated between 9 waterline levels; per-interval tide spread as uncertainty; confidence filter"),
     "es": ("Salida",
            "Cota por píxel y su dispersión; el perfil de retardo tau(s) con intervalo bootstrap",
            "Cota por píxel y su dispersión",
            "MDT interpolado entre 9 niveles de línea de agua; dispersión de la marea por intervalo como "
            "incertidumbre; filtro de confianza")},
]


def _fmt_n(v, lang="en"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    s = f"{int(round(float(v))):,}"
    return s.replace(",", ".") if lang == "es" else s


def _fmt_s(v, lang="en"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    v = float(v)
    s = (f"{v:,.0f}" if v >= 100 else f"{v:,.1f}" if v >= 1 else f"{v:.2f}" if v >= 0.01 or v == 0
         else f"{v:.1e}")
    if lang == "es":
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s


def inputs_table(facts, lang):
    """The requirements table (rows) with the per-site numbers filled in."""
    by = {f["site"]: f for f in facts}

    def per_site(key, fmt=_fmt_n):
        vals = [fmt(by.get(s, {}).get(key), lang) for s in SITE_ORDER]
        return " / ".join(vals)
    sites = " / ".join(SITE_NAMES[s].split(" (")[0] for s in SITE_ORDER)
    gauges = ", ".join(f"{SITE_NAMES[s].split(' (')[0]} {by.get(s, {}).get('in_estuary_gauges', 0)}"
                       for s in SITE_ORDER)
    min90 = per_site("marea_min_N_cov90_screened")
    ncs = [s for s in SITE_ORDER if str(by.get(s, {}).get("nidem_all", "")).startswith("NOT")]
    nc_text = (("Ems with all its scenes (interval 1 of the observed tidal range holds one scene, valid on almost "
                "no pixel); computable at Ems on MAREA's 54-scene record") if lang == "en" else
               ("el Ems con todas sus escenas (el intervalo 1 del rango de marea observado tiene una sola escena, "
                "válida en casi ningún píxel); calculable en el Ems con el registro de 54 escenas de MAREA"))
    if not ncs:
        nc_text = ("when an interval is empty or almost unobserved" if lang == "en" else
                   "cuando un intervalo está vacío o casi sin observar")
    fill = dict(gauges=gauges, scr=per_site("scenes_screened"), all=per_site("scenes_all"),
                keep=per_site("keep_px"), frame=per_site("frame_px"), min90=min90, nc_text=nc_text)
    rows = []
    for r in INPUT_ROWS:
        cells = [c.format(**fill) for c in r[lang]]
        rows.append(cells)
    head = T[lang]["cols_inputs"]
    note = (f"Per-site values in the order {sites}." if lang == "en" else f"Valores por sitio en el orden {sites}.")
    return head, rows, note


def md_table(head, rows):
    esc = [[str(c).replace("|", "\\|").replace("\n", " ") for c in r] for r in rows]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in esc]
    return "\n".join(out)


def _rec_label(record, n, lang):
    lab = T[lang]["rec_screened"] if record == "screened" else T[lang]["rec_all"]
    return f"{lab} ({n})" if n else lab


def controlled_rows(summary, lang):
    """The headline table: per site and record, MAREA and NIDEM computing
    time, the same with their tide-model calls, the ratio, working memory."""
    by = {(r["site"], r["record"], r["method"]): r for r in summary}
    rows = []
    for site in SITE_ORDER:
        for record in RECORDS:
            m, n = by.get((site, record, "marea")), by.get((site, record, "nidem"))
            if m is None and n is None:
                continue
            ref = m or n
            ok = m and n and m["status"] == "ok" and n["status"] == "ok"
            ratio = m["compute_s"] / n["compute_s"] if ok and n["compute_s"] > 0 else None

            def cell(r, key):
                if r is None:
                    return T[lang]["not_run"]
                if r["status"] == "not_computable":
                    return f"{T[lang]['nc_short']} ({_fmt_s(r['detect_s'], lang)} s)" if key == "wall_s" else "-"
                return _fmt_s(r[key], lang) if r["status"] == "ok" else r["status"]

            def tide(r):
                return "-" if r is None else f"{r['n_tide_calls']} / {_fmt_s(r['tide_s'], lang)} s"
            notes = "; ".join(x for x in ((m or {}).get("note", ""), (n or {}).get("note", "")) if x)[:200]
            rat = (f"{ratio:.2f}".replace(".", ",") if lang == "es" else f"{ratio:.2f}") if ratio else "-"
            rows.append([SITE_NAMES.get(site, site), _rec_label(record, None, lang), _fmt_n(ref["n_scenes"], lang),
                         _fmt_n(ref["n_px"], lang), cell(m, "compute_s"), cell(n, "compute_s"), rat,
                         cell(m, "wall_s"), cell(n, "wall_s"), f"{tide(m)} · {tide(n)}",
                         _fmt_s((m or {}).get("work_mb"), lang), _fmt_s((n or {}).get("work_mb"), lang), notes])
    return rows


def stage_rows(df, lang):
    rows = []
    if df is None:
        return rows
    order = {s: i for i, s in enumerate(("setup", "load_inputs", "scene_levels", "lagged_levels", "lag_fit",
                                         "bootstrap_resample", "inversion", "prepass", "preflight",
                                         "composites", "contours_tin", "extract"))}
    d = df.copy()
    d["_o"] = d["stage"].map(lambda s: order.get(s, 99))
    d["_s"] = d["site"].map(lambda s: SITE_ORDER.index(s) if s in SITE_ORDER else 9)
    d["_r"] = d["record"].map(lambda r: {"": 0, "screened": 1, "all": 2}.get(r, 3))
    for r in d.sort_values(["_s", "_r", "_o"]).itertuples():
        tag = str(r.tag) if isinstance(r.tag, str) and r.tag else ""
        rows.append([SITE_NAMES.get(r.site, r.site), r.record or "-", r.method + (f" [{tag}]" if tag else ""),
                     r.stage,
                     _fmt_s(r.wall_s, lang) if r.status in ("ok", "setup") else r.status,
                     _fmt_s(r.cpu_s, lang), _fmt_s(r.work_mb, lang), str(r.note if isinstance(r.note, str) else "")[:160]])
    return rows


def _log_ticks(lo, hi):
    """Readable ticks of a log axis: 1-2-5 steps when it spans under two
    decades, decades otherwise."""
    k0, k1 = int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi)))
    steps = (1, 2, 5) if math.log10(hi / lo) < 2.2 else (1,)
    return [m * 10.0 ** k for k in range(k0, k1 + 1) for m in steps if lo <= m * 10.0 ** k <= hi]


def cost_figure(summary, lang, out_stem):
    """Dot plot in two panels (computing time, working memory): one header
    row per site and one row per record; MAREA circle (PDeep), NIDEM diamond
    (PSand); the MAREA / NIDEM time ratio as the one direct label per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    t = T[lang]
    by = {(r["site"], r["record"], r["method"]): r for r in summary}
    timed = {r["site"] for r in summary}
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    items = []
    for s in SITE_ORDER:
        items.append(("h", s, None))
        items += [("r", s, r) for r in RECORDS]
    ny = len(items)
    import textwrap

    fig = plt.figure(figsize=(8.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.13, left=0.215, right=0.975,
                          top=0.80, bottom=0.165)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    styles = {"marea": dict(marker="o", color=PAL["deep"], ms=8.5),
              "nidem": dict(marker="D", color=PAL["sand"], ms=7.5)}

    def good(p, key):
        return p is not None and p["status"] == "ok" and p.get(key) is not None and np.isfinite(p[key]) and p[key] > 0

    vals = {"compute_s": [], "work_mb": []}
    for (_, s, r) in items:
        for m in ("marea", "nidem"):
            for k in vals:
                if r and good(by.get((s, r, m)), k):
                    vals[k].append(by[(s, r, m)][k])
    dec = "," if lang == "es" else "."
    for ax, key, xl in ((axes[0], "compute_s", t["ax_time"]), (axes[1], "work_mb", t["ax_mem"])):
        v = vals[key] or [1.0, 10.0]
        lo, hi = min(v) / 2.0, max(v) * (3.2 if key == "compute_s" else 2.0)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(ny - 0.5, -0.6)
        for i, (kind, s, r) in enumerate(items):
            if kind == "h":
                if i:
                    ax.axhline(i - 0.5, color=PAL["rule"], lw=0.8, zorder=0)
                continue
            ax.axhline(i, color=PAL["mist"], lw=0.8, zorder=0)
            pts = {m: by.get((s, r, m)) for m in ("marea", "nidem")}
            xs = [p[key] for p in pts.values() if good(p, key)]
            if len(xs) == 2:
                ax.plot(xs, [i, i], color=PAL["rule"], lw=2.0, zorder=1, solid_capstyle="round")
            for m, p in pts.items():
                if good(p, key):
                    ax.plot([p[key]], [i], ls="none", mec="white", mew=1.6, zorder=3, **styles[m])
            if key != "compute_s":
                continue
            m_, n_ = pts["marea"], pts["nidem"]
            if good(m_, key) and good(n_, key):
                ratio = m_[key] / n_[key]
                ax.annotate(f"×{ratio:.1f}".replace(".", dec), (max(xs), i), xytext=(9, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=8.6, color=PAL["slate"])
            elif n_ is not None and n_["status"] == "not_computable":
                xm = m_[key] if good(m_, key) else lo * 1.5
                # beside the MAREA dot, on the side with room
                left = math.log(xm / lo) / math.log(hi / lo) > 0.5
                ax.annotate(t["nc"], (xm, i), xytext=(-11 if left else 9, 0), textcoords="offset points",
                            va="center", ha="right" if left else "left", fontsize=8.6, color=PAL["slate"],
                            style="italic")
            elif s not in timed:
                ax.text(math.sqrt(lo * hi), i, t["not_timed"], va="center", ha="center", fontsize=8.6,
                        color=PAL["slate"], style="italic")
        def tick_text(x, _pos):
            if x >= 1000:
                s = f"{x:,.0f}"
                return s.replace(",", ".") if lang == "es" else s
            return f"{x:g}".replace(".", dec)

        # ticks kept off the panel edges (no label runs into the next panel)
        ax.xaxis.set_major_locator(FixedLocator(_log_ticks(lo * 1.08, hi / 1.08)))
        ax.xaxis.set_major_formatter(FuncFormatter(tick_text))
        ax.xaxis.set_minor_locator(NullLocator())
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(PAL["rule"])
        ax.tick_params(axis="x", colors=PAL["slate"], labelcolor=PAL["ink"], length=3, labelsize=8.8)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", which="major", color=PAL["rule"], lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xl, color=PAL["ink"], fontsize=9.3)
        ax.set_yticks(range(ny))
    labels = []
    for (kind, s, r) in items:
        if kind == "h":
            labels.append(SITE_NAMES[s].split(" (")[0])
            continue
        ref = by.get((s, r, "marea")) or by.get((s, r, "nidem"))
        labels.append(_rec_label(r, ref["n_scenes"] if ref else None, lang))
    axes[0].set_yticklabels(labels, fontsize=8.8)
    for lab, (kind, _, _) in zip(axes[0].get_yticklabels(), items):
        lab.set_color(PAL["deep"] if kind == "h" else PAL["ink"])
        if kind == "h":
            lab.set_fontweight("bold")
            lab.set_fontsize(9.6)
    axes[1].set_yticklabels([])
    fig.text(0.02, 0.962, t["fig_title"], fontsize=13, color=PAL["deep"], weight="bold", ha="left", va="top")
    fig.text(0.02, 0.918, textwrap.fill(t["fig_sub"], 118), fontsize=8.4, color=PAL["slate"], ha="left",
             va="top", linespacing=1.3)
    handles = [Line2D([], [], ls="none", mec="white", mew=1.6, label="MAREA", **styles["marea"]),
               Line2D([], [], ls="none", mec="white", mew=1.6, label="NIDEM", **styles["nidem"])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.205, 0.872), ncol=2, frameon=False,
               fontsize=9.5, handletextpad=0.4, columnspacing=1.6)
    fig.text(0.02, 0.016, textwrap.fill(t["foot"], 150), fontsize=7.6, color=PAL["slate"], ha="left",
             va="bottom", linespacing=1.3)
    fig.savefig(out_stem + ".png", dpi=200)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)
    return out_stem + ".png"


def write_markdown(path, lang, summary, timing_df, timing_json, asrun, harv, facts, gates_rows, fig_png):
    t = T[lang]
    L = [f"# {t['title']}", ""]
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L.append(("Generated by `experiments/v18_compute_cost.py report` on " if lang == "en" else
              "Generado por `experiments/v18_compute_cost.py report` el ") + stamp + ".")
    L.append("")
    machines = sorted({(j.get("machine") or {}).get("node") or "?" for j in timing_json.values()})
    if summary:
        L += [f"## {t['h_ctrl']}", ""]
        mi = next(iter(timing_json.values())).get("machine", {}) if timing_json else {}
        L.append((f"Machine(s): {', '.join(machines)}; CPU {mi.get('cpu_model') or mi.get('machine')}; "
                  f"{mi.get('cpu_count')} cores; RAM {mi.get('ram_gb')} GB; numpy {mi.get('numpy')}. ")
                 if lang == "en" else
                 (f"Máquina(s): {', '.join(machines)}; CPU {mi.get('cpu_model') or mi.get('machine')}; "
                  f"{mi.get('cpu_count')} núcleos; RAM {mi.get('ram_gb')} GB; numpy {mi.get('numpy')}. "))
        L.append("")
        L.append(md_table(t["cols_ctrl"], controlled_rows(summary, lang)))
        L.append("")
        if fig_png:
            L += [f"![{t['fig_title']}]({os.path.basename(fig_png)})", ""]
        boot = [r for r in summary if r["method"] == "marea_bootstrap"]
        if boot:
            L += [f"### {t['h_boot']}", ""]
            brows = []
            for r in boot:
                nprod = (timing_json.get(r["site"], {}).get("outputs", {}).get("screened", {})
                         .get("marea", {}).get("bootstrap_n_production") or N_BOOT_DEFAULT)
                logged = _hval(_h(harv, site=r["site"]) if harv is not None else None,
                               source="production notebook", stage="bootstrap", quantity="wall_s")
                brows.append([SITE_NAMES.get(r["site"], r["site"]), _fmt_n(r["n_scenes"], lang),
                              _fmt_s(r["wall_s"], lang), f"{_fmt_s(r['wall_s'] * nprod, lang)} (x{nprod})",
                              _fmt_s(logged, lang)])
            L += [md_table(t["cols_boot"], brows), ""]
        L += [f"### {t['h_stage']}", "", md_table(t["cols_stage"], stage_rows(timing_df, lang)), ""]
        if gates_rows:
            L += [f"### {t['h_gates']}", "", md_table(t["cols_gates"], gates_rows), ""]
    else:
        L += [f"## {t['h_ctrl']}", "", t["no_timing"], ""]
    if asrun:
        L += [f"## {t['h_asrun']}", ""]
        L.append(md_table(t["cols_asrun"],
                          [[SITE_NAMES.get(r["site"], r["site"]), r["method"],
                            _rec_label(r["record"], None, lang) if r["record"] in RECORDS else r["record"],
                            _fmt_n(r["n_scenes"], lang), _fmt_s(r["total_s"], lang), r["breakdown"], r["memory"],
                            r["condition"]] for r in asrun]))
        L.append("")
    if harv is not None and not harv.empty:
        sc = harv[(harv["stage"] == "job_vs_N") & (harv["quantity"] == "job_s_per_scene")]
        if not sc.empty:
            rows = []
            for r in sc.itertuples():
                same = ((harv["stage"] == "job_vs_N") & (harv["site"] == r.site) & (harv["source"] == r.source)
                        & (harv["record"] == r.record))
                ic = harv[same & (harv["quantity"] == "job_s_intercept")]
                pm = harv[same & (harv["quantity"] == "job_s_per_scene_per_Mpx")]
                lags = r.source.split("(")[-1].rstrip(")")
                rows.append([SITE_NAMES.get(r.site, r.site), lags, r.record, _fmt_s(pd_float(r.value), lang),
                             # per inverted Mpx only where the job is the inversion (fixed lags)
                             _fmt_s(pd_float(pm["value"].iloc[0]), lang) if lags == "fixed" and not pm.empty else "-",
                             _fmt_s(pd_float(ic["value"].iloc[0]) if not ic.empty else None, lang), r.condition])
            L += [f"## {t['h_scaling']}", "", md_table(t["cols_scaling"], rows), ""]
    head, rows, note = inputs_table(facts, lang)
    L += [f"## {t['h_inputs']}", "", md_table(head, rows), "", note, ""]
    frows = []
    for f in facts:
        frows.append([f["site_name"], f"{_fmt_n(f['scenes_screened'], lang)} / {_fmt_n(f['scenes_all'], lang)}",
                      _fmt_n(f["keep_px"], lang), _fmt_n(f["frame_px"], lang), str(f["in_estuary_gauges"]),
                      f["nidem_all"] or "-", f["nidem_screened"] or "-",
                      f"{_fmt_n(f['marea_min_N_cov90_screened'], lang)} / {_fmt_n(f['marea_min_N_cov90_all'], lang)}"])
    L += [f"## {t['h_facts']}", "", md_table(t["cols_facts"], frows), ""]
    L += [f"## {t['h_caveats']}", ""] + [f"- {c}" for c in HARVEST_CAVEATS] + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def gate_rows_of(runs, lang):
    """Every gate of every timing run (a run's tag after the site name)."""
    rows = []
    order = {s: i for i, s in enumerate(SITE_ORDER)}
    for j in sorted(runs, key=lambda q: (order.get(q.get("site"), 9), q.get("tag") or "")):
        site = j.get("site")
        label = SITE_NAMES.get(site, site) + (f" [{j['tag']}]" if j.get("tag") else "")

        def walk(prefix, d, label=label):
            for k, v in d.items():
                if isinstance(v, dict) and "ok" in v:
                    det = {kk: vv for kk, vv in v.items() if kk not in ("ok", "pools", "env", "mine_min",
                                                                         "stored_min", "sizes_mine", "sizes_stored")}
                    rows.append([label, f"{prefix}{k}", T[lang]["yes"] if v["ok"] else T[lang]["no"],
                                 json.dumps(v14._clean(det), default=str)[:220]])
                elif isinstance(v, dict):
                    walk(f"{prefix}{k}.", v)
        walk("", j.get("gates") or {})
    return rows


def cmd_report(args):
    import pandas as pd

    out = guard_out(args.out)
    harv_path = args.harvest or os.path.join(out, "harvest.csv")
    harv = pd.read_csv(harv_path) if os.path.exists(harv_path) else None
    tpaths = args.timing or sorted(glob.glob(os.path.join(out, "timing_*.csv")))
    tdf = load_timing(tpaths)
    runs = []
    for p in sorted(glob.glob(os.path.join(out, "timing_*.json")), key=os.path.getmtime):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("site") and j.get("status") != "refused":
            runs.append(j)
    # one run per site for the facts: the untagged (core) run when there is one
    tjson = {}
    for j in sorted(runs, key=lambda q: 0 if q.get("tag") else 1):
        tjson[j["site"]] = j
    summary = summarize_timing(tdf)
    asrun = as_run_rows(harv)
    v14_dir = args.v14_dir or V14_DIRS[0]
    facts = site_facts(tjson, v14_dir, gauges_per_site(), nidem_status_per_site())
    # CSVs
    cc = [dict(r, source="controlled (single thread, equal inputs)") for r in summary]
    cc += [dict(site=r["site"], record=r["record"], method=r["method"], status="as run", n_scenes=r["n_scenes"],
                wall_s=r["total_s"], stages=r["breakdown"], note=f"{r['memory']}; {r['condition']}",
                source="as run (logs)") for r in asrun]
    pd.DataFrame(cc).to_csv(os.path.join(out, "compute_cost.csv"), index=False)
    pd.DataFrame(facts).to_csv(os.path.join(out, "site_facts.csv"), index=False)
    head, rows, _ = inputs_table(facts, "en")
    pd.DataFrame(rows, columns=head).to_csv(os.path.join(out, "inputs_table.csv"), index=False)
    figs = {}
    if summary:
        for lang in ("en", "es"):
            figs[lang] = cost_figure(summary, lang, os.path.join(out, f"compute_cost_{lang}"))
    for lang, name in (("en", "compute_cost.md"), ("es", "compute_cost.es.md")):
        write_markdown(os.path.join(out, name), lang, summary, tdf, tjson, asrun, harv, facts,
                       gate_rows_of(runs, lang), figs.get(lang))
    log(f"report: {len(summary)} controlled rows, {len(asrun)} as-run rows -> {out}")
    return summary, asrun, facts


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    ap = argparse.ArgumentParser(prog="python -m experiments.v18_compute_cost",
                                 description="Computing cost and inputs of MAREA vs NIDEM on equal inputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="costs already logged -> harvest.csv")
    h.add_argument("--out", default=OUT)
    h.add_argument("--v14-dir", nargs="*", default=None)
    h.add_argument("--nidem-log-dir", nargs="*", default=None)
    h.add_argument("--notebook-dir", nargs="*", default=None)
    t = sub.add_parser("time", help="controlled timing of one site (single thread)")
    t.add_argument("site", choices=list(SITE_NAMES))
    t.add_argument("--records", default="screened,all", help="screened,all (default both)")
    t.add_argument("--stages", default="core", help=f"comma list of {STAGE_GROUPS} (default core)")
    t.add_argument("--warmup", choices=WARMUPS, default="small")
    t.add_argument("--boot", type=int, default=2, help="bootstrap resamples to time (0 = none)")
    t.add_argument("--boot-records", default="screened")
    t.add_argument("--max-load", type=float, default=DEFAULT_MAX_LOAD,
                   help="refuse to start above this 1-min load average (POSIX; < 0 disables)")
    t.add_argument("--allow-threads", action="store_true",
                   help="record but do not stop on a thread pool with more than one thread (debugging only)")
    t.add_argument("--tag", default="", help="suffix of the output names (timing_<site>_<tag>)")
    t.add_argument("--out", default=OUT)
    t.add_argument("--product-dir", dest="product_override", default=None)
    t.add_argument("--extract", dest="extract_override", default=None)
    t.add_argument("--comparison-dir", dest="comparison_override", default=None)
    t.add_argument("--nidem-dir", dest="nidem_override", default=None)
    t.add_argument("--cube", default=None, help="cube for nidem_frame / marea_extract (default v7.SITES)")
    r = sub.add_parser("report", help="summary CSV, inputs table, markdown (EN/ES), figure (EN/ES)")
    r.add_argument("--out", default=OUT)
    r.add_argument("--harvest", default=None)
    r.add_argument("--timing", nargs="*", default=None)
    r.add_argument("--v14-dir", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "harvest":
        cmd_harvest(args)
    elif args.cmd == "time":
        cmd_time(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
