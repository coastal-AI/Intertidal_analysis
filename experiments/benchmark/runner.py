# -*- coding: utf-8 -*-
"""v13 benchmark — the unit queue: plan, claims, resume, logs, resources (§1.9, §7.4).

Question
--------
How do several workers on one machine (tmux panes on the server, one
process on the laptop) run thousands of units, each one adapter group on one
record subset, so that any worker can be killed and restarted at any moment,
no unit runs twice at the same time, a finished unit is never recomputed
while its inputs are unchanged, a failed attempt is kept for inspection, and
every unit leaves a log with its stage timings, its peak memory and the
thread settings it ran with?

Pre-registered rules (§3.0, §7.4, FROZEN['threads'])
----------------------------------------------------
* A unit is DONE when, for every product it writes, ``run.json`` holds a
  final status (ok, empty, not_computable) and the same ``inputs_key``
  (product.inputs_key: package, subset, method, variant, config and code
  keys). Done units are skipped; r_failed, budget and error are rerun.
* Before running a unit a worker takes its claim file with
  ``O_CREAT | O_EXCL`` (host, pid, start, token) and touches it every 60 s.
  A claim whose last heartbeat is older than 30 min and whose process is
  gone (another host: by age alone) is stale; ``--steal-stale`` takes it
  over under a short ``.steal`` lock, so two workers never steal one claim.
* Products are published by :func:`product.save`: written to
  ``<dir>.tmp-<host>-<pid>/``, an existing ``<dir>`` (a failed attempt)
  renamed to ``<dir>.old-<utc>``, then ``os.replace``. Nothing is deleted.
* Statuses: the adapter's own products; ``NotComputable`` -> not_computable
  for every product of the unit (with its stage); an R job that exits with
  rc != 0 (outside the GAM, which the adapter maps to its fallback) ->
  r_failed, with the R log tail; :class:`BudgetExceeded` -> budget; any other
  exception -> error, with the traceback. A :class:`Deferred` unit (its
  dependency, e.g. the N = all unit of an F-mode Granadeiro curve, is not
  done yet) writes nothing and is retried by a later pass.
* Threads: the CLI sets OMP/OPENBLAS/MKL_NUM_THREADS before numpy is
  imported (to the largest count of the requested groups); each unit then
  runs with its group's BLAS threads (FROZEN['threads']['blas']) through
  threadpoolctl in this process and through the environment for child
  processes (the R workers). The count is part of the config key, because a
  multithreaded dgemv partitions columns by thread and the last bits of
  ``invert_series`` depend on it. ``--blas-threads`` (tests and smoke only),
  ``--smoke`` and ``--window`` change the results, so they enter the config
  key and mark the products smoke: a smoke product never satisfies a real
  resume.
* Gates: ``run`` refuses a group whose gates have not passed at the site
  (``gates/<site>/<gate>.json`` with ``"verdict": "PASS"`` on the same
  package); smoke runs only warn, and the positive control (site ``pc``) is
  exempt. A method is *validated* (its controlled row may be printed) when
  its gates and its positive control passed (:func:`validated`).
* Peak memory per unit: Linux VmHWM, reset at the start of the unit through
  ``/proc/self/clear_refs`` (else ``ru_maxrss``, the process peak); Windows
  psutil ``peak_wset`` (``GetProcessMemoryInfo`` without psutil), the process
  peak; for R adapters the maximum with RRunner's Python + R tree peak. A
  1-s sampler records the unit's own RSS peak next to it.

Adapter contract (spec §1.10; the adapters arrive in M2/M3)
-----------------------------------------------------------
Every module of ``experiments/benchmark/adapters`` registered in
``adapters.REGISTRY = {GROUP: module}`` exposes ``GROUP``, ``METHODS``,
``VARIANTS``, ``CONFIG`` (JSON-ready), ``run(pkg, unit, out_root, **opts) ->
list[Product]`` and ``gate(pkg, out_root, **opts) -> dict``. Optional
attributes the runner reads: ``DEFAULT_VARIANT`` (else ``VARIANTS[0]`` or
""), ``CODE_DEPS`` (repository paths of the pyintertidal modules and R
script behind the code key), ``outputs(unit) -> [(method, variant)]`` (else
``METHODS`` x ``unit.variant``), ``MODES`` (modes with distinct units, e.g.
("A",) for NIDEM, whose A unit serves both), ``SITES`` (sites the group runs
at) and ``SENS_KEYS`` (sensitivity keys it accepts). ``run`` receives
``work_dir`` (scratch, removed after publishing), ``boot_threads``,
``r_workers``, ``rscript``, ``tmpdir``, ``resume``, ``max_r_hours``,
``project_only``, ``smoke``, ``window``, ``no_wait`` and ``worker_id``; it
may time its stages with :func:`stage`, put ``RRunner.stats`` in
``product.diag["r_runner"]`` and attach ``product.files = {name: path}`` (files
moved into the unit directory, e.g. full-frame rasters at N = all).

A file ``<root>/STOP`` (or ``<root>/STOP-<worker_id>``) stops a queue after
its current unit.
"""
from __future__ import annotations

import contextlib
import contextvars
import csv
import glob
import io
import json
import math
import os
import re
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from experiments.benchmark import config, product
from experiments.benchmark.product import NotComputable, Product, Unit

GROUPS = ("marea_pair", "nidem", "granadeiro", "gm20", "hd26", "baselines", "published")
#: the milestone that delivers each group's adapter (the CLI's message)
MILESTONE = {g: ("M3" if g == "granadeiro" else "M2") for g in GROUPS}
#: methods by group, for the gate and control checks of :func:`validated`
#: (the names of FROZEN['positive_control']['must_pass'])
METHOD_GROUP = {"nodelay": "marea_pair", "marea": "marea_pair", "nidem": "nidem",
                "granadeiro": "granadeiro", "gm20": "gm20", "hd26": "hd26",
                "zero": "baselines", "eot20": "baselines", "got56": "baselines"}
#: the methods with a lag field (the positive control's "every lag method")
LAG_METHODS = ("marea", "granadeiro", "gm20", "hd26")

HEARTBEAT_S = 60.0
STALE_AFTER_S = 30 * 60.0
#: a steal lock older than this was left by a crashed worker
STEAL_LOCK_STALE_S = 10 * 60.0
RSS_SAMPLE_S = 1.0
CLAIM_FORMAT = "v13-claim-1"
CLAIM_DIR = "_claims"
WORK_DIR = "_work"

#: gates every group needs at its site (§4.1), the published loaders excepted
SITE_GATES = ("record", "levels", "keep")
#: the group's own gates at the same site and resolution
GROUP_GATES = {"marea_pair": ("marea",), "nidem": ("nidem",), "granadeiro": (),
               "gm20": (), "hd26": (), "baselines": (), "published": ()}
#: gates that run at one site and grid only and validate the method everywhere
#: (G-granadeiro: paper mode at Villaviciosa, 10 m, on the server)
GLOBAL_GATES = {"granadeiro": (("granadeiro", "villaviciosa", 10),)}
#: sites whose units are exempt from the gate check (the positive control)
GATE_EXEMPT_SITES = (config.CONTROL_SITE,)

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
#: the thread variables when this module was imported: the CLI sets them
#: before importing it, and numpy (imported just above) read them then
_ENV_AT_IMPORT = {k: os.environ.get(k) for k in THREAD_ENV_VARS}

if set(GROUPS) != set(config.BLAS_THREADS) or set(GROUPS) != set(config.METHOD_SECTION):
    raise AssertionError("runner.GROUPS differ from config.BLAS_THREADS / METHOD_SECTION")


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  exceptions an adapter may raise
# ─────────────────────────────────────────────────────────────────────────────

class RFailed(RuntimeError):
    """An R job other than the GAM exited with rc != 0 (status r_failed)."""

    def __init__(self, message, log_tail=None, stage=None):
        super().__init__(message)
        self.log_tail = log_tail
        self.stage = stage


class BudgetExceeded(RuntimeError):
    """Stopped by ``--max-r-hours`` after the projection (status budget;
    resumable)."""

    def __init__(self, message, projected_r_hours=None, stage="projection"):
        super().__init__(message)
        self.projected_r_hours = projected_r_hours
        self.stage = stage


class Deferred(Exception):
    """The unit cannot start yet (a dependency is not done); nothing is
    written and a later pass retries it."""


class ClaimLost(RuntimeError):
    """This worker's claim file is gone or belongs to another worker."""


class ThreadSettingError(RuntimeError):
    """The group's BLAS thread count cannot be enforced in this process."""


#: RRunner's message for a failed job (pyintertidal.granadeiro2021.RRunner.run)
_R_JOB_FAILED = re.compile(r"^R job \S+ failed \(rc=")


def is_r_failure(exc):
    """True for :class:`RFailed` and RRunner's 'R job ... failed (rc=...)'."""
    return isinstance(exc, RFailed) or (isinstance(exc, RuntimeError)
                                        and bool(_R_JOB_FAILED.match(str(exc))))


# ─────────────────────────────────────────────────────────────────────────────
#  memory
# ─────────────────────────────────────────────────────────────────────────────

def _win_memory_counters():
    """``(WorkingSetSize, PeakWorkingSetSize)`` in bytes of this process,
    through GetProcessMemoryInfo (Windows without psutil)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if fn is None:
        fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not fn(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)


def _proc_status_kb(field_name):
    with open("/proc/self/status", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith(field_name + ":"):
                return float(line.split()[1])
    raise OSError(f"{field_name} not in /proc/self/status")


def current_rss_mb():
    """Resident set size of this process now (MB; NaN when unknown)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        pass
    except Exception:                       # psutil present but failing: fall through
        pass
    try:
        if sys.platform.startswith("linux"):
            return _proc_status_kb("VmRSS") / 1024.0
        if os.name == "nt":
            return _win_memory_counters()[0] / 2 ** 20
    except (OSError, ValueError, AttributeError):
        pass
    return float("nan")


def process_peak_rss_mb():
    """Peak RSS of this process (MB; NaN when unknown).

    Linux: VmHWM (the high-water mark :func:`reset_peak_rss` resets), else
    ``ru_maxrss``; other POSIX: ``ru_maxrss``; Windows: psutil ``peak_wset``,
    or ``GetProcessMemoryInfo`` when psutil is not installed.
    """
    if sys.platform.startswith("linux"):
        try:
            return _proc_status_kb("VmHWM") / 1024.0
        except (OSError, ValueError):
            pass
    if os.name == "posix":
        try:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # kilobytes on Linux, bytes on macOS
            return r / 2 ** 20 if sys.platform == "darwin" else r / 1024.0
        except (ImportError, OSError):
            return float("nan")
    if os.name == "nt":
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2 ** 20
        except ImportError:
            pass
        except Exception:
            pass
        try:
            return _win_memory_counters()[1] / 2 ** 20
        except (OSError, AttributeError, ValueError):
            pass
    return float("nan")


def reset_peak_rss():
    """Reset the process high-water mark to the current RSS (Linux >= 4.0,
    ``echo 5 > /proc/self/clear_refs``), so the next :func:`process_peak_rss_mb`
    is the unit's own peak. Returns True when the reset took effect."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
        return True
    except OSError:
        return False


def peak_rss_mb(runner_stats=None):
    """The unit's peak memory (MB): :func:`process_peak_rss_mb`, or its
    maximum with ``RRunner.stats["python_plus_r_peak_mb"]`` (Granadeiro)."""
    p = process_peak_rss_mb()
    if isinstance(runner_stats, Mapping):
        r = runner_stats.get("python_plus_r_peak_mb")
        try:
            r = float(r)
        except (TypeError, ValueError):
            r = float("nan")
        if np.isfinite(r) and (not np.isfinite(p) or r > p):
            p = r
    return float(p)


# ─────────────────────────────────────────────────────────────────────────────
#  threads
# ─────────────────────────────────────────────────────────────────────────────

def group_threads(group):
    """The group's pre-registered BLAS thread count (FROZEN['threads'])."""
    return int(config.FROZEN["threads"]["blas"][group])


def thread_env(env=None):
    """The thread variables of ``env`` (default: os.environ)."""
    env = os.environ if env is None else env
    return {k: env.get(k) for k in THREAD_ENV_VARS}


def _env_threads(env):
    """The BLAS thread count an environment imposes (None = library default)."""
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        v = env.get(k)
        if v not in (None, ""):
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _threadpoolctl():
    try:
        import threadpoolctl
        return threadpoolctl
    except ImportError:
        return None


def _preload_blas():
    """Load scipy's BLAS before limiting it: threadpoolctl limits only the
    libraries already loaded, and scipy ships its own OpenBLAS."""
    try:
        import scipy.linalg  # noqa: F401
    except ImportError:
        pass


def can_enforce_threads(n):
    """``(ok, why)``: can a unit run with ``n`` BLAS threads in this process?"""
    if _threadpoolctl() is not None:
        return True, "threadpoolctl"
    have = _env_threads(_ENV_AT_IMPORT)
    if have == int(n):
        return True, "environment"
    return False, (f"threadpoolctl is not installed and the BLAS was loaded with "
                   f"{have if have is not None else 'its default'} threads, not {n}: "
                   f"install threadpoolctl or run this group in its own process")


def _pools_info(tpc):
    keep = ("user_api", "internal_api", "num_threads", "version", "threading_layer")
    return [{k: p.get(k) for k in keep if k in p} for p in tpc.threadpool_info()]


@contextlib.contextmanager
def blas_limits(n):
    """Run the body with ``n`` BLAS threads: threadpoolctl for the libraries
    of this process, the thread variables for its children (R workers).

    Yields a dict recording the request and the effective pools; the
    environment is restored afterwards. Raises :class:`ThreadSettingError`
    when ``n`` cannot be enforced (no threadpoolctl and another count in the
    environment numpy was loaded with).
    """
    n = int(n)
    if n < 1:
        raise ValueError("the BLAS thread count must be >= 1")
    tpc = _threadpoolctl()
    if tpc is None:
        ok, why = can_enforce_threads(n)
        if not ok:
            raise ThreadSettingError(why)
    saved = thread_env()
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(n)
    try:
        _preload_blas()
        if tpc is None:
            yield {"requested": n, "api": "environment",
                   "effective": [_env_threads(_ENV_AT_IMPORT)], "env_children": n}
        else:
            with tpc.threadpool_limits(limits=n, user_api="blas"):
                pools = _pools_info(tpc)
                eff = sorted({p.get("num_threads") for p in pools
                              if p.get("user_api") == "blas"} - {None})
                yield {"requested": n, "api": "threadpoolctl", "effective": eff,
                       "pools": pools, "env_children": n}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  claims
# ─────────────────────────────────────────────────────────────────────────────

#: claims this process holds: absolute path -> token
_OWNED = {}
_OWNED_LOCK = threading.Lock()


def unit_leaf(unit):
    """``N<NNN>_r<RR>[__<senstag>]`` of a unit (product.unit_dir's leaf)."""
    return os.path.basename(product.unit_dir("x", unit, "x", ""))


def claim_path(root, unit):
    """``<root>/runs/<site>/_claims/<group>[__<variant>]/<family>-<mode>/<leaf>.claim``.

    One claim per unit (a group run may write several products); kept apart
    from the product directories so ``iter_runs`` never sees them.
    """
    g = f"{unit.group}__{unit.variant}" if unit.variant else unit.group
    return "/".join([str(root).rstrip("/\\").replace("\\", "/"), "runs", unit.site,
                     CLAIM_DIR, g, f"{unit.family}-{unit.mode}", unit_leaf(unit) + ".claim"])


def _create_time():
    try:
        import psutil
        return float(psutil.Process().create_time())
    except Exception:                       # absent or unsupported: pid only
        return None


def _claim_info(info=None):
    d = {"format": CLAIM_FORMAT, "host": socket.gethostname(), "pid": os.getpid(),
         "create_time": _create_time(), "token": uuid.uuid4().hex,
         "start_utc": product.utc_stamp()}
    for k, v in dict(info or {}).items():
        d.setdefault(str(k), v)
    return d


def _read_claim(path):
    """The claim's JSON, {} while it is being written, None when absent."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def _pid_alive(pid, create_time=None):
    """True alive, False gone, None unknown (this host only)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            if not psutil.pid_exists(int(pid)):
                return False
            if create_time is not None:
                ct = psutil.Process(int(pid)).create_time()
                if abs(float(ct) - float(create_time)) > 1.0:
                    return False             # the pid was reused by another process
            return True
        except psutil.NoSuchProcess:
            return False
        except Exception:                    # access denied etc.: it exists
            return True
    if os.name == "posix":
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, ValueError, TypeError):
            return None
        return True
    return None


def claim_state(path, stale_after_s=STALE_AFTER_S, now=None):
    """The state of a claim file, or None when there is none.

    Returns
    -------
    dict
        The claim's fields plus ``age_s`` (since the last heartbeat, i.e.
        the file's mtime), ``alive`` (True / False / None = unknown: another
        host, or no way to tell) and ``stale``: older than ``stale_after_s``
        and its process not known to be alive.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    d = _read_claim(path)
    if d is None:
        return None
    now = time.time() if now is None else now
    age = max(0.0, now - st.st_mtime)
    alive = None
    if d.get("host") == socket.gethostname() and d.get("pid") is not None:
        alive = _pid_alive(d["pid"], d.get("create_time"))
    out = dict(d)
    out.update(path=str(path), age_s=age, alive=alive,
               stale=bool(age > float(stale_after_s) and alive is not True))
    return out


def _exclusive_create(path, info):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, (json.dumps(info, indent=1) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    with _OWNED_LOCK:
        _OWNED[os.path.abspath(path)] = info["token"]
    return True


def _steal(path, info, stale_after_s):
    """Take over a stale claim under the ``<path>.steal`` lock."""
    lock = f"{path}.steal"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            if time.time() - os.stat(lock).st_mtime > STEAL_LOCK_STALE_S:
                os.remove(lock)               # left by a crashed stealer; next pass
        except OSError:
            pass
        return False
    os.close(fd)
    try:
        st = claim_state(path, stale_after_s)
        if st is None:                        # released meanwhile: an ordinary claim
            return _exclusive_create(path, info)
        if not st["stale"]:
            return False
        log(f"[claim] taking over a stale claim: {path} (host {st.get('host')}, pid "
            f"{st.get('pid')}, started {st.get('start_utc')}, last heartbeat "
            f"{st['age_s'] / 60:.0f} min ago)")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return _exclusive_create(path, info)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def claim(path, info=None, steal_stale=False, stale_after_s=STALE_AFTER_S):
    """Take the claim file ``path`` exclusively (``O_CREAT | O_EXCL``).

    Parameters
    ----------
    path : str
        The unit's claim (:func:`claim_path`).
    info : dict, optional
        Extra fields recorded in the claim (unit label, worker id).
    steal_stale : bool
        Take over a stale claim (:func:`claim_state`).
    stale_after_s : float
        Heartbeat age after which a claim of a dead process is stale.

    Returns
    -------
    bool
        True when this process now holds the claim.
    """
    path = os.fspath(path)
    info = _claim_info(info)
    if _exclusive_create(path, info):
        return True
    if steal_stale:
        return _steal(path, info, stale_after_s)
    return False


def _owned_token(path):
    with _OWNED_LOCK:
        return _OWNED.get(os.path.abspath(path))


def heartbeat(path):
    """Touch our claim (its mtime is the heartbeat). Raises :class:`ClaimLost`
    when the file is gone or now belongs to another worker."""
    tok = _owned_token(path)
    d = _read_claim(path)
    if tok is None or d is None or (d and d.get("token") != tok):
        raise ClaimLost(f"claim {path} is no longer ours")
    os.utime(path, None)


def release(path):
    """Remove our claim (a claim taken over by another worker is left alone)."""
    tok = _owned_token(path)
    with _OWNED_LOCK:
        _OWNED.pop(os.path.abspath(path), None)
    if tok is None:
        return
    d = _read_claim(path)
    if d is None:
        return
    if d.get("token") != tok:
        log(f"[claim] WARNING: {path} now belongs to another worker; left in place")
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def list_claims(root, site=None, stale_after_s=STALE_AFTER_S):
    """States (:func:`claim_state`) of every claim under ``<root>/runs``."""
    base = os.path.join(root, "runs", site or "*", CLAIM_DIR)
    out = []
    for p in sorted(glob.glob(os.path.join(base, "**", "*.claim"), recursive=True)):
        st = claim_state(p, stale_after_s)
        if st is not None:
            out.append(st)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  per-unit log, stages and the monitor thread
# ─────────────────────────────────────────────────────────────────────────────

_CURRENT = contextvars.ContextVar("v13_unit_log", default=None)


class _Tee(io.TextIOBase):
    """Writes to the console and to the unit's buffer."""

    def __init__(self, stream, buf):
        super().__init__()
        self._stream, self._buf = stream, buf

    def write(self, s):
        self._buf.write(s)
        try:
            self._stream.write(s)
        except (ValueError, OSError):       # a closed console must not fail the unit
            pass
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class UnitLog:
    """Everything printed while a unit runs (still shown on the console) and
    the unit's stage timings."""

    def __init__(self, label):
        self.label = label
        self.buf = io.StringIO()
        self.timings = {}
        self.monitor = None

    @contextlib.contextmanager
    def capture(self):
        token = _CURRENT.set(self)
        out, err = _Tee(sys.stdout, self.buf), _Tee(sys.stderr, self.buf)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                yield self
        finally:
            _CURRENT.reset(token)

    def add(self, name, seconds):
        self.timings[name] = round(self.timings.get(name, 0.0) + float(seconds), 3)

    def text(self):
        return self.buf.getvalue()


def current_log():
    """The :class:`UnitLog` of the unit running in this context, or None."""
    return _CURRENT.get()


@contextlib.contextmanager
def stage(name):
    """Time one stage of the running unit: logged with the RSS, and saved
    in run.json's ``timings`` (summed when a stage repeats). Usable outside
    a unit (logged only)."""
    ul = _CURRENT.get()
    t0 = time.perf_counter()
    log(f"[stage] {name}: start")
    ok = False
    try:
        yield
        ok = True
    finally:
        dt = time.perf_counter() - t0
        if ul is not None:
            ul.add(name, dt)
        peak = ul.monitor.peak_mb if (ul is not None and ul.monitor is not None) else float("nan")
        log(f"[stage] {name}: {'done' if ok else 'FAILED'} in {dt:.1f} s; rss "
            f"{current_rss_mb():.0f} MB, unit peak {peak:.0f} MB")


class _Monitor:
    """Samples the RSS every second and touches the claim every heartbeat."""

    def __init__(self, claim_file=None, heartbeat_s=HEARTBEAT_S, sample_s=RSS_SAMPLE_S):
        self.claim_file = claim_file
        self.heartbeat_s = float(heartbeat_s)
        self.sample_s = float(min(sample_s, heartbeat_s))
        self.start_mb = current_rss_mb()
        self.peak_mb = self.start_mb
        self.beats = 0
        self.lost = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        r = current_rss_mb()
        if np.isfinite(r) and not (np.isfinite(self.peak_mb) and self.peak_mb >= r):
            self.peak_mb = r

    def _loop(self):
        next_beat = time.monotonic() + self.heartbeat_s
        while not self._stop.wait(self.sample_s):
            self.sample()
            if self.claim_file and time.monotonic() >= next_beat:
                next_beat = time.monotonic() + self.heartbeat_s
                try:
                    heartbeat(self.claim_file)
                    self.beats += 1
                except ClaimLost as e:
                    if self.lost is None:
                        self.lost = str(e)
                        log(f"[claim] WARNING: {e}; another worker may be running this unit")
                except OSError as e:
                    log(f"[claim] heartbeat failed: {e}")

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, name="v13-unit-monitor",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        self.sample()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  adapters and keys
# ─────────────────────────────────────────────────────────────────────────────

def registry():
    """``adapters.REGISTRY`` ({group: module}); ImportError before M2."""
    from experiments.benchmark.adapters import REGISTRY
    return REGISTRY


def default_variant(adapter):
    v = getattr(adapter, "DEFAULT_VARIANT", None)
    if v is None:
        vs = tuple(getattr(adapter, "VARIANTS", ()) or ())
        v = vs[0] if vs else ""
    return str(v or "")


def outputs(adapter, unit):
    """The ``(method, variant)`` products a unit writes, in order."""
    fn = getattr(adapter, "outputs", None)
    if callable(fn):
        outs = [(str(m), str(v or "")) for m, v in fn(unit)]
    else:
        outs = [(str(m), unit.variant) for m in adapter.METHODS]
    if not outs:
        raise ValueError(f"{unit.group}: a unit must write at least one product")
    if len(set(outs)) != len(outs):
        raise ValueError(f"{unit.group}: duplicate outputs {outs}")
    return outs


def smoke_settings(opts, group):
    """The non-controlled settings of a run (they change the results, so
    they enter the config key and mark the products smoke)."""
    s = {}
    if getattr(opts, "smoke", False):
        s["smoke"] = True
    bt = getattr(opts, "blas_threads", None)
    if bt is not None and int(bt) != group_threads(group):
        s["blas_threads"] = int(bt)
    if getattr(opts, "window", None):
        s["window"] = str(opts.window)
    return s


def effective_config(adapter, opts, group):
    """The adapter's CONFIG, plus the smoke settings when there are any."""
    cfg = dict(getattr(adapter, "CONFIG", {}) or {})
    smoke = smoke_settings(opts, group)
    if smoke:
        cfg["_smoke"] = smoke
    return cfg


def adapter_file(adapter):
    f = getattr(adapter, "__file__", None)
    if not f:
        raise TypeError(f"adapter {adapter!r} has no __file__ (its source keys the code key)")
    return f


def package_key_of(pkg):
    """The package key of a CommonPackage (``pkg.keys['package_key']``)."""
    keys = getattr(pkg, "keys", None)
    if isinstance(keys, Mapping) and keys.get("package_key"):
        return str(keys["package_key"])
    raise TypeError("the package carries no keys['package_key']")


def unit_keys(pkg, unit, adapter, method, variant, opts, code=None):
    """Every key of one product of a unit (§2.11)."""
    section = config.METHOD_SECTION[unit.group]
    ck = product.config_key(effective_config(adapter, opts, unit.group), section,
                            unit.mode, unit.sens, variant, group=unit.group)
    if code is None:
        code = product.code_key(adapter_file(adapter),
                                deps=tuple(getattr(adapter, "CODE_DEPS", ()) or ()))
    pk = package_key_of(pkg)
    frozen = {"threads": config.frozen_sha("threads")}
    if section is not None:
        frozen[section] = config.frozen_sha(section)
    return {"package_key": pk, "subset_key": unit.subset_key, "config_key": ck,
            "code_key": code,
            "inputs_key": product.inputs_key(pk, unit, method, variant, ck, code),
            "frozen": frozen}


# ─────────────────────────────────────────────────────────────────────────────
#  planning
# ─────────────────────────────────────────────────────────────────────────────

def _split(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        out = []
        for t in text:
            out += _split(t)
        return out
    return [s.strip() for s in str(text).split(",") if s.strip()]


def parse_N(text):
    """``"all"``, ``"grid"`` (N_GRID + all), ``"gran"`` (GRAN_N_GRID + all)
    or integers, comma-separated and mixed. Returns a list without
    duplicates, in the given order."""
    out = []
    for tok in _split(text):
        low = tok.lower()
        if low == config.N_ALL:
            vals = [config.N_ALL]
        elif low == "grid":
            vals = list(config.N_GRID) + [config.N_ALL]
        elif low == "gran":
            vals = list(config.GRAN_N_GRID) + [config.N_ALL]
        else:
            try:
                v = int(tok)
            except ValueError:
                raise ValueError(f"--N: {tok!r} is not all, grid, gran or an integer") from None
            if v < 1:
                raise ValueError("--N values must be >= 1")
            vals = [v]
        for v in vals:
            if v not in out:
                out.append(v)
    if not out:
        raise ValueError("--N is empty")
    return out


def parse_reps(text):
    """``"0-9"``, ``"0,2,5"`` or ``"3"`` -> sorted unique rep indices."""
    out = set()
    for tok in _split(text):
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise ValueError(f"--reps: empty range {tok!r}")
            out.update(range(a, b + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            raise ValueError(f"--reps: {tok!r} is not an index or a range a-b")
    if not out:
        raise ValueError("--reps is empty")
    return sorted(out)


def _sens_value(v):
    v = str(v).strip()
    if re.fullmatch(r"[+-]?\d+", v):
        return int(v)
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except ValueError:
        pass
    return v


def parse_sens(texts):
    """``--sens`` settings -> list of dicts (one per occurrence; {} = the
    controlled setting when none is given).

    ``"mask=blocklist,floor=80"`` -> ``{"mask": "blocklist", "floor": 80}``.
    Integer- and float-looking values become numbers, so the directory tag
    and the config key never depend on how a number was typed. The primary
    settings are refused (they ARE the controlled row): mask=marea_clear
    (D1), index=floor0 (D2) and MAREA's floor 40 (D4).
    """
    fo = config.FROZEN["obs"]
    primary = {"mask": fo["primary_mask"], "index": fo["primary_index"],
               "floor": int(config.FROZEN["marea"]["floor"])}
    allowed = {"mask": tuple(fo["mask_rules"]), "index": tuple(fo["index_modes"])}
    out = []
    for text in (texts or []):
        d = {}
        for tok in _split(text):
            if "=" not in tok:
                raise ValueError(f"--sens: {tok!r} is not key=value")
            k, v = (s.strip() for s in tok.split("=", 1))
            if not k or not re.fullmatch(r"[A-Za-z0-9_.-]+", k):
                raise ValueError(f"--sens: bad key {k!r}")
            v = _sens_value(v)
            if k in allowed and v not in allowed[k]:
                raise ValueError(f"--sens {k}={v}: expected one of {allowed[k]}")
            if k == "floor" and (not isinstance(v, int) or v < 0):
                raise ValueError("--sens floor= takes a non-negative integer")
            if k in primary and v == primary[k]:
                other = {"mask": "mask=blocklist", "index": "index=raw",
                         "floor": "floor=80 or floor=0"}[k]
                raise ValueError(f"--sens {k}={v} is the controlled setting; the "
                                 f"sensitivity is {other}")
            d[k] = v
        if d in out:
            continue
        out.append(d)
    return out or [{}]


def _site_list(sites, res_m):
    out = []
    for s in _split(sites):
        if s == "all":
            out += [x for x in config.RES_SITES[int(res_m)] if x != config.CONTROL_SITE]
        else:
            config.check_res(s, res_m)
            out.append(s)
    return list(dict.fromkeys(out))


def plan(sites, groups, families, modes, N_list, reps, variants, sens, res_m, *,
         load=None, registry_=None, scenario=None):
    """Expand the requested settings into units, in queue order.

    Parameters
    ----------
    sites, groups, families, modes : list of str
    N_list : list of int or "all"
    reps : list of int
    variants : list of str or None
        None: each group's default variant.
    sens : list of dict
        [{}] is the controlled setting.
    res_m : int
    load : callable, optional
        ``site -> package`` (needs ``subset(family, N, rep)`` and ``T``);
        default ``package.load``.
    registry_ : mapping, optional
        {group: adapter}; default :func:`registry`.

    Rules (one unit per distinct record subset and setting)
    -------------------------------------------------------
    * N = "all" is ``arange(T)`` in every family and rep: only rep 0 of the
      rand family is planned (win and other reps would repeat it);
    * the scene-rule families take N = "all" and rep 0 only;
    * an integer N > T is skipped;
    * a group that declares ``MODES`` runs only those modes (NIDEM: one A
      unit serves both), ``SITES`` only at those sites, ``SENS_KEYS`` only
      those sensitivities;
    * "all" is queued first in each (group, setting), so an F-mode unit
      that needs the N = all unit of its site comes after it.
    """
    if load is None:
        from experiments.benchmark import package
        load = (lambda site: package.load(site, res_m) if scenario is None
                else package.load(site, res_m, scenario=scenario))
    reg = registry() if registry_ is None else registry_
    sens = list(sens or [{}])
    N_list = list(N_list)
    Ns = ([config.N_ALL] if config.N_ALL in N_list else []) + \
        [n for n in N_list if n != config.N_ALL]
    units, seen, notes = [], set(), set()

    def note(msg):
        if msg not in notes:
            notes.add(msg)
            log(f"[plan] {msg}")

    for site in sites:
        pkg = None
        for group in groups:
            if group not in GROUPS:
                raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
            if group not in reg:
                raise KeyError(f"no adapter registered for {group} (available from "
                               f"milestone {MILESTONE[group]})")
            ad = reg[group]
            ad_sites = getattr(ad, "SITES", None)
            if ad_sites is not None and site not in ad_sites:
                note(f"{group} does not run at {site}")
                continue
            ad_modes = tuple(getattr(ad, "MODES", ("A", "F")) or ("A", "F"))
            sens_keys = getattr(ad, "SENS_KEYS", None)
            vs = list(variants) if variants else [default_variant(ad)]
            known = tuple(getattr(ad, "VARIANTS", ()) or ())
            for variant in vs:
                if known and variant not in known:
                    note(f"{group} has no variant {variant!r} (variants: {known})")
                    continue
                for s in sens:
                    if sens_keys is not None and not set(s) <= set(sens_keys):
                        note(f"{group} takes no sensitivity {sorted(set(s) - set(sens_keys))}")
                        continue
                    for family in families:
                        rule = family.startswith("rule-")
                        for mode in modes:
                            m = mode if mode in ad_modes else ad_modes[0]
                            if m != mode:
                                note(f"{group} is mode-independent: its {m} unit serves {mode}")
                            for N in Ns:
                                if N != config.N_ALL and rule:
                                    note(f"{family} takes N = all only; N = {N} skipped")
                                    continue
                                if N == config.N_ALL and family == "win":
                                    note("win N = all is the rand N = all unit; skipped")
                                    continue
                                if pkg is None:
                                    pkg = load(site)
                                if N != config.N_ALL and int(N) > int(pkg.T):
                                    note(f"{site}: N = {N} exceeds the record T = {pkg.T}; skipped")
                                    continue
                                for rep in reps:
                                    if rep > 0 and (N == config.N_ALL or rule):
                                        continue
                                    key = (site, group, variant, family, m, N, rep,
                                           product.senstag(s))
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    sel = pkg.subset(family, N, rep)
                                    units.append(Unit(site=site, res_m=res_m, group=group,
                                                      variant=variant, family=family,
                                                      mode=m, N=N, rep=rep, sens=dict(s),
                                                      sel=sel))
    log(f"[plan] {len(units)} unit(s)")
    return units


# ─────────────────────────────────────────────────────────────────────────────
#  gates and the positive control
# ─────────────────────────────────────────────────────────────────────────────

def gate_path(site, res_m, gate, root=None):
    """``<root>/gates/<site>/<gate>.json`` (root: out_root(res_m))."""
    root = config.out_root(res_m) if root is None else root
    return f"{str(root).rstrip('/')}/gates/{site}/{gate}.json"


def read_gate(site, res_m, gate, root=None):
    """The gate's JSON record, or None when the gate has not run."""
    p = gate_path(site, res_m, gate, root)
    if not os.path.exists(p):
        return None
    try:
        return product.read_json(p)
    except (OSError, ValueError):
        return {"verdict": "UNREADABLE"}


def gate_status(rec, package_key=None):
    """"pass", "fail", "missing" or "stale" (run on another package)."""
    if rec is None:
        return "missing"
    if str(rec.get("verdict", "")).upper() != "PASS":
        return "fail"
    pk = rec.get("package_key")
    if package_key is not None and pk is not None and pk != package_key:
        return "stale"
    return "pass"


def required_gates(site, res_m, group):
    """``[(gate, site, res_m)]`` a group needs before it runs at a site."""
    if group == "published":
        return []
    out = [(g, site, int(res_m)) for g in SITE_GATES + GROUP_GATES[group]]
    out += [(g, s, int(r)) for g, s, r in GLOBAL_GATES.get(group, ())]
    return out


def check_gates(site, res_m, group, package_key=None, root=None):
    """Have the group's gates passed at the site?

    Returns
    -------
    dict
        ``{"ok", "missing", "failed", "stale"}``; a gate run at another site
        (GLOBAL_GATES) is checked without a package key.
    """
    res = {"ok": True, "missing": [], "failed": [], "stale": []}
    for g, s, r in required_gates(site, res_m, group):
        here = (s == site and r == int(res_m))
        rec = read_gate(s, r, g, root if (here or root is not None) else None)
        st = gate_status(rec, package_key if here else None)
        name = g if here else f"{g}@{s}/{r}m"
        if st != "pass":
            res["ok"] = False
            res[{"missing": "missing", "fail": "failed", "stale": "stale"}[st]].append(name)
    return res


def control_verdict(scenario, control_root=None):
    """``control/<scenario>/verdict.json`` (``{"methods": {method: {"verdict":
    "PASS" | "FAIL" | "REPORTED", ...}}}``), or None."""
    base = config.control_dir(scenario) if control_root is None else \
        f"{str(control_root).rstrip('/')}/{scenario}"
    p = f"{base}/verdict.json"
    if not os.path.exists(p):
        return None
    try:
        return product.read_json(p)
    except (OSError, ValueError):
        return None


def control_required(method, scenario):
    """Must ``method`` PASS the positive control's ``scenario`` (FROZEN)?"""
    mp = config.FROZEN["positive_control"]["must_pass"]
    if scenario == "zero":
        return method in mp["zero_elevation"] or method in LAG_METHODS
    if scenario == "profile":
        return method in mp["profile_elevation"] or method in mp["profile_lag"]
    raise ValueError(f"unknown control scenario {scenario!r}")


def control_status(method, control_root=None):
    """``{scenario: "pass" | "fail" | "missing" | "not_required"}``."""
    out = {}
    for sc in config.CONTROL_SCENARIOS:
        if not control_required(method, sc):
            out[sc] = "not_required"
            continue
        v = control_verdict(sc, control_root)
        m = ((v or {}).get("methods") or {}).get(method)
        if m is None:
            out[sc] = "missing"
        else:
            verdict = m.get("verdict") if isinstance(m, Mapping) else m
            out[sc] = "pass" if str(verdict).upper() == "PASS" else "fail"
    return out


def validated(site, res_m, method, group=None, package_key=None, root=None,
              control_root=None):
    """True when the method's gates and its positive control passed (§0.6):
    only then is its controlled row printed."""
    group = group or METHOD_GROUP.get(method)
    if group is None:
        raise KeyError(f"unknown method {method!r}; pass group=")
    if not check_gates(site, res_m, group, package_key, root)["ok"]:
        return False
    return all(v in ("pass", "not_required")
               for v in control_status(method, control_root).values())


# ─────────────────────────────────────────────────────────────────────────────
#  running
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunOptions:
    """Settings of a queue (the ``run`` command's options)."""

    res_m: int
    root: Optional[str] = None            # default: config.site_root(site, res_m[, scenario])
    scenario: Optional[str] = None        # the positive control's scenario (site pc)
    worker_id: str = ""
    steal_stale: bool = False
    stale_after_s: float = STALE_AFTER_S
    heartbeat_s: float = HEARTBEAT_S
    force: bool = False                   # rerun done units (the old attempt is kept)
    smoke: bool = False
    blas_threads: Optional[int] = None    # tests and smoke only
    boot_threads: int = 1
    r_workers: int = 16
    rscript: Optional[str] = None
    tmpdir: Optional[str] = None
    resume: bool = False
    max_r_hours: Optional[float] = None
    project_only: bool = False
    window: Optional[str] = None
    no_wait: bool = False
    check_gates: bool = True
    registry: Optional[Mapping] = None    # tests: {group: adapter}
    load: Optional[Callable] = None       # tests: site -> package
    max_units: Optional[int] = None       # stop after this many units ran


@dataclass
class Outcome:
    """What happened to one unit in a queue."""

    unit: Unit
    outcome: str      # ran | done | claimed | stale_claim | refused | deferred | no_package | unavailable
    products: list = field(default_factory=list)
    message: str = ""


def unit_root(unit, opts):
    if opts.root is not None:
        return str(opts.root)
    sc = opts.scenario if unit.site == config.CONTROL_SITE else None
    return config.site_root(unit.site, unit.res_m, sc)


def adapter_opts(opts, work_dir):
    """The keyword arguments every adapter's ``run`` receives."""
    return {"work_dir": work_dir, "boot_threads": int(opts.boot_threads),
            "r_workers": int(opts.r_workers), "rscript": opts.rscript,
            "tmpdir": opts.tmpdir, "resume": bool(opts.resume),
            "max_r_hours": opts.max_r_hours, "project_only": bool(opts.project_only),
            "smoke": bool(opts.smoke), "window": opts.window,
            "no_wait": bool(opts.no_wait), "worker_id": opts.worker_id}


class QueueState:
    """Per-process caches of a queue: packages, adapters, code keys, gate
    checks, git state and runtime versions."""

    def __init__(self, opts):
        self.opts = opts
        self._pkgs, self._pkg_err = {}, {}
        self._gates, self._code = {}, {}
        self._git = None
        self._runtime = None

    def adapter(self, group):
        reg = self.opts.registry if self.opts.registry is not None else registry()
        if group not in reg:
            raise KeyError(group)
        return reg[group]

    def package(self, site):
        if site in self._pkg_err:
            raise self._pkg_err[site]
        if site not in self._pkgs:
            try:
                if self.opts.load is not None:
                    self._pkgs[site] = self.opts.load(site)
                else:
                    from experiments.benchmark import package
                    if site == config.CONTROL_SITE:
                        self._pkgs[site] = package.load(site, self.opts.res_m,
                                                        scenario=self.opts.scenario)
                    else:
                        self._pkgs[site] = package.load(site, self.opts.res_m)
            except Exception as e:
                self._pkg_err[site] = e
                raise
        return self._pkgs[site]

    def code_key(self, adapter):
        # the code this process loaded: hashed once, at first use
        k = id(adapter)
        if k not in self._code:
            self._code[k] = product.code_key(
                adapter_file(adapter), deps=tuple(getattr(adapter, "CODE_DEPS", ()) or ()))
        return self._code[k]

    def gates(self, unit, pkg):
        k = (unit.site, unit.res_m, unit.group)
        if k not in self._gates:
            root = self.opts.root
            self._gates[k] = check_gates(unit.site, unit.res_m, unit.group,
                                         package_key_of(pkg), root)
        return self._gates[k]

    def git(self):
        if self._git is None:
            self._git = product.git_state()
        return self._git

    def runtime(self):
        if self._runtime is None:
            self._runtime = product.runtime_provenance()
        return self._runtime


def _final_done(run, inputs_key, smoke):
    return (run is not None and run.get("status") in product.DONE_STATUSES
            and (run.get("keys") or {}).get("inputs_key") == inputs_key
            and bool(run.get("smoke", False)) == bool(smoke))


def _all_done(dirs, keys, smoke):
    return all(_final_done(product.read_run(dirs[o]), keys[o]["inputs_key"], smoke)
               for o in dirs)


def _status_products(unit, outs, status, stage_=None, reason=None, diag=None):
    return [Product(method=m, variant=v, unit=unit, status=status, stage=stage_,
                    reason=reason, diag=dict(diag or {})) for m, v in outs]


def _check_products(result, unit, outs):
    if not isinstance(result, (list, tuple)) or not all(isinstance(p, Product) for p in result):
        raise TypeError(f"{unit.group}.run must return a list of Product, got {type(result).__name__}")
    got = [(p.method, p.variant) for p in result]
    if sorted(got) != sorted(outs) or len(set(got)) != len(got):
        raise AssertionError(f"{unit.group}.run returned products {got}, expected {outs}")
    for p in result:
        if p.unit is not unit and p.unit != unit:
            raise AssertionError(f"product {p.method} carries another unit ({p.unit.label})")
    order = {o: i for i, o in enumerate(outs)}
    return sorted(result, key=lambda p: order[(p.method, p.variant)])


def _scalar(v):
    return v if isinstance(v, (bool, int, float, str)) or v is None else None


def _execute(unit, adapter, pkg, outs, keys, dirs, opts, state, cpath, root):
    """Run one claimed unit and publish its products. Returns the products,
    or None when the adapter deferred it."""
    work = "/".join([root.rstrip("/"), "runs", unit.site, WORK_DIR,
                     f"{socket.gethostname()}-{os.getpid()}",
                     f"{unit.group}__{unit.variant}" if unit.variant else unit.group,
                     f"{unit.family}-{unit.mode}", unit_leaf(unit)])
    if os.path.exists(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    n_blas = int(opts.blas_threads) if opts.blas_threads is not None else group_threads(unit.group)
    smoke = smoke_settings(opts, unit.group)
    ulog = UnitLog(unit.label)
    deferred = None
    try:
        with ulog.capture(), _Monitor(cpath, opts.heartbeat_s) as mon, \
                blas_limits(n_blas) as blas:
            ulog.monitor = mon
            reset_ok = reset_peak_rss()
            log(f"=== unit {unit.label}")
            log(f"    {unit.n} scenes, subset {unit.subset_key[:16]}, outputs "
                f"{', '.join(m + ('__' + v if v else '') for m, v in outs)}")
            k0 = keys[outs[0]]
            log(f"    package {k0['package_key'][:16]}, code {k0['code_key'][:16]}, "
                f"config {k0['config_key'][:16]}, inputs {k0['inputs_key'][:16]}")
            log(f"    host {socket.gethostname()}, pid {os.getpid()}, worker "
                f"{opts.worker_id or '-'}, start {product.utc_stamp()}")
            log(f"    BLAS threads {n_blas} via {blas['api']} (effective {blas['effective']}), "
                f"boot threads {opts.boot_threads}, R workers {opts.r_workers}"
                + (f"; SMOKE {smoke}" if smoke else ""))
            t0 = time.perf_counter()
            try:
                result = adapter.run(pkg, unit, root, **adapter_opts(opts, work))
                prods = _check_products(result, unit, outs)
            except NotComputable as e:
                log(f"[unit] NOT COMPUTABLE at {e.stage}: {e.reason or ''}")
                prods = _status_products(unit, outs, "not_computable", e.stage, e.reason, e.diag)
            except Deferred as e:
                deferred = str(e) or "deferred"
                log(f"[unit] deferred: {deferred}")
                prods = []
            except BudgetExceeded as e:
                log(f"[unit] BUDGET: {e}")
                prods = _status_products(unit, outs, "budget", e.stage, str(e),
                                         {"projected_r_hours": e.projected_r_hours})
            except Exception as e:                   # noqa: BLE001 -- every failure is recorded
                tb = traceback.format_exc()
                log(tb.rstrip())
                kind = "r_failed" if is_r_failure(e) else "error"
                tail = getattr(e, "log_tail", None) or (str(e)[-4000:] if kind == "r_failed" else None)
                diag = {"traceback": tb, "exception": type(e).__name__}
                if tail:
                    diag["r_log_tail"] = tail
                prods = _status_products(unit, outs, kind, getattr(e, "stage", None) or "run",
                                         f"{type(e).__name__}: {e}"[:2000], diag)
                log(f"[unit] {kind.upper()}: {type(e).__name__}: {str(e)[:300]}")
            total = time.perf_counter() - t0
            if deferred is not None:
                return None
            mon.sample()
            r_stats = None
            for p in prods:
                if isinstance(p.diag, Mapping) and isinstance(p.diag.get("r_runner"), Mapping):
                    r_stats = p.diag["r_runner"]
            peak = peak_rss_mb(r_stats)
            memory = {"peak_rss_mb": peak, "peak_scope": "unit" if reset_ok else "process",
                      "process_peak_mb": process_peak_rss_mb(),
                      "sampled_unit_peak_mb": mon.peak_mb, "rss_start_mb": mon.start_mb,
                      "rss_end_mb": current_rss_mb()}
            if r_stats is not None:
                memory["r_runner"] = {k: _scalar(v) for k, v in r_stats.items()}
            timings = dict(ulog.timings)
            claim_note = {"path": cpath, "heartbeats": mon.beats, "lost": mon.lost}
            for p in prods:
                own = {k: v for k, v in (p.keys or {}).items() if k not in keys[(p.method, p.variant)]}
                p.keys = {**keys[(p.method, p.variant)], **own}
                p.timings = {**timings, **(p.timings or {}), "total": round(total, 3)}
                p.peak_rss_mb = peak
                p.smoke = bool(p.smoke or smoke)
                p.provenance = {**(p.provenance or {}), "runtime": state.runtime(),
                                "git": state.git(), "threads": {**blas, "boot_threads": int(opts.boot_threads),
                                                                 "r_workers": int(opts.r_workers)},
                                "memory": memory, "claim": claim_note,
                                "worker": {"host": socket.gethostname(), "pid": os.getpid(),
                                           "worker_id": opts.worker_id},
                                "smoke_settings": smoke,
                                "options": {k: _scalar(v) for k, v in adapter_opts(opts, work).items()}}
            log(f"[unit] {', '.join(f'{p.method}={p.status}' for p in prods)} in {total:.1f} s; "
                f"peak RSS {peak:.0f} MB ({memory['peak_scope']}), sampled unit peak "
                f"{mon.peak_mb:.0f} MB")
            if timings:
                log("[unit] stages: " + ", ".join(f"{k} {v:.1f} s" for k, v in timings.items()))
        text = ulog.text()
        for p in prods:
            product.save(p, dirs[(p.method, p.variant)], log_text=text,
                         files=getattr(p, "files", None))
        return prods
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process_unit(unit, opts, state=None):
    """Resume check, claim, run, publish, release (one unit).

    Returns
    -------
    Outcome
    """
    state = QueueState(opts) if state is None else state
    root = unit_root(unit, opts)
    try:
        adapter = state.adapter(unit.group)
    except (ImportError, KeyError):
        return Outcome(unit, "unavailable",
                       message=f"{unit.group}: available from milestone {MILESTONE.get(unit.group, 'M2')}")
    try:
        pkg = state.package(unit.site)
    except Exception as e:                           # noqa: BLE001 -- reported per site
        return Outcome(unit, "no_package", message=f"{type(e).__name__}: {e}")
    outs = outputs(adapter, unit)
    code = state.code_key(adapter)
    keys = {o: unit_keys(pkg, unit, adapter, o[0], o[1], opts, code=code) for o in outs}
    dirs = {o: product.unit_dir(root, unit, o[0], o[1]) for o in outs}
    smoke = bool(smoke_settings(opts, unit.group))
    if not opts.force and _all_done(dirs, keys, smoke):
        return Outcome(unit, "done")
    if opts.check_gates and unit.site not in GATE_EXEMPT_SITES:
        g = state.gates(unit, pkg)
        if not g["ok"]:
            why = (f"gates not passed at {unit.site} {unit.res_m} m: missing {g['missing']}, "
                   f"failed {g['failed']}, other package {g['stale']}")
            if not opts.smoke:
                return Outcome(unit, "refused", message=why)
            log(f"[run] WARNING (smoke): {why}")
    n_blas = int(opts.blas_threads) if opts.blas_threads is not None else group_threads(unit.group)
    ok, why = can_enforce_threads(n_blas)
    if not ok:
        return Outcome(unit, "refused", message=why)
    cpath = claim_path(root, unit)
    if not claim(cpath, {"unit": unit.label, "worker_id": opts.worker_id},
                 steal_stale=opts.steal_stale, stale_after_s=opts.stale_after_s):
        st = claim_state(cpath, opts.stale_after_s)
        if st is not None and st["stale"]:
            return Outcome(unit, "stale_claim",
                           message=f"stale claim of {st.get('host')}:{st.get('pid')} "
                                   f"({st['age_s'] / 60:.0f} min); use --steal-stale")
        return Outcome(unit, "claimed",
                       message="" if st is None else f"{st.get('host')}:{st.get('pid')}")
    try:
        # another worker may have finished it between the check and the claim
        if not opts.force and _all_done(dirs, keys, smoke):
            return Outcome(unit, "done")
        prods = _execute(unit, adapter, pkg, outs, keys, dirs, opts, state, cpath, root)
        if prods is None:
            return Outcome(unit, "deferred")
        return Outcome(unit, "ran", prods)
    finally:
        release(cpath)


def run_unit(unit, opts):
    """Run one unit (resume, claim, publish); returns its products ([] when
    it was skipped: done, claimed elsewhere, refused or deferred)."""
    return process_unit(unit, opts).products


def _stop_requested(opts, roots):
    names = ["STOP"] + ([f"STOP-{opts.worker_id}"] if opts.worker_id else [])
    for r in roots:
        for n in names:
            if os.path.exists(os.path.join(r, n)):
                return os.path.join(r, n)
    return None


def run_queue(units, opts):
    """Run a list of units in order, sharing the queue with other workers
    through the claims.

    Returns
    -------
    dict
        Counts of outcomes (``ran``, ``done``, ``claimed``, ...) and of the
        product statuses written (``status:ok``, ...).
    """
    state = QueueState(opts)
    counts = Counter()
    told = set()
    t0 = time.time()
    n_ran = 0
    log(f"[run] {len(units)} unit(s) in the queue; worker {opts.worker_id or '-'} "
        f"on {socket.gethostname()} (pid {os.getpid()})")
    for unit in units:
        stop = _stop_requested(opts, [unit_root(unit, opts)])
        if stop:
            log(f"[run] {stop} exists: stopping before {unit.label}")
            counts["stopped"] += 1
            break
        if opts.max_units is not None and n_ran >= int(opts.max_units):
            log(f"[run] --max-units {opts.max_units} reached")
            break
        out = process_unit(unit, opts, state)
        counts[out.outcome] += 1
        for p in out.products:
            counts[f"status:{p.status}"] += 1
        if out.outcome == "ran":
            n_ran += 1
        elif out.outcome in ("refused", "no_package", "unavailable", "stale_claim"):
            key = (out.outcome, unit.site, unit.group, out.message)
            if key not in told:
                told.add(key)
                log(f"[run] {unit.site} {unit.group}: {out.outcome}: {out.message}")
        elif out.outcome == "deferred":
            log(f"[run] deferred {unit.label}")
    log(f"[run] finished in {time.time() - t0:.0f} s: " +
        ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
#  status
# ─────────────────────────────────────────────────────────────────────────────

STATUS_COLUMNS = ("site", "res_m", "group", "method", "variant", "family", "mode", "N",
                  "rep", "sens", "status", "stage", "reason", "fallback", "n_scenes",
                  "n_keep_sel", "clear_obs_median", "seconds", "projected_r_hours",
                  "peak_rss_mb", "package_key", "subset_key", "code_key", "smoke", "dir")


def status_rows(root, site=None):
    """One row per published product under ``<root>/runs[/<site>]`` (the
    columns of §7.2's status.csv, plus smoke and dir)."""
    rows = []
    for d, run in product.iter_runs(root, site):
        u = run.get("unit") or {}
        keys = run.get("keys") or {}
        diag = run.get("diag") or {}
        dom = run.get("domain") or {}
        tim = run.get("timings") or {}
        rows.append({
            "site": u.get("site"), "res_m": u.get("res_m"), "group": u.get("group"),
            "method": run.get("method"), "variant": run.get("variant", ""),
            "family": u.get("family"), "mode": u.get("mode"), "N": u.get("N"),
            "rep": u.get("rep"), "sens": product.senstag(u.get("sens") or {}),
            "status": run.get("status"), "stage": run.get("stage"),
            "reason": run.get("reason"), "fallback": run.get("fallback"),
            "n_scenes": u.get("n_sel"), "n_keep_sel": _scalar(dom.get("n_keep_sel")),
            "clear_obs_median": _scalar(diag.get("clear_obs_median")),
            "seconds": _scalar(tim.get("total")),
            "projected_r_hours": _scalar(diag.get("projected_r_hours")),
            "peak_rss_mb": _scalar(run.get("peak_rss_mb")),
            "package_key": keys.get("package_key"),
            "subset_key": keys.get("subset_key") or u.get("subset_key"),
            "code_key": keys.get("code_key"), "smoke": bool(run.get("smoke", False)),
            "dir": d})
    return rows


def write_status_csv(rows, path):
    """Write status rows to ``path`` (atomic)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(STATUS_COLUMNS), extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    product.write_text(path, buf.getvalue())
    return path


def unit_state(unit, opts, state=None):
    """"done", "failed:<status>", "partial", "claimed", "stale_claim" or
    "pending" for a planned unit (no claim is taken)."""
    state = QueueState(opts) if state is None else state
    adapter = state.adapter(unit.group)
    pkg = state.package(unit.site)
    root = unit_root(unit, opts)
    outs = outputs(adapter, unit)
    code = state.code_key(adapter)
    keys = {o: unit_keys(pkg, unit, adapter, o[0], o[1], opts, code=code) for o in outs}
    dirs = {o: product.unit_dir(root, unit, o[0], o[1]) for o in outs}
    smoke = bool(smoke_settings(opts, unit.group))
    if _all_done(dirs, keys, smoke):
        return "done"
    st = claim_state(claim_path(root, unit), opts.stale_after_s)
    if st is not None:
        return "stale_claim" if st["stale"] else "claimed"
    runs = [product.read_run(dirs[o]) for o in outs]
    failed = [r.get("status") for r in runs if r is not None and
              r.get("status") not in product.DONE_STATUSES]
    if failed:
        return f"failed:{failed[0]}"
    if any(r is not None for r in runs):
        return "partial"                    # done under other keys, or not all products
    return "pending"
