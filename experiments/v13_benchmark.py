# -*- coding: utf-8 -*-
"""v13 — controlled benchmark of intertidal lag and elevation methods: the CLI.

Question
--------
Do the tide-lag and elevation methods (no delay, MAREA, NIDEM, Granadeiro et
al. 2021, GM20, HD26 and the lag baselines) rank the same when every one of
them receives the same scenes, the same per-pixel cloud mask, the same band
preprocessing, the same boundary tide and the same date subsets, and only
the estimator changes? The design is ``docs/v13_benchmark_spec.md``
(revision 2 with the authors' final decisions of 2026-10-08); this file is
its only entry point and the only module that changes directory.

usage (from the repository root; on the server after ``source env.sh`` and
``export TMPDIR=$PWD/tmp``):

  python -X utf8 -m experiments.v13_benchmark check-env --res 10
  python -X utf8 -m experiments.v13_benchmark prepare escalda|all --res 10 [--steps a,b] [--force]
  python -X utf8 -m experiments.v13_benchmark truth escalda|all --res 10 [--part elev,lag]
  python -X utf8 -m experiments.v13_benchmark status [escalda|all] --res 10 [--verify] [--csv]

Validation v2 (docs/validation_v2_spec.md) has its own entry point,
``python -X utf8 -m experiments.validation_v2``; this file is kept as the
v13 continuity alias (spec §12.1) and lends it its thread and check-env
helpers (``set_thread_env``, ``Checks``, ``_check_r``).

Milestone M1 delivers check-env, prepare, truth and status. ``run``,
``gate``, ``control``, ``lock``, ``score`` and ``report`` are in the parser
with the arguments of §7.4 and exit with "available from milestone M2"
(Granadeiro: M3) until their modules exist.

Pre-registered rules this file enforces (§3.0, §7.4)
----------------------------------------------------
* OMP/OPENBLAS/MKL_NUM_THREADS are set before numpy is imported: for
  ``run`` the largest FROZEN BLAS count of the requested groups (each unit
  then enforces its own group's count, runner.blas_limits), 4 for the other
  commands; ``--blas-threads`` overrides it (tests and smoke runs only).
* Every command works from the repository root (``main`` chdirs there;
  importing this module does nothing).
* ``prepare`` and ``truth`` write FROZEN.json next to their outputs first; a
  plan that changed since keeps the previous file as ``FROZEN.<utc>.json``.
* ``check-env`` never stops at the first problem: it lists every missing
  input and every encoding violation (FAIL), every condition that limits a
  group or the resources (WARN), and exits 1 when anything FAILed. Absent R
  is a warning (the Granadeiro group is then skipped), as is an absent
  GOT5.6 model (its baseline is skipped).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: BLAS threads of the commands that run no adapter group
DEFAULT_THREADS = 4
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
GROUPS = ("marea_pair", "nidem", "granadeiro", "gm20", "hd26", "baselines", "published")
#: free disk the project and TMPDIR need beyond the packages (R exchange
#: files, v7 memmaps, products)
SPACE_MARGIN_GB = 20.0
#: P_dom / keep_prod measured on Escalda 20 m (§2.5), for the size estimate
PDOM_OVER_KEEP = 1.4
#: the files of a common package (§2.10)
PACKAGE_FILES = ("record.json", "levels.npz", "boundary_1min_cache.npz", "marks.npz",
                 "dom.npy", "b03.npy", "b08.npy", "scl.npy", "frame_valid.npy",
                 "counts.npz", "pixels.npz", "scene_stats.csv", "judges.json",
                 "subsets.npz", "manifest.json")
TRUTH_FILES = ("truth_elev.npz", "truth_elev.json", "support.npy", "truth_lag.json",
               "truth_lag_curves.npz")


def log(msg):
    """Print ``msg`` and flush, so that progress shows at once in a log."""
    print(msg, flush=True)


def _split(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        out = []
        for t in text:
            out += _split(t)
        return out
    return [s.strip() for s in str(text).split(",") if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  threads (before numpy)
# ─────────────────────────────────────────────────────────────────────────────

def thread_count(args):
    """``(n, why)``: the BLAS threads the process starts with."""
    if getattr(args, "blas_threads", None):
        return int(args.blas_threads), "--blas-threads"
    groups = [g for g in _split(getattr(args, "groups", None)) if g in GROUPS]
    if args.command in ("run", "control") and groups:
        from experiments.benchmark import config      # stdlib only: numpy not loaded
        n = max(int(config.BLAS_THREADS[g]) for g in groups)
        return n, f"largest FROZEN count of {','.join(groups)}"
    return DEFAULT_THREADS, "default"


def set_thread_env(n, why=""):
    """Set the BLAS/OMP thread variables (effective only before numpy loads)."""
    old = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(int(n))
    changed = {k: v for k, v in old.items() if v not in (None, str(int(n)))}
    msg = f"[threads] {', '.join(THREAD_ENV_VARS[:3])} = {int(n)} ({why})"
    if changed:
        msg += f"; was {changed}"
    if "numpy" in sys.modules:
        msg += "; numpy is already loaded: the units enforce their count with threadpoolctl"
    log(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _module(name):
    """``experiments.benchmark.<name>``, or None (with a message) when it is
    not in this build."""
    import importlib
    try:
        return importlib.import_module(f"experiments.benchmark.{name}")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(f"benchmark.{name}"):
            log(f"[v13] experiments/benchmark/{name}.py is not in this build")
            return None
        raise


def _call(fn, *args, **kwargs):
    """Call ``fn`` with the keyword arguments that are not None; a keyword
    the function does not take is an interface mismatch and raises."""
    import inspect
    kw = {k: v for k, v in kwargs.items() if v is not None}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        bad = sorted(k for k in kw if k not in params)
        if bad:
            raise TypeError(f"{fn.__module__}.{fn.__qualname__} takes no {bad}")
    return fn(*args, **kw)


def _sites(text, res_m, scenarios=None):
    """``[(site, scenario)]`` of a site argument (site, comma list or all)."""
    from experiments.benchmark import config
    out = []
    for s in _split(text) or ["all"]:
        if s == "all":
            names = [x for x in config.RES_SITES[int(res_m)] if x != config.CONTROL_SITE]
        else:
            config.check_res(s, res_m)
            names = [s]
        for name in names:
            if name == config.CONTROL_SITE:
                scs = _split(scenarios) or list(config.CONTROL_SCENARIOS)
                for sc in scs:
                    if sc not in config.CONTROL_SCENARIOS:
                        raise ValueError(f"unknown scenario {sc!r}; expected one of "
                                         f"{config.CONTROL_SCENARIOS}")
                    out.append((name, sc))
            else:
                out.append((name, None))
    return list(dict.fromkeys(out))


def _label(site, res_m, scenario=None):
    return f"{site} {res_m} m" + (f" [{scenario}]" if scenario else "")


def _gb(nbytes):
    return f"{nbytes / 1e9:.2f} GB"


def _inside(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == \
            os.path.realpath(root)
    except ValueError:                       # another drive on Windows
        return False


def write_frozen(res_m):
    """Write ``FROZEN.json`` of a resolution (§2.11); a different existing
    file is kept as ``FROZEN.<utc>.json``. Returns the path."""
    from experiments.benchmark import config, product
    path = config.frozen_path(res_m)
    doc = config.frozen_document()
    if os.path.exists(path):
        try:
            old = product.read_json(path)
        except (OSError, ValueError):
            old = None
        if old is not None and config.sha256_json(old) == config.sha256_json(doc):
            return path
        keep = path[:-len(".json")] + f".{product.utc_stamp()}.json"
        os.replace(path, keep)
        changed = sorted(s for s, v in doc["sha256"].items()
                         if (old or {}).get("sha256", {}).get(s) != v)
        log(f"[v13] WARNING: the FROZEN plan changed in {changed}; the previous "
            f"FROZEN.json is kept as {keep}")
    product.write_json(path, doc)
    log(f"[v13] {path} written")
    return path


def _log_command(args, rc, seconds):
    """Append the command to ``<out_root>/logs/commands.jsonl`` (provenance)."""
    try:
        from experiments.benchmark import config, product
        res = getattr(args, "res", None)
        if res not in config.RES_CHOICES:
            return
        d = config.logs_dir(res)
        os.makedirs(d, exist_ok=True)
        rec = {"utc": product.utc_stamp(), "host": socket.gethostname(), "pid": os.getpid(),
               "command": args.command, "argv": sys.argv[1:], "exit": rc,
               "seconds": round(seconds, 1), "git": product.git_state()}
        with open(os.path.join(d, "commands.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                   # provenance must never fail a command
        log(f"[v13] could not log the command: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  check-env
# ─────────────────────────────────────────────────────────────────────────────

class Checks:
    """Findings of check-env: OK, INFO, WARN (limits a group or a resource)
    and FAIL (a missing or invalid input), printed as they are found."""

    def __init__(self):
        self.items = []

    def add(self, level, what, detail=""):
        """Record a finding at ``level`` and print it."""
        self.items.append({"level": level, "what": what, "detail": str(detail)})
        log(f"[check-env] {level:<4} {what}" + (f": {detail}" if detail != "" else ""))

    def ok(self, what, detail=""):
        """Record an OK finding."""
        self.add("OK", what, detail)

    def info(self, what, detail=""):
        """Record an INFO finding."""
        self.add("INFO", what, detail)

    def warn(self, what, detail=""):
        """Record a WARN finding (it limits a group or a resource)."""
        self.add("WARN", what, detail)

    def fail(self, what, detail=""):
        """Record a FAIL finding (a missing or invalid input)."""
        self.add("FAIL", what, detail)

    def count(self, level):
        """Number of findings at ``level``."""
        return sum(1 for i in self.items if i["level"] == level)


def inspect_cube(path, period=None, n_sample=5):
    """Encoding and sampled-scene checks of a production-layout cube.

    The bands must be int16 with ``_FillValue = -32768`` and no scale or
    offset; the time axis must decode to unique increasing dates; a crs
    variable must carry the WKT. ``n_sample`` scenes of ``period``, evenly
    spread, are read raw one band at a time: no non-fill band value below
    -1000 (``obs.check_boa``), SCL codes in 0..11 (``obs.scl_u8``), and the
    SCL 1 count (the only class on which the blocklist and ITEM's mask
    differ).

    Returns
    -------
    dict
        ``problems`` (each one a FAIL), ``notes``, ``shape`` (T, H, W),
        ``dates``, ``n_in_period``, ``chunks``, ``samples`` (per scene: date,
        band minima, no-data shares, SCL counts), ``scl_counts`` (codes
        0..11 and no data, summed over the samples).
    """
    import numpy as np
    import xarray as xr
    from experiments.benchmark import config, obs

    period = tuple(period or config.PERIOD)
    out = {"path": path, "problems": [], "notes": [], "samples": [], "chunks": {},
           "shape": None, "dates": [], "n_in_period": 0,
           "scl_counts": [0] * 13}
    probs = out["problems"]
    ds = xr.open_dataset(path, mask_and_scale=False)
    try:
        fills = {}
        for band in ("B03", "B08", "SCL"):
            if band not in ds:
                probs.append(f"{band} is missing")
                continue
            da = ds[band]
            if da.dtype != np.int16:
                probs.append(f"{band} is {da.dtype}, expected int16")
            fill = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
            if fill is None:
                probs.append(f"{band} has no _FillValue")
            elif int(fill) != obs.FILL:
                probs.append(f"{band} _FillValue is {fill}, expected {obs.FILL}")
            fills[band] = obs.FILL if fill is None else int(fill)
            for a in ("scale_factor", "add_offset"):
                if a in da.attrs or a in da.encoding:
                    probs.append(f"{band} carries {a}")
            ch = da.encoding.get("chunksizes")
            out["chunks"][band] = None if ch is None else tuple(int(c) for c in ch)
            if ch is not None and int(ch[0]) != 1:
                out["notes"].append(f"{band} is chunked {tuple(ch)}: per-scene reads "
                                    f"decompress {ch[0]} scenes")
        if probs:
            return out
        da = ds["B03"]
        t_dim = da.dims[0]
        if tuple(da.dims[1:]) != ("y", "x"):
            probs.append(f"B03 dims are {da.dims}, expected (time, y, x)")
        for band in ("B08", "SCL"):
            if ds[band].dims != da.dims:
                probs.append(f"{band} dims {ds[band].dims} differ from B03's {da.dims}")
        tv = ds[t_dim].values
        if not np.issubdtype(tv.dtype, np.datetime64):
            probs.append(f"the time axis {t_dim!r} does not decode to dates ({tv.dtype})")
            return out
        dates = [str(np.datetime64(v, "D")) for v in tv]
        out["dates"] = dates
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            probs.append("cube dates are not unique and increasing")
        out["shape"] = tuple(int(s) for s in da.shape)
        if "crs" not in ds or not (ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref")):
            probs.append("no crs variable with crs_wkt/spatial_ref")
        inp = [i for i, d in enumerate(dates) if period[0] <= d <= period[1]]
        out["n_in_period"] = len(inp)
        if not inp:
            probs.append(f"no scene in {period}")
            return out
        k = max(1, int(n_sample))
        pick = sorted({inp[int(j)] for j in np.round(np.linspace(0, len(inp) - 1, k))})
        tot = np.zeros(13, np.int64)
        for i in pick:
            rec = {"index": int(i), "date": dates[i]}
            for band in ("B03", "B08"):
                raw = ds[band].isel({t_dim: int(i)}).values
                v = raw[raw != fills[band]]
                rec[f"{band}_min"] = int(v.min()) if v.size else None
                rec[f"{band}_nodata"] = round(1.0 - v.size / max(raw.size, 1), 6)
                try:
                    obs.check_boa(raw, fills[band], band)
                except AssertionError as e:
                    probs.append(f"{dates[i]}: {e} (minimum {rec[f'{band}_min']})")
                del raw, v
            s = ds["SCL"].isel({t_dim: int(i)}).values
            try:
                c = obs.scl_counts(obs.scl_u8(s, fills["SCL"]))
                rec["scl_counts"] = c.tolist()
                tot += c
            except ValueError as e:
                probs.append(f"{dates[i]} SCL: {e}")
            del s
            out["samples"].append(rec)
        out["scl_counts"] = tot.tolist()
    finally:
        ds.close()
    return out


def _check_versions(ck):
    import importlib
    from experiments.benchmark import product
    rp = product.runtime_provenance()
    ck.info("python", f"{rp['python']} on {rp['platform']} ({rp['host']})")
    mods = ("numpy", "scipy", "pandas", "xarray", "netCDF4", "h5netcdf", "pyTMD", "pyproj",
            "rasterio", "yaml", "skimage", "psutil", "threadpoolctl", "joblib", "eo_tides")
    vers, missing = {}, []
    for m in mods:
        try:
            vers[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:                    # absent or broken
            vers[m] = None
            missing.append(m)
    ck.info("versions", ", ".join(f"{m} {v}" for m, v in vers.items() if v is not None))
    for m in ("numpy", "scipy", "pandas", "xarray", "pyTMD", "pyproj", "rasterio"):
        if vers[m] is None:
            ck.fail("python package", f"{m} is not importable")
    if vers["netCDF4"] is None and vers["h5netcdf"] is None:
        ck.fail("python package", "neither netCDF4 nor h5netcdf can read the cubes")
    if vers["psutil"] is None:
        ck.warn("psutil", "not installed: a dead worker's claim is recognised by age only, "
                          "and Windows peak memory falls back to GetProcessMemoryInfo")
    if vers["threadpoolctl"] is None:
        ck.warn("threadpoolctl", "not installed: run each BLAS-thread group (granadeiro = 1, "
                                 "the others = 4) in its own process")
    ck.info("threads", ", ".join(f"{k}={os.environ.get(k)}" for k in THREAD_ENV_VARS))
    if vers["threadpoolctl"] is not None:
        try:
            import threadpoolctl
            pools = [f"{p.get('internal_api')} {p.get('version')} x{p.get('num_threads')}"
                     for p in threadpoolctl.threadpool_info() if p.get("user_api") == "blas"]
            ck.info("BLAS", "; ".join(pools) or "no BLAS loaded yet")
        except Exception as e:               # informational only
            ck.info("BLAS", f"threadpool_info failed: {e}")
    return vers


def _check_git(ck):
    from experiments.benchmark import product
    gs = product.git_state()
    ck.info("git", f"HEAD {gs['head']}, {'dirty' if gs['dirty'] else 'clean'} working tree")
    try:
        st = subprocess.run(["git", "status", "--porcelain", "--", "experiments/benchmark",
                             "experiments/v13_benchmark.py", "pyintertidal", "tests"],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
        dirty = [ln[3:] for ln in st.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        dirty = []
    if dirty:
        ck.warn("git", f"{len(dirty)} uncommitted path(s) in the benchmark, pyintertidal or "
                       f"tests ({', '.join(dirty[:4])}{', ...' if len(dirty) > 4 else ''}): "
                       f"`lock` refuses a dirty tree")
    return gs


def _check_frozen(ck, res_m):
    from experiments.benchmark import config, product
    path = config.frozen_path(res_m)
    if not os.path.exists(path):
        ck.info("FROZEN", f"{path} not written yet (prepare writes it)")
    else:
        try:
            old = product.read_json(path).get("sha256", {})
        except (OSError, ValueError, AttributeError):
            old = {}
        cur = config.frozen_shas()
        changed = [s for s in cur if old.get(s) != cur[s]]
        if changed:
            pk = [s for s in changed if s in config.PACKAGE_SECTIONS]
            ck.warn("FROZEN", f"{path} differs from the code in {changed}"
                    + (f"; packages keyed on {pk} will be rebuilt" if pk else ""))
        else:
            ck.ok("FROZEN", f"{path} matches the code")
    lock = config.lock_path(res_m)
    if os.path.exists(lock):
        ck.info("LOCK", f"{lock} exists")


def _stations():
    from experiments.benchmark import config, product
    if not os.path.exists(config.STATIONS_JSON):
        return None
    return {s["code"]: s for s in product.read_json(config.STATIONS_JSON)}


def check_site(ck, cfg, n_sample=5, stations=None):
    """Every input of one site at one grid (cube, extraction, overpass,
    boundary and IOC chunks, judges, elevation truth, transition record).

    Returns the numbers of the package-size estimate."""
    import numpy as np
    from experiments.benchmark import config, levels, product

    tag = _label(cfg.site, cfg.res_m, cfg.scenario)
    info = {"site": cfg.site, "T_cand": None, "HW": None, "keep": None,
            "built": os.path.exists(f"{cfg.common_dir}/manifest.json")}
    if cfg.site == config.CONTROL_SITE:
        for name, p in (("cube", cfg.cube), ("overpass", cfg.overpass),
                        ("planted", cfg.extra.get("planted"))):
            if p and os.path.exists(p):
                ck.ok(f"{tag} {name}", p)
            else:
                ck.info(f"{tag} {name}", f"{p} not built yet (control --synth-only, M2)")
        return info
    cube_dates = None
    if not os.path.exists(cfg.cube):
        ck.fail(f"{tag} cube", f"{cfg.cube} is missing")
    else:
        t0 = time.time()
        r = inspect_cube(cfg.cube, n_sample=n_sample)
        for p in r["problems"]:
            ck.fail(f"{tag} cube", p)
        for n in r["notes"]:
            ck.warn(f"{tag} cube", n)
        if r["shape"] is not None:
            T, H, W = r["shape"]
            cube_dates = r["dates"]
            info["HW"] = H * W
            if not r["problems"]:
                ck.ok(f"{tag} cube", f"{cfg.cube} ({_gb(os.path.getsize(cfg.cube))}): {T} "
                      f"scenes, {r['n_in_period']} in {config.PERIOD[0]}..{config.PERIOD[1]}, "
                      f"{H} x {W} px; int16, fill -32768, no scale; chunks "
                      f"{r['chunks'].get('B03')}")
        if r["samples"]:
            c = r["scl_counts"]
            n_px = sum(c)
            mins = [(s.get("B03_min"), s.get("B08_min")) for s in r["samples"]]
            lo = min((m for pair in mins for m in pair if m is not None), default=None)
            ck.info(f"{tag} sampled scenes",
                    f"{', '.join(s['date'] for s in r['samples'])}: smallest band value {lo} "
                    f"(>= -1000 required); SCL 1 in {c[1]:,} of {n_px:,} px "
                    f"({c[1] / max(n_px, 1):.2e}); SCL 2 {c[2]:,}, SCL 11 {c[11]:,}, no data "
                    f"{c[12]:,} ({time.time() - t0:.0f} s)")
    # the production extraction: keep_prod (gates) and the cube dates
    if not cfg.extract or not os.path.exists(cfg.extract):
        ck.fail(f"{tag} extraction", f"{cfg.extract} is missing")
    else:
        try:
            with np.load(cfg.extract, allow_pickle=False) as z:
                members = set(z.files)
                need = {"Y", "C", "keep", "dates", "shape", "sea", "inter"}
                if need - members:
                    ck.fail(f"{tag} extraction", f"members {sorted(need - members)} missing")
                ext_dates = [str(d) for d in z["dates"]]
                keep_n = int(z["keep"].shape[0])
                shape = tuple(int(v) for v in z["shape"])
            info["keep"] = keep_n
            bad = []
            if cube_dates is not None and ext_dates != cube_dates:
                bad.append(f"its {len(ext_dates)} dates differ from the cube's {len(cube_dates)}")
            if info["HW"] is not None and shape[0] * shape[1] != info["HW"]:
                bad.append(f"its shape {shape} differs from the cube's")
            for b in bad:
                ck.fail(f"{tag} extraction", b)
            if not bad:
                ck.ok(f"{tag} extraction", f"{cfg.extract}: {len(ext_dates)} dates, "
                      f"keep_prod {keep_n:,} px, shape {shape}")
        except Exception as e:               # unreadable: reported, not raised
            ck.fail(f"{tag} extraction", f"{cfg.extract}: {type(e).__name__}: {e}")
    # overpass instants (C1: a cube date without one is not in the record)
    if not os.path.exists(cfg.overpass):
        ck.fail(f"{tag} overpass", f"{cfg.overpass} is missing")
    else:
        try:
            over = product.read_json(cfg.overpass)
            if cube_dates is not None:
                inp = [d for d in cube_dates if config.PERIOD[0] <= d <= config.PERIOD[1]]
                cand = [d for d in inp if d in over]
                info["T_cand"] = len(cand)
                ck.ok(f"{tag} overpass", f"{cfg.overpass}: {len(over)} instants; record "
                      f"candidates {len(cand)} of {len(inp)} scenes in the period"
                      + (f" ({len(inp) - len(cand)} without an overpass time)"
                         if len(cand) < len(inp) else ""))
            else:
                ck.ok(f"{tag} overpass", f"{cfg.overpass}: {len(over)} instants")
        except Exception as e:
            ck.fail(f"{tag} overpass", f"{type(e).__name__}: {e}")
    # boundary recipe and the IOC chunks it reads
    try:
        levels.check_boundary_spec(cfg)
        b = dict(cfg.boundary)
        desc = (f"make_boundary {b.get('tide_model')} + {b.get('n_gauges')} gauge(s), "
                f"exclude {list(b.get('exclude', ()))}" if b["kind"] == "make_boundary"
                else f"PyTMDBoundary {b.get('model')} at the AOI centroid")
        chunks = levels.ioc_chunks(cfg)
        codes = sorted({c["code"] for c in chunks} - {"ioc_stations"})
        ck.ok(f"{tag} boundary", desc + (f"; {len(chunks)} IOC file(s) for {codes}, "
                                         f"{sum(c['size'] for c in chunks) / 1e6:.1f} MB"
                                         if chunks else ""))
    except Exception as e:
        ck.fail(f"{tag} boundary", f"{type(e).__name__}: {e}")
    tide = os.path.join(config.TIDE_DIR, str(cfg.boundary.get("tide_model")
                                             or cfg.boundary.get("model")))
    if not os.path.isdir(tide):
        ck.fail(f"{tag} tide model", f"{tide} is missing")
    # judges (estuary_truth)
    if cfg.judges:
        if stations is None:
            ck.fail(f"{tag} judges", f"{config.STATIONS_JSON} is missing")
        else:
            miss, nofile = [], []
            for code in cfg.judges:
                s = stations.get(code)
                if s is None:
                    miss.append(code)
                elif not os.path.exists(os.path.join(config.ESTUARY_TRUTH, s.get("file", ""))):
                    nofile.append(f"{code} ({s.get('file')})")
            if miss:
                ck.fail(f"{tag} judges", f"not in stations.json: {miss}")
            if nofile:
                ck.fail(f"{tag} judges", f"series missing: {nofile}")
            if not miss and not nofile:
                ck.ok(f"{tag} judges", f"{len(cfg.judges)} station(s) with series in "
                      f"{config.ESTUARY_TRUTH}")
    else:
        ck.info(f"{tag} judges", "none listed (no lag judge at this site)")
    # elevation truth
    for kind in cfg.elevation_truth:
        try:
            if kind == "vaklodingen":
                from experiments import v0_vaklodingen_fetch as v0
                from experiments import v1_vaklodingen_validate as v1
                sheets = [n for n, _, _ in v0.sheets_for(cfg.site)]
                miss = [n for n in sheets if not os.path.exists(os.path.join(v1.TRUTH_DIR, n))]
                if not sheets or miss:
                    ck.fail(f"{tag} Vaklodingen", f"{len(miss)} of {len(sheets)} sheet(s) "
                            f"missing in {v1.TRUTH_DIR}: {miss[:6]}")
                else:
                    ck.ok(f"{tag} Vaklodingen", f"{len(sheets)} sheet(s) in {v1.TRUTH_DIR}")
            elif kind == "lidar":
                from experiments import v10_validation_report as v10
                from pyintertidal import rtk
                for p in (v10.MDT02, rtk.DEFAULT_GRID):
                    (ck.ok if os.path.exists(p) else ck.fail)(f"{tag} LiDAR", p + (
                        "" if os.path.exists(p) else " is missing"))
            elif kind == "rtk":
                from pyintertidal import rtk
                for p in (rtk.DEFAULT_CSV, rtk.DEFAULT_ALICANTE_CSV, rtk.DEFAULT_GRID):
                    (ck.ok if os.path.exists(p) else ck.fail)(f"{tag} RTK", p + (
                        "" if os.path.exists(p) else " is missing"))
        except Exception as e:
            ck.fail(f"{tag} {kind}", f"{type(e).__name__}: {e}")
    if not cfg.elevation_truth:
        ck.info(f"{tag} elevation truth", "none (lag only)")
    # the transition-zone record (rule-tz subsets)
    if cfg.transition_result:
        p = config.resolve(cfg.transition_result)
        try:
            rd = product.read_json(p).get("record_dates") if os.path.exists(p) else None
        except (OSError, ValueError):
            rd = None
        if rd:
            ck.ok(f"{tag} transition record", f"{p}: {len(rd)} record_dates")
        else:
            ck.fail(f"{tag} transition record", f"{cfg.transition_result} (or its mirror) "
                    f"is missing or has no record_dates")
    # as-published products (reference rows and gate references): informational
    pub = [f"{k} {'present' if os.path.exists(config.resolve(v)) else 'absent'}"
           for k, v in cfg.published.items()]
    if pub:
        ck.info(f"{tag} as published", ", ".join(pub))
    if info["built"]:
        try:
            m = product.read_json(f"{cfg.common_dir}/manifest.json")
            ck.info(f"{tag} package", f"built: package_key {product.package_key(m)[:16]}")
        except Exception as e:
            ck.warn(f"{tag} package", f"manifest unreadable: {e}")
    return info


def _package_bytes(info):
    """Estimated package size (§2.10): T P_dom 5 B + T ceil(HW / 8) B."""
    if info["built"] or not info["T_cand"] or not info["HW"]:
        return 0
    keep = info["keep"] or 0
    T = info["T_cand"]
    return int(T * PDOM_OVER_KEEP * keep * 5 + T * -(-info["HW"] // 8))


def _check_space(ck, infos):
    from experiments.benchmark import config
    tmp_env = os.environ.get("TMPDIR")
    tmp = tmp_env or tempfile.gettempdir()
    if not tmp_env:
        ck.warn("TMPDIR", f"not set: R exchange files and memmaps go to {tmp}; run "
                          f"`mkdir -p tmp && export TMPDIR=$PWD/tmp` first")
    elif not _inside(tmp, ROOT):
        ck.warn("TMPDIR", f"{tmp} is outside the project {ROOT}")
    else:
        ck.ok("TMPDIR", f"{tmp} (inside the project)")
    try:
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp, prefix="v13_probe_"):
            pass
    except OSError as e:
        ck.fail("TMPDIR", f"{tmp} is not writable: {e}")
    est = sum(_package_bytes(i) for i in infos)
    unknown = [i["site"] for i in infos if not i["built"] and i["site"] != config.CONTROL_SITE
               and not _package_bytes(i)]
    need = SPACE_MARGIN_GB * 1e9 + est
    ck.info("package size", f"estimated {_gb(est)} for the packages not built yet"
            + (f" (unknown for {unknown})" if unknown else ""))
    for name, path in (("project disk", ROOT), ("TMPDIR disk", tmp)):
        try:
            free = shutil.disk_usage(path).free
        except OSError as e:
            ck.warn(name, f"free space unknown: {e}")
            continue
        (ck.ok if free >= need else ck.warn)(
            name, f"{_gb(free)} free at {path}; needs >= {_gb(need)} "
                  f"({SPACE_MARGIN_GB:.0f} GB + packages)")


def _check_r(ck, rscript=None):
    from pyintertidal import granadeiro2021 as g21
    try:
        rs = g21.find_rscript(rscript)
    except FileNotFoundError:
        ck.warn("Rscript", "not found: the Granadeiro group (M3) will be skipped; pass "
                           "--rscript or set PYINTERTIDAL_RSCRIPT")
        return None
    ck.ok("Rscript", rs)
    tmp = os.environ.get("TMPDIR") or None
    try:
        with g21.RRunner(workers=1, tmpdir=tmp, rscript=rs) as runner:
            v = g21.r_versions(runner)
        miss = [p for p in ("nplr", "lmodel2", "mgcv") if v.get(p) in (None, "MISSING")]
        desc = ", ".join(f"{k} {v[k]}" for k in ("r_version", "nplr", "lmodel2", "mgcv") if k in v)
        if miss:
            ck.warn("R packages", f"{miss} missing ({desc}): the Granadeiro group will be "
                                  f"skipped")
        else:
            ck.ok("R packages", desc)
    except Exception as e:
        ck.warn("R packages", f"granadeiro_fit.R (versions mode) failed: {type(e).__name__}: "
                              f"{str(e)[-300:]}")
        v = None
    # from a script file: Rscript.exe on Windows breaks quoted -e expressions
    script = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".R", prefix="v13_blas_", dir=tmp,
                                         delete=False, encoding="utf-8") as f:
            f.write('si <- sessionInfo()\n'
                    'pick <- function(a, b) if (nzchar(a)) a else if (length(b) && nzchar(b)) b '
                    'else "(R internal)"\n'
                    'cat("BLAS", pick(extSoftVersion()[["BLAS"]], si$BLAS), "\\n")\n'
                    'cat("LAPACK", pick(La_library(), si$LAPACK), "\\n")\n')
            script = f.name
        out = subprocess.run([rs, script], capture_output=True, text=True, timeout=180)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        ck.info("R BLAS/LAPACK", " | ".join(lines) or out.stderr.strip()[-200:])
    except (OSError, subprocess.SubprocessError) as e:
        ck.info("R BLAS/LAPACK", f"not determined: {e}")
    finally:
        if script and os.path.exists(script):
            os.remove(script)
    return v


def _check_models(ck, cfg):
    """EOT20 (every boundary) and GOT5.6 (a lag baseline) answer one instant."""
    import numpy as np
    import pandas as pd
    from experiments.benchmark import config
    from pyintertidal import net, sites
    from pyintertidal.boundary import PyTMDBoundary

    net.use_system_certificates()
    lat, lon = sites.get(cfg.boundary["aoi"]).centroid
    for model, required in (("EOT20", True), ("GOT5.6", False)):
        t0 = time.time()
        try:
            h = PyTMDBoundary(model, lat, lon, directory=config.TIDE_DIR).levels(
                pd.DatetimeIndex(["2024-06-01 12:00:00"]))
            if not np.all(np.isfinite(h)):
                raise ValueError("non-finite level")
            ck.ok(f"tide model {model}", f"{float(h[0]):+.3f} m at {cfg.boundary['aoi']} "
                                         f"({time.time() - t0:.0f} s)")
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            if required:
                ck.fail(f"tide model {model}", msg)
            else:
                ck.warn(f"tide model {model}", f"{msg}: the {model} baseline will be skipped")


def check_env(res_m, sites=None, rscript=None, n_sample=5, models=True):
    """Run every check of §7.4 for one resolution; returns the Checks."""
    from experiments.benchmark import config
    ck = Checks()
    log(f"[check-env] resolution {res_m} m, root {ROOT}")
    _check_versions(ck)
    _check_git(ck)
    _check_frozen(ck, res_m)
    stations = None
    try:
        stations = _stations()
    except Exception as e:
        ck.fail("stations.json", f"{type(e).__name__}: {e}")
    if stations is None:
        ck.fail("estuary_truth", f"{config.STATIONS_JSON} is missing")
    infos, cfgs = [], []
    for site, sc in sites:
        cfg = config.site_config(site, res_m, sc)
        cfgs.append(cfg)
        try:
            infos.append(check_site(ck, cfg, n_sample=n_sample, stations=stations))
        except Exception as e:               # a bug in a check must not hide the others
            ck.fail(f"{_label(site, res_m, sc)}", f"check crashed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
    _check_space(ck, infos)
    _check_r(ck, rscript)
    if models and cfgs:
        _check_models(ck, cfgs[0])
    elif not models:
        ck.info("tide models", "not loaded (--no-models)")
    return ck


def cmd_check_env(args):
    """``check-env``: check the environment and every input of the sites, and
    write the findings to ``check_env_<host>.json`` in the logs folder.

    Returns 1 when anything FAILed, else 0.
    """
    from experiments.benchmark import config, product
    sites = _sites(args.sites, args.res, args.scenario)
    ck = check_env(args.res, sites, rscript=args.rscript, n_sample=args.sample_scenes,
                   models=not args.no_models)
    n_fail, n_warn = ck.count("FAIL"), ck.count("WARN")
    path = f"{config.logs_dir(args.res)}/check_env_{socket.gethostname()}.json"
    try:
        product.write_json(path, {"utc": product.utc_stamp(), "res_m": args.res,
                                  "sites": [s for s, _ in sites], "items": ck.items,
                                  "fail": n_fail, "warn": n_warn})
    except OSError as e:
        log(f"[check-env] could not write {path}: {e}")
        path = None
    log(f"[check-env] {n_fail} FAIL, {n_warn} WARN, {ck.count('OK')} OK"
        + (f"; written to {path}" if path else ""))
    return 1 if n_fail else 0


# ─────────────────────────────────────────────────────────────────────────────
#  prepare, truth, status
# ─────────────────────────────────────────────────────────────────────────────

def cmd_prepare(args):
    """``prepare``: write FROZEN.json, then build the input package of each
    site (every step, or those of ``--steps``).

    Returns 1 when a site failed, 2 for an unknown step or a missing module,
    else 0.
    """
    from experiments.benchmark import config, product
    package = _module("package")
    if package is None:
        return 2
    targets = _sites(args.site, args.res, args.scenario)
    steps = _split(args.steps) or None
    known = tuple(getattr(package, "STEPS", ()))
    if steps and known:
        bad = [s for s in steps if s not in known]
        if bad:
            log(f"[prepare] unknown step(s) {bad}; steps: {known}")
            return 2
    write_frozen(args.res)
    failed = []
    for site, sc in targets:
        label = _label(site, args.res, sc)
        t0 = time.time()
        log(f"[prepare] {label}: steps {', '.join(steps) if steps else 'all'}"
            + (" (force)" if args.force else ""))
        try:
            key = _call(package.prepare, site, args.res, steps=steps, force=bool(args.force),
                        scenario=sc, chunk=args.level_chunk)
            log(f"[prepare] {label}: package_key {key} ({time.time() - t0:.0f} s)")
            man = f"{config.common_dir(site, args.res, sc)}/manifest.json"
            if os.path.exists(man):
                m = product.read_json(man)
                size = sum(int(e.get("size", 0)) for e in m.get("files", {}).values())
                log(f"[prepare] {label}: {len(m.get('files', {}))} package files, {_gb(size)}")
        except Exception as e:
            failed.append(label)
            log(traceback.format_exc().rstrip())
            log(f"[prepare] {label}: FAILED ({type(e).__name__}: {e})")
    if failed:
        log(f"[prepare] failed: {failed}")
    return 1 if failed else 0


def _print_judges(label, js):
    rows = js if isinstance(js, list) else (js or {}).get("stations", [])
    inc = []
    for j in rows:
        if not isinstance(j, dict):
            continue
        code = j.get("code") or j.get("station") or "?"
        ok = j.get("included")
        why = j.get("exclusion_rule") or j.get("excluded_by") or j.get("reason") or ""
        if ok:
            inc.append(code)
        log(f"[truth] {label} judge {code}: {'included' if ok else 'excluded'}"
            + (f" ({why})" if why and not ok else ""))
    log(f"[truth] {label}: {len(inc)} of {len(rows)} judge(s) included")


def cmd_truth(args):
    """``truth``: write FROZEN.json, then build the judges and the lag truth
    and/or the elevation truth (``--part``) of each site's package.

    Returns 1 when a site failed, 2 for an unknown part or a missing module,
    else 0.
    """
    package, judges, truth = _module("package"), _module("judges"), _module("truth")
    if package is None or judges is None or truth is None:
        return 2
    parts = _split(args.part) or ["elev", "lag"]
    bad = [p for p in parts if p not in ("elev", "lag")]
    if bad:
        log(f"[truth] unknown part(s) {bad}; parts: elev, lag")
        return 2
    write_frozen(args.res)
    failed = []
    for site, sc in _sites(args.site, args.res, args.scenario):
        label = _label(site, args.res, sc)
        t0 = time.time()
        try:
            pkg = _call(package.load, site, args.res, scenario=sc)
            if "lag" in parts:
                _print_judges(label, judges.build(pkg))
            if "elev" in parts:
                log(f"[truth] {label}: elevation truth -> {truth.build_elevation_truth(pkg)}")
            if "lag" in parts:
                log(f"[truth] {label}: lag truth -> {truth.build_lag_truth(pkg)}")
            log(f"[truth] {label}: done in {time.time() - t0:.0f} s")
        except Exception as e:
            failed.append(label)
            log(traceback.format_exc().rstrip())
            log(f"[truth] {label}: FAILED ({type(e).__name__}: {e})")
    return 1 if failed else 0


def _status_package(cfg, verify):
    from experiments.benchmark import config, product
    d = cfg.common_dir
    present = [f for f in PACKAGE_FILES if os.path.exists(os.path.join(d, f))]
    size = sum(os.path.getsize(os.path.join(d, f)) for f in present)
    line = f"{len(present)}/{len(PACKAGE_FILES)} files, {_gb(size)}"
    out = {"files": len(present), "package_key": None}
    man = os.path.join(d, "manifest.json")
    if os.path.exists(man):
        try:
            m = product.read_json(man)
            out["package_key"] = key = product.package_key(m)
            line += f"; package_key {key[:16]}"
            stale = [s for s in config.PACKAGE_SECTIONS
                     if (m.get("frozen") or {}).get(s) != config.frozen_sha(s)]
            if stale:
                line += f"; FROZEN changed in {stale}: rebuild"
            from experiments.benchmark import obs
            if m.get("obs_sha") != obs.source_sha256():
                line += "; obs.py changed: rebuild"
        except Exception as e:
            line += f"; manifest unreadable ({type(e).__name__}: {e})"
        if verify:
            package = _module("package")
            if package is not None:
                try:
                    _call(package.verify_manifest, d, full=False)
                    line += "; verified (size + mtime)"
                except Exception as e:
                    line += f"; VERIFY FAILED ({type(e).__name__}: {e})"
    elif present:
        line += "; no manifest yet (prepare incomplete)"
    missing = [f for f in PACKAGE_FILES if f not in present]
    if present and missing:
        line += f"; missing {missing}"
    log(f"  package : {d}: {line}")
    return out


def _status_truth(cfg):
    d = cfg.truth_dir
    present = [f for f in TRUTH_FILES if os.path.exists(os.path.join(d, f))]
    log(f"  truth   : {d}: " + (", ".join(present) if present else "not built"))


def _status_judges(cfg):
    from experiments.benchmark import product
    p = os.path.join(cfg.common_dir, "judges.json")
    if not cfg.judges:
        return
    if not os.path.exists(p):
        log("  judges  : not built")
        return
    try:
        js = product.read_json(p)
        rows = js if isinstance(js, list) else js.get("stations", [])
        inc = [j.get("code") for j in rows if isinstance(j, dict) and j.get("included")]
        log(f"  judges  : {len(rows)} station(s), {len(inc)} included {inc}")
    except Exception as e:
        log(f"  judges  : unreadable ({type(e).__name__}: {e})")


def _status_gates(cfg, package_key):
    import glob
    from experiments.benchmark import config, runner
    if cfg.site == config.CONTROL_SITE:
        return
    files = sorted(glob.glob(os.path.join(config.gates_dir(cfg.site, cfg.res_m), "*.json")))
    if not files:
        log("  gates   : none run")
        return
    parts = []
    for f in files:
        g = os.path.basename(f)[:-5]
        st = runner.gate_status(runner.read_gate(cfg.site, cfg.res_m, g), package_key)
        parts.append(f"{g} {st.upper()}")
    log("  gates   : " + ", ".join(parts))


def _status_runs(cfg):
    from collections import Counter
    from experiments.benchmark import runner
    rows = runner.status_rows(cfg.root, cfg.site)
    if not rows:
        log("  runs    : none")
    else:
        by = Counter((r["method"] + (f"__{r['variant']}" if r["variant"] else ""),
                      r["status"]) for r in rows)
        methods = sorted({m for m, _ in by})
        for m in methods:
            parts = [f"{st} {by[(m, st)]}" for st in
                     ("ok", "empty", "not_computable", "r_failed", "budget", "error")
                     if by.get((m, st))]
            fb = sum(1 for r in rows if r["fallback"] and
                     r["method"] + (f"__{r['variant']}" if r["variant"] else "") == m)
            sm = sum(1 for r in rows if r["smoke"] and
                     r["method"] + (f"__{r['variant']}" if r["variant"] else "") == m)
            log(f"  runs    : {m}: " + ", ".join(parts)
                + (f"; fallback {fb}" if fb else "") + (f"; smoke {sm}" if sm else ""))
    claims = runner.list_claims(cfg.root, cfg.site)
    stale = [c for c in claims if c["stale"]]
    if claims:
        log(f"  claims  : {len(claims) - len(stale)} active, {len(stale)} stale"
            + "".join(f"\n            stale: {c['path']} ({c.get('host')}:{c.get('pid')}, "
                      f"{c['age_s'] / 60:.0f} min)" for c in stale))
    return rows


def cmd_status(args):
    """``status``: print the state of each site's package, judges, truths,
    gates and runs.

    With ``--groups`` it also counts the planned units; with ``--csv`` it
    writes ``scores/status.csv``. Returns 0.
    """
    from experiments.benchmark import config, runner
    targets = _sites(args.site or "all", args.res, args.scenario)
    if (args.site or "all") == "all" and int(args.res) in (20,) and args.scenario is None:
        # the positive control lives under results/v13_benchmark_20m/control/
        targets += [(config.CONTROL_SITE, sc) for sc in config.CONTROL_SCENARIOS
                    if os.path.isdir(config.control_dir(sc))]
    all_rows = []
    for site, sc in targets:
        cfg = config.site_config(site, args.res, sc)
        log(f"=== {_label(site, args.res, sc)} ({cfg.root})")
        pk = _status_package(cfg, args.verify)
        _status_judges(cfg)
        _status_truth(cfg)
        _status_gates(cfg, pk["package_key"])
        all_rows += _status_runs(cfg) or []
    if args.groups:
        _status_pending(args)
    if args.csv:
        path = f"{config.out_root(args.res)}/scores/status.csv"
        runner.write_status_csv(all_rows, path)
        log(f"[status] {len(all_rows)} row(s) written to {path}")
    return 0


def _status_pending(args):
    from experiments.benchmark import runner
    try:
        runner.registry()
    except ImportError:
        log("[status] pending units: available from milestone M2 (no adapters yet)")
        return
    from collections import Counter
    units, opts = _plan_from_args(args)
    state = runner.QueueState(opts)
    c = Counter(runner.unit_state(u, opts, state) for u in units)
    log("[status] planned units: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


# ─────────────────────────────────────────────────────────────────────────────
#  run (M2) and the commands of later milestones
# ─────────────────────────────────────────────────────────────────────────────

def _plan_from_args(args):
    from experiments.benchmark import product, runner
    groups = _split(args.groups)
    families = _split(args.family) or ["rand"]
    bad = [f for f in families if f not in product.FAMILIES]
    if bad:
        raise ValueError(f"unknown family {bad}; families: {product.FAMILIES}")
    modes = _split(args.mode) or ["A"]
    bad = [m for m in modes if m not in product.MODES]
    if bad:
        raise ValueError(f"unknown mode {bad}; modes: {product.MODES}")
    targets = _sites(args.site or "all", args.res, getattr(args, "scenario", None))
    scen = {sc for _, sc in targets} - {None}
    if len(scen) > 1:
        raise ValueError("run one positive-control scenario at a time (--scenario)")
    sc = next(iter(scen), None)
    opts = runner.RunOptions(
        res_m=args.res, scenario=sc, worker_id=str(getattr(args, "worker_id", "") or ""),
        steal_stale=bool(getattr(args, "steal_stale", False)),
        force=bool(getattr(args, "force", False)), smoke=bool(getattr(args, "smoke", False)),
        blas_threads=getattr(args, "blas_threads", None),
        boot_threads=int(getattr(args, "boot_threads", 1) or 1),
        r_workers=int(getattr(args, "r_workers", 16) or 16),
        rscript=getattr(args, "rscript", None), tmpdir=getattr(args, "tmpdir", None),
        resume=bool(getattr(args, "resume", False)),
        max_r_hours=getattr(args, "max_r_hours", None),
        project_only=bool(getattr(args, "project_only", False)),
        window=getattr(args, "window", None), no_wait=bool(getattr(args, "no_wait", False)),
        max_units=getattr(args, "max_units", None))
    units = runner.plan([s for s, _ in targets], groups, families, modes,
                        runner.parse_N(args.N), runner.parse_reps(args.reps),
                        _split(args.variant) or None, runner.parse_sens(args.sens),
                        args.res, scenario=sc)
    return units, opts


def cmd_run(args):
    """``run``: plan the units of the requested groups and run them through the
    queue (``--dry-run`` only lists their state).

    Returns 2 for an unknown group or one whose milestone is not built, 1 when
    a unit failed, else 0.
    """
    from experiments.benchmark import runner
    groups = _split(args.groups)
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        log(f"[run] unknown group(s) {bad}; groups: {GROUPS}")
        return 2
    try:
        reg = runner.registry()
    except ImportError:
        reg = {}
    missing = [g for g in groups if g not in reg]
    if missing:
        ms = sorted({runner.MILESTONE[g] for g in missing})
        log(f"v13_benchmark run: available from milestone {' / '.join(ms)} "
            f"({', '.join(f'{g}: {runner.MILESTONE[g]}' for g in missing)}); this build "
            f"is M1: check-env, prepare, truth, status")
        return 2
    units, opts = _plan_from_args(args)
    if args.dry_run:
        state = runner.QueueState(opts)
        for u in units:
            log(f"[run] {runner.unit_state(u, opts, state):<14} {u.label}")
        return 0
    counts = runner.run_queue(units, opts)
    failed = sum(v for k, v in counts.items() if k in ("status:error", "status:r_failed"))
    return 1 if failed else 0


def cmd_later(args):
    """The commands of later milestones: a clear message, exit 2."""
    ms = "M2"
    if args.command == "gate":
        gates = set(_split(args.gate))
        if gates and gates <= {"granadeiro"}:
            ms = "M3"
        elif "granadeiro" in gates:
            ms = "M2 (granadeiro: M3)"
    elif args.command == "control":
        groups = set(_split(args.groups))
        if groups and groups <= {"granadeiro"}:
            ms = "M3"
        elif "granadeiro" in groups:
            ms = "M2 (granadeiro: M3)"
    log(f"v13_benchmark {args.command}: available from milestone {ms}; this build is M1: "
        f"check-env, prepare, truth, status")
    return 2


# ─────────────────────────────────────────────────────────────────────────────
#  parser and main
# ─────────────────────────────────────────────────────────────────────────────

def _res(sp):
    sp.add_argument("--res", type=int, required=True, choices=(10, 20),
                    help="grid: 10 (reported numbers, server) or 20 (local tests)")


def _threads(sp):
    sp.add_argument("--blas-threads", type=int, default=None,
                    help="BLAS/OMP threads (tests and smoke runs only; marks run products smoke)")


def _plan_args(sp, groups_required):
    sp.add_argument("--groups", required=groups_required, default=None,
                    help=f"comma list of {', '.join(GROUPS)}")
    sp.add_argument("--variant", default=None, help="comma list (default: each group's own)")
    sp.add_argument("--family", default="rand",
                    help="rand | win | rule-tz | rule-g21 | rule-ard (comma list allowed)")
    sp.add_argument("--mode", default="A", help="A | F (comma list allowed)")
    sp.add_argument("--N", default="all", help="all | grid | gran | 10,20,... (mixed allowed)")
    sp.add_argument("--reps", default="0", help="0-9 | 0,2,5")
    sp.add_argument("--sens", action="append", default=None,
                    help="key=value[,key=value] (repeatable), e.g. mask=blocklist, index=raw, "
                         "floor=80")
    sp.add_argument("--scenario", default=None, help="positive-control scenario (site pc)")


def build_parser():
    """The argument parser: ``check-env``, ``prepare``, ``truth``,
    ``status`` and ``run``, plus ``gate``, ``control``, ``lock``, ``score``
    and ``report``, which answer that they belong to a later milestone
    (:func:`cmd_later`)."""
    p = argparse.ArgumentParser(
        prog="python -X utf8 -m experiments.v13_benchmark",
        description="v13 controlled benchmark of intertidal lag and elevation methods "
                    "(docs/v13_benchmark_spec.md)")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    sp = sub.add_parser("check-env", help="versions, R, TMPDIR and disk, every input, cube "
                                          "encodings and sampled scenes, git")
    _res(sp)
    sp.add_argument("--sites", default="all", help="site, comma list or all")
    sp.add_argument("--scenario", default=None, help="positive-control scenario(s) (site pc)")
    sp.add_argument("--rscript", default=None)
    sp.add_argument("--sample-scenes", type=int, default=5)
    sp.add_argument("--no-models", action="store_true",
                    help="do not load EOT20 / GOT5.6 (each load takes 5-20 s)")
    _threads(sp)
    sp.set_defaults(func=cmd_check_env)

    sp = sub.add_parser("prepare", help="build the common input package (§2), resumable")
    sp.add_argument("site", help="site, comma list or all")
    _res(sp)
    sp.add_argument("--steps", default=None, help="comma list of package.STEPS (default all)")
    sp.add_argument("--force", action="store_true", help="rebuild the given steps")
    sp.add_argument("--scenario", default=None, help="positive-control scenario(s) (site pc)")
    sp.add_argument("--level-chunk", type=int, default=None,
                    help="provider instants per batched call (default FROZEN 150,000; it "
                         "enters levels.npz and the package key)")
    _threads(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("truth", help="elevation and lag truths (§5.1, §5.5)")
    sp.add_argument("site", help="site, comma list or all")
    _res(sp)
    sp.add_argument("--part", default="elev,lag", help="elev, lag or both")
    sp.add_argument("--scenario", default=None, help="positive-control scenario(s) (site pc)")
    _threads(sp)
    sp.set_defaults(func=cmd_truth)

    sp = sub.add_parser("status", help="package, truth, gates, runs and claims")
    sp.add_argument("site", nargs="?", default="all", help="site, comma list or all")
    _res(sp)
    sp.add_argument("--verify", action="store_true", help="check package files (size + mtime)")
    sp.add_argument("--csv", action="store_true", help="write scores/status.csv")
    _plan_args(sp, groups_required=False)
    _threads(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("run", help="queue of units (milestone M2; Granadeiro M3)")
    sp.add_argument("site", help="site, comma list or all")
    _res(sp)
    _plan_args(sp, groups_required=True)
    sp.add_argument("--worker-id", default="")
    sp.add_argument("--r-workers", type=int, default=16)
    sp.add_argument("--rscript", default=None)
    sp.add_argument("--tmpdir", default=None)
    sp.add_argument("--resume", action="store_true", help="keep finished R stages (StageCache)")
    sp.add_argument("--max-r-hours", type=float, default=None)
    sp.add_argument("--project-only", action="store_true")
    sp.add_argument("--smoke", action="store_true")
    sp.add_argument("--window", default=None, help="r0:r1,c0:c1 (smoke)")
    sp.add_argument("--no-wait", action="store_true")
    sp.add_argument("--boot-threads", type=int, default=1)
    sp.add_argument("--steal-stale", action="store_true")
    sp.add_argument("--force", action="store_true", help="rerun done units (old kept)")
    sp.add_argument("--max-units", type=int, default=None)
    sp.add_argument("--dry-run", action="store_true", help="list the units and their state")
    _threads(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("gate", help="reproduction gates (§4.1; milestone M2, Granadeiro M3)")
    sp.add_argument("site", help="site, comma list or all")
    _res(sp)
    sp.add_argument("--gate", required=True,
                    help="record,levels,keep,marea,marea_legacy,nidem,granadeiro,score")
    sp.add_argument("--rscript", default=None)
    sp.add_argument("--r-workers", type=int, default=16)
    sp.add_argument("--tmpdir", default=None)
    _threads(sp)
    sp.set_defaults(func=cmd_later)

    sp = sub.add_parser("control", help="positive control (§4.2; milestone M2)")
    sp.add_argument("--scenario", default="zero,profile")
    sp.add_argument("--groups", default=None)
    sp.add_argument("--r-workers", type=int, default=16)
    sp.add_argument("--rscript", default=None)
    sp.add_argument("--tmpdir", default=None)
    sp.add_argument("--smoke", action="store_true")
    sp.add_argument("--synth-only", action="store_true")
    sp.add_argument("--from-existing", action="store_true")
    _threads(sp)
    sp.set_defaults(func=cmd_later)

    sp = sub.add_parser("lock", help="pre-registration lock (§2.11; milestone M2)")
    _res(sp)
    sp.set_defaults(func=cmd_later)

    sp = sub.add_parser("score", help="scoring (§5; milestone M2)")
    sp.add_argument("site", help="site, comma list or all")
    _res(sp)
    sp.add_argument("--no-boot", action="store_true")
    sp.add_argument("--allow-smoke", action="store_true")
    _threads(sp)
    sp.set_defaults(func=cmd_later)

    sp = sub.add_parser("report", help="tables and figures (§7.3; milestone M2)")
    _res(sp)
    sp.add_argument("--paper", action="store_true")
    sp.add_argument("--granadeiro-status", default=None)
    sp.set_defaults(func=cmd_later)
    return p


def main(argv=None):
    """Parse, set the BLAS threads (before numpy), chdir to the repository
    root and run the command. Returns the exit code."""
    args = build_parser().parse_args(argv)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    n, why = thread_count(args)
    set_thread_env(n, why)
    t0 = time.time()
    log(f"[v13] {args.command} on {socket.gethostname()} (pid {os.getpid()}), "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        rc = args.func(args)
    except KeyboardInterrupt:
        log(f"[v13] {args.command} interrupted")
        rc = 130
    except ValueError as e:                  # bad arguments found after parsing
        log(f"[v13] {args.command}: {e}")
        rc = 2
    if args.command in ("check-env", "prepare", "truth", "run") and rc != 2:
        _log_command(args, rc, time.time() - t0)     # 2: not available or bad arguments
    log(f"[v13] {args.command} finished in {time.time() - t0:.0f} s (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
