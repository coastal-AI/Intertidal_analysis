# -*- coding: utf-8 -*-
"""DEA Intertidal 2.5.3 (Bishop-Taylor et al.) as a fixed-configuration competitor.

``run(obs, cfg) -> dict(z, sigma, meta)`` and ``register(METHODS)`` for
``experiments/fixedcfg/methods.py`` (``METHODS = {id: (label, run, card)}``).
The method id is ``"dea"``; :data:`CARD` is its method card.

DEA as published, on the common inputs
--------------------------------------
The elevation is DEA Intertidal's own code, ``dea-intertidal==2.5.3``
(``intertidal/elevation.py``, Apache-2.0), with ``elevation()``'s default
parameters and nothing tuned: candidate filters (wet frequency of NDWI > 0.0
in [0.01, 0.99], Pearson correlation of wetness with tide >= 0.15), a
per-pixel rolling median of NDWI along tide height (105 windows of radius
0.15 of the pixel's tide range, more than 5 observations), 200 interpolated
intervals and a trailing mean of 20, elevation = the lowest smoothed tide at
which the smoothed NDWI exceeds 0.1, the MAD uncertainty, and the upper-edge
cleaning. Two inputs change, both forced by the common configuration and
declared in the card: (1) our NDWI = (B03 - B08)/(B03 + B08) with our clear
mask (SCL 4-7) and scene screening instead of DEA's own loading; (2) the
site's boundary tide h_B at each scene, without lag, instead of DEA's
per-pixel modelled tides. Also declared: the window (the n most recent
screened images) and the analysis domain (the common keep pixels).

Why a subprocess
----------------
DEA needs its own environment (``venv-dea``: dea-intertidal[datacube]
and its pins), and this repository has an unrelated top-level package also
called ``intertidal`` that shadows DEA's whenever Python runs from the
repository root. :func:`run` therefore writes the observations to an npz in
a temp dir OUTSIDE the repository and runs ``dea_worker.py`` there with the
side interpreter started with ``-P`` (no script or working directory on
``sys.path``) and without PYTHONPATH; the worker re-checks all of that and
the installed version and source hash before it runs DEA.

Side interpreter: ``$MAREA_DEA_PYTHON``, else ``venv-dea`` inside or next to
the repository (server: /media/HDD/jalias/Intertidal_analysis/venv-dea/bin/
python; laptop: C:/Users/Jorge/sketch_fitton/venv-dea/Scripts/python.exe).

Optional ``cfg`` keys (any mapping; unknown keys are ignored):
``dea_max_workers`` (DEA's process pool, default 4 or $MAREA_DEA_WORKERS),
``dea_chunk_px`` (pixels per chunk, default 6.25e6 / T clipped to
5,000..100,000, or $MAREA_DEA_CHUNK_PX), ``dea_gate`` (also run DEA's
unchunked ``elevation()`` and compare bitwise; tiles only), ``dea_tmp``
(temp root, default $MAREA_DEA_TMP or the system temp dir),
``dea_timeout_s``, ``dea_keep_tmp``.

Server smoke (one site, one window; no truth is read)::

    cd /media/HDD/jalias/Intertidal_analysis && source env.sh
    $PY -X utf8 experiments/fixedcfg/dea_intertidal.py --site villaviciosa --n 40 --gate
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORKER_PATH = HERE / "dea_worker.py"

try:
    from . import dea_worker as worker
except ImportError:  # run as a script or loaded by path
    import importlib.util

    _spec = importlib.util.spec_from_file_location("dea_worker", WORKER_PATH)
    worker = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(worker)

METHOD_ID = "dea"
LABEL = "DEA Intertidal 2.5.3"
ENV_PYTHON = "MAREA_DEA_PYTHON"
ENV_TMP = "MAREA_DEA_TMP"
ENV_WORKERS = "MAREA_DEA_WORKERS"
ENV_CHUNK = "MAREA_DEA_CHUNK_PX"
ENV_KEEP_TMP = "MAREA_DEA_KEEP_TMP"
DEFAULT_MAX_WORKERS = 4
#: Windows fallback named by the first author's set-up
WINDOWS_SIDE_PYTHON = Path("C:/Users/Jorge/sketch_fitton/venv-dea/Scripts/python.exe")
#: environment variables that could point the side interpreter at the
#: repository or at the main venv
_DROP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "PYTHONUSERBASE",
             "__PYVENV_LAUNCHER__", "VIRTUAL_ENV")


class DEAEnvironmentError(RuntimeError):
    """The DEA side interpreter or its temp dir is missing or unusable."""


class DEAWorkerError(RuntimeError):
    """The DEA worker subprocess failed; the message carries its log tail."""


# ─────────────────────────────────────────────────────────────────────────────
#  the side interpreter
# ─────────────────────────────────────────────────────────────────────────────

def side_python_candidates():
    """Default locations of the side interpreter, in search order."""
    exe = ("Scripts/python.exe", "bin/python") if os.name == "nt" else ("bin/python", "Scripts/python.exe")
    out = [root / e for root in (REPO_ROOT / "venv-dea", REPO_ROOT.parent / "venv-dea") for e in exe]
    if os.name == "nt":
        out.append(WINDOWS_SIDE_PYTHON)
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(str(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def find_side_python():
    """The DEA side interpreter: ``$MAREA_DEA_PYTHON`` if set (it must exist),
    else the first default location that exists. Raises
    :class:`DEAEnvironmentError` with what was tried."""
    explicit = os.environ.get(ENV_PYTHON, "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise DEAEnvironmentError(
                f"{ENV_PYTHON}='{explicit}' is not a file. Point it at the python of the side venv "
                f"that has dea-intertidal[datacube]=={worker.DEA_VERSION} (e.g. venv-dea/bin/python), "
                "or unset it to use the default locations.")
        return p
    tried = side_python_candidates()
    for p in tried:
        if p.is_file():
            return p
    raise DEAEnvironmentError(
        "DEA Intertidal side interpreter not found. Tried:\n  " + "\n  ".join(str(p) for p in tried)
        + f"\nCreate the side venv (python -m venv venv-dea; venv-dea/bin/pip install "
        f"'dea-intertidal[datacube]=={worker.DEA_VERSION}') or set {ENV_PYTHON} to its python. "
        "Never the main venv: the repository's own 'intertidal' package would shadow DEA's.")


def child_env(base=None):
    """The worker's environment: no PYTHONPATH/PYTHONHOME/venv hints, tqdm
    progress bars off, unbuffered output, a non-interactive matplotlib."""
    env = dict(os.environ if base is None else base)
    for k in _DROP_ENV:
        env.pop(k, None)
    env.update(TQDM_DISABLE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    return env


def _launch(cmd, cwd, env, log_path, timeout):
    """Run the worker; stdout and stderr go to ``log_path``. Returns the
    exit code (``subprocess.run`` is looked up here so tests can fake it)."""
    with open(log_path, "wb") as logf:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=logf,
                              stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return proc.returncode


def _tail(path, n=40):
    try:
        lines = Path(path).read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


# ─────────────────────────────────────────────────────────────────────────────
#  observations
# ─────────────────────────────────────────────────────────────────────────────

def _get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def obs_arrays(obs):
    """Checked arrays of a fixed-configuration ``obs``: (Y float32 (T, P),
    C bool (T, P), h float64 (T,), keep int64 (P,), shape (H, W), t)."""
    Y = np.asarray(_get(obs, "Y"))
    C = np.asarray(_get(obs, "C"))
    h = np.asarray(_get(obs, "h"), np.float64).ravel()
    keep = np.asarray(_get(obs, "keep"), np.int64).ravel()
    shape = tuple(int(v) for v in np.asarray(_get(obs, "shape")).ravel())
    if Y.ndim != 2:
        raise ValueError(f"obs Y must be (T, P), got {Y.shape}")
    T, P = Y.shape
    if C.shape != Y.shape:
        raise ValueError(f"obs C {C.shape} differs from Y {Y.shape}")
    if C.dtype != bool:
        C = C.astype(bool)
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32)
    if h.shape != (T,):
        raise ValueError(f"obs h has {h.shape}, expected ({T},) (one h_B per scene)")
    if keep.shape != (P,):
        raise ValueError(f"obs keep has {keep.shape}, expected ({P},)")
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"obs shape must be (H, W), got {shape}")
    if P and (keep.min() < 0 or keep.max() >= shape[0] * shape[1]):
        raise ValueError(f"obs keep outside the {shape[0]} x {shape[1]} frame")
    if P and len(np.unique(keep)) != P:
        raise ValueError("obs keep has repeated pixels")
    return Y, C, h, keep, shape, _get(obs, "t")


def time_ns(t, T):
    """Scene times as int64 ns (UTC, naive) and their source. DEA's
    elevation path never reads the time values (correct_seasonality=False),
    so a missing or partial ``t`` becomes a placeholder daily sequence."""
    placeholder = (np.datetime64("2000-01-01T00:00", "ns")
                   + np.arange(T) * np.timedelta64(1, "D")).astype(np.int64)
    if t is None:
        return placeholder, "placeholder (obs has no t)"
    import pandas as pd

    ti = pd.DatetimeIndex(t)
    if len(ti) != T:
        raise ValueError(f"obs t has {len(ti)} times for {T} scenes")
    if ti.tz is not None:
        ti = ti.tz_convert("UTC").tz_localize(None)
    if ti.isna().any():
        return placeholder, "placeholder (obs t has NaT)"
    return np.asarray(ti.values.astype("datetime64[ns]")).astype(np.int64), "obs t"


def write_input(path, Y, C, h, t_ns, keep, shape, params, block=65536):
    """The worker's input npz, streamed: NDWI set to NaN where not clear,
    written pixel-major (P, T) block by block (never a second full copy)."""
    T, P = Y.shape
    small = (("h", np.asarray(h, np.float64)), ("time_ns", np.asarray(t_ns, np.int64)),
             ("keep", np.asarray(keep, np.int64)), ("shape", np.asarray(shape, np.int64)),
             ("params", np.array(json.dumps(params))))
    nan = np.float32(np.nan)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in small:
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, arr, allow_pickle=False)
        with zf.open("ndwi_pt.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
                                                     "fortran_order": False, "shape": (int(P), int(T))})
            for p0 in range(0, P, block):
                p1 = min(P, p0 + block)
                blk = np.where(C[:, p0:p1], Y[:, p0:p1], nan).astype("<f4", copy=False)
                f.write(np.ascontiguousarray(blk.T).tobytes())


def default_chunk_px(T):
    """Pixels per chunk: about 6.25 M pixel-scenes (1.5-2 GB with 4 pool
    workers at T = 125), clipped to 5,000..100,000."""
    return int(min(100_000, max(5_000, 6_250_000 // max(1, int(T)))))


def _cfg(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict) or hasattr(cfg, "get"):
        try:
            v = cfg.get(key, None)
        except TypeError:
            v = None
    else:
        v = getattr(cfg, key, None)
    return default if v is None else v


def _options(cfg, T):
    def env_int(name, default):
        v = os.environ.get(name, "").strip()
        return int(v) if v else default

    tmp_root = _cfg(cfg, "dea_tmp") or os.environ.get(ENV_TMP, "").strip() or tempfile.gettempdir()
    timeout = _cfg(cfg, "dea_timeout_s")
    return {
        "max_workers": max(1, int(_cfg(cfg, "dea_max_workers", env_int(ENV_WORKERS, DEFAULT_MAX_WORKERS)))),
        "chunk_px": max(1, int(_cfg(cfg, "dea_chunk_px", env_int(ENV_CHUNK, default_chunk_px(T))))),
        "gate": bool(_cfg(cfg, "dea_gate", False)),
        "gate_max_cells": int(_cfg(cfg, "dea_gate_max_cells", worker.GATE_MAX_CELLS)),
        "tmp_root": str(tmp_root),
        "timeout": None if timeout is None else float(timeout),
        "keep_tmp": bool(_cfg(cfg, "dea_keep_tmp", False)) or os.environ.get(ENV_KEEP_TMP, "") == "1",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  run
# ─────────────────────────────────────────────────────────────────────────────

def run(obs, cfg=None):
    """DEA Intertidal 2.5.3 elevations on the common observations.

    Returns ``dict(z=float32 (P,), sigma=float32 (P,), meta=dict)``: ``z``
    is DEA's ``elevation`` and ``sigma`` its ``elevation_uncertainty`` (see
    :data:`SIGMA_TEXT`) at the keep pixels, NaN where DEA gives none.
    ``obs['W']`` is not used (DEA thresholds the NDWI itself)."""
    t_start = time.time()
    Y, C, h, keep, shape, t = obs_arrays(obs)
    T, P = Y.shape
    py = find_side_python()
    t_ns, t_src = time_ns(t, T)
    opt = _options(cfg, T)
    meta = {"method": METHOD_ID, "label": LABEL, "package": f"{worker.DEA_DIST}=={worker.DEA_VERSION}",
            "interpreter": str(py), "worker": str(WORKER_PATH), "n_px": int(P), "n_scenes": int(T),
            "site": _get(obs, "site"), "time_source": t_src, "sigma": SIGMA_TEXT,
            "inputs": dict(CARD["inputs"]), "max_workers": opt["max_workers"], "chunk_px": opt["chunk_px"]}
    if P == 0:
        meta.update(n_z=0, coverage=float("nan"), seconds={"total": 0.0})
        return dict(z=np.zeros(0, np.float32), sigma=np.zeros(0, np.float32), meta=meta)

    tmp_root = Path(opt["tmp_root"]).expanduser()
    if worker._inside(tmp_root, REPO_ROOT):
        raise DEAEnvironmentError(f"temp root {tmp_root} is inside the repository {REPO_ROOT}; the DEA "
                                  f"worker must run outside it (set {ENV_TMP} or cfg['dea_tmp'])")
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="marea_dea_", dir=str(tmp_root))).resolve()
    try:
        if worker._inside(work, REPO_ROOT):   # e.g. a symlinked temp root
            raise DEAEnvironmentError(f"DEA work dir {work} resolves inside the repository {REPO_ROOT}")
        inp, out, logp = work / "in.npz", work / "out.npz", work / "worker.log"
        params = {"chunk_px": opt["chunk_px"], "max_workers": opt["max_workers"], "gate": opt["gate"],
                  "gate_max_cells": opt["gate_max_cells"], "repo_root": str(REPO_ROOT)}
        t0 = time.time()
        write_input(inp, Y, C, h, t_ns, keep, shape, params)
        t_write = time.time() - t0
        cmd = [str(py), "-P", "-X", "utf8", str(WORKER_PATH), "--input", str(inp), "--output", str(out)]
        t0 = time.time()
        try:
            rc = _launch(cmd, str(work), child_env(), str(logp), opt["timeout"])
        except subprocess.TimeoutExpired:
            raise DEAWorkerError(f"DEA worker timed out after {opt['timeout']} s; log tail:\n"
                                 + "\n".join(_tail(logp))) from None
        t_worker = time.time() - t0
        tail = _tail(logp)
        if rc != 0 or not out.is_file():
            kept = ("kept" if opt["keep_tmp"] else
                    f"removed; set {ENV_KEEP_TMP}=1 or cfg['dea_keep_tmp'] to keep it")
            raise DEAWorkerError(f"DEA worker failed (exit code {rc}): {' '.join(cmd)}\n"
                                 f"working directory {work} ({kept}); log tail:\n" + "\n".join(tail))
        t0 = time.time()
        with np.load(out, allow_pickle=False) as R:
            z = np.asarray(R["z"], np.float32)
            sigma = np.asarray(R["sigma"], np.float32)
            wmeta = json.loads(R["meta"].item())
        t_read = time.time() - t0
    finally:
        if not opt["keep_tmp"]:
            shutil.rmtree(work, ignore_errors=True)
    if z.shape != (P,) or sigma.shape != (P,):
        raise DEAWorkerError(f"DEA worker returned z {z.shape}, sigma {sigma.shape} for {P} pixels")
    n_z = int(np.isfinite(z).sum())
    meta.update({
        "n_z": n_z, "coverage": n_z / P,
        "n_candidates": wmeta["n_candidates"], "n_no_obs": wmeta["n_no_obs"],
        "n_edge_removed": wmeta["n_edge_removed"], "n_sigma_zero": wmeta["n_sigma_zero"],
        "n_nonfinite_h": wmeta["n_nonfinite_h"], "n_chunks": wmeta["n_chunks"],
        "sigma_nan_where_z_nan": bool(np.array_equal(np.isnan(z), np.isnan(sigma))),
        "workdir": str(work) + ("" if opt["keep_tmp"] else " (removed)"),
        "seconds": {"write_input": round(t_write, 2), "worker": round(t_worker, 2),
                    "read_output": round(t_read, 2), "total": round(time.time() - t_start, 2),
                    "worker_detail": wmeta["seconds"]},
        "gate": wmeta.get("gate"),
        "dea": {k: wmeta[k] for k in ("dea_version", "elevation_py", "elevation_py_sha256", "python",
                                      "executable", "cwd", "sys_path", "versions", "start_method",
                                      "xarray_use_bottleneck_default", "parameters", "used", "tide_dtype")},
        "log_tail": tail[-12:],
    })
    return dict(z=z, sigma=sigma, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
#  the method card
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_TEXT = ("DEA's elevation_uncertainty (pixel_uncertainty, method 'mad'): the median absolute "
              "deviation, from the elevation, of the tide heights of the observations the elevation "
              "misclassifies at NDWI 0.1 (wet below it, dry above it), 0 when fewer than 3 are "
              "misclassified; DEA's one-sided uncertainty, not a standard error. NaN where z is NaN.")

CARD = {
    "id": METHOD_ID,
    "label": LABEL,
    "name": "DEA Intertidal elevation (Bishop-Taylor et al.), dea-intertidal 2.5.3",
    "family": "per-pixel rolling-median first-wet tide",
    "kind": "elev",
    "output": "elev",
    "version": worker.DEA_VERSION,
    "doi": "10.26186/149403",
    "references": [
        {"cite": ("Bishop-Taylor R, Phillips C, Sagar S, Newey V (2026). Remote Sensing of Environment "
                  "347:115663 (as listed in docs/validation_v2_spec.md)"),
         "doi_or_url": "https://doi.org/10.1016/j.rse.2026.115663"},
        {"cite": ("Bishop-Taylor R, Phillips C, Newey V, Sagar S (2024). Digital Earth Australia "
                  "Intertidal. Geoscience Australia, Canberra (dataset; the citation in the package)"),
         "doi_or_url": "https://dx.doi.org/10.26186/149403"},
        {"cite": ("dea-intertidal 2.5.3, Python package (Apache-2.0), GeoscienceAustralia/dea-intertidal; "
                  "intertidal/elevation.py executed unmodified"),
         "doi_or_url": "https://pypi.org/project/dea-intertidal/2.5.3/"},
        {"cite": ("Bishop-Taylor R, Sagar S, Lymburner L, Beaman RJ (2019). Between the tides: modelling the "
                  "elevation of Australia's exposed intertidal zone at continental scale. ECSS 223:115-128 "
                  "(NIDEM, the predecessor; cited by the package)"),
         "doi_or_url": "https://doi.org/10.1016/j.ecss.2019.03.006"},
    ],
    "code": {
        "package": f"dea-intertidal[datacube]=={worker.DEA_VERSION}", "license": "Apache-2.0",
        "module": "intertidal/elevation.py", "sha256": worker.ELEVATION_PY_SHA256,
        "functions": ["ds_to_flat", "pixel_rolling_median (-> rolling_tide_window)", "pixel_dem",
                      "pixel_uncertainty", "clean_edge_pixels"],
        "gate_reference": "elevation() (with pixel_tides replaced by the h_B broadcast)",
        "environment": ("side venv venv-dea; experiments/fixedcfg/dea_worker.py run with python -P in a "
                        "temp dir outside the repository; version, source hash and defaults checked"),
    },
    "fidelity": "F1",
    "fidelity_text": ("authors' code executed, not a port: DEA's own functions in DEA's own environment; "
                      "only the inputs change (see deviations)"),
    "evidence": [
        ("identity gate (G-dea): the chunked adapter output equals intertidal.elevation.elevation() "
         "run unchunked on the same frame with only pixel_tides replaced, bitwise in z and sigma, "
         "on the synthetic cube of tests/test_fixedcfg_dea.py (two chunks, holes in keep)"),
        ("real tile, laptop 2026-10-20: a 64 x 64 px Villaviciosa tile of the extraction, 40 scenes "
         "(stand-in record: the product folder is server-only), bitwise in z and sigma"),
        ("whole real site: pending, the server smoke with --gate (Villaviciosa at n = 40), the first "
         "gate on a fork platform, where DEA's pool workers use bottleneck medians"),
        ("positive control: a planted ramp under a step NDWI is recovered within 2 % of the tide range "
         "in RMSE (tests/test_fixedcfg_dea.py)"),
    ],
    "algorithm": [
        "tide_m = h_B at each scene for every pixel (float32), NaN where the NDWI is NaN",
        ("ds_to_flat: wet frequency of NDWI > 0.0 in [0.01, 0.99]; Pearson correlation of that wetness "
         "with tide_m >= 0.15"),
        ("pixel_rolling_median: 105 windows (window_offset 5 below the lowest tide, spacing range/100, "
         "radius 0.15 x the pixel's observed tide range); median NDWI and tide; > 5 observations"),
        ("pixel_dem: interpolation to 200 intervals; trailing rolling mean (radius 20, min_periods 5); "
         "elevation = lowest smoothed tide with smoothed NDWI > 0.1; always-wet / always-dry removed"),
        "pixel_uncertainty: MAD of the misclassified observations' tide heights (0 below 3), one-sided",
        "clean_edge_pixels: elevation > 0 pixels 4-adjacent to no data removed (elevation and uncertainty)",
    ],
    "parameters": {"elevation": dict(worker.ELEVATION_DEFAULTS),
                   **{k: dict(v) for k, v in worker.INNER_DEFAULTS.items()}},
    "tuning": [],
    "inputs": {
        "Y": "the common NDWI (B03 - B08)/(B03 + B08), set to NaN where C is False (DEA's no-data)",
        "C": "the common per-pixel clear mask (SCL 4-7), applied to Y",
        "h": "h_B at each scene, without lag, the same for every pixel (in place of DEA's per-pixel tides)",
        "keep, shape": "the pixels and their frame (clean_edge_pixels sees no data outside keep)",
        "t": "labels the time coordinate only (correct_seasonality=False: DEA never reads it)",
        "W": ("not used: DEA thresholds its NDWI itself; its candidate wetness NDWI > 0.0 equals the "
              "common W = C & (Y > 0) wherever Y is finite (DEA counts a clear scene with no NDWI as "
              "no data, W as dry), and its elevation threshold is 0.1"),
    },
    "deviations": [
        {"item": "imagery, masks, index",
         "published": ("DEA's load_data: GA ARD NBART Sentinel-2 (and Landsat 5/7/8/9 in the DEA CLI), "
                       "s2cloudless cloud and NBART contiguity masks (Landsat: fmask with opening/dilation), "
                       "datasets with <= 90 % cloud cover and the GQA filter, cubic resampling, solar-day "
                       "groups; NDWI = (green - nir_1)/(green + nir_1)"),
         "ours": ("the common NDWI (B03 - B08)/(B03 + B08) from Sentinel-2 L2A only, NaN where the common "
                  "clear mask (SCL 4-7) is False, on MAREA's production scene screening; the same NDWI "
                  "formula as DEA's Sentinel-2 branch (B03 = green, B08 = nir_1)"),
         "why": "forced by the common configuration (one external configuration for every method)"},
        {"item": "tide",
         "published": ("per-pixel modelled tides (eo_tides.pixel_tides: EOT20 by default in elevation(), a "
                       "locally ranked model ensemble in the DEA product), float32, set to NaN where the "
                       "satellite data are NaN"),
         "ours": ("the site's boundary tide h_B at each scene time, without lag, identical for every pixel, "
                  "cast to float32 and masked with the NDWI no-data exactly as elevation() masks its "
                  "modelled tides; passed in place of pixel_tides' output, not through "
                  "elevation(tide_data=...), because in 2.5.3 that path skips the no-data masking "
                  "(cloudy scenes would enter the correlation filter as dry and widen every pixel's "
                  "window range), which would change DEA's algorithm rather than its input"),
         "why": "forced by the common configuration (the same boundary tide for every method)"},
        {"item": "window",
         "published": "a 3-year epoch of all observations",
         "ours": "the n most recent screened images of the site (n = 125, 100, 75, 60, 50, 40)",
         "why": "forced by the common configuration (one table per n)"},
        {"item": "analysis domain",
         "published": ("elevation(valid_mask=None) by default; the DEA CLI adds Australian masks "
                       "(AusBathyTopo 250 m, dilated, and a coastal connectivity mask) over a whole tile"),
         "ours": ("valid_mask=None; DEA sees only the common keep pixels, the rest of the frame being no "
                  "data, also for clean_edge_pixels, whose upper-edge cleaning therefore acts at keep's "
                  "boundary (meta n_edge_removed counts it)"),
         "why": "forced by the common configuration (the common domain); no European equivalent of the masks"},
    ],
    "execution": [
        ("per-pixel steps in chunks of pixels (one-row images whose x labels are the keep positions); "
         "clean_edge_pixels once on the assembled frame: bitwise equal to elevation() (identity gate)"),
        "DEA's process pool capped at max_workers (default 4; DEA's own default is one per CPU)",
        ("xarray use_bottleneck=True only inside pixel_rolling_median, whose reductions (median, max, "
         "min, integer counts) are exact: bitwise neutral, about 6x faster"),
    ],
    "sigma": SIGMA_TEXT,
    "estimand": ("the lowest tide (h_B datum, no lag) at which the trailing-smoothed rolling median of NDWI "
                 "exceeds 0.1: the onset of inundation as seen through windows 0.3 of the tide range wide. "
                 "Relative to the common wet threshold (NDWI > 0) the 0.1 threshold and the trailing mean "
                 "shift it by a fraction of the smoothing length (about 0.1 of the tide range); for a step "
                 "NDWI whose dry and wet levels straddle 0.1 symmetrically the shift is about zero"),
    "needs_full_frame": False,
    "needs_full_frame_why": ("every DEA step is per pixel except clean_edge_pixels, which needs only the "
                             "elevation's no-data pattern on the frame; keep and shape give it"),
    "cost": ("laptop probe: about 4 us of CPU per pixel-scene, 80 % of it the rolling medians (bottleneck); "
             "with 4 pool workers expect roughly 1.5-2 us wall per pixel-scene on the server, i.e. "
             "6-8 min for the Ems at n = 125 (1.89 M px) and about 25 min for its six windows: an "
             "estimate the smoke replaces (seconds.worker_detail.us_per_px_scene); 1.5-2 GB at "
             "T = 125 with the default chunk of 50,000 px"),
    "low_n": ("DEA needs more than 5 clear observations in each rolling window (radius 0.15 of the "
              "pixel's tide range), so its coverage collapses when a pixel has fewer than about 20 "
              "clear observations: 1.5 % on a cloudy 40-scene Villaviciosa tile (about 15 clear "
              "observations per pixel). This is DEA's own behaviour, reported as coverage"),
}


def register(METHODS, method_id=METHOD_ID):
    """Add DEA Intertidal to the fixed-configuration registry
    ``METHODS = {id: (label, run, card)}`` and return it."""
    METHODS[method_id] = (LABEL, run, CARD)
    return METHODS


# ─────────────────────────────────────────────────────────────────────────────
#  tiles and the server smoke
# ─────────────────────────────────────────────────────────────────────────────

def tile_obs(obs, r0, r1, c0, c1):
    """The keep pixels of ``obs`` inside rows r0:r1 and columns c0:c1, as an
    obs on that (r1 - r0, c1 - c0) frame (for the gate on a real tile)."""
    Y, C, h, keep, (H, W), t = obs_arrays(obs)
    r, c = keep // W, keep % W
    m = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    w = c1 - c0
    out = dict(obs) if isinstance(obs, dict) else {}
    out.update(Y=Y[:, m], C=C[:, m], h=h, t=t, keep=((r[m] - r0) * w + (c[m] - c0)).astype(np.int64),
               shape=(r1 - r0, w))
    Wet = _get(obs, "W")
    if Wet is not None:
        out["W"] = np.asarray(Wet)[:, m]
    return out


def site_obs(site, n, tile=None):
    """Smoke-test observations of one site and window, built from the stored
    production files as the fixed configuration defines them: MAREA's
    screened record (result.json record_dates), its n most recent scenes,
    Y and C of the extraction, h_B = v7.build_record's boundary levels
    (lag 0). ``tile`` cuts a tile x tile box around keep's median pixel.
    Reads no truth. Runs in the main venv from the repository root."""
    import pandas as pd

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments import v14_scene_count_curves as v14
    from experiments import v7_waterline_contour as v7

    cfg = v14.site_config(site)
    if site != "villaviciosa":
        _, missing = v14.gauge_cache(site)
        if missing:
            raise SystemExit(f"[{site}] gauge cache missing for {missing}: refusing to download")
    with np.load(cfg["extract"], allow_pickle=False) as ex:
        keep_all = np.asarray(ex["keep"], np.int64)
        ext_dates = np.asarray(ex["dates"]).astype(str)
        H, W = (int(v) for v in ex["shape"])
    rec = v7.build_record(site, v7.SITES[site], ext_dates)
    res = json.load(open(f"{cfg['product']}/result.json", encoding="utf-8"))
    rec_dates = [str(d)[:10] for d in res["record_dates"]]
    pos = {d: i for i, d in enumerate(np.asarray(rec["dates"]).astype(str))}
    lost = [d for d in rec_dates if d not in pos]
    if lost:
        raise RuntimeError(f"{len(lost)} screened date(s) not in the record: {lost[:5]}")
    idx = np.array([pos[d] for d in rec_dates], np.int64)
    if np.any(np.diff(idx) <= 0):
        raise RuntimeError("screened record dates are not ascending")
    sel = idx[-int(n):]
    rows = np.asarray(rec["time_index"], np.int64)[sel]
    h = np.asarray(rec["h"], float)[sel]
    t = pd.DatetimeIndex(rec["t_real"])[sel]
    cols = np.arange(len(keep_all))
    frame = (H, W)
    keep = keep_all
    if tile:
        r, c = keep_all // W, keep_all % W
        rm, cm = int(np.median(r)), int(np.median(c))
        r0 = min(max(0, rm - tile // 2), max(0, H - tile))
        c0 = min(max(0, cm - tile // 2), max(0, W - tile))
        r1, c1 = min(H, r0 + tile), min(W, c0 + tile)
        m = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
        cols = np.flatnonzero(m)
        keep = ((r[m] - r0) * (c1 - c0) + (c[m] - c0)).astype(np.int64)
        frame = (r1 - r0, c1 - c0)
    Y = v14.stream_columns(cfg["extract"], "Y", rows, [cols])[0]
    C = v14.stream_columns(cfg["extract"], "C", rows, [cols])[0].astype(bool)
    return dict(Y=Y, C=C, W=C & (Y > 0), h=h, t=t, keep=keep, shape=frame, pixel_m=10, site=site,
                smoke={"n": int(n), "n_screened": len(idx), "tile": tile, "boundary": rec["boundary"],
                       "dates": [str(rec["dates"][i]) for i in (sel[0], sel[-1])]})


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Smoke test of the DEA Intertidal adapter on one site and "
                                             "window (no truth is read)")
    ap.add_argument("--site", required=True, choices=("villaviciosa", "escalda", "wadden", "ems"))
    ap.add_argument("--n", type=int, required=True, help="the n most recent screened images")
    ap.add_argument("--tile", type=int, default=None, help="cut a tile x tile box around keep's median")
    ap.add_argument("--gate", action="store_true", help="also run DEA's unchunked elevation() and compare")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk-px", type=int, default=None)
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--json", default=None, help="also write the summary here")
    a = ap.parse_args(argv)
    t0 = time.time()
    obs = site_obs(a.site, a.n, a.tile)
    t_obs = time.time() - t0
    cfg = {"dea_gate": a.gate, "dea_max_workers": a.workers, "dea_chunk_px": a.chunk_px,
           "dea_keep_tmp": a.keep_tmp}
    res = run(obs, cfg)
    m = res["meta"]
    out = {"site": a.site, "n": a.n, "tile": a.tile, "obs": obs["smoke"], "obs_seconds": round(t_obs, 1),
           "n_px": m["n_px"], "n_scenes": m["n_scenes"], "n_z": m["n_z"], "coverage": round(m["coverage"], 4),
           "n_candidates": m["n_candidates"], "n_edge_removed": m["n_edge_removed"],
           "n_sigma_zero": m["n_sigma_zero"], "z_quantiles": [
               float(v) for v in np.nanquantile(res["z"], [0.05, 0.5, 0.95])] if m["n_z"] else None,
           "seconds": m["seconds"], "gate": m["gate"], "dea": {k: m["dea"][k] for k in (
               "dea_version", "python", "start_method", "versions", "cwd")}}
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    if a.gate and not (m["gate"] or {}).get("passed"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
