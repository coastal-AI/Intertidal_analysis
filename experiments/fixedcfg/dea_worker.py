# -*- coding: utf-8 -*-
"""DEA Intertidal 2.5.3 elevation worker: runs in the SIDE interpreter only.

``experiments/fixedcfg/dea_intertidal.py`` (main venv) writes the common
observations to an npz OUTSIDE the repository and starts this script with
the side interpreter (``python -P``, working directory = that temp dir), so
``import intertidal`` finds DEA's package in the side venv's site-packages
and never this repository's unrelated top-level ``intertidal/`` package.
The script checks that at start-up, plus the installed version, the hash of
``intertidal/elevation.py`` and the defaults of every function it calls.

What runs is DEA Intertidal's own code (dea-intertidal==2.5.3,
``intertidal/elevation.py``, Apache-2.0), called exactly as
``elevation()`` calls it, with ``elevation()``'s default parameters:

* per pixel chunk (every step below is per pixel, so chunking changes
  nothing; the identity gate checks this bitwise):

  - ``tide_m`` = h_B broadcast to every pixel as float32, NaN where the
    NDWI is NaN. This stands in for ``eo_tides.eo.pixel_tides`` (float32
    heights) followed by ``elevation()``'s own masking of the modelled
    tides with the satellite no-data;
  - ``ds_to_flat(min_freq=0.01, max_freq=0.99, min_correlation=0.15,
    correct_seasonality=False, valid_mask=None)``, its own NDWI threshold
    0.0 (``elevation()`` does not pass its 0.1), Pearson correlation of
    the thresholded wetness with the tide;
  - ``pixel_rolling_median(windows_n=100, window_prop_tide=0.15,
    max_workers)``, i.e. ``rolling_tide_window`` over 105 windows
    (``window_offset`` 5), median, more than ``min_count`` 5 observations;
  - ``pixel_dem(ndwi_thresh=0.1)``: 200 interpolated intervals, trailing
    rolling mean (``smooth_radius`` 20, ``min_periods`` 5), lowest
    smoothed tide with NDWI > 0.1, always-wet / always-dry removed;
  - ``pixel_uncertainty(ndwi_thresh=0.1)``: MAD of the misclassified
    observations' tide heights (0 below 3 misclassified), one-sided;

* once on the assembled frame (pixels outside keep are no data):
  ``clean_edge_pixels`` on ``elevation`` and ``elevation_uncertainty``.

``pixel_rolling_median`` runs under ``xarray.set_options(use_bottleneck=
True)``: inside it only medians, maxima, minima and integer counts are
reduced, which bottleneck computes exactly as numpy does, so the result is
bitwise the same (the gate checks it) and about six times faster. On a
``fork`` platform (the server) DEA's pool workers inherit the option; on
``spawn`` (Windows) they use xarray's default.

The gate (``params["gate"]``) also runs ``intertidal.elevation.elevation()``
itself on the whole frame (cropped to keep's bounding box plus one pixel,
which leaves ``clean_edge_pixels`` unchanged), with ``pixel_tides``
replaced by the same h_B broadcast and every other step DEA's own code
path under default options, and compares bitwise (G-dea). Tiles only:
it holds the (T, H, W) cube in memory.

Input npz (written by the adapter)::

    ndwi_pt  float32 (P, T)  pixel-major NDWI, NaN where not clear or no data
    h        float64 (T,)    h_B at the scene times
    time_ns  int64   (T,)    scene times, ns since 1970 (UTC, naive)
    keep     int64   (P,)    flat frame indices of the pixels
    shape    int64   (2,)    frame (H, W)
    params   str             JSON: chunk_px, max_workers, gate,
                             gate_max_cells, repo_root

Output npz: ``z``, ``sigma`` (float32 (P,), NaN where DEA gives none) and
the diagnostics ``z_raw``, ``sigma_raw`` (before ``clean_edge_pixels``),
``freq``, ``corr``, ``clear``, ``candidate``, ``misclassified``,
``edge_removed``, plus ``z_gate``/``sigma_gate`` with the gate, and
``meta`` (JSON string).

Usage (the adapter does this)::

    <venv-dea python> -P -X utf8 dea_worker.py --input in.npz --output out.npz
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile

import numpy as np

DEA_DIST = "dea-intertidal"
DEA_VERSION = "2.5.3"
#: sha256 of intertidal/elevation.py in the dea_intertidal-2.5.3 wheel on PyPI
#: (identical in the laptop and server side venvs, checked 2026-10-20)
ELEVATION_PY_SHA256 = "34b2b92461cb2889e540bdeb612780157c0cd97cd307815a7cee126711d218f4"

#: ``elevation()``'s 2.5.3 defaults that reach the per-pixel steps
ELEVATION_DEFAULTS = {
    "tide_data": None, "valid_mask": None, "ndwi_thresh": 0.1, "min_freq": 0.01, "max_freq": 0.99,
    "min_correlation": 0.15, "windows_n": 100, "window_prop_tide": 0.15, "correct_seasonality": False,
}
#: defaults of the inner functions that ``elevation()`` leaves at their defaults
INNER_DEFAULTS = {
    "ds_to_flat": {"ndwi_thresh": 0.0, "index": "ndwi", "min_freq": 0.01, "max_freq": 0.99,
                   "min_correlation": 0.15, "corr_method": "pearson", "apply_threshold": True,
                   "correct_seasonality": False, "valid_mask": None},
    "pixel_rolling_median": {"windows_n": 100, "window_prop_tide": 0.15, "window_offset": 5,
                             "min_count": 5},
    "rolling_tide_window": {"min_count": 5, "statistic": "median"},
    "pixel_dem": {"ndwi_thresh": 0.1, "interp_intervals": 200, "smooth_radius": 20, "min_periods": 5},
    "pixel_uncertainty": {"ndwi_thresh": 0.1, "method": "mad", "min_misclassified": 3},
}
#: the functions of intertidal.elevation that this worker calls
DEA_FUNCTIONS = ("ds_to_flat", "pixel_rolling_median", "rolling_tide_window", "pixel_dem",
                 "pixel_uncertainty", "clean_edge_pixels", "elevation")
#: T * H * W above which the unchunked elevation() gate refuses to run
GATE_MAX_CELLS = 60_000_000
_MISSING = "<missing>"


def log(msg):
    print(f"[dea-worker {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  npz streaming (no DEA needed: also imported by the main-venv tests)
# ─────────────────────────────────────────────────────────────────────────────

def _read_exact(f, n):
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError("truncated npy member")
        parts.append(b)
        got += len(b)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def npy_header(f):
    """(shape, fortran_order, dtype) of the .npy stream ``f``."""
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"unsupported .npy format version {version}")


def iter_pixel_blocks(npz_path, name, block):
    """Yield ``(p0, rows)`` over the C-order pixel-major (P, T) member
    ``name`` of the npz, ``block`` pixels at a time; the member is streamed,
    never held whole."""
    block = max(1, int(block))
    with zipfile.ZipFile(npz_path) as zf, zf.open(name + ".npy") as f:
        shape, fortran, dtype = npy_header(f)
        if fortran or len(shape) != 2:
            raise ValueError(f"{npz_path}:{name}: expected a C-order 2-D array, got {shape} "
                             f"(fortran={fortran})")
        P, T = (int(v) for v in shape)
        row_bytes = T * dtype.itemsize
        for p0 in range(0, P, block):
            n = min(block, P - p0)
            buf = _read_exact(f, n * row_bytes) if n * row_bytes else b""
            yield p0, np.frombuffer(buf, dtype=dtype).reshape(n, T)


def _inside(path, root):
    path, root = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  environment
# ─────────────────────────────────────────────────────────────────────────────

def check_environment(repo_root):
    """Import DEA's ``intertidal.elevation`` and check that it is the side
    venv's dea-intertidal 2.5.3, byte for byte, and that neither the working
    directory nor ``sys.path`` can reach the repository's own ``intertidal``
    package. Returns (module, xarray, info); raises SystemExit otherwise."""
    import importlib.metadata as md
    import sysconfig

    problems = []
    cwd = os.getcwd()
    if not sys.flags.safe_path:
        problems.append("the interpreter was not started with -P (sys.flags.safe_path is off)")
    if repo_root:
        if _inside(cwd, repo_root):
            problems.append(f"working directory {cwd} is inside the repository {repo_root}")
        for p in sys.path:
            if p in ("", ".") or (os.path.isdir(p) and os.path.normcase(os.path.realpath(p))
                                  == os.path.normcase(os.path.realpath(repo_root))):
                problems.append(f"sys.path holds {p!r} (the working directory or the repository root); "
                                "start the interpreter with -P")
    if problems:
        raise SystemExit("DEA worker environment check failed:\n  " + "\n  ".join(problems))

    import xarray as xr
    import intertidal.elevation as E

    efile = os.path.realpath(E.__file__)
    libs = sorted({os.path.realpath(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")})
    if not any(_inside(efile, d) for d in libs):
        problems.append(f"intertidal.elevation imported from {efile}, outside the interpreter's "
                        f"site-packages {libs}")
    if repo_root and _inside(efile, os.path.join(repo_root, "intertidal")):
        problems.append(f"intertidal.elevation is the repository's own package ({efile})")
    version = md.version(DEA_DIST)
    if version != DEA_VERSION:
        problems.append(f"{DEA_DIST} {version} installed, {DEA_VERSION} required")
    sha = _sha256(efile)
    if sha != ELEVATION_PY_SHA256:
        problems.append(f"{efile} sha256 {sha} differs from the 2.5.3 wheel's {ELEVATION_PY_SHA256}")
    missing = [n for n in DEA_FUNCTIONS if not callable(getattr(E, n, None))]
    if missing:
        problems.append(f"intertidal.elevation lacks {missing}")
    if problems:
        raise SystemExit("DEA worker environment check failed:\n  " + "\n  ".join(problems))

    vers = {}
    for p in ("numpy", "xarray", "pandas", "scipy", "scikit-image", "bottleneck", "odc-algo", "odc-geo",
              "eo-tides", "dea-tools"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    import multiprocessing as mp

    info = {"dea_version": version, "elevation_py": efile, "elevation_py_sha256": sha,
            "python": sys.version.split()[0], "executable": sys.executable, "cwd": cwd,
            "sys_path": list(sys.path), "versions": vers, "start_method": mp.get_start_method(),
            "xarray_use_bottleneck_default": bool(xr.get_options()["use_bottleneck"])}
    return E, xr, info


def _defaults(fn):
    import inspect

    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


def dea_parameters(E):
    """The installed defaults of the functions called, checked against the
    2.5.3 values above; returns {function: {parameter: value}}."""
    got = {"elevation": _defaults(E.elevation)}
    for name in INNER_DEFAULTS:
        got[name] = _defaults(getattr(E, name))
    bad = []
    for name, want in [("elevation", ELEVATION_DEFAULTS)] + list(INNER_DEFAULTS.items()):
        for k, v in want.items():
            g = got[name].get(k, _MISSING)
            if not (g is v or (type(g) is type(v) and g == v)):
                bad.append(f"{name}({k}={g!r}), expected {v!r}")
    if bad:
        raise SystemExit("DEA defaults differ from 2.5.3: " + "; ".join(bad))
    return {name: {k: got[name][k] for k in want}
            for name, want in [("elevation", ELEVATION_DEFAULTS)] + list(INNER_DEFAULTS.items())}


# ─────────────────────────────────────────────────────────────────────────────
#  DEA's per-pixel steps on one chunk, then the frame step
# ─────────────────────────────────────────────────────────────────────────────

def _positions(da):
    return np.asarray(da["x"].values, np.int64)


def dea_chunk(E, xr, block, cols, h32, times, use, max_workers):
    """DEA's per-pixel steps on one chunk, as ``elevation()`` runs them.

    ``block`` float32 (n, T) NDWI (NaN = no clear observation), ``cols``
    int64 (n,) the pixels' positions in keep (they become the x labels of a
    one-row image, so DEA's stacked ``z`` maps straight back), ``h32``
    float32 (T,) h_B, ``use`` elevation()'s parameters. Returns
    {name: DataArray on the stacked z} (empty if nothing was observed)."""
    n, T = block.shape
    nd = np.ascontiguousarray(block.T)[:, None, :]                 # (time, y=1, x=n)
    coords = {"time": times, "y": np.zeros(1, np.int64), "x": np.asarray(cols, np.int64)}
    sat = xr.Dataset({"ndwi": (("time", "y", "x"), nd)}, coords=coords)
    if not np.isfinite(nd).any():
        # ds_to_flat would divide by a zero pixel count; nothing observed,
        # nothing for DEA to model (the unchunked run drops these too)
        return {}
    # h_B in place of eo_tides.pixel_tides (float32 heights), masked with the
    # satellite no-data as elevation() masks its modelled tides
    tide = xr.DataArray(np.broadcast_to(h32[:, None, None], nd.shape), dims=("time", "y", "x"),
                        coords=coords)
    sat["tide_m"] = tide.where(~sat["ndwi"].isnull())
    flat_ds, freq, corr, clear = E.ds_to_flat(
        sat, min_freq=use["min_freq"], max_freq=use["max_freq"], min_correlation=use["min_correlation"],
        correct_seasonality=use["correct_seasonality"], valid_mask=use["valid_mask"])
    out = {"freq": freq, "corr": corr, "clear": clear}
    if flat_ds.sizes.get("z", 0) == 0 or flat_ds.sizes.get("time", 0) == 0:
        return out
    with xr.set_options(use_bottleneck=True):   # exact medians/max/min/counts: bitwise neutral
        interval_ds = E.pixel_rolling_median(flat_ds, windows_n=use["windows_n"],
                                             window_prop_tide=use["window_prop_tide"],
                                             max_workers=max_workers)
    flat_dem = E.pixel_dem(interval_ds, use["ndwi_thresh"])
    _, _, unc, mis = E.pixel_uncertainty(flat_ds, flat_dem, use["ndwi_thresh"])
    out.update(elevation=flat_dem["elevation"], uncertainty=unc, misclassified=mis)
    return out


def frame_clean(E, xr, z_raw, s_raw, keep, shape):
    """``clean_edge_pixels`` once on the (H, W) frame (no data outside keep),
    as ``elevation()`` applies it to its unflattened output."""
    H, W = shape
    fe = np.full(H * W, np.nan, np.float32)
    fu = np.full(H * W, np.nan, np.float32)
    fe[keep] = z_raw
    fu[keep] = s_raw
    ds = xr.Dataset({"elevation": (("y", "x"), fe.reshape(H, W)),
                     "elevation_uncertainty": (("y", "x"), fu.reshape(H, W))},
                    coords={"y": np.arange(H), "x": np.arange(W)})
    cl = E.clean_edge_pixels(ds)
    z = np.asarray(cl["elevation"].values, np.float32).reshape(-1)[keep]
    s = np.asarray(cl["elevation_uncertainty"].values, np.float32).reshape(-1)[keep]
    return z, s


def run_chunked(E, xr, npz_path, h32, times, keep, shape, use, chunk_px, max_workers):
    P, T = len(keep), len(h32)
    o = {"z_raw": np.full(P, np.nan, np.float32), "sigma_raw": np.full(P, np.nan, np.float32),
         "freq": np.full(P, np.nan, np.float32), "corr": np.full(P, np.nan, np.float32),
         "clear": np.zeros(P, np.int32), "candidate": np.zeros(P, bool),
         "misclassified": np.full(P, -1, np.int32)}
    n_chunks = max(1, math.ceil(P / chunk_px))
    per_chunk = []
    t_all = time.time()
    for k, (p0, block) in enumerate(iter_pixel_blocks(npz_path, "ndwi_pt", chunk_px)):
        t0 = time.time()
        cols = np.arange(p0, p0 + len(block), dtype=np.int64)
        r = dea_chunk(E, xr, block, cols, h32, times, use, max_workers)
        if "clear" in r:
            o["clear"][_positions(r["clear"])] = np.asarray(r["clear"].values, np.int32)
            o["freq"][_positions(r["freq"])] = np.asarray(r["freq"].values, np.float32)
            o["corr"][_positions(r["corr"])] = np.asarray(r["corr"].values, np.float32)
        n_cand = 0
        if "elevation" in r:
            pos = _positions(r["elevation"])
            n_cand = len(pos)
            o["candidate"][pos] = True
            o["z_raw"][pos] = np.asarray(r["elevation"].values, np.float32)
            o["sigma_raw"][_positions(r["uncertainty"])] = np.asarray(r["uncertainty"].values, np.float32)
            m = np.asarray(r["misclassified"].values, float)
            o["misclassified"][_positions(r["misclassified"])] = np.where(np.isfinite(m), m, -1).astype(np.int32)
        dt = time.time() - t0
        per_chunk.append(round(dt, 2))
        log(f"chunk {k + 1}/{n_chunks}: {len(block):,} px, {n_cand:,} candidates, "
            f"{int(np.isfinite(o['z_raw'][p0:p0 + len(block)]).sum()):,} elevations [{dt:.1f} s]")
    t_px = time.time() - t_all
    t0 = time.time()
    z, s = frame_clean(E, xr, o["z_raw"], o["sigma_raw"], keep, shape)
    o["z"], o["sigma"] = z, s
    o["edge_removed"] = np.isfinite(o["z_raw"]) & ~np.isfinite(z)
    timing = {"pixel_steps": round(t_px, 2), "clean_edge_pixels": round(time.time() - t0, 2),
              "per_chunk": per_chunk, "us_per_px_scene": round(1e6 * t_px / max(1, P * T), 3)}
    return o, n_chunks, timing


# ─────────────────────────────────────────────────────────────────────────────
#  the identity gate: DEA's elevation() itself, unchunked
# ─────────────────────────────────────────────────────────────────────────────

def crop_box(keep, shape):
    """keep's bounding box plus one pixel, clipped to the frame: every keep
    pixel keeps its four neighbours, so clean_edge_pixels is unchanged."""
    H, W = shape
    r, c = keep // W, keep % W
    return (max(0, int(r.min()) - 1), min(H, int(r.max()) + 2), max(0, int(c.min()) - 1),
            min(W, int(c.max()) + 2))


def check_gate_size(keep, shape, T, max_cells):
    """Refuse (before any work) a gate whose unchunked cube would be too big."""
    r0, r1, c0, c1 = crop_box(keep, shape)
    cells = int(T) * (r1 - r0) * (c1 - c0)
    if cells > max_cells:
        raise SystemExit(f"gate: {T} x {r1 - r0} x {c1 - c0} = {cells:,} cells > {max_cells:,}; the "
                         "unchunked elevation() gate is for tiles (cut one with dea_intertidal.tile_obs "
                         "or the smoke's --tile)")
    return cells


def run_gate(E, xr, npz_path, h32, times, keep, shape, max_workers, max_cells):
    """``intertidal.elevation.elevation()`` on the cropped frame, with only
    ``pixel_tides`` replaced by the h_B broadcast. Returns (z, sigma, info)."""
    H, W = shape
    T = len(h32)
    check_gate_size(keep, shape, T, max_cells)
    r0, r1, c0, c1 = crop_box(keep, shape)
    h, w = r1 - r0, c1 - c0
    lk = (keep // W - r0) * w + (keep % W - c0)
    cube = np.full((T, h * w), np.nan, np.float32)
    for p0, blk in iter_pixel_blocks(npz_path, "ndwi_pt", 65536):
        cube[:, lk[p0:p0 + len(blk)]] = blk.T
    sat = xr.Dataset({"ndwi": (("time", "y", "x"), cube.reshape(T, h, w))},
                     coords={"time": times, "y": np.arange(r0, r1), "x": np.arange(c0, c1)})
    del cube

    def pixel_tides_hB(data, *args, **kwargs):
        shp = (T, data.sizes["y"], data.sizes["x"])
        return xr.DataArray(np.broadcast_to(h32[:, None, None], shp), dims=("time", "y", "x"),
                            coords={"time": data["time"].values, "y": data["y"].values,
                                    "x": data["x"].values})

    saved = E.pixel_tides
    E.pixel_tides = pixel_tides_hB
    t0 = time.time()
    try:
        ds, _ = E.elevation(sat, max_workers=max_workers)
    finally:
        E.pixel_tides = saved
    ze = np.asarray(ds["elevation"].transpose("y", "x").values, np.float32).reshape(-1)[lk]
    zs = np.asarray(ds["elevation_uncertainty"].transpose("y", "x").values, np.float32).reshape(-1)[lk]
    return ze, zs, {"box_rows": [r0, r1], "box_cols": [c0, c1], "seconds": round(time.time() - t0, 2)}


def _compare(a, b):
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    fin = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[fin].astype(np.float64) - b[fin].astype(np.float64))
    return {"bitwise": bool(np.array_equal(a, b, equal_nan=True)), "same_nan": same_nan,
            "n_differ": int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))),
            "max_abs_diff": float(d.max()) if d.size else 0.0}


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="DEA Intertidal 2.5.3 worker (side interpreter only)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    t_start = time.time()
    with np.load(a.input, allow_pickle=False) as f:
        prm = json.loads(f["params"].item())
        h = np.asarray(f["h"], np.float64)
        time_ns = np.asarray(f["time_ns"], np.int64)
        keep = np.asarray(f["keep"], np.int64)
        shape = tuple(int(v) for v in np.asarray(f["shape"]).ravel())
    T, P = len(h), len(keep)
    gate_max_cells = int(prm.get("gate_max_cells") or GATE_MAX_CELLS)
    if prm.get("gate"):
        check_gate_size(keep, shape, T, gate_max_cells)
    t0 = time.time()
    E, xr, info = check_environment(prm.get("repo_root"))
    params = dea_parameters(E)
    t_import = time.time() - t0
    use = dict(params["elevation"])
    chunk_px = max(1, int(prm["chunk_px"]))
    max_workers = max(1, int(prm["max_workers"]))
    log(f"dea-intertidal {info['dea_version']} from {info['elevation_py']} (python {info['python']}, "
        f"{info['start_method']}); {P:,} px x {T} scenes, chunks of {chunk_px:,} px, "
        f"{max_workers} pool worker(s) [import {t_import:.1f} s]")

    h32 = h.astype(np.float32)          # eo-tides' pixel_tides returns float32 heights
    times = time_ns.astype("datetime64[ns]")
    o, n_chunks, timing = run_chunked(E, xr, a.input, h32, times, keep, shape, use, chunk_px, max_workers)
    timing["import"] = round(t_import, 2)

    gate = None
    if prm.get("gate"):
        log("gate: intertidal.elevation.elevation() unchunked, pixel_tides -> h_B broadcast")
        zg, sg, ginfo = run_gate(E, xr, a.input, h32, times, keep, shape, max_workers, gate_max_cells)
        o["z_gate"], o["sigma_gate"] = zg, sg
        gate = {"z": _compare(o["z"], zg), "sigma": _compare(o["sigma"], sg), **ginfo,
                "reference": ("intertidal.elevation.elevation(satellite_ds, max_workers) with only "
                              "pixel_tides replaced by the h_B broadcast; xarray default options")}
        gate["passed"] = bool(gate["z"]["bitwise"] and gate["sigma"]["bitwise"])
        log(f"gate: z bitwise {gate['z']['bitwise']} ({gate['z']['n_differ']} differ), "
            f"sigma bitwise {gate['sigma']['bitwise']} ({gate['sigma']['n_differ']} differ) "
            f"[{ginfo['seconds']:.1f} s]")

    no_obs = ~np.isfinite(o["freq"])
    meta = dict(info)
    meta.update({
        "parameters": params, "used": {k: use[k] for k in ELEVATION_DEFAULTS},
        "tide_dtype": "float32 (h_B cast, as eo-tides' pixel_tides returns float32)",
        "chunk_px": chunk_px, "n_chunks": n_chunks, "max_workers": max_workers,
        "n_px": P, "n_scenes": T, "shape": list(shape), "n_nonfinite_h": int((~np.isfinite(h)).sum()),
        "n_no_obs": int(no_obs.sum()), "n_candidates": int(o["candidate"].sum()),
        "n_z_raw": int(np.isfinite(o["z_raw"]).sum()), "n_z": int(np.isfinite(o["z"]).sum()),
        "n_edge_removed": int(o["edge_removed"].sum()),
        "n_sigma_zero": int(np.count_nonzero(o["sigma"] == 0)),
        "seconds": dict(timing, total=round(time.time() - t_start, 2)),
        "gate": gate,
    })
    tmp = a.output + ".part"
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta)), **o)
    os.replace(tmp, a.output)
    log(f"done: {meta['n_z']:,} of {P:,} px with an elevation ({meta['n_edge_removed']:,} removed by "
        f"clean_edge_pixels) [{meta['seconds']['total']:.1f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
