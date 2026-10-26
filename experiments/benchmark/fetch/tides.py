# -*- coding: utf-8 -*-
"""Validation v2 — ``fetch tides``: the per-pixel global tide model LT (§1.6).

Question
--------
What does a global ocean tide model predict at every pixel of a site at the
record instants, computed the way DEA Intertidal, NIDEM and Coast X-Ray
height their scenes (eo-tides ``pixel_tides``, Bishop-Taylor et al. 2025,
doi:10.21105/joss.07786)?

Pre-registered rules (settings.LEVEL['LT'], levels.LEVEL_RULES['LT'])
---------------------------------------------------------------------
* ``eo_tides.eo.pixel_tides`` on the frame's GeoBox with ``resample=False``,
  ``resolution=1000`` m and ``buffer=1000`` m: the 1 km node grid. Every
  pixel takes its NEAREST node (declared: eo-tides would resample the nodes
  bilinearly onto the pixels).
* ``extrapolate=True, cutoff=inf, parallel=False``, as the production
  boundary ``PyTMDBoundary``: nodes on model land take the nearest ocean
  value, and the result does not depend on a process pool.
* Instants are passed in time chunks of at most :data:`TIME_CHUNK`
  (memory); the values do not depend on the chunking (each instant is an
  independent prediction).
* Models: the installed members of :data:`LT_CANDIDATES`, one eo-tides name
  per model (GOT4.8 and GOT4.7 are superseded by GOT4.10 of the same series
  and are not candidates, declared). EOT20 is the primary model.
* Ensemble rule: with fewer than 3 candidate models installed, LT is EOT20
  alone. With 3 or more, LT is the locally ranked ensemble: the equal-weight
  mean of the models ranked <= top_n at each node (eo-tides' default
  ``ensemble_func``). eo-tides' default ranking points (DEA's 2017-2019
  altimetry ranking) cover Australia only, so at these sites the ranks come
  from the truth-free BT26-rank of L12 with top_n = 1 (the spec's "model
  choice", declared). Without that ranking an ensemble LT is NOT COMPUTABLE.
* G-level (§10.1): the node series equal a direct ``PyTMDBoundary`` call at
  the node's longitude and latitude within 1 mm (:func:`check_points`).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

TIDE_DIR = "tide_models"
#: the primary LT model
PRIMARY = "EOT20"
#: (model, eo-tides names in order of preference); one name per model is used
LT_CANDIDATES = (
    ("EOT20", ("EOT20",)),
    ("FES2022", ("FES2022", "FES2022_extrapolated")),
    ("FES2014", ("FES2014", "FES2014_extrapolated")),
    ("GOT5.6", ("GOT5.6", "GOT5.6_extrapolated")),
    ("GOT5.5", ("GOT5.5", "GOT5.5_extrapolated")),
    ("GOT4.10", ("GOT4.10_nc", "GOT4.10", "GOT4.10_SAL")),
    ("TPXO10", ("TPXO10-atlas-v2-nc", "TPXO10-atlas-v2")),
    ("TPXO9", ("TPXO9-atlas-v5-nc", "TPXO9-atlas-v5")),
    ("HAMTIDE11", ("HAMTIDE11",)),
    ("DTU23", ("DTU23",)),
)
#: models needed for the ranked ensemble (§1.6)
ENSEMBLE_MIN_MODELS = 3
#: equal-weight mean of the models ranked <= top_n: eo-tides' default ranking
#: (3, its default) or the BT26-rank of L12 (1, the spec's model choice)
TOP_N = {"eo-tides": 3, "bt26": 1}
NODE_M = 1000.0
BUFFER_M = 1000.0
TIME_CHUNK = 256
#: G-level tolerance (m)
CHECK_TOL_M = 0.001


def log(msg):
    print(msg, flush=True)


def installed_models(directory=TIDE_DIR):
    """The eo-tides model names whose files are found in ``directory``."""
    import contextlib
    import io

    from eo_tides.utils import list_models

    with contextlib.redirect_stdout(io.StringIO()):
        available, _ = list_models(directory=directory, show_available=False,
                                   show_supported=False, raise_error=False)
    return sorted(str(m) for m in available)


def lt_models(directory=TIDE_DIR, available=None):
    """``[(model, eo-tides name)]`` of the installed candidates, primary first."""
    available = set(installed_models(directory) if available is None else available)
    out = []
    for model, names in LT_CANDIDATES:
        hit = [n for n in names if n in available]
        if hit:
            out.append((model, hit[0]))
    if not out or out[0][0] != PRIMARY:
        raise RuntimeError(f"the primary LT model {PRIMARY} is not installed in the tide "
                           f"model directory (found {sorted(available)})")
    return out


def lt_rule(models):
    """"primary" (EOT20 alone) or "ensemble" (>= 3 candidate models)."""
    return "ensemble" if len(models) >= ENSEMBLE_MIN_MODELS else "primary"


# ─────────────────────────────────────────────────────────────────────────────
#  the node grid
# ─────────────────────────────────────────────────────────────────────────────

def grid_axes(grid):
    """``(xs, ys, crs)`` of an ``inputs.Grid`` or a ``{"xs", "ys", "crs"}``
    mapping (pixel centres; ys descending)."""
    if isinstance(grid, dict):
        xs, ys, crs = grid["xs"], grid["ys"], grid.get("crs", grid.get("crs_wkt"))
    else:
        xs, ys = grid.xs, grid.ys
        crs = getattr(grid, "crs_wkt", None) or getattr(grid, "crs", None)
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("the grid needs at least 2 x 2 pixels")
    if crs is None:
        raise ValueError("the grid has no CRS")
    return xs, ys, crs


def geobox(grid):
    """The frame's odc-geo GeoBox (regular grid, ys descending)."""
    from affine import Affine
    from odc.geo.geobox import GeoBox

    xs, ys, crs = grid_axes(grid)
    rx, ry = float(xs[1] - xs[0]), float(ys[0] - ys[1])
    if rx <= 0 or ry <= 0 or not np.allclose(np.diff(xs), rx) or \
            not np.allclose(-np.diff(ys), ry):
        raise ValueError("the grid must be regular with x ascending and y descending")
    aff = Affine(rx, 0.0, float(xs[0]) - rx / 2, 0.0, -ry, float(ys[0]) + ry / 2)
    return GeoBox((len(ys), len(xs)), aff, crs)


def nearest_index(nodes, coords):
    """Index of the nearest node for each coordinate along one axis (ties to
    the lower index). On a rectilinear node grid the per-axis nearest nodes
    are the 2-D nearest node."""
    nodes = np.asarray(nodes, np.float64)
    coords = np.asarray(coords, np.float64)
    return np.abs(coords[:, None] - nodes[None, :]).argmin(axis=1).astype(np.int64)


def pixel_nodes(grid, pix_flat, x_nodes, y_nodes):
    """The node (flat index ``row * nx + col`` of the node grid) of each pixel."""
    xs, ys, _ = grid_axes(grid)
    W = len(xs)
    pix_flat = np.asarray(pix_flat, np.int64)
    cx = nearest_index(x_nodes, xs)
    cy = nearest_index(y_nodes, ys)
    return cy[pix_flat // W] * len(x_nodes) + cx[pix_flat % W]


def node_tides(grid, t, models, directory=TIDE_DIR, node_m=NODE_M, buffer_m=BUFFER_M,
               time_chunk=TIME_CHUNK, pixel_tides=None):
    """Tide heights of every model at every 1 km node and instant.

    Parameters
    ----------
    grid : inputs.Grid or mapping
    t : DatetimeIndex
        Naive-UTC record instants.
    models : list of (model, eo-tides name)
    pixel_tides : callable, optional
        ``eo_tides.eo.pixel_tides`` (default); tests inject a fake with its
        signature.

    Returns
    -------
    dict
        ``x`` (nx,), ``y`` (ny,) node centres in the frame CRS; ``models``
        (labels); ``names`` (eo-tides names); ``H`` (M, ny * nx, T) float64.
    """
    if pixel_tides is None:
        from eo_tides.eo import pixel_tides
    t = pd.DatetimeIndex(t)
    if t.tz is not None:
        raise ValueError("instants must be naive UTC")
    gb = geobox(grid)
    names = [n for _, n in models]
    x_nodes = y_nodes = None
    H = None
    t0 = time.time()
    for a in range(0, len(t), int(time_chunk)):
        tt = t[a:a + int(time_chunk)]
        da = pixel_tides(gb, time=tt, model=names, directory=directory, resample=False,
                         resolution=float(node_m), buffer=float(buffer_m), extrapolate=True,
                         cutoff=np.inf, parallel=False)
        if "tide_model" not in da.dims:
            da = da.expand_dims(tide_model=names)
        y_dim, x_dim = gb.dimensions
        da = da.transpose("tide_model", "time", y_dim, x_dim)
        got = [str(m) for m in da["tide_model"].values]
        if sorted(got) != sorted(names):
            raise AssertionError(f"pixel_tides returned models {got}, asked {names}")
        da = da.sel(tide_model=names)
        xn = np.asarray(da[x_dim].values, np.float64)
        yn = np.asarray(da[y_dim].values, np.float64)
        if x_nodes is None:
            x_nodes, y_nodes = xn, yn
            H = np.empty((len(names), len(yn) * len(xn), len(t)), np.float64)
        elif not (np.array_equal(xn, x_nodes) and np.array_equal(yn, y_nodes)):
            raise AssertionError("the node grid changed between time chunks")
        if not np.array_equal(pd.DatetimeIndex(da["time"].values).asi8, tt.asi8):
            raise AssertionError("pixel_tides returned other instants than asked")
        v = np.asarray(da.values, np.float64)                  # (M, n, ny, nx)
        H[:, :, a:a + len(tt)] = v.reshape(len(names), len(tt), -1).transpose(0, 2, 1)
        log(f"[fetch tides] {min(a + len(tt), len(t))}/{len(t)} instants x "
            f"{len(names)} model(s) x {H.shape[1]} nodes ({time.time() - t0:.0f} s)")
    return {"x": x_nodes, "y": y_nodes, "models": [m for m, _ in models], "names": names,
            "H": H}


def combine(H_members, ranks, top_n):
    """The ranked ensemble: equal-weight mean of the members ranked <= top_n.

    Parameters
    ----------
    H_members : (M, G, T)
    ranks : (M,) or (G, M) array
        1 = best; per node (G, M) or one ranking for the site (M,).
    top_n : int

    Returns
    -------
    (G, T) float64 and the (G, M) boolean membership.
    """
    H_members = np.asarray(H_members, np.float64)
    M, G, _ = H_members.shape
    r = np.asarray(ranks, np.float64)
    if r.shape == (M,):
        r = np.broadcast_to(r, (G, M))
    if r.shape != (G, M):
        raise ValueError(f"ranks must have shape ({M},) or ({G}, {M}); got {r.shape}")
    use = r <= int(top_n)
    if not use.any(axis=1).all():
        raise ValueError("a node has no member ranked <= top_n")
    w = use.astype(np.float64) / use.sum(axis=1, keepdims=True)    # (G, M)
    out = np.einsum("gm,mgt->gt", w, H_members)
    return out, use


def check_points(nodes, crs, t, H, model_name, directory=TIDE_DIR, n_points=5,
                 provider=None, tol_m=CHECK_TOL_M):
    """G-level for LT: node series = a direct ``PyTMDBoundary`` call at the
    node's (lat, lon) within ``tol_m``.

    Parameters
    ----------
    nodes : (k, 2) array of node (x, y) in ``crs``; the first ``n_points``
        are checked
    H : (k, T) the node series of ``model_name``
    provider : callable, optional
        ``provider(model, lat, lon)`` -> object with ``levels`` (default
        ``PyTMDBoundary``).
    """
    from pyproj import Transformer

    if provider is None:
        from pyintertidal.boundary import PyTMDBoundary as provider
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    rows = []
    for i in range(min(int(n_points), len(nodes))):
        lon, lat = tr.transform(float(nodes[i][0]), float(nodes[i][1]))
        ref = np.asarray(provider(model_name, lat, lon, directory=directory).levels(t), float)
        d = float(np.nanmax(np.abs(ref - np.asarray(H[i], float))))
        rows.append({"node": i, "lon": lon, "lat": lat, "max_abs_m": d})
    worst = max((r["max_abs_m"] for r in rows), default=float("nan"))
    return {"pass": bool(rows) and worst <= tol_m, "tol_m": tol_m, "max_abs_m": worst,
            "points": rows}
