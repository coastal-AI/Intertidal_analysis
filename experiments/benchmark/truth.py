# -*- coding: utf-8 -*-
"""v13 benchmark — the truths every method is scored against (§1.7, §5.1, §5.5, §3.7).

Question
--------
Against what is each method's elevation and lag judged? The truths, the
scoring domains and the global-model baselines are built once per site from
the common package and never look at a method, so every method meets the
same yardstick at every N.

Pre-registered rules (FROZEN['truth_elev'], FROZEN['truth_lag'],
FROZEN['judges'], FROZEN['baselines'])
------------------------------------------------------------------
Elevation (§5.1):
* Vaklodingen (Escalda, Wadden, Ems): ``v1.truth_grid(site)`` (latest finite
  survey per 20 m RD cell in 2019-2025, NAP); every frame pixel centre is
  transformed to EPSG:28992 and looked up with the arithmetic of
  ``v1.sample_truth``, extended to return the survey year and the id of the
  cell (sheet, iy, ix) the value comes from.
* LiDAR (Villaviciosa): ``v10.lidar_truth(rtk.DEFAULT_GRID)`` (PNOA MDT02,
  ground-only, averaged to the grid); its grid must be the cube's.
* RTK (Villaviciosa): ``rtk.load_rtk()`` with its default arguments only
  (development points), heights ``elev_alicante``, blocks of 150 m; never
  range-filtered.
* Planted (positive control): the planted z of ``control.synth``.
* Range filter, the same for every method and N: [min h0 - 0.25,
  max h0 + 0.25] over the full record's boundary levels.
* D_truth = frame pixels whose truth passes the filter (RTK: every
  development point); D_keep = D_truth ∩ keep_full; the support
  S = D_truth ∪ keep_full ∪ every judge neighbourhood ∪ RTK pixels, on
  which every product is stored (``support.npy``).

Lags (§5.5), per judge, fixed at the full record:
* (a) the best constant lag at the record's S2 instants: the gauge as
  delivered, nearest within 6 min; E(tau) = sqrt(mean_i (d_i - median d)^2),
  d_i = h_B(t_i - tau) - g_i from the exact level table, tau = -60..240 min
  at 1 min; tau_a = argmin E; I_a = [min, max] of {tau : E <= E_min + 1 cm}.
* tau_ab (gain-aware): the same with a free gain, E_ab(tau) = min over b of
  the median-centred RMS of g - b h_B(t - tau) (a = the median, as in E;
  b > 0 searched on a 0.01 grid over [0, 4] and refined); I_ab by the same
  1-cm tolerance. E_ab <= E at every tau (b = 1 is on the grid).
* (b) the M2 harmonic phase lag: gauge (centred 10-min series) and h_B
  (the 10-min series) on the instants where both are finite;
  ``pyTMD.solve.constants`` with :data:`CONS_B` (52 constituents), GOT
  nodal corrections, polynomial order 1; tau_b = wrap180(g_gauge - g_B) x
  60 / 28.9841042; SE by a delete-one jackknife over the calendar quarters
  of 2023-2025; I_b = tau_b +- 1.96 SE.
* Also recorded: 200 calendar-month block bootstraps of (a) (seed 20261011),
  edge flags, (a) with gaps <= 20 min interpolated, tau_all over every 10-min
  sample, E0, E_min, A_M2 of gauge and h_B, M4/M2, HW-only and LW-only lags,
  and (c), diagnostic: the GM20 frame (fold phase of the gauge on hours since
  the boundary's low-passed HW marks, raw and minus that of h0) and the HD26
  oracle (HD26's M2 phase of the gauge minus that of h0 on the same instants).
* Baselines (§3.7), lower bounds only: EOT20 and GOT5.6 (when loadable) at
  each gauge through the same (a), tau_ab and (b) machinery with the model's
  levels in place of the gauge, with the model's nearest ocean cell distance.

Outputs (``truth/<site>/``): truth_elev.{npz,json}, support.npy,
judges.json (the score-side judge table), truth_lag.json,
truth_lag_curves.npz and lag_truth.csv.

Validation v2 (docs/validation_v2_spec.md §1.7-§1.9, §6.2, §7.1; the
section at the end of this module, frozen in :data:`TRUTH_V2`; everything
above stays as v13 for the continuity view and the gates)
----------------------------------------------------------------------
* Multi-epoch Vaklodingen: every survey of 2016-2025 per 20 m cell
  (:func:`vaklodingen_cells`); the default view is SURVEY-MATCHED, the survey
  whose epoch (year + 0.5) is nearest the window centre, ties to the later
  (:func:`survey_matched`), with its epoch gap; v13's latest-survey view is
  kept for continuity; stable cells (>= 2 surveys within the window +- 1
  year, range <= 0.20 m). At the Dutch sites the truth CELL is the scoring
  support: a product's cell value is the mean of its mapped 10 m pixels,
  scored when >= 2 and >= half of them are mapped (:func:`cell_aggregate`).
* PNOA MDT02 ground-only (:func:`pnoa_truth`): water = 4-connected patches of
  >= 50 px whose values round to the same mm, plus every pixel below the
  flight-time water level + 0.15 m (:func:`flight_water_level`: EOT20 at the
  site plus the Gijon residual); without a flight time only the patch mask
  applies and the site is descriptive.
* RTK: ``load_rtk()`` defaults, Alicante datum; the campaign epoch from the
  file's time column (:func:`rtk_epoch`).
* Range filter: [q_0.001(h_B) - 0.25, q_0.999(h_B) + 0.25] over the frozen
  1-min h_B (:func:`range_from_series`), with counts per band; the local
  (LG) range as a sensitivity.
* Truth-error model (:func:`truth_error_model`): sigma2_s from the D(dt)
  curve of yearly pairs (the larger of the isotonic and power-law
  intercepts), same-year overlaps, or PNOA's stated accuracy; D_W(Y) with
  each survey's noise counted once; transfer by the D(dt) ratio;
  lambda = 1 - sigma2_T / Var(t); multipliers 0.5, 1, 1.5.
* Strata (:func:`strata_arrays`): s and band, reference lag and amplification
  interpolated along s between the judges, hydroperiod zone, position
  relative to the sampled range, truth slope, distance to permanent water,
  substrate (low-tide dry rho11 terciles) and vegetation (NDVI), turbidity
  (channel-anchor rho4 terciles), n_clear, epoch gap, source class.
* Per-window lag truths (:func:`lag_truths_window`): (a), tau_ab, (b) with
  quarter (month) jackknife, tau_all, HW/LW from both series low-passed with
  the 61-min mean, the M2 amplitude ratio; the gauge centred with its
  provider's time-stamp convention; the coverage rule per window.

v2 outputs: truth_elev_v2.{npz,json}, truth_lag_v2.json,
truth_lag_v2_curves.npz and lag_truth_v2.csv.
"""
from __future__ import annotations

import os
import time
import warnings
import zlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from experiments.benchmark import config, judges, levels, product

_TE = config.FROZEN["truth_elev"]
_TL = config.FROZEN["truth_lag"]
TRUTH_ELEV_FORMAT = "v13-truth-elev-1"
TRUTH_LAG_FORMAT = "v13-truth-lag-1"
#: the range filter's margin around the record's boundary range (m)
RANGE_PAD_M = float(_TE["range_pad_m"])
#: (a): the lag grid (min), the gauge tolerance (min), the interval (m)
TAUS_A = np.arange(int(_TL["a"]["tau_range_min"][0]), int(_TL["a"]["tau_range_min"][1]) + 1,
                   int(_TL["a"]["step_min"]), dtype=np.int64)
TOL_MIN = float(_TL["a"]["tolerance_min"])
INTERVAL_M = float(_TL["a"]["interval_m"])
#: (a) with interpolated gauge values: largest bridged gap (min)
GAP_INTERP_MIN = 20.0
#: tau_ab: the gain grid searched before refinement
GAIN_GRID = np.round(np.arange(0, 401) * 0.01, 2)
#: (b)
Z_B = float(_TL["b"]["z"])
B_CORRECTIONS = str(_TL["b"]["corrections"])
B_ORDER = int(_TL["b"]["order"])
#: the 52 constituents of truth (b) (pyTMD names; duplicate frequencies
#: such as mks2 = delta2, mns2 = eps2, 2mk3 = mo3 left out)
CONS_B = ("sa", "ssa", "mm", "msf", "mf",
          "2q1", "sigma1", "q1", "rho1", "o1", "m1", "chi1", "pi1", "p1", "s1", "k1",
          "psi1", "phi1", "theta1", "j1", "oo1",
          "eps2", "2n2", "mu2", "n2", "nu2", "m2", "lambda2", "l2", "t2", "s2", "r2",
          "k2", "2sm2",
          "mo3", "m3", "mk3", "sk3",
          "mn4", "m4", "ms4", "mk4", "s4", "sk4",
          "2mn6", "m6", "msn6", "2ms6", "2mk6", "msk6",
          "3ms8", "m8")
#: the month-block bootstrap of (a)
N_BOOT = int(_TL["bootstrap"]["n"])
BOOT_SEED = int(_TL["bootstrap"]["seed"])
#: M2 speed (deg/h) and minutes per degree of M2 phase
OMEGA_M2 = 28.9841042
MIN_PER_DEG_M2 = 60.0 / OMEGA_M2
#: pyTMD's time origin
EPOCH_1992 = np.datetime64("1992-01-01T00:00:00", "ns")
#: GM20's fold period (h) and wrap window (min), for truth (c)
GM20_PERIOD_H = float(config.FROZEN["gm20"]["period_h"])
GM20_PERIOD_MIN = 60.0 * GM20_PERIOD_H
GM20_WRAP_LO = float(config.FROZEN["gm20"]["wrap_lo_min"])
#: HW/LW marks of gauge and h_B are paired within this many minutes
MARK_MATCH_MIN = 180.0
#: baselines
MODELS = tuple(config.FROZEN["baselines"]["models"])
del _TE, _TL


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  small helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap180(deg):
    """Phase difference in degrees, wrapped to [-180, 180)."""
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def wrap_gm20(minutes):
    """GM20's wrap of a fold phase (min) to [-120, 625.24)."""
    return (np.asarray(minutes, float) - GM20_WRAP_LO) % GM20_PERIOD_MIN + GM20_WRAP_LO


def days_since_1992(t):
    """Days since 1992-01-01 (pyTMD's time variable), float64."""
    v = pd.DatetimeIndex(t).as_unit("ns").values
    return (v - EPOCH_1992) / np.timedelta64(1, "D")


def lookup(sorted_flat, flat):
    """Positions of ``flat`` in an ascending array and whether they are
    there, without the IndexError of ``a[searchsorted(a, x)]`` past the end."""
    a = np.asarray(sorted_flat, np.int64)
    q = np.asarray(flat, np.int64)
    pos = np.searchsorted(a, q)
    if not len(a):
        return pos, np.zeros(len(q), bool)
    found = a[np.minimum(pos, len(a) - 1)] == q
    found &= pos < len(a)
    return pos, found


def truth_dir(pkg):
    return pkg.cfg.truth_dir


def _package_key(pkg):
    keys = getattr(pkg, "keys", None)
    if isinstance(keys, Mapping):
        return keys.get("package_key")
    return None


def _code_shas():
    here = os.path.dirname(os.path.abspath(__file__))
    return {name: config.sha256_source(os.path.join(here, name))
            for name in ("truth.py", "judges.py")}


def _file_entry(path):
    """Provenance of a source file: path, size and content sha256."""
    if path is None or not os.path.exists(path):
        return {"path": path, "missing": True}
    e = product.file_entry(path, sha=True)
    return {"path": str(path).replace("\\", "/"), "size": e["size"], "sha256": e["sha256"]}


def _file_id(path):
    if path is None or not os.path.exists(path):
        return {"path": path, "missing": True}
    st = os.stat(path)
    return {"path": str(path).replace("\\", "/"), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def _save_npy(path, arr):
    """``np.save`` atomically (temporary file + os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}.npy"
    np.save(tmp, np.asarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _done(json_path, key):
    """True when ``json_path`` exists and records ``inputs_key == key``."""
    if not os.path.exists(json_path):
        return False
    try:
        return product.read_json(json_path).get("inputs_key") == key
    except (OSError, ValueError):
        return False


# package accessors: the attributes of §1.4, with the fallbacks a package
# under construction may need ---------------------------------------------

def series1(pkg):
    """(t_min, h): the 1-min boundary series of the package."""
    s = getattr(pkg, "series1", None)
    if s is None:
        raise AttributeError("the package has no 1-min boundary series (series1)")
    if isinstance(s, Mapping):
        return np.asarray(s["t_min"], float), np.asarray(s["h"], float)
    return np.asarray(s[0], float), np.asarray(s[1], float)


def series10(pkg):
    """(DatetimeIndex, h): the 10-min boundary series of the package (from
    the 1-min series when the package does not carry it)."""
    s = getattr(pkg, "series10", None)
    if s is None:
        idx, h = levels.series_10min(*series1(pkg))
    elif isinstance(s, Mapping):
        idx, h = s["t"], s["h"]
    else:
        idx, h = s[0], s[1]
    return pd.DatetimeIndex(idx).as_unit("ns"), np.asarray(h, float)


def marks_lp(pkg):
    """The low-passed boundary marks (``levels.build_marks``' marks_lp)."""
    mk = getattr(pkg, "marks_lp", None)
    if mk is None:
        raise AttributeError("the package has no marks_lp")
    return mk


def record_t(pkg):
    return pd.DatetimeIndex(pkg.t).as_unit("ns")


# ─────────────────────────────────────────────────────────────────────────────
#  elevation truth (§5.1)
# ─────────────────────────────────────────────────────────────────────────────

def sample_cells(cells, xq, yq):
    """``v1.sample_truth`` plus the cell each value comes from.

    The same arithmetic: for each sheet in ``cells`` order, the nearest
    cell by rounding ``(q - axis[0]) / median step``; the first sheet with a
    finite value wins.

    Returns
    -------
    z (n,) float64, year (n,) int (-1 none), sheet (n,) int16 (index in
    ``cells`` order, -1 none), iy (n,) int32, ix (n,) int32
    """
    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)
    n = len(xq)
    out = np.full(n, np.nan)
    yr = np.full(n, -1)
    sheet = np.full(n, -1, np.int16)
    IY = np.full(n, -1, np.int32)
    IX = np.full(n, -1, np.int32)
    with np.errstate(invalid="ignore"):          # NaN query points never match
        for k, (x, y, zt, yt) in enumerate(cells.values()):
            dx = float(np.median(np.diff(x)))
            dy = float(np.median(np.diff(y)))
            ix = np.round((xq - x[0]) / dx).astype(int)
            iy = np.round((yq - y[0]) / dy).astype(int)
            ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
            v = np.full(n, np.nan)
            v[ok] = zt[iy[ok], ix[ok]]
            take = np.isfinite(v) & ~np.isfinite(out)
            out[take] = v[take]
            yr[take] = yt[iy[take], ix[take]]
            sheet[take] = k
            IY[take] = iy[take]
            IX[take] = ix[take]
    return out, yr, sheet, IY, IX


def cell_ids(sheet, iy, ix):
    """One int64 id per cell: sheet * 1e12 + iy * 1e6 + ix (-1 none)."""
    sheet = np.asarray(sheet, np.int64)
    return np.where(sheet >= 0,
                    sheet * 1_000_000_000_000 + np.asarray(iy, np.int64) * 1_000_000
                    + np.asarray(ix, np.int64), -1)


def vaklodingen_frame(pkg, cells=None, block_rows=256):
    """Vaklodingen truth on the frame, with survey year and cell ids.

    Parameters
    ----------
    pkg : CommonPackage
    cells : dict, optional
        ``v1.truth_grid(site)`` (read when not given).
    block_rows : int
        Frame rows transformed per block (memory only; results are the same).

    Returns
    -------
    dict
        z_raw (H, W) float64, year (H, W) int16, sheet (H, W) int16,
        iy/ix (H, W) int32, cell_id (H, W) int64, sheets (names, ``cells``
        order).
    """
    import pyproj

    from experiments import v1_vaklodingen_validate as v1

    years = tuple(int(y) for y in config.FROZEN["truth_elev"]["vaklodingen"]["years"])
    if years != tuple(v1.YEARS):
        raise AssertionError(f"v1.YEARS {v1.YEARS} differs from FROZEN {years}")
    if cells is None:
        cells = v1.truth_grid(pkg.cfg.site)
    if not cells:
        raise FileNotFoundError(f"no Vaklodingen sheet of {pkg.cfg.site} in {v1.TRUTH_DIR}")
    grid = judges.grid_of(pkg)
    H, W = judges.frame_shape(pkg)
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xs, ys = np.asarray(grid["xs"], float), np.asarray(grid["ys"], float)
    z = np.full(H * W, np.nan)
    year = np.full(H * W, -1, np.int16)
    sheet = np.full(H * W, -1, np.int16)
    iy = np.full(H * W, -1, np.int32)
    ix = np.full(H * W, -1, np.int32)
    for r0 in range(0, H, int(block_rows)):
        r1 = min(H, r0 + int(block_rows))
        rows = np.repeat(np.arange(r0, r1), W)
        cols = np.tile(np.arange(W), r1 - r0)
        xq, yq = to_rd.transform(xs[cols], ys[rows])
        zz, yy, ss, ii, jj = sample_cells(cells, xq, yq)
        sl = slice(r0 * W, r1 * W)
        z[sl], year[sl], sheet[sl], iy[sl], ix[sl] = zz, yy, ss, ii, jj
    return {"z_raw": z.reshape(H, W), "year": year.reshape(H, W),
            "sheet": sheet.reshape(H, W), "iy": iy.reshape(H, W), "ix": ix.reshape(H, W),
            "cell_id": cell_ids(sheet, iy, ix).reshape(H, W), "sheets": list(cells)}


def check_rtk_grid(grid, shape):
    """The RTK/LiDAR grid (``rtk.DEFAULT_GRID``) must be the cube grid, as
    v8 asserts before it maps the survey."""
    import rasterio

    from pyintertidal import rtk

    with rasterio.open(rtk.DEFAULT_GRID) as s:
        ok = tuple(s.shape) == tuple(shape) and s.transform.almost_equals(grid["transform"])
    if not ok:
        raise AssertionError(f"the cube grid differs from {rtk.DEFAULT_GRID}")


def lidar_frame(pkg):
    """PNOA LiDAR truth on the frame: ``v10.lidar_truth(rtk.DEFAULT_GRID)``
    (ground pixels only, about 1.2-1.5 GB peak)."""
    from experiments import v10_validation_report as v10
    from pyintertidal import rtk

    H, W = judges.frame_shape(pkg)
    check_rtk_grid(judges.grid_of(pkg), (H, W))
    if not os.path.exists(v10.MDT02):
        raise FileNotFoundError(f"{v10.MDT02} is missing: copy data_v4/truth/pnoa/ from the "
                                "laptop before the Villaviciosa truth (spec §7.5 step 0)")
    truth, _, gshape, water_share = v10.lidar_truth(rtk.DEFAULT_GRID)
    if tuple(gshape) != (H, W):
        raise AssertionError(f"LiDAR truth shape {gshape} differs from the frame {(H, W)}")
    return {"z_raw": np.asarray(truth, np.float64), "water_share_2m": float(water_share),
            "sources": [v10.MDT02, rtk.DEFAULT_GRID]}


def rtk_points(pkg):
    """RTK development points on the frame (``rtk.load_rtk()`` defaults).

    Returns
    -------
    dict
        flat (frame flat index, int64), row, col, z (elev_alicante),
        z_ellipsoidal (elev), block150 (the survey's 150 m blocks),
        n_outside (points off the frame, expected 0).
    """
    from pyintertidal import rtk

    H, W = judges.frame_shape(pkg)
    check_rtk_grid(judges.grid_of(pkg), (H, W))
    dev = rtk.load_rtk()
    if "elev_alicante" not in dev:
        raise FileNotFoundError(f"{rtk.DEFAULT_ALICANTE_CSV} is missing: the RTK truth is "
                                "scored in the Alicante datum")
    return map_points(dev["row"], dev["col"], dev["elev_alicante"], dev["elev"],
                      dev["block"], (H, W))


def map_points(row, col, z, z_ellipsoidal, block, shape):
    """Frame flat indices of surveyed points (row * W + col); points off the
    frame are dropped and counted."""
    H, W = int(shape[0]), int(shape[1])
    row = np.asarray(row, np.int64)
    col = np.asarray(col, np.int64)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    return {"flat": row[inside] * W + col[inside], "row": row[inside], "col": col[inside],
            "z": np.asarray(z, np.float64)[inside],
            "z_ellipsoidal": np.asarray(z_ellipsoidal, np.float64)[inside],
            "block150": np.asarray(block, np.int64)[inside],
            "n_outside": int((~inside).sum())}


def planted_frame(pkg):
    """The positive control's planted elevation (``planted.npz["z"]``, (H, W))."""
    path = pkg.cfg.extra.get("planted") if getattr(pkg.cfg, "extra", None) else None
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"planted truth {path!r} is missing (control.synth writes it)")
    with np.load(path, allow_pickle=False) as z:
        zp = np.asarray(z["z"], np.float64)
    H, W = judges.frame_shape(pkg)
    if zp.shape != (H, W):
        raise ValueError(f"planted z has shape {zp.shape}, the frame {(H, W)}")
    return {"z_raw": zp, "sources": [path]}


def record_range(pkg):
    """(LO, HI): min and max of the boundary level over the full record."""
    h0 = np.asarray(pkg.h0, float)
    if not np.all(np.isfinite(h0)):
        raise ValueError("the record's h0 must be finite")
    return float(h0.min()), float(h0.max())


def range_filter(pkg, z_raw, pad=None):
    """The method-independent truth range filter (D11).

    Returns
    -------
    truth : array like ``z_raw``
        ``z_raw`` where LO - pad <= z_raw <= HI + pad, NaN elsewhere.
    counts : dict
        LO, HI, lo, hi, pad_m, n_finite, n_below, n_above, n_in_range.
    """
    pad = RANGE_PAD_M if pad is None else float(pad)
    LO, HI = record_range(pkg)
    lo, hi = LO - pad, HI + pad
    z = np.asarray(z_raw, np.float64)
    fin = np.isfinite(z)
    with np.errstate(invalid="ignore"):
        below = fin & (z < lo)
        above = fin & (z > hi)
    inr = fin & ~below & ~above
    return np.where(inr, z, np.nan), {
        "LO": LO, "HI": HI, "lo": lo, "hi": hi, "pad_m": pad,
        "n_finite": int(fin.sum()), "n_below": int(below.sum()),
        "n_above": int(above.sum()), "n_in_range": int(inr.sum())}


def strata_bands(pkg, flat):
    """The stratum band (§2.6) of frame pixels: the package's band_frame
    (merged band of any pixel with a finite s; -1 = "no s"). Without it,
    keep_full pixels get their band and every other pixel -1 (logged)."""
    flat = np.asarray(flat, np.int64)
    bf = getattr(pkg, "band_frame", None)
    if bf is None and getattr(pkg, "full", None) is not None:
        bf = getattr(pkg.full, "band_frame", None)
    if bf is None:
        d = getattr(pkg, "dir", None)
        p = os.path.join(d, "pixels.npz") if d else None
        if p and os.path.exists(p):
            with np.load(p, allow_pickle=False) as z:
                if "band_frame" in z.files:
                    bf = z["band_frame"]
    if bf is not None:
        return np.asarray(bf).ravel()[flat].astype(np.int16)
    log("[truth] WARNING: the package has no band_frame; strata outside keep_full are 'no s'")
    keep = np.asarray(pkg.full.keep, np.int64)
    pos, found = lookup(keep, flat)
    out = np.full(len(flat), -1, np.int16)
    out[found] = np.asarray(pkg.full.band_of)[pos[found]]
    return out


def support(d_truth, keep, nbhds=(), rtk_flat=None):
    """S = D_truth ∪ keep_full ∪ every neighbourhood ∪ RTK pixels (ascending
    unique frame flat indices, int64)."""
    parts = [np.asarray(d_truth, np.int64).ravel(), np.asarray(keep, np.int64).ravel()]
    parts += [np.asarray(nb, np.int64).ravel() for nb in nbhds]
    if rtk_flat is not None:
        parts.append(np.asarray(rtk_flat, np.int64).ravel())
    return np.unique(np.concatenate(parts)).astype(np.int64)


def _elev_sources(cfg):
    """The source files of a site's elevation truth (paths)."""
    kinds = tuple(cfg.elevation_truth)
    paths = []
    if "vaklodingen" in kinds:
        from experiments import v1_vaklodingen_validate as v1
        from experiments.v0_vaklodingen_fetch import sheets_for

        paths.append("data_v4/truth/vaklodingen_catalog.nc")
        paths += [os.path.join(v1.TRUTH_DIR, name).replace("\\", "/")
                  for name, _, _ in sheets_for(cfg.site)]
    if "lidar" in kinds or "rtk" in kinds:
        from pyintertidal import rtk
        paths.append(rtk.DEFAULT_GRID)
    if "lidar" in kinds:
        from experiments import v10_validation_report as v10
        paths.append(v10.MDT02)
    if "rtk" in kinds:
        from pyintertidal import rtk
        paths += [rtk.DEFAULT_CSV, rtk.DEFAULT_ALICANTE_CSV]
    if "planted" in kinds:
        paths.append((cfg.extra or {}).get("planted"))
    return paths


def elev_inputs_key(pkg, table_doc):
    """What the elevation truth depends on (package key, FROZEN sections,
    this code, the judge table, the source files' size and time)."""
    return config.sha256_json({
        "package_key": _package_key(pkg),
        "frozen": config.frozen_shas(("truth_elev", "judges")),
        "code": _code_shas(),
        "judges_inputs": table_doc.get("inputs"),
        "sources": [_file_id(p) for p in _elev_sources(pkg.cfg)],
        "format": TRUTH_ELEV_FORMAT})


def build_elevation_truth(pkg, table=None, force=False):
    """truth/<site>/truth_elev.npz + .json and support.npy (§5.1).

    Parameters
    ----------
    pkg : CommonPackage
        Reads ``cfg``, ``h0``, ``full.keep``, the frame shape, the grid and
        the strata bands (:func:`strata_bands`).
    table : list of dict, optional
        The judge table (neighbourhoods); default the one ``truth`` wrote.
    force : bool
        Rebuild even when the stored inputs key matches.

    Returns
    -------
    str
        The path of truth_elev.json.
    """
    t0 = time.time()
    cfg = pkg.cfg
    out = truth_dir(pkg)
    jpath = f"{out}/truth_elev.json"
    jt_path = f"{out}/judges.json"
    if table is None:
        table = judges.load(jt_path, judges.grid_of(pkg))
    table_doc = judges.read(jt_path) if os.path.exists(jt_path) else {}
    key = elev_inputs_key(pkg, table_doc)
    if not force and _done(jpath, key) and os.path.exists(f"{out}/support.npy"):
        log(f"[truth] {cfg.site}: elevation truth up to date ({jpath})")
        return jpath
    H, W = judges.frame_shape(pkg)
    kinds = tuple(cfg.elevation_truth)
    grid_kind = next((k for k in ("vaklodingen", "lidar", "planted") if k in kinds), None)
    keep = np.asarray(pkg.full.keep, np.int64)
    log(f"[truth] {cfg.site} {cfg.res_m} m: elevation truth {kinds or '(none)'} "
        f"on {H} x {W} px, keep_full {len(keep):,}")

    fr = None
    if grid_kind == "vaklodingen":
        fr = vaklodingen_frame(pkg)
    elif grid_kind == "lidar":
        fr = lidar_frame(pkg)
    elif grid_kind == "planted":
        fr = planted_frame(pkg)
    if fr is not None:
        truth_frame, counts = range_filter(pkg, fr["z_raw"])
        d_truth = np.flatnonzero(np.isfinite(truth_frame.ravel())).astype(np.int64)
    else:
        LO, HI = record_range(pkg)
        counts = {"LO": LO, "HI": HI, "lo": LO - RANGE_PAD_M, "hi": HI + RANGE_PAD_M,
                  "pad_m": RANGE_PAD_M, "n_finite": 0, "n_below": 0, "n_above": 0,
                  "n_in_range": 0}
        truth_frame = None
        d_truth = np.empty(0, np.int64)
    rtk = rtk_points(pkg) if "rtk" in kinds else None
    nbhds = [r["nbhd"] for r in table]
    S = support(d_truth, keep, nbhds, None if rtk is None else rtk["flat"])
    nS = len(S)

    arrays = {"format": np.array(TRUTH_ELEV_FORMAT), "support": S,
              "shape": np.array([H, W], np.int64),
              "grid_kind": np.array(grid_kind or ""),
              "LO": np.float64(counts["LO"]), "HI": np.float64(counts["HI"]),
              "lo": np.float64(counts["lo"]), "hi": np.float64(counts["hi"])}
    _, in_keep = lookup(keep, S)
    if fr is not None:
        tr_S = truth_frame.ravel()[S]
        arrays["truth_raw"] = fr["z_raw"].ravel()[S].astype(np.float64)
        arrays["truth"] = tr_S.astype(np.float64)
    else:
        tr_S = np.full(nS, np.nan)
        arrays["truth_raw"] = np.full(nS, np.nan)
        arrays["truth"] = tr_S
    in_D = np.isfinite(tr_S)
    arrays["in_D_truth"] = in_D
    arrays["in_keep"] = in_keep
    arrays["in_D_keep"] = in_D & in_keep
    if grid_kind == "vaklodingen":
        arrays["year"] = fr["year"].ravel()[S].astype(np.int16)
        arrays["cell_id"] = fr["cell_id"].ravel()[S].astype(np.int64)
        arrays["cell_sheet"] = fr["sheet"].ravel()[S].astype(np.int16)
        arrays["cell_iy"] = fr["iy"].ravel()[S].astype(np.int32)
        arrays["cell_ix"] = fr["ix"].ravel()[S].astype(np.int32)
        arrays["sheets"] = np.array(fr["sheets"], dtype=str)
        # the 20 m cells of D_truth: id, truth (the cell value), year, and how
        # many D_truth pixels each one holds (the cells view's half-mapped rule)
        cid = arrays["cell_id"][in_D]
        uc, first, cnt = np.unique(cid, return_index=True, return_counts=True)
        arrays["cells_id"] = uc.astype(np.int64)
        arrays["cells_truth"] = tr_S[in_D][first]
        arrays["cells_year"] = arrays["year"][in_D][first]
        arrays["cells_n_dtruth"] = cnt.astype(np.int64)
    else:
        arrays["year"] = np.full(nS, -1, np.int16)
        arrays["cell_id"] = np.full(nS, -1, np.int64)
    arrays["band"] = strata_bands(pkg, S)
    if rtk is not None:
        pos, found = lookup(S, rtk["flat"])
        if not found.all():
            raise AssertionError("RTK pixels missing from the support")
        _, rk = lookup(keep, rtk["flat"])
        arrays.update({"rtk_flat": rtk["flat"], "rtk_pos": pos.astype(np.int64),
                       "rtk_row": rtk["row"], "rtk_col": rtk["col"], "rtk_z": rtk["z"],
                       "rtk_z_ellipsoidal": rtk["z_ellipsoidal"],
                       "rtk_block150": rtk["block150"], "rtk_in_keep": rk})

    os.makedirs(out, exist_ok=True)
    product.save_npz(f"{out}/truth_elev.npz", arrays, compress=True)
    _save_npy(f"{out}/support.npy", S)
    yrs = {}
    if grid_kind == "vaklodingen":
        u, c = np.unique(arrays["year"][in_D], return_counts=True)
        yrs = {int(a): int(b) for a, b in zip(u, c)}
    nb_union = (np.unique(np.concatenate(nbhds)) if nbhds and any(len(n) for n in nbhds)
                else np.empty(0, np.int64))
    doc = {
        "format": TRUTH_ELEV_FORMAT, "inputs_key": key,
        "site": cfg.site, "res_m": int(cfg.res_m), "scenario": getattr(cfg, "scenario", None),
        "kinds": list(kinds), "grid_kind": grid_kind, "shape": [H, W],
        "range": counts,
        "n_S": int(nS), "n_D_truth": int(in_D.sum()), "n_D_keep": int((in_D & in_keep).sum()),
        "n_keep_full": int(len(keep)), "n_nbhd_union": int(len(nb_union)),
        "n_support_outside_D_truth_and_keep": int((~in_D & ~in_keep).sum()),
        "years_D_truth": yrs,
        "n_cells_D_truth": int(len(arrays["cells_id"])) if "cells_id" in arrays else None,
        "rtk": None if rtk is None else {
            "n_points": int(len(rtk["flat"])), "n_pixels": int(len(np.unique(rtk["flat"]))),
            "n_points_in_keep": int(arrays["rtk_in_keep"].sum()),
            "n_points_outside_frame": rtk["n_outside"], "datum": "Alicante (elev_alicante)",
            "range_filtered": False},
        "lidar_water_share_2m": fr.get("water_share_2m") if fr else None,
        "support_sha256": config.sha256_bytes(S.astype("<i8").tobytes()),
        "sources": [_file_entry(p) for p in _elev_sources(cfg)],
        "package_key": _package_key(pkg),
        "frozen": config.frozen_shas(("truth_elev", "judges")),
        "code": _code_shas(),
        "judges_table": jt_path,
        "built_utc": product.utc_stamp(), "seconds": round(time.time() - t0, 1),
    }
    product.write_json(jpath, doc)
    log(f"[truth] {cfg.site}: D_truth {doc['n_D_truth']:,} px (range [{counts['lo']:.3f}, "
        f"{counts['hi']:.3f}] m; {counts['n_below']:,} below, {counts['n_above']:,} above), "
        f"D_keep {doc['n_D_keep']:,}, support {nS:,} px"
        + (f", RTK {doc['rtk']['n_points']} points" if rtk is not None else "")
        + f" [{doc['seconds']} s]")
    return jpath


def load_support(pkg_or_cfg):
    """The support S of a site (``truth/<site>/support.npy``)."""
    cfg = getattr(pkg_or_cfg, "cfg", pkg_or_cfg)
    return np.load(f"{cfg.truth_dir}/support.npy", allow_pickle=False)


def load_elevation_truth(pkg_or_cfg):
    """``(arrays, doc)`` of a site's elevation truth."""
    cfg = getattr(pkg_or_cfg, "cfg", pkg_or_cfg)
    arrays = product.load_npz(f"{cfg.truth_dir}/truth_elev.npz")
    if str(arrays.get("format")) != TRUTH_ELEV_FORMAT:
        raise ValueError(f"{cfg.truth_dir}/truth_elev.npz: not a {TRUTH_ELEV_FORMAT} file")
    return arrays, product.read_json(f"{cfg.truth_dir}/truth_elev.json")


# ─────────────────────────────────────────────────────────────────────────────
#  lag truth (a) and tau_ab (§5.5)
# ─────────────────────────────────────────────────────────────────────────────

def interp_at(series, t, max_gap_min=GAP_INTERP_MIN):
    """The gauge linearly interpolated at ``t`` between its two bracketing
    samples when they are at most ``max_gap_min`` apart (an exact sample is
    taken as is); NaN elsewhere."""
    x = series.index.asi8
    v = series.to_numpy(np.float64)
    q = pd.DatetimeIndex(t).as_unit("ns").asi8
    n = len(x)
    out = np.full(len(q), np.nan)
    if not n:
        return out
    j = np.searchsorted(x, q, side="left")
    exact = (j < n) & (x[np.minimum(j, n - 1)] == q)
    out[exact] = v[j[exact]]
    inner = ~exact & (j > 0) & (j < n)
    jl, jr = j[inner] - 1, j[inner]
    gap = (x[jr] - x[jl]).astype(np.float64)
    w = (q[inner] - x[jl]).astype(np.float64) / gap
    val = v[jl] + (v[jr] - v[jl]) * w
    out[inner] = np.where(gap <= max_gap_min * judges.NS_PER_MIN, val, np.nan)
    return out


def e_rows(H, g):
    """E for each row of ``H`` (n_tau, T): sqrt(mean (d - median d)^2) with
    d = H[k] - g over the instants where g is finite."""
    H = np.asarray(H, np.float64)
    g = np.asarray(g, np.float64)
    m = np.isfinite(g)
    if not m.any():
        return np.full(H.shape[0], np.nan)
    d = H[:, m] - g[m]
    med = np.median(d, axis=1, keepdims=True)
    return np.sqrt(np.mean((d - med) ** 2, axis=1))


def E_curve(LEV, g_at_instants, taus=TAUS_A):
    """E(tau) of a gauge at the record instants.

    Parameters
    ----------
    LEV : levels.LevelTable or (n_tau, T) array
        The exact boundary levels; a table is read at ``taus``.
    g_at_instants : (T,) float
        The gauge (or a model) at the record instants, NaN where absent.
    taus : int array
        Lags (min).
    """
    H = LEV.stack(taus) if hasattr(LEV, "stack") else np.asarray(LEV, np.float64)
    return e_rows(H, g_at_instants)


def best_lag(taus, E, tol=None):
    """argmin of an error curve and its tolerance interval.

    Returns
    -------
    dict
        tau, lo, hi ([min, max] of the taus with E <= E_min + tol), Emin_m,
        edge (minimum on the grid edge), interval_edge, contiguous.
    """
    tol = INTERVAL_M if tol is None else float(tol)
    taus = np.asarray(taus, float)
    E = np.asarray(E, float)
    nan = float("nan")
    if not np.isfinite(E).any():
        return {"tau": nan, "lo": nan, "hi": nan, "Emin_m": nan, "edge": False,
                "interval_edge": False, "contiguous": False}
    k = int(np.nanargmin(E))
    with np.errstate(invalid="ignore"):
        within = np.flatnonzero(E <= E[k] + tol)
    return {"tau": float(taus[k]), "lo": float(taus[within.min()]),
            "hi": float(taus[within.max()]), "Emin_m": float(E[k]),
            "edge": bool(k in (0, len(taus) - 1)),
            "interval_edge": bool(within.min() == 0 or within.max() == len(taus) - 1),
            "contiguous": bool(np.all(np.diff(within) == 1))}


def _rms_med(r):
    return float(np.sqrt(np.mean((r - np.median(r)) ** 2)))


def gain_aware(H, g, gains=None):
    """E_ab(tau) = min over b of the median-centred RMS of g - b H[tau].

    The gain is searched on ``gains`` (default 0..4 by 0.01, which holds
    b = 1, so E_ab <= E everywhere) and refined by a bounded Brent search
    between the neighbours of the best grid gain.

    Returns
    -------
    E_ab (n_tau,), gain (n_tau,)
    """
    from scipy.optimize import minimize_scalar

    gains = GAIN_GRID if gains is None else np.asarray(gains, float)
    H = np.asarray(H, np.float64)
    g = np.asarray(g, np.float64)
    m = np.isfinite(g)
    E = np.full(H.shape[0], np.nan)
    B = np.full(H.shape[0], np.nan)
    if m.sum() < 3:
        return E, B
    G = g[m]
    for k in range(H.shape[0]):
        h = H[k, m]
        R = G[None, :] - gains[:, None] * h[None, :]
        med = np.median(R, axis=1, keepdims=True)
        e = np.sqrt(np.mean((R - med) ** 2, axis=1))
        j = int(np.argmin(e))
        lo, hi = gains[max(j - 1, 0)], gains[min(j + 1, len(gains) - 1)]
        best_e, best_b = float(e[j]), float(gains[j])
        if hi > lo:
            res = minimize_scalar(lambda b: _rms_med(G - b * h), bounds=(lo, hi),
                                  method="bounded", options={"xatol": 1e-7})
            if float(res.fun) < best_e:
                best_e, best_b = float(res.fun), float(res.x)
        E[k], B[k] = best_e, best_b
    return E, B


def month_boot(H, g, t, taus=TAUS_A, n=N_BOOT, rng=None):
    """tau_a of ``n`` calendar-month block bootstraps of the record instants
    (months drawn with replacement; the instants of a drawn month enter as a
    block, repeated as often as it is drawn)."""
    g = np.asarray(g, np.float64)
    m = np.isfinite(g)
    out = np.full(int(n), np.nan)
    if m.sum() < 3:
        return out
    t = pd.DatetimeIndex(t)
    lab = (t.year * 12 + t.month).to_numpy()[m]
    um, inv = np.unique(lab, return_inverse=True)
    groups = [np.flatnonzero(inv == j) for j in range(len(um))]
    Hm, gm = np.asarray(H, np.float64)[:, m], g[m]
    rng = np.random.default_rng(rng)
    for b in range(int(n)):
        draw = rng.integers(0, len(um), len(um))
        sel = np.concatenate([groups[j] for j in draw])
        E = e_rows(Hm[:, sel], gm[sel])
        out[b] = float(taus[int(np.nanargmin(E))]) if np.isfinite(E).any() else np.nan
    return out


def boot_rng(site, code):
    """The bootstrap generator of one judge: [seed, SITE_ID, crc32(code)]."""
    return np.random.default_rng([BOOT_SEED, int(config.SITE_ID[site]),
                                  int(zlib.crc32(str(code).encode("utf-8")))])


def truth_s2(pkg, station, g, H=None, boot=True):
    """Truth (a) and tau_ab of one gauge at the record's S2 instants.

    Parameters
    ----------
    pkg : CommonPackage
        ``t`` and ``levels`` (exact h_B(t_i - tau)).
    station : dict
        ``code`` (bootstrap seed).
    g : pd.Series
        The gauge as delivered (:func:`judges.read_series`).
    H : (len(TAUS_A), T) array, optional
        ``pkg.levels.stack(TAUS_A)``, passed to share it between judges.
    boot : bool
        Run the month-block bootstrap.

    Returns
    -------
    dict
        ``a`` (n_instants, coverage, tau, lo, hi, E0_m, Emin_m, edges, boot,
        interp), ``ab`` (tau, lo, hi, gain, Emin_m, edge, gain_edge) and
        ``curves`` (E_a, E_ab, gain_ab, E_interp, boot_tau).
    """
    t = record_t(pkg)
    if H is None:
        H = pkg.levels.stack(TAUS_A)
    g_s2 = judges.at_instants(g, t, TOL_MIN)
    E = e_rows(H, g_s2)
    a = best_lag(TAUS_A, E)
    k0 = int(np.flatnonzero(TAUS_A == 0)[0])
    a.update(n_instants=int(np.isfinite(g_s2).sum()),
             coverage=float(np.isfinite(g_s2).mean()) if len(t) else 0.0,
             E0_m=float(E[k0]))
    curves = {"E_a": E}
    if boot:
        bt = month_boot(H, g_s2, t, rng=boot_rng(pkg.cfg.site, station["code"]))
        fin = bt[np.isfinite(bt)]
        a["boot"] = {"n": int(len(fin)), "seed": [BOOT_SEED, int(config.SITE_ID[pkg.cfg.site]),
                                                  int(zlib.crc32(str(station["code"]).encode()))],
                     "ci_lo": float(np.percentile(fin, 2.5)) if len(fin) else float("nan"),
                     "ci_hi": float(np.percentile(fin, 97.5)) if len(fin) else float("nan"),
                     "sd": float(np.std(fin)) if len(fin) else float("nan")}
        curves["boot_tau"] = bt
    gi = interp_at(g, t)
    Ei = e_rows(H, gi)
    ai = best_lag(TAUS_A, Ei)
    ai["n_instants"] = int(np.isfinite(gi).sum())
    a["interp"] = ai
    curves["E_interp"] = Ei
    Eab, B = gain_aware(H, g_s2)
    ab = best_lag(TAUS_A, Eab)
    kb = int(np.nanargmin(Eab)) if np.isfinite(Eab).any() else None
    ab["gain"] = float(B[kb]) if kb is not None else float("nan")
    ab["gain_edge"] = bool(kb is not None and (B[kb] <= GAIN_GRID[0] or B[kb] >= GAIN_GRID[-1]))
    curves.update(E_ab=Eab, gain_ab=B)
    return {"g_s2": g_s2, "a": a, "ab": ab, "curves": curves}


# ─────────────────────────────────────────────────────────────────────────────
#  lag truth (b): the M2 harmonic phase lag
# ─────────────────────────────────────────────────────────────────────────────

def harmonic_fit(t_days, y, constituents=None, order=None, corrections=None):
    """Amplitudes and Greenwich phase lags (deg) by ``pyTMD.solve.constants``.

    The series is centred by its mean first (the polynomial absorbs it
    anyway). Returns None when there are too few finite samples for the
    design (pyTMD needs more than 2 per constituent).

    Returns
    -------
    dict ``{name: (amp, phase_deg)}`` or None
    """
    import pyTMD.solve

    cons = list(CONS_B if constituents is None else constituents)
    order = B_ORDER if order is None else int(order)
    corrections = B_CORRECTIONS if corrections is None else corrections
    t = np.asarray(t_days, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) <= 2 * len(cons) + order + 1:
        return None
    amp, ph = pyTMD.solve.constants(t, y - np.mean(y), cons, corrections=corrections,
                                    order=order)
    return {c: (float(a), float(p)) for c, a, p in zip(cons, amp, ph)}


def _quarters(idx):
    """Calendar-quarter labels of instants, 0 = the first quarter of PERIOD."""
    y0 = int(str(config.PERIOD[0])[:4])
    idx = pd.DatetimeIndex(idx)
    return ((idx.year - y0) * 4 + (idx.month - 1) // 3).to_numpy()


def truth_m2(pkg, station, g10, constituents=None):
    """Truth (b): the M2 phase lag of the gauge relative to h_B.

    Parameters
    ----------
    pkg : CommonPackage
        ``series10`` (the 10-min h_B).
    station : dict
    g10 : array or Series on the package's 10-min grid
        :func:`judges.centred_10min` of the gauge.
    constituents : sequence, optional
        Default :data:`CONS_B`.

    Returns
    -------
    dict
        tau, se, lo, hi (I_b), n_samples, span_days, n_quarters, jackknife
        (leave-one-quarter-out taus), phases, A_M2 / M4 of both series,
        flags; ``instants`` (the boolean mask of the fit, not saved).
    """
    cons = list(CONS_B if constituents is None else constituents)
    idx10, h10 = series10(pkg)
    g = np.asarray(g10, np.float64)
    if len(g) != len(idx10):
        raise ValueError("g10 is not on the package's 10-min grid")
    m = np.isfinite(g) & np.isfinite(h10)
    nan = float("nan")
    out = {"n_samples": int(m.sum()), "constituents": len(cons), "order": B_ORDER,
           "corrections": B_CORRECTIONS, "tau": nan, "se": nan, "lo": nan, "hi": nan,
           "flags": [], "instants": m}
    if not m.any():
        out["flags"].append("no_samples")
        return out
    td = days_since_1992(idx10[m])
    out["span_days"] = float(td.max() - td.min())
    if out["span_days"] < 365.0:
        out["flags"].append("short_record")
    fg = harmonic_fit(td, g[m], cons)
    fb = harmonic_fit(td, h10[m], cons)
    if fg is None or fb is None:
        out["flags"].append("too_few_samples")
        return out
    dphi = float(wrap180(fg["m2"][1] - fb["m2"][1]))
    tau = dphi * MIN_PER_DEG_M2
    out.update(tau=tau, phase_gauge_deg=fg["m2"][1], phase_B_deg=fb["m2"][1],
               A_M2_gauge=fg["m2"][0], A_M2_B=fb["m2"][0])
    if "m4" in fg:
        out.update(A_M4_gauge=fg["m4"][0], A_M4_B=fb["m4"][0],
                   M4_M2_gauge=fg["m4"][0] / fg["m2"][0] if fg["m2"][0] else nan,
                   M4_M2_B=fb["m4"][0] / fb["m2"][0] if fb["m2"][0] else nan,
                   asym_gauge_deg=float(wrap180(2 * fg["m2"][1] - fg["m4"][1])),
                   asym_B_deg=float(wrap180(2 * fb["m2"][1] - fb["m4"][1])))
    # delete-one-quarter jackknife: the same fit without each calendar quarter
    q = _quarters(idx10[m])
    groups = np.unique(q)
    jack = []
    for gq in groups:
        keep = q != gq
        a_ = harmonic_fit(td[keep], g[m][keep], cons)
        b_ = harmonic_fit(td[keep], h10[m][keep], cons)
        if a_ is None or b_ is None:
            jack.append(nan)
            continue
        d = float(wrap180(wrap180(a_["m2"][1] - b_["m2"][1]) - dphi))
        jack.append(tau + d * MIN_PER_DEG_M2)
    jack = np.asarray(jack, float)
    fin = jack[np.isfinite(jack)]
    G = len(fin)
    out["n_quarters"] = int(len(groups))
    out["jackknife"] = jack
    if G >= 2:
        se = float(np.sqrt((G - 1) / G * np.sum((fin - fin.mean()) ** 2)))
        out.update(se=se, lo=tau - Z_B * se, hi=tau + Z_B * se)
    else:
        out["flags"].append("no_jackknife")
    return out


def tau_all(pkg, g10, taus=TAUS_A):
    """E over every 10-min sample of the record period, h_B(t - tau) read
    from the exact 1-min series (t on the 10-min grid, tau whole minutes).

    Returns
    -------
    (dict, E)
        best_lag of the curve plus E0_m and n_samples (at tau = 0).
    """
    idx10, _ = series10(pkg)
    t1, h1 = series1(pkg)
    m10 = levels.minutes_since_epoch(idx10)
    p = np.searchsorted(t1, m10)
    if (p >= len(t1)).any() or not np.array_equal(t1[p], m10):
        raise ValueError("the 10-min grid is not on the 1-min boundary series")
    g = np.asarray(g10, np.float64)
    fg = np.isfinite(g)
    E = np.full(len(taus), np.nan)
    n = np.zeros(len(taus), np.int64)
    for k, tau in enumerate(np.asarray(taus, np.int64)):
        q = p - int(tau)
        valid = (q >= 0) & (q < len(h1))
        hb = np.full(len(g), np.nan)
        hb[valid] = h1[q[valid]]
        mm = fg & np.isfinite(hb)
        if mm.any():
            d = hb[mm] - g[mm]
            E[k] = np.sqrt(np.mean((d - np.median(d)) ** 2))
        n[k] = int(mm.sum())
    res = best_lag(taus, E)
    k0 = int(np.flatnonzero(np.asarray(taus) == 0)[0])
    res.update(E0_m=float(E[k0]), n_samples=int(n[k0]))
    return res, E


def _pair_marks(tg, tb, max_min=MARK_MATCH_MIN):
    """tg - nearest tb, for the tg with a tb within ``max_min`` minutes."""
    tg, tb = np.asarray(tg, float), np.asarray(tb, float)
    if not len(tg) or not len(tb):
        return np.empty(0)
    j = np.searchsorted(tb, tg)
    jl, jr = np.clip(j - 1, 0, len(tb) - 1), np.clip(j, 0, len(tb) - 1)
    near = np.where(np.abs(tg - tb[jl]) <= np.abs(tb[jr] - tg), jl, jr)
    d = tg - tb[near]
    return d[np.abs(d) <= max_min]


def hw_lw_lags(pkg, g10):
    """HW-only and LW-only lags: ``g21.tide_marks`` of the centred 10-min
    gauge against the boundary's low-passed 1-min marks (marks_lp), each
    gauge mark paired with the nearest boundary mark of its kind within 3 h.

    Returns
    -------
    dict
        ``{"hw": {median_min, mean_min, n}, "lw": {...}}``.
    """
    from pyintertidal import granadeiro2021 as g21

    idx10, _ = series10(pkg)
    g = np.asarray(g10, np.float64)
    fin = np.isfinite(g)
    nan = float("nan")
    empty = {"hw": {"median_min": nan, "mean_min": nan, "n": 0},
             "lw": {"median_min": nan, "mean_min": nan, "n": 0}}
    if fin.sum() < 3:
        return empty
    try:
        mg = g21.tide_marks(levels.minutes_since_epoch(idx10)[fin], g[fin])
    except ValueError:
        return empty
    mb = marks_lp(pkg)
    tb_all, hw_b = np.asarray(mb["t"], float), np.asarray(mb["is_hw"], bool)
    tg_all, hw_g = np.asarray(mg["t"], float), np.asarray(mg["is_hw"], bool)
    out = {}
    for name, flag in (("hw", True), ("lw", False)):
        d = _pair_marks(tg_all[hw_g == flag], tb_all[hw_b == flag])
        out[name] = {"median_min": float(np.median(d)) if len(d) else nan,
                     "mean_min": float(np.mean(d)) if len(d) else nan, "n": int(len(d))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  (c): the frame truths of GM20 and HD26 (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def fold_phase(x_hours, y, period_h=GM20_PERIOD_H):
    """GM20's sinusoid on hours since HW: y ~ c + a cos(2 pi x / P) +
    b sin(2 pi x / P) by least squares.

    Returns
    -------
    dict
        phase_min = 60 P atan2(b, a) / (2 pi) (a crest x0 hours after HW
        gives +60 x0), amp, r2, n.
    """
    x = np.asarray(x_hours, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    nan = float("nan")
    if len(y) < 4:
        return {"phase_min": nan, "amp": nan, "r2": nan, "n": int(len(y))}
    w = 2 * np.pi * x / float(period_h)
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"phase_min": float(60.0 * period_h * np.arctan2(beta[2], beta[1]) / (2 * np.pi)),
            "amp": float(np.hypot(beta[1], beta[2])),
            "r2": float(1.0 - np.sum(res ** 2) / sst) if sst > 0 else nan, "n": int(len(y))}


def hd26_m2_phase(t_days, y):
    """HD26's M2 phase of a series (FROZEN['hd26']): linear detrend, one
    3-sigma pass, at least 30 points over at least 365 days, then
    ``pyTMD.solve.constants`` with M2, N2, K2, O1, Q1 and GOT corrections.

    Returns
    -------
    dict
        phase_deg (NaN when the series is too short), amp, n, span_days.
    """
    import pyTMD.solve

    p = config.FROZEN["hd26"]
    cons = list(p["constituents"])
    t = np.asarray(t_days, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    nan = float("nan")
    if len(y) < 3:
        return {"phase_deg": nan, "amp": nan, "n": int(len(y)), "span_days": nan}
    r = y - np.polyval(np.polyfit(t, y, 1), t)
    keep = np.abs(r - r.mean()) <= float(p["outlier_sigma"]) * r.std()
    t, r = t[keep], r[keep]
    span = float(t.max() - t.min()) if len(t) else nan
    if len(r) < int(p["min_points"]) or not span >= float(p["min_span_days"]) \
            or len(r) <= 2 * len(cons):
        return {"phase_deg": nan, "amp": nan, "n": int(len(r)), "span_days": span}
    amp, ph = pyTMD.solve.constants(t, r - r.mean(), cons, corrections=p["corrections"])
    i = cons.index("m2")
    return {"phase_deg": float(ph[i]), "amp": float(amp[i]), "n": int(len(r)),
            "span_days": span}


def frame_truths(pkg, g_s2):
    """Truth (c), diagnostic: what GM20 and HD26 would read with a perfect
    level proxy at the record instants.

    * GM20 frame: the fold phase of the gauge on hours since the last HW
      mark of marks_lp (``gm20_raw``) and minus the fold phase of h0 on the
      same instants (``gm20_rel``), wrapped to [-120, 625.24) min;
    * HD26 oracle: HD26's M2 phase of the gauge minus that of h0 on the same
      instants, wrapped to +-180 deg, in minutes.
    """
    t = record_t(pkg)
    h0 = np.asarray(pkg.h0, float)
    g = np.asarray(g_s2, float)
    mk = marks_lp(pkg)
    HW = np.asarray(mk["t"], float)[np.asarray(mk["is_hw"], bool)]
    tm = levels.minutes_since_epoch(t)
    j = np.searchsorted(HW, tm, side="right") - 1
    m = np.isfinite(g) & (j >= 0)
    x = (tm[m] - HW[j[m]]) / 60.0
    fg = fold_phase(x, g[m])
    fb = fold_phase(x, h0[m])
    td = days_since_1992(t[m])
    hg = hd26_m2_phase(td, g[m])
    hb = hd26_m2_phase(td, h0[m])
    return {"n": int(m.sum()),
            "gm20_raw": float(wrap_gm20(fg["phase_min"])),
            "gm20_rel": float(wrap_gm20(fg["phase_min"] - fb["phase_min"])),
            "gm20_r2_gauge": fg["r2"], "gm20_r2_h0": fb["r2"],
            "hd26_oracle": float(wrap180(hg["phase_deg"] - hb["phase_deg"]) * MIN_PER_DEG_M2),
            "hd26_n": hg["n"]}


# ─────────────────────────────────────────────────────────────────────────────
#  baselines: global models at the gauge (§3.7)
# ─────────────────────────────────────────────────────────────────────────────

def model_provider(model, lat, lon):
    """A global tide model at a point (``PyTMDBoundary``, nearest ocean cell)."""
    from pyintertidal.boundary import PyTMDBoundary

    return PyTMDBoundary(model, lat, lon, directory=config.TIDE_DIR)


def nearest_ocean_km(lat, lon, model):
    """``tidecheck.nearest_ocean_km`` of a model at a point (km)."""
    from pyintertidal import tidecheck

    return float(tidecheck.nearest_ocean_km(lat, lon, model=model, directory=config.TIDE_DIR,
                                            verbose=False)["distance_km"])


def model_at_gauge(pkg, station, model, H=None, hb_fit=None, ocean=True, constituents=None):
    """A global model at the gauge as a lag "method" (a lower bound).

    The model's levels replace the gauge in the (a), tau_ab and (b)
    machinery: at the record instants for (a) and tau_ab, on the 10-min
    grid (every instant where h_B is finite) for (b). One sorted provider
    call serves both.

    Parameters
    ----------
    hb_fit : dict, optional
        :func:`harmonic_fit` of h_B on every finite 10-min instant (shared
        by the judges of a site).

    Returns
    -------
    dict
        available (False with a reason when the model cannot be loaded),
        a, ab, b (tau, A_M2), ocean_km; ``curves`` (E_a, E_ab).
    """
    t = record_t(pkg)
    idx10, h10 = series10(pkg)
    try:
        prov = model_provider(model, station["lat"], station["lon"])
        u = t.union(idx10)
        hu = np.asarray(prov.levels(u), np.float64)
        if hu.shape != (len(u),):
            raise ValueError(f"{model} returned {hu.shape} levels for {len(u)} instants")
    except Exception as e:                    # an unloadable model is reported, not fatal
        log(f"[truth] {station['code']}: {model} unavailable ({type(e).__name__}: {e})")
        return {"model": model, "available": False, "reason": f"{type(e).__name__}: {e}"}
    m_s2 = hu[u.get_indexer(t)]
    m10 = hu[u.get_indexer(idx10)]
    if H is None:
        H = pkg.levels.stack(TAUS_A)
    E = e_rows(H, m_s2)
    a = best_lag(TAUS_A, E)
    Eab, B = gain_aware(H, m_s2)
    ab = best_lag(TAUS_A, Eab)
    kb = int(np.nanargmin(Eab)) if np.isfinite(Eab).any() else None
    ab["gain"] = float(B[kb]) if kb is not None else float("nan")
    cons = list(CONS_B if constituents is None else constituents)
    mm = np.isfinite(m10) & np.isfinite(h10)
    td = days_since_1992(idx10[mm])
    fm = harmonic_fit(td, m10[mm], cons)
    if hb_fit is None or hb_fit.get("n") != int(mm.sum()):
        hb_fit = {"fit": harmonic_fit(td, h10[mm], cons), "n": int(mm.sum())}
    b = {"tau": float("nan"), "A_M2_model": float("nan"), "n_samples": int(mm.sum())}
    if fm is not None and hb_fit["fit"] is not None:
        b.update(tau=float(wrap180(fm["m2"][1] - hb_fit["fit"]["m2"][1]) * MIN_PER_DEG_M2),
                 A_M2_model=fm["m2"][0])
    out = {"model": model, "available": True, "a": a, "ab": ab, "b": b,
           "curves": {"E_a": E, "E_ab": Eab}}
    if ocean:
        try:
            out["ocean_km"] = nearest_ocean_km(station["lat"], station["lon"], model)
        except Exception as e:                # a diagnostic; its failure is recorded
            out["ocean_km"] = float("nan")
            out["ocean_error"] = f"{type(e).__name__}: {e}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  the lag truth of a site
# ─────────────────────────────────────────────────────────────────────────────

def _model_key(model):
    return "".join(c for c in str(model).lower() if c.isalnum())


def lag_inputs_key(pkg, table_doc, models):
    return config.sha256_json({
        "package_key": _package_key(pkg),
        "frozen": config.frozen_shas(("truth_lag", "judges", "baselines", "gm20", "hd26")),
        "code": _code_shas(), "judges_inputs": table_doc.get("inputs"),
        "models": list(models), "cons_b": list(CONS_B), "format": TRUTH_LAG_FORMAT})


def lag_truth_station(pkg, rec, H, models=MODELS, hb_fit=None, constituents=None):
    """Every lag truth of one judge (``rec``: a judge-table record).

    Returns
    -------
    (dict, curves)
    """
    t0 = time.time()
    series = judges.read_series(rec)
    idx10, _ = series10(pkg)
    s2 = truth_s2(pkg, rec, series, H=H)
    g10 = judges.centred_10min(series, rec["sampling_min"], grid=idx10).to_numpy(np.float64)
    b = truth_m2(pkg, rec, g10, constituents=constituents)
    b.pop("instants", None)
    jack = b.pop("jackknife", np.empty(0))
    al, E_all = tau_all(pkg, g10)
    hwlw = hw_lw_lags(pkg, g10)
    c = frame_truths(pkg, s2["g_s2"])
    curves = dict(s2["curves"], E_all=E_all, jack_tau=jack)
    mods = {}
    for model in models:
        r = model_at_gauge(pkg, rec, model, H=H, hb_fit=hb_fit, constituents=constituents)
        if r.get("available"):
            for name, v in r.pop("curves").items():
                curves[f"{_model_key(model)}__{name}"] = v
        mods[model] = r
    keep_keys = ("code", "name", "provider", "location", "included", "exclusion", "flags",
                 "mouth", "k_g", "k_g_share", "centre_k_km", "s_g_km", "d_min_km", "R_g_m",
                 "n_nbhd", "n_keep_in_nbhd", "in_frame", "coverage", "dry_share",
                 "nearest_member", "nearest_member_km", "sampling_min", "datum", "file")
    out = {k: rec.get(k) for k in keep_keys}
    out.update(a=s2["a"], ab=s2["ab"], b=b, all=al, hw_lw=hwlw, c=c, models=mods,
               seconds=round(time.time() - t0, 1))
    a, ab_ = s2["a"], s2["ab"]
    log(f"[truth] {pkg.cfg.site} {rec['code']}: (a) {a['tau']:.0f} min [{a['lo']:.0f}, "
        f"{a['hi']:.0f}] (n {a['n_instants']}, E {a['Emin_m']:.3f} m, E0 {a['E0_m']:.3f}); "
        f"tau_ab {ab_['tau']:.0f} (b {ab_['gain']:.2f}); (b) {b['tau']:.1f} +- "
        f"{b['se']:.2f} min; all-samples {al['tau']:.0f}; (c) GM20 rel {c['gm20_rel']:.1f}, "
        f"HD26 {c['hd26_oracle']:.1f}; "
        + ", ".join(f"{k} {v['a']['tau']:.0f}/{v['b']['tau']:.1f}" if v.get("available")
                    else f"{k} n/a" for k, v in mods.items())
        + f" [{out['seconds']} s]")
    return out, curves


def planted_lag(pkg):
    """The positive control's planted lag per stratum band (the mean of the
    planted tau over keep_full ∩ band k), when ``planted.npz`` holds ``tau``."""
    path = (pkg.cfg.extra or {}).get("planted") if getattr(pkg.cfg, "extra", None) else None
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if "tau" not in z.files:
            return None
        tau = np.asarray(z["tau"], float).ravel()
    keep = np.asarray(pkg.full.keep, np.int64)
    band = np.asarray(pkg.full.band_of)
    tk = tau[keep]
    bands = sorted(int(b) for b in np.unique(band) if b >= 0)
    return {"bands": bands,
            "tau_mean_min": [float(np.nanmean(tk[band == b])) for b in bands],
            "n_px": [int((band == b).sum()) for b in bands]}


CSV_COLUMNS = ("site", "judge", "location", "included", "exclusion_rule", "flags", "k_g",
               "s_g_km", "d_min_km", "R_g_m", "n_nbhd", "n_instants", "tau_a", "I_a_lo",
               "I_a_hi", "tau_ab", "I_ab_lo", "I_ab_hi", "gain_ab", "tau_b", "se_tau_b",
               "I_b_lo", "I_b_hi", "E0_m", "Emin_m", "tau_all", "A_M2_gauge", "A_M2_B",
               "tau_c_raw", "tau_c_rel", "tau_hd26_oracle")


def csv_rows(site, stations_out, models=MODELS):
    """The flat lag_truth.csv rows (§7.2) of a site's lag truths."""
    rows = []
    for s in stations_out:
        r = {"site": site, "judge": s["code"], "location": s["location"],
             "included": s["included"], "exclusion_rule": ";".join(s["exclusion"] or []),
             "flags": ";".join(s["flags"] or []), "k_g": s["k_g"], "s_g_km": s["s_g_km"],
             "d_min_km": s["d_min_km"], "R_g_m": s["R_g_m"], "n_nbhd": s["n_nbhd"],
             "n_instants": s["a"]["n_instants"], "tau_a": s["a"]["tau"],
             "I_a_lo": s["a"]["lo"], "I_a_hi": s["a"]["hi"], "tau_ab": s["ab"]["tau"],
             "I_ab_lo": s["ab"]["lo"], "I_ab_hi": s["ab"]["hi"], "gain_ab": s["ab"]["gain"],
             "tau_b": s["b"]["tau"], "se_tau_b": s["b"]["se"], "I_b_lo": s["b"]["lo"],
             "I_b_hi": s["b"]["hi"], "E0_m": s["a"]["E0_m"], "Emin_m": s["a"]["Emin_m"],
             "tau_all": s["all"]["tau"], "A_M2_gauge": s["b"].get("A_M2_gauge"),
             "A_M2_B": s["b"].get("A_M2_B"), "tau_c_raw": s["c"]["gm20_raw"],
             "tau_c_rel": s["c"]["gm20_rel"], "tau_hd26_oracle": s["c"]["hd26_oracle"]}
        for model in models:
            k = _model_key(model)
            v = s["models"].get(model, {})
            ok = bool(v.get("available"))
            r[f"{k}_available"] = ok
            r[f"{k}_tau_a"] = v["a"]["tau"] if ok else np.nan
            r[f"{k}_tau_ab"] = v["ab"]["tau"] if ok else np.nan
            r[f"{k}_tau_b"] = v["b"]["tau"] if ok else np.nan
            r[f"{k}_ocean_km"] = v.get("ocean_km", np.nan) if ok else np.nan
        first = s["models"].get(models[0], {}) if models else {}
        r["ocean_km"] = first.get("ocean_km", np.nan) if first.get("available") else np.nan
        rows.append(r)
    return rows


def build_lag_truth(pkg, table=None, models=None, force=False, constituents=None):
    """truth/<site>/truth_lag.json, truth_lag_curves.npz, lag_truth.csv (§5.5).

    Parameters
    ----------
    pkg : CommonPackage
        Reads ``cfg``, ``t``, ``h0``, ``levels``, ``series1``,
        ``series10``, ``marks_lp`` and ``full`` (planted control only).
    table : list of dict, optional
        The judge table (default the one ``truth`` wrote).
    models : sequence, optional
        Global models of the baselines (default FROZEN['baselines']).
    force : bool
    constituents : sequence, optional
        Override of :data:`CONS_B` (tests only).

    Returns
    -------
    str
        The path of truth_lag.json.
    """
    t0 = time.time()
    cfg = pkg.cfg
    out = truth_dir(pkg)
    jpath = f"{out}/truth_lag.json"
    jt_path = f"{out}/judges.json"
    if table is None:
        table = judges.load(jt_path, judges.grid_of(pkg))
    table_doc = judges.read(jt_path) if os.path.exists(jt_path) else {}
    models = tuple(MODELS if models is None else models)
    key = lag_inputs_key(pkg, table_doc, models)
    if constituents is not None:
        key = config.sha256_json({"key": key, "constituents": list(constituents)})
    if not force and _done(jpath, key):
        log(f"[truth] {cfg.site}: lag truth up to date ({jpath})")
        return jpath
    stations_out, curves = [], {"taus": TAUS_A}
    if table:
        H = pkg.levels.stack(TAUS_A)
        idx10, h10 = series10(pkg)
        mm = np.isfinite(h10)
        hb_fit = {"fit": harmonic_fit(days_since_1992(idx10[mm]), h10[mm],
                                      CONS_B if constituents is None else constituents),
                  "n": int(mm.sum())}
        log(f"[truth] {cfg.site}: lag truths of {len(table)} judges at {len(record_t(pkg))} "
            f"record instants, tau {TAUS_A[0]}..{TAUS_A[-1]} min")
        for rec in table:
            res, cv = lag_truth_station(pkg, rec, H, models=models, hb_fit=hb_fit,
                                        constituents=constituents)
            stations_out.append(res)
            for name, v in cv.items():
                curves[f"{rec['code']}__{name}"] = np.asarray(v)
    planted = planted_lag(pkg) if cfg.site == config.CONTROL_SITE else None
    os.makedirs(out, exist_ok=True)
    product.save_npz(f"{out}/truth_lag_curves.npz", curves, compress=True)
    rows = csv_rows(cfg.site, stations_out, models)
    if rows:
        pd.DataFrame(rows).to_csv(f"{out}/lag_truth.csv", index=False)
    locs = {}
    for s in stations_out:
        locs.setdefault(s["location"], []).append(s["code"])
    sources = {r["code"]: _file_entry(judges.series_path(r)) for r in table}
    doc = {
        "format": TRUTH_LAG_FORMAT, "inputs_key": key,
        "site": cfg.site, "res_m": int(cfg.res_m), "scenario": getattr(cfg, "scenario", None),
        "n_record": int(len(record_t(pkg))),
        "taus_min": [int(TAUS_A[0]), int(TAUS_A[-1]), 1],
        "rules": {
            "a": {"tolerance_min": TOL_MIN, "interval_m": INTERVAL_M,
                  "E": "sqrt(mean_i (d_i - median d)^2), d_i = h_B(t_i - tau) - g_i",
                  "interp_max_gap_min": GAP_INTERP_MIN,
                  "bootstrap": {"n": N_BOOT, "block": "calendar month", "seed": BOOT_SEED}},
            "ab": {"gain_grid": [float(GAIN_GRID[0]), float(GAIN_GRID[-1]), 0.01],
                   "offset": "median (as E)"},
            "b": {"constituents": list(CONS_B if constituents is None else constituents),
                  "corrections": B_CORRECTIONS, "order": B_ORDER,
                  "jackknife": "delete one calendar quarter", "z": Z_B,
                  "series": ("10-min: as delivered; 1-min: mean over [T-5, T+5] min with "
                             ">= 6 samples; 5-min: same window with >= 2 samples")},
            "all": "E over every 10-min sample, h_B from the exact 1-min series",
            "hw_lw": "g21.tide_marks of the centred 10-min gauge vs marks_lp, paired within 3 h",
            "c": {"gm20": "fold on hours since marks_lp HW, period 12.4206012 h",
                  "hd26": "FROZEN hd26 phase of the gauge minus that of h0"},
            "baselines": list(models)},
        "locations": locs,
        "headline_locations": sorted({s["location"] for s in stations_out if s["included"]}),
        "stations": stations_out,
        "planted": planted,
        "sources": sources,
        "package_key": _package_key(pkg),
        "frozen": config.frozen_shas(("truth_lag", "judges", "baselines", "gm20", "hd26")),
        "code": _code_shas(),
        "pyTMD": _pytmd_version(),
        "built_utc": product.utc_stamp(), "seconds": round(time.time() - t0, 1),
    }
    product.write_json(jpath, doc)
    log(f"[truth] {cfg.site}: lag truth of {len(stations_out)} judges, "
        f"{len(doc['headline_locations'])} headline locations [{doc['seconds']} s] -> {jpath}")
    return jpath


def _pytmd_version():
    try:
        import pyTMD
        return getattr(pyTMD, "__version__", "?")
    except Exception:                          # recorded, never fatal
        return None


def load_lag_truth(pkg_or_cfg):
    """The lag truth document of a site (truth_lag.json)."""
    cfg = getattr(pkg_or_cfg, "cfg", pkg_or_cfg)
    doc = product.read_json(f"{cfg.truth_dir}/truth_lag.json")
    if doc.get("format") != TRUTH_LAG_FORMAT:
        raise ValueError(f"{cfg.truth_dir}/truth_lag.json: not a {TRUTH_LAG_FORMAT} file")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  the `truth` command
# ─────────────────────────────────────────────────────────────────────────────

PARTS = ("elev", "lag")


def build(pkg, parts=PARTS, force=False, models=None):
    """The ``truth`` command for one site (§5.1, §5.5).

    1. the score-side judge table, ``truth/<site>/judges.json`` (rebuilt
       from the package, the stations and their files; compared with the
       package's prepare-time copy when that exists);
    2. the elevation truth, the domains and the support S;
    3. the lag truths and the global-model baselines.

    Parameters
    ----------
    pkg : CommonPackage
    parts : sequence of {"elev", "lag"}
    force : bool
        Rebuild parts whose stored inputs key matches.
    models : sequence, optional
        Baseline models (default FROZEN['baselines']['models']).

    Returns
    -------
    dict
        ``{"judges": path, "elev": path, "lag": path}`` of what was built.
    """
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown truth part(s) {unknown}; expected {PARTS}")
    t0 = time.time()
    cfg = pkg.cfg
    out = truth_dir(pkg)
    os.makedirs(out, exist_ok=True)
    jt_path = f"{out}/judges.json"
    table = judges.build(pkg, path=jt_path)
    pkg_copy = judges.judges_path(pkg)
    if os.path.exists(pkg_copy):
        try:
            a = {s["code"]: s for s in judges.read(pkg_copy)["stations"]}
            b = {s["code"]: s for s in judges.read(jt_path)["stations"]}
            fields = ("included", "k_g", "R_g_m", "nbhd_sha256", "location")
            diff = sorted(c for c in set(a) | set(b)
                          if c not in a or c not in b
                          or any(a[c].get(f) != b[c].get(f) for f in fields))
            log(f"[truth] {cfg.site}: judge table "
                + ("equals the package copy" if not diff else f"differs from the package copy "
                                                               f"for {diff}"))
        except (OSError, ValueError, KeyError) as e:
            log(f"[truth] {cfg.site}: package judge table unreadable ({e})")
    paths = {"judges": jt_path}
    if "elev" in parts:
        paths["elev"] = build_elevation_truth(pkg, table=table, force=force)
    if "lag" in parts:
        paths["lag"] = build_lag_truth(pkg, table=table, models=models, force=force)
    log(f"[truth] {cfg.site} {cfg.res_m} m done [{time.time() - t0:.0f} s]")
    return paths


def run(site, res_m, scenario=None, parts=PARTS, force=False, models=None):
    """Load a site's package and build its truths (the CLI's ``truth``)."""
    from experiments.benchmark import package

    kw = {} if scenario is None else {"scenario": scenario}
    pkg = package.load(site, res_m, **kw)
    return build(pkg, parts=parts, force=force, models=models)


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION v2 (docs/validation_v2_spec.md §1.7-§1.9, §6.2, §7.1)
# ═════════════════════════════════════════════════════════════════════════════
#
# Everything above is v13 and is kept unchanged: the v13 continuity view and
# the reproduction gates still call it. Everything below is new, reads no
# method output and is frozen in TRUTH_V2 (merged into FROZEN at lock-A as
# the sections truth_elev, truth_lag, truth_error and strata).

TRUTH_ELEV_V2_FORMAT = "v2-truth-elev-1"
TRUTH_LAG_V2_FORMAT = "v2-truth-lag-1"
TRUTH_ERROR_FORMAT = "v2-truth-error-1"

#: Vaklodingen years kept per cell (§1.8): every survey of 2016-2025
VAK_YEARS = (2016, 2025)
#: v13's view, kept for continuity: the latest finite survey of 2019-2025
LATEST_YEARS = (2019, 2025)
#: a survey labelled year Y has epoch Y + 0.5 (mid-year, +-0.5 year)
EPOCH_OFFSET_Y = 0.5
#: the 20 m truth cell is scored when at least 2 of its pixels, and at least
#: half of them, are mapped ("2 of its 4"; declared for cells holding 3, 5 or
#: 6 pixel centres of the rotated 10 m grid)
CELL_MIN_PX = 2
CELL_MIN_SHARE = 0.5
#: stable-cell stratum [V4]
STABLE_MIN_SURVEYS = 2
STABLE_PAD_YEARS = 1
STABLE_MAX_RANGE_M = 0.20
#: range filter quantiles of the frozen 1-min h_B
RANGE_Q = (0.001, 0.999)
#: PNOA (§1.8): constant-value patches and the flight-level mask
PATCH_TOL_M = 0.001
PATCH_MIN_PX = 50
FLIGHT_MARGIN_M = 0.15
PNOA_MIN_VALID = 0.95           # v10: a 10 m pixel needs 95 % valid 2 m area
PNOA_MIN_GROUND = 0.8           # v10: and 80 % of it ground
#: strata class limits (§7.1); a value v gets class k = #(edges <= v)
LAG_EDGES_MIN = (10.0, 30.0, 60.0, 120.0)
AMP_EDGES = (0.95, 1.05, 1.15)
SLOPE_EDGES_PCT = (0.1, 0.5, 2.0)
SLOPE_SMOOTH_M = 40.0
EPOCH_GAP_EDGES_Y = (1.0, 2.0, 4.0)
NDVI_EDGES = (0.1, 0.3)
PERMANENT_WATER_FREQ = 0.9
LOW_TIDE_QUANTILE = 0.25
#: hydroperiod zones of hydroperiod.zonation: (lo, hi] of the submerged share
ZONES = ((0.90, 1.01, "subtidal"), (0.50, 0.90, "lower intertidal"),
         (0.20, 0.50, "mid intertidal"), (0.05, 0.20, "upper intertidal"),
         (-0.01, 0.05, "supratidal"))
#: the truth-error sensitivity multipliers of sigma2_T (§1.9)
SIGMA_T_SENS = (0.5, 1.0, 1.5)

TRUTH_V2 = config._freeze({
    "truth_elev": {
        "rule": ("Vaklodingen all epochs per cell 2016-2025; survey-matched default (nearest "
                 "the window centre, ties to the later survey); 20 m cell support (>= 2 and "
                 ">= half of the cell's pixels mapped); PNOA ground-only with the flight-level "
                 "mask (+0.15 m) and constant-patch water (>= 50 px equal at 1 mm); RTK via "
                 "load_rtk() defaults; range [q_0.001(h_B) - 0.25, q_0.999(h_B) + 0.25] over "
                 "the 1-min series; stable = >= 2 surveys within window +- 1 year and range "
                 "<= 0.20 m"),
        "vaklodingen": {"years": list(VAK_YEARS), "continuity_years": list(LATEST_YEARS),
                        "within_year": "the latest stamp of a year per cell",
                        "cell": ("first sheet (sheets_for order) whose cell holds a finite "
                                 "survey in any kept year; nearest cell by rounding (v1)"),
                        "epoch_offset_y": EPOCH_OFFSET_Y,
                        "matching": "nearest epoch to the window centre, ties to the later"},
        "cell_support": {"min_px": CELL_MIN_PX, "min_share": CELL_MIN_SHARE,
                         "value": "mean of the cell's mapped 10 m pixels"},
        "stable": {"min_surveys": STABLE_MIN_SURVEYS, "pad_years": STABLE_PAD_YEARS,
                   "max_range_m": STABLE_MAX_RANGE_M},
        "range": {"quantiles": list(RANGE_Q), "pad_m": RANGE_PAD_M,
                  "series": "the frozen 1-min h_B over UNION_PERIOD",
                  "domain": ("D_truth(window) = cells/pixels whose window truth lies in the "
                             "fixed bounds; S uses the union over the registered windows")},
        "pnoa": {"patch_tol_m": PATCH_TOL_M, "patch_min_px": PATCH_MIN_PX, "connectivity": 4,
                 "patch_rule": "4-connected pixels whose values round to the same mm",
                 "flight_margin_m": FLIGHT_MARGIN_M, "min_valid": PNOA_MIN_VALID,
                 "min_ground": PNOA_MIN_GROUND,
                 "flight_level": ("EOT20 at the site + the Gijon (gij2) residual (gauge - "
                                  "EOT20 at the gauge, minus its median over +-15 days); the "
                                  "highest strip level when several strips cover the site"),
                 "no_flight_time": "constant-patch mask only; Villaviciosa elevation descriptive"},
        "rtk": ("rtk.load_rtk() with default arguments, elev_alicante, never range-filtered; "
                "campaign epoch from the CSV's time column"),
    },
    "strata": {
        "reference_lag_edges_min": list(LAG_EDGES_MIN),
        "amplification_edges": list(AMP_EDGES),
        "along_s": "linear in s between (0, mouth value) and the judges, constant beyond",
        "zones": [[lo, hi, name] for lo, hi, name in ZONES],
        "zone_series": "h_ref at 10 min (LG at judged sites, else h_B)",
        "sampled_range": "below / inside / above [min, max] of h_ref at the window's S2 instants",
        "slope_edges_pct": list(SLOPE_EDGES_PCT), "slope_smooth_m": SLOPE_SMOOTH_M,
        "permanent_water_freq": PERMANENT_WATER_FREQ,
        "substrate": "per-site terciles of the low-tide median rho11 of dry P0 observations",
        "vegetation_ndvi_edges": list(NDVI_EDGES),
        "low_tide_quantile": LOW_TIDE_QUANTILE,
        "turbidity": ("per-site terciles of the per-band median (over clear scenes) of the "
                      "per-scene median rho4 on channel anchors"),
        "epoch_gap_edges_y": list(EPOCH_GAP_EDGES_Y),
        "n_clear": "per site and window terciles over D_truth",
    },
    "truth_lag": {
        "rule": ("per window: (a), tau_ab, (b), HW/LW, tau_all, amplitude ratio; centred gauge "
                 "series; the record instants (catalogue / product-name times by default, "
                 "user decision 2026-10-14); 200 month-block bootstraps (seed 20261011, "
                 "[seed, SITE_ID, crc32(code), window_num])"),
        "coverage": "the judge inclusion rules re-evaluated on each window's record",
        "jackknife": "quarter-years; months in 1-year windows",
        "hw_lw": ("gauge interpolated to 1 min (gaps <= 20 min), both low-passed with the "
                  "61-min mean, marks paired with the nearest boundary mark of the same kind "
                  "within 3 h; median, IQR, the mean of HW and LW, LW - HW"),
        "amplitude_ratio": "A_M2(gauge) / A_M2(h_B) of the (b) fit",
    },
    "truth_error": {
        "survey_noise": ("Escalda: D(dt) = Var(t_Y1 - t_Y2) pooled per dt; 2 sigma2_s = the "
                         "larger intercept of (i) the increasing isotonic fit extrapolated "
                         "linearly to dt = 0 and (ii) c + a dt^p; Wadden/Ems: same-year "
                         "overlaps, else Escalda; Villaviciosa: PNOA accuracy with its random "
                         "part reduced by the 2 m -> 10 m aggregation"),
        "epoch_mismatch": ("D_W(Y) = Var(mean_W t - t_Y) minus the noise of the window mean "
                           "and of survey Y (each counted once); transferred by the ratio of "
                           "the D(dt) curves above their noise floors"),
        "total": "sigma2_T = sigma2_s + D_W(Y); lambda = 1 - sigma2_T / Var(t)",
        "sensitivity": list(SIGMA_T_SENS),
    },
})


def truth_v2_plain(section=None):
    """A JSON-ready copy of :data:`TRUTH_V2` (or of one of its sections)."""
    return config._plain(TRUTH_V2 if section is None else TRUTH_V2[section])


def truth_v2_sha(section=None):
    """sha256 of :func:`truth_v2_plain` (what lock-A freezes)."""
    return config.sha256_json(truth_v2_plain(section))


def classify(values, edges):
    """Class index of each value: ``k = #(edges <= v)`` (0..len(edges)), -1
    where the value is not finite."""
    v = np.asarray(values, np.float64)
    out = np.full(v.shape, -1, np.int8)
    fin = np.isfinite(v)
    out[fin] = np.searchsorted(np.asarray(edges, np.float64), v[fin], side="right")
    return out


def epoch_of(year):
    """The epoch (decimal year) of a survey labelled ``year``."""
    return np.asarray(year, np.float64) + EPOCH_OFFSET_Y


def window_of(wid):
    """The :class:`settings.Window` of a window id."""
    from experiments.benchmark import settings

    return settings.window(wid)


def window_years(wid):
    """(first, last) calendar year of a window (inclusive)."""
    y = window_of(wid).years
    return int(y[0]), int(y[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  v2: multi-epoch Vaklodingen (§1.8)
# ─────────────────────────────────────────────────────────────────────────────

def read_sheet_epochs(path, years=VAK_YEARS):
    """Every survey of one Vaklodingen sheet in ``years`` (inclusive).

    Surveys stamped in the same calendar year (e.g. 2005-03-02 and
    2005-09-07) are merged per cell: the later stamp's finite value wins,
    as v1 merges years.

    Returns
    -------
    dict
        x (nx,), y (ny,) RD cell centres; years (n,) int; z (n, ny, nx)
        float64; src (n, ny, nx) int16 (the ``isource`` index of the value,
        -1 none); stamps {year: [ISO dates]}; sources (``isource``
        flag_meanings, the source file names) and source_attr; path.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        zvar = "z" if "z" in ds else [v for v in ds.data_vars if ds[v].ndim == 3][0]
        stamps = pd.DatetimeIndex(ds["time"].values)
        yrs = stamps.year.to_numpy()
        sel = np.flatnonzero((yrs >= int(years[0])) & (yrs <= int(years[1])))
        x = np.asarray(ds["x"].values, np.float64)
        y = np.asarray(ds["y"].values, np.float64)
        uy = np.unique(yrs[sel]).astype(np.int64)
        Z = np.full((len(uy), len(y), len(x)), np.nan)
        SRC = np.full(Z.shape, -1, np.int16)
        has_src = "isource" in ds
        for i in sel[np.argsort(stamps[sel].asi8, kind="stable")]:
            k = int(np.searchsorted(uy, yrs[i]))
            z = np.asarray(ds[zvar].isel(time=int(i)).values, np.float64)
            m = np.isfinite(z)
            Z[k][m] = z[m]
            if has_src:
                s = np.asarray(ds["isource"].isel(time=int(i)).values, np.float64)[m]
                SRC[k][m] = np.where(np.isfinite(s), s, -1).astype(np.int16)
        meanings = (str(ds["isource"].attrs.get("flag_meanings", "")).split()
                    if has_src else [])
        attr = str(ds["isource"].attrs.get("source", "")).split() if has_src else []
    st = {}
    for i in sel:
        st.setdefault(int(yrs[i]), []).append(str(stamps[i].date()))
    return {"x": x, "y": y, "years": uy, "z": Z, "src": SRC, "stamps": st,
            "sources": meanings, "source_attr": attr, "path": str(path).replace("\\", "/")}


def vaklodingen_sheets(site, truth_dir_=None):
    """``{sheet name: path}`` of a site's sheets in ``sheets_for`` order
    (missing files are logged and skipped). The sheets are read one at a
    time by :func:`vaklodingen_cells`."""
    from experiments import v1_vaklodingen_validate as v1
    from experiments.v0_vaklodingen_fetch import sheets_for

    base = truth_dir_ or v1.TRUTH_DIR
    out = {}
    for name, _, _ in sheets_for(site):
        p = os.path.join(base, name).replace("\\", "/")
        if os.path.exists(p):
            out[name] = p
        else:
            log(f"[truth] {site}: missing Vaklodingen sheet {name}")
    return out


def rd_coords(grid, shape, block_rows=256):
    """RD (EPSG:28992) coordinates of every frame pixel centre, flat (H*W,)."""
    import pyproj

    H, W = int(shape[0]), int(shape[1])
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xs, ys = np.asarray(grid["xs"], float), np.asarray(grid["ys"], float)
    XQ = np.empty(H * W)
    YQ = np.empty(H * W)
    for r0 in range(0, H, int(block_rows)):
        r1 = min(H, r0 + int(block_rows))
        rows = np.repeat(np.arange(r0, r1), W)
        cols = np.tile(np.arange(W), r1 - r0)
        xq, yq = to_rd.transform(xs[cols], ys[rows])
        XQ[r0 * W:r1 * W] = xq
        YQ[r0 * W:r1 * W] = yq
    return XQ, YQ


def vaklodingen_cells(xq, yq, sheets, years=VAK_YEARS):
    """The multi-epoch truth cells behind a set of query points.

    Each point takes, among the sheets in order, the first whose nearest
    cell (v1's rounding arithmetic) holds a finite survey in any kept year;
    that cell keeps EVERY survey year (v13 kept only the latest).

    Parameters
    ----------
    xq, yq : (P,) float
        RD coordinates (e.g. :func:`rd_coords` of the frame).
    sheets : mapping
        ``{name: path or read_sheet_epochs dict}`` in sheet order; a path
        is read, used and released before the next sheet.
    years : (int, int)

    Returns
    -------
    dict
        years (n_y,) int (only years surveyed somewhere); Z (n_c, n_y)
        float64 (NaN = not surveyed); SRC (n_c, n_y) int16; cell_id (n_c,)
        int64 ascending (:func:`cell_ids`); sheet, iy, ix (n_c,); x_rd, y_rd
        (n_c,); point_cell (P,) int32 (index into the cells, -1 none);
        n_px (n_c,) points per cell; sheets (names); sources {name: [...]};
        stamps {name: {...}}.
    """
    xq = np.asarray(xq, np.float64)
    yq = np.asarray(yq, np.float64)
    P = len(xq)
    y0, y1 = int(years[0]), int(years[1])
    ny_all = y1 - y0 + 1
    sheet_of = np.full(P, -1, np.int16)
    parts, names, sources, stamps = [], [], {}, {}
    for k, (name, sh) in enumerate(sheets.items()):
        if not isinstance(sh, Mapping):
            sh = read_sheet_epochs(sh, years)
        names.append(name)
        sources[name] = list(sh.get("sources", []))
        stamps[name] = dict(sh.get("stamps", {}))
        x, y = np.asarray(sh["x"], float), np.asarray(sh["y"], float)
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        free = np.flatnonzero(sheet_of < 0)
        with np.errstate(invalid="ignore"):
            fx = np.round((xq[free] - x[0]) / dx)
            fy = np.round((yq[free] - y[0]) / dy)
        ok = np.isfinite(fx) & np.isfinite(fy)
        ok &= (fx >= 0) & (fx < len(x)) & (fy >= 0) & (fy < len(y))
        ix = np.where(ok, fx, 0).astype(np.int64)
        iy = np.where(ok, fy, 0).astype(np.int64)
        Zs = np.asarray(sh["z"], np.float64)
        anyf = np.isfinite(Zs).any(axis=0)
        ok &= anyf[iy, ix]
        take = free[ok]
        sheet_of[take] = k
        if not len(take):
            continue
        cid = cell_ids(np.full(len(take), k), iy[ok], ix[ok])
        uc, first = np.unique(cid, return_index=True)
        ciy, cix = iy[ok][first], ix[ok][first]
        Zc = np.full((len(uc), ny_all), np.nan)
        Sc = np.full((len(uc), ny_all), -1, np.int16)
        src = sh.get("src")
        for j, yr in enumerate(np.asarray(sh["years"], np.int64)):
            if y0 <= yr <= y1:
                Zc[:, yr - y0] = Zs[j][ciy, cix]
                if src is not None:
                    Sc[:, yr - y0] = np.asarray(src[j])[ciy, cix]
        parts.append({"cell_id": uc, "sheet": np.full(len(uc), k, np.int16),
                      "iy": ciy.astype(np.int32), "ix": cix.astype(np.int32),
                      "x_rd": x[cix], "y_rd": y[ciy], "Z": Zc, "SRC": Sc,
                      "pid": take, "pcid": cid})
    keys = ("cell_id", "sheet", "iy", "ix", "x_rd", "y_rd", "Z", "SRC", "pid", "pcid")
    if parts:
        cat = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    else:
        cat = {"cell_id": np.empty(0, np.int64), "sheet": np.empty(0, np.int16),
               "iy": np.empty(0, np.int32), "ix": np.empty(0, np.int32),
               "x_rd": np.empty(0), "y_rd": np.empty(0), "Z": np.empty((0, ny_all)),
               "SRC": np.empty((0, ny_all), np.int16), "pid": np.empty(0, np.int64),
               "pcid": np.empty(0, np.int64)}
    point_cell = np.full(P, -1, np.int32)
    point_cell[cat["pid"]] = np.searchsorted(cat["cell_id"], cat["pcid"]).astype(np.int32)
    keep_y = np.isfinite(cat["Z"]).any(axis=0)
    yrs = np.arange(y0, y1 + 1)[keep_y]
    n_c = len(cat["cell_id"])
    return {"years": yrs.astype(np.int64), "Z": cat["Z"][:, keep_y],
            "SRC": cat["SRC"][:, keep_y], "cell_id": cat["cell_id"].astype(np.int64),
            "sheet": cat["sheet"], "iy": cat["iy"], "ix": cat["ix"],
            "x_rd": cat["x_rd"], "y_rd": cat["y_rd"], "point_cell": point_cell,
            "n_px": np.bincount(point_cell[point_cell >= 0], minlength=n_c).astype(np.int32),
            "sheets": names, "sources": sources, "stamps": stamps}


def survey_matched(Z, years, centre):
    """The survey-matched truth (§1.8): per cell, the finite survey whose
    epoch (year + 0.5) is nearest ``centre``; ties go to the later survey.

    Returns
    -------
    z (n_c,) float64 (NaN none), year (n_c,) int16 (-1 none), gap (n_c,)
    float64 = |epoch - centre| in years (NaN none).
    """
    Z = np.asarray(Z, np.float64)
    years = np.asarray(years, np.int64)
    n = Z.shape[0]
    z = np.full(n, np.nan)
    yr = np.full(n, -1, np.int16)
    gap = np.full(n, np.nan)
    if not len(years):
        return z, yr, gap
    d = np.abs(epoch_of(years) - float(centre))
    order = np.lexsort((-years, d))           # nearest first, the later on ties
    for j in order:
        take = ~np.isfinite(z) & np.isfinite(Z[:, j])
        z[take] = Z[take, j]
        yr[take] = years[j]
        gap[take] = d[j]
    return z, yr, gap


def latest_survey(Z, years, span=LATEST_YEARS):
    """v13's continuity view: the latest finite survey in ``span``.

    Returns z (n_c,), year (n_c,) int16 (-1 none).
    """
    Z = np.asarray(Z, np.float64)
    years = np.asarray(years, np.int64)
    z = np.full(Z.shape[0], np.nan)
    yr = np.full(Z.shape[0], -1, np.int16)
    for j in np.flatnonzero((years >= span[0]) & (years <= span[1])):
        m = np.isfinite(Z[:, j])
        z[m] = Z[m, j]
        yr[m] = years[j]
    return z, yr


def survey_in_year(Z, years, year):
    """The survey of one year per cell (NaN where that year was not
    surveyed): one point of the mismatch curve (§2.7)."""
    years = np.asarray(years, np.int64)
    j = np.flatnonzero(years == int(year))
    if not len(j):
        return np.full(np.asarray(Z).shape[0], np.nan)
    return np.asarray(Z, np.float64)[:, int(j[0])].copy()


def stable_cells(Z, years, span, pad=STABLE_PAD_YEARS, min_surveys=STABLE_MIN_SURVEYS,
                 max_range=STABLE_MAX_RANGE_M):
    """The stable-cell stratum [V4]: at least ``min_surveys`` surveys in the
    window's years +- ``pad`` and their range at most ``max_range`` m;
    every other cell is "not assessable".

    Parameters
    ----------
    span : (int, int)
        The window's first and last calendar year.

    Returns
    -------
    dict
        stable (n_c,) bool, n (n_c,) surveys in the span, range_m (n_c,)
        (NaN with fewer than 2), years [lo, hi].
    """
    Z = np.asarray(Z, np.float64)
    years = np.asarray(years, np.int64)
    lo, hi = int(span[0]) - int(pad), int(span[1]) + int(pad)
    sub = Z[:, (years >= lo) & (years <= hi)]
    n = np.isfinite(sub).sum(axis=1)
    rng = np.full(Z.shape[0], np.nan)
    m = n >= 2
    if m.any():
        rng[m] = np.nanmax(sub[m], axis=1) - np.nanmin(sub[m], axis=1)
    with np.errstate(invalid="ignore"):
        stable = (n >= int(min_surveys)) & (rng <= float(max_range))
    return {"stable": stable, "n": n.astype(np.int16), "range_m": rng, "years": [lo, hi]}


def cell_aggregate(values, point_cell, n_cells, n_px, mapped=None,
                   min_px=CELL_MIN_PX, min_share=CELL_MIN_SHARE):
    """A product on the 20 m truth cells (§7.1).

    The cell value is the mean of its mapped 10 m pixels; the cell is scored
    when at least ``min_px`` and at least ``min_share`` of its ``n_px``
    pixels are mapped.

    Parameters
    ----------
    values : (P,) float
        The product on a set of pixels (NaN = not mapped).
    point_cell : (P,) int
        The cell of each pixel (-1 none).
    n_cells : int
    n_px : (n_cells,) int
        Pixels per cell over the whole frame (``vaklodingen_cells``).
    mapped : (P,) bool, optional
        Default ``isfinite(values)``.

    Returns
    -------
    mean (n_cells,) float64 (NaN where not scored), n_mapped (n_cells,),
    scored (n_cells,) bool.
    """
    v = np.asarray(values, np.float64)
    pc = np.asarray(point_cell, np.int64)
    m = np.isfinite(v) if mapped is None else (np.asarray(mapped, bool) & np.isfinite(v))
    m &= pc >= 0
    n_cells = int(n_cells)
    cnt = np.bincount(pc[m], minlength=n_cells)
    tot = np.bincount(pc[m], weights=v[m], minlength=n_cells)
    n_px = np.asarray(n_px, np.int64)
    scored = (cnt >= int(min_px)) & (cnt >= float(min_share) * n_px)
    mean = np.full(n_cells, np.nan)
    mean[scored] = tot[scored] / cnt[scored]
    return mean, cnt.astype(np.int32), scored


def dated_epochs(SRC, years, sheet, src_dates):
    """The dated epoch of each cell's survey per year: the survey date's
    decimal year where ``fetch vak-meta`` found one (a stamp other than
    1 January, or a date embedded in the source file name), else mid-year.

    Parameters
    ----------
    SRC : (n_c, n_y) int16
    years : (n_y,) int
    sheet : (n_c,) int (index in the sheet order)
    src_dates : list
        Per sheet (same order), ``{source index: decimal year or None}``.

    Returns
    -------
    (n_c, n_y) float64
    """
    SRC = np.asarray(SRC, np.int64)
    out = np.broadcast_to(epoch_of(years)[None, :], SRC.shape).copy()
    sheet = np.asarray(sheet, np.int64)
    for k, table in enumerate(src_dates or []):
        rows = (sheet == k)[:, None]
        for s_idx, dec in (table or {}).items():
            if dec is not None:
                out[(SRC == int(s_idx)) & rows] = float(dec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  v2: range filter from the 1-min h_B (§1.8)
# ─────────────────────────────────────────────────────────────────────────────

def range_from_series(h, q=RANGE_Q, pad=None):
    """The fixed truth range [q_0.001(h_B) - pad, q_0.999(h_B) + pad].

    Parameters
    ----------
    h : array
        The frozen 1-min boundary series over the union period.

    Returns
    -------
    dict
        q, q_lo, q_hi (the quantiles, linear interpolation), lo, hi, pad_m,
        n, n_finite.
    """
    pad = RANGE_PAD_M if pad is None else float(pad)
    h = np.asarray(h, np.float64)
    fin = h[np.isfinite(h)]
    if not len(fin):
        raise ValueError("the boundary series has no finite level")
    qlo, qhi = (float(v) for v in np.quantile(fin, [float(q[0]), float(q[1])]))
    return {"q": [float(q[0]), float(q[1])], "q_lo": qlo, "q_hi": qhi,
            "lo": qlo - pad, "hi": qhi + pad, "pad_m": pad,
            "n": int(len(h)), "n_finite": int(len(fin))}


def apply_range(z_raw, lo, hi, band=None):
    """``z_raw`` inside [lo, hi], NaN elsewhere, with the counts (and the
    counts per stratum band when ``band`` is given, -1 = no band).

    Returns
    -------
    truth (like z_raw), counts dict (n_finite, n_below, n_above,
    n_in_range, lo, hi[, by_band {band: {...}}]).
    """
    z = np.asarray(z_raw, np.float64)
    fin = np.isfinite(z)
    with np.errstate(invalid="ignore"):
        below = fin & (z < lo)
        above = fin & (z > hi)
    inr = fin & ~below & ~above
    counts = {"lo": float(lo), "hi": float(hi), "n_finite": int(fin.sum()),
              "n_below": int(below.sum()), "n_above": int(above.sum()),
              "n_in_range": int(inr.sum())}
    if band is not None:
        b = np.asarray(band).ravel()
        by = {}
        for k in np.unique(b):
            m = b == k
            by[int(k)] = {"n_finite": int(fin.ravel()[m].sum()),
                          "n_below": int(below.ravel()[m].sum()),
                          "n_above": int(above.ravel()[m].sum()),
                          "n_in_range": int(inr.ravel()[m].sum())}
        counts["by_band"] = by
    return np.where(inr, z, np.nan), counts


def local_range(h_bins, q=RANGE_Q, pad=None):
    """Per-bin bounds of the LG sensitivity: the same quantile rule on each
    row of ``h_bins`` (n_bins, n_t), the local level series of a bin.

    Returns lo (n_bins,), hi (n_bins,) (NaN for a bin without levels).
    """
    pad = RANGE_PAD_M if pad is None else float(pad)
    Hb = np.asarray(h_bins, np.float64)
    lo = np.full(Hb.shape[0], np.nan)
    hi = np.full(Hb.shape[0], np.nan)
    for k in range(Hb.shape[0]):
        f = Hb[k][np.isfinite(Hb[k])]
        if len(f):
            a, b = np.quantile(f, [float(q[0]), float(q[1])])
            lo[k], hi[k] = a - pad, b + pad
    return lo, hi


def apply_local_range(z_raw, bin_of, lo_bins, hi_bins):
    """``z_raw`` inside its bin's bounds (NaN elsewhere, and where the
    point has no bin or its bin no bounds)."""
    z = np.asarray(z_raw, np.float64)
    b = np.asarray(bin_of, np.int64)
    lo_b = np.asarray(lo_bins, np.float64)
    hi_b = np.asarray(hi_bins, np.float64)
    ok = (b >= 0) & (b < len(lo_b))
    lo = np.full(z.shape, np.nan)
    hi = np.full(z.shape, np.nan)
    lo[ok] = lo_b[b[ok]]
    hi[ok] = hi_b[b[ok]]
    with np.errstate(invalid="ignore"):
        inr = np.isfinite(z) & (z >= lo) & (z <= hi)
    return np.where(inr, z, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
#  v2: PNOA ground-only truth with the water-surface masks (§1.8)
# ─────────────────────────────────────────────────────────────────────────────

def constant_patches(z, tol=PATCH_TOL_M, min_px=PATCH_MIN_PX):
    """Flattened water bodies: 4-connected patches of equal value.

    Two neighbouring pixels belong to one patch when their values round to
    the same multiple of ``tol`` (a water surface flattened to one value is
    always a single patch; a gently sloping surface never chains into one).

    Parameters
    ----------
    z : (H, W) float
        The 2 m MDT (NaN = no data).

    Returns
    -------
    water (H, W) bool, info {n_patches, n_px, values (value and size of the
    largest patches, at most 20)}.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = np.asarray(z)
    H, W = z.shape
    fin = np.isfinite(z)
    q = np.zeros(z.shape, np.int64)
    q[fin] = np.round(z[fin].astype(np.float64) / float(tol)).astype(np.int64)
    eh = fin[:, :-1] & fin[:, 1:] & (q[:, :-1] == q[:, 1:])
    ev = fin[:-1, :] & fin[1:, :] & (q[:-1, :] == q[1:, :])
    del q
    rh, ch = np.nonzero(eh)
    rv, cv = np.nonzero(ev)
    a = np.concatenate([rh.astype(np.int64) * W + ch, rv.astype(np.int64) * W + cv])
    b = np.concatenate([rh.astype(np.int64) * W + ch + 1, (rv.astype(np.int64) + 1) * W + cv])
    water = np.zeros(H * W, bool)
    info = {"n_patches": 0, "n_px": 0, "values": []}
    if not len(a):
        return water.reshape(H, W), info
    nodes = np.unique(np.concatenate([a, b]))
    ia = np.searchsorted(nodes, a)
    ib = np.searchsorted(nodes, b)
    g = coo_matrix((np.ones(len(ia), np.int8), (ia, ib)), shape=(len(nodes), len(nodes)))
    ncomp, lab = connected_components(g, directed=False)
    size = np.bincount(lab, minlength=ncomp)
    big = size >= int(min_px)
    water[nodes[big[lab]]] = True
    comps = np.flatnonzero(big)
    order = comps[np.argsort(-size[comps], kind="stable")][:20]
    first = np.full(ncomp, -1, np.int64)
    first[lab[::-1]] = nodes[::-1]
    zf = z.ravel()
    info = {"n_patches": int(big.sum()), "n_px": int(water.sum()),
            "values": [[round(float(zf[first[c]]), 4), int(size[c])] for c in order]}
    return water.reshape(H, W), info


def flight_mask(z, level_m, margin=FLIGHT_MARGIN_M):
    """Pixels under water at flight time: finite z < level + margin (an
    unknown level, None or NaN, masks nothing)."""
    z = np.asarray(z, np.float64)
    if level_m is None or not np.isfinite(level_m):
        return np.zeros(z.shape, bool)
    with np.errstate(invalid="ignore"):
        return np.isfinite(z) & (z < float(level_m) + float(margin))


def flight_water_level(t_flight, local_tide, gauge=None, gauge_tide=None, window_days=15.0,
                       tol_min=None):
    """The water level at the site at flight time.

    level = local_tide(t) + r(t), with the gauge's non-tidal residual
    r(t) = [g(t) - gauge_tide(t)] - the median over +- ``window_days`` of
    (g - gauge_tide) (the gauge's datum cancels). Without a gauge, r = 0
    (declared).

    Parameters
    ----------
    t_flight : datetime-like or sequence
        Naive-UTC flight instants (one per strip).
    local_tide, gauge_tide : callable
        ``levels(DatetimeIndex) -> array`` (a boundary provider's
        ``levels``): the tide model at the site and at the gauge.
    gauge : pd.Series, optional
        The Gijon gauge (gij2) as delivered.

    Returns
    -------
    dict
        level (the highest over the instants), levels, tide, residual (per
        instant), n_residual, instants.
    """
    t = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(t_flight))).as_unit("ns")
    tide = np.asarray(local_tide(t), np.float64)
    res = np.zeros(len(t))
    n_res = 0
    if gauge is not None and gauge_tide is not None and len(gauge):
        g = gauge.sort_index()
        lo = t.min() - pd.Timedelta(days=float(window_days))
        hi = t.max() + pd.Timedelta(days=float(window_days))
        gw = g[(g.index >= lo) & (g.index <= hi)]
        if len(gw):
            r_all = gw.to_numpy(np.float64) - np.asarray(gauge_tide(gw.index), np.float64)
            med = float(np.nanmedian(r_all))
            g_t = judges.at_instants(g, t, tol_min)
            res = g_t - np.asarray(gauge_tide(t), np.float64) - med
            n_res = int(np.isfinite(r_all).sum())
    lev = tide + res
    return {"level": float(np.nanmax(lev)) if np.isfinite(lev).any() else float("nan"),
            "levels": lev.tolist(), "tide": tide.tolist(), "residual": res.tolist(),
            "n_residual": n_res, "instants": [str(x) for x in t]}


def pnoa_ground(src, flight_level=None, tol=PATCH_TOL_M, min_px=PATCH_MIN_PX,
                margin=FLIGHT_MARGIN_M):
    """The ground mask of a 2 m MDT window.

    ground = valid & not in a constant patch & not under water at flight
    time (this replaces the 3 x 3 flatness rule of ``v10.lidar_truth``).

    Returns
    -------
    dict
        valid, water_patch, below_flight (outside the patches), ground (bool
        arrays), patches (the :func:`constant_patches` info).
    """
    valid = np.isfinite(src)
    water, info = constant_patches(src, tol, min_px)
    below = flight_mask(src, flight_level, margin)
    return {"valid": valid, "water_patch": water, "below_flight": below & ~water,
            "ground": valid & ~water & ~below, "patches": info}


def aggregate_ground(src, src_transform, src_crs, ground, valid, dst_transform, dst_crs,
                     dst_shape, min_valid=PNOA_MIN_VALID, min_ground=PNOA_MIN_GROUND):
    """The 10 m mean of the ground 2 m pixels (``v10.lidar_truth``'s
    averaging): area-weighted reprojection of the sum and the counts; a
    pixel needs ``min_valid`` of its area valid and ``min_ground`` of that
    ground.

    Returns
    -------
    z (dst_shape) float64 (NaN where the rules fail), ground share, valid share.
    """
    from rasterio.warp import Resampling, reproject

    kw = dict(src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform,
              dst_crs=dst_crs, resampling=Resampling.average, src_nodata=None,
              dst_nodata=None)
    gsum, gcnt, vcnt = (np.zeros(tuple(dst_shape)) for _ in range(3))
    reproject(np.where(ground, src, 0).astype("float64"), gsum, **kw)
    reproject(np.asarray(ground, "float64"), gcnt, **kw)
    reproject(np.asarray(valid, "float64"), vcnt, **kw)
    mean = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1e-9), np.nan)
    gshare = gcnt / np.maximum(vcnt, 1e-9)
    ok = (vcnt > float(min_valid)) & (gshare >= float(min_ground))
    return np.where(ok, mean, np.nan), gshare, vcnt


def pnoa_truth(grid_path=None, mdt_path=None, flight_level=None):
    """The v2 PNOA truth on the frame grid (§1.8): ground-only MDT02 with
    the constant-patch and flight-level masks, averaged to 10 m.

    Parameters
    ----------
    grid_path : str
        A GeoTIFF on the frame grid (default ``rtk.DEFAULT_GRID``).
    mdt_path : str
        The MDT02 sheet (default ``v10.MDT02``).
    flight_level : float, optional
        The water level at flight time (:func:`flight_water_level`); None
        applies the constant-patch mask only.

    Returns
    -------
    dict
        z_raw (H, W); the 2 m shares of water patches, flight-masked and
        ground pixels; the patch info; flight_level; descriptive (True when
        the flight level is unknown); sources.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    from experiments import v10_validation_report as v10
    from pyintertidal import rtk

    grid_path = grid_path or rtk.DEFAULT_GRID
    mdt_path = mdt_path or v10.MDT02
    with rasterio.open(grid_path) as g:
        gt, gcrs, gshape, gb = g.transform, g.crs, g.shape, g.bounds
    with rasterio.open(mdt_path) as s:
        w = from_bounds(*transform_bounds(gcrs, s.crs, *gb, densify_pts=21),
                        transform=s.transform)
        w = w.round_offsets().round_lengths()
        src = s.read(1, window=w, boundless=True, fill_value=s.nodata).astype("float32")
        src[src == s.nodata] = np.nan
        st, scrs = s.window_transform(w), s.crs
    gr = pnoa_ground(src, flight_level)
    z, _, _ = aggregate_ground(src, st, scrs, gr["ground"], gr["valid"], gt, gcrs, gshape)
    nv = max(int(gr["valid"].sum()), 1)
    return {"z_raw": z, "water_patch_share_2m": float(gr["water_patch"].sum() / nv),
            "flight_masked_share_2m": float(gr["below_flight"].sum() / nv),
            "ground_share_2m": float(gr["ground"].sum() / nv),
            "patches": gr["patches"], "flight_level": flight_level,
            "descriptive": flight_level is None or not np.isfinite(flight_level),
            "sources": [str(mdt_path), str(grid_path)]}


def rtk_epoch(csv_path=None):
    """The RTK campaign epoch from the survey file's time column (every
    point's ``Averaging start``; no height is read).

    Returns
    -------
    dict
        first_utc, last_utc, dates (UTC calendar dates), n_points, epoch
        (decimal year of the first point).
    """
    from pyintertidal import rtk

    path = csv_path or rtk.DEFAULT_CSV
    col = pd.read_csv(path, usecols=["Averaging start"])["Averaging start"].astype(str)
    s = col.str.replace(r"\s*UTC", "", regex=True)
    t = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f%z", utc=True, errors="coerce")
    t = t.dropna().dt.tz_convert(None)
    if not len(t):
        raise ValueError(f"{path}: no readable 'Averaging start' time")
    t0 = t.min()
    return {"first_utc": str(t0), "last_utc": str(t.max()),
            "dates": sorted({str(d) for d in t.dt.date}), "n_points": int(len(t)),
            "epoch": float(t0.year + (t0.dayofyear - 0.5) / 365.25)}


# ─────────────────────────────────────────────────────────────────────────────
#  v2: the truth-error model (§1.9; method-free, frozen at lock-A)
# ─────────────────────────────────────────────────────────────────────────────

def pair_variances(Z, years, mask=None, min_cells=100):
    """Var(t_Y1 - t_Y2) over the cells surveyed in both years, for every
    pair of years.

    Returns
    -------
    pd.DataFrame
        y1, y2, dt, n, var (pairs with fewer than ``min_cells`` cells are
        dropped).
    """
    Z = np.asarray(Z, np.float64)
    years = np.asarray(years, np.int64)
    if mask is not None:
        Z = Z[np.asarray(mask, bool)]
    rows = []
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            d = Z[:, j] - Z[:, i]
            d = d[np.isfinite(d)]
            if len(d) >= int(min_cells):
                rows.append({"y1": int(years[i]), "y2": int(years[j]),
                             "dt": int(years[j] - years[i]), "n": int(len(d)),
                             "var": float(np.var(d))})
    return pd.DataFrame(rows, columns=["y1", "y2", "dt", "n", "var"])


def d_curve(pairs):
    """D(dt): the cell-weighted mean of the pair variances per year gap.

    Returns ``(dt (k,), D (k,), n (k,))`` sorted by dt.
    """
    if not len(pairs):
        return np.empty(0), np.empty(0), np.empty(0, np.int64)
    dts = sorted(int(v) for v in pairs["dt"].unique())
    D, n = [], []
    for k in dts:
        g = pairs[pairs["dt"] == k]
        D.append(float(np.average(g["var"], weights=g["n"])))
        n.append(int(g["n"].sum()))
    return np.asarray(dts, np.float64), np.asarray(D), np.asarray(n, np.int64)


def isotonic(y, w=None):
    """Increasing least-squares isotonic fit (pool-adjacent-violators)."""
    y = np.asarray(y, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1 + c2)
    return np.repeat(np.asarray(vals), cnt)


def isotonic_intercept(dt, D, w=None):
    """D(0) from the increasing isotonic fit of D(dt), extrapolated linearly
    to dt = 0 from its first two distinct levels and clipped to [0, the fit
    at the smallest dt] (g(0) = 0 and g increasing)."""
    dt = np.asarray(dt, np.float64)
    o = np.argsort(dt)
    dt, D = dt[o], np.asarray(D, np.float64)[o]
    if not len(D):
        return float("nan")
    fit = isotonic(D, None if w is None else np.asarray(w, np.float64)[o])
    k = np.flatnonzero(fit > fit[0])
    if not len(k):
        return float(max(fit[0], 0.0))
    first = np.flatnonzero(fit == fit[0])
    x0 = float(dt[first].mean())                 # the centre of the first pooled block
    j = int(k[0])
    nxt = np.flatnonzero(fit == fit[j])
    x1 = float(dt[nxt].mean())
    slope = (fit[j] - fit[0]) / (x1 - x0)
    return float(np.clip(fit[0] - slope * x0, 0.0, fit[0]))


def powerlaw_intercept(dt, D, w=None):
    """D(0) = c of the weighted least-squares fit D(dt) = c + a dt^p (c,
    a >= 0, 0.05 <= p <= 3)."""
    from scipy.optimize import curve_fit

    dt = np.asarray(dt, np.float64)
    D = np.asarray(D, np.float64)
    if len(D) < 3:
        return float("nan")
    sigma = None if w is None else 1.0 / np.sqrt(np.asarray(w, np.float64))
    p0 = [max(float(D.min()) * 0.5, 1e-6), max(float(D.max() - D.min()), 1e-6), 1.0]
    try:
        par, _ = curve_fit(lambda x, c, a, p: c + a * x ** p, dt, D, p0=p0, sigma=sigma,
                           bounds=([0.0, 0.0, 0.05], [np.inf, np.inf, 3.0]), maxfev=20000)
    except (RuntimeError, ValueError):
        return float("nan")
    return float(par[0])


def survey_noise(dt, D, w=None):
    """sigma2_s from the D(dt) curve (§1.9): half the larger of the
    isotonic and power-law intercepts (conservative).

    Returns
    -------
    dict
        sigma2_s, sigma_s, iso_intercept, power_intercept, chosen.
    """
    iso = isotonic_intercept(dt, D, w)
    pw = powerlaw_intercept(dt, D, w)
    cand = {k: v for k, v in (("isotonic", iso), ("power", pw)) if np.isfinite(v)}
    if not cand:
        return {"sigma2_s": float("nan"), "sigma_s": float("nan"), "iso_intercept": iso,
                "power_intercept": pw, "chosen": None}
    chosen = max(cand, key=cand.get)
    s2 = cand[chosen] / 2.0
    return {"sigma2_s": float(s2), "sigma_s": float(np.sqrt(s2)), "iso_intercept": iso,
            "power_intercept": pw, "chosen": chosen}


def overlap_noise(a, b, min_cells=100):
    """sigma2_s from same-year overlapping sheets: Var(z_a - z_b) / 2 over
    the cells (same RD centre) both sheets surveyed in the same year.

    Parameters
    ----------
    a, b : dict
        ``read_sheet_epochs`` dicts (x, y, years, z).

    Returns
    -------
    dict
        sigma2_s (NaN without enough overlap), n, years.
    """
    xa, ya = np.round(np.asarray(a["x"], float), 3), np.round(np.asarray(a["y"], float), 3)
    xb, yb = np.round(np.asarray(b["x"], float), 3), np.round(np.asarray(b["y"], float), 3)
    _, ia_x, ib_x = np.intersect1d(xa, xb, return_indices=True)
    _, ia_y, ib_y = np.intersect1d(ya, yb, return_indices=True)
    diffs, yrs = [], []
    if len(ia_x) and len(ia_y):
        yb_all = np.asarray(b["years"])
        for j, yr in enumerate(np.asarray(a["years"])):
            kb = np.flatnonzero(yb_all == yr)
            if not len(kb):
                continue
            za = np.asarray(a["z"])[j][np.ix_(ia_y, ia_x)]
            zb = np.asarray(b["z"])[int(kb[0])][np.ix_(ib_y, ib_x)]
            d = (za - zb).ravel()
            d = d[np.isfinite(d)]
            if len(d):
                diffs.append(d)
                yrs.append(int(yr))
    d = np.concatenate(diffs) if diffs else np.empty(0)
    if len(d) < int(min_cells):
        return {"sigma2_s": float("nan"), "n": int(len(d)), "years": yrs}
    return {"sigma2_s": float(np.var(d) / 2.0), "n": int(len(d)), "years": yrs}


def pnoa_noise(rmse_m, random_share, n_agg=25.0):
    """sigma2_s of the aggregated PNOA truth: the stated RMSE split into a
    systematic part (kept) and a random part (divided by the number of 2 m
    pixels per 10 m pixel). ``random_share`` is a pre-registered input."""
    v = float(rmse_m) ** 2
    rs = float(random_share)
    if not 0.0 <= rs <= 1.0:
        raise ValueError("random_share must lie in [0, 1]")
    return v * (1.0 - rs) + v * rs / float(n_agg)


def window_mismatch(Z, years, span, Y, sigma2_s, min_cells=100):
    """D_W(Y): the method-free variance between the window-averaged truth
    and survey Y, with each survey's noise counted once (§1.9).

    On the cells surveyed in every year of the window and in Y, with
    t_y = s_y + e_y (Var e = sigma2_s): Var(mean_W t - t_Y) = D_W(Y) +
    sigma2_s (1/n + 1) when Y is outside the window, D_W(Y) + sigma2_s
    (n - 1)/n when Y is one of its n years.

    Returns
    -------
    dict
        D_W (clipped at 0), var_obs, noise, n_cells, n_years, Y_in_window,
        window_years (and a reason when it cannot be estimated).
    """
    Z = np.asarray(Z, np.float64)
    years = np.asarray(years, np.int64)
    wy = list(range(int(span[0]), int(span[1]) + 1))
    have = [y for y in wy if y in set(years.tolist())]
    nan = float("nan")
    out = {"D_W": nan, "var_obs": nan, "noise": nan, "n_cells": 0, "n_years": len(have),
           "Y_in_window": int(Y) in wy, "window_years": wy}
    jY = np.flatnonzero(years == int(Y))
    if len(have) < len(wy) or not len(jY):
        out["reason"] = "the window's years or Y are not all surveyed"
        return out
    cols = [int(np.flatnonzero(years == y)[0]) for y in have]
    sub = Z[:, cols]
    tY = Z[:, int(jY[0])]
    m = np.isfinite(sub).all(axis=1) & np.isfinite(tY)
    n = len(cols)
    out["n_cells"] = int(m.sum())
    if m.sum() < int(min_cells):
        out["reason"] = "too few cells surveyed in every year"
        return out
    d = sub[m].mean(axis=1) - tY[m]
    var_obs = float(np.var(d))
    noise = float(sigma2_s) * ((n - 1) / n if int(Y) in wy else (1.0 / n + 1.0))
    out.update(D_W=max(var_obs - noise, 0.0), var_obs=var_obs, noise=noise)
    return out


def transfer_ratio(curve_site, curve_ref, s2_site, s2_ref):
    """The D(dt) ratio that transfers Escalda's D_W to a site (§1.9): the
    mean over the common year gaps of (D_site - 2 s2_site) / (D_ref - 2
    s2_ref) (numerator clipped at 0). ``curve_*`` = (dt, D)."""
    ds, Ds = (np.asarray(v, np.float64) for v in curve_site[:2])
    dr, Dr = (np.asarray(v, np.float64) for v in curve_ref[:2])
    r = []
    for g in np.intersect1d(ds, dr):
        num = Ds[ds == g][0] - 2 * s2_site
        den = Dr[dr == g][0] - 2 * s2_ref
        if den > 0:
            r.append(max(num, 0.0) / den)
    return float(np.mean(r)) if r else float("nan")


def reliability(var_t, sigma2_T, mult=1.0):
    """lambda = 1 - mult * sigma2_T / Var(t) (the attenuation of b)."""
    return 1.0 - float(mult) * float(sigma2_T) / float(var_t)


def truth_error_model(ref, sites=None, pnoa=None, windows=(), survey_years=None,
                      min_cells=100):
    """The frozen truth-error model (FROZEN['truth_error'], §1.9).

    Parameters
    ----------
    ref : dict
        Escalda's cells: ``Z`` (n_c, n_y), ``years``, optional ``mask``
        (e.g. the intertidal band).
    sites : dict, optional
        ``{site: {"Z", "years", ["mask"], ["overlap_sigma2_s"]}}`` (Wadden,
        Ems).
    pnoa : dict, optional
        ``{"rmse_m", "random_share", ["n_agg"]}`` for Villaviciosa.
    windows : sequence of str
        Window ids of the D_W(Y) table.
    survey_years : sequence of int, optional
        The Y of the table (default every year of ``ref``).

    Returns
    -------
    dict (JSON-ready)
        format, rules, sites {site: sigma2_s, method, curve, ratio}, D_W
        {site: {window: {Y: {...}}}}, sensitivity, sha256.
    """
    sites = sites or {}
    pr = pair_variances(ref["Z"], ref["years"], ref.get("mask"), min_cells)
    dt, D, n = d_curve(pr)
    noise_ref = survey_noise(dt, D, n if len(n) else None)
    s2_ref = noise_ref["sigma2_s"]
    doc = {"format": TRUTH_ERROR_FORMAT, "rules": truth_v2_plain("truth_error"),
           "sites": {"escalda": {"sigma2_s": s2_ref, "method": "D(dt) intercept",
                                 "noise": noise_ref,
                                 "curve": {"dt": dt.tolist(), "D": D.tolist(), "n": n.tolist()},
                                 "ratio": 1.0}},
           "D_W": {"escalda": {}}, "sensitivity": list(SIGMA_T_SENS)}
    ys = [int(y) for y in (ref["years"] if survey_years is None else survey_years)]
    for wid in windows:
        span = window_years(wid)
        doc["D_W"]["escalda"][wid] = {str(Y): window_mismatch(ref["Z"], ref["years"], span, Y,
                                                              s2_ref, min_cells) for Y in ys}
    for site, d in sites.items():
        prs = pair_variances(d["Z"], d["years"], d.get("mask"), min_cells)
        dts, Ds, ns = d_curve(prs)
        ov = d.get("overlap_sigma2_s")
        if ov is not None and np.isfinite(ov):
            s2, method = float(ov), "same-year overlapping sheets"
        else:
            s2, method = s2_ref, "Escalda's value (no same-year overlap; declared)"
        ratio = transfer_ratio((dts, Ds), (dt, D), s2, s2_ref)
        doc["sites"][site] = {"sigma2_s": s2, "method": method,
                              "curve": {"dt": dts.tolist(), "D": Ds.tolist(), "n": ns.tolist()},
                              "ratio": ratio}
        doc["D_W"][site] = {
            wid: {Y: dict(v, D_W=v["D_W"] * ratio, transferred_from="escalda")
                  for Y, v in tab.items()}
            for wid, tab in doc["D_W"]["escalda"].items()}
    if pnoa is not None:
        s2 = pnoa_noise(pnoa["rmse_m"], pnoa["random_share"], pnoa.get("n_agg", 25.0))
        doc["sites"]["villaviciosa"] = {"sigma2_s": s2, "method": "PNOA stated accuracy",
                                        "pnoa": dict(pnoa), "ratio": None}
    doc["sha256"] = config.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    return doc


def sigma2_T(model, site, wid=None, Y=None, mult=1.0):
    """sigma2_T(site, window, view) = mult * (sigma2_s + D_W(Y)) from a
    :func:`truth_error_model` document (D_W = 0 when the view has no
    epoch-mismatch entry, e.g. a PNOA or RTK view)."""
    s = model["sites"][site]["sigma2_s"]
    dw = 0.0
    if wid is not None and Y is not None:
        e = model.get("D_W", {}).get(site, {}).get(wid, {}).get(str(int(Y)))
        if e is not None and np.isfinite(e.get("D_W", np.nan)):
            dw = float(e["D_W"])
    return float(mult) * (float(s) + dw)


# ─────────────────────────────────────────────────────────────────────────────
#  v2: strata (§7.1; method-free, defined before any score)
# ─────────────────────────────────────────────────────────────────────────────

def along_s(s_km, s_nodes, v_nodes, v_mouth):
    """A judge quantity interpolated along s: linear between (0, v_mouth)
    and the judges' (s_g, v_g) in s order, constant beyond the last judge
    (NaN where s is not finite; judges with NaN are skipped)."""
    s = np.asarray(s_km, np.float64)
    sn = np.asarray(s_nodes, np.float64)
    vn = np.asarray(v_nodes, np.float64)
    ok = np.isfinite(sn) & np.isfinite(vn)
    sn, vn = sn[ok], vn[ok]
    o = np.argsort(sn, kind="stable")
    xs = np.concatenate([[0.0], sn[o]])
    vs = np.concatenate([[float(v_mouth)], vn[o]])
    out = np.full(s.shape, np.nan)
    fin = np.isfinite(s)
    out[fin] = np.interp(s[fin], xs, vs)
    return out


def reference_lag(s_km, s_nodes, tau_nodes):
    """tau_ref(s) (0 at the mouth) and its class (§7.1)."""
    t = along_s(s_km, s_nodes, tau_nodes, 0.0)
    return t, classify(t, LAG_EDGES_MIN)


def amplification(s_km, s_nodes, ratio_nodes):
    """The M2 amplitude ratio along s (1 at the mouth) and its class."""
    a = along_s(s_km, s_nodes, ratio_nodes, 1.0)
    return a, classify(a, AMP_EDGES)


def hydroperiod_zone(z, h_ref):
    """The hydroperiod zone of each truth elevation under ``h_ref`` (the
    10-min reference level): codes 1..5 of ``hydroperiod.zonation`` (1 =
    subtidal ... 5 = supratidal), 0 where z is not finite.

    Returns
    -------
    zone (int8), hp (the submerged share, float64)
    """
    z = np.asarray(z, np.float64)
    h = np.asarray(h_ref, np.float64)
    h = np.sort(h[np.isfinite(h)])
    if not len(h):
        raise ValueError("the reference level series is empty")
    hp = np.full(z.shape, np.nan)
    fin = np.isfinite(z)
    hp[fin] = 1.0 - np.searchsorted(h, z[fin], side="right") / len(h)
    zone = np.zeros(z.shape, np.int8)
    for code, (lo, hi, _) in enumerate(ZONES, start=1):
        zone[fin & (hp > lo) & (hp <= hi)] = code
    return zone, hp


def sampled_range_position(z, h_at_instants):
    """-1 below the lowest h_ref sampled at the window's S2 instants, 0
    inside, 1 above the highest, -2 where z is not finite."""
    z = np.asarray(z, np.float64)
    h = np.asarray(h_at_instants, np.float64)
    h = h[np.isfinite(h)]
    out = np.full(z.shape, -2, np.int8)
    if not len(h):
        return out
    fin = np.isfinite(z)
    out[fin] = 0
    out[fin & (z < h.min())] = -1
    out[fin & (z > h.max())] = 1
    return out


def slope_percent(z_frame, pixel_m, smooth_m=SLOPE_SMOOTH_M):
    """Truth slope (%) after a NaN-aware box smoothing over ``smooth_m``
    (normalised convolution; NaN where the pixel itself has no truth)."""
    from scipy import ndimage

    z = np.asarray(z_frame, np.float64)
    fin = np.isfinite(z)
    k = max(1, int(round(float(smooth_m) / float(pixel_m))))
    if k % 2 == 0:
        k += 1
    num = ndimage.uniform_filter(np.where(fin, z, 0.0), k, mode="constant")
    den = ndimage.uniform_filter(fin.astype(np.float64), k, mode="constant")
    zs = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    gy, gx = np.gradient(zs, float(pixel_m))
    s = 100.0 * np.hypot(gx, gy)
    return np.where(fin, s, np.nan).astype(np.float32)


def distance_to(mask_frame, pixel_m):
    """Euclidean distance (m) of every pixel to the nearest True pixel of
    ``mask_frame`` (inf when the mask is empty): the distance to permanent
    water (full-record water frequency >= 0.9)."""
    from scipy import ndimage

    m = np.asarray(mask_frame, bool)
    if not m.any():
        return np.full(m.shape, np.inf)
    return ndimage.distance_transform_edt(~m) * float(pixel_m)


def tercile_edges(values):
    """The per-site tercile limits (q 1/3, q 2/3) of the finite values."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return (float("nan"), float("nan"))
    a, b = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return (float(a), float(b))


def low_tide_positions(h_at_positions, q=LOW_TIDE_QUANTILE):
    """Record positions whose boundary level lies in the lowest quantile
    ``q`` (ties included): the scenes of the substrate and vegetation strata."""
    h = np.asarray(h_at_positions, np.float64)
    thr = np.nanquantile(h, float(q))
    return np.flatnonzero(np.isfinite(h) & (h <= thr)).astype(np.int64)


def lattice_row_bounds(H, block_rows, py):
    """Row-block bounds whose inner boundaries sit on 20 m lattice rows
    (r = py mod 2), so no 2 x 2 block straddles two row blocks."""
    step = max(2, int(block_rows) // 2 * 2)
    inner = range(int(py) % 2 + step, int(H), step)
    return [0] + [b for b in inner if 0 < b < H] + [int(H)]


def _nanmedian0(a):
    """``np.nanmedian`` over axis 0 without the all-NaN warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=0)


def low_tide_dry_medians(src, positions, setting="C1.floor0.ndwi.D1", block_rows=64,
                         lattice=None):
    """Per-pixel medians of the dry observations at low tide (§7.1):
    rho11 (B11 / 10000 under floor0, at 20 m by the lattice rule) and NDVI
    ((B08 - B04) / (B08 + B04), floor0) where the setting's V & ~W.

    Parameters
    ----------
    src : inputs.SceneSource
        Reads ``read_block``, ``offsets``, ``grid`` (H, W), ``lattice20``.
    positions : int array
        The low-tide record positions (:func:`low_tide_positions`),
        strictly increasing.
    setting : str
        The observation setting (P0 by default).
    block_rows : int
        Rows per block (memory: positions x rows x W float32 per band).
    lattice : (int, int), optional
        The 20 m phase (default ``src.lattice20``, else (0, 0)).

    Returns
    -------
    dict
        rho11, ndvi (H, W) float32 (NaN without a dry observation), n_dry
        (H, W) int32.
    """
    from experiments.benchmark import obs

    pos = np.asarray(positions, np.int64)
    H, W = int(src.grid.H), int(src.grid.W)
    lat = lattice if lattice is not None else (getattr(src, "lattice20", None) or (0, 0))
    rho = np.full((H, W), np.nan, np.float32)
    ndvi = np.full((H, W), np.nan, np.float32)
    ndry = np.zeros((H, W), np.int32)
    if not len(pos):
        return {"rho11": rho, "ndvi": ndvi, "n_dry": ndry}
    offs = np.asarray(src.offsets, np.int64)[pos]
    bounds = lattice_row_bounds(H, block_rows, lat[0])
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        raw = src.read_block(pos, ["B03", "B04", "B08", "B11", "SCL"], slice(r0, r1))
        ph = ((int(lat[0]) - r0) % 2, int(lat[1]))
        o = obs.observe_setting(raw, setting, offset=offs, lattice=ph)
        dry = np.asarray(o.V, bool) & ~np.asarray(o.W, bool)
        b11 = obs.prep_band(raw["B11"], offs, "floor0")
        b11 = obs.expand20(obs.block_mean20(b11, ph), ph, b11.shape[-2:]) / np.float32(1e4)
        b4 = obs.prep_band(raw["B04"], offs, "floor0")
        b8 = obs.prep_band(raw["B08"], offs, "floor0")
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = (b8 - b4) / (b8 + b4)
        rho[r0:r1] = _nanmedian0(np.where(dry, b11, np.nan))
        ndvi[r0:r1] = _nanmedian0(np.where(dry & np.isfinite(nd), nd, np.nan))
        ndry[r0:r1] = dry.sum(axis=0)
    return {"rho11": rho, "ndvi": ndvi, "n_dry": ndry}


def anchor_band_medians(src, positions, anchors_flat, anchor_band, clear, band="B04",
                        block_rows=256):
    """The turbidity value of each 3-km band (§7.1): the median over clear
    scenes of the per-scene median rho (``band`` / 10000, floor0, valid
    under C1) on the band's channel anchors.

    Parameters
    ----------
    positions : int array
        Record positions (strictly increasing).
    anchors_flat : int array
        Frame flat indices of the channel anchors (EMODnet depth >= 2 m
        below LAT).
    anchor_band : int array
        The 3-km band of each anchor.
    clear : bool array (len(positions),)
        The clear scenes.

    Returns
    -------
    dict ``{band id: median}`` (NaN for a band without clear observations).
    """
    from experiments.benchmark import obs

    pos = np.asarray(positions, np.int64)[np.asarray(clear, bool)]
    a = np.asarray(anchors_flat, np.int64)
    ab = np.asarray(anchor_band, np.int64)
    W = int(src.grid.W)
    H = int(src.grid.H)
    ub = [int(b) for b in np.unique(ab)]
    if not len(pos) or not len(a):
        return {b: float("nan") for b in ub}
    per = {b: [[] for _ in range(len(pos))] for b in ub}
    offs = np.asarray(src.offsets, np.int64)[pos]
    arow = a // W
    for r0 in range(0, H, int(block_rows)):
        r1 = min(H, r0 + int(block_rows))
        sel = (arow >= r0) & (arow < r1)
        if not sel.any():
            continue
        raw = src.read_block(pos, [band, "SCL"], slice(r0, r1))
        x = obs.prep_band(raw[band], offs, "floor0") / np.float32(1e4)
        valid = obs.pixel_valid("C1", raw["SCL"])
        x = np.where(valid, x, np.nan).reshape(len(pos), -1)
        loc = a[sel] - r0 * W
        for b in ub:
            m = ab[sel] == b
            if m.any():
                for i in range(len(pos)):
                    per[b][i].append(x[i, loc[m]])
    out = {}
    for b in ub:
        meds = []
        for vals in per[b]:
            v = np.concatenate(vals) if vals else np.empty(0)
            v = v[np.isfinite(v)]
            if len(v):
                meds.append(float(np.median(v)))
        out[b] = float(np.median(meds)) if meds else float("nan")
    return out


def strata_arrays(z, *, s_km=None, band=None, judges_s=None, judges_tau=None,
                  judges_ratio=None, h_ref=None, h_at_instants=None, slope_pct=None,
                  dist_water_m=None, rho11=None, ndvi=None, turbidity_by_band=None,
                  n_clear=None, epoch_gap=None, source_class=None, water_flag=None):
    """Every stratum of a set of truth points (pixels or cells), §7.1.

    Each input is optional: a stratum whose inputs are missing is left out,
    never guessed. Arrays are aligned with ``z``; none comes from a method.

    Returns
    -------
    dict
        Class arrays (int8, -1 = undefined): tau_ref_class, amp_class,
        zone, sampled_range, slope_class, substrate_class,
        vegetation_class, turbidity_class, n_clear_class, epoch_gap_class;
        the continuous values (tau_ref_min, amp_ratio, hp, slope_pct, ...);
        and ``edges`` (the class limits used, including per-site terciles).
    """
    z = np.asarray(z, np.float64)
    out = {"edges": {}}
    if s_km is not None:
        out["s_km"] = np.asarray(s_km, np.float64)
    if band is not None:
        out["band"] = np.asarray(band)
    if s_km is not None and judges_s is not None and len(judges_s):
        if judges_tau is not None:
            out["tau_ref_min"], out["tau_ref_class"] = reference_lag(s_km, judges_s, judges_tau)
            out["edges"]["tau_ref_class"] = list(LAG_EDGES_MIN)
        if judges_ratio is not None:
            out["amp_ratio"], out["amp_class"] = amplification(s_km, judges_s, judges_ratio)
            out["edges"]["amp_class"] = list(AMP_EDGES)
    if h_ref is not None:
        out["zone"], out["hp"] = hydroperiod_zone(z, h_ref)
        out["edges"]["zone"] = [[lo, hi, nm] for lo, hi, nm in ZONES]
    if h_at_instants is not None:
        out["sampled_range"] = sampled_range_position(z, h_at_instants)
    if slope_pct is not None:
        out["slope_pct"] = np.asarray(slope_pct, np.float64)
        out["slope_class"] = classify(slope_pct, SLOPE_EDGES_PCT)
        out["edges"]["slope_class"] = list(SLOPE_EDGES_PCT)
    if dist_water_m is not None:
        out["dist_permanent_water_m"] = np.asarray(dist_water_m, np.float64)
    if rho11 is not None:
        e = tercile_edges(rho11)
        out["substrate_class"] = classify(rho11, e)
        out["edges"]["substrate_class"] = list(e)
    if ndvi is not None:
        out["vegetation_class"] = classify(ndvi, NDVI_EDGES)
        out["edges"]["vegetation_class"] = list(NDVI_EDGES)
    if turbidity_by_band is not None and band is not None:
        e = tercile_edges(list(turbidity_by_band.values()))
        per = np.array([turbidity_by_band.get(int(b), np.nan) for b in np.asarray(band)],
                       np.float64)
        out["turbidity_class"] = classify(per, e)
        out["edges"]["turbidity_class"] = list(e)
    if n_clear is not None:
        nc = np.asarray(n_clear, np.float64)
        e = tercile_edges(np.where(np.isfinite(z), nc, np.nan))
        out["n_clear_class"] = classify(nc, e)
        out["edges"]["n_clear_class"] = list(e)
    if epoch_gap is not None:
        out["epoch_gap_y"] = np.asarray(epoch_gap, np.float64)
        out["epoch_gap_class"] = classify(epoch_gap, EPOCH_GAP_EDGES_Y)
        out["edges"]["epoch_gap_class"] = list(EPOCH_GAP_EDGES_Y)
    if source_class is not None:
        out["source_class"] = np.asarray(source_class)
    if water_flag is not None:
        out["lidar_water_flag"] = np.asarray(water_flag, bool)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  v2: the elevation truth of a site (§1.8, §7.1)
# ─────────────────────────────────────────────────────────────────────────────

#: the Campaign-1 windows an elevation unit is scored in (designs: base and
#: levels at W3, native cells and LM rows at W1, calendar years): D_truth,
#: hence S0 and dom, is the union over all of them
ELEV_WINDOWS_1A = ("W3", "W1", "Y2023", "Y2024", "Y2025")
#: the Campaign-1 windows of the lag units (the lag design at W1, W2, W3,
#: Y2023, Y2024; the calendar-year designs at Y2023-Y2025)
LAG_WINDOWS_1A = ("W3", "W2", "W1", "Y2023", "Y2024", "Y2025")


def build_elevation_truth_v2(pkg, windows=ELEV_WINDOWS_1A, h1=None, table=None, sheets=None,
                             src_dates=None, pnoa=None, out=None, force=False):
    """truth/<site>/truth_elev_v2.{npz,json} (§1.8, §7.1).

    At the Dutch sites the support is the 20 m cell: every frame pixel is
    mapped to its cell, every cell keeps all its surveys of 2016-2025, and
    per window the survey-matched truth (range-filtered with the fixed
    bounds), its year, its epoch gap and the stable flag are stored per
    cell, next to v13's latest-survey view. At Villaviciosa the v2 PNOA
    truth is stored per pixel, and the RTK development points with their
    campaign epoch.

    Parameters
    ----------
    pkg : CommonPackage
        Reads ``cfg``, the frame shape and grid, ``full.keep`` and the strata
        bands.
    windows : sequence of str
        The registered windows (``settings.WINDOW`` ids; default every
        Campaign-1 elevation window, :data:`ELEV_WINDOWS_1A`).
    h1 : array
        The frozen 1-min h_B over the union period (the range filter).
    table : list of dict, optional
        The judge table (neighbourhoods; default the truth copy, if any).
    sheets : mapping, optional
        ``{name: path or read_sheet_epochs dict}`` (default the site's).
    src_dates : list, optional
        :func:`dated_epochs`' per-sheet date tables (``fetch vak-meta``).
    pnoa : dict, optional
        Villaviciosa: ``flight_level`` (None = unknown) and optionally
        ``grid_path`` and ``mdt_path``.
    out : str, optional
        Output directory (default the site's truth directory).
    force : bool

    Returns
    -------
    str
        The path of truth_elev_v2.json.
    """
    t0 = time.time()
    cfg = pkg.cfg
    out = out or truth_dir(pkg)
    jpath = f"{out}/truth_elev_v2.json"
    if h1 is None:
        raise ValueError("the range filter needs the frozen 1-min h_B (h1)")
    rng = range_from_series(h1)
    kinds = tuple(cfg.elevation_truth)
    H, W = judges.frame_shape(pkg)
    grid = judges.grid_of(pkg)
    keep = np.asarray(pkg.full.keep, np.int64)
    if table is None:
        jt = f"{truth_dir(pkg)}/judges.json"
        table = judges.load(jt, grid) if os.path.exists(jt) else []
    nbhds = [r["nbhd"] for r in table]
    sheet_ids = None
    if sheets is not None:
        sheet_ids = [v if isinstance(v, str) else v.get("path", k) for k, v in sheets.items()]
    key = config.sha256_json({"package_key": _package_key(pkg), "windows": list(windows),
                              "range": rng, "truth_v2": truth_v2_sha(), "code": _code_shas(),
                              "pnoa": pnoa, "sheets": sheet_ids, "src_dates": src_dates,
                              "nbhd": [config.sha256_bytes(np.asarray(n, "<i8").tobytes())
                                       for n in nbhds],
                              "format": TRUTH_ELEV_V2_FORMAT})
    if not force and _done(jpath, key):
        log(f"[truth] {cfg.site}: v2 elevation truth up to date ({jpath})")
        return jpath
    arrays = {"format": np.array(TRUTH_ELEV_V2_FORMAT), "shape": np.array([H, W], np.int64),
              "lo": np.float64(rng["lo"]), "hi": np.float64(rng["hi"])}
    doc = {"format": TRUTH_ELEV_V2_FORMAT, "inputs_key": key, "site": cfg.site,
           "res_m": int(cfg.res_m), "kinds": list(kinds), "windows": list(windows),
           "range": rng, "truth_v2_sha256": truth_v2_sha(), "views": {}}
    d_union = np.empty(0, np.int64)
    C = None
    if "vaklodingen" in kinds:
        sheets = vaklodingen_sheets(cfg.site) if sheets is None else sheets
        XQ, YQ = rd_coords(grid, (H, W))
        C = vaklodingen_cells(XQ, YQ, sheets)
        del XQ, YQ
        nc = len(C["cell_id"])
        pc = C["point_cell"]
        arrays.update({"years": C["years"], "Z": C["Z"], "SRC": C["SRC"],
                       "cell_id": C["cell_id"], "cell_sheet": C["sheet"],
                       "cell_iy": C["iy"], "cell_ix": C["ix"], "cell_x_rd": C["x_rd"],
                       "cell_y_rd": C["y_rd"], "cell_n_px": C["n_px"],
                       "sheets": np.array(C["sheets"], dtype=str)})
        ep = dated_epochs(C["SRC"], C["years"], C["sheet"], src_dates) if src_dates else None
        cell_in = np.zeros(nc, bool)
        for wid in windows:
            w = window_of(wid)
            z, yr, gap = survey_matched(C["Z"], C["years"], w.centre)
            tr, counts = apply_range(z, rng["lo"], rng["hi"])
            st = stable_cells(C["Z"], C["years"], window_years(wid))
            arrays[f"{wid}.truth"] = tr
            arrays[f"{wid}.year"] = yr
            arrays[f"{wid}.gap"] = gap
            arrays[f"{wid}.stable"] = st["stable"]
            if ep is not None:
                ok = yr >= 0
                jj = np.searchsorted(C["years"], yr[ok])
                dg = np.full(nc, np.nan)
                dg[ok] = np.abs(ep[np.flatnonzero(ok), jj] - w.centre)
                arrays[f"{wid}.gap_dated"] = dg
            fin = np.isfinite(tr)
            cell_in |= fin
            u, c = np.unique(yr[fin], return_counts=True)
            doc["views"][wid] = {"centre": w.centre, "n_cells_D_truth": int(fin.sum()),
                                 "range_counts": counts,
                                 "years": {int(a): int(b) for a, b in zip(u, c)},
                                 "n_stable": int((st["stable"] & fin).sum()),
                                 "stable_years": st["years"],
                                 "gap_mean_y": float(np.mean(gap[fin])) if fin.any() else None}
        zl, yl = latest_survey(C["Z"], C["years"])
        trl, cl = apply_range(zl, rng["lo"], rng["hi"])
        arrays["latest.truth"], arrays["latest.year"] = trl, yl
        doc["views"]["latest_2019_2025"] = {"n_cells_D_truth": int(np.isfinite(trl).sum()),
                                            "range_counts": cl}
        arrays["cell_in_D_union"] = cell_in
        d_union = np.flatnonzero((pc >= 0) & cell_in[np.maximum(pc, 0)]).astype(np.int64)
        doc.update(n_cells=int(nc), n_cells_D_union=int(cell_in.sum()),
                   years_kept=[int(y) for y in C["years"]], sheets=C["sheets"],
                   cell_px={str(int(k)): int(v) for k, v in
                            zip(*np.unique(C["n_px"], return_counts=True))})
    elif "lidar" in kinds:
        p = dict(pnoa or {})
        fr = pnoa_truth(p.get("grid_path"), p.get("mdt_path"), p.get("flight_level"))
        if fr["z_raw"].shape != (H, W):
            raise AssertionError(f"PNOA truth shape {fr['z_raw'].shape} differs from {(H, W)}")
        tr, counts = apply_range(fr["z_raw"], rng["lo"], rng["hi"])
        d_union = np.flatnonzero(np.isfinite(tr.ravel())).astype(np.int64)
        arrays["pixel_flat"] = d_union
        arrays["pixel_truth"] = tr.ravel()[d_union]
        doc["pnoa"] = {k: v for k, v in fr.items() if k != "z_raw"}
        doc["views"]["pixels"] = {"range_counts": counts, "n_D_truth": int(len(d_union))}
    rtk_doc, rtk_flat = None, None
    if "rtk" in kinds:
        rk = rtk_points(pkg)
        rtk_flat = rk["flat"]
        arrays.update({"rtk_flat": rk["flat"], "rtk_z": rk["z"],
                       "rtk_block150": rk["block150"]})
        try:
            rtk_doc = rtk_epoch()
        except (OSError, ValueError, KeyError) as e:
            rtk_doc = {"error": f"{type(e).__name__}: {e}"}
        rtk_doc.update(n_points=int(len(rk["flat"])), datum="Alicante (elev_alicante)",
                       range_filtered=False)
    S = support(d_union, keep, nbhds, rtk_flat)
    arrays["support"] = S
    arrays["d_truth_union"] = d_union
    if C is not None:
        arrays["support_cell"] = C["point_cell"][S]
    arrays["band"] = strata_bands(pkg, S)
    os.makedirs(out, exist_ok=True)
    product.save_npz(f"{out}/truth_elev_v2.npz", arrays, compress=True)
    doc.update(n_S=int(len(S)), n_D_union_px=int(len(d_union)), rtk=rtk_doc,
               support_sha256=config.sha256_bytes(S.astype("<i8").tobytes()),
               package_key=_package_key(pkg), code=_code_shas(),
               built_utc=product.utc_stamp(), seconds=round(time.time() - t0, 1))
    product.write_json(jpath, doc)
    log(f"[truth] {cfg.site}: v2 elevation truth, range [{rng['lo']:.3f}, {rng['hi']:.3f}] m, "
        f"{len(d_union):,} D_truth px (union over {list(windows)}), support {len(S):,} px "
        f"[{doc['seconds']} s] -> {jpath}")
    return jpath


def load_elevation_truth_v2(pkg_or_cfg, out=None):
    """``(arrays, doc)`` of a site's v2 elevation truth."""
    cfg = getattr(pkg_or_cfg, "cfg", pkg_or_cfg)
    d = out or cfg.truth_dir
    arrays = product.load_npz(f"{d}/truth_elev_v2.npz")
    if str(arrays.get("format")) != TRUTH_ELEV_V2_FORMAT:
        raise ValueError(f"{d}/truth_elev_v2.npz: not a {TRUTH_ELEV_V2_FORMAT} file")
    return arrays, product.read_json(f"{d}/truth_elev_v2.json")


# ─────────────────────────────────────────────────────────────────────────────
#  v2: per-window lag truths (§6.2)
# ─────────────────────────────────────────────────────────────────────────────

def window_positions(t, wid):
    """Record positions of the instants ``t`` inside the window [start, end)."""
    w = window_of(wid)
    t = pd.DatetimeIndex(t)
    m = (t >= pd.Timestamp(w.start)) & (t < pd.Timestamp(w.end))
    return np.flatnonzero(m).astype(np.int64)


def window_boot_seed(site, code, wid):
    """The bootstrap seed of one judge and window:
    [20261011, SITE_ID, crc32(code), window_num]."""
    from experiments.benchmark import settings

    return [BOOT_SEED, int(config.SITE_ID[site]), int(zlib.crc32(str(code).encode("utf-8"))),
            int(settings.window_num(wid))]


def _jack_labels(idx, group):
    idx = pd.DatetimeIndex(idx)
    if group == "month":
        return (idx.year * 12 + idx.month - 1).to_numpy()
    return (idx.year * 4 + (idx.month - 1) // 3).to_numpy()


def m2_lag(idx10, h10, g10, constituents=None, group="quarter"):
    """Truth (b) on arrays: the M2 phase lag of a centred 10-min gauge
    relative to the 10-min h_B on their common finite instants, the
    delete-one-group jackknife (calendar quarters; months for 1-year
    windows) and the M2 amplitude ratio gauge / boundary.

    Returns
    -------
    dict
        tau, se, lo, hi, n_samples, span_days, n_groups, jackknife (list),
        A_M2_gauge, A_M2_B, amp_ratio, phases, flags.
    """
    cons = list(CONS_B if constituents is None else constituents)
    idx10 = pd.DatetimeIndex(idx10)
    h10 = np.asarray(h10, np.float64)
    g = np.asarray(g10, np.float64)
    if len(g) != len(idx10) or len(h10) != len(idx10):
        raise ValueError("the gauge, h_B and the 10-min grid must be aligned")
    m = np.isfinite(g) & np.isfinite(h10)
    nan = float("nan")
    out = {"n_samples": int(m.sum()), "group": group, "tau": nan, "se": nan, "lo": nan,
           "hi": nan, "amp_ratio": nan, "flags": []}
    if not m.any():
        out["flags"].append("no_samples")
        return out
    td = days_since_1992(idx10[m])
    out["span_days"] = float(td.max() - td.min())
    fg = harmonic_fit(td, g[m], cons)
    fb = harmonic_fit(td, h10[m], cons)
    if fg is None or fb is None:
        out["flags"].append("too_few_samples")
        return out
    dphi = float(wrap180(fg["m2"][1] - fb["m2"][1]))
    tau = dphi * MIN_PER_DEG_M2
    out.update(tau=tau, phase_gauge_deg=fg["m2"][1], phase_B_deg=fb["m2"][1],
               A_M2_gauge=fg["m2"][0], A_M2_B=fb["m2"][0],
               amp_ratio=fg["m2"][0] / fb["m2"][0] if fb["m2"][0] else nan)
    lab = _jack_labels(idx10[m], group)
    jack = []
    for gq in np.unique(lab):
        k = lab != gq
        a_ = harmonic_fit(td[k], g[m][k], cons)
        b_ = harmonic_fit(td[k], h10[m][k], cons)
        if a_ is None or b_ is None:
            jack.append(nan)
            continue
        d = float(wrap180(wrap180(a_["m2"][1] - b_["m2"][1]) - dphi))
        jack.append(tau + d * MIN_PER_DEG_M2)
    jack = np.asarray(jack, float)
    fin = jack[np.isfinite(jack)]
    G = len(fin)
    out["n_groups"] = int(len(jack))
    out["jackknife"] = jack.tolist()
    if G >= 2:
        se = float(np.sqrt((G - 1) / G * np.sum((fin - fin.mean()) ** 2)))
        out.update(se=se, lo=tau - Z_B * se, hi=tau + Z_B * se)
    else:
        out["flags"].append("no_jackknife")
    return out


def hw_lw_window(series, marks_b, start, end, width=None, max_min=MARK_MATCH_MIN):
    """HW and LW lags of a gauge in a window (§6.2).

    The gauge is interpolated to 1 min (gaps <= 20 min bridged), low-passed
    with the boundary's centred running mean (``levels.lowpass``, 61 min),
    and its ``g21.tide_marks`` in [start, end) are paired with the nearest
    boundary mark of the same kind (``marks_lp``) within 3 h.

    Returns
    -------
    dict
        hw / lw {median_min, iqr_min, mean_min, n}, mean_min (the mean of
        the two medians), distortion_min (LW - HW), lowpass_width.
    """
    from pyintertidal import granadeiro2021 as g21

    width = levels.LOWPASS_WIDTH if width is None else int(width)
    nan = float("nan")
    empty = {"median_min": nan, "iqr_min": nan, "mean_min": nan, "n": 0}
    res = {"hw": dict(empty), "lw": dict(empty), "mean_min": nan, "distortion_min": nan,
           "lowpass_width": width}
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    pad = pd.Timedelta(hours=12)
    grid = pd.date_range(t0 - pad, t1 + pad, freq="1min", inclusive="left").as_unit("ns")
    gap = pd.Timedelta(minutes=GAP_INTERP_MIN)
    s = series[(series.index >= grid[0] - gap) & (series.index <= grid[-1] + gap)]
    if len(s) < 3:
        return res
    lp = levels.lowpass(interp_at(s, grid), width)
    if np.isfinite(lp).sum() < 3:
        return res
    try:
        mg = g21.tide_marks(levels.minutes_since_epoch(grid), lp)
    except ValueError:
        return res
    lo_m = float(levels.minutes_since_epoch(pd.DatetimeIndex([t0]))[0])
    hi_m = float(levels.minutes_since_epoch(pd.DatetimeIndex([t1]))[0])
    tg = np.asarray(mg["t"], float)
    hwg = np.asarray(mg["is_hw"], bool)
    inw = (tg >= lo_m) & (tg < hi_m)
    tb = np.asarray(marks_b["t"], float)
    hwb = np.asarray(marks_b["is_hw"], bool)
    for name, flag in (("hw", True), ("lw", False)):
        d = _pair_marks(tg[inw & (hwg == flag)], tb[hwb == flag], max_min)
        if len(d):
            q1, q3 = np.percentile(d, [25, 75])
            res[name] = {"median_min": float(np.median(d)), "iqr_min": float(q3 - q1),
                         "mean_min": float(np.mean(d)), "n": int(len(d))}
    if res["hw"]["n"] and res["lw"]["n"]:
        res["mean_min"] = 0.5 * (res["hw"]["median_min"] + res["lw"]["median_min"])
        res["distortion_min"] = res["lw"]["median_min"] - res["hw"]["median_min"]
    return res


def tau_all_window(idx10, g10, t1, h1, taus=TAUS_A):
    """tau_all on a window's 10-min samples, h_B(t - tau) read from the
    exact 1-min series ``(t1, h1)`` (minutes since the epoch)."""
    m10 = levels.minutes_since_epoch(idx10)
    t1 = np.asarray(t1, np.float64)
    h1 = np.asarray(h1, np.float64)
    p = np.searchsorted(t1, m10)
    if (p >= len(t1)).any() or not np.array_equal(t1[np.minimum(p, len(t1) - 1)], m10):
        raise ValueError("the 10-min grid is not on the 1-min boundary series")
    g = np.asarray(g10, np.float64)
    fg = np.isfinite(g)
    E = np.full(len(taus), np.nan)
    for k, tau in enumerate(np.asarray(taus, np.int64)):
        q = p - int(tau)
        valid = (q >= 0) & (q < len(h1))
        hb = np.full(len(g), np.nan)
        hb[valid] = h1[q[valid]]
        mm = fg & np.isfinite(hb)
        if mm.any():
            d = hb[mm] - g[mm]
            E[k] = np.sqrt(np.mean((d - np.median(d)) ** 2))
    res = best_lag(taus, E)
    k0 = int(np.flatnonzero(np.asarray(taus) == 0)[0])
    res["E0_m"] = float(E[k0])
    return res, E


def lag_truths_window(pkg, rec, series, wid, H=None, positions=None, constituents=None,
                      boot=True, conventions=None, shift_min=0.0):
    """Every lag truth of one judge in one window (§6.2).

    Parameters
    ----------
    pkg : CommonPackage
        ``t``, ``levels``, ``series1``, ``series10``, ``marks_lp``, ``cfg``.
    rec : dict
        A judge-table (or stations.json) record: code, sampling_min,
        provider.
    series : pd.Series
        The gauge as delivered (:func:`judges.read_series`); it is centred
        with its provider's time-stamp convention first.
    wid : str
        The window; its record = the package instants in [start, end).
    H : (len(TAUS_A), T) array, optional
        ``pkg.levels.stack(TAUS_A)`` (shared between judges).
    positions : int array, optional
        Replace the window's record (e.g. a scene-rule record).
    conventions : mapping, optional
        Time-stamp conventions (default :data:`judges.TIMESTAMP_CONVENTIONS`).
    shift_min : float
        The declared timing sensitivity (+- half the averaging window).

    Returns
    -------
    (dict, curves)
    """
    t_all = record_t(pkg)
    pos = window_positions(t_all, wid) if positions is None else np.asarray(positions, np.int64)
    w = window_of(wid)
    series = judges.centred(series, rec, conventions, shift_min=shift_min)
    if H is None:
        H = pkg.levels.stack(TAUS_A)
    t = t_all[pos]
    Hw = np.asarray(H)[:, pos]
    g_s2 = judges.at_instants(series, t, TOL_MIN)
    E = e_rows(Hw, g_s2)
    a = best_lag(TAUS_A, E)
    k0 = int(np.flatnonzero(TAUS_A == 0)[0])
    a.update(n_instants=int(np.isfinite(g_s2).sum()), n_record=int(len(t)),
             coverage=float(np.isfinite(g_s2).mean()) if len(t) else 0.0, E0_m=float(E[k0]))
    curves = {"E_a": E}
    if boot and len(t):
        seed = window_boot_seed(pkg.cfg.site, rec["code"], wid)
        bt = month_boot(Hw, g_s2, t, rng=np.random.default_rng(seed))
        fin = bt[np.isfinite(bt)]
        a["boot"] = {"n": int(len(fin)), "seed": seed,
                     "ci_lo": float(np.percentile(fin, 2.5)) if len(fin) else float("nan"),
                     "ci_hi": float(np.percentile(fin, 97.5)) if len(fin) else float("nan"),
                     "sd": float(np.std(fin)) if len(fin) else float("nan")}
        curves["boot_tau"] = bt
    Eab, B = gain_aware(Hw, g_s2)
    ab = best_lag(TAUS_A, Eab)
    kb = int(np.nanargmin(Eab)) if np.isfinite(Eab).any() else None
    ab["gain"] = float(B[kb]) if kb is not None else float("nan")
    ab["gain_edge"] = bool(kb is not None and (B[kb] <= GAIN_GRID[0] or B[kb] >= GAIN_GRID[-1]))
    curves.update(E_ab=Eab, gain_ab=B)
    idx10, h10 = series10(pkg)
    m10 = (idx10 >= pd.Timestamp(w.start)) & (idx10 < pd.Timestamp(w.end))
    idx_w, h_w = idx10[m10], h10[m10]
    g10 = judges.centred_10min(series, rec["sampling_min"], grid=idx_w).to_numpy(np.float64)
    b = m2_lag(idx_w, h_w, g10, constituents=constituents,
               group="month" if len(w.years) == 1 else "quarter")
    curves["jack_tau"] = np.asarray(b.pop("jackknife", []), float)
    try:
        t1, h1 = series1(pkg)
        al, E_all = tau_all_window(idx_w, g10, t1, h1)
        curves["E_all"] = E_all
    except (AttributeError, ValueError) as e:
        al = {"tau": float("nan"), "error": f"{type(e).__name__}: {e}"}
    try:
        hl = hw_lw_window(series, marks_lp(pkg), w.start, w.end)
    except AttributeError as e:
        hl = {"error": f"{type(e).__name__}: {e}"}
    res = {"code": rec["code"], "window": wid, "start": w.start, "end": w.end,
           "shift_min": float(shift_min), "a": a, "ab": ab, "b": b, "all": al, "hw_lw": hl,
           "amp_ratio": b.get("amp_ratio", float("nan"))}
    return res, curves


LAG_V2_CSV_COLUMNS = ("site", "judge", "location", "window", "n_instants", "tau_a", "I_a_lo",
                      "I_a_hi", "tau_ab", "gain_ab", "tau_b", "se_tau_b", "tau_all", "hw_min",
                      "lw_min", "hwlw_mean_min", "distortion_min", "amp_ratio")


def build_lag_truth_v2(pkg, windows=LAG_WINDOWS_1A, table=None,
                       constituents=None, out=None, force=False, conventions=None):
    """truth/<site>/truth_lag_v2.json, truth_lag_v2_curves.npz and
    lag_truth_v2.csv (§6.2).

    The judge inclusion is re-evaluated per window (the coverage and dry
    rules on the window's record, :func:`judges.window_inclusion`, on the
    series centred with its provider's time-stamp convention, as the
    truths read it); a window where a judge fails is recorded with its
    exclusion and no truth. Every included judge gets (a), tau_ab, (b),
    tau_all, HW/LW and the amplitude ratio in each window (default every
    Campaign-1 lag window, :data:`LAG_WINDOWS_1A`).

    Returns
    -------
    str
        The path of truth_lag_v2.json.
    """
    t0 = time.time()
    cfg = pkg.cfg
    out = out or truth_dir(pkg)
    jpath = f"{out}/truth_lag_v2.json"
    jt = f"{truth_dir(pkg)}/judges.json"
    if table is None:
        table = judges.load(jt, judges.grid_of(pkg))
    table_doc = judges.read(jt) if os.path.exists(jt) else {}
    key = config.sha256_json({"base": lag_inputs_key(pkg, table_doc, ()),
                              "windows": list(windows), "truth_v2": truth_v2_sha("truth_lag"),
                              "table": [(r["code"], r.get("included"), r.get("in_frame"),
                                         r.get("nearest_member_km")) for r in table],
                              "constituents": None if constituents is None
                              else list(constituents), "format": TRUTH_LAG_V2_FORMAT,
                              "conventions": judges.conventions_sha(conventions)})
    if not force and _done(jpath, key):
        log(f"[truth] {cfg.site}: v2 lag truth up to date ({jpath})")
        return jpath
    H = pkg.levels.stack(TAUS_A) if table else None
    t_all = record_t(pkg)
    stations, curves, rows = [], {"taus": TAUS_A}, []
    for k_rec, rec in enumerate(table):
        series = judges.read_series(rec)
        st = {"code": rec["code"], "location": rec.get("location"),
              "s_g_km": rec.get("s_g_km"), "mouth": rec.get("mouth"), "windows": {}}
        for wid in windows:
            tw = time.time()
            pos = window_positions(t_all, wid)
            inc = judges.window_inclusion(rec, series, t_all[pos], wid,
                                          conventions=conventions or "default")
            entry = {"inclusion": inc}
            if inc["included"]:
                r, cv = lag_truths_window(pkg, rec, series, wid, H=H, positions=pos,
                                          constituents=constituents, conventions=conventions)
                entry.update(r)
                for name, v in cv.items():
                    curves[f"{rec['code']}__{wid}__{name}"] = np.asarray(v)
                hl = r["hw_lw"]
                rows.append({"site": cfg.site, "judge": rec["code"],
                             "location": rec.get("location"), "window": wid,
                             "n_instants": r["a"]["n_instants"], "tau_a": r["a"]["tau"],
                             "I_a_lo": r["a"]["lo"], "I_a_hi": r["a"]["hi"],
                             "tau_ab": r["ab"]["tau"], "gain_ab": r["ab"]["gain"],
                             "tau_b": r["b"]["tau"], "se_tau_b": r["b"]["se"],
                             "tau_all": r["all"].get("tau"),
                             "hw_min": hl.get("hw", {}).get("median_min"),
                             "lw_min": hl.get("lw", {}).get("median_min"),
                             "hwlw_mean_min": hl.get("mean_min"),
                             "distortion_min": hl.get("distortion_min"),
                             "amp_ratio": r["amp_ratio"]})
            st["windows"][wid] = entry
            log(f"  [truth lag] {cfg.site} {rec['code']} ({k_rec + 1}/{len(table)}) {wid}: "
                + (f"tau_a {entry['a']['tau']}, tau_b {entry['b'].get('tau')}"
                   if inc["included"] else f"excluded {inc.get('exclusion')}")
                + f" ({time.time() - tw:.0f} s)")
        stations.append(st)
    os.makedirs(out, exist_ok=True)
    product.save_npz(f"{out}/truth_lag_v2_curves.npz", curves, compress=True)
    pd.DataFrame(rows, columns=list(LAG_V2_CSV_COLUMNS)).to_csv(f"{out}/lag_truth_v2.csv",
                                                                index=False)
    doc = {"format": TRUTH_LAG_V2_FORMAT, "inputs_key": key, "site": cfg.site,
           "windows": list(windows), "rules": truth_v2_plain("truth_lag"),
           "conventions": judges.conventions_doc(conventions), "stations": stations,
           "development_judges": judges.development_list(),
           "code": _code_shas(), "built_utc": product.utc_stamp(),
           "seconds": round(time.time() - t0, 1)}
    product.write_json(jpath, doc)
    log(f"[truth] {cfg.site}: v2 lag truths of {len(table)} judges x {len(windows)} windows "
        f"[{doc['seconds']} s] -> {jpath}")
    return jpath


def load_lag_truth_v2(pkg_or_cfg, out=None):
    """The v2 lag truth document of a site (truth_lag_v2.json)."""
    cfg = getattr(pkg_or_cfg, "cfg", pkg_or_cfg)
    d = out or cfg.truth_dir
    doc = product.read_json(f"{d}/truth_lag_v2.json")
    if doc.get("format") != TRUTH_LAG_V2_FORMAT:
        raise ValueError(f"{d}/truth_lag_v2.json: not a {TRUTH_LAG_V2_FORMAT} file")
    return doc


def judge_nodes(lag_doc, wid, truth="a"):
    """``(s_g, tau, amp_ratio)`` of the included judges of a window, one
    value per location (the mean over its gauges): the inputs of the
    reference-lag and amplification strata."""
    by = {}
    for st in lag_doc["stations"]:
        e = st["windows"].get(wid, {})
        if not e.get("inclusion", {}).get("included"):
            continue
        loc = st.get("location") or st["code"]
        s = st.get("s_g_km")
        by.setdefault(loc, []).append((np.nan if s is None else s,
                                       e.get(truth, {}).get("tau", np.nan),
                                       e.get("amp_ratio", np.nan)))
    s, t, r = [], [], []
    for loc in sorted(by):
        a = np.asarray(by[loc], float)
        s.append(float(np.nanmean(a[:, 0])))
        t.append(float(np.nanmean(a[:, 1])))
        r.append(float(np.nanmean(a[:, 2])))
    return np.asarray(s), np.asarray(t), np.asarray(r)
