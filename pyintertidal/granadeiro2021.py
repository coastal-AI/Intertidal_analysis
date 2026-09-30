"""
granadeiro2021.py -- a faithful reimplementation of Granadeiro et al. (2021)
============================================================================

Granadeiro, J.P.; Belo, J.; Henriques, M.; Catalao, J.; Catry, T. (2021).
*Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and
Exposure Periods over Large Intertidal Areas.* Remote Sensing 13(2):320,
doi:10.3390/rs13020320 (pages below are those of the published PDF).

The paper's recipe, step by step, with the R packages it names
(:mod:`pyintertidal.r` ``granadeiro_fit.R`` runs every one of them):

1.  Imagery: green (B03) and NIR (B08) reflectance, 10 m (p.4 section 2.2).
    ONE declared exception: Sentinel-2 L2A (Sen2Cor) reflectance from the
    project's own cubes instead of ACOLITE, at the cube's resolution.
2.  Scenes: "a cloud cover lower than 10%" (p.4); every passing scene used.
3.  Inter-calibration (p.4, Fig. 2): one reference scene "informally selected
    to have an average reflectance range in the green and NIR bands";
    band-wise major-axis regressions (lmodel2) of every other scene on it,
    fitted on pixels with NIR < 0.05 (sea) or > 0.2 (land) outside the
    intertidal area; each scene corrected with the coefficients
    (x_cal = (x - intercept) / slope, as the co-author's
    Calibration_satimg_DEM.R: ``x*(1/slope)-inter/slope``).
4.  Water height at each scene (p.4, Eq. 1): cosine interpolation between
    the bracketing high- and low-water marks of ONE reference location
    (a tide table in the paper):
    ``h = hHW - (hHW - hLW) * (cos(pi (T - TLW) / (THW - TLW)) + 1) / 2``.
5.  Intertidal area (p.4-5, Eqs. 2-3; p.10): NDWI = (G - NIR) / (G + NIR);
    temporal population SD over all scenes (Eq. 3, 1/M); SD > 0.2.
6.  Pixel height (p.6, Eq. 4): 4-parameter logistic of the standardised
    NIR against the water heights, fitted with nplr (weighted least squares,
    nlm Newton-type minimiser); the height is the inflection; a pixel whose
    fit does not converge is outside the intertidal zone.  Run on every
    intertidal pixel -> preliminary DEM.
7.  Tidal-stage lags (p.7, Figs. 4-5): pixels whose preliminary height is
    within (mean water height) +- 0.25 m ("2.47 +- 0.25 m"); 50,000 random
    ones; lag grid -90..+90 min every 5 min; at each lag the heights of all
    scenes are recomputed from Eq. 1 at (t - lag) and the logistic is
    fitted separately to the rising-tide and to the ebbing-tide scenes; the
    lag is the one with the smallest |h_rising - h_ebbing|.
8.  Lag surface (p.7-8): mgcv GAM of the lags on longitude and latitude,
    penalised thin-plate regression splines, smoothing by GCV.
9.  Final DEM (p.8 section 2.7): heights re-estimated with Eq. 1 at each
    pixel's own lag.
10. Exposure (p.8-9, Eq. 5):
    ``E = C * (1 - acos(2 (h - hLW) / (hHW - hLW) - 1) / pi)`` with the mean
    cycle C and the mean HW / LW marks of the scene set.

Where the paper leaves a step open, the decisions of the reimplementation
are frozen in ``experiments/v8_granadeiro2021.py`` (``FROZEN``, keyed by
the step numbers of the fidelity audit) and implemented here as the
default constants.  Nothing is subsampled: every masked pixel is fitted,
the lag sample is min(50,000, pool) and every calibration pixel is used.

Exchange with R is through raw float64 little-endian files written with
``numpy.tofile`` and read with ``readBin`` (no R/Python bridge package).
"""
from __future__ import annotations

import heapq
import os
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  constants (paper page / equation; audit step in brackets)
# ─────────────────────────────────────────────────────────────────────────────

PAPER = ("Granadeiro, Belo, Henriques, Catalao & Catry 2021, Remote Sens. "
         "13(2):320, doi:10.3390/rs13020320")

#: [5] SCL classes counted as cloud in the scene rule (cloud shadow, cloud
#: medium / high probability, thin cirrus) = pyintertidal.water.BAD_CLASSES
SCENE_CLOUD_SCL = (3, 8, 9, 10)
#: [4-5] p.4 "a cloud cover lower than 10%": full-frame share, strict <
SCENE_CLOUD_MAX = 0.10
#: [5] our reading of the paper's whole-tile scenes: no-data share <= 2 %
SCENE_NODATA_MAX = 0.02
#: [1] L2A reflectance = cube value / 10000 (the cubes hold DN + BOA_ADD_OFFSET,
#: i.e. offset-harmonised values; no-data is the int16 fill)
REFLECTANCE_SCALE = 10000.0
#: [9] reference scene: range = p99 - p1 of the finite frame pixels
REF_RANGE_PCT = (1.0, 99.0)
#: [10] p.4 calibration pixels: NIR "<0.05" (sea) or "high (0.2)" (land)
CAL_NIR_SEA_MAX = 0.05
CAL_NIR_LAND_MIN = 0.2
#: [11] p.10 "a threshold of SD (NDWI) = 0.2"; Eq. 3 is a 1/M (ddof=0) SD
SD_NDWI_THRESHOLD = 0.2
#: [14] tide-table emulation: 1-min series, >= 3 h between HW/LW marks
MARK_STEP_MIN = 1.0
MARK_MIN_SEP_MIN = 180.0
#: [17] nplr settings of the paper's fit (p.6): 4 parameters, linear heights
#: (useLog=FALSE: heights are on an MSL datum and can be negative), the
#: package defaults LPweight = 0.25 and method "res"
NPLR_NPARS = 4
NPLR_USELOG = False
NPLR_LPWEIGHT = 0.25
NPLR_METHOD = "res"
#: [19] p.7 "within 2.47 +- 0.25 m"
POOL_HALF_WIDTH_M = 0.25
#: [20] p.7 "repeated for 50,000 random pixels"
N_LAG_SAMPLE = 50_000
#: [21] p.7 "ranging from -90 min to +90 min (at 5 min intervals)"
LAG_GRID_MIN = np.arange(-90.0, 90.0 + 1e-9, 5.0)

#: columns of one logistic fit returned by the R side
FIT_COLS = ("status", "inflection", "bottom", "top", "xmid", "scal", "s",
            "iterations", "code", "n_used", "minimum")
#: fit status codes (R side): 0 ok, 1 nlm iterations == 0 ("'nlm' failed to
#: estimate parameters"), 2 constant fitted values, 3 any other error
FIT_OK, FIT_NLM_NO_ITER, FIT_CONSTANT, FIT_ERROR = 0, 1, 2, 3

R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r",
                        "granadeiro_fit.R")
RSCRIPT_DEFAULT = r"C:/Program Files/R/R-4.4.3/bin/Rscript.exe"


# ─────────────────────────────────────────────────────────────────────────────
#  scenes and reflectance
# ─────────────────────────────────────────────────────────────────────────────

def scene_frame_shares(scl, fill=None):
    """Full-frame shares of one scene's SCL: ``(cloud, nodata)``.

    ``cloud`` = share of ALL frame pixels whose SCL is in
    :data:`SCENE_CLOUD_SCL`; ``nodata`` = share whose SCL is the fill value,
    NaN or 0 (SCL 0 = no data).  Denominator: every pixel of the frame
    (the paper selected whole tiles, p.4)."""
    scl = np.asarray(scl)
    n = scl.size
    if np.issubdtype(scl.dtype, np.floating):
        nodata = ~np.isfinite(scl) | (scl == 0)
    else:
        nodata = scl == 0
        if fill is not None:
            nodata |= scl == fill
    cloud = np.isin(scl, SCENE_CLOUD_SCL)
    return float(np.count_nonzero(cloud) / n), float(np.count_nonzero(nodata) / n)


def scene_rule(cloud, nodata, in_period, has_time):
    """[4-6] A scene is used iff cloud < 0.10 (strict), no-data <= 0.02, it
    lies in the period and it has an overpass time; every passing scene is
    used (p.4)."""
    cloud, nodata = np.asarray(cloud, float), np.asarray(nodata, float)
    return ((cloud < SCENE_CLOUD_MAX) & (nodata <= SCENE_NODATA_MAX)
            & np.asarray(in_period, bool) & np.asarray(has_time, bool))


def to_reflectance(raw, fill):
    """Cube value -> L2A reflectance (float64); the int16 fill -> NaN.

    Negative reflectances (dark water with the BOA offset) are VALID data
    and are kept; only no-data is NaN (nplr drops NAs, nothing else)."""
    raw = np.asarray(raw)
    out = raw.astype(np.float64) / REFLECTANCE_SCALE
    if fill is not None:
        out[raw == fill] = np.nan
    return out


def calibrate(x, coef):
    """[10] Apply per-scene calibration ``(x - intercept) / slope``.

    ``x`` is (T, n) with scenes on axis 0; ``coef`` is (T, 2) =
    (intercept, slope) of the major-axis fit scene ~ reference (the
    reference scene carries (0, 1), which leaves it bit-identical)."""
    coef = np.asarray(coef, float)
    return (x - coef[:, 0:1]) / coef[:, 1:2]


def band_ranges(raw, fill, pct=REF_RANGE_PCT):
    """[9] Per-scene reflectance range p99 - p1 over the finite frame pixels
    (``np.percentile`` linear = R ``quantile`` type 7).  ``raw`` (T, P)."""
    T = raw.shape[0]
    q = np.empty((T, 2))
    for i in range(T):
        v = raw[i]
        v = v[v != fill].astype(np.float64) / REFLECTANCE_SCALE
        q[i] = np.percentile(v, pct)
    return q[:, 1] - q[:, 0], q


def reference_scene(range_green, range_nir):
    """[9] p.4 "informally selected to have an average reflectance range in
    the green and NIR bands": the scene minimising
    |range_G - med_G| / med_G + |range_NIR - med_NIR| / med_NIR
    (first on ties, as argmin).  Returns (index, score)."""
    rg, rn = np.asarray(range_green, float), np.asarray(range_nir, float)
    mg, mn = np.median(rg), np.median(rn)
    score = np.abs(rg - mg) / mg + np.abs(rn - mn) / mn
    return int(np.argmin(score)), score


def calibration_pixels(nir_ref, exclude):
    """[10] Calibration pixels: finite reference-scene NIR < 0.05 (sea) or
    > 0.2 (land), outside ``exclude`` (the intertidal mask).  Returns
    (flat indices, n_sea, n_land)."""
    nir_ref = np.asarray(nir_ref, float)
    ok = np.isfinite(nir_ref) & ~np.asarray(exclude, bool)
    with np.errstate(invalid="ignore"):
        sea = ok & (nir_ref < CAL_NIR_SEA_MAX)
        land = ok & (nir_ref > CAL_NIR_LAND_MIN)
    return np.flatnonzero(sea | land), int(sea.sum()), int(land.sum())


def major_axis(x, y):
    """[10] Major-axis regression ``y = b0 + b1 x`` as lmodel2 1.7.4 computes
    its "MA" row (``MA.reg``), on the finite pairs.

    lmodel2: ``b.ols`` of ``lm(y ~ x)``, ``r = cor(y, x)``,
    ``d = (b.ols^2 - r^2) / (r^2 b.ols)``,
    ``b.ma = (d + sign(r) sqrt(d^2 + 4)) / 2``, ``b0 = ybar - b.ma xbar``;
    NA when r^2 <= .Machine$double.eps; an error when var(x) or var(y) is 0.
    Here b.ols = s_xy / s_xx in closed form; the equality with the R package
    (to 1e-10) is proven on the real calibration data by the v8 script.
    Returns ``(b0, b1, n)``."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        raise ValueError("major_axis needs at least 3 finite pairs")
    xbar, ybar = x.mean(), y.mean()
    dx, dy = x - xbar, y - ybar
    sxx, syy, sxy = dx @ dx, dy @ dy, dx @ dy
    if sxx == 0 or syy == 0:
        raise ValueError("variance of x or y is 0 (lmodel2 stops here)")
    b_ols = sxy / sxx
    r = sxy / np.sqrt(sxx * syy)
    r2 = r * r
    if r2 <= np.finfo(float).eps:
        return np.nan, np.nan, n
    d = (b_ols ** 2 - r2) / (r2 * b_ols)
    b_ma = 0.5 * (d + np.sign(r) * np.sqrt(d ** 2 + 4.0))
    return float(ybar - b_ma * xbar), float(b_ma), n


def ndwi_sd_block(g, n):
    """[11] Eq. 2 NDWI = (G - NIR) / (G + NIR) and Eq. 3 temporal population
    SD (1/M, ddof = 0) per pixel, over the scenes with a finite NDWI.

    ``g``, ``n``: (T, npx) reflectances.  Returns (sd, count, n_nonfinite)
    where n_nonfinite counts observations with both bands finite but a
    non-finite NDWI (G + NIR == 0), which are left out like no-data."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (g - n) / (g + n)
    fin = np.isfinite(nd)
    both = np.isfinite(g) & np.isfinite(n)
    n_nonfinite = int(np.count_nonzero(both & ~fin))
    nd[~fin] = np.nan
    cnt = fin.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(nd, axis=0, ddof=0)
    return sd, cnt, n_nonfinite


def frame_ndwi_sd(G_raw, N_raw, fill, coef_g=None, coef_n=None,
                  block=25_000):
    """[11] SD(NDWI) of every frame pixel over the scenes (rows) of the
    int16 arrays ``G_raw``, ``N_raw`` (T, P), optionally after calibration.
    Returns (sd float64 (P,), count int32 (P,), n_nonfinite)."""
    T, P = G_raw.shape
    sd = np.full(P, np.nan)
    cnt = np.zeros(P, np.int32)
    n_nf = 0
    for a in range(0, P, block):
        b = min(a + block, P)
        g = to_reflectance(G_raw[:, a:b], fill)
        n = to_reflectance(N_raw[:, a:b], fill)
        if coef_g is not None:
            g = calibrate(g, coef_g)
        if coef_n is not None:
            n = calibrate(n, coef_n)
        sd[a:b], cnt[a:b], k = ndwi_sd_block(g, n)
        n_nf += k
    return sd, cnt, n_nf


def sd_histogram_valleys(sd, bin_width=0.01, upper=2.0, smooth_bins=5):
    """[11] Histogram of SD(NDWI) and its valleys, a CHECK of the 0.2
    threshold (p.9-10, Fig. 7: land, water and intertidal peaks).  Returns
    counts, edges, local maxima and minima of the moving-average-smoothed
    histogram, and the valley closest to 0.2."""
    v = np.asarray(sd, float)
    v = v[np.isfinite(v)]
    edges = np.arange(0.0, upper + bin_width / 2, bin_width)
    counts, _ = np.histogram(np.clip(v, 0.0, upper - 1e-12), bins=edges)
    k = np.ones(smooth_bins) / smooth_bins
    sm = np.convolve(counts, k, mode="same")
    centres = 0.5 * (edges[1:] + edges[:-1])
    inner = np.arange(1, len(sm) - 1)
    peaks = inner[(sm[inner] > sm[inner - 1]) & (sm[inner] >= sm[inner + 1])]
    valleys = inner[(sm[inner] < sm[inner - 1]) & (sm[inner] <= sm[inner + 1])]
    top = peaks[np.argsort(sm[peaks])[::-1]][:3] if len(peaks) else peaks
    out = {"bin_width": bin_width, "n": int(len(v)),
           "share_above_threshold": float(np.mean(v > SD_NDWI_THRESHOLD)) if len(v) else None,
           "peaks_sd": [float(centres[i]) for i in sorted(top)],
           "peaks_count_smoothed": [float(sm[i]) for i in sorted(top)],
           "valleys_sd": [float(centres[i]) for i in valleys][:50],
           "counts": counts.astype(int).tolist(), "edges": edges.tolist()}
    if len(valleys):
        j = valleys[np.argmin(np.abs(centres[valleys] - SD_NDWI_THRESHOLD))]
        out["valley_closest_to_threshold"] = float(centres[j])
        if len(top) >= 2:
            a, b = sorted(top[:2])
            between = valleys[(valleys > a) & (valleys < b)]
            if len(between):
                jj = between[np.argmin(sm[between])]
                out["deepest_valley_between_two_highest_peaks"] = float(centres[jj])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  tide-table marks and Eq. 1
# ─────────────────────────────────────────────────────────────────────────────

def tide_marks(t_min, h, min_sep_min=MARK_MIN_SEP_MIN):
    """[14] Tide-table-like high/low-water marks of a dense level series.

    ``t_min``: strictly increasing times in minutes (1-min sampling);
    ``h``: levels.  Steps:

    1. turning points of the series: sign changes of ``diff(h)`` with zero
       differences forward-filled (a plateau gives ONE turning point, timed
       at the plateau centre);
    2. they alternate HW / LW by construction; wiggles are removed by
       persistence simplification: while two neighbouring marks are less
       than ``min_sep_min`` apart, the neighbouring pair with the smallest
       |dh| is deleted (alternation is kept, because a pair holds one HW and
       one LW), so each half-cycle keeps its global extreme;
    3. checks: alternation, separations >= ``min_sep_min``, and that every
       inner HW (LW) is the maximum (minimum) of the series between its two
       neighbouring marks (``n_not_global_extreme``, expected 0).

    Returns a dict with ``t`` (minutes), ``h``, ``is_hw`` and bookkeeping."""
    t = np.asarray(t_min, np.float64)
    h = np.asarray(h, np.float64)
    if t.ndim != 1 or t.shape != h.shape:
        raise ValueError("t_min and h must be 1-D and aligned")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_min must be strictly increasing")
    ok = np.isfinite(h)
    n_nonfinite = int(np.count_nonzero(~ok))
    t, h = t[ok], h[ok]
    if len(h) < 3:
        raise ValueError("too few finite samples")
    d = np.diff(h)
    s = np.sign(d)
    nz = s != 0
    if not nz.any():
        raise ValueError("constant series")
    idx = np.arange(len(s))
    pos = np.maximum.accumulate(np.where(nz, idx, -1))
    pos[pos < 0] = int(np.argmax(nz))
    s = s[pos]
    k = np.flatnonzero(s[1:] != s[:-1]) + 1          # sample of each turn
    is_hw_c = s[k - 1] > 0
    zero = d == 0
    last_nz = np.maximum.accumulate(np.where(~zero, idx, -1))
    zrun = idx - last_nz                              # zeros ending at i
    m = zrun[k - 1]                                   # plateau length (diffs)
    start = k - m
    tc = 0.5 * (t[start] + t[k])
    hc = h[k]
    n = len(k)

    # persistence simplification, smallest |dh| first, only pairs < min_sep
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    if n:
        nxt[-1] = -1
    alive = [True] * n
    tcl, hcl = tc.tolist(), hc.tolist()
    heap = [(abs(hcl[i + 1] - hcl[i]), i, i + 1) for i in range(n - 1)
            if tcl[i + 1] - tcl[i] < min_sep_min]
    heapq.heapify(heap)
    n_pairs_removed = 0
    while heap:
        _, a, b = heapq.heappop(heap)
        if not (alive[a] and alive[b] and nxt[a] == b):
            continue
        p, q = prev[a], nxt[b]
        alive[a] = alive[b] = False
        n_pairs_removed += 1
        if p >= 0:
            nxt[p] = q
        if q >= 0:
            prev[q] = p
        if p >= 0 and q >= 0 and tcl[q] - tcl[p] < min_sep_min:
            heapq.heappush(heap, (abs(hcl[q] - hcl[p]), p, q))
    sel = np.flatnonzero(np.array(alive, bool))
    tm, hm, hw, km = tc[sel], hc[sel], is_hw_c[sel], k[sel]
    if len(tm) < 3:
        raise ValueError("fewer than 3 marks")
    if not np.all(hw[1:] != hw[:-1]):
        raise AssertionError("marks do not alternate")
    sep = np.diff(tm)
    if np.any(sep < min_sep_min):
        raise AssertionError("marks closer than the minimum separation")
    # check: every inner mark is the global extreme between its neighbours
    bad = 0
    for i in range(1, len(km) - 1):
        seg = h[km[i - 1]:km[i + 1] + 1]
        if hw[i]:
            bad += int(h[km[i]] < seg.max())
        else:
            bad += int(h[km[i]] > seg.min())
    return {"t": tm, "h": hm, "is_hw": hw, "sample_index": km,
            "n_samples": int(len(h)), "n_nonfinite_dropped": n_nonfinite,
            "n_turning_points": int(n), "n_pairs_removed": int(n_pairs_removed),
            "n_marks": int(len(tm)), "n_hw": int(hw.sum()),
            "n_lw": int((~hw).sum()), "min_sep_min": float(min_sep_min),
            "separation_min_minutes": float(sep.min()),
            "separation_median_minutes": float(np.median(sep)),
            "n_not_global_extreme": int(bad),
            "n_plateau_turns": int(np.count_nonzero(m > 0))}


def eq1(t_sat, t_hw, h_hw, t_lw, h_lw):
    """Eq. 1 (p.4): water height at the sensing time,
    ``hHW - (hHW - hLW) * (cos(pi (Tsat - TLW) / (THW - TLW)) + 1) / 2``,
    with the operation order of the co-author's R line
    (``hHW-((hHW-hLW)*(cos((pi*(Tsat-TLW)/(THW-TLW)))+1))/2``).  Valid for
    rising (TLW < Tsat < THW) and ebbing (THW < Tsat < TLW) brackets."""
    return h_hw - (h_hw - h_lw) * (np.cos(np.pi * (t_sat - t_lw)
                                          / (t_hw - t_lw)) + 1.0) / 2.0


def eq1_bracket(t, marks):
    """[12-15] The bracketing pair of marks of each time ``t`` (minutes):
    previous mark <= t < next mark.  Returns arrays ``k`` (index of the
    previous mark), ``t_hw, h_hw, t_lw, h_lw`` and ``rising`` (the direction
    of the pair: previous mark a LW)."""
    tm, hm, hw = marks["t"], marks["h"], marks["is_hw"]
    t = np.asarray(t, np.float64)
    pos = np.searchsorted(tm, t, side="right")
    if np.any(pos < 1) or np.any(pos >= len(tm)):
        raise ValueError("a time falls outside the span of the marks")
    k0, k1 = pos - 1, pos
    prev_hw = hw[k0]
    return {"k": k0,
            "t_hw": np.where(prev_hw, tm[k0], tm[k1]),
            "h_hw": np.where(prev_hw, hm[k0], hm[k1]),
            "t_lw": np.where(prev_hw, tm[k1], tm[k0]),
            "h_lw": np.where(prev_hw, hm[k1], hm[k0]),
            "rising": ~prev_hw}


def water_height(t, marks):
    """Eq. 1 water height at times ``t`` (minutes, any shape) and the bracket."""
    b = eq1_bracket(t, marks)
    return eq1(np.asarray(t, np.float64), b["t_hw"], b["h_hw"], b["t_lw"],
               b["h_lw"]), b


def scene_cycles_hours(t_scene, marks):
    """[24] Duration (h) of each scene's bracketing tidal cycle: from the
    scene's previous mark to the next mark of the same type (t[k+2] - t[k])."""
    b = eq1_bracket(t_scene, marks)
    k = b["k"]
    if np.any(k + 2 >= len(marks["t"])):
        raise ValueError("a scene's cycle runs past the last mark")
    return (marks["t"][k + 2] - marks["t"][k]) / 60.0


def lag_heights(t_scene, marks, lag_grid=LAG_GRID_MIN):
    """[21] Eq. 1 heights of every scene at (t - lag) for every lag of the
    grid: (n_lag, T).  Positive lag = later local tide (p.7, Fig. 4)."""
    tt = np.asarray(t_scene, np.float64)[None, :] - np.asarray(lag_grid,
                                                               np.float64)[:, None]
    return water_height(tt, marks)[0]


# ─────────────────────────────────────────────────────────────────────────────
#  standardisation, lag choice, exposure
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_prop(y, axis=0):
    """[16] nplr::convertToProp: ``(y - min) / (max - min)`` with NAs ignored,
    per pixel over its scenes (IEEE-identical to the R function)."""
    y = np.asarray(y, np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mn = np.nanmin(y, axis=axis, keepdims=True)
        mx = np.nanmax(y, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (y - mn) / (mx - mn)


def choose_lags(status, infl, lag_grid=LAG_GRID_MIN):
    """[21] Per pixel, the lag with the smallest |h_rising - h_ebbing| over
    the lags where both fits succeeded; the FIRST minimum on ties (R
    ``which.min``, which also skips NA).  ``status``, ``infl``: (n, n_lag, 2)
    with limb 0 = rising, 1 = ebbing.  Returns (lag, gap (n, n_lag), index)."""
    ok = (status == FIT_OK) & np.isfinite(infl)
    hr = np.where(ok[:, :, 0], infl[:, :, 0], np.nan)
    he = np.where(ok[:, :, 1], infl[:, :, 1], np.nan)
    gap = np.abs(hr - he)
    has = np.isfinite(gap).any(axis=1)
    idx = np.full(len(gap), -1, np.int64)
    if has.any():
        idx[has] = np.nanargmin(gap[has], axis=1)
    lag = np.where(has, np.asarray(lag_grid, float)[np.maximum(idx, 0)], np.nan)
    return lag, gap, idx


def exposure_eq5(h_pixel, h_hw, h_lw, cycle_h):
    """Eq. 5 (p.9): hours exposed per cycle,
    ``C * (1 - acos(2 (h - hLW) / (hHW - hLW) - 1) / pi)``; NaN outside
    [hLW, hHW] (acos of |x| > 1 is NaN in R and numpy; no clipping)."""
    with np.errstate(invalid="ignore"):
        return cycle_h * (1.0 - np.arccos(2.0 * (np.asarray(h_pixel, float) - h_lw)
                                          / (h_hw - h_lw) - 1.0) / np.pi)


# ─────────────────────────────────────────────────────────────────────────────
#  R workers
# ─────────────────────────────────────────────────────────────────────────────

def find_rscript(explicit=None):
    """Rscript executable: ``explicit``, $PYINTERTIDAL_RSCRIPT, R 4.4.3's
    default Windows path, or Rscript on the PATH."""
    for c in (explicit, os.environ.get("PYINTERTIDAL_RSCRIPT"), RSCRIPT_DEFAULT,
              shutil.which("Rscript")):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Rscript not found (pass rscript= or set "
                            "PYINTERTIDAL_RSCRIPT)")


def _tree_rss_mb(proc):
    try:
        tot = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except Exception:
                pass
        return tot / 2 ** 20
    except Exception:
        return 0.0


class RRunner:
    """Parallel ``Rscript granadeiro_fit.R <spec>`` worker processes.

    Jobs are ``(prepare, collect)`` pairs: ``prepare()`` writes the job's
    input files (raw float64 LE) and returns its spec dict (key ``_cleanup``
    lists files to delete after ``collect(spec)`` has read the outputs).
    At most ``workers`` processes run at once; inputs are written just
    before launch, so the temporary directory holds about ``workers`` jobs.
    The directory is created under ``tmpdir`` and removed by :meth:`close`.
    Memory of the worker process trees is sampled once a second."""

    def __init__(self, workers=8, tmpdir=None, rscript=None, log=None,
                 max_temp_bytes=3e9):
        self.workers = max(1, int(workers))
        self.rscript = find_rscript(rscript)
        self.script = R_SCRIPT
        if tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="granadeiro2021_", dir=tmpdir)
        self.log = log or (lambda msg: None)
        self.max_temp_bytes = float(max_temp_bytes)
        self.stats = {"jobs": 0, "worker_seconds": 0.0, "temp_peak_bytes": 0,
                      "r_worker_peak_mb": 0.0, "r_workers_concurrent_peak_mb": 0.0,
                      "python_plus_r_peak_mb": 0.0}
        self._n = 0

    def path(self, stem, ext=".bin"):
        self._n += 1
        return os.path.join(self.root, f"{stem}_{self._n:06d}{ext}")

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def temp_bytes(self):
        tot = 0
        for e in os.scandir(self.root):
            try:
                tot += e.stat().st_size
            except OSError:
                pass
        return tot

    def run(self, jobs, desc="R", progress_every_s=120.0, workers=None):
        """Run the jobs, at most ``workers`` (default: the runner's) at once."""
        import psutil

        n_workers = self.workers if workers is None else max(1, int(workers))

        me = psutil.Process()
        pending = list(enumerate(jobs))[::-1]
        n_jobs = len(pending)
        running = {}
        done = 0
        t_start = last_log = last_mem = time.time()
        try:
            while pending or running:
                while pending and len(running) < n_workers:
                    i, (prepare, collect) = pending.pop()
                    spec = prepare()
                    cleanup = list(spec.pop("_cleanup", []))
                    spec_path = self.path(f"{desc}_spec", ".txt")
                    with open(spec_path, "w", encoding="utf-8") as f:
                        for k, v in spec.items():
                            f.write(f"{k}={v}\n")
                    log_path = spec_path[:-4] + ".log"
                    logf = open(log_path, "w", encoding="utf-8")
                    proc = subprocess.Popen([self.rscript, self.script, spec_path],
                                            stdout=logf, stderr=subprocess.STDOUT,
                                            cwd=self.root)
                    try:
                        pp = psutil.Process(proc.pid)
                    except Exception:
                        pp = None
                    running[i] = dict(proc=proc, pp=pp, spec=spec, collect=collect,
                                      files=cleanup + [spec_path, log_path],
                                      logf=logf, t0=time.time(), log_path=log_path)
                    tb = self.temp_bytes()
                    self.stats["temp_peak_bytes"] = max(self.stats["temp_peak_bytes"], tb)
                    if tb > self.max_temp_bytes:
                        raise RuntimeError(f"temporary files exceed {self.max_temp_bytes / 1e9:.1f} GB")
                finished = [i for i, r in running.items() if r["proc"].poll() is not None]
                for i in finished:
                    r = running.pop(i)
                    r["logf"].close()
                    rc = r["proc"].returncode
                    if rc != 0:
                        with open(r["log_path"], encoding="utf-8", errors="replace") as f:
                            tail = f.read()[-4000:]
                        raise RuntimeError(f"R job {desc}#{i} failed (rc={rc}):\n{tail}")
                    r["collect"](r["spec"])
                    self.stats["worker_seconds"] += time.time() - r["t0"]
                    self.stats["jobs"] += 1
                    for p in r["files"]:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    done += 1
                now = time.time()
                if now - last_mem >= 1.0 and running:
                    last_mem = now
                    rs = [_tree_rss_mb(r["pp"]) for r in running.values() if r["pp"] is not None]
                    if rs:
                        self.stats["r_worker_peak_mb"] = max(self.stats["r_worker_peak_mb"], max(rs))
                        self.stats["r_workers_concurrent_peak_mb"] = max(
                            self.stats["r_workers_concurrent_peak_mb"], sum(rs))
                        own = me.memory_info().rss / 2 ** 20
                        self.stats["python_plus_r_peak_mb"] = max(
                            self.stats["python_plus_r_peak_mb"], own + sum(rs))
                if now - last_log >= progress_every_s:
                    last_log = now
                    el = now - t_start
                    eta = el / max(done, 1) * (n_jobs - done) if done else float("nan")
                    self.log(f"    {desc}: {done}/{n_jobs} R jobs, {el:.0f} s, ETA {eta:.0f} s")
                if not finished:
                    time.sleep(0.1)
        finally:
            for r in running.values():
                try:
                    r["proc"].kill()
                    r["proc"].wait(10)
                except Exception:
                    pass
                try:
                    r["logf"].close()
                except Exception:
                    pass
        return self.stats


def _f64(a):
    return np.ascontiguousarray(a, dtype="<f8")


def fit_logistic(runner, Y, x=None, x_fn=None, reference=False, prop=False,
                 chunk=2000, desc="fit"):
    """[17] Eq. 4 logistic per pixel on the R workers.

    ``Y`` (n_px, T): standardised NIR (lean path) or, with ``reference`` and
    ``prop``, the raw calibrated NIR (convertToProp in R, then ``nplr()``).
    ``x`` (T,) heights shared by every pixel, or ``x_fn(a, b)`` returning
    the (b - a, T) heights of pixels a..b-1.  ``reference`` runs
    ``try(nplr(x, y, useLog=FALSE, npars=4, silent=TRUE))`` +
    ``getInflexion`` instead of the lean path.  Returns (n_px, 11) with the
    columns :data:`FIT_COLS`."""
    if not hasattr(Y, "shape"):
        Y = np.asarray(Y)
    n_px, T = Y.shape
    out = np.full((n_px, len(FIT_COLS)), np.nan)
    if (x is None) == (x_fn is None):
        raise ValueError("give exactly one of x or x_fn")

    def make(a, b):
        def prepare():
            p_in, p_out = runner.path(f"{desc}_in"), runner.path(f"{desc}_out")
            with open(p_in, "wb") as f:
                if x is not None:
                    _f64(x).tofile(f)
                else:
                    X = _f64(x_fn(a, b))
                    if X.shape != (b - a, T):
                        raise ValueError(f"x_fn returned {X.shape}")
                    X.tofile(f)
                _f64(Y[a:b]).tofile(f)
            return {"mode": "nplr" if reference else "fit", "in_file": p_in,
                    "out_file": p_out, "n_px": b - a, "n_obs": T,
                    "x_shared": int(x is not None), "prop": int(prop),
                    "_cleanup": [p_in, p_out]}

        def collect(spec):
            out[a:b] = np.fromfile(spec["out_file"], dtype="<f8").reshape(b - a, len(FIT_COLS))
        return prepare, collect

    chunk = max(1, int(chunk))
    runner.run([make(a, min(a + chunk, n_px)) for a in range(0, n_px, chunk)], desc=desc)
    return out


def lag_fits(runner, Y, H_lag, rising, reference=False, chunk=250, desc="lag"):
    """[21] Rising-tide and ebbing-tide logistics at every lag, per pixel.

    ``Y`` (n_px, T) standardised NIR; ``H_lag`` (n_lag, T) Eq. 1 heights at
    (t - lag); ``rising`` (T,) the fixed split.  Returns ``status`` and
    ``infl`` (n_px, n_lag, 2), limb 0 = rising, 1 = ebbing."""
    if not hasattr(Y, "shape"):
        Y = np.asarray(Y)
    n_px, T = Y.shape
    H_lag = _f64(H_lag)
    L = H_lag.shape[0]
    if H_lag.shape[1] != T:
        raise ValueError("H_lag and Y disagree on the number of scenes")
    rise = _f64(np.asarray(rising, bool).astype(float))
    out = np.full((n_px, L, 2, 2), np.nan)

    def make(a, b):
        def prepare():
            p_in, p_out = runner.path(f"{desc}_in"), runner.path(f"{desc}_out")
            with open(p_in, "wb") as f:
                H_lag.tofile(f)
                rise.tofile(f)
                _f64(Y[a:b]).tofile(f)
            return {"mode": "nplr_lag" if reference else "lag", "in_file": p_in,
                    "out_file": p_out, "n_px": b - a, "n_obs": T, "n_lag": L,
                    "_cleanup": [p_in, p_out]}

        def collect(spec):
            out[a:b] = np.fromfile(spec["out_file"], dtype="<f8").reshape(b - a, L, 2, 2)
        return prepare, collect

    chunk = max(1, int(chunk))
    runner.run([make(a, min(a + chunk, n_px)) for a in range(0, n_px, chunk)], desc=desc)
    return out[..., 0], out[..., 1]


def lmodel2_rows(runner, x, y_fn, n_jobs, jobs_per_process=4, desc="lmodel2",
                 workers=None):
    """[10] ``lmodel2(y ~ x)`` for ``n_jobs`` response vectors (``y_fn(k)``
    returns job k's y, aligned with ``x``; NaN pairs are dropped by
    model.frame's na.omit).  Returns (n_jobs, 8): n, OLS (b0, b1),
    MA (b0, b1), SMA (b0, b1), r."""
    x = _f64(x)
    n_obs = len(x)
    p_x = runner.path(f"{desc}_x")
    x.tofile(p_x)
    out = np.full((n_jobs, 8), np.nan)

    def make(ks):
        def prepare():
            p_y, p_out = runner.path(f"{desc}_y"), runner.path(f"{desc}_out")
            with open(p_y, "wb") as f:
                for k in ks:
                    y = _f64(y_fn(k))
                    if y.shape != (n_obs,):
                        raise ValueError("y_fn returned a wrong shape")
                    y.tofile(f)
            return {"mode": "lmodel2", "in_x": p_x, "in_y": p_y, "out_file": p_out,
                    "n_obs": n_obs, "n_jobs": len(ks), "_cleanup": [p_y, p_out]}

        def collect(spec):
            out[ks] = np.fromfile(spec["out_file"], dtype="<f8").reshape(len(ks), 8)
        return prepare, collect

    groups = [list(range(a, min(a + jobs_per_process, n_jobs)))
              for a in range(0, n_jobs, jobs_per_process)]
    try:
        runner.run([make(g) for g in groups], desc=desc, workers=workers)
    finally:
        try:
            os.remove(p_x)
        except OSError:
            pass
    return out


def gam_lag_surface(runner, lon, lat, lag, lon_new, lat_new, seed):
    """[22] ``mgcv::gam(lag ~ s(lon, lat))`` with the defaults (thin-plate
    regression spline, k = 30, GCV.Cp) on the lag sample; predictions at
    (lon_new, lat_new).  Returns (prediction, fitted, summary dict)."""
    S = _f64(np.column_stack([lon, lat, lag]))
    Pn = _f64(np.column_stack([lon_new, lat_new]))
    n, m = len(S), len(Pn)
    res = {}

    def prepare():
        p_s, p_p = runner.path("gam_sample"), runner.path("gam_pred_in")
        S.tofile(p_s)
        Pn.tofile(p_p)
        p_o, p_f = runner.path("gam_pred"), runner.path("gam_fitted")
        p_sum = runner.path("gam_summary", ".txt")
        return {"mode": "gam", "in_sample": p_s, "in_pred": p_p, "out_file": p_o,
                "out_fitted": p_f, "out_summary": p_sum, "n": n, "m": m,
                "seed": int(seed), "_cleanup": [p_s, p_p, p_o, p_f, p_sum]}

    def collect(spec):
        res["pred"] = np.fromfile(spec["out_file"], dtype="<f8")
        res["fitted"] = np.fromfile(spec["out_fitted"], dtype="<f8")
        info = {}
        with open(spec["out_summary"], encoding="utf-8") as f:
            for line in f:
                k, _, v = line.rstrip("\n").partition("=")
                try:
                    info[k] = float(v) if k not in ("r_version", "mgcv_version", "formula",
                                                    "family", "method", "optimizer",
                                                    "basis", "tprs_xt_seed", "converged",
                                                    "mgcv_conv_fully_converged") else v
                except ValueError:
                    info[k] = v
        res["summary"] = info

    runner.run([(prepare, collect)], desc="gam")
    return res["pred"], res["fitted"], res["summary"]


def r_eq1(runner, t_sat, t_lw, t_hw, h_lw, h_hw):
    """The co-author's R Eq. 1 line evaluated in R (for the tests)."""
    M = _f64(np.column_stack([t_sat, t_lw, t_hw, h_lw, h_hw]))
    res = {}

    def prepare():
        p_in, p_out = runner.path("eq1_in"), runner.path("eq1_out")
        M.tofile(p_in)
        return {"mode": "eq1", "in_file": p_in, "out_file": p_out, "n": len(M),
                "_cleanup": [p_in, p_out]}

    def collect(spec):
        res["h"] = np.fromfile(spec["out_file"], dtype="<f8")

    runner.run([(prepare, collect)], desc="eq1")
    return res["h"]


def r_versions(runner):
    """R and package versions used by the R side."""
    res = {}

    def prepare():
        p = runner.path("versions", ".txt")
        return {"mode": "versions", "out_file": p, "_cleanup": [p]}

    def collect(spec):
        with open(spec["out_file"], encoding="utf-8") as f:
            res.update(dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line))

    runner.run([(prepare, collect)], desc="versions")
    return res
