"""
elevation.py — Intertidal elevation from the NDWI-vs-tide relationship
======================================================================

Two estimators, both computed LOCALLY from the cached cube (no cloud jobs),
both turning "how wet is this pixel at each tide level?" into metres:

**HSR (Hypsometric Super-Resolution)** — our estimator. It exploits the
physics of MIXED pixels: a 10 m pixel on the waterline is *partially*
flooded, and its NDWI is the linear mixture of its wet and dry parts. As the
tide rises the wet fraction follows the pixel's internal hypsometry, so the
per-pixel NDWI-vs-tide curve is a sigmoid whose parameters are physical:

.. math:: \\mathrm{NDWI}_p(t) = a_p + b_p\\,\\Phi\\!\\left(\\frac{h_t-\\mu_p}{\\sigma_p}\\right)

* centre ``μ`` → the pixel's median elevation. Because the FULL curve is
  used (no threshold crossing), precision is sub-decimetre.
* width ``σ`` → the relief INSIDE the pixel: a sub-pixel roughness product
  no other method provides.
* fit quality → a Cramér–Rao uncertainty per pixel, which doubles as the
  method's own quality control.

**Step (DEA-style)** — the published baseline: take a rolling median of NDWI
across tide-height windows and read the tide at which the pixel flips from
dry to wet. It needs a clean monotonic transition, so it answers for fewer
pixels and gives no uncertainty — but on the pixels it does answer for it is
every bit as accurate as HSR, and it is the reference method of the
literature. :func:`fit_step` follows the published algorithm exactly; see its
docstring for why that matters.

Fit per EPOCH, not per decade
-----------------------------
Intertidal morphology migrates, so fitting ten years at once blurs the
transition it is trying to measure. Measured on the Ría de Villaviciosa —
same cube, same mask, same grids, only the date range differing — a ten-year
fit leaves 15.6 % of pixels pinned at the widest σ the search offers against
10.8 % over three years, and the median fit residual drops from 0.215 to
0.191 NDWI. The three-year fit also resolves slightly MORE pixels (33 313
against 31 487), so the shorter window costs no coverage.

A pixel pinned at the top of the σ grid has a transition wider than the grid
can express: its width, and with it its elevation, is bounded rather than
measured. Use :func:`epochs` to split the archive and fit the recent epoch.

What the two estimators are actually worth
------------------------------------------
Against an RTK GNSS field survey of the Ría de Villaviciosa (361 fixed
solutions at 1.3 cm, occupied at the lowest spring tide of the month), scored
on the 107 pixels BOTH methods resolve, HSR and the step method **tie**:
0.135 m RMSE each, and a paired bootstrap over 20 000 resamples cannot
separate them (p = 0.97). Changing the tide model moves the result more than
changing the estimator — see :mod:`pyintertidal.tidecheck`.

An earlier version of this docstring quoted "HSR 0.76 m, step 0.85 m, legacy
methods 1.14–3.4 m" against the IGN 5 m LiDAR. **Those figures are withdrawn.**
Across our intertidal mask that LiDAR takes only seven distinct int16 values
and 83 % of it sits at exactly +2.00 m — it recorded the water surface at
flight time, not the bed. Any ranking built on it says nothing about the
methods. Separately, the step implementation those numbers were measured
against was incomplete (see :func:`fit_step`), so the comparison was unfair in
the other direction too.

What HSR still offers over the step method is not a better RMSE: it is the
sub-pixel width σ, a per-pixel uncertainty, and elevations for the mixed
pixels a threshold crossing has to discard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import time

import numpy as np
import rasterio
from scipy.special import erf

from .cube import open_cube, grid_from_dataset, write_geotiff
from . import water as W


# ─────────────────────────────────────────────────────────────────────────────
#  Epochs
# ─────────────────────────────────────────────────────────────────────────────

def epochs(dates, epoch_years=3):
    """Split dates into epochs of ``epoch_years`` years, anchored at the END.

    Anchoring at the end guarantees the LATEST epoch — the one production
    DEMs are built on — is always a full window (2016..2025 with 3-year
    epochs → …, 2020-2022, 2023-2025; the short remainder falls at the start
    of the record where it does least harm).

    Returns ``[{"label": "2023-2025", "dates": [...]}, ...]``, oldest first.
    """
    if not dates:
        return []
    years = sorted({int(d[:4]) for d in dates})
    out, end = [], years[-1]
    while end >= years[0]:
        start = end - epoch_years + 1
        sel = [d for d in dates if start <= int(d[:4]) <= end]
        if sel:
            out.insert(0, {"label": f"{max(start, years[0])}-{end}",
                           "dates": sel})
        end = start - 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Getting the pixels that matter into memory, once
# ─────────────────────────────────────────────────────────────────────────────

def extract_intertidal_ndwi(nc_path, dates, mask, max_pixels=40_000,
                            clear_classes=W.CLEAR_CLASSES, row_chunk=32,
                            keep_always=None, seed=3, verbose=True):
    """Pull the intertidal pixels' NDWI into memory in one pass.

    A fit over a decade of imagery streams gigabytes and takes ten minutes.
    That is fine to do once; it is ruinous when the question is "what happens
    if the tide is shifted by twenty minutes?" and you want to ask it twenty
    times. But the answer to that question only ever involves the intertidal
    pixels, and there are few enough of them to hold in RAM.

    So: stream the cube ONCE, keep only the pixels inside ``mask``, and every
    later experiment becomes a matrix operation. Measured on the Ría de
    Villaviciosa — 1379 dates, 39 951 pixels kept — this turned ten-minute
    fits into seconds, which is what made a nineteen-point search over tidal
    phase lag take four minutes instead of three hours.

    The returned arrays are exactly what :func:`_fit_block` and
    :mod:`pyintertidal.tidecheck` consume.

    Parameters
    ----------
    nc_path : str
        Cached NDWI cube (B03/B08/SCL).
    dates : list
        Every date in the cube, in cube order.
    mask : (H, W) bool
        Which pixels to keep — normally
        :func:`pyintertidal.frequency.intertidal_mask`.
    max_pixels : int
        Cap on pixels kept, subsampled at random beyond it. 40 000 float32
        values over 1379 dates is about 220 MB for ``Y`` and ``C`` together.
    keep_always : array of flat indices, optional
        Pixels exempt from the subsampling — field-survey locations, say,
        which are the only independent check there is and must not be
        thinned away.

    Returns
    -------
    dict with ``Y`` (T, P) float32 NDWI, ``C`` (T, P) bool usable-observation
    mask, ``index`` the flat indices of the kept pixels, ``dates``, and
    ``shape`` of the full grid.
    """
    ds, arrays, t_dim = open_cube(nc_path, ("B03", "B08", "SCL"))
    try:
        b03, b08, scl = arrays["B03"], arrays["B08"], arrays["SCL"]
        T = scl.sizes[t_dim]
        H = scl.sizes[scl.dims[1]]
        Wd = scl.sizes[scl.dims[2]]

        keep = np.flatnonzero(np.asarray(mask, bool).ravel())
        if keep_always is not None:
            forced = np.asarray(keep_always, dtype=np.int64)
            forced = forced[np.asarray(mask, bool).ravel()[forced]]
        else:
            forced = np.empty(0, dtype=np.int64)
        if keep.size > max_pixels:
            rng = np.random.default_rng(seed)
            pool = np.setdiff1d(keep, forced)
            n = max(0, int(max_pixels) - forced.size)
            keep = np.union1d(forced, rng.choice(pool, n, replace=False))
        keep = np.sort(keep)
        kr, kc = keep // Wd, keep % Wd
        if verbose:
            print(f"[extract] {int(mask.sum()):,} intertidal px -> "
                  f"{keep.size:,} kept ({forced.size} forced)")

        Y = np.full((T, keep.size), np.nan, np.float32)
        C = np.zeros((T, keep.size), bool)
        t0 = time.time()
        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            sel = (kr >= y0) & (kr < y1)
            if not sel.any():
                continue
            sl = {scl.dims[1]: slice(y0, y1)}
            g = np.asarray(b03.isel(sl).values, np.float32)
            n_ = np.asarray(b08.isel(sl).values, np.float32)
            s = np.nan_to_num(np.asarray(scl.isel(sl).values),
                              nan=0.0).astype(np.int16)
            den = g + n_
            with np.errstate(invalid="ignore", divide="ignore"):
                nd = np.where(den != 0, (g - n_) / den, np.nan).astype(np.float32)
            ok = np.isin(s, list(clear_classes)) & np.isfinite(nd)
            rr, cc = kr[sel] - y0, kc[sel]
            Y[:, sel] = nd[:, rr, cc]
            C[:, sel] = ok[:, rr, cc]
            if verbose:
                done = y1 / H
                print(f"[extract] rows {y1}/{H} ({100 * done:4.1f} %)  "
                      f"~{time.time() - t0:.0f}s elapsed", flush=True)
    finally:
        ds.close()

    if verbose:
        print(f"[extract] {T} dates x {keep.size:,} px, "
              f"{100 * C.mean():.1f} % usable observations")
    return {"Y": Y, "C": C, "index": keep, "dates": list(dates),
            "shape": (H, Wd)}


# ─────────────────────────────────────────────────────────────────────────────
#  HSR — stage 1: per-pixel hypsometric fit
# ─────────────────────────────────────────────────────────────────────────────

def _fit_block(Y, C, tide, mu_grid, sg_grid):
    """Least-squares sigmoid fit for a block of pixels.

    ``Y`` (T, P) are NDWI values, ``C`` (T, P) marks usable observations.
    For each candidate ``(μ, σ)`` the amplitude terms ``(a, b)`` have a
    closed-form least-squares solution, so the search is a cheap sweep over
    a 2-D grid rather than an iterative optimiser — this is what makes the
    method fast enough for millions of pixels.

    Returns ``(a, b, mu, sigma, rmse, n_obs)``, each of length P.
    """
    T, P = Y.shape
    # C may be a boolean mask (every observation counts the same) or a float
    # weight per observation. Inverse-variance weighting is the principled
    # version of the smoothing the published step method applies: a bin
    # averaged from many scenes deserves more say than a noisy one.
    Cf = C.astype(np.float64)
    Yc = np.where(Cf > 0, Y, 0.0).astype(np.float64) * Cf

    N = Cf.sum(axis=0)
    Sy = Yc.sum(axis=0)
    Syy = (np.where(Cf > 0, Y, 0.0) * Yc).sum(axis=0)

    best_loss = np.full(P, np.inf)
    best = np.zeros((4, P))

    for mu in mu_grid:
        for sg in sg_grid:
            phi = 0.5 * (1.0 + erf((tide - mu) / (sg * np.sqrt(2.0))))    # (T,)
            Sp = Cf.T @ phi
            Spp = Cf.T @ (phi * phi)
            Syp = Yc.T @ phi
            det = N * Spp - Sp * Sp
            ok = det > 1e-9
            b = np.where(ok, (N * Syp - Sp * Sy) / np.where(ok, det, 1.0), 0.0)
            a = np.where(N > 0, (Sy - b * Sp) / np.where(N > 0, N, 1.0), 0.0)
            loss = (Syy - 2 * a * Sy - 2 * b * Syp + a * a * N
                    + 2 * a * b * Sp + b * b * Spp)
            upd = ok & (loss < best_loss)
            best_loss = np.where(upd, loss, best_loss)
            best[0] = np.where(upd, a, best[0])
            best[1] = np.where(upd, b, best[1])
            best[2] = np.where(upd, mu, best[2])
            best[3] = np.where(upd, sg, best[3])

    rmse = np.sqrt(np.maximum(best_loss, 0.0) / np.maximum(N, 1.0))
    return best[0], best[1], best[2], best[3], rmse, N


def auto_tide_bins(tide, samples_per_bin=5, lo=20, hi=400):
    """How finely to bin the tide axis, from how finely it was sampled.

    Binning trades two errors against each other, and the balance is a
    property of the ARCHIVE, not a number to pick by hand:

    * too few bins quantises the tide axis. Once a bin is wider than a
      pixel's transition, the sigmoid fits entirely inside one bin, the
      elevation stops being identifiable within it, and the fit snaps to an
      arbitrary value — a horizontal band in any scatter against truth;
    * too many bins (or none) averages nothing, so scene noise — thin cirrus,
      glint, turbidity — passes straight into the fit.

    Measured on the Tagus, against a real survey: 40 bins gave RMSE 0.339 m
    with 39 % of pixels pinned at the bottom of the σ grid, no binning gave
    0.289 m, and the optimum near 150 bins gave 0.275 m. That optimum is
    ``samples_per_bin`` observations per bin — enough to average noise down,
    not so much that the axis coarsens below the transitions being measured.

    Parameters
    ----------
    tide : array
        Tide height of every observation that will enter the fit.
    samples_per_bin : float
        Target number of observations averaged into each bin.
    lo, hi : int
        Bounds, so a sparse archive still gets a usable axis and a dense one
        does not produce thousands of near-empty bins.

    Returns
    -------
    int
    """
    t = np.sort(np.asarray(tide, dtype=float))
    t = t[np.isfinite(t)]
    if t.size < 3:
        return int(lo)
    gap = np.median(np.diff(t))
    span = t[-1] - t[0]
    if gap <= 0 or span <= 0:
        return int(lo)
    return int(np.clip(round(span / (samples_per_bin * gap)), lo, hi))


def tide_composite(Y, C, tide, n_bins=40):
    """Collapse observations into per-tide-bin medians before fitting.

    Individual scenes carry noise that has nothing to do with elevation:
    thin cirrus the cloud mask missed, sun glint, turbidity plumes, a
    seasonal shift in the algae on the flat. Because that noise is
    independent of tide height while the SIGNAL is a function of it,
    averaging every observation that shares a tide level cancels the first
    and keeps the second — the trick the published step method uses, applied
    here to the sigmoid fit.

    Trade-off: the tide axis is quantised to ``n_bins`` levels, so bins far
    finer than the fit's own precision buy nothing, and bins so coarse that
    the transition falls inside a single bin destroy it. Forty over a 3–4 m
    range gives ~8 cm steps, which sits below the transition width of a
    typical pixel.

    Parameters
    ----------
    Y, C : (T, P) arrays
        NDWI values and the mask of usable observations.
    tide : (T,) array
        Tide height per observation.
    n_bins : int
        Number of equal-width tide bins.

    Returns
    -------
    ``(Yb, Cb, tide_b, Nb)`` with ``n_bins`` rows. ``Nb`` counts the raw
    observations behind each bin, so the fit can weight a bin averaged from
    twenty scenes above one averaged from two instead of trusting both
    equally.
    """
    edges = np.linspace(tide.min(), tide.max(), int(n_bins) + 1)
    idx = np.clip(np.digitize(tide, edges) - 1, 0, int(n_bins) - 1)

    P = Y.shape[1]
    Yb = np.zeros((int(n_bins), P), np.float64)
    Cb = np.zeros((int(n_bins), P), bool)
    Nb = np.zeros((int(n_bins), P), np.float64)
    tide_b = 0.5 * (edges[:-1] + edges[1:])
    for k in range(int(n_bins)):
        rows = np.flatnonzero(idx == k)
        if rows.size == 0:
            continue
        block = np.where(C[rows], Y[rows], np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(block, axis=0)
        ok = np.isfinite(med)
        Yb[k] = np.where(ok, med, 0.0)
        Cb[k] = ok
        Nb[k] = np.isfinite(block).sum(axis=0)
        # The bin's own mean tide is more faithful than its centre when the
        # observations inside it are not evenly spread.
        tide_b[k] = float(tide[rows].mean())
    return Yb, Cb, tide_b, Nb


def hsr_stage1(
    nc_path,
    dates,
    tide_heights,
    valid_dates=None,
    clear_classes=W.CLEAR_CLASSES,
    mu_points=60,
    sg_grid=(0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65),
    robust_iters=1,
    row_chunk=32,
    min_obs=8,
    min_b=0.15,
    tide_bins=None,
    min_bin_count=1,
    min_correlation=None,
    weight_bins=True,
    verbose=True,
):
    """Fit the hypsometric sigmoid per pixel, streaming the cube by row blocks.

    Parameters
    ----------
    nc_path : str
        Cached NDWI cube (B03/B08/SCL).
    dates : list
        Every date in the cube, in cube order.
    tide_heights : dict {date: metres}
        Tide height per date; dates missing here are ignored.
    valid_dates : list, optional
        Restrict the fit to these dates (an epoch, after cloud filtering).
    mu_points, sg_grid
        Search grid: ``mu_points`` elevations spanning the observed tide
        range, and the candidate sigmoid widths (metres of within-pixel
        relief). A σ landing on the last grid value is SATURATED — the fit
        could not represent the transition and stage 2 will not trust it.
    robust_iters : int
        Re-fits after discarding observations beyond 2.5×RMSE (residual
        clouds, sun glint, mis-flagged scenes).
    row_chunk : int
        Rows per streaming block (memory control).
    min_obs, min_b
        A pixel is only considered intertidal-fittable with at least
        ``min_obs`` clear observations and a dry→wet contrast above
        ``min_b``.
    tide_bins : int | "auto" | None
        Fit per-tide-bin medians instead of individual scenes — see
        :func:`tide_composite`. ``"auto"`` derives the count from the tide
        sampling of this archive (:func:`auto_tide_bins`), which is the only
        defensible way to set it: too coarse and the elevation stops being
        identifiable inside a bin, too fine and nothing is averaged. None
        disables binning. ``min_obs`` still counts the RAW observations, so a
        pixel cannot pass the test on the strength of the averaging.

    Returns
    -------
    dict with ``mu``, ``sigma``, ``a``, ``b``, ``rmse``, ``n_obs``,
    ``sigma_mu`` (Cramér–Rao uncertainty), ``valid``, ``transform``, ``crs``.
    """
    ds, arrays, t_dim = open_cube(nc_path, ("B03", "B08", "SCL"))
    b03, b08, scl = arrays["B03"], arrays["B08"], arrays["SCL"]
    try:
        T = scl.sizes[t_dim]
        H = scl.sizes[scl.dims[1]]
        Wd = scl.sizes[scl.dims[2]]

        # Tide per observation (NaN where the date is unusable).
        vd = set(valid_dates) if valid_dates else None
        tide = np.full(T, np.nan, np.float64)
        for i, d in enumerate(dates):
            if vd is not None and d not in vd:
                continue
            if d in tide_heights and tide_heights[d] is not None:
                tide[i] = float(tide_heights[d])
        has_tide = np.isfinite(tide)
        tide_f = np.where(has_tide, tide, 0.0)

        tmin = float(np.nanmin(tide)) if has_tide.any() else -1.0
        tmax = float(np.nanmax(tide)) if has_tide.any() else 1.0
        mu_grid = np.linspace(tmin, tmax, int(mu_points))
        sg_grid = np.asarray(sg_grid, np.float64)

        if tide_bins == "auto":
            tide_bins = auto_tide_bins(tide[has_tide])
            print(f"[hsr] tide_bins='auto' → {tide_bins} bins "
                  f"({(tmax - tmin) / tide_bins * 100:.2f} cm each)")

        out = {k: np.zeros((H, Wd), np.float32)
               for k in ("a", "b", "mu", "sigma", "rmse", "n_obs", "corr")}

        t_start = time.time()
        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            sl = {scl.dims[1]: slice(y0, y1)}
            g = np.asarray(b03.isel(sl).values, np.float32)
            n = np.asarray(b08.isel(sl).values, np.float32)
            s = np.nan_to_num(np.asarray(scl.isel(sl).values),
                              nan=0.0).astype(np.int16)
            den = g + n
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)
            usable = (np.isin(s, list(clear_classes)) & np.isfinite(ndwi)
                      & has_tide[:, None, None])

            hh = y1 - y0
            Y = ndwi.reshape(T, hh * Wd)
            C = usable.reshape(T, hh * Wd)
            raw_obs = C.sum(axis=0).astype(np.float32)

            # Pearson NDWI-vs-tide on the RAW observations. A pixel whose
            # wetness does not track the tide is not intertidal, whatever a
            # sigmoid can be made to fit through it. Borrowed from the
            # published method, which filters candidates this way before
            # doing any work.
            Yv = np.where(C, Y, 0.0).astype(np.float64)
            okf = C.astype(np.float64)
            n_c = np.maximum(okf.sum(0), 1)
            Sx = tide_f @ okf
            Sxx = (tide_f ** 2) @ okf
            Sy_ = Yv.sum(0)
            Syy_ = (Yv * Yv).sum(0)
            Sxy = tide_f @ Yv
            cov = Sxy - Sx * Sy_ / n_c
            vx = Sxx - Sx * Sx / n_c
            vy = Syy_ - Sy_ * Sy_ / n_c
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where((vx > 0) & (vy > 0),
                                cov / np.sqrt(np.maximum(vx * vy, 1e-12)), 0.0)
            out["corr"][y0:y1] = corr.reshape(hh, Wd)
            del Yv, okf

            if tide_bins:
                # Only the observations that carry a tide height can be
                # binned; the rest are already excluded by `usable`.
                keep = has_tide
                Yf, Cb, tf, Nb = tide_composite(
                    np.nan_to_num(Y[keep]), C[keep], tide[keep],
                    n_bins=tide_bins)
                # A bin backed by fewer than `min_bin_count` scenes says too
                # little to be worth a vote; the rest are weighted by how
                # many scenes went into them.
                keep_bin = Cb & (Nb >= min_bin_count)
                # Weight by how many scenes back each bin, or treat every
                # surviving bin alike. `weight_bins=False` exists so the
                # question "do the weights help?" can actually be answered.
                Cf = np.where(keep_bin, Nb if weight_bins else 1.0, 0.0)
            else:
                Yf, Cf, tf = Y, C.astype(np.float64), tide_f

            a, b, mu, sg, rmse, N = _fit_block(Yf, Cf, tf, mu_grid, sg_grid)

            for _ in range(int(robust_iters)):
                sg_safe = np.maximum(sg[None], 1e-3)
                phi = 0.5 * (1 + erf((tf[:, None] - mu[None])
                                     / (sg_safe * np.sqrt(2))))
                resid = np.abs(Yf - (a[None] + b[None] * phi))
                C2 = np.where(resid < 2.5 * np.maximum(rmse[None], 0.02),
                              Cf, 0.0)
                a, b, mu, sg, rmse, N = _fit_block(Yf, C2, tf, mu_grid, sg_grid)

            # Report the real evidence behind each pixel, not the bin count.
            if tide_bins:
                N = raw_obs

            shp = (hh, Wd)
            out["a"][y0:y1] = a.reshape(shp)
            out["b"][y0:y1] = b.reshape(shp)
            out["mu"][y0:y1] = mu.reshape(shp)
            out["sigma"][y0:y1] = sg.reshape(shp)
            out["rmse"][y0:y1] = rmse.reshape(shp)
            out["n_obs"][y0:y1] = N.reshape(shp)

            if verbose:
                # A fit over a decade of imagery runs for many minutes with
                # nothing to show. Silence is indistinguishable from a hung
                # process — we once let a sleeping laptop masquerade as a
                # 21-hour computation — so say where we are and when.
                done = y1 / H
                el = time.time() - t_start
                print(f"[hsr] filas {y1}/{H} ({100 * done:4.1f} %)  "
                      f"{time.strftime('%H:%M:%S')}  "
                      f"faltan ~{el * (1 - done) / max(done, 1e-9) / 60:.1f} min",
                      flush=True)

        transform, crs = grid_from_dataset(ds)
    finally:
        ds.close()

    # Fittable intertidal pixels: enough contrast, enough observations, and an
    # elevation inside the observed tide range (outside it the fit extrapolates).
    rng = tmax - tmin
    corr_ok = (np.ones_like(out["corr"], bool) if min_correlation is None
               else out["corr"] >= min_correlation)
    valid = (corr_ok
             & (out["b"] > min_b)
             & (out["mu"] > tmin + 0.02 * rng)
             & (out["mu"] < tmax - 0.02 * rng)
             & (out["n_obs"] >= min_obs))

    # Cramér–Rao uncertainty of the elevation:
    #     var(μ) ≈ rmse² / Σ (∂model/∂μ)² ,  ∂model/∂μ = -b·pdf((h-μ)/σ)/σ
    # Computed per row block so a full (H, W, T) array is never materialised.
    # With tide binning the fit saw `tide_bins` observations, not hundreds:
    # using the raw tide list here would credit the fit with information it
    # never had and make an already-optimistic bound worse.
    tt = tide[np.isfinite(tide)]
    if tide_bins:
        edges = np.linspace(tt.min(), tt.max(), int(tide_bins) + 1)
        tt = 0.5 * (edges[:-1] + edges[1:])
    sigma_mu = np.full(out["mu"].shape, np.nan, np.float32)
    if tt.size:
        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            mu_b = out["mu"][y0:y1][..., None]
            sg_b = np.maximum(out["sigma"][y0:y1][..., None], 1e-3)
            z = (tt[None, None, :] - mu_b) / sg_b
            pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            ssum = ((out["b"][y0:y1][..., None] * pdf / sg_b) ** 2).sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sigma_mu[y0:y1] = np.where(
                    ssum > 1e-9, out["rmse"][y0:y1] / np.sqrt(ssum), np.nan
                ).astype(np.float32)

    out["sigma_mu"] = sigma_mu
    out["valid"] = valid
    out["transform"] = transform
    out["crs"] = crs
    out["tide_range"] = (tmin, tmax)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  HSR — stage 2: super-resolution
# ─────────────────────────────────────────────────────────────────────────────

def hsr_stage2(mu, sigma, valid, scale=4, sigma_floor=0.05, sigma_cap=0.30,
               sigma_saturated=0.60, scale_max=1.5, iters=400, lam=0.55):
    """Super-resolve the DEM to ``10/scale`` m by alternating projections.

    Each iteration alternates two constraints: Jacobi smoothing (the surface
    should be continuous) and a per-block statistical constraint (each coarse
    block must keep mean ``μ`` and standard deviation ``σ_eff``). The σ term
    is what distinguishes this from plain interpolation: it injects the
    within-pixel relief the fit MEASURED instead of inventing a smooth ramp.

    Guard rails, all learned the hard way:

    ``sigma_floor``
        Subtracts the noise floor of the fit, so flat pixels flatten instead
        of shimmering.
    ``sigma_saturated``
        A σ at the top of the search grid means the fit could NOT represent
        the transition (tide error or morphological change absorbed into the
        width, not real relief). Those pixels inject nothing — smoothing
        only. Without this the DEM fills with checkerboard noise.
    ``sigma_cap`` / ``scale_max``
        Bound how much relief and how much amplification a single projection
        step may apply.
    """
    K = int(scale)
    CH, CW = mu.shape
    sig = sigma.astype(np.float64)
    sg_eff = np.sqrt(np.maximum(sig ** 2 - float(sigma_floor) ** 2, 0.0))
    sg_eff = np.minimum(sg_eff, float(sigma_cap))
    sg_eff = np.where(sig >= float(sigma_saturated), 0.0, sg_eff)

    def up(c):
        """Upsample a coarse array to the fine grid."""
        return np.repeat(np.repeat(c, K, 0), K, 1)

    zf = up(np.where(valid, mu, np.nan)).astype(np.float64)
    fill = float(np.nanmean(zf)) if np.isfinite(zf).any() else 0.0
    zf = np.nan_to_num(zf, nan=fill)
    vf = up(valid)

    for _ in range(int(iters)):
        nb = (np.roll(zf, 1, 0) + np.roll(zf, -1, 0) +
              np.roll(zf, 1, 1) + np.roll(zf, -1, 1)) / 4.0
        zf = np.where(vf, (1 - lam) * zf + lam * nb, zf)

        zb = zf.reshape(CH, K, CW, K)
        m = zb.mean(axis=(1, 3), keepdims=True)
        s = zb.std(axis=(1, 3), keepdims=True)
        sc = np.where(valid[:, None, :, None],
                      np.minimum(sg_eff[:, None, :, None] / np.maximum(s, 1e-6),
                                 float(scale_max)), 1.0)
        zb = (zb - m) * sc + np.where(valid[:, None, :, None],
                                      mu[:, None, :, None].astype(np.float64), m)
        zf = zb.reshape(CH * K, CW * K)

    return np.where(vf, zf, np.nan).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ElevationResult:
    """Elevation products on the analysis grid.

    ``mu`` is the DEM (metres above the tide-model MSL datum), ``sigma`` the
    sub-pixel relief, ``sigma_mu`` the per-pixel uncertainty, ``z_fine`` the
    super-resolved DEM at ``10/scale`` m.
    """

    mu: np.ndarray
    sigma: np.ndarray
    sigma_mu: np.ndarray
    valid: np.ndarray
    z_fine: np.ndarray | None
    scale: int
    transform: object
    crs: object
    rmse: np.ndarray
    n_obs: np.ndarray
    method: str = "hsr"
    epoch: str = ""

    @property
    def transform_fine(self):
        """Affine transform of the super-resolved grid."""
        t = self.transform
        return rasterio.Affine(t.a / self.scale, t.b, t.c,
                               t.d, t.e / self.scale, t.f)

    def clip_to(self, mask):
        """Restrict every product to ``mask`` (e.g. the intertidal mask)."""
        m = np.asarray(mask, dtype=bool)
        keep = np.zeros(self.mu.shape, bool)
        h = min(m.shape[0], keep.shape[0])
        w = min(m.shape[1], keep.shape[1])
        keep[:h, :w] = m[:h, :w]
        self.valid = self.valid & keep
        for name in ("mu", "sigma", "sigma_mu", "rmse"):
            setattr(self, name,
                    np.where(keep, getattr(self, name), np.nan).astype(np.float32))
        if self.z_fine is not None:
            kf = np.repeat(np.repeat(keep, self.scale, 0), self.scale, 1)
            self.z_fine = np.where(kf, self.z_fine, np.nan).astype(np.float32)
        return self

    def save(self, out_dir="."):
        """Write every product as GeoTIFF (fine DEM on its own finer grid)."""
        os.makedirs(out_dir, exist_ok=True)
        pre = self.method
        write_geotiff(os.path.join(out_dir, f"{pre}_elevation.tif"), self.mu,
                      self.transform, self.crs, "float32", np.nan)
        write_geotiff(os.path.join(out_dir, f"{pre}_sigma.tif"), self.sigma,
                      self.transform, self.crs, "float32", np.nan)
        write_geotiff(os.path.join(out_dir, f"{pre}_uncertainty.tif"),
                      self.sigma_mu, self.transform, self.crs, "float32", np.nan)
        if self.z_fine is not None:
            write_geotiff(os.path.join(out_dir, f"{pre}_dem_fine.tif"),
                          self.z_fine, self.transform_fine, self.crs,
                          "float32", np.nan)
        return out_dir

    def summary(self):
        """Print a short report of the DEM."""
        v = np.isfinite(self.mu)
        print(f"[{self.method}{' · ' + self.epoch if self.epoch else ''}] "
              f"{int(v.sum()):,} px")
        if v.any():
            print(f"  elevation [{np.nanmin(self.mu):.2f}, "
                  f"{np.nanmax(self.mu):.2f}] m")
            print(f"  sub-pixel relief σ: median "
                  f"{np.nanmedian(self.sigma[v]) * 100:.0f} cm | "
                  f"saturated {100 * np.nanmean(self.sigma[v] >= 0.60):.0f}%")
            print(f"  uncertainty: median "
                  f"{np.nanmedian(self.sigma_mu[v]) * 100:.1f} cm")
        return self


#: Backwards-compatible alias (the HSR-specific name used in earlier work).
HsrResult = ElevationResult


# ─────────────────────────────────────────────────────────────────────────────
#  High-level fits
# ─────────────────────────────────────────────────────────────────────────────

def fit_hsr(cube, tide_heights, dates=None, scale=4, sigma_floor=0.05,
            max_uncertainty=0.10, clip_mask=None, epoch_label="",
            superresolve=True, **stage1_kwargs):
    """Fit HSR elevation on a cached cube (local computation, no cloud job).

    Parameters
    ----------
    cube : SentinelCube
        Must be an NDWI cube (``water="ndwi"``, i.e. B03/B08/SCL).
    tide_heights : dict {date: metres}
        From :meth:`pyintertidal.tides.TideService.heights_for`.
    dates : list, optional
        The epoch's dates (see :func:`epochs`); defaults to every date with a
        tide height.
    scale : int
        Super-resolution factor (4 → 2.5 m grid).
    max_uncertainty : float
        Quality control: drop pixels whose Cramér–Rao uncertainty exceeds
        this (metres). The method flags its own bad fits.
    clip_mask : array, optional
        Restrict to this mask (pass the intertidal mask).
    superresolve : bool
        Compute stage 2. Set False for a fast 10 m-only run.

    Returns
    -------
    ElevationResult
    """
    if cube.water != "ndwi":
        raise ValueError("HSR needs an NDWI cube (water='ndwi')")
    if dates is None:
        dates = sorted(tide_heights.keys())

    s1 = hsr_stage1(cube.cache_path, cube.dates, tide_heights,
                    valid_dates=dates, **stage1_kwargs)

    valid = s1["valid"]
    if clip_mask is not None:
        m = np.zeros_like(valid)
        h = min(m.shape[0], clip_mask.shape[0])
        w = min(m.shape[1], clip_mask.shape[1])
        m[:h, :w] = clip_mask[:h, :w]
        valid = valid & m
    bad = ~np.isfinite(s1["sigma_mu"]) | (s1["sigma_mu"] > max_uncertainty)
    valid = valid & ~bad

    mu = np.where(valid, s1["mu"], np.nan).astype(np.float32)
    z_fine = None
    if superresolve:
        z_fine = hsr_stage2(np.nan_to_num(mu, nan=0.0),
                            np.where(valid, s1["sigma"], 0.0),
                            valid, scale=scale, sigma_floor=sigma_floor)
    # Every per-pixel product is masked to the SAME valid set. Stage 1 fills
    # its arrays everywhere — a sigmoid can be fitted to a roof or a wet
    # field — and leaking that through would put land noise on the
    # uncertainty map, in the exported GeoTIFF, and into the significance
    # test of `morphodynamics.dem_difference`, which reads sigma_mu directly.
    def masked(name):
        return np.where(valid, s1[name], np.nan).astype(np.float32)

    return ElevationResult(
        mu=mu,
        sigma=masked("sigma"),
        sigma_mu=masked("sigma_mu"), valid=valid, z_fine=z_fine,
        scale=int(scale), transform=s1["transform"], crs=s1["crs"],
        rmse=masked("rmse"), n_obs=np.where(valid, s1["n_obs"], np.nan
                                            ).astype(np.float32),
        method="hsr", epoch=epoch_label,
    )


def _interp_axis0(a, src, dst):
    """Linear interpolation along axis 0, NaN-aware, for (n, y, x) stacks."""
    n, h, w = a.shape
    flat = a.reshape(n, -1)
    out = np.empty((len(dst), flat.shape[1]), np.float32)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        out[:, j] = (np.interp(dst, src[ok], col[ok], left=np.nan,
                               right=np.nan) if ok.sum() >= 2 else np.nan)
    return out.reshape(len(dst), h, w)


def _rolling_mean0(a, window, min_periods):
    """Trailing NaN-aware rolling mean along axis 0.

    Trailing, not centred, because that is what the reference
    implementation does (``center=False``); a centred window would shift
    the extracted transition relative to theirs.
    """
    n = a.shape[0]
    ok = np.isfinite(a)
    vals = np.where(ok, a, 0.0).astype(np.float64)
    csum = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(vals, 0)])
    ccnt = np.concatenate([np.zeros((1,) + a.shape[1:]),
                           np.cumsum(ok, 0, dtype=np.float64)])
    lo = np.maximum(np.arange(n) - window + 1, 0)
    hi = np.arange(n) + 1
    s = csum[hi] - csum[lo]
    c = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c >= min_periods, s / np.maximum(c, 1), np.nan)
    return out.astype(np.float32)


def fit_step(cube, tide_heights, dates=None, threshold=0.1, n_windows=100,
             window_frac=0.15, min_obs=5, row_chunk=32, clip_mask=None,
             epoch_label="", clear_classes=W.CLEAR_CLASSES,
             freq_range=(0.01, 0.99), min_correlation=0.15,
             erode_upper=True, min_count=5, window_offset=5,
             interp_intervals=200, smooth_radius=20, min_periods=5,
             verbose=True):
    """The DEA Intertidal STEP method, as published (Bishop-Taylor et al.).

    This exists to be a FAIR baseline, so it follows the paper rather than
    our own conventions — a comparison against a weakened version of someone
    else's method proves nothing. Per pixel: summarise NDWI as a rolling
    median over ``n_windows`` tide-height windows each spanning
    ``window_frac`` of the observed tide range, then take the lowest tide at
    which that median first reads wet.

    .. warning::

       An earlier version of this function was **not** faithful: it lacked the
       interpolation to ``interp_intervals`` steps, the trailing rolling mean
       of radius ``smooth_radius``, ``min_count`` and ``min_correlation``.
       Measured on the Tagus, correcting it moved the baseline from 0.218 m
       RMSE over 133 083 pixels to **0.184 m over 272 325** — better on both
       axes at once. Every HSR-versus-step comparison produced before that fix
       was against a weakened opponent and must not be reused.

    Faithful to the paper
    ---------------------
    ``threshold=0.1``
        Their value, not the textbook 0.0. They state it captures the
        dry–wet transition more cleanly, and it is not ours to second-guess
        when the point is to reproduce their method.
    ``freq_range=(0.01, 0.99)``
        Candidate pixels must have a dynamic inundation frequency; a pixel
        that is always wet or always dry is not intertidal.
    ``require_positive_corr``
        Candidates must also show a positive Pearson correlation between
        NDWI and tide height — a pixel whose wetness falls as the tide rises
        is not responding to the tide.
    ``erode_upper``
        Their correction for the bias mixed pixels introduce along the upper
        landward edge, where part of the pixel is permanently dry so a
        higher tide is needed before it looks flooded.

    Known deviation
    ---------------
    The window is scaled to the tide range observed over the WHOLE area,
    where the paper scales it per pixel. Across a continent that matters
    (tidal range varies by an order of magnitude); across one estuary the
    two are nearly identical, and the per-pixel version costs a rolling
    median per pixel per window.

    Returns an :class:`ElevationResult` (``sigma`` is NaN: the method does
    not estimate within-pixel relief).
    """
    if cube.water != "ndwi":
        raise ValueError("the step method needs an NDWI cube (water='ndwi')")
    if dates is None:
        dates = sorted(tide_heights.keys())
    keep = set(dates)

    all_dates = cube.dates
    tide = np.array([tide_heights.get(d, np.nan) if d in keep else np.nan
                     for d in all_dates], dtype=np.float64)
    has_tide = np.isfinite(tide)
    if has_tide.sum() < 2:
        raise ValueError("not enough dates with tide heights")
    tmin, tmax = float(np.nanmin(tide)), float(np.nanmax(tide))
    tide_f = np.where(has_tide, tide, 0.0)

    ds, arrays, t_dim = open_cube(cube.cache_path, cube.bands)
    b03, b08, scl = arrays["B03"], arrays["B08"], arrays["SCL"]
    try:
        H = scl.sizes[scl.dims[1]]
        Wd = scl.sizes[scl.dims[2]]
        elevation = np.full((H, Wd), np.nan, np.float32)
        obs_count = np.zeros((H, Wd), np.float32)

        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            sl = {scl.dims[1]: slice(y0, y1)}
            g = np.asarray(b03.isel(sl).values, np.float32)
            n = np.asarray(b08.isel(sl).values, np.float32)
            s = np.nan_to_num(np.asarray(scl.isel(sl).values),
                              nan=0.0).astype(np.int16)
            den = g + n
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.where(den != 0, (g - n) / den, np.nan).astype(np.float32)
            clear = np.isin(s, list(clear_classes)) & np.isfinite(ndwi)
            clear &= has_tide[:, None, None]
            nd = np.where(clear, ndwi, np.nan)
            obs_count[y0:y1] = clear.sum(0)

            # ── the paper's two candidate criteria ──────────────────────
            n_ok = np.maximum(clear.sum(0), 1)
            freq = (clear & (ndwi > threshold)).sum(0) / n_ok
            candidate = (freq > freq_range[0]) & (freq < freq_range[1])
            if min_correlation is not None:
                # Pearson NDWI-vs-tide per pixel, from plain sums: the
                # obvious formulation materialises several (T, y, x) float64
                # temporaries and is twice as slow for the same answer.
                Yv = np.where(clear, ndwi, 0.0).astype(np.float32)
                okf = clear.astype(np.float32)
                n_ = np.maximum(okf.sum(0), 1)
                Sx = np.tensordot(tide_f, okf, axes=([0], [0]))
                Sxx = np.tensordot(tide_f ** 2, okf, axes=([0], [0]))
                Sy = Yv.sum(0)
                Syy = (Yv * Yv).sum(0)
                Sxy = np.tensordot(tide_f, Yv, axes=([0], [0]))
                cov = Sxy - Sx * Sy / n_
                vx = Sxx - Sx * Sx / n_
                vy = Syy - Sy * Sy / n_
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.where((vx > 0) & (vy > 0),
                                    cov / np.sqrt(np.maximum(vx * vy, 1e-12)),
                                    0.0)
                candidate &= corr >= min_correlation
                del Yv, okf

            hh = y1 - y0
            # ── rolling median per tide window ──────────────────────────
            # Their geometry: radius = window_prop x range (not half of it),
            # spacing = range / n, and the sweep starts `window_offset`
            # windows BELOW the lowest tide so the dry end is covered.
            n_int = int(n_windows) + window_offset
            med_ndwi = np.full((n_int, hh, Wd), np.nan, np.float32)
            med_tide = np.full((n_int, hh, Wd), np.nan, np.float32)
            radius = window_frac * (tmax - tmin)
            spacing = (tmax - tmin) / n_windows
            for k, i in enumerate(range(-window_offset, int(n_windows))):
                tc = tmin + i * spacing
                sel = has_tide & (tide >= tc - radius) & (tide <= tc + radius)
                if not sel.any():
                    continue
                block = nd[sel]
                cnt = np.isfinite(block).sum(0)
                with np.errstate(invalid="ignore"):
                    m_ = np.nanmedian(block, axis=0)
                # A window with too few observations says nothing.
                enough = cnt >= min_count
                med_ndwi[k] = np.where(enough, m_, np.nan)
                med_tide[k] = np.where(enough, np.median(tide[sel]), np.nan)

            # ── interpolate to a denser interval axis ───────────────────
            if interp_intervals:
                src = np.arange(n_int, dtype=np.float64)
                dst = np.linspace(0, n_int - 1, int(interp_intervals))
                med_ndwi = _interp_axis0(med_ndwi, src, dst)
                med_tide = _interp_axis0(med_tide, src, dst)

            # ── trailing rolling mean, exactly as they smooth ───────────
            if smooth_radius:
                med_ndwi = _rolling_mean0(med_ndwi, smooth_radius, min_periods)
                med_tide = _rolling_mean0(med_tide, smooth_radius, min_periods)

            # Elevation = the LOWEST tide at which the smoothed median reads
            # wet. Pixels whose threshold lands on either end of their own
            # observed range are always-dry or always-wet, not intertidal.
            with np.errstate(invalid="ignore"):
                wet_tide = np.where(med_ndwi > threshold, med_tide, np.nan)
                z = np.nanmin(wet_tide, axis=0)
                t_hi = np.nanmax(med_tide, axis=0)
                t_lo = np.nanmin(med_tide, axis=0)
            good = (np.isfinite(z) & (z < t_hi) & (z > t_lo)
                    & (obs_count[y0:y1] >= min_obs) & candidate)
            elevation[y0:y1] = np.where(good, z, np.nan).astype(np.float32)
            if verbose:
                import time as _t
                print(f"[step] filas {y1}/{H}  "
                      f"{_t.strftime('%H:%M:%S')}", flush=True)

        transform, crs = grid_from_dataset(ds)
    finally:
        ds.close()

    if erode_upper:
        # A pixel on the landward edge is partly permanent land, so it needs
        # a higher tide before it reads as flooded and its elevation comes
        # out too high. The paper removes that fringe rather than model it.
        from scipy import ndimage

        has = np.isfinite(elevation)
        inner = ndimage.binary_erosion(has, np.ones((3, 3), bool),
                                       border_value=1)
        edge = has & ~inner
        upper = edge & (elevation > np.nanmedian(elevation[has]))
        elevation = np.where(upper, np.nan, elevation).astype(np.float32)

    valid = np.isfinite(elevation)
    if clip_mask is not None:
        m = np.zeros_like(valid)
        h = min(m.shape[0], clip_mask.shape[0])
        w = min(m.shape[1], clip_mask.shape[1])
        m[:h, :w] = clip_mask[:h, :w]
        valid &= m
        elevation = np.where(valid, elevation, np.nan).astype(np.float32)

    nan_like = np.full(elevation.shape, np.nan, np.float32)
    return ElevationResult(
        mu=elevation, sigma=nan_like.copy(), sigma_mu=nan_like.copy(),
        valid=valid, z_fine=None, scale=1, transform=transform, crs=crs,
        rmse=nan_like.copy(), n_obs=obs_count, method="step",
        epoch=epoch_label,
    )
