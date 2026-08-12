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
dry to wet. Simpler and more selective (it needs a clean monotonic
transition), included for comparison.

Fit per EPOCH, not per decade
-----------------------------
Intertidal morphology migrates, so fitting ten years at once blurs the
transition. Measured on our data: 87 % of pixels saturate the sigmoid-width
grid at 10 years versus 12 % at 3 years, and the fit uncertainty halves.
Use :func:`epochs` to split the archive and fit the recent epoch.

Benchmark (Ría de Villaviciosa vs IGN 5 m LiDAR, datum bias removed):
HSR RMSE 0.76 m, step 0.85 m, legacy bracketing methods 1.14–3.4 m.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    Cf = C.astype(np.float64)
    Yc = np.where(C, Y, 0.0).astype(np.float64)

    N = Cf.sum(axis=0)
    Sy = Yc.sum(axis=0)
    Syy = (Yc * Yc).sum(axis=0)

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

        out = {k: np.zeros((H, Wd), np.float32)
               for k in ("a", "b", "mu", "sigma", "rmse", "n_obs")}

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
            a, b, mu, sg, rmse, N = _fit_block(Y, C, tide_f, mu_grid, sg_grid)

            for _ in range(int(robust_iters)):
                sg_safe = np.maximum(sg[None], 1e-3)
                phi = 0.5 * (1 + erf((tide_f[:, None] - mu[None])
                                     / (sg_safe * np.sqrt(2))))
                resid = np.abs(Y - (a[None] + b[None] * phi))
                C2 = C & (resid < 2.5 * np.maximum(rmse[None], 0.02))
                a, b, mu, sg, rmse, N = _fit_block(Y, C2, tide_f, mu_grid, sg_grid)

            shp = (hh, Wd)
            out["a"][y0:y1] = a.reshape(shp)
            out["b"][y0:y1] = b.reshape(shp)
            out["mu"][y0:y1] = mu.reshape(shp)
            out["sigma"][y0:y1] = sg.reshape(shp)
            out["rmse"][y0:y1] = rmse.reshape(shp)
            out["n_obs"][y0:y1] = N.reshape(shp)

        transform, crs = grid_from_dataset(ds)
    finally:
        ds.close()

    # Fittable intertidal pixels: enough contrast, enough observations, and an
    # elevation inside the observed tide range (outside it the fit extrapolates).
    rng = tmax - tmin
    valid = ((out["b"] > min_b)
             & (out["mu"] > tmin + 0.02 * rng)
             & (out["mu"] < tmax - 0.02 * rng)
             & (out["n_obs"] >= min_obs))

    # Cramér–Rao uncertainty of the elevation:
    #     var(μ) ≈ rmse² / Σ (∂model/∂μ)² ,  ∂model/∂μ = -b·pdf((h-μ)/σ)/σ
    # Computed per row block so a full (H, W, T) array is never materialised.
    tt = tide[np.isfinite(tide)]
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
    return ElevationResult(
        mu=mu,
        sigma=np.where(valid, s1["sigma"], np.nan).astype(np.float32),
        sigma_mu=s1["sigma_mu"], valid=valid, z_fine=z_fine,
        scale=int(scale), transform=s1["transform"], crs=s1["crs"],
        rmse=s1["rmse"], n_obs=s1["n_obs"], method="hsr", epoch=epoch_label,
    )


def fit_step(cube, tide_heights, dates=None, threshold=0.0, n_windows=100,
             window_frac=0.15, min_obs=5, row_chunk=32, clip_mask=None,
             epoch_label="", clear_classes=W.CLEAR_CLASSES):
    """Fit the DEA-style STEP elevation locally (published baseline).

    For each pixel the NDWI observations are summarised as a rolling median
    over ``n_windows`` tide-height windows (each spanning ``window_frac`` of
    the observed tide range); the elevation is the tide height at which that
    median first crosses ``threshold`` from dry to wet, linearly interpolated
    between the bracketing windows.

    More selective than HSR — a pixel needs a clean, monotonic dry→wet
    transition — and it provides no sub-pixel relief or formal uncertainty,
    but it is the reference method of the published literature.

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
    half = 0.5 * window_frac * (tmax - tmin)
    centres = np.linspace(tmin, tmax, int(n_windows))

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
            nd = np.where(clear, ndwi, np.nan)
            obs_count[y0:y1] = clear.sum(0)

            hh = y1 - y0
            rolling = np.full((int(n_windows), hh, Wd), np.nan, np.float32)
            for k, tc in enumerate(centres):
                sel = has_tide & (tide >= tc - half) & (tide <= tc + half)
                if sel.any():
                    with np.errstate(invalid="ignore"):
                        rolling[k] = np.nanmedian(nd[sel], axis=0)

            wet = rolling >= threshold
            dry_low = rolling[0] < threshold
            wet_high = rolling[-1] >= threshold
            crosses = (dry_low & wet_high & np.isfinite(rolling[0])
                       & np.isfinite(rolling[-1]))
            k1 = np.argmax(wet, axis=0)
            k0 = np.clip(k1 - 1, 0, int(n_windows) - 1)
            r1 = np.take_along_axis(rolling, k1[None], 0)[0]
            r0 = np.take_along_axis(rolling, k0[None], 0)[0]
            d = r1 - r0
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(np.abs(d) > 1e-6, (threshold - r0) / d, 0.0)
            frac = np.clip(np.nan_to_num(frac), 0, 1)
            z = centres[k0] + frac * (centres[k1] - centres[k0])
            good = crosses & (obs_count[y0:y1] >= min_obs)
            elevation[y0:y1] = np.where(good, z, np.nan).astype(np.float32)

        transform, crs = grid_from_dataset(ds)
    finally:
        ds.close()

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
