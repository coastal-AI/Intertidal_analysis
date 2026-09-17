"""C0 — Granadeiro et al. (2021), implemented to the letter.

*Using Sentinel-2 Images to Estimate Topography, Tidal-Stage Lags and
Exposure Periods over Large Intertidal Areas*, Remote Sens. 13, 320.
The state of the art this project must beat honestly, so it is implemented
from the paper's own recipe, not approximated:

1.  scene INTER-CALIBRATION: major-axis regression of every scene's NIR
    against one reference scene, fitted only on stable pixels (open sea
    NIR < 0.05, land NIR > 0.2, intertidal excluded), scenes corrected
    with the coefficients (their Fig. 2);
2.  water height per scene by their Eq. (1): cosine interpolation between
    the bracketing low- and high-water marks at ONE reference location;
3.  pixel elevation by their Eq. (4): 4-parameter logistic of the
    per-pixel standardised NIR against water height — the elevation is
    the INFLECTION; non-converging pixels dropped, as they do;
4.  tide-stage lags by their rising/ebbing procedure: two separate
    logistics per sampled pixel, a lag grid of -90..+90 min step 5, the
    lag chosen to MINIMISE |h_ebb - h_rise| (their Figs. 4-5), sampled at
    pixels whose preliminary elevation sits near the mean water height
    (±0.25 m), then extended with a thin-plate spline on lon/lat;
5.  final DEM re-estimated with the per-pixel lag; exposure by Eq. (5).

Declared substitutions (unavoidable, and stated in any comparison table):
Sen2Cor L2A instead of ACOLITE (the cube is already L2A); scipy
least-squares instead of R nplr's Newton-Raphson; scipy thin-plate RBF
with k-fold-CV smoothing instead of mgcv's GCV; the lag sample scaled to
this flat's size (their 50,000 pixels served a 100-km archipelago).

Run:  python -m experiments.sota_granadeiro          (villaviciosa)
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scipy.optimize import least_squares

LAG_GRID_MIN = np.arange(-90.0, 91.0, 5.0)      # their sequence exactly


# ── 1 · inter-calibration ────────────────────────────────────────────────
def major_axis(x, y):
    """lmodel2-style major-axis regression: slope and intercept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    sxx, syy = x.var(), y.var()
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) \
        / (2 * sxy)
    return slope, y.mean() - slope * x.mean()


def intercalibrate(NIR, calib, ref_idx):
    """Correct every scene's NIR into the reference scene's units."""
    T = NIR.shape[0]
    ref = calib[ref_idx]
    out = np.array(NIR, np.float32)
    coeffs = np.full((T, 2), np.nan)
    for t in range(T):
        ok = np.isfinite(calib[t]) & np.isfinite(ref)
        if ok.sum() < 200 or t == ref_idx:
            coeffs[t] = (1.0, 0.0)
            continue
        m, b = major_axis(ref[ok], calib[t][ok])
        if not np.isfinite(m) or abs(m) < 0.2:
            m, b = 1.0, 0.0
        coeffs[t] = (m, b)
        out[t] = (out[t] - b) / m
    return out, coeffs


# ── 2 · their Eq. (1): cosine interpolation between tide extremes ────────
def tide_extremes(times, heights):
    """(t, h) of every local high/low water of a dense tide series."""
    h = np.asarray(heights, float)
    d = np.sign(np.diff(h))
    turn = np.flatnonzero(np.diff(d) != 0) + 1
    return times[turn], h[turn]


def hsat_eq1(t_sat, ext_t, ext_h):
    """Water height at t_sat from the bracketing extremes (their Eq. 1)."""
    t_sat = np.asarray(t_sat)
    pos = np.searchsorted(ext_t, t_sat)
    pos = np.clip(pos, 1, len(ext_t) - 1)
    t1, t2 = ext_t[pos - 1], ext_t[pos]
    h1, h2 = ext_h[pos - 1], ext_h[pos]
    frac = (np.cos(np.pi * ((t_sat - t1) / (t2 - t1))) + 1.0) / 2.0
    return h2 + (h1 - h2) * frac


# ── 3 · their Eq. (4): the 4-parameter logistic ──────────────────────────
def fit_logistic4(h, rho):
    """(bottom, top, a, h0) by least squares; None when not converged —
    their rule: a pixel that does not converge is outside the intertidal."""
    ok = np.isfinite(h) & np.isfinite(rho)
    h, rho = h[ok], rho[ok]
    if len(h) < 12 or np.ptp(h) < 0.5:
        return None

    def resid(p):
        bot, top, a, h0 = p
        return bot + (top - bot) / (1.0 + np.exp(-a * (h - h0))) - rho

    p0 = (np.percentile(rho, 95), np.percentile(rho, 5), -3.0,
          float(np.median(h)))
    # parameter order (bottom, top, ...) with bottom>top and a<0 both
    # describe the same DECREASING curve; leave the optimiser free and
    # only demand a monotone-decreasing result afterwards
    try:
        r = least_squares(resid, p0, max_nfev=200)
    except Exception:
        return None
    bot, top, a, h0 = r.x
    decreasing = a * (top - bot) < 0
    if (not r.success or not decreasing
            or not (h.min() - 1.0 < h0 < h.max() + 1.0)):
        return None
    return float(bot), float(top), float(a), float(h0)


def standardise(x):
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


# ── 4 · their rising/ebbing lag search ───────────────────────────────────
def pixel_lag(t_sat_s, rho, rising, ext_t, ext_h, lag_grid=LAG_GRID_MIN):
    """The lag minimising |h0_ebb - h0_rise| (their Figs. 4-5)."""
    best = (np.inf, np.nan)
    for lag in lag_grid:
        h = hsat_eq1(t_sat_s - lag * 60.0, ext_t, ext_h)
        f_r = fit_logistic4(h[rising], rho[rising])
        f_e = fit_logistic4(h[~rising], rho[~rising])
        if f_r is None or f_e is None:
            continue
        gap = abs(f_r[3] - f_e[3])
        if gap < best[0]:
            best = (gap, float(lag))
    return best[1], best[0]


def tps_lag_map(xy_known, lags, xy_all, seed=3):
    """Thin-plate spline on coordinates, smoothing by 5-fold CV (their GAM
    with mgcv's GCV, in scipy's vocabulary)."""
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lags))
    folds = np.array_split(idx, 5)
    best = (np.inf, 1.0)
    for smooth in (0.1, 1.0, 10.0, 100.0, 1000.0):
        err = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            f = RBFInterpolator(xy_known[tr], lags[tr],
                                kernel="thin_plate_spline",
                                smoothing=smooth)
            err.append(np.mean((f(xy_known[te]) - lags[te]) ** 2))
        m = float(np.mean(err))
        if m < best[0]:
            best = (m, smooth)
    f = RBFInterpolator(xy_known, lags, kernel="thin_plate_spline",
                        smoothing=best[1])
    return f(xy_all), best[1]


def exposure_eq5(h_pixel, h_hw, h_lw, cycle_h=12.40):
    """Their Eq. (5): exposure hours per average tide cycle."""
    x = 2.0 * (h_pixel - h_lw) / (h_hw - h_lw) - 1.0
    return cycle_h * (1.0 - np.arccos(np.clip(x, -1, 1)) / np.pi)


# ── runner ───────────────────────────────────────────────────────────────
def main(site="villaviciosa", n_lag_px=2500, seed=13, years=None,
         boundary=None, tag="", base="marea_demo_extract.npz",
         gran="granadeiro_extract.npz", out_dir=".", cloud_rule="scene",
         px_stride=1):
    """``years=(y0, y1)`` restricts the archive to one epoch — the
    epoch-matched variant for a fair comparison against products fitted on
    2023-2025 (morphology migrates; HSR limits itself deliberately, and
    the SOTA must be scored under the same rule as well as its own).

    ``boundary=(times, heights)`` replaces the EOT20 series that plays
    Bubaque's role in their Eq. (1) — a tide gauge, or a consensus of
    gauges; ``tag`` names the output file. ``base``/``gran`` point at the
    NDWI extraction (keep, C, dates, s_km, bbox, shape) and the NIR
    extraction of the SAME cube; ``out_dir`` receives the product.

    ``cloud_rule``: "scene" is theirs (whole frame < 10 % cloud, fully
    covered). ``("intertidal", f)`` is a DECLARED SUBSTITUTION for a box
    whose frame is mostly sea and far land: a scene enters when at most a
    fraction ``f`` of the intertidal pixels are cloudy. On the Scheldt the
    frame rule leaves 11 rising scenes and their logistic needs 12.

    ``px_stride``: fit the per-pixel logistics on every k-th intertidal
    pixel only (the lag sample and the TPS lag map still cover them all);
    a DECLARED shortcut for very large flats (the Ems has 478k pixels at
    20 m, 13 h at k=1). The product is NaN on the skipped pixels."""
    import pandas as pd

    from pyintertidal import marea, overpass
    import pyintertidal as pit
    pit.net.use_system_certificates()

    t0 = time.time()
    base = np.load(base, allow_pickle=True)
    gran = np.load(gran, allow_pickle=True)
    keep, (H, W) = base["keep"], (int(v) for v in base["shape"])
    H, W = (int(v) for v in base["shape"])
    dates, s_km = base["dates"], base["s_km"]
    bbox = base["bbox"]
    if bbox.dtype == object:                 # dict saved by marea.extract
        bb = bbox.item()
        bbox = np.array([bb["west"], bb["south"], bb["east"], bb["north"]])
    C = base["C"].astype(bool)
    NIR, calib = gran["NIR"], gran["calib"]
    ref_idx = int(gran["ref_idx"])
    bad, nodata = gran["bad_frac"], gran["nodata_frac"]

    # their scene rule: cloud cover below 10 % (plus full coverage)
    if cloud_rule == "scene":
        use = (bad <= 0.10) & (nodata <= 0.02)
    else:
        f_bad = 1.0 - C.mean(axis=1)
        use = (f_bad <= float(cloud_rule[1])) & (nodata <= 0.02)
        print(f"cloud rule: <= {cloud_rule[1]:.0%} cloudy over the intertidal "
              f"pixels (declared substitution for their frame rule)")
    if years is not None:
        yr = np.array([int(d[:4]) for d in dates])
        use &= (yr >= years[0]) & (yr <= years[1])
    print(f"{use.sum()} of {len(dates)} scenes pass their <10% cloud rule"
          + (f" within {years[0]}-{years[1]}" if years else ""))

    times = overpass.get_overpass_times(
        {"west": float(bbox[0]), "south": float(bbox[1]),
         "east": float(bbox[2]), "north": float(bbox[3]),
         "crs": "EPSG:4326"},
        ("2016-01-01", "2025-12-31"), verbose=False)
    has_t = np.array([d in times for d in dates])
    print(f"{int(has_t.sum())} of {len(dates)} scenes have a catalogue "
          f"overpass time", flush=True)
    use &= has_t
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[d] for d in dates[use]])).tz_localize(None)
    NIRu = np.array(NIR[use], np.float32)
    calibu = calib[use]
    Cu = C[use]
    ref_pos = int(np.flatnonzero(np.flatnonzero(use) == ref_idx)[0]) \
        if use[ref_idx] else 0

    # 1 · inter-calibration
    NIRc, coeffs = intercalibrate(NIRu, calibu, ref_pos)
    NIRc[~Cu] = np.nan                       # clouds out, per pixel
    print(f"inter-calibrated {use.sum()} scenes "
          f"(median slope {np.nanmedian(coeffs[:, 0]):.3f})")

    # 2 · dense reference tide + extremes at the mouth (EOT20 = the
    # single-reference-location role Bubaque plays in the paper)
    lon = 0.5 * (float(bbox[0]) + float(bbox[2]))
    lat = 0.5 * (float(bbox[1]) + float(bbox[3]))
    tides = pit.TideService(model="EOT20", directory="./tide_models",
                            location=(lat, lon))
    aoi = pit.sites.get(site)
    if boundary is None:
        st, sh = tides.series(aoi, "2016-01-01", "2026-01-01",
                              every_hours=0.25)
        st = pd.DatetimeIndex(st).tz_localize(None)
    elif hasattr(boundary, "levels"):
        # a BoundaryProvider: sample it every 15 min over the period
        st = pd.date_range("2016-01-01", "2026-01-01", freq="15min")
        sh = np.asarray(boundary.levels(st), float)
    else:
        st, sh = boundary
        st = pd.DatetimeIndex(st)
        if st.tz is not None:
            st = st.tz_localize(None)
        sh = np.asarray(sh, float)
    ext_t_dt, ext_h = tide_extremes(st, sh)
    ext_t = ext_t_dt.view("int64") / 1e9          # seconds
    t_sat_s = t_real.view("int64") / 1e9
    h_scene = hsat_eq1(t_sat_s, ext_t, ext_h)
    # rising flag from the same series (their ebb/rise split)
    h_before = hsat_eq1(t_sat_s - 1800.0, ext_t, ext_h)
    rising = h_scene > h_before
    print(f"Eq.(1) heights: range [{np.nanmin(h_scene):+.2f}, "
          f"{np.nanmax(h_scene):+.2f}] m · {int(rising.sum())} rising / "
          f"{int((~rising).sum())} ebbing scenes")

    # 3 · preliminary DEM: logistic per pixel on standardised NIR
    P = NIRc.shape[1]
    rho = np.empty_like(NIRc)
    for j in range(P):
        rho[:, j] = standardise(NIRc[:, j])
    # checkpoint: a killed run (network, session) resumes here instead of
    # paying the two slow stages again
    ck_path = os.path.join(out_dir, f"granadeiro_checkpoint{'_' + tag if tag else ''}.npz")
    ck = dict(np.load(ck_path, allow_pickle=True)) if os.path.exists(ck_path) else {}
    if "prelim" in ck and len(ck["prelim"]) == P:
        prelim = ck["prelim"]
        kept = int(np.isfinite(prelim).sum())
        print(f"preliminary DEM: {kept}/{P} px (from checkpoint)")
    else:
        prelim = np.full(P, np.nan)
        kept = 0
        for j in range(0, P, int(px_stride)):
            f = fit_logistic4(h_scene, rho[:, j])
            if f is not None:
                prelim[j] = f[3]
                kept += 1
            if (j + 1) % 4000 == 0:
                print(f"  prelim DEM {j + 1}/{P} px ({kept} converged)",
                      flush=True)
        print(f"preliminary DEM: {kept}/{P} px converged "
              f"({time.time() - t0:.0f} s)")
        ck = {"prelim": prelim}
        np.savez_compressed(ck_path, **ck)

    # 4 · lag sample: elevation near the mean water height ±0.25 (theirs)
    h_mean = float(np.nanmean(h_scene))
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(np.abs(prelim - h_mean) <= 0.25)
    samp = rng.choice(pool, min(n_lag_px, len(pool)), replace=False)
    print(f"lag sample: {len(samp)} px near h_mean = {h_mean:+.2f} m")
    if "lags" in ck and len(ck["lags"]) == len(samp):
        lags, gaps = ck["lags"], ck["gaps"]
        print(f"lags: {int(np.isfinite(lags).sum())} px (from checkpoint)")
    else:
        lags, gaps = np.full(len(samp), np.nan), np.full(len(samp), np.nan)
        for k, j in enumerate(samp):
            lags[k], gaps[k] = pixel_lag(t_sat_s, rho[:, j], rising,
                                         ext_t, ext_h)
            if (k + 1) % 250 == 0:
                print(f"  lags {k + 1}/{len(samp)} "
                      f"(median so far {np.nanmedian(lags[:k + 1]):+.0f} min)",
                      flush=True)
        ck.update({"lags": lags, "gaps": gaps})
        np.savez_compressed(ck_path, **ck)
    ok = np.isfinite(lags)
    print(f"lags estimated on {ok.sum()} px · "
          f"median {np.nanmedian(lags):+.0f} min")

    # 5 · thin-plate lag map over all intertidal pixels
    rows, cols = keep // W, keep % W
    xy_all = np.column_stack([cols, rows]).astype(float)
    lag_all, smooth = tps_lag_map(xy_all[samp[ok]], lags[ok], xy_all)
    print(f"lag map: TPS smoothing {smooth} · "
          f"range [{lag_all.min():+.0f}, {lag_all.max():+.0f}] min")

    # 6 · final DEM with the per-pixel lag
    final = np.full(P, np.nan)
    for j in range(0, P, int(px_stride)):
        h_j = hsat_eq1(t_sat_s - lag_all[j] * 60.0, ext_t, ext_h)
        f = fit_logistic4(h_j, rho[:, j])
        if f is not None:
            final[j] = f[3]
        if (j + 1) % 4000 == 0:
            print(f"  final DEM {j + 1}/{P} px", flush=True)

    # 7 · exposure with the mean extremes of the used scenes (their rule)
    hw = float(np.nanmean(ext_h[ext_h > np.nanmedian(ext_h)]))
    lw = float(np.nanmean(ext_h[ext_h < np.nanmedian(ext_h)]))
    expo = exposure_eq5(final, hw, lw)

    out_name = ("granadeiro_products.npz" if years is None
                else f"granadeiro_products_{years[0]}-{years[1]}.npz")
    if tag:
        out_name = out_name.replace(".npz", f"_{tag}.npz")
    out_name = os.path.join(out_dir, out_name)
    np.savez_compressed(
        out_name, prelim=prelim, final=final,
        lag_all=lag_all, lag_sample_idx=samp, lag_sample=lags,
        lag_gap=gaps, exposure=expo, h_scene=h_scene, rising=rising,
        use=use, keep=keep, shape=np.array([H, W]), s_km=s_km)
    print(f"-> {out_name} ({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "villaviciosa"
    yrs = ((int(sys.argv[2]), int(sys.argv[3]))
           if len(sys.argv) > 3 else None)
    main(site, years=yrs)
