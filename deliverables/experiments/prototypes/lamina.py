"""Joint inversion of the water surface: the level is estimated, not assumed.

Every published method for intertidal topography from optical archives — the
DEA step, NIDEM, and the four per-pixel sigmoid papers — assumes the water
level at each acquisition is a single number, the same everywhere in the
scene, taken from an ocean tide model. Inside an estuary that is false, and
this project measured it: the relief compresses upstream by the same amount
in two independent estimators, so the error is in what they share.

Here the level is an unknown. It is written as a profile along the channel,

    h_t(p) = h0_t + gamma_t * s_p

with s_p the geodesic distance from open water, through ground water can
actually reach (canal.py). One new unknown per date.

Why this is identifiable when the naive version was not
-------------------------------------------------------
The affine result established earlier says the fit is invariant under
h -> alpha*h + c: datum and scale cannot be recovered from imagery. A free
h_t per scene is exactly degenerate with a global shift of z, which is why
the earlier tide-gauge attempt traded information between the two and had to
be abandoned.

gamma_t is not degenerate. z is not constrained to be linear in s, so no
global transformation of z can imitate a slope along the channel. The
unidentifiable subspace is exactly two-dimensional — constant and scale — and
gamma is orthogonal to it. So h0_t stays FIXED at the ocean model's value,
which is trustworthy at the mouth, and only the inland deviation is inferred.
The affine theorem is not a wall; it is a specification of the minimum that
must come from outside.

Second piece of physics: flooding needs a path
----------------------------------------------
Everyone assumes wet iff z < h — a bathtub. Water has to arrive. A pixel is
reachable at level h only if some path from the sea stays below h, so the
level at which it first floods is the minimax elevation along the best path,
which is exactly what depression filling computes. One morphological
reconstruction per iteration over the whole grid, not one per scene.

Nothing here is validated yet. That is what the gates in puertas.py are for.
"""
import os
import sys
import json
import time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
from scipy.special import erf

from pyintertidal.net import use_system_certificates
from pyintertidal.elevation import _fit_block
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

N_BANDS = 24            # channel bands sharing one tide vector
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
MIN_B = 0.15
GAMMA_GRID = np.linspace(-0.8, 0.8, 33)   # m/km
GAMMA_PIX = 9000        # pixels used in the scene step
PIX_CHUNK = 3000        # keep every (T, chunk) temporary small: RAM is 1 GB
LAMBDA_GAMMA = 0.02     # damping, in units of the residual sum
HUBER_K = 2.5


def phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


# ── the pixel step ───────────────────────────────────────────────────────
def band_edges(s_km, n_bands):
    """Quantile bands, so each carries a similar number of pixels.

    Width matters only through gamma*(s - s_band), which must stay small
    against sigma. With gamma of order 0.2 m/km and sigma of order 0.1-0.3 m,
    bands must be well under half a kilometre; quantile bands are far
    narrower than that except in the sparse upstream tail.
    """
    q = np.linspace(0, 1, n_bands + 1)
    e = np.unique(np.quantile(s_km[np.isfinite(s_km)], q))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def pixel_step(Y, C, h0, gamma, s_km, edges, mu_grid, sg_grid):
    """Fit (a, b, z, sigma) per pixel with the water surface held fixed.

    Each band shares one tide vector, so the closed form in _fit_block applies
    unchanged — 24 calls instead of 40 000. Without this the per-pixel tide
    would make every iteration a (T x P x grid) computation and the method
    would not run at all.
    """
    P = Y.shape[1]
    a = np.full(P, np.nan); b = np.full(P, np.nan)
    z = np.full(P, np.nan); sg = np.full(P, np.nan)
    nobs = np.zeros(P)
    for i in range(len(edges) - 1):
        sel = np.where((s_km > edges[i]) & (s_km <= edges[i + 1]))[0]
        if len(sel) == 0:
            continue
        s_band = float(np.median(s_km[sel]))
        tide_b = h0 + gamma * s_band
        for j in range(0, len(sel), PIX_CHUNK):
            idx = sel[j:j + PIX_CHUNK]
            aa, bb, mm, ss, _, NN = _fit_block(
                Y[:, idx], C[:, idx], tide_b, mu_grid, sg_grid)
            a[idx], b[idx], z[idx], sg[idx], nobs[idx] = aa, bb, mm, ss, NN
    return a, b, z, sg, nobs


# ── the scene step ───────────────────────────────────────────────────────
def gamma_from_tide(theta, h0, rng_t):
    """gamma as a function of the tidal state: three numbers, not 465.

    A free gamma per date turned out to be badly under-determined — 23 % of
    dates ended pinned at the edge of the search grid and 44 % sat exactly at
    zero, which is what an unidentifiable parameter looks like. The cause is
    geometric: the median pixel lies 158 m along the channel, so a physical
    slope of 0.1 m/km moves it by 1.6 cm, far under the NDWI noise. Only the
    thin upstream tail carries information, and it is the noisiest part of
    the scene.

    The physics never asked for a free parameter per date anyway. The slope of
    the water surface in an estuary is set by the tidal state — how high the
    water is and how big the tide is that day — so:

        gamma(t) = g0 + g1 * h0_t + g2 * (range_t - mean range)

    g1 is the interesting one: it is exactly the coefficient that compresses
    relief, since z_hat = (z - g0*s) / (1 + g1*s).
    """
    g0, g1, g2 = theta
    return g0 + g1 * h0 + g2 * (rng_t - rng_t.mean())


def scene_step_tidal(Y, C, h0, rng_t, a, b, z, sg, s_km, good):
    """Fit the three tidal coefficients against every scene at once."""
    from scipy.optimize import minimize
    idx = np.where(good)[0]
    if len(idx) > GAMMA_PIX:
        idx = np.sort(np.random.default_rng(2).choice(idx, GAMMA_PIX,
                                                      replace=False))
    ap, bp, zp, sgp, sp = a[idx], b[idx], z[idx], sg[idx], s_km[idx]

    def cost(theta):
        gam = gamma_from_tide(theta, h0, rng_t)
        tot = 0.0
        for j in range(0, len(idx), PIX_CHUNK):
            k = idx[j:j + PIX_CHUNK]
            sl = slice(j, j + len(k))
            hh = h0[:, None] + gam[:, None] * sp[None, sl]
            pred = ap[None, sl] + bp[None, sl] * phi(
                (hh - zp[None, sl]) / np.maximum(sgp[None, sl], 1e-3))
            r = (Y[:, k] - pred) * (C[:, k] > 0)
            s = np.abs(r)
            d = HUBER_K * 0.05
            tot += np.where(s <= d, 0.5 * r * r, d * (s - 0.5 * d)).sum()
        return float(tot)

    res = minimize(cost, np.zeros(3), method="Powell",
                   options={"maxfev": 120, "xtol": 1e-4, "ftol": 1e-6})
    return np.asarray(res.x), float(res.fun)


def scene_step(Y, C, h0, a, b, z, sg, s_km, good, gamma_prev):
    """One gamma per scene, by robust 1-D search. Cheap and non-degenerate.

    gamma_t carries a single degree of freedom against tens of thousands of
    pixels, so it cannot absorb pixel-level error — which is precisely the
    failure that sank the earlier free-level version.
    """
    idx = np.where(good)[0]
    if len(idx) > GAMMA_PIX:
        idx = np.sort(np.random.default_rng(1).choice(idx, GAMMA_PIX,
                                                      replace=False))
    ap, bp, zp, sgp, sp = a[idx], b[idx], z[idx], sg[idx], s_km[idx]
    T = Y.shape[0]
    cost = np.zeros((len(GAMMA_GRID), T))
    for gi, g in enumerate(GAMMA_GRID):
        for j in range(0, len(idx), PIX_CHUNK):
            k = idx[j:j + PIX_CHUNK]
            sl = slice(j, j + len(k))
            hh = h0[:, None] + g * sp[None, sl]
            pred = ap[None, sl] + bp[None, sl] * phi(
                (hh - zp[None, sl]) / np.maximum(sgp[None, sl], 1e-3))
            r = (Y[:, k] - pred) * (C[:, k] > 0)
            # Huber: one bad scene must not drag its own gamma
            s = np.abs(r)
            d = HUBER_K * 0.05
            cost[gi] += np.where(s <= d, 0.5 * r * r,
                                 d * (s - 0.5 * d)).sum(axis=1)
    scale = np.maximum(cost.min(axis=0), 1e-9)
    cost = cost / scale + LAMBDA_GAMMA * (
        GAMMA_GRID[:, None] - gamma_prev[None, :]) ** 2
    return GAMMA_GRID[np.argmin(cost, axis=0)]


# ── connectivity ─────────────────────────────────────────────────────────
def flood_threshold(z_map, sea, valid):
    """Level at which each pixel first floods, given it needs a path.

    This is the minimax elevation over paths from the sea, which is exactly
    what depression filling returns. Pixels sitting in a pit get a threshold
    ABOVE their own elevation — the bathtub model has no way to express that.
    """
    from skimage.morphology import reconstruction
    zz = np.where(valid, z_map, np.nan)
    hi = np.nanmax(zz) + 1.0
    mask = np.where(np.isfinite(zz), zz, hi)
    seed = np.full_like(mask, hi)
    seed[sea] = mask[sea]
    filled = reconstruction(seed, mask, method="erosion")
    return np.where(valid, filled, np.nan)


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    t_start = time.time()
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    ch = np.load(os.path.join(SC, "canal.npz"))
    s_km = ch["s_keep"].astype(np.float64) / 1000.0

    cache = os.path.join(SC, "mareas_villa.npz")
    rng_all = None
    if os.path.exists(cache):
        cc = np.load(cache)
        h0_all, rng_all = cc["tide"], cc["day_range"]
    else:
        aoi = pit.sites.get("villaviciosa")
        lat_c, lon_c = aoi.centroid
        times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
        h0_all = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                             directory="tide_models", crs="EPSG:4326",
                             extrapolate=True, cutoff=np.inf,
                             parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(h0_all)
    if rng_all is not None:
        ep &= np.isfinite(rng_all)
    Y = np.nan_to_num(Y[ep], nan=0.0).astype(np.float64)
    C = C[ep].astype(np.float64)
    h0 = h0_all[ep]
    rng_t = rng_all[ep] if rng_all is not None else np.zeros_like(h0)
    dates_ep = dates[ep]

    # pixels with no channel distance are hydraulically isolated: the profile
    # is undefined for them, so they are carried at gamma = 0 and reported
    fin_s = np.isfinite(s_km)
    s_use = np.where(fin_s, s_km, 0.0)
    print(f"{Y.shape[0]} escenas · {Y.shape[1]:,} px "
          f"({int((~fin_s).sum())} sin distancia de canal)")
    print(f"marea en la boca {h0.min():+.2f} a {h0.max():+.2f} m · "
          f"canal 0 a {np.nanmax(s_km):.2f} km")

    edges = band_edges(s_use, N_BANDS)
    print(f"{len(edges)-1} bandas de canal, de {edges[1]*1000:.0f} m "
          f"a {edges[-1]*1000:.0f} m\n")

    # One global mu grid, wide enough for every band's shifted tide, so that
    # gamma = 0 reproduces the plain fit EXACTLY rather than approximately.
    span = float(np.abs(GAMMA_GRID).max() * np.nanmax(s_km))
    mu_grid = np.linspace(h0.min() - span, h0.max() + span, MU_POINTS)
    mu_grid_flat = np.linspace(h0.min(), h0.max(), MU_POINTS)

    gamma = np.zeros(len(h0))
    hist = []
    N_ITER = int(os.environ.get("N_ITER", 4))
    SYNTH = os.environ.get("SYNTH")   # parametric bootstrap under a FLAT surface
    MODE = os.environ.get("MODE", "tidal")   # "tidal" (3 params) or "per_scene"
    theta = np.zeros(3)
    print(f"modo del scene step: {MODE}\n")
    for it in range(N_ITER + 1):
        grid = mu_grid_flat if it == 0 else mu_grid
        a, b, z, sg, nobs = pixel_step(Y, C, h0, gamma, s_use, edges,
                                       grid, SG_GRID)
        good = (nobs >= MIN_OBS) & (b > MIN_B) & np.isfinite(z)
        rec = {"iter": it, "n_good": int(good.sum()),
               "gamma_mean": float(gamma.mean()),
               "gamma_sd": float(gamma.std()),
               "z_mean": float(np.nanmean(z[good]))}
        print(f"iter {it}: {rec['n_good']:,} px utiles · "
              f"gamma media {rec['gamma_mean']:+.3f} sd {rec['gamma_sd']:.3f} "
              f"m/km · {(time.time()-t_start)/60:.1f} min", flush=True)
        hist.append(rec)
        if it == 0:
            np.savez_compressed(os.path.join(SC, "lamina_base.npz"),
                                a=a, b=b, z=z, sg=sg, nobs=nobs, good=good)
            if SYNTH:
                # Rebuild the observations from this fit with a PERFECTLY FLAT
                # water surface, then let the solver loose on them. Any gamma
                # it reports now is manufactured by the estimator, because the
                # data it is reading contain none. This is the only honest way
                # to know whether a fitted gamma is physics or self-portrait:
                # correlations between gamma and the tide can arise from the
                # fit alone, since near high water almost everything is
                # submerged, the sigmoid saturates and gamma loses its
                # leverage.
                rng = np.random.default_rng(int(SYNTH))
                pred = a[None, :] + b[None, :] * phi(
                    (h0[:, None] - z[None, :])
                    / np.maximum(sg[None, :], 1e-3))
                # Noise PER PIXEL, not one global sigma. Upstream pixels are
                # turbid and thinly observed, so their residuals are larger —
                # and they are also the ones with the long lever arm that
                # dominates gamma. A homoscedastic null would be quieter than
                # the real data exactly where it matters, and would flatter
                # the result.
                w = C > 0
                resid = np.where(w, Y - pred, np.nan)
                d2 = np.where(w, (Y - pred) ** 2, 0.0).sum(0)
                sd_p = np.sqrt(d2 / np.maximum(w.sum(0), 1))
                sd_p = np.where(np.isfinite(sd_p) & (sd_p > 0), sd_p,
                                np.nanmedian(sd_p))
                # Scene-level systematics matter as much as pixel noise. Haze,
                # sun angle and turbidity move a WHOLE scene together, and a
                # null built from independent per-pixel noise cannot produce
                # that — so it would be quieter than the real archive and
                # would flatter the result all over again. The size of the
                # effect is taken straight from the data: the spread of each
                # scene's own median residual.
                scene_off = np.nanmedian(resid, axis=1)
                sd_scene = float(np.nanstd(scene_off))
                print(f"  SINTETICO: lamina plana · ruido por pixel mediana "
                      f"{np.median(sd_p):.4f} p90 {np.percentile(sd_p,90):.4f}"
                      f" · sistematico por escena sd {sd_scene:.4f} NDWI",
                      flush=True)
                Y = np.nan_to_num(
                    pred
                    + rng.normal(0.0, 1.0, Y.shape) * sd_p[None, :]
                    + rng.normal(0.0, sd_scene, (Y.shape[0], 1)),
                    nan=0.0)
        if it == N_ITER:
            break
        if MODE == "tidal":
            theta, cst = scene_step_tidal(Y, C, h0, rng_t, a, b, z, sg,
                                          s_use, good)
            gamma = gamma_from_tide(theta, h0, rng_t)
            print(f"    g0 {theta[0]:+.4f}  g1 {theta[1]:+.4f}  "
                  f"g2 {theta[2]:+.4f}  coste {cst:.1f}", flush=True)
        else:
            gamma = scene_step(Y, C, h0, a, b, z, sg, s_use, good, gamma)

    tag = (f"_{MODE}" if MODE != "per_scene" else "") + \
          (f"_sint{SYNTH}" if SYNTH else "")
    np.savez_compressed(os.path.join(SC, f"lamina{tag}.npz"),
                        a=a, b=b, z=z, sg=sg, nobs=nobs, good=good,
                        gamma=gamma, h0=h0, rng_t=rng_t, theta=theta,
                        dates=dates_ep, s_km=s_km, keep=keep)
    json.dump(hist, open(os.path.join(SC, f"lamina_hist{tag}.json"), "w"),
              indent=1)
    r = np.corrcoef(h0, gamma)[0, 1]
    print(f"\ncorrelacion gamma-h0: {r:+.3f}"
          f"{'   (SINTETICO: deberia ser ~0)' if SYNTH else ''}")
    print(f"guardado lamina{tag}.npz tras {(time.time()-t_start)/60:.1f} min")
    print(f"gamma: min {gamma.min():+.3f} max {gamma.max():+.3f} "
          f"mediana {np.median(gamma):+.3f} m/km")


if __name__ == "__main__":
    main()
