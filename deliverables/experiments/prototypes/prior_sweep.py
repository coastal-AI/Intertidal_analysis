"""Break the h(t) <-> mu(p) coupling, or admit the method does not work.

Where it stands: with a prior and per-scene haze calibration, inverting the
water level improved agreement with the RTK survey (r 0.845 -> 0.890) but the
compression slope got WORSE (0.812 -> 0.618), and the RMSE gain was not
significant. A worse slope while the NDWI residual falls is the signature of
the two unknowns trading information: the fit buys a lower residual by moving
the water level and the elevations together in a direction that suits the
data and not the ground.

The Cramer-Rao bound says one scene's level is readable to 5 mm against the
harmonic model's 140 mm, so the information is there. The alternation is what
squanders it: every iteration re-fits the pixels on the levels it just
invented, and nothing anchors them.

Four variants, judged on the SAME held-out field pixels, on two axes at once
because either alone is misleading:

  A  model tide                      the baseline to beat
  B  alternating, pixels re-fitted   what we have now
  C  pixels FROZEN after one fit     h is the only free thing
  D  frozen + haze calibration       C plus the scene-offset removal

Success is RMSE significantly better AND the slope not worse. A method that
wins one and loses the other does not belong in the package.
"""
import os, sys, json, time

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
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
N_CAND = 240
PRIOR_SIGMA = 0.20
MIN_PIX_PER_SCENE = 400


def phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def solve_levels(A, Cc, SSy, a, b, mu, sg, cand, h_prior, lam):
    z = (cand[:, None] - mu[None, :]) / np.maximum(sg[None, :], 1e-3)
    pred = (a[None, :] + b[None, :] * phi(z)).astype(np.float32)
    loss = (SSy[:, None] - 2.0 * (A @ pred.T) + (Cc @ (pred * pred).T))
    loss = loss + lam * (cand[None, :] - h_prior[:, None]) ** 2
    k = np.argmin(loss, axis=1)
    k1 = np.clip(k, 1, len(cand) - 2)
    rows = np.arange(loss.shape[0])
    y0, y1, y2 = loss[rows, k1 - 1], loss[rows, k1], loss[rows, k1 + 1]
    den = y0 - 2 * y1 + y2
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    return cand[k1] + np.clip(shift, -1, 1) * (cand[1] - cand[0])


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    base = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=base, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide)
    Yf = np.nan_to_num(Y[ep], nan=0.0).astype(np.float64)
    Cf = C[ep].astype(np.float64)
    h_model = tide[ep]
    print(f"{Yf.shape[0]} escenas x {Yf.shape[1]:,} px", flush=True)

    A = (Yf * Cf).astype(np.float32)
    SSy = (Yf * A).sum(axis=1)
    usable = Cf.sum(axis=1) >= MIN_PIX_PER_SCENE
    cand = np.linspace(h_model.min() - 0.3, h_model.max() + 0.3, N_CAND)

    def fit_pixels(h):
        grid = np.linspace(np.nanmin(h), np.nanmax(h), MU_POINTS)
        return _fit_block(Yf, Cf, h, grid, SG_GRID)

    def haze_offset(h, a, b, mu, sg, good):
        zz = (h[:, None] - mu[None, good]) / np.maximum(sg[None, good], 1e-3)
        blind = (np.abs(zz) > 2.5) & (Cf[:, good] > 0)
        expect = a[None, good] + b[None, good] * phi(zz)
        num = np.where(blind, Yf[:, good] - expect, 0.0).sum(axis=1)
        den = blind.sum(axis=1)
        return np.where(den >= 200, num / np.maximum(den, 1), 0.0)

    def invert(freeze, haze, n_iter):
        h = h_model.copy()
        a, b, mu, sg, rmse, N = fit_pixels(h)
        good = (N >= 8) & (b > 0.15)
        noise = float(np.nanmedian(rmse[good])) or 0.05
        lam = float(Cf[:, good].sum(axis=1).mean()) * noise ** 2 / PRIOR_SIGMA ** 2
        for _ in range(n_iter):
            if haze:
                off = haze_offset(h, a, b, mu, sg, good).astype(np.float32)
                Ac = (Yf - off[:, None]) * Cf
                SS = ((Yf - off[:, None]) * Ac).sum(axis=1)
            else:
                Ac, SS = A, SSy
            h_new = solve_levels(Ac[:, good], Cf[:, good].astype(np.float32),
                                 SS, a[good], b[good], mu[good], sg[good],
                                 cand, h_model, lam)
            h_new = np.where(usable, h_new, h_model)
            if np.max(np.abs(h_new - h)) < 0.002:
                h = h_new
                break
            h = h_new
            if not freeze:
                a, b, mu, sg, rmse, N = fit_pixels(h)
                good = (N >= 8) & (b > 0.15)
        # final elevation always from a fit on the recovered levels
        a2, b2, mu2, sg2, r2, N2 = fit_pixels(h)
        return np.where((N2 >= 8) & (b2 > 0.15), mu2, np.nan), h

    runs = {}
    a0, b0, mu0, sg0, r0, N0 = fit_pixels(h_model)
    runs["A - marea del modelo"] = (np.where((N0 >= 8) & (b0 > 0.15), mu0,
                                             np.nan), h_model)
    global PRIOR_SIGMA
    for label, (fr, hz, it, ps) in {
            "D prior 0.05": (True, True, 6, 0.05),
            "D prior 0.10": (True, True, 6, 0.10),
            "D prior 0.20": (True, True, 6, 0.20),
            "D prior 0.40": (True, True, 6, 0.40)}.items():
        PRIOR_SIGMA = ps
        t0 = time.time()
        runs[label] = invert(fr, hz, it)
        print(f"  {label:26s} {(time.time()-t0):.0f}s", flush=True)

    # ── held-out field survey ────────────────────────────────────────────
    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, G = idx[have], gnss[have]
    ok = np.all([np.isfinite(v[idx]) for v, _ in runs.values()], axis=0)
    idx, G = idx[ok], G[ok]
    n = len(idx)

    print(f"\nSOBRE LOS MISMOS {n} PIXELES DE CAMPO")
    print(f"{'variante':26s} {'RMSE':>7s} {'r':>7s} {'pendiente':>10s} "
          f"{'RMS h-modelo':>13s}")
    print("-" * 70)
    res = {}
    for k, (v, h) in runs.items():
        e = v[idx]
        dd = G - e
        dd = dd - np.median(dd)
        res[k] = dd
        dh = np.sqrt(np.mean((h - h_model) ** 2))
        print(f"{k:26s} {np.sqrt(np.mean(dd**2)):7.3f} "
              f"{np.corrcoef(G, e)[0,1]:7.3f} "
              f"{np.polyfit(G, e, 1)[0]:10.3f} {dh:13.3f}")

    rng = np.random.default_rng(11)
    bi = rng.integers(0, n, size=(20000, n))
    base_sq = res[list(res)[0]] ** 2
    print(f"\nfrente a la marea del modelo (negativo = mejora):")
    for k in list(res)[1:]:
        sq = res[k] ** 2
        obs = np.sqrt(sq.mean()) - np.sqrt(base_sq.mean())
        boot = np.sqrt(sq[bi].mean(1)) - np.sqrt(base_sq[bi].mean(1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {k:26s} {obs:+.4f} m  IC [{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICATIVO' if lo*hi > 0 else 'no significativo'}")

    json.dump({k: {"rmse": float(np.sqrt(np.mean(v ** 2)))}
               for k, v in res.items()},
              open(os.path.join(SC, "prior_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
