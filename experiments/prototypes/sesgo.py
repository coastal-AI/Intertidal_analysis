"""A censoring-aware estimator, calibrated without a single ground point.

What the day established
------------------------
The relief compresses — fitted elevations span only ~0.82 of the surveyed
range — and this project had been reading that as evidence of a wrong water
level. Every version of that reading has now failed its own control:

  * spring-versus-neap disagreement growing upstream: 3 of 12 noise-matched
    re-deals equal or beat it
  * a free water-surface tilt per date: 23 % of dates pinned at the edge of
    the search grid, 44 % at exactly zero
  * a tilt driven by the tidal state: physically sized at last, and it clears
    its synthetic null only by about 15 % with two nulls, but it predicts a
    compression of 0.985 against the 0.82 observed — one part in seventy
  * the real water level, tide plus inverse-barometer surge: no better than
    the SAME surge dealt to the wrong dates

What survives is duller and much stronger. The underestimate grows by 0.189 m
for every metre of tidal headroom the pixel lacks, t = -5.30, and it is
independent of how often the pixel was observed — the two correlate at
r = 0.011, so they separate cleanly, and selecting on observation count alone
does nothing. Restricted to pixels with more than 0.8 m of water above them,
the slope is 0.963 with a confidence interval spanning 1: no compression left.

So the elevations are censored. A pixel near the top of the archive's water
range is submerged rarely or never, nothing constrains the upper tail of its
transition, and the fit settles low. Every published method has this, which is
exactly why HSR and the step method compress by the same amount.

The estimator
-------------
Censoring is a property of the ARCHIVE, not of the ground, so its size can be
computed by simulation before anyone visits the site:

  1. take the real tide series and the real distribution of fitted (a, b,
     sigma) and per-pixel noise — all of it derived from the imagery alone
  2. plant pixels at KNOWN elevations across and beyond the water range
  3. fit them with the same estimator and record how far the fit falls, as a
     function of how much water sat above them
  4. invert that curve to correct real fits

No survey enters steps 1 to 4. The survey is spent once, at the end, on
blocks held out before anything else happened, and the datum offset comes
from the training half only — in production there is no survey at the site to
take it from.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy.special import erf

from pyintertidal.elevation import _fit_block

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817
N_PER_LEVEL = 400
N_LEVELS = 44


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def simulate_bias(tide, a_d, b_d, sg_d, noise_d, obs_frac, seed=0):
    """How far does the fit fall, as a function of tidal headroom?

    Pixels are planted at known elevations reaching ABOVE the highest water in
    the archive, because that is the regime the correction has to cover and
    the regime no real pixel can teach us about.
    """
    rng = np.random.default_rng(seed)
    lo, hi = tide.min(), tide.max()
    span = hi - lo
    z_true = np.repeat(np.linspace(lo - 0.15 * span, hi + 0.35 * span,
                                   N_LEVELS), N_PER_LEVEL)
    P = len(z_true)
    a = rng.choice(a_d, P)
    b = rng.choice(b_d, P)
    sg = rng.choice(sg_d, P)
    nz = rng.choice(noise_d, P)

    clean = a[None, :] + b[None, :] * phi(
        (tide[:, None] - z_true[None, :]) / np.maximum(sg[None, :], 1e-3))
    Y = clean + rng.normal(0.0, 1.0, clean.shape) * nz[None, :]
    C = (rng.random(clean.shape) < obs_frac).astype(np.float64)

    grid = np.linspace(lo, hi, MU_POINTS)
    z_hat = np.full(P, np.nan)
    keep = np.zeros(P, bool)
    for j in range(0, P, 4000):
        s = slice(j, j + 4000)
        aa, bb, mu, ss, _, N = _fit_block(Y[:, s], C[:, s], tide, grid,
                                          SG_GRID)
        z_hat[s] = mu
        keep[s] = (N >= MIN_OBS) & (bb > 0.15)
    return z_true, z_hat, keep


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    keep_idx, field_flat, gnss, dates = (d["keep"], d["field_flat"],
                                         d["gnss"], d["dates"])
    SH = tuple(int(v) for v in d["shape"])
    Y, C = d["Y"], d["C"]
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]
    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide_all)
    tide = tide_all[ep]
    lo, hi = float(tide.min()), float(tide.max())

    B = np.load(os.path.join(SC, "lamina_base.npz"))
    a, b, z, sg, nobs, good = (B["a"], B["b"], B["z"], B["sg"], B["nobs"],
                               B["good"])

    # noise per pixel, straight from the real residuals
    Yr = np.nan_to_num(Y[ep], nan=0.0).astype(np.float64)
    Cr = C[ep].astype(np.float64)
    noise = np.zeros(Yr.shape[1])
    for j in range(0, Yr.shape[1], 4000):
        s = slice(j, j + 4000)
        pr = a[None, s] + b[None, s] * phi(
            (tide[:, None] - z[None, s]) / np.maximum(sg[None, s], 1e-3))
        w = Cr[:, s] > 0
        noise[s] = np.sqrt(np.where(w, (Yr[:, s] - pr) ** 2, 0.0).sum(0)
                           / np.maximum(w.sum(0), 1))
    obs_frac = float((Cr > 0).mean())
    del Yr, Cr

    g = good & np.isfinite(noise) & (noise > 0)
    print(f"archivo: {len(tide)} escenas, agua de {lo:+.2f} a {hi:+.2f} m")
    print(f"plantilla: {int(g.sum()):,} px reales dan (a, b, sigma, ruido); "
          f"fraccion observada {obs_frac:.2f}\n")

    z_true, z_hat, ok = simulate_bias(tide, a[g], b[g], sg[g], noise[g],
                                      obs_frac)
    ok &= np.isfinite(z_hat)
    head_true = hi - z_true
    print(f"simulados {len(z_true):,} px, resueltos {int(ok.sum()):,}")

    # ── the curve ────────────────────────────────────────────────────────
    edges = np.linspace(head_true.min(), head_true.max(), 26)
    cent, bias, n_in = [], [], []
    for i in range(len(edges) - 1):
        m = ok & (head_true >= edges[i]) & (head_true < edges[i + 1])
        if m.sum() < 60:
            continue
        cent.append(0.5 * (edges[i] + edges[i + 1]))
        bias.append(float(np.median(z_hat[m] - z_true[m])))
        n_in.append(int(m.sum()))
    cent, bias = np.array(cent), np.array(bias)
    print(f"\n{'holgura (m)':>12s} {'sesgo (m)':>11s} {'n':>7s}")
    print("-" * 34)
    for c, v, n in zip(cent, bias, n_in):
        print(f"{c:12.2f} {v:+11.3f} {n:7d}")
    print("\nsesgo negativo = el ajuste cae por debajo de la cota real.")
    print("Nada de esto ha visto el GNSS.")

    # ── invert it: observed z_hat -> corrected z ─────────────────────────
    zt = hi - cent                       # true elevation at each bin centre
    zh = zt + bias                       # what the fit reports there
    order = np.argsort(zh)
    zh_s, zt_s = zh[order], zt[order]

    def correct(v):
        return np.interp(v, zh_s, zt_s, left=zt_s[0], right=zt_s[-1])

    # ── spend the held-out survey, once ─────────────────────────────────
    pos = {int(k): i for i, k in enumerate(keep_idx)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
        tr = s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea = s.read(1)[rr, cc]
    zf = np.where(good, z, np.nan)[fi]
    m = np.isfinite(zf) & np.isfinite(y) & np.isfinite(hsr) & np.isfinite(dea)
    y, zf, hsr, dea, rr, cc = y[m], zf[m], hsr[m], dea[m], rr[m], cc[m]

    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    hold = np.isin(block, rng.choice(
        blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))),
        replace=False))
    print(f"\nCONJUNTO RESERVADO — {len(blocks)} bloques de {BLOCK_M:.0f} m, "
          f"entrenamiento {int((~hold).sum())} / reservado {int(hold.sum())}")
    print("-" * 62)
    print(f"{'metodo':32s} {'RMSE':>7s} {'pend':>7s} {'etiquetas':>10s}")

    out = {}
    for name, v, lab in (("HSR (producto)", hsr, 0),
                         ("escalon DEA (producto)", dea, 0),
                         ("base gamma=0 (este codigo)", zf, 0),
                         ("corregido por censura", correct(zf), 0)):
        off = float(np.median(y[~hold] - v[~hold]))   # datum from train only
        r = y[hold] - (v[hold] + off)
        sl = float(np.polyfit(y[hold], v[hold], 1)[0])
        rms = float(np.sqrt(np.mean(r ** 2)))
        print(f"{name:32s} {rms:7.3f} {sl:7.3f} {lab:10d}")
        out[name] = {"rmse": rms, "slope": sl}

    json.dump({"curve": {"headroom": cent.tolist(), "bias": bias.tolist()},
               "holdout": out, "tide_max": hi,
               "n_holdout": int(hold.sum())},
              open(os.path.join(SC, "sesgo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
