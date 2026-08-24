"""Which pixels can this archive not measure? Ask a simulation, not a survey.

sesgo.py established, with no ground truth anywhere, that the estimator is
unbiased across the interior of the water range — bias under 3 cm from 0.35 m
to 3.3 m of tidal headroom — and collapses at both ends, reaching -2.2 m for
pixels above the highest water the archive ever saw and +0.9 m for pixels
below the lowest.

It also established that the collapse cannot be undone. Correcting a fitted
elevation by inverting the mean bias curve made things far worse — RMSE 0.958
against 0.133 — and the reason is not a bug but the definition of censoring:
a fitted 0.26 m may be an ordinary mid-flat pixel or a censored pixel truly at
2.45 m, and nothing in the data separates them. The information is gone.

What can be done is to say WHICH pixels are in that state, and this is where
the simulation earns its place: it knows each planted pixel's true elevation,
so it can be asked which OBSERVABLE quantities give the censored ones away.
Observable means available for a real pixel with no survey:

    b          amplitude of the fitted transition — a pixel that barely ever
               changes state has little of it
    sigma      fitted sub-pixel relief, which inflates when the transition is
               unresolved
    wet        fraction of observations in which the pixel looked wet; near 0
               or near 1 means the transition was never watched
    n_obs      how many usable observations it had
    z_hat      where the fit landed relative to the water range

The rule is fitted on simulated pixels only. The survey is then spent once,
to check that the pixels the rule flags really are the ones that disagree
with the ground — which is a prediction made before looking, not a fit.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy.special import erf
from scipy import stats

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
N_PER_LEVEL = 500
N_LEVELS = 44
BAD_M = 0.25          # a fit this far out is called censored
NDWI_WET = 0.0


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def observables(Y, C, a, b, z, sg, nobs, lo, hi):
    """Everything the rule may use. None of it needs ground truth."""
    wet = np.where(C > 0, Y > NDWI_WET, 0).sum(0) / np.maximum(
        (C > 0).sum(0), 1)
    return np.column_stack([
        b,
        sg,
        wet,
        np.log(np.maximum(nobs, 1)),
        (hi - z) / (hi - lo),          # headroom, in units of the range
        (z - lo) / (hi - lo),
    ])


NAMES = ["b (amplitud)", "sigma", "fraccion mojada", "log n_obs",
         "holgura arriba", "holgura abajo"]


def simulate(tide, a_d, b_d, sg_d, noise_d, obs_frac, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = tide.min(), tide.max()
    span = hi - lo
    z_true = np.repeat(np.linspace(lo - 0.15 * span, hi + 0.35 * span,
                                   N_LEVELS), N_PER_LEVEL)
    P = len(z_true)
    a = rng.choice(a_d, P); b = rng.choice(b_d, P)
    sg = rng.choice(sg_d, P); nz = rng.choice(noise_d, P)
    clean = a[None, :] + b[None, :] * phi(
        (tide[:, None] - z_true[None, :]) / np.maximum(sg[None, :], 1e-3))
    Y = clean + rng.normal(0.0, 1.0, clean.shape) * nz[None, :]
    C = (rng.random(clean.shape) < obs_frac).astype(np.float64)

    grid = np.linspace(lo, hi, MU_POINTS)
    zh = np.full(P, np.nan); bh = np.zeros(P); sh = np.zeros(P)
    ah = np.zeros(P); nn = np.zeros(P)
    for j in range(0, P, 4000):
        s = slice(j, j + 4000)
        aa, bb, mu, ss, _, N = _fit_block(Y[:, s], C[:, s], tide, grid,
                                          SG_GRID)
        zh[s], bh[s], sh[s], ah[s], nn[s] = mu, bb, ss, aa, N
    ok = (nn >= MIN_OBS) & (bh > 0.15) & np.isfinite(zh)
    X = observables(Y, C, ah, bh, zh, sh, nn, lo, hi)
    return z_true, zh, ok, X


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
    g = good & np.isfinite(noise) & (noise > 0)

    # ── learn the rule on simulated pixels ───────────────────────────────
    z_true, zh, ok, Xs = simulate(tide, a[g], b[g], sg[g], noise[g], obs_frac)
    bad = ok & (np.abs(zh - z_true) > BAD_M)
    print(f"simulacion: {int(ok.sum()):,} px resueltos, "
          f"{int(bad.sum()):,} mal por mas de {BAD_M} m "
          f"({100*bad.sum()/max(ok.sum(),1):.0f} %)\n")

    print("que observable delata a un pixel censurado")
    print(f"{'observable':22s} {'AUC':>7s}")
    print("-" * 32)
    aucs = []
    for i, nm in enumerate(NAMES):
        v = Xs[ok, i]
        yb = bad[ok]
        if yb.sum() < 20 or (~yb).sum() < 20:
            continue
        u = stats.mannwhitneyu(v[yb], v[~yb], alternative="two-sided")
        auc = u.statistic / (yb.sum() * (~yb).sum())
        aucs.append((abs(auc - 0.5), i, nm, auc))
        print(f"{nm:22s} {auc:7.3f}")
    print("\nAUC lejos de 0.5 = separa. 1.0 o 0.0 = separa perfectamente.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xs[ok], bad[ok])
    p_sim = clf.predict_proba(Xs[ok])[:, 1]
    u = stats.mannwhitneyu(p_sim[bad[ok]], p_sim[~bad[ok]])
    auc = u.statistic / (bad[ok].sum() * (~bad[ok]).sum())
    print(f"\nregla combinada, AUC en simulacion: {auc:.3f}")

    # ── apply it to the real pixels ──────────────────────────────────────
    Xr = observables(Yr, Cr, a, b, z, sg, nobs, lo, hi)
    p_real = np.full(len(z), np.nan)
    m = good & np.isfinite(Xr).all(1)
    p_real[m] = clf.predict_proba(Xr[m])[:, 1]
    thr = float(np.quantile(p_sim, 0.90))
    print(f"umbral (percentil 90 de la simulacion): {thr:.3f}")
    print(f"px reales marcados: {int(np.nansum(p_real > thr)):,} de "
          f"{int(m.sum()):,} ({100*np.nansum(p_real>thr)/max(m.sum(),1):.1f} %)")

    # ── the survey, spent once, on a prediction made before looking ──────
    pos = {int(k): i for i, k in enumerate(keep_idx)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
        tr = s.transform
    zf, pf = np.where(good, z, np.nan)[fi], p_real[fi]
    mm = np.isfinite(zf) & np.isfinite(y) & np.isfinite(pf) & np.isfinite(hsr)
    y, zf, hsr, pf, rr, cc = y[mm], zf[mm], hsr[mm], pf[mm], rr[mm], cc[mm]

    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rngh = np.random.default_rng(SEED)
    hold = np.isin(block, rngh.choice(
        blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))),
        replace=False))

    flag = pf > thr
    print(f"\nCONJUNTO RESERVADO — {int(hold.sum())} puntos, "
          f"{int((flag & hold).sum())} marcados")
    print("-" * 62)
    off = float(np.median(y[~hold] - zf[~hold]))
    err = np.abs(y - zf - off)
    print(f"{'grupo':28s} {'n':>4s} {'|error| med':>12s} {'RMSE':>7s}")
    for lab, sel in (("marcados (reservado)", flag & hold),
                     ("no marcados (reservado)", ~flag & hold)):
        if sel.sum() == 0:
            print(f"{lab:28s}    0")
            continue
        print(f"{lab:28s} {int(sel.sum()):4d} {np.median(err[sel]):12.3f} "
              f"{np.sqrt(np.mean(err[sel]**2)):7.3f}")

    out = {"auc_sim": float(auc), "thr": thr,
           "frac_flagged_real": float(np.nansum(p_real > thr) / max(m.sum(), 1))}
    if (flag & hold).sum() >= 3 and (~flag & hold).sum() >= 3:
        u2 = stats.mannwhitneyu(err[flag & hold], err[~flag & hold],
                                alternative="greater")
        print(f"\nMann-Whitney (los marcados son peores): p = {u2.pvalue:.3g}")
        out["mwu_p"] = float(u2.pvalue)

    for lab, sel in (("todos (reservado)", hold),
                     ("solo no marcados", hold & ~flag)):
        if sel.sum() < 10:
            continue
        sl = float(np.polyfit(y[sel], zf[sel], 1)[0])
        rm = float(np.sqrt(np.mean((y[sel] - zf[sel] - off) ** 2)))
        print(f"{lab:28s} pendiente {sl:6.3f}  RMSE {rm:6.3f}  "
              f"n {int(sel.sum())}")
        out[lab] = {"slope": sl, "rmse": rm, "n": int(sel.sum())}

    json.dump(out, open(os.path.join(SC, "bandera.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(SC, "bandera.npz"),
                        p_real=p_real, thr=thr, keep=keep_idx)


if __name__ == "__main__":
    main()
