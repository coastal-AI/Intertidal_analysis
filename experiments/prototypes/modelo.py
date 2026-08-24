"""A physics-informed residual learner for intertidal elevation.

Design, and why it is this and not something else
-------------------------------------------------
The literature review found that every ML regressor for intertidal elevation
which reports an honest transfer degrades by two to three times, while the
label-free physical fit does not. The reason is always the same: the models
learn WHERE they are rather than HOW A PIXEL RESPONDS TO THE TIDE, so they
interpolate beautifully inside the surveyed area and collapse outside it.
This project has already produced one instance of that failure — a model fed
neighbourhood water-frequency and distance-to-water features scored well and
was void, because with adjacent 150 m blocks those features let it copy the
answer from a neighbour.

So this model is built to make that failure impossible rather than to be
caught later:

  * NO spatial features. Not coordinates, not distance to water, not
    neighbourhood statistics. Nothing that answers "where in the ria am I".
    Every feature describes how ONE pixel's radiometry responded to the tide,
    which is a question whose answer transfers.

  * It predicts a RESIDUAL, not an elevation. The physical fit supplies the
    estimate; the model only learns what the physics gets systematically
    wrong, from the physics' own diagnostics. A model that learns nothing
    returns the physical estimate unchanged, which is a safe floor. A model
    that learns elevation from scratch has no such floor.

Two versions, and the comparison between them is the point
----------------------------------------------------------
  A  SUPERVISED — trained on RTK points from the training blocks. The
     conventional approach, and the ceiling of what labels can buy here.

  B  SIMULATION-TRAINED — trained on pixels planted at known elevations in a
     simulation calibrated from the archive alone: real tide series, real
     distribution of fitted (a, b, sigma), real per-pixel noise. It sees ZERO
     ground truth and can therefore be carried to a coast nobody has surveyed.

A prediction worth writing down before running: B will find little to correct.
The same simulation showed this estimator is unbiased to within 3 cm across
the interior of the water range, which is where every surveyed pixel sits. If
A gains and B does not, the gap is not something the model failed to learn —
it is physics missing from the simulation, and that localises what is still
unexplained about the compression.

Validation
----------
The split is the one fixed earlier and not revisited: spatial blocks of 150 m,
35 % held out, seed 20260817. Model and hyperparameters are chosen by nested
cross-validation INSIDE the training blocks. The datum offset comes from the
training half only, for every method including the baselines, because in
production there is no survey at the site to take it from. The held-out blocks
are scored once.
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
MU_POINTS = 60
SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
MIN_OBS = 8
BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817
N_TIDE_BINS = 8
SIM_PER_LEVEL = 500
SIM_LEVELS = 44

FEATURES = None       # filled in by build_features


def phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def build_features(Y, C, tide, a, b, z, sg, nobs):
    """Per-pixel description of how radiometry answered the tide.

    Two families, and nothing else:
      fit diagnostics  what the physical estimator itself reports, including
                       how well it fitted and where it landed inside the
                       water range
      response shape   distribution-free summaries of NDWI against tide,
                       which is the raw evidence the fit compressed
    """
    global FEATURES
    lo, hi = float(tide.min()), float(tide.max())
    rng_ = max(hi - lo, 1e-6)
    feats, names = [], []

    def add(v, n):
        feats.append(np.asarray(v, dtype=float))
        names.append(n)

    obs = C > 0
    V = np.where(obs, Y, np.nan)
    n = np.maximum(obs.sum(0), 1)

    # residual of the physical fit: how well the sigmoid actually described
    # this pixel is the single most informative diagnostic it can offer
    pred = a[None, :] + b[None, :] * phi(
        (tide[:, None] - z[None, :]) / np.maximum(sg[None, :], 1e-3))
    resid = np.where(obs, Y - pred, 0.0)
    add(np.sqrt((resid ** 2).sum(0) / n), "rmse_ajuste")
    add(np.abs(resid).max(0), "residuo_max")

    add(a, "a")
    add(b, "b")
    add(sg, "sigma")
    add(np.log(np.maximum(nobs, 1)), "log_nobs")
    add((hi - z) / rng_, "holgura_arriba")
    add((z - lo) / rng_, "holgura_abajo")

    with np.errstate(invalid="ignore"):
        for q in (5, 25, 50, 75, 95):
            add(np.nanpercentile(V, q, axis=0), f"p{q}")
        add(np.nanstd(V, 0), "sd_ndwi")

    edges = np.quantile(tide, np.linspace(0, 1, N_TIDE_BINS + 1))
    edges[0] -= 1e-9
    for i in range(N_TIDE_BINS):
        m = (tide > edges[i]) & (tide <= edges[i + 1])
        if m.sum() == 0:
            add(np.zeros(Y.shape[1]), f"bin{i}")
            add(np.zeros(Y.shape[1]), f"mojado{i}")
            continue
        with np.errstate(invalid="ignore"):
            add(np.nan_to_num(np.nanmean(V[m], 0)), f"bin{i}")
        add(np.where(obs[m], Y[m] > 0, 0).sum(0)
            / np.maximum(obs[m].sum(0), 1), f"mojado{i}")

    t = tide[:, None]
    w = obs.astype(float)
    mt = (w * t).sum(0) / n
    my = np.where(obs, Y, 0).sum(0) / n
    cov = np.where(obs, (t - mt) * (Y - my), 0).sum(0) / n
    vt = (w * (t - mt) ** 2).sum(0) / n
    vy = np.where(obs, (Y - my) ** 2, 0).sum(0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        add(cov / np.sqrt(np.maximum(vt * vy, 1e-12)), "corr_marea")
        add(cov / np.maximum(vt, 1e-12), "pend_marea")
    add(np.where(obs, Y > 0, 0).sum(0) / n, "frac_mojada")

    FEATURES = names
    return np.column_stack(feats)


def candidates():
    return {
        "Ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=400,
                                              min_samples_leaf=4,
                                              random_state=0, n_jobs=2),
        "GradBoosting": GradientBoostingRegressor(random_state=0),
    }


def pick_inside(X, y, groups, label):
    """Choose the model by cross-validation that never sees the held-out set."""
    k = min(4, len(np.unique(groups)))
    if k < 2:
        return list(candidates().items())[0]
    inner = GroupKFold(n_splits=k)
    best, best_s, bname = None, np.inf, None
    for nm, m in candidates().items():
        p = np.full(len(y), np.nan)
        for a_, b_ in inner.split(X, y, groups=groups):
            m.fit(X[a_], y[a_])
            p[b_] = m.predict(X[b_])
        s = float(np.sqrt(np.mean((y - p) ** 2)))
        print(f"    {nm:14s} CV interna {s:.3f}")
        if s < best_s:
            best_s, best, bname = s, m, nm
    print(f"    elegido para {label}: {bname}")
    return bname, best


def simulate_training_set(tide, a_d, b_d, sg_d, noise_d, obs_frac, seed=0,
                          sd_scene=0.0):
    """Pixels at known elevations, so the residual to learn is known exactly.

    Two things this gets right that a casual version would not:

    * the four pixel properties are drawn as a BLOCK, by resampling whole real
      pixels. Drawing a, b, sigma and noise independently would manufacture
      combinations that do not occur — a bright amplitude with a tiny sigma,
      say — and the model would spend its capacity on a population that does
      not exist.
    * a scene-wide term is added on top of the per-pixel noise. Haze, sun
      angle and turbidity move an entire scene together, and a simulation of
      independent pixel noise is a quieter world than the archive really is.
    """
    rng = np.random.default_rng(seed)
    lo, hi = tide.min(), tide.max()
    span = hi - lo
    z_true = np.repeat(np.linspace(lo - 0.10 * span, hi + 0.25 * span,
                                   SIM_LEVELS), SIM_PER_LEVEL)
    P = len(z_true)
    pick = rng.integers(0, len(a_d), P)
    a, b, sg, nz = a_d[pick], b_d[pick], sg_d[pick], noise_d[pick]
    clean = a[None, :] + b[None, :] * phi(
        (tide[:, None] - z_true[None, :]) / np.maximum(sg[None, :], 1e-3))
    Y = (clean
         + rng.normal(0.0, 1.0, clean.shape) * nz[None, :]
         + rng.normal(0.0, sd_scene, (clean.shape[0], 1)))
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
    X = build_features(Y, C, tide, ah, bh, zh, sh, nn)
    return X[ok], (z_true - zh)[ok], zh[ok]


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    tide_all = np.load(os.path.join(SC, "mareas_villa.npz"))["tide"]
    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide_all)
    tide = tide_all[ep]
    Yr = np.nan_to_num(Y[ep], nan=0.0).astype(np.float64)
    Cr = C[ep].astype(np.float64)

    B = np.load(os.path.join(SC, "lamina_base.npz"))
    a, b, z, sg, nobs, good = (B["a"], B["b"], B["z"], B["sg"], B["nobs"],
                               B["good"])

    # ── the surveyed pixels, and only them, for the supervised model ─────
    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat])
    have = fi >= 0
    fi, y, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]

    Xf = build_features(Yr[:, fi], Cr[:, fi], tide, a[fi], b[fi], z[fi],
                        sg[fi], nobs[fi])
    zf = np.where(good, z, np.nan)[fi]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
        tr = s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea = s.read(1)[rr, cc]

    m = (np.isfinite(Xf).all(1) & np.isfinite(y) & np.isfinite(zf)
         & np.isfinite(hsr) & np.isfinite(dea))
    Xf, y, zf, hsr, dea, rr, cc = (Xf[m], y[m], zf[m], hsr[m], dea[m],
                                   rr[m], cc[m])
    print(f"{len(y)} puntos de campo · {Xf.shape[1]} variables, "
          f"ninguna espacial")
    print(f"variables: {', '.join(FEATURES[:8])}, ...\n")

    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rngh = np.random.default_rng(SEED)
    hold = np.isin(block, rngh.choice(
        blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))),
        replace=False))
    print(f"{len(blocks)} bloques de {BLOCK_M:.0f} m · entrenamiento "
          f"{int((~hold).sum())} puntos / RESERVADO {int(hold.sum())}\n")

    # the target is the CORRECTION to the physical estimate, with the datum
    # taken from the training half only
    off = float(np.median(y[~hold] - zf[~hold]))
    target = (y - off) - zf

    print("A — modelo supervisado (aprende de los puntos de entrenamiento)")
    nameA, mdlA = pick_inside(Xf[~hold], target[~hold], block[~hold],
                              "supervisado")
    mdlA.fit(Xf[~hold], target[~hold])
    predA = zf + mdlA.predict(Xf)

    # ── B: trained on simulation only ────────────────────────────────────
    print("\nB — modelo entrenado SOLO en simulacion (cero etiquetas)")
    noise = np.zeros(Yr.shape[1])
    for j in range(0, Yr.shape[1], 4000):
        s = slice(j, j + 4000)
        pr = a[None, s] + b[None, s] * phi(
            (tide[:, None] - z[None, s]) / np.maximum(sg[None, s], 1e-3))
        w = Cr[:, s] > 0
        noise[s] = np.sqrt(np.where(w, (Yr[:, s] - pr) ** 2, 0.0).sum(0)
                           / np.maximum(w.sum(0), 1))
    g = good & np.isfinite(noise) & (noise > 0)
    obs_frac = float((Cr > 0).mean())
    # size of the scene-wide systematic, taken from the archive itself
    sd_scene = 0.0
    try:
        med = np.zeros(Yr.shape[0])
        sub = np.where(g)[0][::7]
        pr = a[None, sub] + b[None, sub] * phi(
            (tide[:, None] - z[None, sub]) / np.maximum(sg[None, sub], 1e-3))
        rr_ = np.where(Cr[:, sub] > 0, Yr[:, sub] - pr, np.nan)
        med = np.nanmedian(rr_, axis=1)
        sd_scene = float(np.nanstd(med))
        del pr, rr_
    except Exception:
        pass
    print(f"    sistematico de escena medido en el archivo: "
          f"sd {sd_scene:.4f} NDWI")
    Xs, ys, _ = simulate_training_set(tide, a[g], b[g], sg[g], noise[g],
                                      obs_frac, sd_scene=sd_scene)
    print(f"    {len(ys):,} px simulados · correccion verdadera "
          f"mediana {np.median(ys):+.3f} m, sd {np.std(ys):.3f}")
    # B chooses its model on a held-out slice of the SIMULATION, so no real
    # data of any kind touches the choice. With 17 000 planted pixels there
    # is no reason to settle for a linear fit without checking.
    rs = np.random.default_rng(9)
    cut = rs.random(len(ys)) < 0.7
    bestB, bestBs, nameB = None, np.inf, None
    for nm, mdl in candidates().items():
        mdl.fit(Xs[cut], ys[cut])
        s = float(np.sqrt(np.mean((ys[~cut] - mdl.predict(Xs[~cut])) ** 2)))
        print(f"    {nm:14s} en simulacion reservada {s:.3f}")
        if s < bestBs:
            bestBs, bestB, nameB = s, mdl, nm
    print(f"    elegido para B: {nameB}")
    mdlB = bestB
    mdlB.fit(Xs, ys)
    corr = mdlB.predict(Xf)
    predB = zf + corr
    print(f"    correccion que propone en los px reales: "
          f"mediana {np.median(corr):+.3f} m, sd {np.std(corr):.3f}, "
          f"|max| {np.abs(corr).max():.3f}")

    # ── the held-out blocks, once ───────────────────────────────────────
    print(f"\n{'CONJUNTO RESERVADO':34s} {'RMSE':>7s} {'pend':>7s} "
          f"{'etiquetas':>10s}")
    print("-" * 62)
    out = {}
    rows = [("HSR (producto)", hsr, 0),
            ("escalon DEA (producto)", dea, 0),
            ("fisica sola (este ajuste)", zf, 0),
            (f"A · supervisado ({nameA})", predA, int((~hold).sum())),
            ("B · solo simulacion", predB, 0)]
    for name, v, lab in rows:
        o = float(np.median(y[~hold] - v[~hold]))
        r = y[hold] - (v[hold] + o)
        rms = float(np.sqrt(np.mean(r ** 2)))
        sl = float(np.polyfit(y[hold], v[hold], 1)[0])
        print(f"{name:34s} {rms:7.3f} {sl:7.3f} {lab:10d}")
        out[name] = {"rmse": rms, "slope": sl, "labels": lab}

    # ── paired bootstrap on the same resamples, so the comparisons match ──
    rb = np.random.default_rng(3)
    yh = y[hold]

    def centred(v):
        return v[hold] + float(np.median(y[~hold] - v[~hold]))

    base, aa, bb2, prod = (centred(zf), centred(predA), centred(predB),
                           centred(hsr))
    idx = rb.integers(0, len(yh), (4000, len(yh)))

    def gain(alt, ref):
        dv = np.array([np.sqrt(np.mean((yh[i] - ref[i]) ** 2))
                       - np.sqrt(np.mean((yh[i] - alt[i]) ** 2))
                       for i in idx])
        return float(dv.mean()), np.percentile(dv, [2.5, 97.5])

    print(f"\n{'comparacion':40s} {'mejora':>8s} {'IC 95%':>18s}")
    print("-" * 70)
    for lab, alt, ref in (
            ("A supervisado sobre la fisica", aa, base),
            ("B solo simulacion sobre la fisica", bb2, base),
            ("B solo simulacion sobre HSR producto", bb2, prod),
            ("A supervisado sobre B", aa, bb2)):
        mn, ci = gain(alt, ref)
        star = "" if ci[0] <= 0 <= ci[1] else "  <-- no cruza cero"
        print(f"{lab:40s} {mn:+8.3f} [{ci[0]:+7.3f},{ci[1]:+7.3f}]{star}")
        out[lab] = {"mean": mn, "ci": ci.tolist()}
    print("\nUna mejora cuyo intervalo cruza cero no es una mejora: con 48")
    print("puntos reservados el poder es corto y hay que decirlo.")

    if hasattr(mdlA, "feature_importances_") or nameA == "RandomForest":
        try:
            imp = mdlA.feature_importances_
            o = np.argsort(imp)[::-1][:8]
            print("\nvariables que mas usa A:")
            for i in o:
                print(f"   {FEATURES[i]:18s} {imp[i]:.3f}")
        except Exception:
            pass

    json.dump({"features": FEATURES, "holdout": out,
               "n_train": int((~hold).sum()), "n_hold": int(hold.sum())},
              open(os.path.join(SC, "modelo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
