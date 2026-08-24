"""A validation set set aside first, touched once, at the end.

Cross-validation reuses every point for both fitting and judging. That is
efficient but it leaves no number that was produced without the data having
influenced the choices. So: split the surveyed blocks in two BEFORE anything
else happens, do all model and feature selection inside the training half,
and evaluate on the other half exactly once.

The split is by spatial BLOCK, not by point. Neighbouring 10 m pixels are
nearly the same measurement, so splitting at random would put a pixel's own
neighbours in the training set and the held-out score would flatter the
model — the same leak that made the first cross-validation optimistic.

HSR is scored on the same held-out blocks. Its datum offset comes from the
training half only, since in production there is no survey at the site to
take it from.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_BINS = 8
BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    aoi = pit.sites.get("villaviciosa"); lat_c, lon_c = aoi.centroid
    times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide)
    Y, C, tide = Y[ep], C[ep], tide[ep]

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat]); have = fi >= 0
    fi, target, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]
    Yf, Cf = Y[:, fi], C[:, fi]

    # features: NDWI against tide only, no spatial context
    feats = []
    V = np.where(Cf, Yf, np.nan)
    for q in (5, 10, 25, 50, 75, 90, 95):
        feats.append(np.nanpercentile(V, q, axis=0))
    feats.append(np.nanstd(V, 0))
    edges = np.quantile(tide, np.linspace(0, 1, N_BINS + 1)); edges[0] -= 1e-6
    for i in range(N_BINS):
        m = (tide > edges[i]) & (tide <= edges[i + 1])
        feats.append(np.nanmean(V[m], 0))
        feats.append(Cf[m].sum(0).astype(float))
    t = tide[:, None]; w = Cf.astype(float); n = np.maximum(w.sum(0), 1)
    mt = (w * t).sum(0) / n; my = np.nansum(np.where(Cf, Yf, 0), 0) / n
    cov = np.where(Cf, (t - mt) * (Yf - my), 0).sum(0) / n
    vt = (w * (t - mt) ** 2).sum(0) / n
    vy = np.where(Cf, (Yf - my) ** 2, 0).sum(0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        feats.append(cov / np.sqrt(np.maximum(vt * vy, 1e-12)))
        feats.append(cov / np.maximum(vt, 1e-12))
    lo, hi = np.quantile(tide, [1 / 3, 2 / 3])
    for m in (tide <= lo, tide >= hi):
        feats.append(np.where(Cf[m], Yf[m] > 0, 0).sum(0)
                     / np.maximum(Cf[m].sum(0), 1))
    feats.append(np.where(Cf, Yf > 0, 0).sum(0) / n)
    feats.append(Cf.sum(0).astype(float))
    X = np.column_stack(feats)

    ok = np.isfinite(X).all(1) & np.isfinite(target)
    X, y, rr, cc = X[ok], target[ok], rr[ok], cc[ok]

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr_v, tr = s.read(1)[rr, cc], s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea_v = s.read(1)[rr, cc]

    east = tr.c + (cc + .5) * tr.a; north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))

    # ── the split, made first and never revisited ────────────────────────
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    n_hold = max(2, int(round(HOLDOUT_FRACTION * len(blocks))))
    hold_blocks = rng.choice(blocks, n_hold, replace=False)
    is_hold = np.isin(block, hold_blocks)
    print(f"{len(y)} puntos en {len(blocks)} bloques de {BLOCK_M:.0f} m")
    print(f"  entrenamiento {int((~is_hold).sum()):3d} puntos "
          f"({len(blocks) - n_hold} bloques)")
    print(f"  RESERVADO     {int(is_hold.sum()):3d} puntos ({n_hold} bloques)")

    Xtr, ytr, btr = X[~is_hold], y[~is_hold], block[~is_hold]
    Xho, yho = X[is_hold], y[is_hold]

    # ── everything below sees ONLY the training half ─────────────────────
    cands = {
        "Ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=400,
                                              min_samples_leaf=3,
                                              random_state=0, n_jobs=2),
        "GradBoosting": GradientBoostingRegressor(random_state=0),
    }
    inner = GroupKFold(n_splits=min(4, len(np.unique(btr))))
    print(f"\nseleccion dentro del entrenamiento:")
    best, best_s, bname = None, np.inf, None
    for nm, m in cands.items():
        p = np.full(len(ytr), np.nan)
        for a, b in inner.split(Xtr, ytr, groups=btr):
            m.fit(Xtr[a], ytr[a]); p[b] = m.predict(Xtr[b])
        s = float(np.sqrt(np.mean((ytr - p) ** 2)))
        print(f"  {nm:14s} CV interna {s:.3f}")
        if s < best_s:
            best_s, best, bname = s, m, nm
    print(f"  elegido: {bname}")
    best.fit(Xtr, ytr)

    # HSR's datum offset also comes from the training half only
    off = float(np.nanmedian(ytr - hsr_v[~is_hold]))
    off_dea = float(np.nanmedian(ytr - dea_v[~is_hold]))

    # ── the held-out set, used once ──────────────────────────────────────
    def rmse(pred, truth):
        m = np.isfinite(pred) & np.isfinite(truth)
        return float(np.sqrt(np.mean((truth[m] - pred[m]) ** 2))), int(m.sum())

    rows = [(f"ML ({bname})", *rmse(best.predict(Xho), yho), len(ytr)),
            ("HSR", *rmse(hsr_v[is_hold] + off, yho), 0),
            ("escalon (DEA)", *rmse(dea_v[is_hold] + off_dea, yho), 0)]

    print(f"\n{'CONJUNTO RESERVADO':22s} {'RMSE':>7s} {'n':>5s} "
          f"{'etiquetas usadas':>17s}")
    print("-" * 56)
    for name, rm, n_, lab in rows:
        print(f"{name:22s} {rm:7.3f} {n_:5d} {lab:17d}")

    # slope on the held-out set: does the model also fix the compression?
    print(f"\n{'pendiente contra el GNSS reservado':40s}")
    for name, pred in [(f"ML ({bname})", best.predict(Xho)),
                       ("HSR", hsr_v[is_hold] + off),
                       ("escalon (DEA)", dea_v[is_hold] + off_dea)]:
        m = np.isfinite(pred) & np.isfinite(yho)
        print(f"  {name:22s} {np.polyfit(yho[m], pred[m], 1)[0]:6.3f}")

    json.dump({"rows": [{"method": r[0], "rmse": r[1], "n": r[2]} for r in rows],
               "n_train": int((~is_hold).sum()),
               "n_holdout": int(is_hold.sum()), "chosen": bname},
              open(os.path.join(SC, "holdout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
