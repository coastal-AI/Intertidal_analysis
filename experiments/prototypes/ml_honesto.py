"""Re-run the comparison without the two advantages I gave the models.

Two things were wrong with the first version, and both flattered the learned
models:

  1. the residual median was subtracted using ALL samples, test included.
     For HSR that is a genuine datum correction — its output sits in the tide
     model's frame and the survey in an ellipsoidal one, a real 52.81 m
     constant. The learned models already predict in the right frame, so
     giving them the same subtraction hands them a free parameter fitted on
     the test data. Here they get the offset from their TRAINING fold only.

  2. model and feature set were chosen by looking at the same 199 points the
     score is reported on. That is selection bias, so this version picks the
     model INSIDE each training fold — nested selection — and reports what
     that procedure achieves, which is what you would actually get on new
     ground.

Both fixes push in the same direction, so if the conclusion survives them it
was not an artefact of the scoring.
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


def build(Yf, Cf, tide, rr, cc, tr, wf_map, ref):
    feats, names = [], []
    def add(v, n):
        feats.append(np.asarray(v, float)); names.append(n)
    V = np.where(Cf, Yf, np.nan)
    for q in (5, 10, 25, 50, 75, 90, 95):
        add(np.nanpercentile(V, q, axis=0), f"p{q}")
    add(np.nanstd(V, 0), "std")
    edges = np.quantile(tide, np.linspace(0, 1, N_BINS + 1)); edges[0] -= 1e-6
    for i in range(N_BINS):
        m = (tide > edges[i]) & (tide <= edges[i + 1])
        add(np.nanmean(V[m], 0), f"bin{i}")
        add(Cf[m].sum(0), f"n{i}")
    t = tide[:, None]; w = Cf.astype(float); n = np.maximum(w.sum(0), 1)
    mt = (w * t).sum(0) / n; my = np.nansum(np.where(Cf, Yf, 0), 0) / n
    cov = np.where(Cf, (t - mt) * (Yf - my), 0).sum(0) / n
    vt = (w * (t - mt) ** 2).sum(0) / n
    vy = np.where(Cf, (Yf - my) ** 2, 0).sum(0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        add(cov / np.sqrt(np.maximum(vt * vy, 1e-12)), "corr")
        add(cov / np.maximum(vt, 1e-12), "slope")
    lo, hi = np.quantile(tide, [1 / 3, 2 / 3])
    for lab, m in [("lo", tide <= lo), ("hi", tide >= hi)]:
        add(np.where(Cf[m], Yf[m] > 0, 0).sum(0) / np.maximum(Cf[m].sum(0), 1),
            f"wet_{lab}")
    add(np.where(Cf, Yf > 0, 0).sum(0) / n, "wf")
    add(Cf.sum(0), "nobs")
    filled = np.where(np.isfinite(wf_map), wf_map, 0)
    for size in (3, 9, 21):
        add(ndimage.uniform_filter(filled, size=size)[rr, cc], f"wf{size}")
    add(ndimage.distance_transform_edt(ref != 1)[rr, cc] * abs(tr.a), "dist")
    return np.column_stack(feats), names


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
    yrs = np.array([int(s[:4]) for s in dates]); ep = (yrs >= 2023) & np.isfinite(tide)
    Y, C, tide = Y[ep], C[ep], tide[ep]

    pos = {int(k): i for i, k in enumerate(keep)}
    fi = np.array([pos.get(int(f), -1) for f in field_flat]); have = fi >= 0
    fi, target, flat = fi[have], gnss[have], field_flat[have]
    rr, cc = flat // SH[1], flat % SH[1]

    with rasterio.open("products_villaviciosa/water_frequency.tif") as s:
        wf_map, tr = s.read(1), s.transform
    with rasterio.open("products_villaviciosa/reference_map.tif") as s:
        ref = s.read(1)
    X, names = build(Y[:, fi], C[:, fi], tide, rr, cc, tr, wf_map, ref)

    # Spatial-context features answer "where in the ria am I", not "how does
    # this pixel respond to the tide". With adjacent 150 m blocks they let a
    # model interpolate the surface from neighbouring blocks instead of
    # inferring elevation from radiometry — a different question, and the one
    # that cannot transfer to a coast where nobody has surveyed anything.
    import os as _os
    if _os.environ.get("DROP_SPATIAL"):
        drop = [i for i, n in enumerate(names)
                if n.startswith("wf") and n != "wf" or n == "dist"]
        X = np.delete(X, drop, axis=1)
        names = [n for i, n in enumerate(names) if i not in drop]
        print(f"variables espaciales retiradas: {len(drop)}")
    ok = np.isfinite(X).all(1) & np.isfinite(target)
    X, y, rr, cc = X[ok], target[ok], rr[ok], cc[ok]

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)
    hsr_v = hsr[rr, cc]

    east = tr.c + (cc + .5) * tr.a; north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    gkf = GroupKFold(n_splits=5)
    print(f"{len(y)} muestras · {X.shape[1]} variables · "
          f"{len(np.unique(block))} bloques de {BLOCK_M:.0f} m\n")

    def candidates():
        return {
            "Ridge": make_pipeline(StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-3, 3, 25))),
            "RandomForest": RandomForestRegressor(n_estimators=400,
                                                  min_samples_leaf=3,
                                                  random_state=0, n_jobs=2),
            "GradBoosting": GradientBoostingRegressor(random_state=0),
        }

    # ── nested: the model is chosen inside the training fold ─────────────
    pred = np.full(len(y), np.nan)
    chosen = []
    for tr_i, te_i in gkf.split(X, y, groups=block):
        inner = GroupKFold(n_splits=4)
        best, best_s = None, np.inf
        for nm, m in candidates().items():
            p = np.full(len(tr_i), np.nan)
            for a, b in inner.split(X[tr_i], y[tr_i], groups=block[tr_i]):
                m.fit(X[tr_i][a], y[tr_i][a]); p[b] = m.predict(X[tr_i][b])
            s = np.sqrt(np.mean((y[tr_i] - p) ** 2))
            if s < best_s:
                best_s, best, bname = s, m, nm
        chosen.append(bname)
        best.fit(X[tr_i], y[tr_i])
        pred[te_i] = best.predict(X[te_i])
    print("modelo elegido en cada fold:", ", ".join(chosen))

    def rmse_no_centre(p):
        m = np.isfinite(p)
        return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))

    # HSR keeps its datum offset, but taken from the training folds only
    hsr_corr = np.full(len(y), np.nan)
    for tr_i, te_i in gkf.split(X, y, groups=block):
        off = np.nanmedian(y[tr_i] - hsr_v[tr_i])
        hsr_corr[te_i] = hsr_v[te_i] + off

    print(f"\n{'metodo':34s} {'RMSE':>7s} {'etiquetas':>10s}")
    print("-" * 54)
    print(f"{'HSR (datum del fold de train)':34s} "
          f"{rmse_no_centre(hsr_corr):7.3f} {'0':>10s}")
    print(f"{'ML, seleccion anidada, CV espacial':34s} "
          f"{rmse_no_centre(pred):7.3f} {len(y):>10d}")

    json.dump({"hsr": rmse_no_centre(hsr_corr),
               "ml_nested": rmse_no_centre(pred),
               "chosen": chosen},
              open(os.path.join(SC, ("ml_sin_espacial.json" if _os.environ.get("DROP_SPATIAL") else "ml_honesto.json")), "w"), indent=1)


if __name__ == "__main__":
    main()
