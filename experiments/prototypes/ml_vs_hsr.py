"""Would a learned model beat HSR? Measure it, with the leakage trap disarmed.

HSR imposes a shape — the wet fraction is the normal CDF of the sub-pixel
elevations — and inverts it. A learned model would impose nothing and work
out the mapping from the data. That is a fair question and it has a number.

The trap: all 193 GNSS pixels sit inside 0.75 km of one ria. Neighbouring
10 m pixels are nearly the same measurement, so with random k-fold every test
pixel has its neighbours in the training set and the model can memorise the
local surface instead of learning anything transferable. The score that comes
out looks excellent and means nothing. Both are reported here precisely so
the gap between them is visible.

Features are deliberately the SAME information HSR uses — NDWI summarised
against tide height — but with no sigmoid imposed, so this compares the
assumption, not the input.
"""
import os, sys, json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_TIDE_BINS = 8
BLOCK_M = 150.0          # spatial block side for the honest CV


def main():
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    Y, C, keep, dates = d["Y"], d["C"], d["keep"], d["dates"]
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(d["shape"])

    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pd.to_datetime([f"{s} 11:00:00" for s in dates])
    tide = model_tides(x=[lon_c], y=[lat_c], time=times, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    yrs = np.array([int(s[:4]) for s in dates])
    ep = (yrs >= 2023) & np.isfinite(tide)
    Y, C, tide = Y[ep], C[ep], tide[ep]

    # ── keep only the pixels the field survey measured ───────────────────
    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, target = idx[have], gnss[have]
    flat = field_flat[have]
    Yf, Cf = Y[:, idx], C[:, idx]
    print(f"{len(idx)} pixeles de campo con serie NDWI")

    # ── features: NDWI summarised against tide, no sigmoid assumed ───────
    edges = np.quantile(tide, np.linspace(0, 1, N_TIDE_BINS + 1))
    edges[0] -= 1e-6
    cols, names = [], []
    for i in range(N_TIDE_BINS):
        m = (tide > edges[i]) & (tide <= edges[i + 1])
        w = Cf[m]
        v = np.where(w, Yf[m], np.nan)
        with np.errstate(invalid="ignore"):
            cols.append(np.nanmean(v, axis=0))
        names.append(f"ndwi_bin{i}")
        cols.append(w.sum(axis=0).astype(float))
        names.append(f"n_bin{i}")
    wf = np.where(Cf, (Yf > 0), 0).sum(axis=0) / np.maximum(Cf.sum(axis=0), 1)
    cols += [wf, Cf.sum(axis=0).astype(float)]
    names += ["water_frequency", "n_obs"]

    X = np.column_stack(cols)
    X = np.where(np.isfinite(X), X, np.nan)
    keep_rows = np.isfinite(X).all(axis=1) & np.isfinite(target)
    X, y, flat = X[keep_rows], target[keep_rows], flat[keep_rows]
    print(f"{X.shape[0]} muestras x {X.shape[1]} variables "
          f"(sin sigmoide impuesto)")

    # ── HSR on the same pixels, for reference ────────────────────────────
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea = s.read(1)
    rr, cc = flat // SH[1], flat % SH[1]
    hsr_v, dea_v = hsr[rr, cc], dea[rr, cc]

    def score(pred, mask=None):
        m = np.isfinite(pred) & np.isfinite(y) if mask is None else mask
        r = y[m] - pred[m]
        r = r - np.median(r)
        return np.sqrt(np.mean(r ** 2)), int(m.sum())

    both = np.isfinite(hsr_v) & np.isfinite(dea_v)
    print(f"\n{'referencia (sin etiquetas)':32s} {'RMSE':>7s} {'n':>5s}")
    print("-" * 48)
    for lab, v in [("HSR", hsr_v), ("escalon (DEA)", dea_v)]:
        rm, n = score(v, both)
        print(f"{lab:32s} {rm:7.3f} {n:5d}")

    # ── spatial blocks, so neighbours cannot leak across the split ───────
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        tr = s.transform
    east = tr.c + (cc + 0.5) * tr.a
    north = tr.f + (rr + 0.5) * tr.e
    block = ((east - east.min()) // BLOCK_M).astype(int) * 1000 + \
            ((north - north.min()) // BLOCK_M).astype(int)
    n_blocks = len(np.unique(block))
    print(f"\n{n_blocks} bloques espaciales de {BLOCK_M:.0f} m")

    models = {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                              random_state=0, n_jobs=2),
    }

    def cv_rmse(model, splits):
        preds = np.full(len(y), np.nan)
        for tr_i, te_i in splits:
            m = model
            m.fit(X[tr_i], y[tr_i])
            preds[te_i] = m.predict(X[te_i])
        r = y - preds
        r = r - np.median(r)
        return float(np.sqrt(np.mean(r ** 2)))

    print(f"\n{'modelo aprendido':22s} {'CV aleatoria':>13s} "
          f"{'CV espacial':>12s}   (etiquetas necesarias)")
    print("-" * 62)
    out = {}
    for name, model in models.items():
        rand = cv_rmse(model, list(KFold(5, shuffle=True,
                                         random_state=0).split(X)))
        spat = cv_rmse(model, list(GroupKFold(n_splits=min(5, n_blocks))
                                   .split(X, y, groups=block)))
        out[name] = {"cv_random": rand, "cv_spatial": spat}
        print(f"{name:22s} {rand:13.3f} {spat:12.3f}")

    print("\nla diferencia entre las dos columnas ES la fuga: con particion")
    print("aleatoria el modelo tiene vecinos del mismo metro cuadrado en el")
    print("entrenamiento y solo tiene que recordarlos.")

    json.dump(out, open(os.path.join(SC, "ml_vs_hsr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
