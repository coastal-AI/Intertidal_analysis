"""Give the learned model every per-pixel feature we can think of.

The first attempt used tide-binned NDWI only. This adds everything else that
varies from pixel to pixel and does not require HSR's sigmoid:

  * distribution of NDWI: quantiles, extremes, spread
  * how NDWI tracks the tide: Pearson r, and the slope of a straight line
  * behaviour at the ends of the tidal frame: wet fraction in the lowest and
    highest terciles, which is where elevation information concentrates
  * spatial context: local water frequency at two scales, and distance to
    permanent water — a pixel deep inside the flat is not like one on the edge

Harmonic constituents are deliberately NOT included. They are a property of
the tide prediction POINT, and all 199 pixels share one, so the column would
be constant: zero variance, zero information. They could only separate sites,
not pixels, and that needs labels at more than one site.

A third variant hands the model HSR's own mu as an extra feature. That asks
the more interesting question: not "can a model replace the physics" but
"can it improve on it".
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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
N_BINS = 8
BLOCK_M = 150.0


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

    pos = {int(k): i for i, k in enumerate(keep)}
    idx = np.array([pos.get(int(f), -1) for f in field_flat])
    have = idx >= 0
    idx, target, flat = idx[have], gnss[have], field_flat[have]
    Yf, Cf = Y[:, idx], C[:, idx]
    rr, cc = flat // SH[1], flat % SH[1]

    feats, names = [], []

    def add(v, name):
        feats.append(np.asarray(v, float))
        names.append(name)

    V = np.where(Cf, Yf, np.nan)

    # distribution of NDWI
    for q in (5, 10, 25, 50, 75, 90, 95):
        add(np.nanpercentile(V, q, axis=0), f"ndwi_p{q}")
    add(np.nanstd(V, axis=0), "ndwi_std")
    add(np.nanmax(V, axis=0) - np.nanmin(V, axis=0), "ndwi_range")

    # tide-binned means and counts
    edges = np.quantile(tide, np.linspace(0, 1, N_BINS + 1)); edges[0] -= 1e-6
    for i in range(N_BINS):
        m = (tide > edges[i]) & (tide <= edges[i + 1])
        add(np.nanmean(V[m], axis=0), f"ndwi_bin{i}")
        add(Cf[m].sum(axis=0), f"n_bin{i}")

    # how NDWI tracks the tide
    t = tide[:, None]
    w = Cf.astype(float)
    n = np.maximum(w.sum(0), 1)
    mt = (w * t).sum(0) / n
    my = np.nansum(np.where(Cf, Yf, 0), 0) / n
    cov = (np.where(Cf, (t - mt) * (Yf - my), 0)).sum(0) / n
    vt = (w * (t - mt) ** 2).sum(0) / n
    vy = (np.where(Cf, (Yf - my) ** 2, 0)).sum(0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        add(cov / np.sqrt(np.maximum(vt * vy, 1e-12)), "corr_ndwi_tide")
        add(cov / np.maximum(vt, 1e-12), "slope_ndwi_tide")

    # the ends of the tidal frame, where elevation information concentrates
    lo, hi = np.quantile(tide, [1 / 3, 2 / 3])
    for lab, m in [("low", tide <= lo), ("high", tide >= hi)]:
        wet = np.where(Cf[m], Yf[m] > 0, 0).sum(0)
        add(wet / np.maximum(Cf[m].sum(0), 1), f"wetfrac_{lab}")

    add(np.where(Cf, Yf > 0, 0).sum(0) / n, "water_frequency")
    add(Cf.sum(0), "n_obs")

    # spatial context
    with rasterio.open("products_villaviciosa/water_frequency.tif") as s:
        wf_map, tr = s.read(1), s.transform
    with rasterio.open("products_villaviciosa/reference_map.tif") as s:
        ref = s.read(1)
    filled = np.where(np.isfinite(wf_map), wf_map, 0)
    for size in (3, 9, 21):
        sm = ndimage.uniform_filter(filled, size=size)
        add(sm[rr, cc], f"wf_local{size}")
    dist = ndimage.distance_transform_edt(ref != 1) * abs(tr.a)
    add(dist[rr, cc], "dist_to_water_m")

    X = np.column_stack(feats)
    ok = np.isfinite(X).all(1) & np.isfinite(target)
    X, y, rr2, cc2 = X[ok], target[ok], rr[ok], cc[ok]
    print(f"{X.shape[0]} muestras x {X.shape[1]} variables")

    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea = s.read(1)
    hsr_v, dea_v = hsr[rr2, cc2], dea[rr2, cc2]

    def rmse(pred, m):
        r = y[m] - pred[m]
        return float(np.sqrt(np.mean((r - np.median(r)) ** 2)))

    both = np.isfinite(hsr_v) & np.isfinite(dea_v)
    print(f"\n{'sin etiquetas':30s} {'RMSE':>7s} {'n':>5s}")
    print("-" * 46)
    for lab, v in [("HSR", hsr_v), ("escalon (DEA)", dea_v)]:
        print(f"{lab:30s} {rmse(v, both):7.3f} {int(both.sum()):5d}")

    east = tr.c + (cc2 + 0.5) * tr.a
    north = tr.f + (rr2 + 0.5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    nb = len(np.unique(block))

    def cv(model, X_, splits):
        p = np.full(len(y), np.nan)
        for a, b in splits:
            model.fit(X_[a], y[a]); p[b] = model.predict(X_[b])
        r = y - p
        return float(np.sqrt(np.mean((r - np.median(r)) ** 2)))

    models = {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                              random_state=0, n_jobs=2),
        "GradBoosting": GradientBoostingRegressor(random_state=0),
    }
    Xh = np.column_stack([X, np.where(np.isfinite(hsr_v), hsr_v,
                                      np.nanmedian(hsr_v))])

    print(f"\n{nb} bloques espaciales de {BLOCK_M:.0f} m")
    print(f"\n{'modelo':16s} {'variables':>10s} {'CV aleat.':>10s} "
          f"{'CV espacial':>12s}")
    print("-" * 54)
    out = {}
    for name, m in models.items():
        for tag, Xi in [("ricas", X), ("ricas+HSR", Xh)]:
            rnd = cv(m, Xi, list(KFold(5, shuffle=True, random_state=0).split(Xi)))
            spa = cv(m, Xi, list(GroupKFold(n_splits=min(5, nb))
                                 .split(Xi, y, groups=block)))
            out[f"{name}|{tag}"] = {"cv_random": rnd, "cv_spatial": spa}
            print(f"{name:16s} {tag:>10s} {rnd:10.3f} {spa:12.3f}")

    json.dump(out, open(os.path.join(SC, "ml_ricas.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
