"""HSR and DEA on the eo-tides ensemble tide, against the RTK survey.

This is the configuration the DEA paper describes. It is worth stating what
the ensemble turned out to be here: over these 1379 scenes it equals the
unweighted mean of its three members to 0.0000 m, because the rankings that
weight it exist only in Australian waters. So this run answers "how does HSR
behave on the paper's ensemble" and "does averaging tide models help" with
the same number — they are the same tide.

Reuses any product already on disk. Judged on the pixels every product
resolves, with a paired bootstrap, because a 15 mm gap over ~110 points is
not something to read off a table and believe.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import pyintertidal as pit
from pyintertidal.elevation import fit_hsr, fit_step, epochs
from pyintertidal import terrain

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CACHE = "ndwi_cube_villaviciosa_grande_10y.nc"
OUT = "products_villaviciosa"

TIDES = {"GOT4.10": "tides.json", "EOT20": "tides_eot20.json",
         "ENSEMBLE": "tides_ensemble.json"}


def main():
    aoi = pit.sites.get("villaviciosa")
    cube = pit.SentinelCube(aoi, ["2016-01-01", "2025-12-31"], water="ndwi",
                            cache_path=CACHE)
    transform, crs = cube.grid
    SH = cube.shape

    wf = rasterio.open(f"{OUT}/water_frequency.tif").read(1)
    with rasterio.open(f"{OUT}/reference_map.tif") as s:
        ref = s.read(1)
    poly = aoi.raster_mask(transform, crs, SH)
    inter = pit.intertidal_mask(wf, ref, 0.01, 0.99, aoi_mask=poly)

    tides = {k: json.load(open(os.path.join(SC, v))) for k, v in TIDES.items()}
    ep = epochs([d for d in cube.dates if d in tides["GOT4.10"]], 3)[-1]
    print(f"epoca {ep['label']}, {len(ep['dates'])} fechas\n", flush=True)

    products = {}
    for tname, tide in tides.items():
        tag = tname.replace(".", "").lower()
        for meth in ("HSR", "DEA"):
            key = f"{meth} / {tname}"
            path = f"{OUT}/{meth.lower()}_{tag}_{ep['label']}.tif"
            if os.path.exists(path):
                products[key] = rasterio.open(path).read(1)
                print(f"  {key:18s} cacheado", flush=True)
                continue
            t0 = time.time()
            if meth == "HSR":
                r = fit_hsr(cube, tide, dates=ep["dates"], clip_mask=inter,
                            epoch_label=ep["label"], superresolve=False,
                            tide_bins="auto")
                a = terrain.drop_small_regions(np.asarray(r.mu, "float32"),
                                               min_region_px=25)
            else:
                st = fit_step(cube, tide, dates=ep["dates"], clip_mask=inter,
                              epoch_label=ep["label"], verbose=False)
                a = np.asarray(st.mu, "float32")
            pit.write_geotiff(path, a, transform, crs, "float32", np.nan)
            products[key] = a
            print(f"  {key:18s} {int(np.isfinite(a).sum()):7,d} px "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ── the RTK survey ───────────────────────────────────────────────────
    fx = pd.read_csv(r"C:\Users\Jorge\Downloads\batimetriavilla.csv")
    fx = fx[fx["Solution status"] == "FIX"]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(fx["Longitude"].values, fx["Latitude"].values)
    col = ((x - transform.c) / transform.a).astype(int)
    row = ((y - transform.f) / transform.e).astype(int)
    uniq, inv = np.unique(row * SH[1] + col, return_inverse=True)
    gnss = (np.bincount(inv, weights=fx["Ellipsoidal height"].values)
            / np.bincount(inv))
    gy = np.bincount(inv, weights=y) / np.bincount(inv)
    rr, cc = uniq // SH[1], uniq % SH[1]

    def clean(a):
        return np.where(np.isfinite(a) & (a > -900), a, np.nan)

    com = np.all([np.isfinite(clean(a)[rr, cc]) for a in products.values()], 0)
    n = int(com.sum())
    print(f"\nSOBRE LOS MISMOS {n} PIXELES")
    print(f"{'producto':18s} {'RMSE':>7s} {'MAE':>7s} {'r':>7s} {'pend':>7s}")
    print("-" * 50)
    res, slopes = {}, {}
    for k, a in products.items():
        v = clean(a)[rr, cc][com]
        d = gnss[com] - v
        d = d - np.median(d)
        res[k] = d
        slopes[k] = float(np.polyfit(gnss[com], v, 1)[0])
        print(f"{k:18s} {np.sqrt(np.mean(d**2)):7.3f} "
              f"{np.mean(np.abs(d)):7.3f} "
              f"{np.corrcoef(gnss[com], v)[0,1]:7.3f} {slopes[k]:7.3f}")

    # ── paired bootstrap ─────────────────────────────────────────────────
    rng = np.random.default_rng(20260816)
    BOOT = 20000
    idx = rng.integers(0, n, size=(BOOT, n))
    print(f"\nBOOTSTRAP PAREADO ({BOOT:,} remuestreos)  negativo = gana el 1o")
    print(f"{'comparacion':40s} {'dif':>7s} {'IC 95%':>17s}  veredicto")
    print("-" * 82)
    keys = list(res)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_, b_ = res[keys[i]] ** 2, res[keys[j]] ** 2
            obs = np.sqrt(a_.mean()) - np.sqrt(b_.mean())
            boot = np.sqrt(a_[idx].mean(1)) - np.sqrt(b_[idx].mean(1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"{keys[i]:>19s} vs {keys[j]:<18s} {obs:+7.3f} "
                  f"[{lo:+6.3f},{hi:+6.3f}]  "
                  f"{'SIGNIFICATIVO' if lo * hi > 0 else 'no significativo'}")

    # ── does the compression worsen upstream? ────────────────────────────
    print("\nCOMPRESION A LO LARGO DE LA RIA (hipotesis de amplificacion)")
    dist = (gy[com].max() - gy[com]) / 1000.0
    order = np.argsort(dist)
    low, high = order[:n // 2], order[n // 2:]
    print(f"  mitad exterior {dist[low].min():.2f}-{dist[low].max():.2f} km   "
          f"mitad interior {dist[high].min():.2f}-{dist[high].max():.2f} km")
    print(f"\n{'producto':18s} {'exterior':>9s} {'interior':>9s} {'cambio':>8s}")
    print("-" * 48)
    for k in keys:
        v = clean(products[k])[rr, cc][com]
        s_lo = np.polyfit(gnss[com][low], v[low], 1)[0]
        s_hi = np.polyfit(gnss[com][high], v[high], 1)[0]
        print(f"{k:18s} {s_lo:9.3f} {s_hi:9.3f} {s_hi - s_lo:+8.3f}")
    print("\nsi la amplificacion estuarina fuese la causa, la pendiente")
    print("interior deberia ser CLARAMENTE menor en todos los productos")

    json.dump({"rmse": {k: float(np.sqrt(np.mean(v ** 2)))
                        for k, v in res.items()},
               "pendiente": slopes, "n": n},
              open(os.path.join(SC, "ensemble_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
