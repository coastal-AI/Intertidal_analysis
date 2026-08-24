"""Re-fit both estimators at Tagus and Vadehavet, faithfully this time.

The numbers in products_*/validation.json predate today's correction to
`fit_step`, which was missing DEA's interpolation to 200 intervals, its
trailing rolling mean of radius 20, `min_count` and `min_correlation`. At
Villaviciosa that correction moved the baseline from 0.401 m to 0.135 m — so
every published HSR-versus-step comparison at these two sites was measured
against a weakened opponent and means nothing.

HSR is re-fitted too, not just the baseline: the `sigma_mu` masking bug fixed
today changed which pixels survive quality control, so the stored HSR rasters
are also stale. Refitting both under identical current code is the only way
the comparison is between methods rather than between vintages.

Scored against the EMODnet survey already on disk, on the pixels BOTH methods
resolve, with a paired bootstrap — a difference in RMSE over hundreds of
thousands of correlated pixels is not something to read off a table and
believe. Coverage is reported alongside, because HSR resolving roughly twice
the pixels is a result in itself that a bare RMSE hides.
"""
import os, sys, json, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio

import pyintertidal as pit
from pyintertidal.elevation import fit_hsr, fit_step, epochs
from pyintertidal.validation import reproject_to_grid
from pyintertidal import terrain

SITES = {
    "tejo": dict(cube="ndwi_cube_tejo_2019_2021.nc",
                 ref="bathy_tagus_native.tif",
                 extent=["2019-01-01", "2021-12-31"]),
    "vadehavet": dict(cube="ndwi_cube_vadehavet_2019_2021.nc",
                      ref="bathy_native.tif",
                      extent=["2019-01-01", "2021-12-31"]),
}
BOOT = 5000


def score(truth, pred, sel):
    d = truth[sel] - pred[sel]
    bias = float(np.median(d))
    d = d - bias
    return dict(n=int(sel.sum()), bias_m=round(bias, 3),
                rmse_m=round(float(np.sqrt(np.mean(d ** 2))), 3),
                mae_m=round(float(np.mean(np.abs(d))), 3),
                pearson_r=round(float(np.corrcoef(truth[sel], pred[sel])[0, 1]), 3))


def main():
    for name, cfg in SITES.items():
        out = f"products_{name}"
        print(f"\n{'='*64}\n{name.upper()}\n{'='*64}", flush=True)
        aoi = pit.sites.get(name)
        cube = pit.SentinelCube(aoi, cfg["extent"], water="ndwi",
                                cache_path=cfg["cube"])
        transform, crs = cube.grid
        SH = cube.shape
        with rasterio.open(f"{out}/intertidal_mask.tif") as s:
            inter = s.read(1).astype(bool)
        tides = json.load(open(f"{out}/tides.json"))
        ep = epochs([d for d in cube.dates if d in tides], 3)[-1]
        print(f"epoca {ep['label']}, {len(ep['dates'])} fechas, "
              f"{SH[0]}x{SH[1]} px, {int(inter.sum()):,} intermareales",
              flush=True)

        products = {}
        for meth in ("HSR", "step (DEA)"):
            tag = "hsr" if meth == "HSR" else "step"
            path = f"{out}/{tag}_elevation_fiel.tif"
            if os.path.exists(path):
                products[meth] = rasterio.open(path).read(1)
                print(f"  {meth:12s} cacheado", flush=True)
                continue
            t0 = time.time()
            if meth == "HSR":
                r = fit_hsr(cube, tides, dates=ep["dates"], clip_mask=inter,
                            epoch_label=ep["label"], superresolve=False,
                            tide_bins="auto")
                a = terrain.drop_small_regions(np.asarray(r.mu, "float32"),
                                               min_region_px=25)
            else:
                r = fit_step(cube, tides, dates=ep["dates"], clip_mask=inter,
                             epoch_label=ep["label"], verbose=True)
                a = np.asarray(r.mu, "float32")
            pit.write_geotiff(path, a, transform, crs, "float32", np.nan)
            products[meth] = a
            print(f"  {meth:12s} {int(np.isfinite(a).sum()):8,d} px "
                  f"({(time.time()-t0)/60:.1f} min) -> {path}", flush=True)

        truth = reproject_to_grid(f"{out}/{cfg['ref']}", transform, crs, SH)
        truth = np.where(np.isfinite(truth) & (truth > -900), truth, np.nan)
        print(f"  referencia EMODnet: {int(np.isfinite(truth).sum()):,} px "
              f"validos en el grid", flush=True)

        def clean(a):
            return np.where(np.isfinite(a) & (a > -900), a, np.nan)

        vals = {k: clean(v) for k, v in products.items()}
        res = {"all": [], "common": [], "epoch": ep["label"],
               "n_dates": len(ep["dates"])}
        for k, v in vals.items():
            sel = np.isfinite(truth) & np.isfinite(v) & inter
            res["all"].append(dict(method=k, **score(truth, v, sel)))

        common = np.isfinite(truth) & inter
        for v in vals.values():
            common &= np.isfinite(v)
        res["n_common"] = int(common.sum())
        for k, v in vals.items():
            res["common"].append(dict(method=k, **score(truth, v, common)))

        print(f"\n  SOBRE LOS MISMOS {int(common.sum()):,} PIXELES")
        print(f"  {'metodo':12s} {'n propio':>10s} {'RMSE':>7s} {'r':>7s}")
        print("  " + "-" * 40)
        for row_all, row_com in zip(res["all"], res["common"]):
            print(f"  {row_com['method']:12s} {row_all['n']:10,d} "
                  f"{row_com['rmse_m']:7.3f} {row_com['pearson_r']:7.3f}")

        # paired bootstrap over the common pixels
        idx = np.flatnonzero(common)
        rng = np.random.default_rng(20260816)
        keep = rng.choice(idx, min(len(idx), 40000), replace=False)
        sq = {}
        for k, v in vals.items():
            d = truth[np.unravel_index(keep, SH)] - v[np.unravel_index(keep, SH)]
            sq[k] = (d - np.median(d)) ** 2
        ks = list(sq)
        bi = rng.integers(0, len(keep), size=(BOOT, len(keep)))
        diff = np.sqrt(sq[ks[0]][bi].mean(1)) - np.sqrt(sq[ks[1]][bi].mean(1))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        obs = float(np.sqrt(sq[ks[0]].mean()) - np.sqrt(sq[ks[1]].mean()))
        res["bootstrap"] = {"comparison": f"{ks[0]} menos {ks[1]}",
                            "delta_rmse_m": round(obs, 4),
                            "ci95": [round(float(lo), 4), round(float(hi), 4)],
                            "significant": bool(lo * hi > 0),
                            "n_resampled": int(len(keep))}
        print(f"\n  {ks[0]} menos {ks[1]}: {obs:+.4f} m  "
              f"IC 95% [{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICATIVO' if lo*hi > 0 else 'no significativo'}",
              flush=True)

        old = f"{out}/validation.json"
        if os.path.exists(old) and not os.path.exists(f"{out}/validation_prefix.json"):
            os.rename(old, f"{out}/validation_prefix.json")
            print(f"  anterior -> validation_prefix.json")
        json.dump(res, open(old, "w"), indent=1)
        print(f"  escrito {old}", flush=True)


if __name__ == "__main__":
    main()
