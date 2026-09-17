"""V1 — judge the tide-adjusted DEMs against the Rijkswaterstaat Vaklodingen.

For each Dutch site (escalda, wadden, ems) the notebook's products
(plain / MAREA applied / MAREA fitted / crossed / Granadeiro, from
products_<site>/comparison_<tag>/products.npz) are sampled against the
Vaklodingen truth: the latest survey of each 20 m cell in 2022-2025 (NAP
datum, EPSG:28992), taken from the sheets fetched by V0.

Metric: median-centred RMSE and OLS slope of product vs truth over the
intertidal pixels with both values (the products sit on the boundary's
datum, the truth on NAP: a constant); also per band of distance from the
mouth, and on a 5 x 5-pixel thinned subset so the numbers are not carried by
spatial autocorrelation.

Run:  python -m experiments.v1_vaklodingen_validate [site ...]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SITES = {"escalda": ("products_escalda/comparison_eot20_g2/products.npz",
                     "ndwi_cube_escalda_2023-2025_20m.nc"),
         "wadden": ("products_wadden/comparison_eot20_g1/products.npz",
                    "ndwi_cube_wadden_2023-2025_20m.nc"),
         "ems": ("products_ems/comparison_eot20_g1/products.npz",
                 "ndwi_cube_ems_2023-2025_20m.nc")}
TRUTH_DIR = "data_v4/truth/vaklodingen"
YEARS = (2019, 2025)      # the German Ems sheets (WSA Emden) stop at 2020
LABELS = {"z_plain": "plain (ocean clock)", "z_marea": "MAREA applied",
          "z_marea_fit": "MAREA fitted, no threshold",
          "z_cross": "crossed (Granadeiro lag, our inversion)",
          "z_gran": "Granadeiro"}


def truth_grid(site):
    """Latest surveyed value per RD cell in YEARS, merged over the sheets."""
    import pyproj
    import xarray as xr
    from experiments.v0_vaklodingen_fetch import sheets_for

    cells = {}
    for name, _, _ in sheets_for(site):
        path = os.path.join(TRUTH_DIR, name)
        if not os.path.exists(path):
            print(f"  missing sheet {name}")
            continue
        ds = xr.open_dataset(path)
        zvar = "z" if "z" in ds else [v for v in ds.data_vars if ds[v].ndim == 3][0]
        yrs = ds["time"].dt.year.values
        sel = np.flatnonzero((yrs >= YEARS[0]) & (yrs <= YEARS[1]))
        if not len(sel):
            print(f"  {name}: no survey in {YEARS}")
            ds.close()
            continue
        z = ds[zvar].isel(time=sel).values          # (t, y, x)
        x, y = ds["x"].values, ds["y"].values
        ds.close()
        # latest finite value per cell, and its year
        zt = np.full(z.shape[1:], np.nan)
        yt = np.full(z.shape[1:], -1)
        for k, i in enumerate(sel):
            m = np.isfinite(z[k])
            zt[m] = z[k][m]
            yt[m] = yrs[i]
        cells[name] = (x, y, zt, yt)
        print(f"  {name}: {int(np.isfinite(zt).sum()):,} surveyed cells, "
              f"years {sorted(set(yt[yt > 0].tolist()))}")
    return cells


def sample_truth(cells, xq, yq):
    """Nearest-cell lookup of the merged truth at RD points."""
    out = np.full(len(xq), np.nan)
    yr = np.full(len(xq), -1)
    for x, y, zt, yt in cells.values():
        dx = float(np.median(np.diff(x))); dy = float(np.median(np.diff(y)))
        ix = np.round((xq - x[0]) / dx).astype(int)
        iy = np.round((yq - y[0]) / dy).astype(int)
        ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        v = np.full(len(xq), np.nan); v[ok] = zt[iy[ok], ix[ok]]
        take = np.isfinite(v) & ~np.isfinite(out)
        out[take] = v[take]; yr[take] = yt[iy[take], ix[take]]
    return out, yr


def main(sites):
    import pyproj
    import xarray as xr
    from pyintertidal import tide_estimators as te

    results = {}
    for site in sites:
        prod_path, cube = SITES[site]
        if not os.path.exists(prod_path):
            print(f"[{site}] no products yet: {prod_path}")
            continue
        print(f"[{site}]")
        P = np.load(prod_path)
        keep, s_km = P["keep"], P["s_km"].astype(float)
        H, W = (int(v) for v in P["shape"])
        ds = xr.open_dataset(cube)
        crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref"))
        xs, ys = ds["x"].values, ds["y"].values
        ds.close()
        rows, cols = keep // W, keep % W
        tr = pyproj.Transformer.from_crs(crs_cube, "EPSG:28992", always_xy=True)
        xq, yq = tr.transform(xs[cols], ys[rows])
        cells = truth_grid(site)
        truth, tyear = sample_truth(cells, xq, yq)
        have = np.isfinite(truth)
        print(f"  truth at {int(have.sum()):,} of {len(keep):,} intertidal px "
              f"(survey years {sorted(set(tyear[have].tolist()))})")
        # a product can only place a pixel inside the tide range its scenes
        # sampled (the inversion grid); truth outside that range - the deep
        # channel a 20 m cell next to the flat edge falls into, or ground
        # above every high water - is not a test of the clock or the
        # estimator. Nearest-cell sampling against steep channels is exactly
        # what made the Westerschelde read 1.4 m RMSE at first pass.
        zp = P["z_plain"].astype(float)
        lo, hi = float(np.nanmin(zp)), float(np.nanmax(zp))
        rng_ok = (truth >= lo - 0.25) & (truth <= hi + 0.25)
        print(f"  tide range of the products [{lo:+.2f}, {hi:+.2f}] m: "
              f"{int((have & ~rng_ok).sum()):,} truth px outside it (excluded; "
              f"{int((have & (truth < lo - 0.25)).sum()):,} below = channel, "
              f"{int((have & (truth > hi + 0.25)).sum()):,} above)")
        truth = np.where(rng_ok, truth, np.nan)
        have = np.isfinite(truth)

        # 5x5 thinning: one pixel per 100 m block
        thin = ((rows // 5) * 7919 + (cols // 5))
        _, first = np.unique(thin, return_index=True)
        thin_mask = np.zeros(len(keep), bool); thin_mask[first] = True

        def score(p, t):
            m = np.isfinite(p) & np.isfinite(t)
            if m.sum() < 30:
                return None
            pc, tc = p[m] - np.median(p[m]), t[m] - np.median(t[m])
            return {"rmse_m": float(np.sqrt(np.mean((pc - tc) ** 2))),
                    "slope": float(np.polyfit(tc, pc, 1)[0]),
                    "bias_m": float(np.median(p[m] - t[m])), "n": int(m.sum())}

        _, centres, band_of = te.make_bands(s_km, 6)
        site_rows = {}
        print(f"  {'product':44s} {'RMSE':>6s} {'slope':>6s} {'n':>7s} | thinned {'RMSE':>6s} {'slope':>6s} {'n':>6s}")
        for key, label in LABELS.items():
            if key not in P.files:
                continue
            p = P[key].astype(float)
            r_all = score(p, truth)
            r_thin = score(np.where(thin_mask, p, np.nan), truth)
            r_band = [score(np.where(band_of == k, p, np.nan), truth) for k in range(len(centres))]
            site_rows[label] = {"all": r_all, "thinned": r_thin,
                                "by_band": [{"s_km": float(centres[k]), **(r_band[k] or {})}
                                            for k in range(len(centres))]}
            if r_all and r_thin:
                print(f"  {label:44s} {r_all['rmse_m']:6.3f} {r_all['slope']:6.3f} {r_all['n']:7d} | "
                      f"{r_thin['rmse_m']:6.3f} {r_thin['slope']:6.3f} {r_thin['n']:6d}")
        # common subset (every product finite)
        common = have.copy()
        for key in LABELS:
            if key in P.files:
                common &= np.isfinite(P[key])
        print(f"  common subset ({int(common.sum()):,} px):")
        for key, label in LABELS.items():
            if key in P.files:
                r = score(np.where(common, P[key].astype(float), np.nan), truth)
                site_rows[label]["common"] = r
                if r:
                    print(f"    {label:42s} {r['rmse_m']:6.3f} {r['slope']:6.3f}")
        results[site] = {"n_truth": int(have.sum()), "years": sorted(set(int(v) for v in tyear[have])),
                         "rows": site_rows}

    # ── figure: RMSE and slope by band, plain vs MAREA vs Granadeiro ────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(results), figsize=(6.2 * len(results), 7.5), dpi=130, squeeze=False)
    cols_ = {"plain (ocean clock)": "#999999", "MAREA applied": "#c1121f",
             "crossed (Granadeiro lag, our inversion)": "#e0a000", "Granadeiro": "#1a76b3"}
    for j, (site, r) in enumerate(results.items()):
        for label, col in cols_.items():
            if label not in r["rows"]:
                continue
            bands = r["rows"][label]["by_band"]
            xs_ = [b["s_km"] for b in bands]
            axs[0, j].plot(xs_, [b.get("rmse_m", np.nan) for b in bands], "o-", color=col, label=label)
            axs[1, j].plot(xs_, [b.get("slope", np.nan) for b in bands], "o-", color=col, label=label)
        axs[0, j].set_title(f"{site}: RMSE vs Vaklodingen by band", loc="left", fontsize=10)
        axs[1, j].set_title(f"{site}: slope vs Vaklodingen by band", loc="left", fontsize=10)
        axs[1, j].axhline(1, color="k", lw=0.8, ls=":")
        axs[1, j].set_xlabel("distance from the mouth s (km)")
        for ax in axs[:, j]:
            ax.grid(alpha=0.25)
        axs[0, j].set_ylabel("RMSE (m), median-centred"); axs[1, j].set_ylabel("slope")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("docs/figures/tide_comparison", exist_ok=True)
    fig.savefig("docs/figures/tide_comparison/vaklodingen_by_band.png", bbox_inches="tight")

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/v1_vaklodingen_validation.json", "w"), indent=1)
    print("-> results/v1_vaklodingen_validation.json")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SITES))
