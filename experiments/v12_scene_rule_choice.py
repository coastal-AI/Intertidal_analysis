# -*- coding: utf-8 -*-
"""Which scene record should MAREA use? A truth-free choice (2026-10-06).

Two records per site, both with the band lag model and the per-pixel
quality rule (QUALITY_MIN_BRACKET), the per-pixel SCL cloud mask always on:
  * "transition": the transition-zone scene rule (the record of the last run,
    read from its result.json);
  * "all": every scene of 2023-2025 with an overpass time and a finite level.

No truth enters. For each record:
  * coverage   = share of the extraction's pixels that get an elevation;
  * noise      = split-half noise of the elevation: the record split into two
                 halves (alternate dates), each inverted with the full record's
                 per-pixel lags; RMS(z_a - z_b) / sqrt(2) on the pixels both
                 halves AND both records resolve (the same pixels for the two
                 records of a site);
  * lag_judge  = lag at the held-out gauge (sites with one; a diagnostic).

Decision, fixed before the run: the record with the lower split-half noise at
the majority of the sites is adopted for every site; coverage is reported.

usage (server, from the repository root):
  python -m experiments.v12_scene_rule_choice [site ...]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pyintertidal as pit  # noqa: E402
from pyintertidal import marea  # noqa: E402
from experiments.validation_grid import DUTCH_RES_M, dutch_cube, dutch_products  # noqa: E402

P = ("2023-01-01", "2025-12-31")
OUT = "results/scene_rule_choice"


def site_cfg(site):
    if site == "villaviciosa":
        lag = sorted(glob.glob("products_marea/marea_eot20_g0_*/result.json"))[-1]
        return dict(extract="products_marea/marea_extract.npz", over="products_marea/overpass_times.json",
                    cube="ndwi_cube_villaviciosa_grande_10y.nc", pixel_m=10.0, lag_result=lag,
                    judge=None, n_g=None, mouth="any")
    if site == "ferrol":
        return dict(extract="products_ferrol/marea_extract.npz", over="products_ferrol/overpass_times.json",
                    cube="ndwi_cube_ferrol_2023-2025_10m.nc", pixel_m=10.0,
                    lag_result="products_ferrol/marea_eot20_g1/result.json", judge="fer2", n_g=1,
                    mouth=None)
    tag = {"escalda": "eot20_g2", "wadden": "eot20_g1", "ems": "eot20_g1"}[site]
    p = dutch_products(site)
    return dict(extract=f"{p}/marea_extract.npz", over=f"{p}/overpass_times.json", cube=dutch_cube(site),
                pixel_m=float(DUTCH_RES_M), lag_result=f"{p}/marea_{tag}/result.json",
                judge={"escalda": "trnz", "wadden": "harl", "ems": "delf"}[site],
                n_g={"escalda": 2, "wadden": 1, "ems": 1}[site], mouth=None)


def notebook_mouth_side(site):
    nb = json.load(open(f"tide_boundary_comparison_{site}.ipynb", encoding="utf-8"))
    for line in "".join(nb["cells"][2]["source"]).splitlines():
        if line.startswith("MOUTH_SIDE"):
            return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    return "any"


def boundary_for(site, cfg):
    if site == "villaviciosa":
        from pyintertidal.boundary import PyTMDBoundary
        return PyTMDBoundary("EOT20", *pit.sites.get("villaviciosa").centroid, directory="tide_models")
    from pyintertidal.boundary import make_boundary
    return make_boundary(pit.sites.get(site), tide_model="EOT20", n_gauges=cfg["n_g"], period=P,
                         exclude=[cfg["judge"]], verbose=False)


def judge_pixels(site, cfg, keep, W):
    if not cfg["judge"]:
        return None
    import pyproj
    import xarray as xr
    st = {s["Code"]: s for s in json.load(open("data_v4/gauges/ioc_stations.json", encoding="utf-8"))[:]}
    g = st[cfg["judge"]]
    with xr.open_dataset(cfg["cube"]) as ds:
        xs, ys = ds["x"].values, ds["y"].values
        crs = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt") or ds["crs"].attrs.get("spatial_ref"))
    gx, gy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(float(g["Lon"]), float(g["Lat"]))
    d = np.hypot(xs[keep % W] - gx, ys[keep // W] - gy)
    return np.argsort(d)[:200]


def run_site(site):
    cfg = site_cfg(site)
    if cfg["mouth"] is None:
        cfg["mouth"] = notebook_mouth_side(site)
    pit.net.use_system_certificates()
    z_ = np.load(cfg["extract"], allow_pickle=True)
    ex = {k: z_[k] for k in z_.files}
    del z_
    ex["shape"] = tuple(int(v) for v in ex["shape"])
    ex["bbox"] = ex["bbox"].item() if "bbox" in ex else {**pit.sites.get(site).bbox, "crs": "EPSG:4326"}
    H, W = ex["shape"]
    keep = ex["keep"]
    dates = np.array([str(d) for d in ex["dates"]])
    over = json.load(open(cfg["over"]))
    bnd = boundary_for(site, cfg)
    rec_rule = set(json.load(open(cfg["lag_result"]))["record_dates"])
    near = judge_pixels(site, cfg, keep, W)
    results, z_by = [], {}
    for name in ("transition", "all"):
        sel = np.array([(P[0] <= d <= P[1]) and d in over and (name == "all" or d in rec_rule) for d in dates])
        t = pd.DatetimeIndex(pd.to_datetime([str(over[d]) for d in dates[sel]])).tz_localize(None)
        h = np.asarray(bnd.levels(t), float)
        idx = np.flatnonzero(sel)[np.isfinite(h)]
        ex_r = {**ex, "Y": ex["Y"][idx], "C": ex["C"][idx], "dates": ex["dates"][idx]}
        out = f"{OUT}/{site}_{name}"
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        res = marea.reconstruct(cfg["cube"], out_dir=out, name=f"{site} {name}", pixel_m=cfg["pixel_m"],
                                boundary=bnd, extraction=ex_r, mouth_side=cfg["mouth"], min_scenes=40,
                                overpass_times=over, n_boot=0, verbose=True)
        mp = np.load(f"{out}/marea.npz")
        tau_px = mp["tau_px_min"].astype(float)
        # split-half: alternate dates, each half inverted with the full record's lags
        t_r = pd.DatetimeIndex(pd.to_datetime([str(over[d]) for d in dates[idx]])).tz_localize(None)
        Y = np.nan_to_num(ex["Y"][idx], nan=0.0).astype(np.float32)
        C = ex["C"][idx].astype(np.float32)
        lo, hi = res["rango_marea_m"]
        halves = (np.arange(len(idx)) % 2 == 0, np.arange(len(idx)) % 2 == 1)
        zh = [np.full(len(keep), np.nan), np.full(len(keep), np.nan)]
        for tv in np.unique(tau_px):
            cols = np.flatnonzero(tau_px == tv)
            hk = np.asarray(bnd.levels(t_r - pd.Timedelta(minutes=float(tv))), float)
            for j, hm in enumerate(halves):
                zh[j][cols], _ = marea.invert_series(Y[hm][:, cols], C[hm][:, cols], hk[hm], lo, hi,
                                                     min_bracket=marea.QUALITY_MIN_BRACKET)
        z_by[name] = (mp["z"].astype(float), zh)
        results.append({"site": site, "record": name, "scenes": res["n_escenas"],
                        "coverage": res["n_px_cota"] / res["n_px"],
                        "lag_judge_min": float(np.median(tau_px[near])) if near is not None else np.nan,
                        "seconds": round(time.time() - t0)})
        del Y, C, ex_r
        print(results[-1], flush=True)
    both = np.ones(len(keep), bool)
    for name, (z, zh) in z_by.items():
        both &= np.isfinite(z) & np.isfinite(zh[0]) & np.isfinite(zh[1])
    for r in results:
        z, zh = z_by[r["record"]]
        r["noise_m"] = float(np.sqrt(np.mean((zh[0][both] - zh[1][both]) ** 2)) / np.sqrt(2))
        r["noise_px"] = int(both.sum())
    return results


def main(sites):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in sites:
        rows += run_site(s)
        pd.DataFrame(rows).to_csv(f"{OUT}/scene_rule_choice.csv", index=False)
    df = pd.DataFrame(rows)
    piv = df.pivot(index="site", columns="record", values="noise_m")
    wins_all = int((piv["all"] < piv["transition"]).sum())
    choice = "all" if wins_all > len(piv) / 2 else "transition"
    print(df.round(4).to_string(index=False))
    print(f"\nlower split-half noise with ALL scenes at {wins_all} of {len(piv)} sites -> adopted record: {choice}")
    json.dump({"choice": choice, "wins_all": wins_all, "n_sites": len(piv),
               "rule": "lower split-half noise at the majority of sites (fixed before the run)"},
              open(f"{OUT}/decision.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["villaviciosa", "ferrol", "escalda", "wadden", "ems"])
