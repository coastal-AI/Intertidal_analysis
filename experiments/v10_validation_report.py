# -*- coding: utf-8 -*-
"""The tables of the validation report (docs/paper/validation_summary.tex).

Reads the results of one validation run (by default the 10 m server run,
extracted under ``--root``) and writes LaTeX table rows to
``docs/paper/tables/`` plus ``numbers.json`` with every figure the text
quotes. One row per method, each with its own published scene rule:
no delay, MAREA, NIDEM (ITEM rule) and Granadeiro et al. (2021).

Tables
* rtk:      Villaviciosa RTK-GNSS (development subset), Alicante datum,
            from ``results/method_comparison/villaviciosa.csv`` (v9).
* lidar:    Villaviciosa against the IGN PNOA LiDAR DTM (MDT02, 2 m),
            aggregated to the 10 m grid from ground pixels only (the water
            the LiDAR saw is removed at 2 m first), all and common cells.
* lag:      tide-gauge pairs, error of the mouth-inner level difference
            E(tau) = RMS[h_M(t - tau) - h_I(t)] (both gauges demeaned) for
            no lag, MAREA's lag at the inner gauge and the best constant
            lag; b and r of the predicted difference on the measured one.
* external: Vaklodingen (Dutch sites), from ``results/method_comparison``.
* stations: the Villaviciosa lag per 3-km section (pressure-sensor plan).

usage: python -m experiments.v10_validation_report [--root DIR] [--dutch-res 10]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from experiments.validation_grid import thin_mask  # noqa: E402

MDT02 = "data_v4/truth/pnoa/MDT02-WGS84-0015-3-COB2.tif"
WATER_STD_M = 0.005          # a 3 x 3 window of 2 m pixels flatter than this is a water surface
MIN_GROUND = 0.8             # a 10 m cell needs this share of ground pixels
OUT = "docs/paper/tables"
MAIN = ("no delay", "MAREA", "NIDEM", "Granadeiro")
LABEL = {"no delay": "no delay", "MAREA": "MAREA", "NIDEM": "NIDEM",
         "Granadeiro": "Granadeiro et al.\\ (2021)"}
PAIRS = {  # site: (label, mouth gauge, inner gauge, products folder template, tag)
    "ferrol": ("Ferrol (Ferrol2)", "fer1", "fer2", "products_ferrol", "eot20_g1"),
    "escalda": ("Westerschelde (Terneuzen)", "brsk", "trnz", "products_escalda{suf}", "eot20_g2"),
    "ems": ("Ems-Dollard (Delfzijl)", "bork", "delf", "products_ems{suf}", "eot20_g1"),
    "wadden": ("Wadden Sea, Vlie (Harlingen)", "ters", "harl", "products_wadden{suf}", "eot20_g1"),
}
DUTCH = {"escalda": "Westerschelde", "wadden": "Wadden Sea (Vlie)", "ems": "Ems-Dollard"}


def fmt(v, d=3, sign=False):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def fmt_n(n):
    return f"{int(n):,}".replace(",", "\\,") if n and np.isfinite(n) else "--"


def scenes(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "--"


# ─────────────────────────────────────────────────────────────────────────────
#  elevation tables from v9
# ─────────────────────────────────────────────────────────────────────────────

def method_rows(df, subset="all"):
    out = {}
    for p in MAIN:
        a = df[(df["product"] == p) & (df["subset"] == subset)]
        out[p] = a.iloc[0] if len(a) else None
    return out


def rtk_table(cmp_dir, numbers):
    df = pd.read_csv(f"{cmp_dir}/villaviciosa.csv")
    info = json.load(open(f"{cmp_dir}/summary.json"))["villaviciosa"]
    rows = []
    for p, a in method_rows(df).items():
        if a is None or not a["n"]:
            state = "not run" if a is None or str(a["scenes"]) == "missing" else "not computable"
            rows.append(f"{LABEL[p]} & \\multicolumn{{6}}{{l}}{{{state}}} \\\\")
            continue
        rows.append(f"{LABEL[p]} & {scenes(a['scenes'])} & {int(a['n'])} & {fmt(a['rmse_m'])} & "
                    f"{fmt(a['slope'], 2)} & {fmt(a['r'], 2)} & {fmt(a['bias_m'], 2)} \\\\")
        numbers.setdefault("rtk", {})[p] = {k: float(a[k]) for k in ("n", "rmse_m", "slope", "r", "bias_m")}
    numbers.setdefault("rtk", {})["truth_datum"] = info.get("truth_datum", "ellipsoidal")
    numbers["rtk"]["common_n"] = info.get("common_n")
    return rows


def external_table(cmp_dir, numbers):
    summ = json.load(open(f"{cmp_dir}/summary.json"))
    rows = []
    for site, name in DUTCH.items():
        f = f"{cmp_dir}/{site}.csv"
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        nc = summ.get(site, {}).get("common_n")
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{name}}} (common subset $n_c$ = {fmt_n(nc)})}} \\\\")
        allr, comr = method_rows(df, "all"), method_rows(df, "common")
        for p in MAIN:
            a, c = allr[p], comr[p]
            if a is None or str(a["scenes"]) == "missing":        # v9: no product file yet
                rows.append(f"{LABEL[p]} & \\multicolumn{{6}}{{l}}{{not run}} \\\\")
                continue
            if not a["n"]:
                rows.append(f"{LABEL[p]} & {scenes(a['scenes'])} & \\multicolumn{{5}}{{l}}{{not computable$^{{b}}$}} \\\\")
                continue
            cc = fmt(c["rmse_m"]) if c is not None else "--"
            rows.append(f"{LABEL[p]} & {scenes(a['scenes'])} & {fmt_n(a['n'])} & {fmt(a['rmse_m'])} & "
                        f"{fmt(a['slope'], 2)} & {fmt(a['r'], 2)} & {cc} \\\\")
            numbers.setdefault("vaklodingen", {}).setdefault(site, {})[p] = {
                "n": float(a["n"]), "rmse_m": float(a["rmse_m"]), "slope": float(a["slope"]),
                "r": float(a["r"]), "rmse_common_m": float(c["rmse_m"]) if c is not None else None}
        numbers["vaklodingen"].setdefault(site, {})["common_n"] = nc
        rows.append("\\midrule")
    return rows[:-1]


# ─────────────────────────────────────────────────────────────────────────────
#  LiDAR (Villaviciosa)
# ─────────────────────────────────────────────────────────────────────────────

def lidar_truth(grid_path):
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds
    from scipy import ndimage

    with rasterio.open(grid_path) as g:
        gt, gcrs, gshape, gb = g.transform, g.crs, g.shape, g.bounds
    with rasterio.open(MDT02) as s:
        w = from_bounds(*transform_bounds(gcrs, s.crs, *gb, densify_pts=21), transform=s.transform)
        w = w.round_offsets().round_lengths()
        src = s.read(1, window=w, boundless=True, fill_value=s.nodata).astype("float32")
        src[src == s.nodata] = np.nan
        st, scrs = s.window_transform(w), s.crs
    valid = np.isfinite(src)
    f = np.where(valid, src, 0).astype("float32")
    k = valid.astype("float32")
    n3 = ndimage.uniform_filter(k, 3)
    m3 = ndimage.uniform_filter(f, 3) / np.maximum(n3, 1e-6)
    q3 = ndimage.uniform_filter(f * f, 3) / np.maximum(n3, 1e-6)
    water = valid & (np.sqrt(np.clip(q3 - m3 * m3, 0, None)) < WATER_STD_M) & (n3 > 0.99)
    ground = valid & ~water
    kw = dict(src_transform=st, src_crs=scrs, dst_transform=gt, dst_crs=gcrs,
              resampling=Resampling.average, src_nodata=None, dst_nodata=None)
    gsum, gcnt, vcnt = (np.zeros(gshape) for _ in range(3))
    reproject(np.where(ground, src, 0).astype("float64"), gsum, **kw)
    reproject(ground.astype("float64"), gcnt, **kw)
    reproject(valid.astype("float64"), vcnt, **kw)
    mean_ground = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1e-9), np.nan)
    ok = (vcnt > 0.95) & (gcnt / np.maximum(vcnt, 1e-9) >= MIN_GROUND)
    return np.where(ok, mean_ground, np.nan), gt, gshape, float(water.sum() / valid.sum())


def lidar_table(root, numbers):
    from pyintertidal import rtk
    from pyintertidal.elevation_validation import ols_against_truth

    E = np.load(f"{root}/products_marea/external_products.npz")
    prods = {"no delay": E["no_delay"].astype(float), "MAREA": E["MAREA"].astype(float)}
    H, W = prods["MAREA"].shape
    keep = None
    nid = np.load("products_marea/nidem/nidem.npz", allow_pickle=True)
    gf = f"{root}/products_marea/granadeiro2021/granadeiro2021.npz"
    if os.path.exists(gf):
        g = np.load(gf, allow_pickle=True)
        keep = g["keep"]
        assert np.array_equal(nid["keep"], keep), "NIDEM and Granadeiro are on different pixels"
        r = np.full(H * W, np.nan); r[keep] = g["final_keep"]
        prods["NIDEM"] = None
        prods["Granadeiro"] = r.reshape(H, W)
    keep = nid["keep"] if keep is None else keep
    r = np.full(H * W, np.nan); r[keep] = nid["filtered_keep"]
    prods["NIDEM"] = r.reshape(H, W)
    truth, gt, gshape, water_share = lidar_truth(rtk.DEFAULT_GRID)
    assert gshape == (H, W)
    lo, hi = np.nanmin(prods["no delay"]), np.nanmax(prods["no delay"])
    truth = np.where((truth >= lo - 0.25) & (truth <= hi + 0.25), truth, np.nan)
    common = np.isfinite(truth)
    for v in prods.values():
        common &= np.isfinite(v)
    rows = []
    numbers["lidar"] = {"water_share_2m": water_share, "common_n": int(common.sum()),
                        "truth_cells": int(np.isfinite(truth).sum())}
    for p in MAIN:
        v = prods.get(p)
        if v is None:
            rows.append(f"{LABEL[p]} & \\multicolumn{{6}}{{l}}{{not run}} \\\\")
            continue
        a = ols_against_truth(truth.ravel(), v.ravel())
        c = ols_against_truth(truth[common], v[common])
        rows.append(f"{LABEL[p]} & {fmt_n(a['n'])} & {fmt(a['rmse_m'])} & {fmt(a['slope'], 2)} & "
                    f"{fmt(a['r'], 2)} & {fmt(c['rmse_m'])} & {fmt(a['bias_m'], 2)} \\\\")
        numbers["lidar"][p] = {"n": a["n"], "rmse_m": a["rmse_m"], "slope": a["slope"], "r": a["r"],
                               "bias_m": a["bias_m"], "rmse_common_m": c["rmse_m"]}
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  lag against gauge pairs: E(tau)
# ─────────────────────────────────────────────────────────────────────────────

def lag_pair_value(path, prefix):
    q = pd.read_csv(path)
    hit = q[q["quantity"].str.startswith(prefix)]
    return str(hit["value"].iloc[0]) if len(hit) else None


def inner_distance_km(root, site):
    nb = f"{root}/tide_boundary_comparison_{site}.ipynb"
    if not os.path.exists(nb):
        return None
    txt = open(nb, encoding="utf-8").read()
    m = re.search(r"inner gauge \w+ \([^)]*\) at s = ([0-9.]+) km", txt)
    return float(m.group(1)) if m else None


def lag_table(root, suf, numbers):
    from pyintertidal import lag_validation as lv

    rows = []
    for site, (label, gm, gi, ptemp, tag) in PAIRS.items():
        pdir = f"{root}/{ptemp.format(suf=suf)}"
        jm = f"{pdir}/comparison_{tag}/judge_metrics.csv"
        if not os.path.exists(jm):
            continue
        j = pd.read_csv(jm).set_index("method")
        tau_m = float(j.loc["MAREA", "lag at the judge (min)"])
        lp = f"{pdir}/comparison_{tag}/lag_pair_metrics.csv"
        tau_g = float(lag_pair_value(lp, "gauge-to-gauge lag, all"))
        ci = lag_pair_value(lp, "MAREA lag, 95 % bootstrap interval") or "--"
        res = json.load(open(f"{pdir}/marea_{tag}/result.json"))
        hM, hI = lv.gauge_series(gm), lv.gauge_series(gi)
        t = hI.index.values.astype("int64") / 1e9
        tm = hM.dropna()
        tms, vms = tm.index.values.astype("int64") / 1e9, tm.values
        vi, vm0 = hI.values, hM.reindex(hI.index).values

        def E(tau, full=False):
            mt = np.interp(t - 60.0 * tau, tms, vms, left=np.nan, right=np.nan)
            ok = np.isfinite(mt) & np.isfinite(vi) & np.isfinite(vm0)
            e = float(np.sqrt(np.mean((mt[ok] - vi[ok]) ** 2)))
            if not full:
                return e
            m, p = vm0[ok] - vi[ok], vm0[ok] - mt[ok]           # measured, predicted difference
            if np.allclose(p, 0):
                return e, np.nan, np.nan
            return e, float(np.polyfit(m, p, 1)[0]), float(np.corrcoef(m, p)[0, 1])

        e0 = E(0.0)
        em, b, r = E(tau_m, full=True)
        taus = np.arange(-30, 181, 1.0)
        es = np.array([E(x) for x in taus])
        k = int(np.argmin(es))
        km = inner_distance_km(root, site)
        name = label.replace(")", f", {km:.1f} km)") if km else label
        ci = ci.replace("[", "$[").replace("]", "]$")
        rows.append(f"{name} & {res['n_escenas']} & ${tau_g:+.0f}$ & ${tau_m:+.0f}$ {ci} & "
                    f"{fmt(e0)} & {fmt(em)} & {fmt(es[k])} (${taus[k]:+.0f}$) & {fmt(b, 2)} & {fmt(r, 2)} \\\\")
        numbers.setdefault("lag", {})[site] = {
            "scenes": res["n_escenas"], "tau_gauges": tau_g, "tau_marea": tau_m, "ci": ci,
            "E0": e0, "E_marea": em, "E_best": float(es[k]), "tau_best": float(taus[k]),
            "b": b, "r": r, "explained": 1 - (em / e0) ** 2 if e0 > 0 else None}
    return rows


def stations_table(root, numbers):
    import glob
    f = sorted(glob.glob(f"{root}/products_marea/marea_eot20_g0_*/result.json"))
    if not f:
        return []
    r = json.load(open(f[-1]))
    rows = ["P0 & mouth, reference & $\\approx 0$--1 & --- \\\\"]
    for k, (c, tu, lo, hi) in enumerate(zip(r["centros_km"], r["tau_usado_min"], r["tau_lo_min"], r["tau_hi_min"])):
        if k == 0:
            continue
        rows.append(f"P{k} & section {k} centre & $\\approx {c:.1f}$ & ${tu:.0f}$ $[{lo:.0f}, {hi:.0f}]$ \\\\")
    numbers["stations"] = {"record": os.path.basename(os.path.dirname(f[-1])), "n_scenes": r["n_escenas"],
                           "centres_km": r["centros_km"], "tau_min": r["tau_usado_min"],
                           "ci": list(zip(r["tau_lo_min"], r["tau_hi_min"]))}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".", help="folder with the extracted run results")
    ap.add_argument("--comparison-dir", default=None, help="default: <root>/results/method_comparison")
    ap.add_argument("--dutch-res", type=int, default=10, help="10 (products_<site>_10m) or 20")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-lidar", action="store_true")
    a = ap.parse_args(argv)
    suf = "" if a.dutch_res == 20 else f"_{a.dutch_res}m"
    cmp_dir = a.comparison_dir or f"{a.root}/results/method_comparison"
    os.makedirs(a.out, exist_ok=True)
    numbers = {"root": os.path.abspath(a.root), "comparison_dir": os.path.abspath(cmp_dir),
               "dutch_res_m": a.dutch_res}
    tables = {"rtk": rtk_table(cmp_dir, numbers), "external": external_table(cmp_dir, numbers),
              "lag": lag_table(a.root, suf, numbers), "stations": stations_table(a.root, numbers)}
    if not a.skip_lidar:
        tables["lidar"] = lidar_table(a.root, numbers)
    for name, rows in tables.items():
        open(f"{a.out}/{name}_rows.tex", "w", encoding="utf-8").write("\n".join(rows) + "\n")
        print(f"== {name}\n" + "\n".join(rows))
    json.dump(numbers, open(f"{a.out}/numbers.json", "w"), indent=1, default=float)
    print(f"-> {a.out}/")


if __name__ == "__main__":
    main()
