# -*- coding: utf-8 -*-
"""The elevation comparison of every method, on the same pixels, against the
same truth, with the same metrics: the single source of the comparison tables.

Principle agreed with the author (2026-09-29): every method works on the same
archive (the site's cube, Sentinel-2 L2A, same resolution, 2023-2025, same
boundary tide) and is scored on the same pixels (the MAREA extraction's keep)
with the same truth and metrics; on top of that, each method applies its own
published rules (scene selection and pixel masks):

* ``no delay``  MAREA's inversion with the lag set to zero, on MAREA's record
                (transition-zone cloud screening, the usable dates);
* ``MAREA``     the same record, with the interior lag;
* ``Granadeiro``  Granadeiro et al. (2021), faithful reimplementation
                (experiments/v8_granadeiro2021.py): its own scene rule
                (< 10 % cloud), own intertidal mask, nplr logistic, 50,000-px
                lag sample, mgcv GAM;
* ``NIDEM``     ITEM/NIDEM waterline-contour interpolation (Sagar et al. 2017;
                Bishop-Taylor et al. 2019), ITEM all-observations rule and PQ
                mask (experiments/v7_waterline_contour.py primary);
* ``NIDEM (S2 scene rule)``  the same with the Digital Earth Africa Sentinel-2
                scene rule (load_ard min_gooddata 0.5), a labelled variant.

Before anything new is scored, the no-delay and MAREA scores are recomputed
and must reproduce the notebooks' own CSVs (products_marea/rtk_onsite_metrics.csv,
products_<site>/comparison_<tag>/vaklodingen_metrics.csv).

Subsets: ``all`` (each product on every scoring pixel/point it resolves),
``thinned`` (Vaklodingen only: one pixel per 100 m block, as the notebooks:
5x5 px at 20 m, 10x10 px at 10 m),
``common`` (the pixels/points every main product resolves: no delay, MAREA,
Granadeiro and NIDEM, the ITEM-rule NIDEM where computable, else its S2
variant).

Writes results/method_comparison/{<site>.csv, summary.json, tables.tex}.
Run from the repository root:  python -m experiments.v9_method_comparison [site ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from experiments.validation_grid import dutch_cube, dutch_products, thin_mask  # noqa: E402

OUT = "results/method_comparison"
TOL = 1e-9
RTK_CSV = "data/villaviciosa_rtk_gnss.csv"

SITES = {
    "villaviciosa": dict(name="Ría de Villaviciosa", truth="RTK-GNSS (development split)",
                         cube="ndwi_cube_villaviciosa_grande_10y.nc",
                         extract="products_marea/marea_extract.npz",
                         products="products_marea/external_products.npz",
                         notebook_csv="products_marea/rtk_onsite_metrics.csv",
                         lag_result=sorted(glob.glob("products_marea/marea_eot20_g0_*/result.json")),
                         granadeiro="products_marea/granadeiro2021",
                         nidem="products_marea/nidem",
                         nidem_variant="products_marea/nidem/s2_scene_rule",
                         nidem_sensitivity="products_marea/nidem/sensitivity_marea_record"),
    "escalda": dict(name="Westerschelde", truth="Vaklodingen", tag="eot20_g2",
                    cube=dutch_cube("escalda")),
    "wadden": dict(name="Wadden Sea (Vlie)", truth="Vaklodingen", tag="eot20_g1",
                   cube=dutch_cube("wadden")),
    "ems": dict(name="Ems-Dollard", truth="Vaklodingen", tag="eot20_g1",
                cube=dutch_cube("ems")),
}
for _s, _c in SITES.items():
    if _s == "villaviciosa":
        continue
    _p = dutch_products(_s)
    _c.update(extract=f"{_p}/marea_extract.npz",
              comparison=f"{_p}/comparison_{_c['tag']}",
              lag_result=[f"{_p}/marea_{_c['tag']}/result.json"],
              granadeiro=f"{_p}/granadeiro2021_{_c['tag']}",
              nidem=f"{_p}/nidem_{_c['tag']}",
              nidem_variant=f"{_p}/nidem_{_c['tag']}_s2_scene_rule")

MAIN = ("no delay", "MAREA", "Granadeiro", "NIDEM")


# ─────────────────────────────────────────────────────────────────────────────
#  loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_keep(cfg):
    z = np.load(cfg["extract"], allow_pickle=True)
    return np.asarray(z["keep"]), tuple(int(v) for v in z["shape"])


def load_nidem(path, keep):
    """The FILTERED NIDEM DEM on the keep pixels (primary product), or None
    when the run is NOT COMPUTABLE / missing. Also returns the scene count."""
    npz, diag = os.path.join(path, "nidem.npz"), os.path.join(path, "nidem_diagnostics.json")
    if not os.path.exists(npz):
        return None, None, "missing"
    d = json.load(open(diag, encoding="utf-8")) if os.path.exists(diag) else {}
    if str(d.get("status", "ok")).upper().startswith("NOT"):
        return None, d.get("n_scenes"), "NOT COMPUTABLE"
    z = np.load(npz, allow_pickle=True)
    assert np.array_equal(z["keep"], keep), f"{path}: keep differs from the extraction"
    n_sc = int(len(z["dates"])) if "dates" in z.files else d.get("n_scenes")
    return z["filtered_keep"].astype(float), n_sc, "ok"


#: the product file experiments/v8_granadeiro2021.py writes; the same folder
#: also holds caches (boundary_1min_cache.npz, ...) that are not products
GRAN_FILE = "granadeiro2021.npz"
GRAN_KEY = "final_keep"


def load_granadeiro(path, keep):
    """The faithful Granadeiro final DEM on the keep pixels, or None."""
    f = os.path.join(path, GRAN_FILE)
    if not os.path.exists(f):
        return None, None, "missing"
    z = np.load(f, allow_pickle=True)
    if GRAN_KEY not in z.files:
        raise KeyError(f"{f}: no {GRAN_KEY!r} in {z.files}")
    assert np.array_equal(z["keep"], keep), f"{f}: keep differs from the extraction"
    v = z[GRAN_KEY].astype(float)
    assert v.shape == keep.shape, (f, v.shape, keep.shape)
    n_sc = None
    for k in ("scene_dates", "dates", "used_dates"):
        if k in z.files:
            n_sc = int(len(z[k])); break
    return v, n_sc, "ok"


def marea_record_size(cfg):
    files = [f for f in cfg["lag_result"] if os.path.exists(f)]
    if not files:
        return None
    return int(json.load(open(files[-1], encoding="utf-8"))["n_escenas"])


# ─────────────────────────────────────────────────────────────────────────────
#  scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_villaviciosa(cfg):
    from pyintertidal import elevation_validation as ev
    from pyintertidal import rtk

    keep, (H, W) = load_keep(cfg)
    E = np.load(cfg["products"])
    rasters = {"no delay": E["no_delay"].astype(float), "MAREA": E["MAREA"].astype(float)}
    dev = rtk.load_rtk(RTK_CSV)
    r_, c_, zt = dev["row"], dev["col"], dev["elev"]

    # 1. the notebook's own numbers must come out again
    ref = pd.read_csv(cfg["notebook_csv"]).set_index("product")
    worst = 0.0
    for name, ras in rasters.items():
        f = ev.ols_against_truth(zt, ras[r_, c_])
        assert f["n"] == int(ref.loc[name, "n"]), (name, f["n"], ref.loc[name, "n"])
        for k in ("rmse_m", "slope", "intercept", "r", "bias_m"):
            worst = max(worst, abs(f[k] - float(ref.loc[name, k])))
    assert worst <= TOL, f"rtk_onsite_metrics.csv not reproduced (max diff {worst})"
    print(f"  reproduced {cfg['notebook_csv']}: max |diff| {worst:.1e}")

    def on_grid(vals):
        r = np.full(H * W, np.nan); r[keep] = vals
        return r.reshape(H, W)

    records = {"no delay": ("MAREA record: transition-zone screening", marea_record_size(cfg)),
               "MAREA": ("MAREA record: transition-zone screening", marea_record_size(cfg))}
    g, n_g, st_g = load_granadeiro(cfg["granadeiro"], keep)
    if g is not None:
        rasters["Granadeiro"] = on_grid(g)
    records["Granadeiro"] = ("own rule: scene cloud < 10 %", n_g if g is not None else st_g)
    for name, key, rule in (("NIDEM", "nidem", "ITEM: all observations"),
                            ("NIDEM (S2 scene rule)", "nidem_variant", "DE Africa load_ard, good >= 50 %"),
                            ("NIDEM (on MAREA's record)", "nidem_sensitivity", "MAREA's record")):
        v, n_sc, st = load_nidem(cfg[key], keep)
        if v is not None:
            rasters[name] = on_grid(v)
        records[name] = (rule, n_sc if v is not None else st)
    if "Granadeiro" in E.files:
        rasters["Granadeiro (superseded implementation)"] = E["Granadeiro"].astype(float)
        records["Granadeiro (superseded implementation)"] = ("frame-rule scenes, old code", None)

    samp = {name: np.asarray(ras[r_, c_], float) for name, ras in rasters.items()}
    main_present = [p for p in MAIN if p in samp]
    common = np.ones(len(zt), bool)
    for p in main_present:
        common &= np.isfinite(samp[p])
    rows = []
    for name, v in samp.items():
        for subset, sel in (("all", np.ones(len(zt), bool)), ("common", common)):
            f = ev.ols_against_truth(zt, np.where(sel, v, np.nan))
            rows.append({"site": "villaviciosa", "product": name, "subset": subset,
                         "rule": records[name][0], "scenes": records[name][1],
                         **({k: f[k] for k in ("n", "rmse_m", "slope", "r", "bias_m")} if f else
                            {"n": int(np.sum(np.isfinite(v) & sel)), "rmse_m": np.nan, "slope": np.nan,
                             "r": np.nan, "bias_m": np.nan})})
    return pd.DataFrame(rows), {"common_members": main_present, "common_n": int(common.sum())}


def score_dutch(site, cfg):
    import pyproj
    from experiments import v1_vaklodingen_validate as v1
    from experiments.v7_waterline_contour import cube_grid, vscore

    os.chdir(ROOT)
    keep, (H, W) = load_keep(cfg)
    P = np.load(f"{cfg['comparison']}/products.npz", allow_pickle=True)
    assert np.array_equal(P["keep"], keep), "products.npz keep differs from the extraction"
    grid = cube_grid(cfg["cube"])
    assert (H, W) == (grid["H"], grid["W"])
    rows_k, cols_k = keep // W, keep % W
    to_rd = pyproj.Transformer.from_crs(grid["crs"], "EPSG:28992", always_xy=True)
    xq, yq = to_rd.transform(grid["xs"][cols_k], grid["ys"][rows_k])
    cells = v1.truth_grid(site)
    truth_raw, _ = v1.sample_truth(cells, xq, yq)
    z_plain = P["z_plain"].astype(float)
    lo_, hi_ = float(np.nanmin(z_plain)), float(np.nanmax(z_plain))
    in_range = (truth_raw >= lo_ - 0.25) & (truth_raw <= hi_ + 0.25)
    truth = np.where(in_range, truth_raw, np.nan)
    thin = thin_mask(rows_k, cols_k, abs(float(grid["xs"][1] - grid["xs"][0])))

    prods = {"no delay": z_plain, "MAREA": P["z_marea"].astype(float)}
    # 1. the notebook's own numbers must come out again (all and thinned)
    ref = pd.read_csv(f"{cfg['comparison']}/vaklodingen_metrics.csv").set_index("product")
    worst = 0.0
    for name, v in prods.items():
        a = vscore(v, truth); t = vscore(np.where(thin, v, np.nan), truth)
        assert a["n"] == int(ref.loc[name, "n"]), (name, a["n"], ref.loc[name, "n"])
        for mine, col in ((a["rmse_m"], "RMSE (m)"), (a["slope"], "slope"), (a["r"], "r"),
                          (a["bias_m"], "bias (m)"), (t["rmse_m"], "thinned RMSE (m)"),
                          (t["slope"], "thinned slope")):
            worst = max(worst, abs(mine - float(ref.loc[name, col])))
    assert worst <= TOL, f"vaklodingen_metrics.csv not reproduced (max diff {worst})"
    print(f"  reproduced {cfg['comparison']}/vaklodingen_metrics.csv: max |diff| {worst:.1e}")

    n_rec = int(len(P["record_dates"])) if "record_dates" in P.files else marea_record_size(cfg)
    records = {"no delay": ("MAREA record: transition-zone screening", n_rec),
               "MAREA": ("MAREA record: transition-zone screening", n_rec)}
    g, n_g, st_g = load_granadeiro(cfg["granadeiro"], keep)
    if g is not None:
        prods["Granadeiro"] = g
    records["Granadeiro"] = ("own rule: scene cloud < 10 %", n_g if g is not None else st_g)
    nid, n_n, st_n = load_nidem(cfg["nidem"], keep)
    if nid is not None:
        prods["NIDEM"] = nid
    records["NIDEM"] = ("ITEM: all observations", n_n if nid is not None else st_n)
    var, n_v, st_v = load_nidem(cfg["nidem_variant"], keep)
    if var is not None:
        prods["NIDEM (S2 scene rule)"] = var
    records["NIDEM (S2 scene rule)"] = ("DE Africa load_ard, good >= 50 %", n_v if var is not None else st_v)
    if "z_gran" in P.files:
        prods["Granadeiro (superseded implementation)"] = P["z_gran"].astype(float)
        records["Granadeiro (superseded implementation)"] = ("old code", None)

    members = ["no delay", "MAREA"] + [p for p in ("Granadeiro",) if p in prods]
    members += ["NIDEM"] if "NIDEM" in prods else (["NIDEM (S2 scene rule)"] if "NIDEM (S2 scene rule)" in prods else [])
    common = np.isfinite(truth)
    for p in members:
        common &= np.isfinite(prods[p])
    rows = []
    for name, v in prods.items():
        for subset, sel in (("all", None), ("thinned", thin), ("common", common)):
            vv = v if sel is None else np.where(sel, v, np.nan)
            s = vscore(vv, truth)
            rows.append({"site": site, "product": name, "subset": subset,
                         "rule": records[name][0], "scenes": records[name][1],
                         **(s if s else {"n": int(np.sum(np.isfinite(vv) & np.isfinite(truth))),
                                         "rmse_m": np.nan, "slope": np.nan, "r": np.nan,
                                         "bias_m": np.nan})})
    for name, st in (("NIDEM", st_n), ("NIDEM (S2 scene rule)", st_v), ("Granadeiro", st_g)):
        if name not in prods:
            rows.append({"site": site, "product": name, "subset": "all", "rule": records[name][0],
                         "scenes": st, "n": 0, "rmse_m": np.nan, "slope": np.nan, "r": np.nan,
                         "bias_m": np.nan})
    info = {"common_members": members, "common_n": int(common.sum()),
            "truth_px": int(np.isfinite(truth_raw).sum()),
            "truth_in_range": int(np.isfinite(truth).sum()), "level_range_nodelay": [lo_, hi_]}
    return pd.DataFrame(rows), info


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX rows
# ─────────────────────────────────────────────────────────────────────────────

def fmt_n(n):
    return f"{int(n):,}".replace(",", "\\,") if n and np.isfinite(n) and n > 0 else "--"


def latex_rows(df):
    lines = []
    for site in df["site"].unique():
        d = df[df["site"] == site]
        lines.append(f"% {site}")
        for prod in ("no delay", "MAREA", "Granadeiro", "NIDEM", "NIDEM (S2 scene rule)"):
            a = d[(d["product"] == prod) & (d["subset"] == "all")]
            c = d[(d["product"] == prod) & (d["subset"] == "common")]
            if a.empty:
                continue
            a, c = a.iloc[0], (c.iloc[0] if not c.empty else None)
            if not a["n"]:
                lines.append(f"{prod} & \\multicolumn{{4}}{{l}}{{not computable ({a['scenes']})}} & & & \\\\")
                continue
            cc = (f"{c['rmse_m']:.3f} & {c['slope']:.2f} & {c['r']:.2f}"
                  if c is not None and np.isfinite(c["rmse_m"]) else "-- & -- & --")
            lines.append(f"{prod} & {fmt_n(a['n'])} & {a['rmse_m']:.3f} & {a['slope']:.2f} & "
                         f"{a['r']:.2f} & {cc} \\\\")
    return "\n".join(lines)


def main(sites):
    os.makedirs(OUT, exist_ok=True)
    frames, summary = [], {}
    for site in sites:
        print(f"[{site}]")
        cfg = SITES[site]
        df, info = score_villaviciosa(cfg) if site == "villaviciosa" else score_dutch(site, cfg)
        df.to_csv(f"{OUT}/{site}.csv", index=False)
        frames.append(df)
        summary[site] = info
        show = df[df["subset"].isin(["all", "common"])][["product", "subset", "scenes", "n", "rmse_m",
                                                         "slope", "r", "bias_m"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    allf = pd.concat(frames, ignore_index=True)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
    open(f"{OUT}/tables.tex", "w", encoding="utf-8").write(latex_rows(allf) + "\n")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SITES))
