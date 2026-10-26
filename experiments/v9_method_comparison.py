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

Same-scenes comparison (2026-10-19, ``--same-scenes``): adds, at every site,
the labelled sensitivity ``NIDEM (on MAREA's record)`` (v7
``--sensitivity-marea-record``: NIDEM on exactly the scenes MAREA scored,
asserted against MAREA's stored record_dates), scored like every other row,
plus the subsets ``common_published`` (truth & no delay & MAREA & NIDEM, the
common subset of docs/paper/tables/external_rows.tex; checked against the
v7 CSV) and ``common_same_scenes`` (truth & no delay & MAREA & NIDEM on
MAREA's record). Writes to results/method_comparison_same_scenes/ only
({<site>.csv, summary.json, same_scenes_summary.csv, same_scenes_summary.md});
the default run and its outputs are unchanged.
    python -m experiments.v9_method_comparison --same-scenes [site ...]
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
#: the same-scenes comparison (--same-scenes) writes here only
OUT_SAME = "results/method_comparison_same_scenes"
#: the sensitivity row: NIDEM on exactly MAREA's scored scene record
SENS = "NIDEM (on MAREA's record)"
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
              nidem_variant=f"{_p}/nidem_{_c['tag']}_s2_scene_rule",
              nidem_sensitivity=f"{_p}/nidem_{_c['tag']}/sensitivity_marea_record")

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
    # a NOT COMPUTABLE run writes its diagnostics but no nidem.npz: read the
    # verdict first, so it is not mistaken for a run that never happened
    d = json.load(open(diag, encoding="utf-8")) if os.path.exists(diag) else {}
    if str(d.get("status", "ok")).upper().startswith("NOT"):
        return None, d.get("n_scenes"), "NOT COMPUTABLE"
    if not os.path.exists(npz):
        return None, None, "missing"
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


def marea_record_dates(cfg):
    """The scene dates MAREA scored (record_dates of its result.json), or None."""
    files = [f for f in cfg["lag_result"] if os.path.exists(f)]
    if not files:
        return None
    d = json.load(open(files[-1], encoding="utf-8")).get("record_dates")
    return [str(x)[:10] for x in d] if d else None


def check_same_scenes(path, marea_dates):
    """The sensitivity's scenes (nidem.npz 'dates') must be exactly MAREA's
    scored record; returns them."""
    f = os.path.join(path, "nidem.npz")
    with np.load(f, allow_pickle=True) as z:
        d = [str(x)[:10] for x in z["dates"]]
    if marea_dates is None:
        print(f"  WARNING: no MAREA record_dates to check {f} against")
    elif d != list(marea_dates):
        raise AssertionError(f"{f}: its {len(d)} scenes are not MAREA's "
                             f"{len(marea_dates)}-scene record")
    else:
        print(f"  {SENS}: its {len(d)} scenes are exactly MAREA's record")
    return d


def same_scene_rows(site, prods, records, truth_ok, score_fn):
    """The same-scenes subsets (``--same-scenes``) for the products in
    ``prods`` (values at the scoring points): ``common_published`` = truth &
    no delay & MAREA & NIDEM (the common subset of external_rows.tex; no
    delay & MAREA where NIDEM is not computable), ``common_same_scenes`` =
    truth & no delay & MAREA & NIDEM on MAREA's record, and ``common4`` =
    both NIDEMs too. ``score_fn(values)`` returns the metrics dict or None."""
    pub = ["no delay", "MAREA"] + (["NIDEM"] if "NIDEM" in prods else [])
    subsets = [("common_published", pub)]
    if SENS in prods:
        subsets.append(("common_same_scenes", ["no delay", "MAREA", SENS]))
        if "NIDEM" in prods:
            subsets.append(("common4", ["no delay", "MAREA", "NIDEM", SENS]))
    rows, info = [], {}
    for label, members in subsets:
        sel = truth_ok.copy()
        for p in members:
            sel &= np.isfinite(prods[p])
        info[label] = {"members": members, "n": int(sel.sum())}
        for p in members:
            v = np.where(sel, prods[p], np.nan)
            s = score_fn(v)
            rows.append({"site": site, "product": p, "subset": label,
                         "rule": records[p][0], "scenes": records[p][1],
                         **({k: s[k] for k in ("n", "rmse_m", "slope", "r", "bias_m")} if s else
                            {"n": int(np.sum(np.isfinite(v) & truth_ok)), "rmse_m": np.nan,
                             "slope": np.nan, "r": np.nan, "bias_m": np.nan})})
    return rows, info


# ─────────────────────────────────────────────────────────────────────────────
#  scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_villaviciosa(cfg, same_scenes=False):
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
    # the notebook (and the reproduction above) scores ellipsoidal heights;
    # the table uses the national datum when the survey carries it, so the
    # bias is the product zero against Alicante (RMSE, slope and r are
    # median-centred and change only by the geoid's few-cm spatial variation)
    if "elev_alicante" in dev:
        zt_t, datum = np.asarray(dev["elev_alicante"], float), "Alicante (REDNAP orthometric)"
    else:
        zt_t, datum = zt, "ETRS89 ellipsoidal"
    rows = []
    for name, v in samp.items():
        for subset, sel in (("all", np.ones(len(zt), bool)), ("common", common)):
            f = ev.ols_against_truth(zt_t, np.where(sel, v, np.nan))
            rows.append({"site": "villaviciosa", "product": name, "subset": subset,
                         "rule": records[name][0], "scenes": records[name][1],
                         **({k: f[k] for k in ("n", "rmse_m", "slope", "r", "bias_m")} if f else
                            {"n": int(np.sum(np.isfinite(v) & sel)), "rmse_m": np.nan, "slope": np.nan,
                             "r": np.nan, "bias_m": np.nan})})
    info = {"common_members": main_present, "common_n": int(common.sum()),
            "truth_datum": datum}
    if same_scenes:
        if SENS in samp:
            check_same_scenes(cfg["nidem_sensitivity"], marea_record_dates(cfg))
        extra, info["same_scenes"] = same_scene_rows(
            "villaviciosa", samp, records, np.isfinite(zt_t),
            lambda v: ev.ols_against_truth(zt_t, v))
        rows += extra
    return pd.DataFrame(rows), info


def score_dutch(site, cfg, same_scenes=False):
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
    if same_scenes:
        # NIDEM on exactly MAREA's scored record (v7 --sensitivity-marea-record)
        sen, n_s, st_s = load_nidem(cfg["nidem_sensitivity"], keep)
        if sen is not None:
            mdates = ([str(x)[:10] for x in P["record_dates"]] if "record_dates" in P.files
                      else marea_record_dates(cfg))
            check_same_scenes(cfg["nidem_sensitivity"], mdates)
            prods[SENS] = sen
        records[SENS] = ("MAREA's record", n_s if sen is not None else st_s)

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
    missing = [("NIDEM", st_n), ("NIDEM (S2 scene rule)", st_v), ("Granadeiro", st_g)]
    if same_scenes:
        missing.append((SENS, records[SENS][1]))
    for name, st in missing:
        if name not in prods:
            rows.append({"site": site, "product": name, "subset": "all", "rule": records[name][0],
                         "scenes": st, "n": 0, "rmse_m": np.nan, "slope": np.nan, "r": np.nan,
                         "bias_m": np.nan})
    info = {"common_members": members, "common_n": int(common.sum()),
            "truth_px": int(np.isfinite(truth_raw).sum()),
            "truth_in_range": int(np.isfinite(truth).sum()), "level_range_nodelay": [lo_, hi_]}
    if same_scenes:
        extra, info["same_scenes"] = same_scene_rows(site, prods, records, np.isfinite(truth),
                                                     lambda v: vscore(v, truth))
        rows += extra
        info["same_scenes"]["common_published_vs_v7_csv_max_abs_diff"] = \
            check_published_common(cfg, extra)
    return pd.DataFrame(rows), info


def check_published_common(cfg, rows):
    """The ``common_published`` rows must reproduce the common subset of the
    v7 CSV behind docs/paper/tables/external_rows.tex (common3 no delay &
    MAREA & NIDEM; common2 where NIDEM is not computable). Returns the max
    |diff| (None if the CSV is absent); warns, does not stop, above TOL."""
    f = f"{cfg['comparison']}/vaklodingen_metrics_with_nidem.csv"
    if not os.path.exists(f):
        print(f"  WARNING: {f} not found; common_published not cross-checked")
        return None
    ref = pd.read_csv(f)
    com = ref[ref["subset"].astype(str).str.startswith("common3")]
    if com.empty:
        com = ref[ref["subset"].astype(str).str.startswith("common2")]
    worst = 0.0
    for r in rows:
        if r["subset"] != "common_published":
            continue
        c = com[com["product"] == r["product"]]
        if c.empty:
            print(f"  WARNING: no {r['product']} row in the common subset of {f}")
            worst = np.inf
            continue
        for k in ("n", "rmse_m", "slope", "r", "bias_m"):
            worst = max(worst, abs(float(r[k]) - float(c[k].iloc[0])))
    msg = f"common_published vs {f}: max |diff| {worst:.1e}"
    print(("  " if worst <= TOL else "  WARNING: NOT REPRODUCED, ") + msg)
    return float(worst)


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


# ─────────────────────────────────────────────────────────────────────────────
#  same-scenes summary (--same-scenes)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_PRODUCTS = (("no delay", "no delay"), ("MAREA", "MAREA"),
                    ("NIDEM", "NIDEM (all observations, as published)"),
                    (SENS, SENS))
COMMONS = ("common_published", "common_same_scenes")


def same_scenes_summary(frames, summary):
    """One row per site and method: scenes, n, RMSE, slope, r, bias on all
    pixels / points, and the RMSE (and n) on the published common subset and
    on the same-scenes common subset."""
    rows = []
    for df in frames:
        site = str(df["site"].iloc[0])
        for prod, label in SUMMARY_PRODUCTS:
            a = df[(df["product"] == prod) & (df["subset"] == "all")]
            if a.empty:
                continue
            a = a.iloc[0]
            r = {"site": site, "method": label, "rule": a["rule"], "scenes": a["scenes"],
                 **{k: a[k] for k in ("n", "rmse_m", "slope", "r", "bias_m")}}
            for key in COMMONS:
                c = df[(df["product"] == prod) & (df["subset"] == key)]
                r[f"n_{key}"] = c["n"].iloc[0] if len(c) else np.nan
                r[f"rmse_{key}_m"] = c["rmse_m"].iloc[0] if len(c) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _f(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v) if v is not None else "--"
    return "--" if not np.isfinite(v) else f"{v:.{d}f}"


def _n(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(v) or v <= 0 else f"{int(v):,}"


def summary_markdown(tab, summary):
    truth = {"villaviciosa": "RTK-GNSS (development split, Alicante datum)"}
    lines = ["# Same-scenes comparison: no delay vs MAREA vs NIDEM",
             "",
             f"NIDEM (on MAREA's record) = NIDEM run on exactly the scenes MAREA scored "
             f"(v7 --sensitivity-marea-record; dates asserted equal). Metrics as the "
             f"published tables (v9): RMSE, slope and r median-centred, bias = median "
             f"(product - truth). common_published = truth & no delay & MAREA & NIDEM "
             f"(all observations), the common subset of external_rows.tex; "
             f"common_same_scenes = truth & no delay & MAREA & NIDEM (on MAREA's record).",
             ""]
    for site in tab["site"].unique():
        d = tab[tab["site"] == site]
        info = summary.get(site, {}).get("same_scenes", {})
        npub = (info.get("common_published") or {}).get("n")
        nss = (info.get("common_same_scenes") or {}).get("n")
        lines += [f"## {SITES[site]['name']} ({site}); truth: {truth.get(site, SITES[site]['truth'])}",
                  "",
                  f"common_published n = {_n(npub)}; common_same_scenes n = {_n(nss)}",
                  "",
                  "| method | scenes | n | RMSE (m) | slope | r | bias (m) | RMSE common published (m) | RMSE common same scenes (m) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in d.iterrows():
            sc = r["scenes"]
            sc = _n(sc) if isinstance(sc, (int, float, np.integer, np.floating)) else str(sc)
            lines.append(f"| {r['method']} | {sc} | {_n(r['n'])} | {_f(r['rmse_m'])} | "
                         f"{_f(r['slope'], 2)} | {_f(r['r'], 2)} | {_f(r['bias_m'])} | "
                         f"{_f(r['rmse_common_published_m'])} | {_f(r['rmse_common_same_scenes_m'])} |")
        lines.append("")
    errs = {s: v["error"] for s, v in summary.items() if isinstance(v, dict) and "error" in v}
    if errs:
        lines += ["## Sites not scored", ""] + [f"* {s}: {e}" for s, e in errs.items()] + [""]
    return "\n".join(lines)


def main(sites, same_scenes=False):
    out = OUT_SAME if same_scenes else OUT
    os.makedirs(out, exist_ok=True)
    frames, summary = [], {}
    for site in sites:
        print(f"[{site}]")
        cfg = SITES[site]
        if same_scenes:
            # one site failing (e.g. a sensitivity not run yet) must not lose the others
            try:
                df, info = (score_villaviciosa(cfg, same_scenes=True) if site == "villaviciosa"
                            else score_dutch(site, cfg, same_scenes=True))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                summary[site] = {"error": f"{type(e).__name__}: {e}"}
                continue
        else:
            df, info = score_villaviciosa(cfg) if site == "villaviciosa" else score_dutch(site, cfg)
        df.to_csv(f"{out}/{site}.csv", index=False)
        frames.append(df)
        summary[site] = info
        shown = ["all", "common"] + (list(COMMONS) if same_scenes else [])
        show = df[df["subset"].isin(shown)][["product", "subset", "scenes", "n", "rmse_m",
                                            "slope", "r", "bias_m"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=str)
    if same_scenes:
        tab = same_scenes_summary(frames, summary)
        tab.to_csv(f"{out}/same_scenes_summary.csv", index=False)
        open(f"{out}/same_scenes_summary.md", "w", encoding="utf-8").write(
            summary_markdown(tab, summary) + "\n")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"-> {out}/same_scenes_summary.{{csv,md}}")
        return
    allf = pd.concat(frames, ignore_index=True)
    open(f"{out}/tables.tex", "w", encoding="utf-8").write(latex_rows(allf) + "\n")
    print(f"-> {out}/")


if __name__ == "__main__":
    _args = sys.argv[1:]
    _same = "--same-scenes" in _args
    _sites = [a for a in _args if a != "--same-scenes"]
    _bad = [s for s in _sites if s not in SITES]
    if _bad:
        raise SystemExit(f"unknown site(s) {_bad}; choose from {list(SITES)} "
                         "(optional flag: --same-scenes)")
    main(_sites or list(SITES), same_scenes=_same)
