# -*- coding: utf-8 -*-
"""experiments/v17_truth_diagnostics.py on tiny synthetic data: the metrics
(published centred RMSE, total RMSE, bias), the elevation thirds read from the
truth, the uncertainty calibration and its sigma bins, a synthetic Dutch site
end to end (gates, tables, a NOT COMPUTABLE primary NIDEM, the report), a
synthetic Villaviciosa through a stand-in survey loader, and the RTK and
output guards.

Run:  python -m pytest -q tests/test_truth_diagnostics.py
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import v17_truth_diagnostics as v17  # noqa: E402
from experiments import v9_method_comparison as v9  # noqa: E402

SENS = v17.SENS
GRID = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])


def _vscore():
    from experiments.v7_waterline_contour import vscore
    return vscore


# ── metrics ──────────────────────────────────────────────────────────────────

def test_metrics_reproduce_published_scores_and_add_the_total():
    from pyintertidal import elevation_validation as ev
    vscore = _vscore()
    rng = np.random.default_rng(0)
    t = rng.uniform(-1, 1, 400)
    p = 0.8 * t + 0.3 + rng.normal(0, 0.1, 400)
    p[:7] = np.nan
    m, ref = v17.metrics("vaklodingen", t, p), vscore(p, t)
    for a, b in (("n", "n"), ("rmse_centred_m", "rmse_m"), ("bias_m", "bias_m"), ("slope", "slope"), ("r", "r")):
        assert m[a] == ref[b]                                            # the published function, unchanged
    d = (p - t)[np.isfinite(p)]
    assert m["rmse_total_m"] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert m["rmse_total_m"] ** 2 == pytest.approx(m["mean_diff_m"] ** 2 + m["sd_diff_m"] ** 2)
    assert m["rmse_total_m"] > m["rmse_centred_m"]                      # the 0.3 m offset counts
    mr, rr = v17.metrics("rtk", t, p), ev.ols_against_truth(t, p)
    assert (mr["rmse_centred_m"], mr["bias_m"], mr["slope"]) == (rr["rmse_m"], rr["bias_m"], rr["slope"])
    # a subset scores exactly as v9's where-masking
    sel = rng.random(400) < 0.5
    a, b = v17.metrics("vaklodingen", t, p, sel), vscore(np.where(sel, p, np.nan), t)
    assert (a["n"], a["rmse_centred_m"], a["slope"], a["r"], a["bias_m"]) == \
        (b["n"], b["rmse_m"], b["slope"], b["r"], b["bias_m"])
    # below the minimum n: NaN, the n reported
    few = v17.metrics("vaklodingen", t[:25], p[:25])
    assert few["n"] == 18 and np.isnan(few["rmse_centred_m"]) and np.isnan(few["rmse_total_m"])
    assert np.isnan(v17.metrics("rtk", t[:25], p[:25])["rmse_total_m"])


def test_published_centring_can_exceed_the_total_rmse():
    # skewed truth and a slope far from 1: centring product and truth by their
    # own medians is not removing the bias (the caveat the tables state)
    rng = np.random.default_rng(4)
    t = rng.exponential(0.5, 5000)
    p = 0.3 * t
    p = p + (np.mean(t) - np.mean(p))                                    # mean(p - t) = 0
    m = v17.metrics("vaklodingen", t, p)
    assert abs(m["mean_diff_m"]) < 1e-12
    assert m["rmse_centred_m"] > m["rmse_total_m"]
    assert m["rmse_total_m"] == pytest.approx(m["sd_diff_m"])


def test_elevation_thirds_are_read_from_the_truth_only():
    t = np.array([-1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0, np.nan])
    sel = np.isfinite(t)
    sel[6] = False                                                       # 2.0 m is not in the set
    cls, edges = v17.truth_classes(t, sel)
    assert edges == pytest.approx([-1.0, -1 / 3, 1 / 3, 1.0])
    assert cls.tolist() == [0, 0, 1, 1, 2, 2, -1, -1]
    assert sum(int((cls == k).sum()) for k in range(3)) == int(sel.sum())  # a partition of the set


# ── calibration ──────────────────────────────────────────────────────────────

def test_calibrated_gaussian_errors_give_68_and_95_percent():
    rng = np.random.default_rng(1)
    n = 60000
    s = GRID[rng.integers(0, 8, n)]
    t = rng.uniform(-1, 1, n)
    p = t + 0.4 + rng.normal(0, 1, n) * s                                # site bias 0.4 m, calibrated sigma
    out, bins = v17.calibrate("vaklodingen", t, p, s, np.ones(n, bool), thin=np.arange(n) % 7 == 0)
    assert out["bias_removed_m"] == pytest.approx(0.4, abs=0.01)
    assert out["within_1sigma"] == pytest.approx(v17.GAUSS_1, abs=0.01)
    assert out["within_2sigma"] == pytest.approx(v17.GAUSS_2, abs=0.01)
    assert out["k68"] == pytest.approx(1.0, abs=0.03) and out["z_rms"] == pytest.approx(1.0, abs=0.03)
    assert out["spearman_rho"] > 0.4 and out["spearman_rho_thinned"] > 0.4
    assert out["ence"] < 0.05 and out["n_bins_scored"] == 8
    assert out["bin_rule"] == "one bin per distinct value"
    assert len(bins) == 8 and all(b["sigma_lo_m"] == b["sigma_hi_m"] for b in bins)
    for b in bins:                                                       # on the 1:1 line, bin by bin
        assert b["q68_abs_e_m"] == pytest.approx(b["sigma_median_m"], rel=0.08)


def test_understated_and_constant_sigma():
    rng = np.random.default_rng(2)
    n = 20000
    t = rng.uniform(-1, 1, n)
    p = t + rng.normal(0, 0.3, n)
    out, bins = v17.calibrate("vaklodingen", t, p, np.full(n, 0.1), np.ones(n, bool))
    assert np.isnan(out["spearman_rho"])                                 # a constant sigma ranks nothing
    assert len(bins) == 1
    assert out["within_1sigma"] == pytest.approx(0.2611, abs=0.015)     # P(|Z| < 1/3)
    assert out["k68"] == pytest.approx(3.0, rel=0.05)
    out2, bins2 = v17.calibrate("rtk", t[:10], p[:10], np.full(10, 0.1), np.ones(10, bool))
    assert out2["n"] == 10 and np.isnan(out2["within_1sigma"]) and bins2 == []


def test_sigma_bins_distinct_values_or_quantiles():
    s = np.repeat([0.03, 0.1, 0.65], [5, 7, 9])
    idx, bounds, _ = v17.sigma_bins(s)
    assert bounds == [(0.03, 0.03), (0.1, 0.1), (0.65, 0.65)] and np.bincount(idx).tolist() == [5, 7, 9]
    rng = np.random.default_rng(3)
    idx, bounds, rule = v17.sigma_bins(rng.uniform(0.05, 0.15, 1000))
    assert len(bounds) == 10 and np.bincount(idx).min() >= 95 and "quantile" in rule
    tied = np.concatenate([np.full(500, 0.09), rng.uniform(0.1, 0.2, 500)])
    idx, _, _ = v17.sigma_bins(tied)
    assert len(set(idx[:500].tolist())) == 1                             # equal values never split


# ── synthetic sites on disk ──────────────────────────────────────────────────

def _published(path, site, truth, prods, kind):
    """The CSV v9 --same-scenes would write for these arrays: 'all' rows and
    its common subsets, with v9's own functions."""
    from pyintertidal import elevation_validation as ev
    vscore = _vscore()

    def fn(v):
        return vscore(v, truth) if kind == "vaklodingen" else ev.ols_against_truth(truth, v)
    rows = []
    for name, v in prods.items():
        s = fn(v)
        rows.append({"site": site, "product": name, "subset": "all", "n": s["n"], "rmse_m": s["rmse_m"],
                     "slope": s["slope"], "r": s["r"], "bias_m": s["bias_m"]})
    extra, _ = v9.same_scene_rows(site, prods, {p: ("", None) for p in prods}, np.isfinite(truth), fn)
    rows += [{k: r[k] for k in ("site", "product", "subset", "n", "rmse_m", "slope", "r", "bias_m")} for r in extra]
    pd.DataFrame(rows).to_csv(path, index=False)


def _nidem_dir(path, keep, dem, unc, dates, status="ok"):
    os.makedirs(path, exist_ok=True)
    if status == "ok":
        np.savez_compressed(os.path.join(path, "nidem.npz"), keep=keep, filtered_keep=dem.astype(np.float32),
                            uncertainty_keep=unc.astype(np.float32), dates=np.array(dates))
    json.dump({"status": status, "n_scenes": len(dates)}, open(os.path.join(path, "nidem_diagnostics.json"), "w"))


def dutch_site(tmp, monkeypatch, primary=True, seed=0):
    """A 30 x 40 synthetic Dutch site: files in the production layout, the
    truth through a stand-in for v14.dutch_truth, the published CSV as v9
    writes it. Returns (overrides, arrays as v17 must load them)."""
    rng = np.random.default_rng(seed)
    H, W, P = 30, 40, 900
    keep = np.sort(rng.choice(H * W, P, replace=False))
    cols = keep % W
    s_km = cols * 0.25 + rng.uniform(0, 0.2, P)
    band = np.minimum((s_km // 3).astype(np.int64), 3)
    z = rng.uniform(-1.0, 1.2, P)
    truth = np.where(rng.random(P) < 0.1, np.nan, z + rng.normal(0, 0.02, P))
    sig = GRID[rng.integers(1, 7, P)]
    z_marea = z + 0.2 + sig * rng.normal(0, 1, P)                       # MAREA's sigma calibrated by construction
    z_marea[rng.random(P) < 0.05] = np.nan
    sig = np.where(np.isfinite(z_marea), sig, np.nan)
    z_plain = z + 0.15 + rng.normal(0, 0.35, P)
    z_plain[rng.random(P) < 0.05] = np.nan
    nid = np.where(rng.random(P) < 0.05, np.nan, z - 0.4 + rng.normal(0, 0.3, P))
    unc = np.where(np.isfinite(nid), 0.1 + 0.01 * rng.random(P), np.nan)
    sens = np.where(rng.random(P) < 0.05, np.nan, z - 0.35 + rng.normal(0, 0.25, P))
    unc_s = np.where(np.isfinite(sens), 0.12 + 0.01 * rng.random(P), np.nan)
    rec = [f"2024-01-{d:02d}" for d in range(1, 13)]
    allrec = [f"2024-{m:02d}-01" for m in range(1, 13)] + rec[:3]
    d = os.path.join(str(tmp), "site")
    os.makedirs(os.path.join(d, "comparison"))
    os.makedirs(os.path.join(d, "marea"))
    np.savez(os.path.join(d, "extract.npz"), keep=keep, shape=np.array([H, W]))
    np.savez_compressed(os.path.join(d, "comparison", "products.npz"), keep=keep, shape=np.array([H, W]),
                        s_km=s_km, band=band, z_plain=z_plain, z_marea=z_marea, record_dates=np.array(rec))
    np.savez_compressed(os.path.join(d, "marea", "marea.npz"), z=z_marea.astype(np.float32),
                        sigma=sig.astype(np.float32), band=band.astype(np.int16), keep=keep,
                        s_km=s_km.astype(np.float32), shape=np.array([H, W]))
    json.dump({"record_dates": rec, "centros_km": [1.5, 4.5, 7.5, 10.5], "tau_usado_min": [0.0, 2.0, 5.0, 9.0]},
              open(os.path.join(d, "marea", "result.json"), "w"))
    _nidem_dir(os.path.join(d, "nidem"), keep, nid, unc, allrec,
               status="ok" if primary else "NOT COMPUTABLE: no clear observation in interval 1")
    _nidem_dir(os.path.join(d, "nidem", v17.SENS_DIR), keep, sens, unc_s, rec)
    truth_info = {"truth": "synthetic", "pixel_m": 10.0}
    monkeypatch.setattr(v17.v14, "dutch_truth", lambda site, cfg, keep_, H_, W_, zp: (truth.copy(), truth_info))
    as_loaded = {"no delay": z_plain, "MAREA": z_marea, SENS: sens.astype(np.float32).astype(float)}
    if primary:
        as_loaded["NIDEM"] = nid.astype(np.float32).astype(float)
    pub = os.path.join(d, "published.csv")
    _published(pub, "escalda", truth, as_loaded, "vaklodingen")
    ov = {"extract": os.path.join(d, "extract.npz"), "product": os.path.join(d, "marea"),
          "comparison": os.path.join(d, "comparison"), "nidem": os.path.join(d, "nidem"),
          "cube": "unused.nc", "published": pub}
    return ov, dict(truth=truth, prods=as_loaded, band=band, sigma=sig, keep=keep)


@pytest.mark.parametrize("primary", [True, False])
def test_dutch_site_end_to_end(tmp_path, monkeypatch, primary):
    ov, A = dutch_site(tmp_path, monkeypatch, primary=primary)
    out = str(tmp_path / "out")
    tables, st = v17.run_site("escalda", out=out, overrides=ov)
    assert st["status"] == "complete" and v17.gates_ok(st["gates"])
    assert st["gates"]["published_all_rows"]["max_abs_diff"] <= v17.TOL_SCORE
    assert st["gates"]["published_common_subsets"]["checked"]
    for f in ("by_band", "by_class", "calibration", "reliability", "totals"):
        assert os.path.exists(os.path.join(out, f"escalda_{f}.csv"))
    t, P = A["truth"], A["prods"]
    members = ["MAREA", SENS] + (["NIDEM"] if primary else [])
    C = np.isfinite(t)
    for p in members:
        C &= np.isfinite(P[p])
    assert st["sets"]["common"] == int(C.sum())
    tot = tables["totals"]
    com = tot[tot["subset"] == "common"].set_index("product")
    assert set(members) <= set(com.index) and "no delay" in com.index
    assert all(int(com.loc[p, "n"]) == int(C.sum()) for p in members)           # identical pixels
    assert int(com.loc["no delay", "n"]) == int((C & np.isfinite(P["no delay"])).sum())
    assert (tot["subset"] == "pair_published").any() == primary
    if not primary:
        nc = tot[(tot["product"] == "NIDEM")]
        assert len(nc) == 1 and nc.iloc[0]["status"].startswith("NOT COMPUTABLE") and nc.iloc[0]["n"] == 0
    # bands and thirds partition the common set; the thirds do not depend on the product
    bb, bc = tables["by_band"], tables["by_class"]
    for p in members + ["no delay"]:
        g = bb[(bb["product"] == p) & (bb["band"].astype(str) != "all")]
        assert int(g["n_set"].sum()) == int(C.sum())
        assert sorted(g["band"].astype(int)) == sorted(set(A["band"][C].tolist()))
    sets_c = bc[bc["class"] != "all"].pivot_table(index="class", columns="product", values="n_set")
    assert (sets_c.nunique(axis=1) == 1).all() and int(sets_c.iloc[:, 0].sum()) == int(C.sum())
    # calibration: MAREA's planted sigma is calibrated, NIDEM's 0.1 m understates its 0.3 m errors
    cal = tables["calibration"].set_index("product")
    assert cal.loc["MAREA", "within_1sigma"] == pytest.approx(v17.GAUSS_1, abs=0.07)
    assert cal.loc[SENS, "within_1sigma"] < 0.5 and cal.loc[SENS, "k68"] > 1.5
    if primary:
        assert cal.loc["NIDEM", "within_1sigma"] < 0.4
    else:
        assert cal.loc["NIDEM", "status"].startswith("NOT COMPUTABLE")
    assert len(tables["reliability"])
    # the report: combined tables, both languages, the three figures
    D, missing = v17.report(out=out, dpi=40, formats=("png",))
    assert list(D) == ["escalda"] and set(missing) == {"villaviciosa", "wadden", "ems"}
    for f in ("truth_diagnostics.md", "truth_diagnostics.es.md", "truth_totals.csv", "truth_gates.csv"):
        assert os.path.exists(os.path.join(out, f))
    for stem in v17.FIG_STEMS.values():
        for lang in ("es", "en"):
            assert os.path.getsize(os.path.join(out, f"{stem}_{lang}.png")) > 1000
    md_es = open(os.path.join(out, "truth_diagnostics.es.md"), encoding="utf-8").read()
    assert "Escalda occidental" in md_es and ("no computable" in md_es) == (not primary)


def test_failed_gates_stop_unless_soft(tmp_path, monkeypatch):
    ov, A = dutch_site(tmp_path, monkeypatch)
    out = str(tmp_path / "out")
    pub = pd.read_csv(ov["published"])
    k = pub.index[(pub["product"] == "MAREA") & (pub["subset"] == "all")][0]
    pub.loc[k, "rmse_m"] += 1e-6                                         # a published row we do not reproduce
    pub.to_csv(ov["published"], index=False)
    with pytest.raises(SystemExit):
        v17.run_site("escalda", out=out, overrides=ov)
    st = json.load(open(os.path.join(out, "escalda_settings.json"), encoding="utf-8"))
    assert st["status"] == "gate_failed" and st["failed_gates"] == ["published_all_rows"]
    assert not os.path.exists(os.path.join(out, "escalda_totals.csv"))
    _, st2 = v17.run_site("escalda", out=out, overrides=ov, soft_gates=True)
    assert st2["status"] == "complete_gates_overridden" and st2["gates_overridden"] == ["published_all_rows"]


def test_sigma_and_record_gates(tmp_path, monkeypatch):
    ov, A = dutch_site(tmp_path, monkeypatch)
    # the z next to MAREA's sigma is not the scored MAREA on a unit with truth
    mz = os.path.join(ov["product"], "marea.npz")
    M = dict(np.load(mz))
    i = int(np.flatnonzero(np.isfinite(A["truth"]) & np.isfinite(A["prods"]["MAREA"]))[0])
    M["z"] = M["z"].copy()
    M["z"][i] += 0.01
    np.savez_compressed(mz, **M)
    with pytest.raises(SystemExit):
        v17.run_site("escalda", out=str(tmp_path / "o1"), overrides=ov)
    st = json.load(open(os.path.join(str(tmp_path / "o1"), "escalda_settings.json"), encoding="utf-8"))
    assert st["failed_gates"] == ["sigma_belongs_to_scored_marea"]
    M["z"][i] -= 0.01
    np.savez_compressed(mz, **M)
    # the sensitivity NIDEM is not on MAREA's record
    sp = os.path.join(ov["nidem"], v17.SENS_DIR, "nidem.npz")
    Z = dict(np.load(sp))
    Z["dates"] = np.array([str(x).replace("2024-01-12", "2024-01-13") for x in Z["dates"]])
    np.savez_compressed(sp, **Z)
    with pytest.raises(SystemExit):
        v17.run_site("escalda", out=str(tmp_path / "o2"), overrides=ov)
    st = json.load(open(os.path.join(str(tmp_path / "o2"), "escalda_settings.json"), encoding="utf-8"))
    assert st["failed_gates"] == ["same_scenes_record"]


def test_villaviciosa_points_through_the_survey_loader(tmp_path, monkeypatch):
    from pyintertidal import rtk
    rng = np.random.default_rng(7)
    H, W = 20, 25
    keep = np.sort(rng.choice(H * W, 300, replace=False))
    z = rng.uniform(-1.0, 1.2, H * W)

    def raster(off, sd):
        r = np.full(H * W, np.nan, np.float32)
        r[keep] = (z[keep] + off + rng.normal(0, sd, len(keep))).astype(np.float32)
        return r.reshape(H, W)
    MA, ND = raster(0.1, 0.15), raster(0.12, 0.2)
    sig = GRID[rng.integers(1, 7, len(keep))].astype(np.float32)
    nid = (z[keep] - 0.05 + rng.normal(0, 0.2, len(keep))).astype(np.float32)
    sens = (z[keep] - 0.04 + rng.normal(0, 0.18, len(keep))).astype(np.float32)
    rec = [f"2023-0{m}-15" for m in range(1, 10)]
    d = str(tmp_path / "vva")
    os.makedirs(os.path.join(d, "marea"))
    np.savez(os.path.join(d, "extract.npz"), keep=keep, shape=np.array([H, W]))
    np.savez(os.path.join(d, "external_products.npz"), MAREA=MA, no_delay=ND)
    np.savez_compressed(os.path.join(d, "marea", "marea.npz"), z=MA.ravel()[keep], sigma=sig,
                        band=np.zeros(len(keep), np.int16), s_km=np.full(len(keep), 4.0, np.float32), keep=keep)
    json.dump({"record_dates": rec, "centros_km": [1.5, 4.3, 7.6], "tau_usado_min": [0.0, 5.0, 20.7]},
              open(os.path.join(d, "marea", "result.json"), "w"))
    _nidem_dir(os.path.join(d, "nidem"), keep, nid, np.full(len(keep), 0.1), [f"2023-{m:02d}-02" for m in range(1, 13)])
    _nidem_dir(os.path.join(d, "nidem", v17.SENS_DIR), keep, sens, np.full(len(keep), 0.12), rec)
    # 80 development points, some outside the extraction, some sharing a pixel
    n = 80
    pix = np.concatenate([rng.choice(keep, 60), rng.choice(np.setdiff1d(np.arange(H * W), keep), 20)])
    r_, c_ = pix // W, pix % W
    alc = z[pix] + rng.normal(0, 0.03, n)
    dev = {"row": r_, "col": c_, "elev": alc + 52.8, "elev_alicante": alc, "block": np.zeros(n, int),
           "split": np.full(n, "dev"), "n_reserved_hidden": 7}
    calls = []

    def fake_loader(*a, **k):
        calls.append((a, k))
        return dict(dev)
    monkeypatch.setattr(rtk, "load_rtk", fake_loader)

    def on_pts(v):
        g = np.full(H * W, np.nan)
        g[keep] = v
        return g.reshape(H, W)[r_, c_]
    prods = {"no delay": ND.astype(float)[r_, c_], "MAREA": MA.astype(float)[r_, c_],
             "NIDEM": on_pts(nid.astype(float)), SENS: on_pts(sens.astype(float))}
    pub = os.path.join(d, "published.csv")
    _published(pub, "villaviciosa", alc, prods, "rtk")
    ov = {"extract": os.path.join(d, "extract.npz"), "product": os.path.join(d, "marea"),
          "nidem": os.path.join(d, "nidem"), "external": os.path.join(d, "external_products.npz"),
          "published": pub}
    S = v17.load_villaviciosa(v17.site_inputs("villaviciosa", ov))
    assert calls == [((), {})]                                           # the survey's default door only
    assert np.array_equal(S["truth"], alc)                               # the national datum, as v9's tables
    for p, v in prods.items():
        assert np.array_equal(S["prods"][p], v, equal_nan=True)
    assert np.isnan(S["prods"]["MAREA"][60:]).all()                     # points outside the extraction
    tables, st = v17.run_site("villaviciosa", out=str(tmp_path / "out"), overrides=ov)
    assert st["status"] == "complete" and v17.gates_ok(st["gates"])
    assert st["truth"]["datum"].startswith("Alicante") and st["truth"]["n_reserved_hidden"] == 7
    a = tables["totals"].query("subset == 'all' and product == 'MAREA'").iloc[0]
    assert int(a["n"]) == 60
    edges = st["classes"]["edges_m"]
    assert edges[0] == pytest.approx(float(np.min(alc[:60]))) and edges[3] == pytest.approx(float(np.max(alc[:60])))


# ── review fixes: band -1, truth error, reliability slope, clamps, units, bootstrap ──

def _set_band_minus_one(ov, share=0.03, seed=0):
    """Give ``share`` of the keep pixels band -1 and no mouth distance (as the
    production products of every Dutch site have), in both band maps."""
    cp = os.path.join(ov["comparison"], "products.npz")
    mp = os.path.join(ov["product"], "marea.npz")
    P, M = dict(np.load(cp)), dict(np.load(mp))
    idx = np.random.default_rng(seed).choice(len(P["band"]), int(share * len(P["band"])), replace=False)
    for D_ in (P, M):
        D_["band"] = D_["band"].copy()
        D_["band"][idx] = -1
        D_["s_km"] = D_["s_km"].astype(float).copy()
        D_["s_km"][idx] = np.nan
    M["s_km"] = M["s_km"].astype(np.float32)
    M["band"] = M["band"].astype(np.int16)
    np.savez_compressed(cp, **P)
    np.savez_compressed(mp, **M)
    return idx


def test_band_minus_one_pixels_pass_the_band_gate_and_get_their_own_row(tmp_path, monkeypatch):
    ov, A = dutch_site(tmp_path, monkeypatch)
    idx = _set_band_minus_one(ov)
    C = np.isfinite(A["truth"])
    for p in ("MAREA", "NIDEM", SENS):
        C &= np.isfinite(A["prods"][p])
    n_m1 = int(C[idx].sum())
    assert n_m1 > 0                                                      # some band -1 pixels hold truth
    out = str(tmp_path / "out")
    tables, st = v17.run_site("escalda", out=out, overrides=ov)
    assert st["status"] == "complete" and v17.gates_ok(st["gates"])
    g = st["gates"]["band_map"]
    assert g["ok"] and g["bands_without_centre_in_common"] == [] and g["n_common_without_mouth_distance"] == n_m1
    bb = tables["by_band"]
    m1 = bb[(bb["product"] == "MAREA") & (bb["band"].astype(str) == "-1")]
    assert len(m1) == 1 and int(m1.iloc[0]["n_set"]) == n_m1 and m1.iloc[0]["tau_applied_min"] == 0.0
    allb = bb[(bb["product"] == "MAREA") & (bb["band"].astype(str) != "all")]
    assert int(allb["n_set"].sum()) == int(C.sum())                     # -1 included: still a partition
    v17.report(out=out, dpi=40, formats=("png",))
    md = open(os.path.join(out, "truth_diagnostics.md"), encoding="utf-8").read()
    assert "no mouth distance, lag 0" in md
    # a band beyond the result.json centres still fails the gate
    cp = os.path.join(ov["comparison"], "products.npz")
    mp = os.path.join(ov["product"], "marea.npz")
    P, M = dict(np.load(cp)), dict(np.load(mp))
    j = int(np.flatnonzero(C)[0])
    for D_ in (P, M):
        D_["band"] = D_["band"].copy()
        D_["band"][j] = 9
    np.savez_compressed(cp, **P)
    np.savez_compressed(mp, **M)
    with pytest.raises(SystemExit):
        v17.run_site("escalda", out=str(tmp_path / "o9"), overrides=ov)


def test_calibration_with_the_truth_error_and_the_reliability_slope():
    rng = np.random.default_rng(11)
    n = 40000
    s = GRID[rng.integers(0, 8, n)]
    t = rng.uniform(-1, 1, n)
    st_true = 0.08                                                       # the truth's own error
    p = t + 0.3 + rng.normal(0, 1, n) * np.sqrt(s ** 2 + st_true ** 2)
    out, bins = v17.calibrate("vaklodingen", t, p, s, np.ones(n, bool), sigma_t_site=st_true)
    # against sigma alone the small sigmas look understated; with sigma_t added they are calibrated
    assert out["within_1sigma"] < v17.GAUSS_1 - 0.05
    assert out["within_1sigma_tsite"] == pytest.approx(v17.GAUSS_1, abs=0.01)
    assert out["k68_tsite"] == pytest.approx(1.0, abs=0.03) and out["sigma_t_site_m"] == st_true
    for st in v17.SIGMA_TS:
        k = v17.t_key(st)
        assert np.isfinite(out[f"within_1sigma_{k}"]) and np.isfinite(out[f"k68_{k}"])
    w = [out[f"within_1sigma_{v17.t_key(st)}"] for st in v17.SIGMA_TS]
    assert w == sorted(w) and out["within_1sigma_t000"] == out["within_1sigma"]
    # the per-bin q68 against sigma: slope below 1 (the floor sigma_t lifts the small bins)
    assert 0.5 < out["reliability_slope"] < 1.0 and out["reliability_intercept_m"] > 0
    assert "does not follow" in out["reliability_verdict"] or "follows" in out["reliability_verdict"]
    # calibrated by construction (no truth error): slope ~ 1, intercept ~ 0, 'follows'
    p2 = t + 0.3 + rng.normal(0, 1, n) * s
    o2, _ = v17.calibrate("vaklodingen", t, p2, s, np.ones(n, bool))
    assert o2["reliability_slope"] == pytest.approx(1.0, abs=0.08) and o2["reliability_verdict"].startswith("q68 follows")
    assert "within_1sigma_tsite" not in o2 and np.isnan(o2["sigma_t_site_m"])
    # a constant sigma: one bin, no slope
    o3, _ = v17.calibrate("vaklodingen", t, p2, np.full(n, 0.1), np.ones(n, bool))
    assert np.isnan(o3["reliability_slope"]) and o3["reliability_verdict"].startswith("fewer than 2")


def test_rtk_truth_error_is_the_pooled_within_pixel_sd():
    W = 10
    rows = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    cols = np.array([0, 0, 0, 5, 5, 1, 2, 2])
    z = np.array([1.0, 1.2, 1.4, 0.0, 0.3, 5.0, 2.0, np.nan])
    keep = np.array([0, 15, 21, 32])                                    # every pixel above is in keep
    te = v17.rtk_truth_error(z, rows, cols, keep, W)
    ss = ((1.0 - 1.2) ** 2 + 0 + (1.4 - 1.2) ** 2) + ((0.0 - 0.15) ** 2 + (0.3 - 0.15) ** 2)
    assert te["sigma_t_m"] == pytest.approx(np.sqrt(ss / 3)) and te["dof"] == 3
    assert te["n_pixels_with_repeats"] == 2 and te["n_points_in_them"] == 5   # the NaN height is not used
    assert np.isnan(v17.rtk_truth_error(z[:1], rows[:1], cols[:1], keep, W)["sigma_t_m"])


def test_clamp_shares_at_each_methods_ceiling():
    cl = v17.marea_clamp({"rango_marea_m": [-2.0, 1.0]})
    step = 3.0 / 49
    assert cl["tol"] == pytest.approx(step / 2) and cl["hi"] == 1.0
    z = np.array([1.0, 1.0 - step, -2.0, 0.0, np.nan])
    sh = v17.clamp_shares(z, np.ones(5, bool), cl)
    assert sh["share_at_upper_clamp"] == pytest.approx(0.25) and sh["share_at_lower_clamp"] == pytest.approx(0.25)
    nc = v17.nidem_clamp(np.linspace(-1.5, 0.9, 9))
    assert v17.clamp_shares(np.array([0.9, 0.95, 0.0]), np.ones(3, bool), nc)["share_at_upper_clamp"] == \
        pytest.approx(2 / 3)
    assert np.isnan(v17.clamp_shares(z, np.ones(5, bool), None)["share_at_upper_clamp"])


def test_units_file_and_the_paired_bootstrap(tmp_path, monkeypatch):
    ov, A = dutch_site(tmp_path, monkeypatch)
    out = str(tmp_path / "out")
    tables, st = v17.run_site("escalda", out=out, overrides=ov)
    U = v17.load_units(os.path.join(out, "escalda_units.npz"))
    fin = np.isfinite(A["truth"])
    assert np.array_equal(U["truth"], A["truth"][fin]) and U["kind"] == "vaklodingen"
    for p in ("MAREA", "NIDEM", SENS, "no delay"):
        assert np.array_equal(U["prods"][p], np.asarray(A["prods"][p], float)[fin], equal_nan=True)
    assert int(U["mask_common"].sum()) == st["sets"]["common"]
    assert int(U["mask_calibration"].sum()) == st["sets"]["calibration"]
    v17.bootstrap_site("escalda", out=out, n_boot=150, seed=3, block_m=50.0)           # 5 x 5 px blocks: 48 clusters
    df = pd.read_csv(os.path.join(out, "escalda_bootstrap.csv"))
    assert set(df["comparison"]) == {"MAREA - NIDEM", f"MAREA - {SENS}", "MAREA - no delay"}
    j = json.load(open(os.path.join(out, "escalda_bootstrap.json"), encoding="utf-8"))
    assert j["gate_points_reproduce_totals"]["ok"]                      # same units, same functions
    com = tables["totals"][tables["totals"]["subset"] == "common"].set_index("product")
    r = df[(df["comparison"] == "MAREA - NIDEM") & (df["stat"] == "rmse_total")].iloc[0]
    assert r["a_value"] == pytest.approx(com.loc["MAREA", "rmse_total_m"], abs=1e-12)
    assert r["b_value"] == pytest.approx(com.loc["NIDEM", "rmse_total_m"], abs=1e-12)
    assert r["ci_lo"] < r["diff"] < r["ci_hi"] and r["n_clusters"] == 48 and not r["indicative"]
    assert "0.05 km blocks" in r["cluster_rule"] and r["n_valid_resamples"] == 150
    assert (df["stat"] == "within_1sigma").sum() == 2                    # the two pairs that state an uncertainty
    # the planted calibrations: MAREA's sigma is the better one on the same units
    w = df[(df["comparison"] == "MAREA - NIDEM") & (df["stat"] == "within_1sigma")].iloc[0]
    assert w["diff"] > 0.1 and w["ci_lo"] > 0
    D, missing = v17.report(out=out, dpi=40, formats=("png",))
    assert os.path.exists(os.path.join(out, "truth_bootstrap.csv"))
    es = open(os.path.join(out, "truth_diagnostics.es.md"), encoding="utf-8").read()
    assert "Diferencias pareadas" in es and " pp" in es


def test_spanish_numbers_follow_one_convention():
    assert v17._int(1379, "es") == "1379" and v17._int(32120, "es") == "32 120"
    assert v17._int(1614155, "es") == "1 614 155" and v17._int(32120) == "32,120"
    assert v17._num(0.1906, 3, lang="es") == "0,191" and v17._num(-0.07, 2, True, "es") == "−0,07"
    assert v17._pct(0.639, "es") == "63,9 %" and v17._pct(0.639) == "63.9 %"


# ── guards ───────────────────────────────────────────────────────────────────

def test_rtk_read_only_through_load_rtk_with_defaults():
    from tests import test_r1_guard
    static = [f for name, f in vars(test_r1_guard).items() if name.startswith("test_static")]
    assert static
    for f in static:
        f()                                                              # the static scan covers this script too
    tree = ast.parse(open(v17.__file__, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "attr", None) == "load_rtk" or getattr(n.func, "id", None) == "load_rtk")]
    assert len(calls) == 1 and not calls[0].args and not calls[0].keywords


def test_outputs_refuse_production_paths():
    for bad in ("products_marea/x", "products_escalda_10m/nidem_eot20_g2", "results/method_comparison_same_scenes"):
        with pytest.raises(SystemExit):
            v17.run_site("escalda", out=bad)
        with pytest.raises(SystemExit):
            v17.report(out=bad)


def test_cli_overrides():
    a = v17.build_parser().parse_args(["run", "ems", "--marea-dir", "m", "--nidem-dir", "n", "--soft-gates"])
    assert (a.site, a.product, a.nidem, a.soft_gates) == ("ems", "m", "n", True)
    cfg = v17.site_inputs("ems", {k: getattr(a, k, None) for k in v17.OVERRIDE_KEYS})
    assert cfg["product"] == "m" and cfg["nidem_sensitivity"] == os.path.join("n", v17.SENS_DIR)
    assert cfg["published"] == "results/method_comparison_same_scenes/ems.csv"
    assert v17.site_inputs("wadden")["product"] == "products_wadden_10m/marea_eot20_g1"
