"""Tests of experiments.benchmark.scoring, the lag scoring of Stage L.

A synthetic site is written in the on-disk layout of the V2 plan (§6 of the
interfaces): a package stand-in on a synthetic boundary tide (records every
2 days of 2024, the exact level table, the 1-min and 10-min series, marks),
the per-window lag truths built by ``truth.lag_truths_window`` itself, the
judges' table (``judges.load`` recomputes the neighbourhoods), the lag plan
(``plans/lag.csv``), one P0 instance and the products of every result kind
(field, band, phase field with a pixel interval, phase band, point with
secondary models and model-internal values, groups, a NOT COMPUTABLE one, a
not applicable / NOT AVAILABLE plan row, a planned row without product),
the positive-control verdicts and the gates. The tests pin:

* E(tau) equals the truth's E_a curve (and ``truth.e_rows``) bitwise at
  integer tau, E_all equals ``truth.tau_all_window``; a real tau reads the
  level rows linearly; dE = 0 and LLE = 0 at tau_a; AGF = 1 at tau_a and 0
  at tau = 0; 'lag not material' when E0 - Emin < 2 cm;
* err_int, the interval score and the coverage; the bootstrap is paired
  (one weight matrix per judge and window, shared by every product) and
  deterministic;
* end to end: the window-year guard rows, judge exclusion, the status texts
  (NOT COMPUTABLE, not applicable, NOT AVAILABLE, implementation not
  validated), the read-out ladder (step 2 for a pixel method with a hole),
  the phase products, L14@<model> and @internal rows, L01v per-orbit rows,
  L20's leave-one-out reference, the Terneuzen pair as one location, the
  pre-lock banner and the descriptive L-P1 without verdict, the coherence
  table of a site without judges, the efficiency table, and the refusal to
  score without LOCK.json phase A;
* profile metrics on 4 synthetic locations; control verdicts (PASS,
  mismatch, fault, phase, offsets, both scenarios); G-lagtruth on a
  synthetic level table.

Run:  python -X utf8 -m pytest tests/test_benchmark_scoring_lag.py -q
"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import (  # noqa: E402
    config, designs, judges, lagtable, levels, product, scoring, truth)
from experiments.benchmark.lags import common as lc  # noqa: E402

SITE = "escalda"
NOJ = "villaviciosa"
CRS = "EPSG:32631"
X0, Y0, PIX = 500_000.0, 5_700_000.0, 100.0
H, W = 40, 60
P0 = "C1.floor0.ndwi.D1"
WIN = "Y2024"
CONS = ("m2", "s2", "o1", "k1", "n2", "m4")
B = 40
P = lc.PERIOD_M2_MIN
#: the judges: pixel (row, col), planted lag (min), datum offset (m)
GAUGES = {"gatvanborssele": dict(px=(12, 20), lag=20, offset=0.3),
          "terneuzen": dict(px=(20, 40), lag=35, offset=-0.2),
          "terneuzen.westsluis.buiten": dict(px=(22, 42), lag=37, offset=0.1),
          "borssele": dict(px=(30, 10), lag=12, offset=0.0, excluded=True)}


def _grid():
    return {"xs": X0 + PIX * np.arange(W), "ys": Y0 - PIX * np.arange(H), "crs": CRS}


def _domain():
    rr, cc = np.meshgrid(np.arange(4, 36), np.arange(2, 58), indexing="ij")
    keep = np.sort((rr * W + cc).ravel()).astype(np.int64)
    s_km = ((keep % W) - 2) * 0.1 + 0.05
    band_of = np.floor(s_km / 1.0).astype(np.int64)
    K = int(band_of.max()) + 1
    centers = np.array([np.median(s_km[band_of == k]) for k in range(K)])
    return keep, s_km, band_of, centers


def _gauge_series(pkg, lag, offset, seed, sampling=10):
    idx = pd.date_range("2024-01-01", "2024-12-31 23:50", freq=f"{sampling}min").as_unit("ns")
    rng = np.random.default_rng(seed)
    v = pkg.tide.value(idx.asi8 - int(lag) * 60_000_000_000) + offset
    return pd.Series(v + rng.normal(0, 0.01, len(idx)), index=idx)


def _judge_records(grid, keep, s_km, band_of, centers):
    from pyproj import Transformer

    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    recs = []
    for code, g in GAUGES.items():
        r, c = g["px"]
        gx, gy = float(grid["xs"][c]), float(grid["ys"][r])
        lon, lat = tr.transform(gx, gy)
        geom = judges.station_geometry(lon, lat, grid, keep, s_km, band_of, centers)
        R = 400.0
        nb = judges.nbhd_flat(grid, gx, gy, R)
        recs.append({"code": code, "name": code, "provider": "RWS", "lon": lon, "lat": lat,
                     "datum": "NAP", "sampling_min": 10.0, "file": f"{code}.csv",
                     "location": judges.location_of(code), "gx": gx, "gy": gy,
                     "s_g_km": geom["s_g_km"], "k_g": geom["k_g"], "R_g_m": R,
                     "nbhd": nb, "nbhd_sha256": config.sha256_bytes(nb.astype("<i8").tobytes()),
                     "mouth": False, "included": not g.get("excluded", False),
                     "in_frame": True, "nearest_member_km": 10.0,
                     "flags": ["sparse"] if code == "terneuzen.westsluis.buiten" else []})
    return recs


def _plan_row(site, window, method, products, status="planned", note="", variant=""):
    u = designs.Unit(design="lag", site=site, window=window, setting=P0, method=method,
                     variant=variant, status=status, note=note)
    row = u.row(1)
    row["products"] = products
    return {k: str(v) for k, v in row.items()}


def _write_plan(root, rows):
    os.makedirs(os.path.join(root, "plans"), exist_ok=True)
    pd.DataFrame(rows, columns=list(designs.CSV_COLUMNS)).to_csv(
        os.path.join(root, "plans", "lag.csv"), index=False, lineterminator="\n")


def _write_instance(root, site, window, keep, s_km, band_of, centers, sel):
    d = os.path.join(root, "runs", site, window, P0, "_instance", "rand-A", "Nall_r00")
    product.save_npz(os.path.join(d, "instance.npz"), {
        "sel": np.asarray(sel, np.int64), "t_ns": np.zeros(len(sel), np.int64),
        "h0": np.zeros(len(sel)), "keep": keep, "pos": np.arange(len(keep), dtype=np.int64),
        "band_of": band_of.astype(np.int16), "s_km": s_km.astype(np.float32),
        "centers": centers, "edges": np.arange(len(centers) + 1, dtype=float),
        "K": np.array(len(centers)), "status": np.array("ok"),
        "orbit": np.zeros(len(sel), np.int64), "platform": np.array(["S2A"] * len(sel))})
    product.write_json(os.path.join(d, "instance.json"),
                       {"format": scoring.INSTANCE_FORMAT, "site": site, "window": window})
    return d


def _write_product(root, site, window, pid, inst_dir, *, status="ok", kind=None,
                   lag_kind="linear", arrays=None, method=None, variant="", nested=False,
                   code_key=None, **outcome):
    d = scoring.product_dir(root, site, window, P0, pid, variant)
    os.makedirs(d, exist_ok=True)
    if arrays:
        product.save_npz(os.path.join(d, "product.npz"), arrays)
    ident = {"format": scoring.LAG_PRODUCT_FORMAT, "product": pid, "method": method or pid,
             "variant": variant, "designs": ["lag"], "unit_id": "u", "site": site,
             "window": window, "setting": P0, "family": "rand", "mode": "A", "N": "all",
             "rep": 0}
    out = {"status": status, "stage": outcome.pop("stage", None),
           "reason": outcome.pop("reason", None), "kind": kind, "lag_kind": lag_kind,
           "edge_share": 0.0, "n_defined": 1, **outcome}
    run = {"instance": {"path": os.path.relpath(inst_dir, d), "instance_sha256": "x"},
           "timings": {f"lag.{pid}.fit": {"wall_s": 2.0, "cpu_self_s": 1.5,
                                          "peak_rss_mb": 300.0, "peak_anon_mb": 120.0}},
           "unit_total": {"wall_s": 3.0},
           # the adapter code of the product (the printing rule compares it with
           # the positive control's)
           "keys": {"code_files_key": code_key or f"ck-{pid}"}}
    if nested:
        run.update(identity=ident, outcome=out)
    else:
        run.update(ident, **out)
    product.write_json(os.path.join(d, "run.json"), run)
    return d


def build_site(root):
    """Write the synthetic scored site under ``root``; return what the tests use."""
    grid = _grid()
    pkg = scoring.synthetic_package(SITE, years=(2024,), grid=grid)
    t1, h1 = pkg.series1
    pkg.marks_lp = levels.build_marks(t1, h1)["marks_lp"]
    pkg_v = scoring.synthetic_package(NOJ, years=(2024,), grid=grid, seed=11)
    keep, s_km, band_of, centers = _domain()
    K = len(centers)
    table = _judge_records(grid, keep, s_km, band_of, centers)
    series = {code: _gauge_series(pkg, g["lag"], g["offset"], k)
              for k, (code, g) in enumerate(GAUGES.items())}
    # lag truths, built by the truth module itself
    stations, curves = [], {"taus": truth.TAUS_A}
    for rec in table:
        st = {"code": rec["code"], "location": rec["location"], "s_g_km": rec["s_g_km"],
              "mouth": False, "windows": {}}
        if not rec["included"]:
            st["windows"][WIN] = {"inclusion": {"included": False, "exclusion": ["coverage"]}}
        else:
            res, cv = truth.lag_truths_window(pkg, rec, series[rec["code"]], WIN,
                                              constituents=CONS, boot=False)
            st["windows"][WIN] = {"inclusion": {"included": True, "exclusion": []}, **res}
            for k, v in cv.items():
                curves[f"{rec['code']}__{WIN}__{k}"] = np.asarray(v)
        stations.append(st)
    tdir = os.path.join(root, "truth", SITE)
    doc = {"format": truth.TRUTH_LAG_V2_FORMAT, "site": SITE, "windows": [WIN],
           "conventions": judges.conventions_doc(), "stations": stations}
    product.write_json(os.path.join(tdir, "truth_lag_v2.json"), doc)
    product.save_npz(os.path.join(tdir, "truth_lag_v2_curves.npz"), curves)
    product.write_json(os.path.join(tdir, "judges.json"),
                       {"format": judges.JUDGES_FORMAT, "site": SITE,
                        "stations": [{k: v for k, v in r.items() if k != "nbhd"}
                                     for r in table]})
    # the plan
    rows = [_plan_row(SITE, WIN, "E0", "E0a E0b L01"), _plan_row(SITE, WIN, "L01v", "L01v"),
            _plan_row(SITE, WIN, "L02", "L02"), _plan_row(SITE, WIN, "L03", "L03"),
            _plan_row(SITE, WIN, "L08", "L08"), _plan_row(SITE, WIN, "L09", "L09 L10"),
            _plan_row(SITE, WIN, "L12", "L12"), _plan_row(SITE, WIN, "L14", "L14 L15"),
            _plan_row(SITE, WIN, "L16", "L16", status="not_applicable",
                      note="NOT AVAILABLE: AMM15 archive starts 2024-08-04"),
            _plan_row(SITE, WIN, "L17", "L17"), _plan_row(SITE, WIN, "L18", "L18",
                                                          status="not_applicable",
                                                          note="EMODnet tiles not fetched"),
            _plan_row(SITE, WIN, "L20", "L20"),
            _plan_row(SITE, "W3", "E0", "E0a E0b L01", variant="boot30"),
            _plan_row(SITE, "W3", "L17", "L17"),
            _plan_row(NOJ, WIN, "E0", "E0a E0b L01"), _plan_row(NOJ, WIN, "L02", "L02"),
            _plan_row(NOJ, WIN, "L17", "L17")]
    _write_plan(root, rows)
    inst = _write_instance(root, SITE, WIN, keep, s_km, band_of, centers,
                           np.arange(pkg.T))
    n = len(keep)
    rng = np.random.default_rng(4)
    tau1 = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])[:K]
    edge = np.zeros(K, bool)
    edge[3] = True
    f1 = lc.broadcast_band(tau1, band_of) + rng.normal(0, 0.2, n).astype(np.float32)
    _write_product(root, SITE, WIN, "L01", inst, kind="field", method="E0",
                   arrays={"field": f1, "band_tau": tau1, "band_n_px": np.bincount(band_of),
                           "band_edge": edge, "band_lo": tau1 - 4, "band_hi": tau1 + 4},
                   interval95={"level": "band", "nominal": "bootstrap30"}, nested=True)
    tau2 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])[:K]
    _write_product(root, SITE, WIN, "L02", inst, kind="band",
                   arrays={"field": lc.broadcast_band(tau2, band_of), "band_tau": tau2,
                           "band_n_px": np.bincount(band_of), "band_edge": np.zeros(K, bool)})
    _write_product(root, SITE, WIN, "L03", inst, status="not_computable", kind="band",
                   stage="min_scenes", reason="fewer than 60 scenes")
    # GM20-like phase field, one period up, with a hole at gatvanborssele
    f8 = (10.0 * s_km + P).astype(np.float32)
    gv = next(r for r in table if r["code"] == "gatvanborssele")
    d = np.hypot(grid["xs"][keep % W] - gv["gx"], grid["ys"][keep // W] - gv["gy"])
    f8[d <= 600.0] = np.nan
    _write_product(root, SITE, WIN, "L08", inst, kind="field", lag_kind="phase",
                   arrays={"field": f8, "iv_lo": f8 - 6.0, "iv_hi": f8 + 6.0},
                   interval95={"level": "pixel", "nominal": "iqr20x1.96/0.674"},
                   variant="")
    tau9 = tau2 + 3.0 - P
    _write_product(root, SITE, WIN, "L09", inst, kind="band", lag_kind="phase",
                   arrays={"field": lc.broadcast_band(tau9, band_of), "band_tau": tau9,
                           "band_n_px": np.bincount(band_of), "band_edge": np.zeros(K, bool)})
    _write_product(root, SITE, WIN, "L10", inst, kind="band",
                   arrays={"field": lc.broadcast_band(tau2 + 1.0, band_of),
                           "band_tau": tau2 + 1.0, "band_n_px": np.bincount(band_of),
                           "band_edge": np.zeros(K, bool)})
    point14 = {}
    for code in GAUGES:
        point14[code] = {"a": 10.0, "ab": 12.0, "b": 11.0, "hwlw": 9.0, "internal_a": 8.0,
                         "internal_b": 7.0, "dist_km": 2.0, "model": "EOT20", "flags": [],
                         "models": {"EOT20": {"a": 10.0, "ab": 12.0, "b": 11.0, "hwlw": 9.0,
                                              "available": True, "reason": None},
                                    "GOT4.10c": {"a": 15.0, "ab": 15.0, "b": 14.0,
                                                 "hwlw": 13.0, "available": True,
                                                 "reason": None},
                                    "FES2022": {"available": False,
                                                "reason": "not installed"}}}
    _write_product(root, SITE, WIN, "L14", inst, kind="point", point=point14)
    _write_product(root, SITE, WIN, "L15", inst, kind="point",
                   point={c: {"a": 12.0, "ab": 13.0, "b": 12.5, "hwlw": 11.0,
                              "flags": ["degenerate"]} for c in GAUGES})
    _write_product(root, SITE, WIN, "L17", inst, kind="point",
                   point={c: {"a": 0.0, "ab": 0.0, "b": 0.0, "hwlw": 0.0} for c in GAUGES})
    _write_product(root, SITE, WIN, "L20", inst, kind="point", point={})
    groups = {"orbit037": {"tau": tau1 + 2.0, "n_scenes": 80, "status": "ok"},
              "orbit137": {"tau": tau1 - 1.0, "n_scenes": 70, "status": "ok"},
              "orbit080": {"tau": np.full(K, np.nan), "n_scenes": 9,
                           "status": "not_computable", "reason": "too few scenes"}}
    _write_product(root, SITE, WIN, "L01v", inst, kind="groups",
                   groups={k: dict(v, tau=np.asarray(v["tau"]).tolist())
                           for k, v in groups.items()})
    # the site without judges: two products for the coherence table
    inst_v = _write_instance(root, NOJ, WIN, keep, s_km, band_of, centers, np.arange(pkg_v.T))
    _write_product(root, NOJ, WIN, "L01", inst_v, kind="band", method="E0",
                   arrays={"field": lc.broadcast_band(tau1, band_of), "band_tau": tau1,
                           "band_n_px": np.bincount(band_of), "band_edge": np.zeros(K, bool)})
    _write_product(root, NOJ, WIN, "L02", inst_v, kind="band",
                   arrays={"field": lc.broadcast_band(tau2, band_of), "band_tau": tau2,
                           "band_n_px": np.bincount(band_of), "band_edge": np.zeros(K, bool)})
    _write_product(root, NOJ, WIN, "L17", inst_v, kind="point", point={})
    # positive-control verdicts (the control driver's format: per-world
    # verdicts and the gating summary; L02 as a hand-written verdict) and gates
    for pid, g in (("L01", "PASS"), ("L03", "PASS"), ("L08", "control mismatch"),
                   ("L01v", "PASS"), ("L12", "PASS"), ("L10", "pending")):
        worlds = {"gating_profile": {"verdict": "PASS" if g != "control mismatch"
                                     else "control mismatch"},
                  "hysteresis_profile": {"verdict": "control mismatch"}}
        if g != "pending":
            worlds["gating_zero"] = {"verdict": "PASS"}
        product.write_json(os.path.join(root, "control", f"{pid}.json"),
                           {"product": pid, "worlds": worlds, "gating": g,
                            "code_files_key": f"ck-{pid}"})
    product.write_json(os.path.join(root, "control", "L02.json"),
                       {"verdict": "PASS", "code_files_key": "ck-L02"})
    from experiments.benchmark import gates
    for g in ("record", "keep", "marea"):
        product.write_json(os.path.join(root, "gates", SITE, f"{g}.json"),
                           {"format": "v2-gate-1", "gate": g, "verdict": "PASS",
                            "code_sha256": gates.gate_code_shas(g)})
    return types.SimpleNamespace(root=root, pkg=pkg, pkg_v=pkg_v, grid=grid, keep=keep,
                                 s_km=s_km, band_of=band_of, centers=centers, table=table,
                                 f8=f8.astype(np.float64),
                                 series=series, doc=product.read_json(
                                     os.path.join(tdir, "truth_lag_v2.json")),
                                 curves=curves, tau1=tau1, tau2=tau2)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    lagtable.USE_CARDS = False
    lagtable._card_info.cache_clear()
    s = build_site(str(tmp_path_factory.mktemp("lagscore")))
    yield s
    lagtable.USE_CARDS = True
    lagtable._card_info.cache_clear()


@pytest.fixture(scope="module")
def scored(site):
    out = scoring.score_lag(site.root, pre_lock=True, B=B,
                            packages={SITE: site.pkg, NOJ: site.pkg_v},
                            truth_docs={NOJ: {"format": truth.TRUTH_LAG_V2_FORMAT,
                                              "stations": []}},
                            judge_tables={NOJ: []}, series_by_code=site.series)
    return out


def _scorer(site, B_=B):
    return scoring.LagScorer(site.pkg, site.doc, site.table, WIN, B=B_,
                             series_by_code=site.series)


# ─────────────────────────────────────────────────────────────────────────────
#  E, E_all, LLE, AGF
# ─────────────────────────────────────────────────────────────────────────────

def test_E_reproduces_the_truth_curve_bitwise(site):
    sc = _scorer(site)
    pos = truth.window_positions(truth.record_t(site.pkg), WIN)
    for code in ("gatvanborssele", "terneuzen"):
        j = sc.judge(code)
        ref = site.curves[f"{code}__{WIN}__E_a"]
        Hw = site.pkg.levels.stack(truth.TAUS_A)[:, pos]
        er = truth.e_rows(Hw, j.g)
        for tau in (-60, -1, 0, 7, 20, 35, 120, 240):
            k = tau - int(truth.TAUS_A[0])
            v = sc.E(code, tau)
            assert np.float64(v).view(np.int64) == np.float64(ref[k]).view(np.int64)
            assert np.float64(v).view(np.int64) == np.float64(er[k]).view(np.int64)
        assert sc.check_truth_curve(code, site.curves) == 0.0
        # a real tau reads the level rows linearly
        h = 0.3 * sc.row(20) + 0.7 * sc.row(21)
        assert np.allclose(sc.h_at(20.7), h, rtol=0, atol=1e-12)
        assert sc.E(code, 20.7) == pytest.approx(float(truth.e_rows(h[None, :], j.g)[0]),
                                                 abs=1e-15)
        # beyond the table: a live provider call, exact
        hb = sc.h_at(260)
        assert np.allclose(hb, site.pkg.tide.value(sc.t.asi8 - 260 * 60_000_000_000),
                           atol=1e-12)
        assert np.isfinite(sc.E(code, 260.5))


def test_E_all_reproduces_tau_all_window(site):
    sc = _scorer(site)
    for code in ("gatvanborssele", "terneuzen.westsluis.buiten"):
        ref = site.curves[f"{code}__{WIN}__E_all"]
        for tau in (-60, 0, 19, 37, 240):
            k = tau - int(truth.TAUS_A[0])
            assert np.float64(sc.E_all(code, tau)).view(np.int64) == \
                np.float64(ref[k]).view(np.int64)
        assert sc.E_all_min(code) == float(np.nanmin(ref))
        lo, hi = sc.E_all(code, 19), sc.E_all(code, 20)
        mid = sc.E_all(code, 19.5)
        assert np.isfinite(mid) and mid <= max(lo, hi) + 0.02
        # at a non-integer tau, linear in minutes between the 1-min samples
        j = sc.judge(code)
        q0 = j.p10 - 19
        hb = 0.5 * j.h1[q0] + 0.5 * j.h1[q0 - 1]
        m = np.isfinite(j.g10)
        dd = hb[m] - j.g10[m]
        assert mid == pytest.approx(float(np.sqrt(np.mean((dd - np.median(dd)) ** 2))),
                                    abs=1e-14)


def test_level_metrics_at_tau_a_and_zero(site):
    sc = _scorer(site)
    for code in ("gatvanborssele", "terneuzen"):
        tau_a = sc.entry(code)["a"]["tau"]
        assert abs(tau_a - GAUGES[code]["lag"]) <= 2
        lm = sc.level_metrics(code, tau_a)
        assert lm["dE"] == 0.0 and lm["LLE"] == 0.0
        assert lm["AGF"] == 1.0 and lm["material"]
        assert lm["explained"] == pytest.approx(1 - (lm["Emin"] / lm["E0"]) ** 2)
        lz = sc.level_metrics(code, 0.0)
        assert lz["AGF"] == 0.0 and lz["dE"] == pytest.approx(lz["E0"] - lz["Emin"])
        assert lz["explained"] == 0.0
        assert lz["LLE"] > 0.05            # a 20-min error at a 1.6 m M2: ~ 25 cm
        # dE_all against the best integer lag of E_all
        assert lm["dE_all"] >= -1e-12


def test_lag_not_material_below_2_cm(site):
    pkg = site.pkg
    rec = dict(site.table[0], code="nearmouth")
    g = _gauge_series(pkg, 1, 0.0, 99)
    res, _ = truth.lag_truths_window(pkg, rec, g, WIN, constituents=CONS, boot=False)
    doc = {"stations": [{"code": "nearmouth", "location": "nearmouth",
                         "windows": {WIN: {"inclusion": {"included": True}, **res}}}]}
    sc = scoring.LagScorer(pkg, doc, [rec], WIN, B=B, series_by_code={"nearmouth": g},
                           site=SITE)
    lm = sc.level_metrics("nearmouth", 5.0)
    assert lm["E0"] - lm["Emin"] < 0.02 and not lm["material"] and np.isnan(lm["AGF"])
    r = sc.score("nearmouth", 5.0)
    assert "lag not material" in r["a"]["flags"] and np.isnan(r["a"]["AGF"])
    assert np.isfinite(r["a"]["dE_cm"])


def test_err_int_interval_score_and_coverage(site):
    sc = _scorer(site)
    code = "terneuzen"
    lo, hi = sc.entry(code)["a"]["lo"], sc.entry(code)["a"]["hi"]
    tau_a = sc.entry(code)["a"]["tau"]
    r = sc.score(code, hi + 5.0, interval=(tau_a - 3.0, tau_a + 2.0))
    assert r["a"]["err_int"] == pytest.approx(5.0)
    assert r["a"]["e"] == pytest.approx(hi + 5.0 - tau_a)
    assert r["a"]["IS"] == pytest.approx(5.0) and r["a"]["covered"] == 1.0
    r = sc.score(code, lo + 0.5)
    assert r["a"]["err_int"] == 0.0 and r["a"]["hitI"] == 1.0
    assert np.isnan(r["a"]["IS"]) and np.isnan(r["hwlw"]["err_int"])
    # the interval score formula
    assert scoring.interval_score((10.0, 20.0), 25.0) == (10.0 + 40.0 * 5.0, 0.0)
    assert scoring.interval_score((10.0, 20.0), 7.0) == (10.0 + 40.0 * 3.0, 0.0)
    assert scoring.interval_score((10.0, 20.0), 12.0) == (10.0, 1.0)
    assert all(np.isnan(scoring.interval_score((np.nan, 20.0), 12.0)))
    # phase: the truth is moved to the interval's branch
    assert scoring.interval_score((P - 3.0, P + 3.0), 1.0, "phase") == (6.0, 1.0)
    # a phase product's error is circular
    rp = sc.score(code, tau_a + 4.0 + P, "phase")
    assert rp["a"]["e"] == pytest.approx(4.0)
    assert rp["a"]["dE_cm"] > 0 and np.isfinite(rp["a"]["LLE_cm"])


def test_bootstrap_is_paired_and_deterministic(site):
    sc = _scorer(site)
    code = "gatvanborssele"
    W1 = sc.weights(code)
    assert W1 is sc.weights(code) and W1.shape == (B, len(sc.t))
    j = sc.judge(code)
    assert (W1[:, ~j.m] == 0).all()
    d10 = sc.boot_draws(code, 10.0)
    d25 = sc.boot_draws(code, 25.0)
    # paired: the same tau_a* in every draw, so the e* differ by the lag difference
    assert np.allclose(d25["e_a"] - d10["e_a"], 15.0)
    assert np.allclose(d10["e_a"], 10.0 - sc.boot(code).tau_a)
    sc2 = _scorer(site)
    assert np.array_equal(sc2.weights(code), W1)
    assert np.array_equal(sc2.boot_draws(code, 10.0)["dE"], d10["dE"])
    # the seed is the truth's window seed with a trailing 1
    assert sc.boot(code).seed == [*truth.window_boot_seed(SITE, code, WIN), 1]
    # the weighted statistics: E0*, Emin* and tau_a* from the weighted E
    bs = sc.boot(code)
    assert (bs.Emin <= bs.E0 + 1e-15).all()
    assert (np.abs(bs.tau_a - GAUGES[code]["lag"]) <= 6).all()
    assert (d10["dE"] >= -1e-12).all()
    r = sc.score(code, 10.0)
    assert np.isfinite(r["a"]["e_lo"]) and r["a"]["e_lo"] <= r["a"]["e_hi"]
    lo, hi = np.percentile(10.0 - bs.tau_a, [2.5, 97.5])
    assert (r["a"]["e_lo"], r["a"]["e_hi"]) == (pytest.approx(lo), pytest.approx(hi))
    assert np.isnan(r["b"]["e_lo"])               # no bootstrap of tau_b in V2


# ─────────────────────────────────────────────────────────────────────────────
#  end to end
# ─────────────────────────────────────────────────────────────────────────────

def _pts(scored):
    return scored["data"]["points"]


def _row(pts, product_, judge, truth_="a", window=WIN, site=SITE):
    r = pts[(pts["site"] == site) & (pts["window"] == window) & (pts["product"] == product_)
            & (pts["judge"] == judge) & (pts["truth"] == truth_)]
    assert len(r) == 1, (product_, judge, truth_, len(r))
    return r.iloc[0]


def test_reads_products_written_by_the_queue(tmp_path):
    """Products published by the run queue's own writer read back as the
    scorer expects (field, band with its interval, point, groups, instance)."""
    q = pytest.importorskip("experiments.benchmark.queue_v2")
    if not all(hasattr(q, n) for n in ("result_arrays", "publish", "load_lag_product",
                                        "PRODUCT_FORMAT")):
        pytest.skip("the queue's writer is not built")
    keep, s_km, band_of, centers = _domain()
    K, n = len(centers), len(keep)
    root = str(tmp_path)
    inst = _write_instance(root, SITE, WIN, keep, s_km, band_of, centers, np.arange(10))
    tau = np.linspace(5, 55, K)
    results = [
        lc.lag_result("L01", "ok", kind="field", n_keep=n, K=K,
                      field=lc.broadcast_band(tau, band_of),
                      band={"tau": tau, "n_px": np.bincount(band_of), "edge": np.zeros(K, bool),
                            "lo": tau - 4, "hi": tau + 4},
                      interval95={"level": "band", "lo": tau - 4, "hi": tau + 4,
                                  "nominal": "bootstrap30"}),
        lc.lag_result("L14", "ok", kind="point",
                      point={"g1": {"a": 10.0, "ab": 11.0, "b": 12.0, "hwlw": 9.0,
                                    "model": "EOT20", "flags": ["regional unresolved"]}}),
        lc.lag_result("L01v", "ok", kind="groups", K=K,
                      groups={"orbit037": {"tau": tau + 1, "n_scenes": 50, "status": "ok"}}),
        lc.lag_result("L03", "not_computable", stage="min_scenes", reason="< 60 scenes")]
    for res in results:
        arrays, outcome, diag = q.result_arrays(res)
        d = scoring.product_dir(root, SITE, WIN, P0, res["product"])
        run = {"format": q.PRODUCT_FORMAT, "product": res["product"], "method": res["product"],
               "variant": "", "designs": ["lag"], "unit_id": "u", "site": SITE, "window": WIN,
               "setting": P0, "family": "rand", "mode": "A", "N": "all", "rep": 0, **outcome,
               "instance": {"path": os.path.relpath(inst, d).replace("\\", "/"),
                            "instance_sha256": "x"},
               "timings": {}, "unit_total": {"wall_s": 1.0}, "diag": diag,
               "arrays": sorted(arrays)}
        q.publish(d, arrays, run)
    store = scoring._Store(root)
    p1 = store.product(scoring.product_dir(root, SITE, WIN, P0, "L01"))
    assert p1["status"] == "ok" and p1["kind"] == "field"
    assert np.allclose(p1["field"], lc.broadcast_band(tau, band_of))
    assert np.allclose(p1["band"]["tau"], tau) and np.allclose(p1["band"]["lo"], tau - 4)
    assert p1["interval95"]["level"] == "band" and np.allclose(p1["interval95"]["hi"], tau + 4)
    ins = store.instance(p1)
    assert np.array_equal(ins["keep"], keep) and np.array_equal(ins["band_of"], band_of)
    assert np.allclose(ins["centers"], centers) and ins["K"] == K
    p14 = store.product(scoring.product_dir(root, SITE, WIN, P0, "L14"))
    assert p14["point"]["g1"]["b"] == 12.0 and "regional unresolved" in \
        p14["point"]["g1"]["flags"]
    pv = store.product(scoring.product_dir(root, SITE, WIN, P0, "L01v"))
    assert np.allclose(pv["groups"]["orbit037"]["tau"], tau + 1)
    p3 = store.product(scoring.product_dir(root, SITE, WIN, P0, "L03"))
    assert p3["status"] == "not_computable" and p3["stage"] == "min_scenes"


def test_a_judged_site_without_lag_truth(site):
    out = scoring.score_lag(site.root, pre_lock=True, B=4, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, truth_docs={SITE: None},
                            series_by_code=site.series, write=False)
    pts = out["data"]["points"]
    assert len(pts) and pts["status"].str.startswith("NOT AVAILABLE (no lag truth").all()
    assert set(pts["judge"]) == set(GAUGES)
    assert out["data"]["family"].empty


def test_failures_become_rows_not_crashes(site, monkeypatch):
    from experiments.benchmark import readout as rd

    table = [dict(r, file="no_such_gauge_file.csv") if r["code"] == "gatvanborssele" else r
             for r in site.table]
    series = {k: v for k, v in site.series.items() if k != "gatvanborssele"}

    def broken(*a, **kw):
        raise ValueError("misaligned field")

    monkeypatch.setattr(rd, "readout", broken)
    out = scoring.score_lag(site.root, pre_lock=True, B=4, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, judge_tables={SITE: table},
                            series_by_code=series, write=False)
    pts = out["data"]["points"]
    gv = pts[pts["judge"] == "gatvanborssele"]
    assert gv["status"].str.startswith("NOT AVAILABLE (gauge: FileNotFoundError").all()
    r = _row(pts, "L02", "terneuzen")
    assert r["status"] == "NOT AVAILABLE (scoring error: ValueError: misaligned field)"
    assert _row(pts, "L17", "terneuzen")["status"] == "ok"       # point products unaffected
    assert any("misaligned field" in w for w in out["warnings"])
    assert any("gatvanborssele" in w and "gauge" in w for w in out["warnings"])


def test_score_refuses_without_lock_phase_a(site):
    with pytest.raises(RuntimeError, match="LOCK"):
        scoring.score_lag(site.root, pre_lock=False, B=B, packages={SITE: site.pkg},
                          series_by_code=site.series, write=False)
    assert scoring.lock_phase_a({"phase": "A"}) and not scoring.lock_phase_a({})


def test_files_columns_and_banner(scored, site):
    for k in ("points", "summary", "family", "repro", "cost", "table_md"):
        assert os.path.exists(scored[k]), k
    with open(scored["points"], encoding="utf-8") as f:
        assert f.readline().strip() == f"# {scoring.SCORING_V2['banner_pre_lock']}"
    pts = lagtable.read_csv(scored["points"])
    assert tuple(pts.columns) == lagtable.POINTS_COLUMNS
    summ = lagtable.read_csv(scored["summary"])
    assert tuple(summ.columns) == lagtable.SUMMARY_COLUMNS
    md = open(scored["table_md"], encoding="utf-8").read()
    assert md.count("PRE-LOCK, descriptive; not the pre-registered analysis") >= 2
    meta = json.load(open(os.path.join(os.path.dirname(scored["points"]), "lag_meta.json"),
                          encoding="utf-8"))
    assert meta["pre_lock"] is True and meta["banner"]


def test_window_year_guard_and_exclusions(scored):
    pts = _pts(scored)
    w3 = pts[(pts["site"] == SITE) & (pts["window"] == "W3")]
    assert len(w3) and (w3["status"] == "NOT AVAILABLE: store years [2023, 2025]").all()
    assert set(w3["product"]) == {"L01", "L17"}
    ex = pts[(pts["judge"] == "borssele") & (pts["window"] == WIN)]
    assert len(ex) and (ex["status"] == "judge excluded (coverage)").all()


def test_status_texts_of_the_plan(scored):
    pts = _pts(scored)
    j = "gatvanborssele"
    assert _row(pts, "L03", j)["status"] == "NOT COMPUTABLE (min_scenes)"
    assert _row(pts, "L12", j)["status"] == "NOT AVAILABLE (not run)"
    assert _row(pts, "L16", j)["status"] == "NOT AVAILABLE (AMM15 archive starts 2024-08-04)"
    assert _row(pts, "L18", j)["status"] == "not applicable (EMODnet tiles not fetched)"
    r9 = _row(pts, "L09", j)
    assert r9["status"] == "ok" and not r9["validated"]
    assert _row(pts, "L01", j)["validated"] and _row(pts, "L02", j)["validated"]
    r8 = _row(pts, "L08", j)
    assert r8["validated"] and "control mismatch" in r8["flags"]
    # a gating world still pending: not validated
    assert not _row(pts, "L10", j)["validated"]
    # model rows have no imagery control
    assert _row(pts, "L17", j)["validated"]
    md = open(scored["table_md"], encoding="utf-8").read()
    for text in ("NOT COMPUTABLE (min_scenes)", "not applicable (EMODnet tiles not fetched)",
                 "NOT AVAILABLE (AMM15 archive starts 2024-08-04)",
                 "implementation not validated", "NOT AVAILABLE: store years [2023, 2025]",
                 "judge excluded (coverage)"):
        assert text in md, text
    assert f"#### {SITE} · {WIN} · gatvanborssele" in md
    assert f"#### {SITE} · {WIN} · terneuzen.westsluis.buiten" in md


def test_readouts_end_to_end(scored, site):
    pts = _pts(scored)
    r = _row(pts, "L01", "gatvanborssele")
    assert r["readout_step"] == 1 and r["n_readout_px"] >= 30
    assert abs(r["tau_m"] - 15.0) < 0.5 and r["tau_m_band"] == pytest.approx(15.0, abs=0.5)
    assert r["e"] == pytest.approx(r["tau_m"] - r["tau_true"])
    assert r["E0_cm"] > r["Etau_cm"] > 0 and r["dE_cm"] >= 0
    assert r["IS"] > 0 and r["covered"] in (0.0, 1.0)
    assert r["B"] == B and r["n_instants"] == site.pkg.T
    # a judge flagged sparse in the judge table carries the flag on every row
    assert "judge sparse" in _row(pts, "L01", "terneuzen.westsluis.buiten")["flags"]
    assert "judge sparse" in _row(pts, "L17", "terneuzen.westsluis.buiten")["flags"]
    assert "judge sparse" not in r["flags"]
    # terneuzen sits in the edge band of L01
    rt = _row(pts, "L01", "terneuzen")
    assert bool(rt["edge"]) and "edge" in rt["flags"]
    # GM20 with a hole at the gauge: step 2, a circular read-out one period down
    r8 = _row(pts, "L08", "gatvanborssele")
    assert r8["readout_step"] == 2 and "step2" in r8["flags"] and np.isnan(r8["tau_m_step1"])
    assert -50 < r8["tau_m"] < 100
    f8 = site.f8
    band1 = (site.band_of == 1) & np.isfinite(f8)
    assert r8["tau_m"] == pytest.approx(float(np.median(lc.wrap_phase(f8[band1]))), abs=1e-9)
    assert np.isfinite(r8["IS"])
    # HD26-like phase band: wrapped back by one period
    assert _row(pts, "L09", "gatvanborssele")["tau_m"] == pytest.approx(13.0)
    # the same read-out for every truth of a field product; e per truth
    rb = _row(pts, "L01", "gatvanborssele", "b")
    assert rb["tau_m"] == r["tau_m"] and rb["dE_cm"] == r["dE_cm"]
    assert rb["e"] == pytest.approx(r["tau_m"] - rb["tau_true"])


def test_point_products_and_descriptive_rows(scored):
    pts = _pts(scored)
    j = "terneuzen"
    assert _row(pts, "L14", j, "a")["tau_m"] == 10.0
    assert _row(pts, "L14", j, "b")["tau_m"] == 11.0
    assert _row(pts, "L14", j, "ab")["tau_m"] == 12.0
    assert _row(pts, "L14", j, "hwlw")["tau_m"] == 9.0
    assert _row(pts, "L14@GOT4.10c", j, "a")["tau_m"] == 15.0
    assert _row(pts, "L14@FES2022", j)["status"] == "NOT AVAILABLE (not installed)"
    ri = _row(pts, "L14@internal", j, "a")
    assert ri["tau_m"] == 8.0 and _row(pts, "L14@internal", j, "b")["tau_m"] == 7.0
    assert np.isnan(_row(pts, "L14@internal", j, "ab")["tau_m"])
    assert "descriptive" in ri["flags"]
    assert "degenerate" in _row(pts, "L15", j)["flags"]
    assert _row(pts, "L17", j)["tau_m"] == 0.0
    r17 = _row(pts, "L17", j)
    assert r17["AGF"] == 0.0 and r17["readout_step"] == "point"
    # per-orbit rows of L01v
    ro = _row(pts, "L01v@orbit037", "gatvanborssele")
    assert ro["tau_m"] == pytest.approx(17.0)
    assert _row(pts, "L01v@orbit080", "gatvanborssele")["status"].startswith("NOT COMPUTABLE")


def test_l20_leave_one_out(scored, site):
    pts = _pts(scored)
    doc = site.doc
    ent = {st["code"]: st["windows"][WIN] for st in doc["stations"]}
    s_tz = np.mean([r["s_g_km"] for r in site.table if r["location"] == "terneuzen"])
    s_gv = [r["s_g_km"] for r in site.table if r["code"] == "gatvanborssele"][0]
    tz_a = np.mean([ent[c]["a"]["tau"] for c in ("terneuzen", "terneuzen.westsluis.buiten")])
    # gatvanborssele: the anchor (0, 0) and the Terneuzen location
    want = np.interp(s_gv, [0.0, s_tz], [0.0, tz_a])
    assert _row(pts, "L20", "gatvanborssele")["tau_m"] == pytest.approx(want)
    # terneuzen: beyond the last other location the value is held constant
    want_tz = ent["gatvanborssele"]["a"]["tau"] if s_tz > s_gv else np.interp(
        s_tz, [0.0, s_gv], [0.0, ent["gatvanborssele"]["a"]["tau"]])
    for j in ("terneuzen", "terneuzen.westsluis.buiten"):
        r = _row(pts, "L20", j)
        assert r["tau_m"] == pytest.approx(want_tz)
        assert "reference, never ranked" in r["flags"]


def test_summary_terneuzen_pair_is_one_location(scored):
    pts = _pts(scored)
    summ = scored["data"]["summary"]
    s = summ[(summ["site"] == SITE) & (summ["window"] == WIN) & (summ["product"] == "L01")
             & (summ["truth"] == "a") & (summ["judge_set"] == "headline")].iloc[0]
    assert s["n_locations"] == 2
    e_gv = abs(_row(pts, "L01", "gatvanborssele")["e"])
    e_tz = np.mean([abs(_row(pts, "L01", j)["e"]) for j in
                    ("terneuzen", "terneuzen.westsluis.buiten")])
    assert s["MAE"] == pytest.approx(np.mean([e_gv, e_tz]))
    assert s["max_abs_e"] == pytest.approx(max(e_gv, e_tz))
    assert s["MAE_lo"] <= s["MAE_hi"] and np.isfinite(s["rms_dE_lo_cm"])
    assert str(s["label"]).startswith("PRE-LOCK")
    # step1_only drops the step-2 read-out of GM20 at gatvanborssele
    s8 = summ[(summ["site"] == SITE) & (summ["product"] == "L08") & (summ["truth"] == "a")]
    assert s8[s8["judge_set"] == "headline"].iloc[0]["n_locations"] == 2
    assert s8[s8["judge_set"] == "step1_only"].iloc[0]["n_locations"] == 1
    # every judge set is there; status rows for products that are not ok
    sets = set(summ.loc[(summ["site"] == SITE) & (summ["product"] == "L01"), "judge_set"])
    assert {"headline", "without_mouth", "development_free", "step1_only"} <= sets
    s3 = summ[(summ["product"] == "L03") & (summ["site"] == SITE)].iloc[0]
    assert s3["status"] == "NOT COMPUTABLE (min_scenes)" and s3["n_locations"] == 0
    # pooled rows (one judged site here): the site values
    pooled = summ[(summ["site"] == "pooled") & (summ["window"] == WIN) &
                  (summ["product"] == "L01") & (summ["truth"] == "a") &
                  (summ["judge_set"] == "headline")]
    assert len(pooled) == 1 and pooled.iloc[0]["MAE"] == pytest.approx(s["MAE"])
    assert len(summ[(summ["site"] == "pooled") & (summ["window"] == "mixed")])


def test_family_is_descriptive_before_the_lock(scored):
    fam = scored["data"]["family"]
    assert tuple(fam.columns) == lagtable.FAMILY_COLUMNS
    main = fam[~fam["label"].str.contains("pseudo-replicated")]
    assert set(main["truth"]) == {"a", "ab", "b"}
    assert set(main["comparison"]) == {"L17", "L14", "L08", "L09"}     # no L16: not available
    assert main["reject"].isna().all() and main["confirmatory"].isna().all()
    assert main["label"].str.contains("descriptive, mixed windows").all()
    assert main["label"].str.startswith("PRE-LOCK").all()
    assert main["label"].str.contains("not validated: L09").all()
    r = main[(main["truth"] == "a") & (main["comparison"] == "L17")].iloc[0]
    assert r["n"] == 2 and r["p_min"] == pytest.approx(0.5)
    pooled = fam[fam["label"].str.contains("pseudo-replicated")]
    assert len(pooled) == 12
    assert scored["data"]["context"]["lp1"]["confirmatory"] is False


def test_repro_coherence_and_cost(scored, site):
    rp = scored["data"]["repro"]
    assert tuple(rp.columns) == lagtable.REPRO_COLUMNS
    orb = rp[(rp["kind"] == "orbits") & (rp["site"] == SITE)]
    allr = orb[orb["band"].astype(str) == "all"].iloc[0]
    assert allr["n_groups"] == 2 and allr["median_abs_diff"] == pytest.approx(3.0)
    assert np.isfinite(allr["icc21"]) and allr["icc21"] > 0.9
    coh = rp[(rp["kind"] == "coherence") & (rp["site"] == NOJ)]
    assert len(coh) == len(site.centers)
    assert np.allclose(coh["spread_range"].to_numpy(float), 5.0)
    assert np.allclose(coh["spread_sd"].to_numpy(float), 2.5)
    assert (coh["label"] == "lag coherence only").all()
    md = open(scored["table_md"], encoding="utf-8").read()
    assert "lag coherence only" in md
    cost = scored["data"]["cost"]
    c1 = cost.set_index("product").loc["L01"]
    assert c1["wall_s_median"] == 2.0 and c1["label"].startswith("contended")
    assert cost.set_index("product").loc["L03", "share_not_computable"] == 1.0


def test_show_unvalidated_prints_numbers(site, tmp_path):
    out = scoring.score_lag(site.root, pre_lock=True, B=8, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, series_by_code=site.series,
                            out_dir=str(tmp_path), show_unvalidated=True)
    md = open(out["table_md"], encoding="utf-8").read()
    assert "not validated (shown)" in md


# ─────────────────────────────────────────────────────────────────────────────
#  profile metrics, control verdicts, G-lagtruth
# ─────────────────────────────────────────────────────────────────────────────

def test_profile_metrics_on_four_locations():
    tau_t = np.array([10.0, 20.0, 40.0, 60.0])
    s_g = np.array([1.0, 5.0, 12.0, 20.0])
    tau_m = np.array([12.0, 21.0, 38.0, 65.0])
    pm = scoring.profile_metrics(tau_m, tau_t, s_g)
    slope = np.polyfit(tau_t, tau_m, 1)[0]
    assert pm["slope"] == pytest.approx(slope)
    e = tau_m - tau_t
    assert pm["shape_err"] == pytest.approx(np.sqrt(np.mean((e - e.mean()) ** 2)))
    assert pm["mono_violations"] == 0
    assert 0.9 < pm["ccc"] <= 1.0
    pm = scoring.profile_metrics(np.array([12.0, 30.0, 25.0, 20.0]), tau_t, s_g)
    assert pm["mono_violations"] == 2
    # s order, not input order
    pm = scoring.profile_metrics(tau_m[::-1], tau_t[::-1], s_g[::-1])
    assert pm["mono_violations"] == 0


def test_control_verdicts():
    band_of = np.repeat(np.arange(4), 50)
    planted = {"bands": [0, 1, 2, 3], "tau_mean_min": [0.0, 30.0, 60.0, 90.0],
               "n_px": [50] * 4}
    ok_band = {"product": "L02", "status": "ok", "kind": "band", "lag_kind": "linear",
               "band": {"tau": np.array([2.0, 33.0, 58.0, 95.0])}}
    off_field = {"product": "L12", "status": "ok", "kind": "field", "lag_kind": "linear",
                 "field": np.repeat(np.array([20.0, 50.0, 80.0, 110.0]), 50)}
    phase = {"product": "L08", "status": "ok", "kind": "field", "lag_kind": "phase",
             "field": np.repeat(np.array([P + 1.0, 31.0 - P, 60.0, 89.0]), 50)}
    nc = {"product": "L07", "status": "not_computable", "kind": "band", "stage": "coverage",
          "reason": "< 30 scenes"}
    out = scoring.control_verdicts([ok_band, off_field, phase, nc], planted, band_of=band_of,
                                   offsets={"L02": 0.0, "L12": 0.0, "L08": 0.0, "L07": 0.0})
    # |err| = 2, 3, 2, 5 min: the median is 2.5
    assert out["L02"]["verdict"] == "PASS" and out["L02"]["median_abs_err_min"] == 2.5
    assert out["L02"]["n_bands"] == 4 and len(out["L02"]["per_band"]) == 4
    assert out["L12"]["verdict"] == "control mismatch"
    assert out["L12"]["median_abs_err_min"] == pytest.approx(20.0)
    assert out["L08"]["verdict"] == "PASS" and out["L08"]["median_abs_err_min"] == \
        pytest.approx(1.0)
    assert out["L07"]["verdict"] == "control mismatch" and "not_computable" in \
        out["L07"]["reason"]
    # the theory offset of the card is removed before the comparison
    out = scoring.control_verdicts({"L12": off_field}, planted, band_of=band_of,
                                   offsets={"L12": 20.0})
    assert out["L12"]["verdict"] == "PASS" and out["L12"]["median_abs_err_min"] == 0.0
    # a documented fault is set by hand only
    out = scoring.control_verdicts({"L02": ok_band}, planted,
                                   manual_fault={"L02": "sign error in the scan"},
                                   offsets={"L02": 0.0})
    assert out["L02"]["verdict"] == "FAIL (fault)"
    with pytest.raises(ValueError):
        scoring.control_verdicts({"L12": off_field}, planted, offsets={"L12": 0.0})
    # both scenarios must pass
    prof = scoring.control_verdicts({"L02": ok_band, "L12": off_field}, planted,
                                    band_of=band_of, offsets={"L02": 0.0, "L12": 20.0})
    zero = scoring.control_verdicts(
        {"L02": dict(ok_band, band={"tau": np.array([1.0, 0.0, -2.0, 30.0])}),
         "L12": dict(off_field, field=np.full(200, 20.0))},
        dict(planted, tau_mean_min=[0.0] * 4), band_of=band_of,
        offsets={"L02": 0.0, "L12": 20.0})
    ov = scoring.control_overall({"profile": prof, "zero": zero})
    assert ov["L12"]["verdict"] == "PASS"
    assert zero["L02"]["verdict"] == "PASS"          # median |err| of (1, 0, 2, 30) = 1.5
    assert ov["L02"]["verdict"] == "PASS"
    ov = scoring.control_overall({"profile": prof})
    assert ov["L12"]["verdict"] == "control mismatch"   # the zero scenario is missing
    assert scoring.control_verdict_of({"scenarios": {"profile": {"verdict": "PASS"},
                                                     "zero": {"verdict": "PASS"}}}) == "PASS"


def test_reading_control_files_and_the_lock():
    cv = scoring.control_verdict_of
    driver = {"product": "L08", "gating": "PASS",
              "worlds": {"gating_profile": {"verdict": "PASS"},
                         "gating_zero": {"verdict": "PASS"},
                         "mnar_profile": {"verdict": "control mismatch"}}}
    assert cv(driver) == "PASS"                    # misspecified worlds never gate
    assert cv(dict(driver, gating="pending")) is None
    assert cv(dict(driver, gating="control mismatch")) == "control mismatch"
    assert cv(dict(driver, fault="sign error in the fold")) == "FAIL (fault)"
    w = dict(driver["worlds"], gating_zero={"verdict": "FAIL (fault)"})
    assert cv(dict(driver, worlds=w)) == "FAIL (fault)"
    assert cv(None) is None and cv({}) is None
    # LOCK.json as lock.write_lock writes it
    assert scoring.lock_phase_a({"phases": {"A": {"utc": "x"}}})
    assert not scoring.lock_phase_a({"phases": {}})
    assert not scoring.lock_phase_a({"phases": {"A": {}}})
    assert scoring.lock_development({"development_judges": ["harlingen.waddenzee", "delfzijl"],
                                     "phases": {}}) == ("harlingen.waddenzee", "delfzijl")


def test_gate_lagtruth_on_a_synthetic_level_table(tmp_path):
    out = scoring.gate_lagtruth("wadden", root=str(tmp_path), synthetic=True,
                                constituents=CONS)
    assert out["format"] == "v2-gate-1" and out["gate"] == "lagtruth"
    assert out["verdict"] == "PASS", out
    c = out["compared"]
    assert abs(c["tau_a"] - 75) <= 1 and abs(c["tau_b"] - 75) <= 1
    assert c["source"].startswith("synthetic") and c["window"] == "Y2024"
    assert "equal = True" in c["nearest_cell"]
    doc = product.read_json(os.path.join(tmp_path, "gates", "wadden", "lagtruth.json"))
    assert doc["verdict"] == "PASS" and doc["first_difference"] is None


def test_gate_lagtruth_fails_and_refuses(tmp_path, monkeypatch):
    pkg = scoring.synthetic_package("wadden")
    # a window whose years the package lacks is not run (window-year guard)
    out = scoring.gate_lagtruth("wadden", root=str(tmp_path), pkg=pkg, constituents=CONS,
                                write=False, window="W3")
    assert out["verdict"] == "NOT RUN" and "store years" in out["reason"]
    # a truth machinery that misses the planted lag fails, with the first difference
    real = truth.lag_truths_window

    def shifted(*a, **kw):
        r, cv = real(*a, **kw)
        r["a"] = dict(r["a"], tau=r["a"]["tau"] - 15.0)
        return r, cv

    monkeypatch.setattr(truth, "lag_truths_window", shifted)
    out = scoring.gate_lagtruth("wadden", root=str(tmp_path), pkg=pkg, constituents=CONS,
                                write=False)
    assert out["verdict"] == "FAIL" and out["first_difference"].startswith("tau_a = 60")


# ─────────────────────────────────────────────────────────────────────────────
#  the review fixes of 2026-10-20
# ─────────────────────────────────────────────────────────────────────────────

def test_the_printing_rule_needs_a_current_control_and_current_gates(site, tmp_path):
    from experiments.benchmark import gates
    ok, flags, text = scoring.validation(site.root, SITE, "L02",
                                         {"keys": {"code_files_key": "ck-L02"}})
    assert ok and text is None
    ok, flags, text = scoring.validation(site.root, SITE, "L02",
                                         {"keys": {"code_files_key": "edited-adapter"}})
    assert not ok and "control stale: the control ran other adapter code" in text
    ok, _, text = scoring.validation(site.root, SITE, "L02", {"keys": {}})
    assert not ok and "records no code key" in text
    # a gate verdict made with other sources (or without hashes) is stale
    root = str(tmp_path)
    product.write_json(os.path.join(root, "control", "L02.json"),
                       {"verdict": "PASS", "code_files_key": "k"})
    for g in ("record", "keep"):
        product.write_json(os.path.join(root, "gates", SITE, f"{g}.json"),
                           {"verdict": "PASS", "code_sha256": gates.gate_code_shas(g)})
    assert scoring.validation(root, SITE, "L02", {"keys": {"code_files_key": "k"}})[0]
    product.write_json(os.path.join(root, "gates", SITE, "keep.json"), {"verdict": "PASS"})
    ok, flags, text = scoring.validation(root, SITE, "L02", {"keys": {"code_files_key": "k"}})
    assert not ok and "G-keep stale" in text and "G-keep stale" in flags
    old = dict(gates.gate_code_shas("keep"))
    old[next(iter(old))] = "0" * 64
    product.write_json(os.path.join(root, "gates", SITE, "keep.json"),
                       {"verdict": "PASS", "code_sha256": old})
    assert "code changed since the verdict" in scoring.validation(
        root, SITE, "L02", {"keys": {"code_files_key": "k"}})[2]


def _readout_world():
    n, K = 300, 4
    s_km = np.linspace(0.2, 11.8, n)
    band_of = np.minimum((s_km // 3.0).astype(np.int64), K - 1)
    centers = np.array([np.mean(s_km[band_of == k]) for k in range(K)])
    inst = {"keep": np.arange(n, dtype=np.int64), "s_km": s_km.astype(np.float32),
            "band_of": band_of, "centers": centers}
    return inst, n, K


def test_the_l01_interval_is_the_bootstrap_of_the_read_out():
    from experiments.benchmark.lags import marea_family as mf
    inst, n, K = _readout_world()
    rng = np.random.default_rng(3)
    boot = np.array([10.0, 20.0, 30.0, 40.0])[None, :] + rng.normal(0, 4.0, (30, K))
    tau_used, _ = mf.causality(np.array([10.0, 20.0, 30.0, 40.0]))
    field = mf.applied_lag(inst["centers"], tau_used, inst["s_km"].astype(float))
    pos = np.arange(120, 160)
    ro = {"step": 1, "positions": pos, "tau": float(np.median(field[pos]))}
    prod = {"product": "L01", "dir": "x", "diag": {"boot": boot},
            "interval95": {"level": "band", "lo": np.zeros(K), "hi": np.zeros(K)}}
    lo, hi = scoring.interval_at_readout(prod, ro, "linear", inst, None)
    reps = []
    for b in range(30):
        f = mf.applied_lag(inst["centers"], mf.causality(boot[b])[0],
                           inst["s_km"].astype(float))
        reps.append(np.median(f[pos]))
    assert (lo, hi) == pytest.approx(tuple(np.percentile(reps, [2.5, 97.5])))
    assert lo < ro["tau"] < hi
    nan2 = scoring.interval_at_readout(prod, dict(ro, step=3), "linear", inst)
    assert all(np.isnan(v) for v in nan2)


def test_the_l08_interval_is_formed_at_the_read_out_with_the_subsample_factor():
    inst, n, K = _readout_world()
    rng = np.random.default_rng(5)
    field = 40.0 + rng.normal(0, 1.0, n)
    refit = field[None, :] + rng.normal(0, 3.0, (20, n))
    pos = np.arange(10, 60)
    tau = float(lc.circ_median(field[pos]))
    ro = {"step": 1, "positions": pos, "tau": tau}
    prod = {"product": "L08", "diag": {"refit_tau": refit.astype(np.float32),
                                       "refit_subset_size": 70, "T": 100}}
    lo, hi = scoring.interval_at_readout(prod, ro, "phase", inst)
    r32 = refit.astype(np.float32)
    reps = np.array([lc.circ_median(r32[k, pos]) for k in range(20)])
    q25, q75 = np.percentile(lc.circ_diff(reps, tau), [25, 75])
    half = 1.96 * (q75 - q25) / 1.349 * np.sqrt(70 / 30)
    assert (lo, hi) == pytest.approx((tau - half, tau + half))
    from experiments.benchmark.lags import gm20
    assert gm20.subsample_factor(70, 100) == pytest.approx(np.sqrt(70 / 30))


def test_model_b_is_scored_circularly(site):
    sc = _scorer(site)
    j = "terneuzen"
    tb = sc.truth_value(j, "b")[0]
    kinds = scoring.point_kinds("L14")
    assert kinds["b"] == "phase" and kinds["a"] == "linear"
    assert scoring.point_kinds("L17") == "linear"
    raw = {"a": 30.0, "ab": 30.0, "b": tb - P, "hwlw": 30.0}
    taus = scoring.point_wrap(raw, kinds)
    assert taus["b"] == pytest.approx(float(lc.wrap_phase(tb - P)))
    out = sc.score(j, taus, kinds)
    assert abs(out["b"]["e"]) < 1e-6                     # one period: the same phase
    lin = sc.score(j, raw, "linear")
    assert abs(lin["b"]["e"]) == pytest.approx(P)


def test_a_stale_truth_makes_the_judge_not_available(site, monkeypatch):
    monkeypatch.setattr(scoring.LagScorer, "check_truth_curve", lambda self, code, c: 0.5)
    out = scoring.score_lag(site.root, pre_lock=True, B=4, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, series_by_code=site.series, write=False)
    pts = out["data"]["points"]
    r = _row(pts, "L02", "terneuzen")
    assert r["status"].startswith("NOT AVAILABLE (truth stale: E_g differs")
    monkeypatch.setattr(scoring.LagScorer, "check_truth_curve", lambda self, code, c: None)
    out = scoring.score_lag(site.root, pre_lock=True, B=4, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, series_by_code=site.series, write=False)
    r = _row(out["data"]["points"], "L02", "terneuzen")
    assert "no E_a curve" in r["status"]


def test_the_first_table_shows_the_co_primary_truths(scored):
    md = open(scored["table_md"], encoding="utf-8").read()
    for text in ("co-primary truths (a), tau_ab and (b)", "MAE (a) (min) [CI]",
                 "MAE tau_ab (min) [CI]", "MAE (b) (min) [CI]", "RMS dE vs tau_a (cm) [CI]",
                 "MAE (min) against truth tau_ab", "MAE (min) against truth (b)"):
        assert text in md, text


def test_imagery_against_non_imagery_and_the_offset_note(scored):
    md = open(scored["table_md"], encoding="utf-8").read()
    assert "Imagery against non-imagery rows" in md and "datatake-start" in md
    ol = scored["data"]["context"]["offset_lines"]
    e = next(x for x in ol if x["site"] == SITE and x["window"] == WIN)
    assert e["n_imagery"] > 0 and e["n_non_imagery"] > 0 and e["sensing"] is None
    fam = scored["data"]["family"]
    assert fam[fam["comparison"] == "L17"]["label"].str.contains(
        scoring.LP1_OFFSET_NOTE, regex=False).all()
    assert not fam[fam["comparison"] == "L08"]["label"].str.contains(
        scoring.LP1_OFFSET_NOTE, regex=False).any()
    assert "Not computed in milestone V2" in md
    assert lagtable.product_info("L08")["fidelity_label"] == "F3 (paper check pending)"
    assert lagtable.product_info("L09")["fidelity_label"].startswith("F2 pending G-hd26")


def test_sensing_offset_reads_the_optional_diagnostic(tmp_path):
    p = os.path.join(tmp_path, "sensing", "ems", "sensing_times.json")
    product.write_json(p, {"dates": {"2024-01-02": {"instant": "x", "minus_start_min": 6.0},
                                     "2024-01-07": {"instant": "y", "minus_start_min": 7.0},
                                     "2024-01-12": {"instant": None}}})
    s = scoring.sensing_offset(str(tmp_path), "ems", cubes_root=str(tmp_path / "cubes"))
    assert s["median_min"] == 6.5 and s["n"] == 2


def test_partial_scores_never_replace_the_full_tables(site):
    out = scoring.score_lag(site.root, pre_lock=True, B=4, sites=[SITE], windows=[WIN],
                            packages={SITE: site.pkg}, series_by_code=site.series)
    want = os.path.join(site.root, "scores", f"partial_{SITE}__{WIN}")
    assert os.path.normpath(os.path.dirname(out["points"])) == os.path.normpath(want)
    assert "ledger" not in out                       # injected packages: not real 10 m data


def test_pre_lock_blinds_the_confirmatory_window_and_records_the_ledger(site, tmp_path):
    import shutil

    from experiments.benchmark import lock
    root = str(tmp_path / "real")
    os.makedirs(os.path.join(root, "plans"))
    shutil.copy(os.path.join(site.root, "plans", "lag.csv"),
                os.path.join(root, "plans", "lag.csv"))
    out = scoring.score_lag(root, pre_lock=True, B=4, invocation="score --pre-lock")
    pts = out["data"]["points"]
    w3 = pts[(pts["site"] == SITE) & (pts["window"] == "W3")]
    assert len(w3) and w3["status"].str.startswith("BLINDED").all()
    assert out["data"]["context"]["blinded"][0] == {"site": SITE, "window": "W3"}
    led = scoring.read_prelock_ledger(root)
    assert len(led) == 1 and led[0]["unblinded"] == [] and led[0]["points_sha256"]
    assert led[0]["invocation"] == "score --pre-lock"
    fp = lock.prelock_forking_paths(root)
    assert len(fp) == 1 and "the L-P1 comparison list" in fp[0]["could_influence"]
    md = open(out["table_md"], encoding="utf-8").read()
    assert "BLINDED before lock-A" in md
    out = scoring.score_lag(root, pre_lock=True, B=4, unblind=("W3",))
    pts = out["data"]["points"]
    assert not pts[pts["window"] == "W3"]["status"].str.startswith("BLINDED").any()
    led = scoring.read_prelock_ledger(root)
    assert len(led) == 2 and led[1]["unblinded"] == ["W3"]
    with pytest.raises(lock.LockRefused) as e:
        lock.write_lock("A", root, git_status=lambda: "")
    assert any("--amend-unblind" in r for r in e.value.reasons)


def test_profile_celerities_are_in_min_per_km():
    tau_t = np.array([10.0, 20.0, 40.0, 60.0])
    s_g = np.array([1.0, 5.0, 12.0, 20.0])
    tau_m = np.array([12.0, 21.0, 38.0, 65.0])
    pm = scoring.profile_metrics(tau_m, tau_t, s_g)
    assert pm["celerity_m_min_per_km"] == pytest.approx(np.polyfit(s_g, tau_m, 1)[0])
    assert pm["celerity_true_min_per_km"] == pytest.approx(np.polyfit(s_g, tau_t, 1)[0])
