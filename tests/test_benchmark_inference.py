"""Tests of experiments.benchmark.inference, the lag inference (spec §6.4-§6.5,
§9.1-§9.2, §9.8).

* month-block bootstrap: multiplicities per calendar month, zero weight for
  instants without a gauge value, deterministic per seed, and, with the
  truth's own seed, the weighted argmin reproduces ``truth.month_boot``'s
  draws exactly; the weighted median equals ``np.median`` of the replicated
  sample and the weighted E equals ``truth.e_rows`` with unit weights;
* the exact sign-flip test equals brute-force enumeration for n <= 10 with
  mixed site sizes (site-weighted statistic), k_zero and p_min =
  2 / 2^(n - k_zero);
* Holm's step-down (adjusted p, rejections, NaN members);
* family L-P1: site agreement (>= 3 of 4 judged sites), the descriptive
  version without verdict, and the ranking rule (all three truths);
* ICC(2,1) on Shrout & Fleiss's (1979) Table 2 example (0.29) and Lin's CCC;
* attain_report on a synthetic D_truth: blocks at 1/4/16 km, bands with
  truth, the window-year guard, the L-P1 attainable p, and the assertion
  2/(B + 1) < alpha/m for every bootstrap family.

Run:  python -X utf8 -m pytest tests/test_benchmark_inference.py -q
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import config, inference, truth  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  bootstrap weights
# ─────────────────────────────────────────────────────────────────────────────

def _instants(n=150, seed=3):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(np.arange(366), n, replace=False))
    return pd.DatetimeIndex(pd.Timestamp("2024-01-01 10:50") + pd.to_timedelta(days, unit="D"))


def test_month_block_weights_are_month_multiplicities():
    t = _instants()
    fin = np.ones(len(t), bool)
    fin[::7] = False
    W = inference.month_block_weights(t, fin, 50, [1, 2, 3])
    assert W.shape == (50, len(t)) and W.dtype == np.int64
    assert (W[:, ~fin] == 0).all()
    lab = inference.month_labels(t)
    for b in range(50):
        for mth in np.unique(lab[fin]):
            vals = W[b, fin & (lab == mth)]
            assert (vals == vals[0]).all()          # one multiplicity per month
        # n_months draws in total
        per_month = [W[b, fin & (lab == m)][0] for m in np.unique(lab[fin])]
        assert sum(per_month) == len(np.unique(lab[fin]))
    assert np.array_equal(W, inference.month_block_weights(t, fin, 50, [1, 2, 3]))
    assert not np.array_equal(W, inference.month_block_weights(t, fin, 50, [1, 2, 4]))
    assert inference.month_block_weights(t, np.zeros(len(t), bool), 5, 1).sum() == 0
    with pytest.raises(ValueError):
        inference.month_block_weights(t, fin[:-1], 5, 1)


def test_weighted_median_equals_the_replicated_median():
    rng = np.random.default_rng(0)
    v = rng.normal(size=23)
    W = rng.integers(0, 4, size=(200, 23))
    W[0] = 0
    W[0, 5] = 2
    med = inference.weighted_median(v, W)
    for b in range(200):
        rep = np.repeat(v, W[b])
        if len(rep):
            assert med[b] == np.median(rep)
        else:
            assert np.isnan(med[b])


def test_weighted_mcrms_unit_weights_is_e_rows_and_replication():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(7, 31))
    ones = np.ones((1, 31), np.int64)
    E = inference.weighted_mcrms(D, ones)[0]
    g = np.zeros(31)
    assert np.allclose(E, truth.e_rows(D, g), rtol=0, atol=1e-15)
    W = rng.integers(0, 3, size=(5, 31))
    Ew = inference.weighted_mcrms(D, W)
    for b in range(5):
        rep = np.repeat(D, W[b], axis=1)
        want = truth.e_rows(rep, np.zeros(rep.shape[1]))
        assert np.allclose(Ew[b], want, rtol=0, atol=1e-12)


def test_weights_with_the_truth_seed_reproduce_month_boot():
    t = _instants(160, seed=5)
    taus = truth.TAUS_A
    rng = np.random.default_rng(9)
    x = t.asi8 / 3.6e12
    H = np.vstack([1.6 * np.cos(2 * np.pi * (x - tau / 60.0) / 12.4206012) for tau in taus])
    g = 1.6 * np.cos(2 * np.pi * (x - 40 / 60.0) / 12.4206012) + rng.normal(0, 0.05, len(t))
    g[::11] = np.nan
    seed = [20261011, 2, 12345, 2024]
    want = truth.month_boot(H, g, t, rng=np.random.default_rng(seed))
    m = np.isfinite(g)
    W = inference.month_block_weights(t, m, truth.N_BOOT, seed)
    Ew = inference.weighted_mcrms(H[:, m] - g[m], W[:, m])
    got = taus[np.argmin(Ew, axis=1)].astype(float)
    assert np.array_equal(got, want)


def test_percentile_ci():
    lo, hi = inference.percentile_ci(np.arange(1001.0))
    assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)
    x = np.full(10, np.nan)
    x[:4] = 1.0
    assert np.isnan(inference.percentile_ci(x)[0])          # fewer than half finite


# ─────────────────────────────────────────────────────────────────────────────
#  sign flips and Holm
# ─────────────────────────────────────────────────────────────────────────────

def _brute(d_by_site):
    sites = [s for s in d_by_site if len(d_by_site[s])]
    vals = [(s, v) for s in sites for v in d_by_site[s]]
    n = len(vals)

    def stat(signs):
        means = []
        for s in sites:
            xs = [sg * v for (ss, v), sg in zip(vals, signs) if ss == s]
            means.append(np.mean(xs))
        return float(np.mean(means))

    S = stat([1] * n)
    count = sum(abs(stat(sg)) >= abs(S) - 1e-12 for sg in itertools.product((1, -1), repeat=n))
    return S, count / 2 ** n


@pytest.mark.parametrize("sizes,seed", [((1, 2, 1, 4), 0), ((3, 3), 1), ((1, 1, 1, 1), 2),
                                        ((5, 1, 2, 2), 3), ((10,), 4)])
def test_sign_flip_equals_brute_force(sizes, seed):
    rng = np.random.default_rng(seed)
    d = {f"s{k}": rng.normal(0.5, 2.0, n) for k, n in enumerate(sizes)}
    r = inference.sign_flip(d)
    S, p = _brute(d)
    assert r["stat"] == pytest.approx(S, abs=1e-12)
    assert r["p"] == p and r["n"] == sum(sizes) and r["n_sites"] == len(sizes)
    assert r["k_zero"] == 0 and r["p_min"] == pytest.approx(2.0 / 2 ** sum(sizes))


def test_sign_flip_zero_differences_and_site_weighting():
    d = {"a": np.array([1.0, 0.0, 2.0]), "b": np.array([0.0, -3.0]), "c": np.array([np.nan])}
    r = inference.sign_flip(d)
    assert r["n"] == 5 and r["k_zero"] == 2 and r["n_sites"] == 2
    assert r["p_min"] == pytest.approx(2.0 / 2 ** 3)
    assert r["p"] >= r["p_min"]
    # site weighting: each site counts once, whatever its size
    assert r["stat"] == pytest.approx(np.mean([np.mean([1, 0, 2]), np.mean([0, -3])]))
    S, p = _brute({k: v[np.isfinite(v)] for k, v in d.items()})
    assert r["p"] == p
    # all positive: the smallest attainable p
    r = inference.sign_flip({"x": np.array([1.0, 2.0, 3.0, 4.0])})
    assert r["p"] == pytest.approx(2.0 / 16)
    r = inference.sign_flip({})
    assert r["n"] == 0 and np.isnan(r["p"])


def test_holm_step_down():
    p = {"A": 0.004, "B": 0.03, "C": 0.012, "D": 0.6, "E": float("nan")}
    h = inference.holm(p, alpha=0.05)
    m = 5
    # sorted: A 0.004 (x5 = 0.02), C 0.012 (x4 = 0.048), B 0.03 (x3 = 0.09), D, E (as 1)
    assert h["A"]["p_adj"] == pytest.approx(0.02) and h["A"]["reject"]
    assert h["C"]["p_adj"] == pytest.approx(0.048) and h["C"]["reject"]
    assert h["B"]["p_adj"] == pytest.approx(0.09) and not h["B"]["reject"]
    assert h["D"]["p_adj"] == pytest.approx(1.0) and not h["D"]["reject"]
    assert not h["E"]["reject"] and np.isnan(h["E"]["p"])
    assert h["A"]["threshold"] == pytest.approx(0.05 / m)
    assert h["C"]["threshold"] == pytest.approx(0.05 / (m - 1))
    # step-down stops at the first failure even if a later p meets its own
    # threshold: x fails 0.05/3, so y (0.04 < 0.05/2) and z are not rejected
    h = inference.holm({"x": 0.02, "y": 0.024, "z": 0.026}, alpha=0.05)
    assert not h["x"]["reject"] and not h["y"]["reject"] and not h["z"]["reject"]
    assert h["z"]["p_adj"] == pytest.approx(0.06)          # max(0.06, 0.048, 0.026)
    h = inference.holm({"x": 0.01, "y": 0.04}, alpha=0.05)
    assert h["x"]["reject"] and h["y"]["reject"]


# ─────────────────────────────────────────────────────────────────────────────
#  family L-P1
# ─────────────────────────────────────────────────────────────────────────────

def _points(abs_e):
    """Judge-level rows from {(site, location): {product: |e|}}."""
    rows = []
    for (site, loc), prods in abs_e.items():
        for p, v in prods.items():
            rows.append({"site": site, "window": "W3", "location": loc, "judge": loc,
                         "product": p, "truth": "a", "abs_e": v, "e": v})
    return pd.DataFrame(rows)


def _family_points(better_at=("escalda", "wadden", "ems", "ferrol"), gap=8.0):
    locs = {"escalda": ["e1", "e2"], "wadden": ["w1", "w2"],
            "ems": [f"m{k}" for k in range(8)], "ferrol": ["f1"]}
    out = {}
    for site, ls in locs.items():
        for k, loc in enumerate(ls):
            m = 3.0 + 0.1 * k
            x = m + gap if site in better_at else m - gap
            out[(site, loc)] = {"L01": m, "L17": x, "L08": m + 0.05 * (-1) ** k}
    return _points(out)


def test_family_lp1_rejects_with_site_agreement():
    pts = _family_points()
    fam = inference.family_lp1(pts, "a", ["L17", "L08"], ["escalda", "wadden", "ems", "ferrol"],
                               label="test")
    r = fam.set_index("comparison")
    assert list(fam.columns) == list(inference.FAMILY_COLUMNS)
    assert r.loc["L17", "n"] == 13 and r.loc["L17", "k_zero"] == 0
    assert r.loc["L17", "S"] == pytest.approx(-8.0)
    assert r.loc["L17", "p"] == pytest.approx(2.0 / 2 ** 13)
    assert r.loc["L17", "agree_sites"] == 4
    assert r.loc["L17", "reject"] and r.loc["L17", "confirmatory"]
    assert not r.loc["L08", "reject"] and not r.loc["L08", "confirmatory"]
    assert r.loc["L17", "holm_p"] == pytest.approx(2 * 2.0 / 2 ** 13)
    assert "ems@W3=-1" in r.loc["L17", "site_signs"]


def test_site_agreement_needs_three_of_four_sites():
    # MAREA is better at ems and ferrol only: the ems locations dominate a
    # pooled test, but the site-weighted statistic and the agreement rule do not
    pts = _family_points(better_at=("ems", "ferrol"), gap=8.0)
    fam = inference.family_lp1(pts, "a", ["L17"], ["escalda", "wadden", "ems", "ferrol"])
    r = fam.iloc[0]
    assert r["agree_sites"] <= 2 and not r["confirmatory"]
    pooled = inference.pooled_sign_test(pts, "a", "L17")
    assert pooled["label"] == "pseudo-replicated (description only)"
    assert pooled["S"] < 0 and pooled["n"] == 13
    # three of four agree: confirmatory (when Holm rejects)
    pts = _family_points(better_at=("escalda", "ems", "ferrol"), gap=8.0)
    fam = inference.family_lp1(pts, "a", ["L17"], ["escalda", "wadden", "ems", "ferrol"])
    assert fam.iloc[0]["agree_sites"] == 3


def test_descriptive_family_issues_no_verdict():
    pts = _family_points()
    fam = inference.family_lp1(pts, "a", ["L17"], ["escalda", "wadden", "ems", "ferrol"],
                               verdict=False, label="descriptive")
    assert fam.iloc[0]["reject"] is None and fam.iloc[0]["confirmatory"] is None
    assert np.isfinite(fam.iloc[0]["p"]) and np.isfinite(fam.iloc[0]["holm_p"])


def test_family_refuses_several_windows_per_site():
    pts = _family_points()
    pts.loc[pts.index[:3], "window"] = "Y2023"
    with pytest.raises(ValueError):
        inference.family_lp1(pts, "a", ["L17"], ["escalda", "wadden", "ems", "ferrol"])


def test_gauges_of_one_location_are_averaged():
    rows = [{"site": "escalda", "window": "W3", "location": "terneuzen", "judge": j,
             "product": p, "truth": "a", "abs_e": v, "e": v}
            for j, p, v in (("terneuzen", "L01", 2.0), ("terneuzen.westsluis.buiten", "L01", 4.0),
                            ("terneuzen", "L17", 20.0),
                            ("terneuzen.westsluis.buiten", "L17", 22.0))]
    loc = inference.location_abs_e(pd.DataFrame(rows))
    assert len(loc) == 2
    assert loc.set_index("product").loc["L01", "abs_e"] == 3.0
    fam = inference.family_lp1(pd.DataFrame(rows), "a", ["L17"], ["escalda"])
    assert fam.iloc[0]["n"] == 1 and fam.iloc[0]["S"] == pytest.approx(-18.0)


def test_ranking_needs_all_three_truths():
    rows = []
    for T in ("a", "ab", "b"):
        rows.append({"truth": T, "comparison": "L17", "S": -5.0, "confirmatory": True})
        rows.append({"truth": T, "comparison": "L08", "S": -2.0,
                     "confirmatory": T != "b"})
        rows.append({"truth": T, "comparison": "L09", "S": 3.0 if T != "ab" else -1.0,
                     "confirmatory": True})
    claims = inference.ranking_claims(pd.DataFrame(rows))
    assert claims == {"L17": "MAREA better", "L08": None, "L09": None}
    rows = [{"truth": T, "comparison": "L14", "S": 4.0, "confirmatory": True}
            for T in ("a", "ab", "b")]
    assert inference.ranking_claims(pd.DataFrame(rows)) == {"L14": "MAREA worse"}


# ─────────────────────────────────────────────────────────────────────────────
#  ICC and CCC
# ─────────────────────────────────────────────────────────────────────────────

def test_icc21_shrout_fleiss_table2():
    Y = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9],
                  [6, 2, 4, 7]], float)
    assert inference.icc21(Y) == pytest.approx(0.29, abs=0.005)
    # perfect agreement, incomplete rows dropped, degenerate cases
    Z = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [np.nan, 3.0]])
    assert inference.icc21(Z) == pytest.approx(1.0)
    assert np.isnan(inference.icc21(np.ones((1, 3))))
    assert np.isnan(inference.icc21(np.ones((4, 1))))


def test_lin_ccc():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert inference.lin_ccc(x, x) == pytest.approx(1.0)
    assert inference.lin_ccc(x, x + 1.0) < 1.0
    y = 2 * x
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    want = 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    assert inference.lin_ccc(x, y) == pytest.approx(want)
    assert np.isnan(inference.lin_ccc([1.0], [1.0]))


# ─────────────────────────────────────────────────────────────────────────────
#  attainability
# ─────────────────────────────────────────────────────────────────────────────

def _truth_doc(codes_included, years_ok=True):
    st = []
    for code, inc in codes_included.items():
        st.append({"code": code, "location": code.split(".")[0] if "terneuzen" in code else code,
                   "windows": {"W3": {"inclusion": {"included": inc,
                                                    "exclusion": [] if inc else ["coverage"]}}}})
    return {"format": truth.TRUTH_LAG_V2_FORMAT, "stations": st}


def test_attain_report_on_a_synthetic_d_truth(tmp_path):
    H, W = 400, 300
    rr, cc = np.meshgrid(np.arange(100, 300), np.arange(50, 250), indexing="ij")
    d_truth = np.sort((rr * W + cc).ravel())
    band = np.full((H, W), -1, np.int64)
    band[:, 50:250] = (np.arange(200) // 30)[None, :]           # 7 bands, 30 px wide
    inputs = {"escalda": {"d_truth": d_truth, "band_frame": band, "shape": (H, W),
                          "pixel_m": 10.0, "years": [2023, 2024, 2025]},
              "ems": {"d_truth": d_truth[:5000], "band_frame": band, "shape": (H, W),
                      "pixel_m": 10.0, "years": [2023]},
              "villaviciosa": {"d_truth": None, "band_frame": band, "shape": (H, W),
                               "pixel_m": 10.0, "years": [2023, 2024, 2025]}}
    docs = {"escalda": _truth_doc({"gatvanborssele": True, "terneuzen": True,
                                   "terneuzen.westsluis.buiten": True, "borssele": False}),
            "ems": _truth_doc({"delfzijl": True, "knock_3990010": True})}
    out = inference.attain_report(str(tmp_path), sites=["escalda", "ems", "villaviciosa"],
                                  inputs=inputs, truth_docs=docs)
    # the attainability assertion of spec §9.1 for every bootstrap family
    B = config.FROZEN["inference"]["boot"]["B_primary"]
    alpha = config.FROZEN["inference"]["holm_alpha"]
    boots = {k: v for k, v in out["families"].items() if v["kind"] == "bootstrap"}
    assert set(boots) == {"P1", "P2", "P3"}
    for name, f in boots.items():
        assert 2.0 / (B + 1) < alpha / f["m"], name
        assert f["attainable"]
    assert out["bootstrap_families_attainable"] is True
    e = out["sites"]["escalda"]
    # 2 x 2 km of truth at 10 m; the block grid is anchored at the frame's first
    # pixel: rows 100-299 touch 2 row blocks of 1 km, cols 50-249 touch 3
    assert e["blocks"]["1km"]["n_blocks"] == 6 and not e["blocks"]["1km"]["testable"]
    assert e["blocks"]["4km"]["n_blocks"] == 1
    assert e["blocks"]["16km"]["n_blocks"] == 1
    want_bands = int(np.sum(np.bincount(band.ravel()[d_truth][band.ravel()[d_truth] >= 0])
                            >= 30))
    assert e["bands_with_truth"] == want_bands == 7 and e["p1_band_test_possible"]
    # the Terneuzen pair is one location; borssele excluded
    assert e["lp1_locations_W3"] == ["gatvanborssele", "terneuzen"]
    # ems holds 2023 only: the window-year guard gives no W3 location
    assert out["sites"]["ems"]["lp1_locations_W3"] == []
    assert "store years" in out["sites"]["ems"]["lp1_note"]
    assert out["sites"]["villaviciosa"]["status"].startswith("no D_truth")
    lp1 = out["lag_families"]["L-P1"]
    assert lp1["n_locations"] == 2 and lp1["p_min_exact"] == pytest.approx(0.5)
    assert not lp1["by_m"]["5"]["attainable"]
    assert lp1["by_m"]["5"]["holm_thresholds"][0] == pytest.approx(alpha / 5)
    with open(tmp_path / "attain.json", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["format"] == inference.ATTAIN_FORMAT and doc["rules"]


def test_attain_with_13_locations_is_attainable(tmp_path):
    docs = {"ems": _truth_doc({f"g{k}": True for k in range(13)})}
    inputs = {"ems": {"d_truth": np.arange(10), "band_frame": np.zeros((10, 10), np.int64),
                      "shape": (10, 10), "pixel_m": 10.0, "years": [2023, 2024, 2025]}}
    out = inference.attain_report(str(tmp_path), sites=["ems"], inputs=inputs, truth_docs=docs,
                                  write=False)
    lp1 = out["lag_families"]["L-P1"]
    assert lp1["n_locations"] == 13
    assert lp1["p_min_exact"] == pytest.approx(2.0 / 8192)
    assert lp1["by_m"]["5"]["attainable"] and lp1["by_m"]["4"]["attainable"]
    assert not os.path.exists(tmp_path / "attain.json")
