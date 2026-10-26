"""Tests of experiments.benchmark.lagtable, the lag score files (scoring spec §9).

* The CSV columns are exactly the scoring spec's lists (lag_points,
  lag_family) and, for lag_summary, the key columns plus the spec's list
  with the CI columns §5 asks for next to RMS dE and RMS LLE and the mean
  explained share; the repro and cost files have fixed columns too.
* Pre-lock, every file carries the banner (the CSVs on their first line,
  skipped by ``read_csv``; the markdown at the top and the bottom;
  lag_meta.json), and without it none does.
* The markdown has one table per site x window x judge, rows grouped in the
  four panels, and rows reading NOT COMPUTABLE (<stage>), not applicable
  (<plan note>), NOT AVAILABLE (<reason>) and "implementation not
  validated" (no numbers unless the development view is asked for), never
  blank; the first lag table and the cross-site summary over calendar-year
  windows; the table notes (home advantage, weak evidence).
* Product names, tiers, fidelity and panels from the catalogue of §3.3 when
  the cards are not used.

Run:  python -X utf8 -m pytest tests/test_benchmark_lagtable.py -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.benchmark import inference, lagtable  # noqa: E402

#: scoring spec §9, verbatim
SPEC_POINTS = ("site, window, judge, location, s_g_km, mouth, development, product, method, "
               "variant, family, fidelity, lag_kind, status, reason, validated, readout_step, "
               "n_readout_px, edge, tau_m, tau_m_band, tau_m_band_interp, tau_m_step1, truth, "
               "tau_true, I_lo, I_hi, e, abs_e, e_lo, e_hi, err_int, E0_cm, Etau_cm, Emin_cm, "
               "dE_cm, dE_lo_cm, dE_hi_cm, Eall_tau_cm, Eall_min_cm, dE_all_cm, LLE_cm, "
               "LLE_lo_cm, LLE_hi_cm, explained, AGF, AGF_lo, AGF_hi, material, IS, covered, "
               "n_instants, B, flags")
SPEC_SUMMARY = ("n_locations, MAE, MAE_lo, MAE_hi, median_abs_e, bias, bias_lo, bias_hi, "
                "max_abs_e, hit10, hitI, rms_dE_cm, rms_dE_all_cm, rms_LLE_cm, mean_AGF, "
                "mean_AGF_lo, mean_AGF_hi, edge_share, step_shares, slope, ccc, shape_err, "
                "mono_violations")
SPEC_FAMILY = ("truth, comparison, n, k_zero, S, p, p_min, holm_p, reject, site_signs, "
               "agree_sites, confirmatory, label")


def _cols(text):
    return tuple(c.strip() for c in text.split(","))


@pytest.fixture(autouse=True)
def no_cards():
    lagtable.USE_CARDS = False
    lagtable._card_info.cache_clear()
    yield
    lagtable.USE_CARDS = True
    lagtable._card_info.cache_clear()


def _point(product, status="ok", validated=True, truth="a", judge="gatvanborssele", **kw):
    r = {c: np.nan for c in lagtable.POINTS_COLUMNS}
    r.update(site="escalda", window="Y2023", judge=judge, location=judge, s_g_km=14.6,
             mouth=False, development=False, product=product, method=product, variant="",
             family="", fidelity="", lag_kind="linear", status=status, reason=None,
             validated=validated, readout_step=1 if status == "ok" else None, n_readout_px=80,
             edge=False, truth=truth, tau_true=10.0, I_lo=8.0, I_hi=12.0, flags="")
    if status == "ok":
        r.update(tau_m=12.5, e=2.5, abs_e=2.5, e_lo=1.5, e_hi=3.5, E0_cm=20.0, Etau_cm=4.1,
                 Emin_cm=3.0, dE_cm=1.1, dE_lo_cm=0.8, dE_hi_cm=1.4, LLE_cm=2.6, AGF=0.94,
                 AGF_lo=0.9, AGF_hi=0.97, B=1000, n_instants=150)
    r.update(kw)
    return r


def _points():
    rows = []
    for T in ("a", "ab", "b", "hwlw"):
        rows += [_point("L01", truth=T), _point("L09", validated=False, truth=T),
                 _point("L03", status="NOT COMPUTABLE (min_scenes)", truth=T),
                 _point("L16", status="NOT AVAILABLE (AMM15 archive starts 2024-08-04)",
                        truth=T),
                 _point("L18", status="not applicable (EMODnet tiles not fetched)", truth=T),
                 _point("L17", truth=T, tau_m=0.0, e=-10.0, abs_e=10.0, readout_step="point"),
                 _point("L01", truth=T, judge="terneuzen"),
                 _point("L01", truth=T, judge="borssele", status="judge excluded (coverage)")]
    rows += [_point("L01", truth="a", window="W3",
                    status="NOT AVAILABLE: store years [2024, 2025]")]
    return pd.DataFrame(rows)


def _summary():
    rows = []
    for site, w, v in (("escalda", "Y2023", True), ("ems", "Y2023", True),
                       ("pooled", "Y2023", True), ("escalda", "Y2023", False)):
        r = {c: np.nan for c in lagtable.SUMMARY_COLUMNS}
        r.update(site=site, window=w, product="L01" if v else "L09", truth="a",
                 judge_set="headline", status="ok", validated=v, n_locations=2, MAE=3.5,
                 MAE_lo=2.0, MAE_hi=5.0, bias=1.0, rms_dE_cm=1.2, rms_LLE_cm=2.2,
                 mean_AGF=0.9, hit10=1.0, label="")
        rows.append(r)
    return pd.DataFrame(rows)


def _plan():
    return [{"products": p} for p in ("E0a E0b L01", "L03", "L09 L10", "L16", "L17", "L18")]


def _write(tmp_path, pre_lock=True, show=False):
    fam = pd.DataFrame([{"truth": "a", "comparison": "L17", "n": 2, "k_zero": 0, "S": -5.0,
                         "p": 0.5, "p_min": 0.5, "holm_p": 1.0, "reject": None,
                         "site_signs": "escalda@Y2023=-1", "agree_sites": 1,
                         "confirmatory": None, "label": "descriptive"}])
    rep = pd.DataFrame([{"site": "villaviciosa", "window": "Y2024", "kind": "coherence",
                         "product": "all", "band": 0, "centre_km": 1.5, "n_groups": 2,
                         "groups": "L01;L02", "values": "L01=5.00;L02=0.00",
                         "median_abs_diff": np.nan, "icc21": np.nan, "spread_sd": 2.5,
                         "spread_range": 5.0, "label": "lag coherence only"}])
    ctx = {"lp1_windows": {"escalda": "Y2023", "ems": "Y2023"}, "B": 1000,
           "show_unvalidated": show, "validated": {"villaviciosa|L01": True,
                                                   "villaviciosa|L02": False}}
    return lagtable.write(_points(), _summary(), fam, rep, _plan(), str(tmp_path), pre_lock,
                          cost=[{"product": "L01", "method": "E0", "n_instances": 3,
                                 "wall_s_median": 12.0}], context=ctx)


def test_csv_columns_are_the_spec_lists():
    assert lagtable.POINTS_COLUMNS == _cols(SPEC_POINTS)
    assert inference.FAMILY_COLUMNS == _cols(SPEC_FAMILY) == lagtable.FAMILY_COLUMNS
    spec = _cols(SPEC_SUMMARY)
    metrics = lagtable.SUMMARY_METRICS
    # the spec's columns in the spec's order, plus the CIs of RMS dE / LLE and
    # the mean explained share that §5 requires
    assert [c for c in metrics if c in spec] == list(spec)
    # ... and the apparent celerities of the profile metrics (spec §6.4, min/km)
    assert set(metrics) - set(spec) == {"rms_dE_lo_cm", "rms_dE_hi_cm", "rms_LLE_lo_cm",
                                        "rms_LLE_hi_cm", "mean_explained",
                                        "celerity_m_min_per_km", "celerity_true_min_per_km"}
    assert lagtable.SUMMARY_COLUMNS[:len(lagtable.SUMMARY_KEYS)] == lagtable.SUMMARY_KEYS
    assert {"site", "window", "product", "truth", "judge_set"} <= set(lagtable.SUMMARY_KEYS)


def test_written_files_have_the_columns_and_the_banner(tmp_path):
    paths = _write(tmp_path)
    assert set(paths) == {"points", "summary", "family", "repro", "cost", "table_md", "meta"}
    banner = f"# {lagtable.BANNER}"
    for name, cols in (("points", lagtable.POINTS_COLUMNS), ("summary", lagtable.SUMMARY_COLUMNS),
                       ("family", lagtable.FAMILY_COLUMNS), ("repro", lagtable.REPRO_COLUMNS),
                       ("cost", lagtable.COST_COLUMNS)):
        with open(paths[name], encoding="utf-8") as f:
            assert f.readline().rstrip("\n") == banner, name
        df = lagtable.read_csv(paths[name])
        assert tuple(df.columns) == cols, name
    with open(paths["points"], "rb") as f:
        assert b"\r\n" not in f.read()
    md = open(paths["table_md"], encoding="utf-8").read()
    assert md.splitlines()[2] == f"> **{lagtable.BANNER}**"
    assert md.rstrip().endswith(f"> **{lagtable.BANNER}**")
    meta = json.load(open(paths["meta"], encoding="utf-8"))
    assert meta["pre_lock"] is True and meta["banner"] == lagtable.BANNER
    # after the lock: no banner anywhere
    out = tmp_path / "locked"
    paths = _write(out, pre_lock=False)
    for name in ("points", "summary", "family", "repro", "cost"):
        with open(paths[name], encoding="utf-8") as f:
            assert not f.readline().startswith("#")
    assert lagtable.BANNER not in open(paths["table_md"], encoding="utf-8").read()


def test_markdown_rows_are_never_blank(tmp_path):
    md = open(_write(tmp_path)["table_md"], encoding="utf-8").read()
    for note in lagtable.TABLE_NOTES:
        assert note in md
    assert "#### escalda · Y2023 · gatvanborssele" in md
    assert "#### escalda · Y2023 · terneuzen" in md
    assert "#### escalda · Y2023 · borssele" not in md          # excluded judge: one line
    assert "- borssele: judge excluded (coverage)" in md
    for text in ("NOT COMPUTABLE (min_scenes)", "not applicable (EMODnet tiles not fetched)",
                 "NOT AVAILABLE (AMM15 archive starts 2024-08-04)",
                 "implementation not validated", "NOT AVAILABLE: store years [2024, 2025]"):
        assert text in md, text
    # the not-validated row carries no numbers
    line = next(ln for ln in md.splitlines() if ln.startswith("| L09 "))
    assert "implementation not validated" in line and "12.5" not in line
    line = next(ln for ln in md.splitlines() if ln.startswith("| L01 ") and "12.5" in ln)
    assert "+2.5 [+1.5, +3.5]" in line and "20.0 / 4.1 / 3.0" in line
    assert "94 [90, 97]" in line
    # panels in order, with the literature apart from the in-house adaptations
    tbl = md.split("#### escalda · Y2023 · gatvanborssele")[1].split("####")[0]
    i0 = tbl.index(lagtable.PANELS[0])
    i1 = tbl.index(lagtable.PANELS[1])
    i3 = tbl.index(lagtable.PANELS[3])
    assert i0 < tbl.index("| L01 ") < i1 < tbl.index("| L09 ") < i3 < tbl.index("| L17 ")
    # first lag table and the calendar-year cross-site summary
    first = md.split("## First lag table")[1].split("## Family")[0]
    assert "**escalda · Y2023**" in first and "**ems · Y2023**" in first
    assert "*Y2023* — MAE (min) against truth (a)" in first
    row = next(ln for ln in first.splitlines() if ln.startswith("| L09 "))
    assert "implementation not validated" in row and "3.5" not in row
    # the coherence table hides the product that is not validated
    coh = md.split("lag coherence only (no included judge)")[1]
    assert "Not shown (implementation not validated): L02." in coh
    assert "| band | centre (km) | L01 |" in coh


def test_development_view_shows_unvalidated_numbers(tmp_path):
    md = open(_write(tmp_path, show=True)["table_md"], encoding="utf-8").read()
    line = next(ln for ln in md.splitlines() if ln.startswith("| L09 ") and "12.5" in ln)
    assert "not validated (shown)" in line


def test_product_info_catalogue_and_panels():
    i = lagtable.product_info("L08")
    assert i["fidelity"] == "F3" and i["lag_kind"] == "phase" and i["panel"] == lagtable.PANELS[1]
    assert lagtable.product_info("L01")["panel"] == lagtable.PANELS[0]
    assert lagtable.product_info("L04w")["panel"] == lagtable.PANELS[2]
    assert lagtable.product_info("L17")["panel"] == lagtable.PANELS[3]
    d = lagtable.product_info("L14@GOT4.10c")
    assert d["base"] == "L14" and "GOT4.10c" in d["name"] and d["method"] == "L14"
    assert lagtable.product_info("L01v@orbit037")["fidelity"] == "F0"
    assert lagtable.product_info("L99")["panel"] == lagtable.PANELS[3]
