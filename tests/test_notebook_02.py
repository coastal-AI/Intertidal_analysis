"""Tests of notebooks/validation/_build_02_lag_benchmark.py, the builder of
notebooks/validation/02_lag_benchmark.ipynb (the lag benchmark run of
2026-10-26).

* The built notebook is valid nbformat v4, deterministic (fixed cell ids, no
  outputs), has the sections 0-10 in order and a papermill-style
  "parameters" cell with LAG_DIR / V2_LAG_DIR, and the repository copy is
  the one the builder writes (no hand edits).
* The section-4 helpers (the cell tagged "section4-helpers"), run on a tiny
  synthetic lag_points sample: the recomputed summary follows the scorer's
  rule (locations = the mean over their gauges, the Terneuzen pair is one
  location; judge-excluded and BLINDED rows never enter), the markdown the
  benchmark's renderer writes from it passes the notebook's comparison, and
  a changed number in that markdown is caught.

Run:  python -X utf8 -m pytest tests/test_notebook_02.py -q
"""
import importlib.util
import os
import re
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BUILD = os.path.join(ROOT, "notebooks", "validation", "_build_02_lag_benchmark.py")


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("_build_02_lag_benchmark", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nb(builder):
    return builder.build()


@pytest.fixture(scope="module")
def helpers(nb):
    cells = [c for c in nb.cells if "section4-helpers" in c.metadata.get("tags", [])]
    assert len(cells) == 1
    ns = {"__name__": "section4_helpers"}
    exec(compile(cells[0].source, "<section4-helpers>", "exec"), ns)
    return ns


# ─────────────────────────────────────────────────────────────────────────────
#  structure
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_and_deterministic(builder, nb):
    nbformat.validate(nb)
    assert nb.nbformat == 4
    assert builder.text(builder.build()) == builder.text(nb)
    ids = [c.id for c in nb.cells]
    assert len(ids) == len(set(ids))
    for c in nb.cells:
        if c.cell_type == "code":
            assert c.outputs == [] and c.execution_count is None


def test_sections_in_order(nb):
    heads = [int(m.group(1)) for c in nb.cells if c.cell_type == "markdown"
             for m in [re.match(r"^## (\d+) · ", c.source)] if m]
    assert heads == list(range(11))
    assert nb.cells[0].source.startswith("# 02 · Lag benchmark")
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for phrase in ("BLINDED", "Home advantage", "weak evidence", "catalogue",
                   "Nothing is corrected", "nbclient", "run_v2_lag_2026-10-26.sh",
                   "score --pre-lock --show-unvalidated"):
        assert phrase in md, phrase


def test_parameters_cell(nb):
    p = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(p) == 1
    src = p[0].source
    assert 'os.environ.get("V2_LAG_DIR", "results/validation_v2")' in src
    assert "LAG_DIR" in src and "RUN_SCRIPT" in src


def test_repository_copy_is_built(builder):
    """The .ipynb in the repository is exactly what the builder writes."""
    with open(builder.NB_PATH, encoding="utf-8") as f:
        assert f.read() == builder.text()


def test_run_script_is_archived_with_its_hash():
    import hashlib

    path = os.path.join(ROOT, "experiments", "runs", "run_v2_lag_2026-10-26.sh")
    lines = open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    i0, i1 = lines.index("# >>> archive header"), lines.index("# <<< archive header")
    stated = re.search(r"sha256 of the original file: ([0-9a-f]{64})", "\n".join(lines)).group(1)
    original = "\n".join(lines[:i0] + lines[i1 + 1:])
    assert lines[0] == "#!/usr/bin/env bash"
    assert hashlib.sha256(original.encode("utf-8")).hexdigest() == stated
    assert "score --pre-lock --show-unvalidated" in original


# ─────────────────────────────────────────────────────────────────────────────
#  section 4 on a synthetic lag_points sample
# ─────────────────────────────────────────────────────────────────────────────

#: (judge, location, s_g_km, status, tau_a)
JUDGES = (("gatvanborssele", "gatvanborssele", 14.6, "ok", 13.0),
          ("terneuzen", "terneuzen", 20.4, "ok", 20.0),
          ("terneuzen.westsluis.buiten", "terneuzen", 20.4, "ok", 18.0),
          ("borssele", "borssele", 13.1, "judge excluded (coverage, dry)", np.nan))
#: product -> (validated, signed e per included judge under truth a)
PRODUCTS = {"L01": (False, (-8.0, -12.0, -10.0)), "L17": (True, (-13.0, -20.0, -18.0)),
            "L14": (True, (-9.0, -11.0, -10.0))}
TRUTH_SHIFT = {"a": 0.0, "ab": 0.5, "b": -1.0}


def _points():
    rows = []
    base = {"site": "escalda", "variant": "", "family": "", "fidelity": "", "lag_kind": "linear",
            "development": False, "mouth": False, "readout_step": "1", "n_readout_px": 100.0,
            "edge": False, "I_lo": np.nan, "I_hi": np.nan, "IS": np.nan, "covered": np.nan,
            "flags": ""}
    for pid, (val, es) in PRODUCTS.items():
        for T, sh in TRUTH_SHIFT.items():
            for k, (j, loc, s, st, tau) in enumerate(JUDGES):
                r = dict(base, window="Y2023", judge=j, location=loc, s_g_km=s, product=pid,
                         method=pid, status=st, validated=val, truth=T)
                if st == "ok":
                    e = es[k] + sh
                    r.update(tau_true=tau, tau_m=tau + e, e=e, abs_e=abs(e), dE_cm=0.1 * abs(e),
                             dE_all_cm=0.12 * abs(e), LLE_cm=abs(e), AGF=0.5, explained=0.4,
                             I_lo=tau - 6, I_hi=tau + 6)
                rows.append(r)
            # a blinded window: one status row per product and truth, no numbers
            rows.append(dict(base, window="W3", judge=None, location=None, s_g_km=np.nan, product=pid,
                             method=pid, validated=val, truth=T,
                             status="BLINDED (pre-lock: W3 is the confirmatory window of L-P1)"))
    df = pd.DataFrame(rows)
    for c in ("tau_true", "tau_m", "e", "abs_e", "dE_cm", "dE_all_cm", "LLE_cm", "AGF", "explained"):
        if c not in df:
            df[c] = np.nan
    return df


def _render(helpers, summ, lpw):
    lagtable = helpers["lagtable"]
    order = {k: i for i, k in enumerate(lagtable.CATALOGUE)}
    lines = lagtable._first_table(summ, {"lp1_windows": lpw}, order, True)
    return "\n".join(["# t", "", "## First lag table", ""] + lines +
                     ["", "## Imagery against non-imagery rows (catalogue instants)", ""])


def test_recompute_follows_the_scoring_rule(helpers):
    summ, scored, lpw, sets_by_sw = helpers["recompute_summary"](_points())
    assert scored == {("escalda", "Y2023")}
    assert lpw == {"escalda": "Y2023"}
    assert sets_by_sw[("escalda", "Y2023")]["headline"] == ["gatvanborssele", "terneuzen"]
    h = summ[(summ["judge_set"] == "headline") & (summ["site"] == "escalda")
             & (summ["window"] == "Y2023")].set_index(["product", "truth"])
    for pid, (_, es) in PRODUCTS.items():
        for T, sh in TRUTH_SHIFT.items():
            e = np.array(es) + sh
            loc_abs = [abs(e[0]), (abs(e[1]) + abs(e[2])) / 2]       # the Terneuzen pair: one location
            loc_e = [e[0], (e[1] + e[2]) / 2]
            r = h.loc[(pid, T)]
            assert r["n_locations"] == 2
            assert r["MAE"] == pytest.approx(np.mean(loc_abs))
            assert r["bias"] == pytest.approx(np.mean(loc_e))
            assert r["hit10"] == pytest.approx(np.mean(np.array(loc_abs) <= 10.0))
    # the blinded window is never scored
    w3 = summ[summ["window"] == "W3"]
    assert (w3["n_locations"].fillna(0) == 0).all()


def test_markdown_comparison_round_trip(helpers):
    summ, _, lpw, _ = helpers["recompute_summary"](_points())
    md = _render(helpers, summ, lpw)
    site_df, cross_df = helpers["parse_first_table"](md)
    assert set(site_df["product"]) >= set(PRODUCTS)
    assert set(cross_df["site"]) == {"escalda", "pooled"}
    cmp_ = helpers["compare_first_table"](summ, md)
    assert len(cmp_) and cmp_["ok"].all(), cmp_[~cmp_["ok"]]
    # L01 is not validated: printed as "not validated (shown)", L17 as "ok"
    st = cmp_[(cmp_["quantity"] == "status") & (cmp_["block"] == "site")].set_index("product")["printed"]
    assert st["L01"] == "not validated (shown)" and st["L17"] == "ok"


def test_markdown_comparison_catches_a_changed_number(helpers):
    summ, _, lpw, _ = helpers["recompute_summary"](_points())
    md = _render(helpers, summ, lpw)
    mae = summ[(summ["judge_set"] == "headline") & (summ["product"] == "L01") & (summ["truth"] == "a")
               & (summ["site"] == "escalda")]["MAE"].iloc[0]
    line = next(ln for ln in md.split("\n") if ln.startswith("| L01 "))
    bad_line = line.replace(f"| {mae:.1f}", f"| {mae + 0.3:.1f}", 1)
    assert bad_line != line
    cmp_ = helpers["compare_first_table"](summ, md.replace(line, bad_line))
    bad = cmp_[~cmp_["ok"]]
    assert len(bad) == 1 and bad.iloc[0]["product"] == "L01" and bad.iloc[0]["quantity"] == "MAE"


def test_parse_cell(helpers):
    pc = helpers["parse_cell"]
    assert pc("9.5 [6.5, 12.0]") == (9.5, 6.5, 12.0)
    assert pc("+32.0")[0] == 32.0 and np.isnan(pc("+32.0")[1])
    assert all(np.isnan(v) for v in pc("–"))
    assert pc("NOT COMPUTABLE (ensemble)") is None
