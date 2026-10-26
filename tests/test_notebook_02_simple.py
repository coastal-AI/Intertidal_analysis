# -*- coding: utf-8 -*-
"""The simple notebooks of the lag benchmark (notebooks/validation/02_lag_vs_gauges_site.ipynb and
02b_lag_summary.ipynb) are valid nbformat v4 notebooks, carry no outputs, and every code cell
compiles. The site notebook reads its site from NB_SITE, never scores the blinded window W3, numbers
its sections 0.., and calls the estimators and truth functions directly. The notebooks are not
executed here: they need the server's data (run_02_on_server.sh).

Run:  python -m pytest -q tests/test_notebook_02_simple.py
"""
import os

import nbformat
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "validation")
NOTEBOOKS = ["02_lag_vs_gauges_site.ipynb", "02b_lag_summary.ipynb"]


def _read(name):
    return nbformat.read(os.path.join(NB_DIR, name), as_version=4)


def _code(name):
    return "\n".join(c.source for c in _read(name).cells if c.cell_type == "code")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_valid_and_clean(name):
    nb = _read(name)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "no code cell"
    for c in code:
        assert not c.get("outputs") and c.get("execution_count") is None, "the committed notebook carries outputs"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_cells_compile(name):
    for k, c in enumerate(_read(name).cells):
        if c.cell_type == "code":
            compile(c.source, f"{name}[cell {k}]", "exec")


def test_site_notebook_parameters_and_sections(monkeypatch):
    nb = _read(NOTEBOOKS[0])
    params = next(c.source for c in nb.cells if c.cell_type == "code")
    assert 'os.environ.get("NB_SITE"' in params
    assert 'assert "W3" not in WINDOWS' in params
    for v in ("NB_SITE", "NB_WINDOWS", "NB_OUT", "NB_SCORES"):
        monkeypatch.delenv(v, raising=False)
    ns = {}
    exec(compile(params, "parameters", "exec"), ns)          # the Parameters cell runs anywhere
    sites = ns["SITES"]
    assert set(sites) == {"escalda", "wadden", "ems", "ferrol", "villaviciosa"}
    assert all("W3" not in s["windows"] for s in sites.values()), "W3 is blinded before lock-A"
    monkeypatch.setenv("NB_WINDOWS", "W3")
    with pytest.raises(AssertionError):
        exec(compile(params, "parameters", "exec"), {})
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown"
                and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings if "·" in h]
    assert numbers == [str(k) for k in range(len(numbers))], f"sections not numbered 0..: {numbers}"


def test_site_notebook_calls_the_functions_directly():
    src = _code(NOTEBOOKS[0])
    for call in ("make_boundary(", "te.m2a_rasch(", "truth.best_lag(", "truth.gain_aware(",
                 "truth.m2_lag(", "aggregate.xc_band(", "aggregate.loop_band(", "hd26.l09_band(",
                 "aggregate.l10_band(", "gm20.fold_fit(", "ppl.logistic_fit(", "bt26.corr_block(",
                 "physics.bin_depths(", "physics.tau_of_s(", "lag_points.csv"):
        assert call in src, f"the site notebook does not call {call}"
    for apparatus in ("experiments.validation_v2", "queue_v2", "scoring.", "build_instance",
                      "_build_02"):
        assert apparatus not in src, f"the site notebook uses the run's apparatus ({apparatus})"


def test_run_script_present():
    sh = open(os.path.join(NB_DIR, "run_02_on_server.sh"), encoding="utf-8").read()
    assert "NB_SITE=$site" in sh and NOTEBOOKS[0] in sh and NOTEBOOKS[1] in sh
    assert "logs/v2/claude/nb02" in sh and "results/lag_nb" in sh
