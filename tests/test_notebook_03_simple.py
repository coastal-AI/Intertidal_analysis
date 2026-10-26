# -*- coding: utf-8 -*-
"""The simple notebooks of the scene-count experiment
(notebooks/validation/03_marea_vs_nidem_scene_count_site.ipynb and
03b_marea_vs_nidem_summary.ipynb) are valid nbformat v4 notebooks, carry no
outputs, and every code cell compiles. The site notebook reads its site from
NB_SITE and calls load_rtk with its default arguments only. The notebooks are
not executed here: they need the server's data (run_03_on_server.sh).

Run:  python -m pytest -q tests/test_notebook_03_simple.py
"""
import os
import re

import nbformat
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "validation")
NOTEBOOKS = ["03_marea_vs_nidem_scene_count_site.ipynb", "03b_marea_vs_nidem_summary.ipynb"]


def _read(name):
    return nbformat.read(os.path.join(NB_DIR, name), as_version=4)


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


def test_site_notebook_parameters_and_rtk():
    nb = _read(NOTEBOOKS[0])
    params = next(c.source for c in nb.cells if c.cell_type == "code")
    assert 'os.environ.get("NB_SITE"' in params
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    calls = re.findall(r"load_rtk\(([^)]*)\)", src)
    assert calls and all(a.strip() == "" for a in calls), "load_rtk must be called with its default arguments"
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown"
                and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings if "·" in h]
    assert numbers == [str(k) for k in range(len(numbers))], f"sections not numbered 0..: {numbers}"


def test_run_script_present():
    sh = open(os.path.join(NB_DIR, "run_03_on_server.sh"), encoding="utf-8").read()
    assert "NB_SITE=$site" in sh and NOTEBOOKS[0] in sh and NOTEBOOKS[1] in sh
