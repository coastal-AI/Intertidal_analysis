# -*- coding: utf-8 -*-
"""The bed-change notebook (notebooks/validation/04_bed_change_and_window_drift_site.ipynb) is a plain notebook.

* it is a valid notebook without outputs and every code cell compiles;
* it never mentions the field survey of the Spanish site (only the Vaklodingen is read);
* the site comes from the Parameters cell, overridable by NB_SITE, and the sections are numbered 0..;
* its time-aware functions give the right numbers on small synthetic cases: the weighted sampled
  exceedance (equal weights = the Hazen quantile, a zero weight drops the look, the step-world bracket holds
  whatever the weights) and the CUSUM break (found where the bed moves, not where it stays).

The notebook is not executed here: it needs the server's data (run_04_on_server.sh).

Run:  python -m pytest -q tests/test_notebook_04_simple.py   (or python tests/test_notebook_04_simple.py)
"""
import ast
import os
import re

import nbformat
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "validation")
NB = os.path.join(NB_DIR, "04_bed_change_and_window_drift_site.ipynb")
ENV = ("NB_SITE", "NB_NS", "NB_OUT", "NB_NTABLES", "NB_YEARS")


def cells():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    return nb.cells


def code_cells():
    return [c.source for c in cells() if c.cell_type == "code"]


def notebook_function(name, **names):
    """A function defined in a code cell of the notebook, compiled alone with the given globals."""
    for src in code_cells():
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                ns = dict(np=np, **names)
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"notebook:{name}", "exec"), ns)
                return ns[name]
    raise LookupError(name)


def parameters(env=None):
    """The Parameters cell and the site cell run with a clean environment (plus `env`)."""
    saved = {k: os.environ.pop(k) for k in ENV if k in os.environ}
    try:
        os.environ.update(env or {})
        ns = {}
        first, sites = code_cells()[:2]
        exec(compile(first, "parameters", "exec"), ns)
        exec(compile(sites, "sites", "exec"), ns)
        return ns
    finally:
        for k in ENV:
            os.environ.pop(k, None)
        os.environ.update(saved)


def test_valid_clean_and_compiles():
    for k, c in enumerate(cells()):
        if c.cell_type == "code":
            assert not c.get("outputs") and c.get("execution_count") is None, "the notebook carries outputs"
            compile(c.source, f"cell {k}", "exec")


def test_only_the_vaklodingen_is_read():
    for c in cells():
        assert "rtk" not in c.source.lower(), "a cell mentions the field survey of the Spanish site"


def test_parameters_and_sections():
    assert 'os.environ.get("NB_SITE"' in code_cells()[0]
    ns = parameters()
    assert ns["SITE"] == "escalda" and ns["NS"] == [40, 60] and ns["SURVEY_YEARS"] == [2023, 2024, 2025]
    assert ns["END"] == "2025-12-31" and ns["ROW"] == "escalda"
    ns = parameters({"NB_SITE": "wadden"})
    assert ns["NS"] == [40, 60] and ns["ROW"] == "wadden-2025"
    ns = parameters({"NB_SITE": "ems"})
    assert ns["NS"] == [40, 50], "60 images are not reachable at the Ems"
    headings = [c.source.splitlines()[0] for c in cells() if c.cell_type == "markdown" and c.source.startswith("## ")]
    numbers = [h.split("·")[0].strip("# ").strip() for h in headings if "·" in h]
    assert numbers == [str(k) for k in range(len(numbers))], f"sections not numbered 0..: {numbers}"


def test_calls_the_method_functions_directly():
    src = "\n".join(code_cells())
    for call in ("marea.invert_series(", "te.m2a_rasch(", 'fm.METHODS[mid][1](', "sample_truth(",
                 "elev_freq.sampled_exceedance_block(", "make_boundary("):
        assert call in src, call


def _random_case(seed, T=40, P=300):
    rng = np.random.default_rng(seed)
    h = np.sort(rng.uniform(-2, 2, T))[rng.permutation(T)]           # distinct levels, not in time order
    V = rng.random((T, P)) < 0.7
    V[:, 0] = False                                                  # a pixel without a valid look
    V[:, 1] = False
    V[3, 1] = True                                                   # a pixel with one valid look
    Wb = V & (rng.random((T, P)) < rng.uniform(0, 1, P))
    return rng, h, V, Wb


def test_weighted_exceedance_equal_weights_is_hazen():
    wex = notebook_function("weighted_exceedance")
    _, h, V, Wb = _random_case(1)
    z = wex(V, Wb, h, np.ones((len(h), 1)))
    assert np.isnan(z[0]) and z[1] == h[3]
    for p in range(2, V.shape[1]):
        lev = h[V[:, p]]
        q = 1 - Wb[V[:, p], p].mean()
        assert np.isclose(z[p], np.quantile(lev, q, method="hazen")), p


def test_weighted_exceedance_zero_weight_drops_the_look():
    wex = notebook_function("weighted_exceedance")
    rng, h, V, Wb = _random_case(2)
    keep = rng.random(V.shape) < 0.6
    z = wex(V, Wb, h, keep.astype(float))
    z_sub = wex(V & keep, Wb & keep, h, np.ones((len(h), 1)))
    assert np.array_equal(np.isnan(z), np.isnan(z_sub)) and np.allclose(z, z_sub, equal_nan=True)


def test_weighted_exceedance_step_world_bracket_any_weights():
    wex = notebook_function("weighted_exceedance")
    rng, h, V, _ = _random_case(3)
    zbed = rng.uniform(-1.5, 1.5, V.shape[1])
    Wb = V & (h[:, None] > zbed[None, :])                            # wet exactly when the level is above the bed
    for w in (np.ones((len(h), 1)), np.exp(-rng.uniform(0, 5, (len(h), 1))), rng.uniform(0.01, 1, V.shape)):
        z = wex(V, Wb, h, w)
        for p in range(2, V.shape[1]):
            dry, wet = h[V[:, p] & ~Wb[:, p]], h[Wb[:, p]]
            lo = dry.max() if len(dry) else h[V[:, p]].min()
            hi = wet.min() if len(wet) else h[V[:, p]].max()
            assert lo - 1e-12 <= z[p] <= hi + 1e-12, p


def test_weighted_exceedance_near_e5c():
    from experiments.fixedcfg.elev_freq import sampled_exceedance_block
    wex = notebook_function("weighted_exceedance")
    _, h, V, Wb = _random_case(4)
    z, z0 = wex(V, Wb, h, np.ones((len(h), 1))), sampled_exceedance_block(V, Wb, h)
    assert np.array_equal(np.isnan(z), np.isnan(z0))
    for p in range(2, V.shape[1]):                                   # within one step of the pixel's sorted levels
        lev = np.sort(h[V[:, p]])
        step = np.max(np.diff(lev)) if len(lev) > 1 else 0.0
        assert abs(z[p] - z0[p]) <= step + 1e-12


def test_cusum_break_where_the_bed_moves():
    cusum = notebook_function("cusum_break", CUSUM_CRIT=1.36, MIN_AFTER=10)
    from experiments.fixedcfg.elev_freq import sampled_exceedance_block
    rng = np.random.default_rng(5)
    T, P = 60, 400
    h = rng.permutation(np.linspace(-2, 2, T))                       # levels spread over the range, in random order
    V = rng.random((T, P)) < 0.85
    bed = np.full((T, P), -1.0)
    bed[:30, :200] = 1.0                                             # pixels 0..199: the bed drops 2 m after look 29
    Wb = (V & (h[:, None] > bed)) ^ (V & (rng.random((T, P)) < 0.02))   # and a little classification noise
    z = sampled_exceedance_block(V, Wb, h)                           # the static estimate (E5c)
    k, stat = cusum(V, Wb, h[:, None], z)
    moved, still = k[:200], k[200:]
    assert np.mean(moved >= 0) > 0.8, "the break is missed where the bed moved"
    assert np.median(np.abs(moved[moved >= 0] - 29)) <= 3, "the break is not where the bed moved"
    assert np.mean(still >= 0) < 0.15, "too many breaks where the bed stayed"
    for p in np.flatnonzero(k >= 0):                                 # at least MIN_AFTER valid looks follow
        assert V[k[p] + 1:, p].sum() >= 10


def test_cusum_no_break_without_misfit():
    cusum = notebook_function("cusum_break", CUSUM_CRIT=1.36, MIN_AFTER=10)
    h = np.linspace(-1, 1, 50)
    V = np.ones((50, 3), bool)
    z = np.array([-0.5, 0.0, 0.5])
    Wb = h[:, None] > z[None, :]
    k, stat = cusum(V, Wb, h[:, None], z)
    assert np.all(k == -1)


def test_run_script():
    sh = open(os.path.join(NB_DIR, "run_04_on_server.sh"), encoding="utf-8").read()
    assert "NB_SITE=$s" in sh and os.path.basename(NB) in sh
    assert re.search(r"SITES=\$\{\*:-escalda wadden ems\}", sh), "the Westerschelde runs first, then the Wadden"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
