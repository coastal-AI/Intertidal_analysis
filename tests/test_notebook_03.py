# -*- coding: utf-8 -*-
"""notebooks/validation/_build_03_scene_count_curves.py: the notebook builds,
is a valid nbformat v4 notebook with numbered sections 0-9 and deterministic
cell ids, its code compiles, the committed notebook equals a fresh build, and
the RESUMEN_marea_vs_nidem.md parsers read a tiny sample correctly (and the
real RESUMEN when it is present). The notebook is not executed here: it needs
the stored per-draw results (see the notebook's section 0).

Run:  python -m pytest -q tests/test_notebook_03.py
"""
import ast
import importlib.util
import math
import os

import nbformat
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "notebooks", "validation", "_build_03_scene_count_curves.py")
RESUMEN = os.path.join(ROOT, "results", "scene_count_curves", "server_2026-10-26", "RESUMEN_marea_vs_nidem.md")


@pytest.fixture(scope="module")
def b():
    spec = importlib.util.spec_from_file_location("nb03_build", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEADER = ("| N | MAREA calc./K (RMSE) | MAREA cob. % | MAREA RMSEc (m) | MAREA sesgo (m) | NIDEM calc./K (RMSE) "
          "| NIDEM cob. % | NIDEM RMSEc (m) | NIDEM sesgo (m) |\n|---|---|---|---|---|---|---|---|---|\n")
SHA = {k: k * 64 for k in "abcd"}
SAMPLE = (
    "# Resumen\n\n## Tablas por sitio\n\nCómo leer: | no es una fila de tabla\n\n"
    "### Villaviciosa — 194 puntos RTK (120 píxeles); K = 10 hasta N = 100\n" + HEADER
    + "| 30 | 10/10 (10) | 87.4 | 0.418 | -0.017 | 1/10 (0) | 0.0 | — | — |\n"
    "| 40 | 10/10 (10) | 92.3 | 0.285 | +0.008 | 4/10 (3) | 0.0 | 0.215† | -0.103† |\n"
    "| 80 (no pedida; primera N con NIDEM dibujado) | 10/10 (10) | 95.6 | 0.243 | -0.030 | 9/10 (9) | 77.3 | 0.233 "
    "| -0.083 |\n"
    "| todas (465) | 1/1 (1) | 100.00 | 0.191 | -0.084 | 1/1 (1) | 100.00 | 0.231 | -0.070 |\n\n"
    "### Ems-Dollard — 100 000 píxeles Vaklodingen; K = 3 hasta N = 80\n" + HEADER
    + "| 20 (en lugar de 30) | 3/3 (3) | 32.9 | 0.727† | +0.379† | 0/3 (0) | 0.0 | — | — |\n\n"
    "Por falta de tiempo, la rejilla del Ems se redujo.\n\n"
    "## Primera N calculable\n\n| Sitio | a | b | c | d | e |\n|---|---|---|---|---|---|\n"
    "| Villaviciosa | 30 (10/10; N = 20: 3/10) | 30 (87.4 %) | 60 (7/10) | 80 (9/10) | 80 (77.3 %) |\n"
    "| Ems | 20 (3/3) | 40 (90.0 %) | 150 (2/2) | 150 (2/2) | — |\n\n"
    "## Salvedades\n\n- **Uno.** primera.\n- **Dos.** segunda.\n\n"
    "## Ficheros\n\n| Sitio | `<sitio>_refit.csv` | `<sitio>_refit_settings.json` | `<sitio>_nidem_all.csv` "
    "| `<sitio>_nidem_all_settings.json` |\n|---|---|---|---|---|\n"
    f"| villaviciosa | `{SHA['a']}` | `{SHA['b']}` | `{SHA['c']}` | `{SHA['d']}` |\n"
)


# ── the notebook ────────────────────────────────────────────────────────────

def test_build_is_valid_and_deterministic(b, tmp_path):
    nb = b.build()
    nbformat.validate(nb)
    p = tmp_path / "nb03.ipynb"
    b.write(nb, str(p))
    back = nbformat.read(str(p), as_version=4)
    nbformat.validate(back)
    assert nbformat.writes(back) == nbformat.writes(nb)
    assert nbformat.writes(b.build()) == nbformat.writes(nb)              # same bytes on every build
    assert b"\r\n" not in p.read_bytes()
    ids = [c["id"] for c in nb.cells]
    assert len(set(ids)) == len(ids) and all(i.startswith(b.CELL_ID_PREFIX) for i in ids)
    assert nb.metadata["kernelspec"]["name"] == "python3"


def test_sections_and_code(b):
    nb = b.build()
    heads = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert [h.split(".")[0] for h in heads] == [f"## {i}" for i in range(10)], heads
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(not c.get("outputs") and c.get("execution_count") is None for c in code)
    for c in code:
        ast.parse(c.source)                                               # every cell compiles
    assert b.parser_cell_source() in [c.source for c in code]             # the tested parsers are the ones used
    src = "\n".join(c.source for c in code)
    for needle in ("v14.summarize", "v15.plot_inputs", "v15.figure", "v15.MAREA_SHA256", "CURVES_CODE_ROOT",
                   "compare_rows(", "parse_first_n_table(", "draw_sha1"):
        assert needle in src, needle
    # the notebook never loads RTK data: the scores are in the CSVs
    assert "load_rtk" not in src and "pyintertidal.rtk" not in src and "import rtk" not in src


def test_committed_notebook_is_a_fresh_build(b):
    if not os.path.exists(b.NB_PATH):
        pytest.skip("notebook not written yet")
    on_disk = nbformat.read(b.NB_PATH, as_version=4)
    assert nbformat.writes(on_disk) == nbformat.writes(b.build()), "re-run the build script"


# ── the RESUMEN parsers ─────────────────────────────────────────────────────

def test_parse_value_and_cells(b):
    assert b.parse_value("—") is None
    assert b.parse_value("0.418") == {"value": 0.418, "decimals": 3, "dagger": False}
    assert b.parse_value("-0.103†") == {"value": -0.103, "decimals": 3, "dagger": True}
    assert b.parse_value("+0.008")["value"] == 0.008
    assert b.parse_value("−0.5")["value"] == -0.5
    assert b.parse_value("100.00")["decimals"] == 2
    with pytest.raises(ValueError):
        b.parse_value("0.2 m")
    assert b.parse_count("4/10 (3)") == (4, 10, 3)
    assert b.parse_n_cell("todas (465)")[:2] == (465, True)
    assert b.parse_n_cell("80 (no pedida)")[:2] == (80, False)
    assert [b.site_key(t) for t in ("Westerschelde (escalda)", "Wadden (Vlie)", "Ems-Dollard", "Sitio")] == \
        ["escalda", "wadden", "ems", None]


def test_parse_tables_sample(b):
    t = b.parse_resumen_tables(SAMPLE)
    assert sorted(t) == ["ems", "villaviciosa"]
    assert [r["N"] for r in t["villaviciosa"]] == [30, 40, 80, 465]
    assert [r["all"] for r in t["villaviciosa"]] == [False, False, False, True]
    r40 = t["villaviciosa"][1]
    assert (r40["marea"]["calc"], r40["marea"]["K"], r40["marea"]["k_rmse"]) == (10, 10, 10)
    assert (r40["nidem"]["calc"], r40["nidem"]["K"], r40["nidem"]["k_rmse"]) == (4, 10, 3)
    assert r40["nidem"]["rmse"] == {"value": 0.215, "decimals": 3, "dagger": True}
    assert r40["marea"]["bias"]["value"] == 0.008 and not r40["marea"]["bias"]["dagger"]
    assert t["villaviciosa"][0]["nidem"]["rmse"] is None and t["villaviciosa"][0]["nidem"]["bias"] is None
    assert t["villaviciosa"][3]["marea"]["cov"]["decimals"] == 2
    ems = t["ems"][0]
    assert ems["N"] == 20 and ems["marea"]["rmse"]["dagger"] and ems["nidem"]["calc"] == 0


def test_compare_rows_rounding_and_daggers(b):
    parsed = b.parse_resumen_tables(SAMPLE)["villaviciosa"]

    def meth(calc, K, k, cov, rmse, bias, dagger):
        return {"calc": calc, "K": K, "k_rmse": k, "cov": cov, "rmse": rmse, "bias": bias, "dagger": dagger}

    nan = float("nan")
    computed = [
        {"N": 30, "all": False, "marea": meth(10, 10, 10, 87.43, 0.41849, -0.0174, False),
         "nidem": meth(1, 10, 0, 0.0, nan, nan, False)},
        {"N": 40, "all": False, "marea": meth(10, 10, 10, 92.3, 0.2851, 0.0081, False),
         "nidem": meth(4, 10, 3, 0.0, 0.21549, -0.1034, True)},
        {"N": 80, "all": False, "marea": meth(10, 10, 10, 95.6, 0.243, -0.030, False),
         "nidem": meth(9, 10, 9, 77.3, 0.233, -0.083, False)},
        {"N": 465, "all": True, "marea": meth(1, 1, 1, 100.0, 0.1914, -0.0836, False),
         "nidem": meth(1, 1, 1, 100.0, 0.2312, -0.0697, False)},
    ]
    n, bad = b.compare_rows("villaviciosa", computed, parsed)
    assert n == 4 * 2 * 6 and bad == []
    computed[1]["nidem"]["rmse"] = 0.2156                                # beyond the rounding of '0.215'
    computed[1]["marea"]["dagger"] = True                                 # a dagger the RESUMEN does not have
    computed[0]["nidem"]["bias"] = -0.1                                   # a value where the RESUMEN has '—'
    computed[3]["marea"]["calc"] = 0
    n, bad = b.compare_rows("villaviciosa", computed, parsed)
    assert len(bad) == 5, bad                                             # rmse, 2 daggers (rmse, bias), '—', count
    assert any("N=40 nidem rmse" in m for m in bad) and any("N=465 marea calc" in m for m in bad)
    n, bad = b.compare_rows("villaviciosa", computed[:3], parsed)
    assert any("N=465: not computed" in m for m in bad)


def test_first_n_hashes_and_sections(b):
    assert b.parse_first_n_table(SAMPLE) == {"villaviciosa": [30, 30, 60, 80, 80], "ems": [20, 40, 150, 150, None]}
    assert b.parse_resumen_hashes(SAMPLE) == {"villaviciosa": {
        "_refit.csv": SHA["a"], "_refit_settings.json": SHA["b"], "_nidem_all.csv": SHA["c"],
        "_nidem_all_settings.json": SHA["d"]}}
    assert b.resumen_section(SAMPLE, "Salvedades") == "- **Uno.** primera.\n- **Dos.** segunda."
    assert b.resumen_section(SAMPLE, "No existe") == ""


def test_real_resumen_if_present(b):
    if not os.path.exists(RESUMEN):
        pytest.skip("RESUMEN_marea_vs_nidem.md not on this machine")
    with open(RESUMEN, encoding="utf-8") as f:
        text = f.read()
    t = b.parse_resumen_tables(text)
    assert {s: [r["N"] for r in rows] for s, rows in t.items()} == {
        "villaviciosa": [30, 40, 60, 80, 100, 150, 465], "escalda": [30, 40, 60, 100, 150, 464],
        "wadden": [30, 40, 60, 100, 150, 462], "ems": [20, 40, 80, 150, 300, 462]}
    assert all(math.isfinite(r["marea"]["cov"]["value"]) for rows in t.values() for r in rows)
    assert sorted(b.parse_first_n_table(text)) == ["ems", "escalda", "villaviciosa", "wadden"]
    h = b.parse_resumen_hashes(text)
    assert sorted(h) == ["ems", "escalda", "villaviciosa", "wadden"] and all(len(v) == 4 for v in h.values())
    assert b.resumen_section(text, "Salvedades").startswith("- **")
