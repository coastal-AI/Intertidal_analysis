# -*- coding: utf-8 -*-
"""Builds notebooks/validation/03_scene_count_curves_marea_nidem.ipynb.

The notebook is the reproducible record of the scene-count experiment of
2026-10-20/26: "MAREA, una curva, NIDEM otra, y ver cómo responde cada uno
ante meter distintos números de imágenes". It reads only the stored per-draw
results of experiments/v14_scene_count_curves.py (MAREA) and
experiments/v15_nidem_scene_curves.py (NIDEM), checks their hashes, gates and
draws, recomputes the summaries with v14's and v15's own functions, asserts
that they equal the verified summary RESUMEN_marea_vs_nidem.md, and redraws
the published figure (plus a bias panel). Nothing heavy is recomputed.

The RESUMEN parsers below are ordinary functions (tested by
tests/test_notebook_03.py); their source is copied into the notebook, which
therefore needs nothing from this file to run.

Run from the repository root:
    python -X utf8 notebooks/validation/_build_03_scene_count_curves.py
        writes the notebook (no outputs; deterministic cell ids)
    python -X utf8 notebooks/validation/_build_03_scene_count_curves.py \\
        --execute results/scene_count_curves/03_scene_count_curves_marea_nidem.executed.ipynb --html
        executes the notebook as it is on disk with nbclient (working directory
        = repository root) into that path, plus an HTML export next to it
        (needs nbconvert). The environment variables CURVES_CODE_ROOT,
        CURVES_FIG_DIR, CURVES_RESUMEN and CURVES_REPO_ROOT reach the kernel.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import math
import os
import platform
import re
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB_NAME = "03_scene_count_curves_marea_nidem.ipynb"
NB_PATH = os.path.join(HERE, NB_NAME)
CELL_ID_PREFIX = "nb03"


# ─────────────────────────────────────────────────────────────────────────────
#  RESUMEN_marea_vs_nidem.md parsers (copied verbatim into the notebook)
# ─────────────────────────────────────────────────────────────────────────────

def site_key(text):
    """The site key named by a RESUMEN heading or table cell, or None."""
    t = str(text).lower()
    for key, words in (("villaviciosa", ("villaviciosa",)), ("escalda", ("escalda", "westerschelde")),
                       ("wadden", ("wadden",)), ("ems", ("ems",))):
        if any(w in t for w in words):
            return key
    return None


def parse_value(cell):
    """'0.418', '+0.008', '-0.103†', '87.4', '100.00' -> dict(value, decimals,
    dagger); '—' (no draw with an RMSE) -> None."""
    c = str(cell).strip()
    if c in ("—", "–", "-", ""):
        return None
    m = re.match(r"^([+\-−]?\d+(?:\.(\d+))?)\s*(†)?$", c)
    if not m:
        raise ValueError(f"not a table value: {cell!r}")
    return {"value": float(m.group(1).replace("−", "-")), "decimals": len(m.group(2) or ""),
            "dagger": bool(m.group(3))}


def parse_count(cell):
    """'10/10 (10)' -> (computable, K, with an RMSE)."""
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*\(\s*(\d+)\s*\)\s*$", str(cell))
    if not m:
        raise ValueError(f"not a count cell: {cell!r}")
    return tuple(int(g) for g in m.groups())


def parse_n_cell(cell):
    """'30', '80 (no pedida; ...)', 'todas (465)' -> (N, whole archive?, label)."""
    c = str(cell).strip()
    m = re.match(r"^(todas|all)\s*\((\d+)\)", c, flags=re.IGNORECASE)
    if m:
        return int(m.group(2)), True, c
    m = re.match(r"^(\d+)", c)
    if not m:
        raise ValueError(f"not an N cell: {cell!r}")
    return int(m.group(1)), False, c


def parse_resumen_tables(text):
    """The per-site tables of the RESUMEN ('## Tablas por sitio', one '###'
    heading per site): {site: [row, ...]} in file order, row = dict(N, all,
    label, marea=..., nidem=...), each method = dict(calc, K, k_rmse, cov,
    rmse, bias) with cov/rmse/bias as :func:`parse_value` returns them."""
    out, site, inside = {}, None, False
    for line in str(text).splitlines():
        s = line.strip()
        if s.startswith("## "):
            inside = s[3:].strip().lower().startswith(("tablas por sitio", "per-site tables"))
            site = None
            continue
        if not inside:
            continue
        if s.startswith("### "):
            site = site_key(s[4:].split("—")[0])
            if site is not None:
                out.setdefault(site, [])
            continue
        if site is None or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 9 or cells[0] == "N" or set(cells[0]) <= set("-: "):
            continue
        N, whole, label = parse_n_cell(cells[0])
        row = {"N": N, "all": whole, "label": label}
        for meth, j in (("marea", 1), ("nidem", 5)):
            calc, K, k_rmse = parse_count(cells[j])
            row[meth] = {"calc": calc, "K": K, "k_rmse": k_rmse, "cov": parse_value(cells[j + 1]),
                         "rmse": parse_value(cells[j + 2]), "bias": parse_value(cells[j + 3])}
        out[site].append(row)
    return out


def parse_first_n_table(text):
    """The '## Primera N calculable' table: {site: [N of each criterion]} (the
    leading integer of each cell, None for a cell without one), columns in
    file order (MAREA with RMSE, MAREA drawn, NIDEM computable, NIDEM with
    RMSE, NIDEM drawn)."""
    out, inside = {}, False
    for line in str(text).splitlines():
        s = line.strip()
        if s.startswith("## "):
            inside = s[3:].strip().lower().startswith(("primera n calculable", "first computable n"))
            continue
        if not inside or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        key = site_key(cells[0])
        if key is None:
            continue
        vals = []
        for c in cells[1:]:
            m = re.match(r"^(\d+)", c)
            vals.append(int(m.group(1)) if m else None)
        out[key] = vals
    return out


def parse_resumen_hashes(text):
    """The file table of the RESUMEN ('| Sitio | `<sitio>_refit.csv` | ...'):
    {site: {suffix: sha256}}, suffix e.g. '_refit.csv'."""
    out, suffixes = {}, None
    for line in str(text).splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if any("<sitio>" in c or "<site>" in c for c in cells):
            suffixes = [re.sub(r"^`?<(sitio|site)>", "", c).strip("`") for c in cells[1:]]
            continue
        if suffixes is None:
            continue
        key = site_key(cells[0])
        shas = [re.sub(r"[`\s]", "", c) for c in cells[1:]]
        if key is None or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in shas):
            continue
        out[key] = dict(zip(suffixes, shas))
    return out


def resumen_section(text, title):
    """The body of the '## <title>' section (up to the next '## '), stripped;
    '' when there is no such section."""
    lines, inside = [], False
    for line in str(text).splitlines():
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().lower().startswith(str(title).lower())
            continue
        if inside:
            lines.append(line)
    return "\n".join(lines).strip()


def compare_rows(site, computed, parsed):
    """Cell-by-cell comparison of a computed per-site table with the parsed
    RESUMEN table. ``computed`` rows: dict(N, all, marea=..., nidem=...),
    each method = dict(calc, K, k_rmse, cov [%], rmse [m or NaN], bias [m or
    NaN], dagger). Counts must be equal; values equal within the rounding of
    the RESUMEN cell (half a unit of its last decimal); '—' iff no value;
    the dagger (median not drawn by the figure) must agree. Returns
    (cells compared, [mismatch messages])."""
    bad, n = [], 0
    by_n = {int(r["N"]): r for r in computed}
    for p in parsed:
        c = by_n.get(int(p["N"]))
        where = f"{site} N={p['N']}"
        if c is None:
            bad.append(f"{where}: not computed")
            continue
        if bool(c["all"]) != bool(p["all"]):
            bad.append(f"{where}: whole archive? RESUMEN {p['all']}, computed {c['all']}")
        for meth in ("marea", "nidem"):
            pc, cc = p[meth], c[meth]
            for k in ("calc", "K", "k_rmse"):
                n += 1
                if int(pc[k]) != int(cc[k]):
                    bad.append(f"{where} {meth} {k}: RESUMEN {pc[k]}, computed {cc[k]}")
            for k in ("cov", "rmse", "bias"):
                n += 1
                pv, cv = pc[k], float(cc[k])
                if pv is None:
                    if math.isfinite(cv):
                        bad.append(f"{where} {meth} {k}: RESUMEN '—', computed {cv:.6g}")
                    continue
                if not math.isfinite(cv):
                    bad.append(f"{where} {meth} {k}: RESUMEN {pv['value']}, computed no value")
                    continue
                tol = 0.5 * 10.0 ** (-pv["decimals"]) + 1e-9
                if abs(cv - pv["value"]) > tol:
                    bad.append(f"{where} {meth} {k}: RESUMEN {pv['value']}, computed {cv:.6g} (tolerance {tol:.2g})")
                if k != "cov" and bool(pv["dagger"]) != bool(cc["dagger"]):
                    bad.append(f"{where} {meth} {k}: dagger RESUMEN {pv['dagger']}, computed {cc['dagger']}")
    return n, bad


PARSERS = (site_key, parse_value, parse_count, parse_n_cell, parse_resumen_tables, parse_first_n_table,
           parse_resumen_hashes, resumen_section, compare_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  the notebook
# ─────────────────────────────────────────────────────────────────────────────

TITLE = "# 03 · Scene-count curves: MAREA and NIDEM against the number of scenes"

MD_HEADER = """\
## 0. Header

**Question** (2026-10-20): *"MAREA, una curva, NIDEM otra, y ver cómo responde cada uno ante meter distintos
números de imágenes"*: one curve for MAREA and one for NIDEM, showing how each responds to being given
different numbers of images.

**Design.** At each site v14 drew random subsets of N scenes from the whole 2023–2025 archive: the pool `all`,
465 / 464 / 462 / 462 scenes at Villaviciosa / Westerschelde / Wadden / Ems-Dollard, cloudy scenes included,
with the per-pixel clear-sky mask SCL 4–7. On every draw:

- **MAREA** (`experiments/v14_scene_count_curves.py`, `--lags refit`): the production inversion with the lags
  re-fitted on the draw and no quality rule (mb 0), as published;
- **NIDEM** (`experiments/v15_nidem_scene_curves.py`): the published filtered product with ITEM's cloud mask, on
  exactly the same draws (the same `draw_sha1`).

Every draw is scored on one fixed set per site: the 194 RTK points at Villaviciosa (already scored inside the
CSVs; this notebook never loads RTK data) and a fixed subsample of 100,000 Vaklodingen pixels at the Dutch sites.

**What this is not.** This is *not* the n-tables design (v19: the n most recent screened images ending at the
truth date, several methods). Here the scenes are random draws from the whole unscreened archive, and there are
only two methods.

**What the notebook does.** It recomputes nothing heavy. It reads the stored per-draw CSVs and their settings,
checks hashes, gates and draw identity, recomputes the summaries with v14's and v15's own functions
(`v14.summarize`, `v15.plot_inputs`, `v15.figure`), asserts that they equal the verified summary
`RESUMEN_marea_vs_nidem.md`, and redraws the published figure plus a bias panel.

**How to reproduce** (server, repository root, after `source env.sh`):

1. MAREA, git 8b4737d, 2026-10-20 (the commands recorded in each `<site>_refit_settings.json`, printed in
   section 1.3): `"$PY" -X utf8 -m experiments.v14_scene_count_curves run <site> --lags refit --reps 10
   --max-minutes 45` at Villaviciosa and `--reps 5` at the Dutch sites. Outputs
   `results/scene_count_curves/<site>_refit.csv` and `<site>_refit_settings.json`.
2. NIDEM, per site: `bash experiments/runs/run_v15_2026-10-26.sh [site ...]`, which runs
   `"$PY" -X utf8 -m experiments.v15_nidem_scene_curves run <site> --workers W --io-lock ... --max-minutes 100`
   for each site (resumable). Outputs `<site>_nidem_all.csv` and `<site>_nidem_all_settings.json`.
3. This notebook, executed on the server with nbclient:
   `"$PY" -X utf8 notebooks/validation/_build_03_scene_count_curves.py --execute
   results/scene_count_curves/03_scene_count_curves_marea_nidem.executed.ipynb --html`.

**Environment variables.** `CURVES_CODE_ROOT`: import `experiments` from a clean copy (for example
`git archive 82121c9 experiments`) instead of the repository. `CURVES_FIG_DIR`: the figure folder (default
`results/scene_count_curves/figs_<date>/`). `CURVES_RESUMEN`: the path of the RESUMEN. `CURVES_REPO_ROOT`: the
repository root (default: found from the working directory)."""

CODE_SETUP = '''\
import ast
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys

T_START = dt.datetime.now(dt.timezone.utc)


def find_repo(start):
    """The first folder at or above ``start`` holding experiments/v14_scene_count_curves.py and pyintertidal/."""
    d = os.path.abspath(start)
    while True:
        if (os.path.isfile(os.path.join(d, "experiments", "v14_scene_count_curves.py"))
                and os.path.isdir(os.path.join(d, "pyintertidal"))):
            return d
        up = os.path.dirname(d)
        if up == d:
            raise SystemExit(f"no repository root at or above {start!r}: set CURVES_REPO_ROOT")
        d = up


REPO = os.path.abspath(os.environ.get("CURVES_REPO_ROOT") or find_repo(os.getcwd()))
#: where the experiments package is imported from: the repository, or (CURVES_CODE_ROOT) a clean copy of it
CODE_ROOT = os.path.abspath(os.environ.get("CURVES_CODE_ROOT") or REPO)
for _p in (REPO, CODE_ROOT):                     # CODE_ROOT ends up first, the repository (pyintertidal) next
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.chdir(REPO)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
import matplotlib                                # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402
from IPython.display import Image, Markdown, display  # noqa: E402

from experiments import v14_scene_count_curves as v14  # noqa: E402
from experiments import v15_nidem_scene_curves as v15  # noqa: E402

os.chdir(REPO)                                   # v14 and v15 chdir to their own root when imported


def same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


for _m in (v14, v15):
    assert same_path(os.path.dirname(os.path.dirname(_m.__file__)), CODE_ROOT), _m.__file__

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.max_rows", 200)


def rel(path):
    """``path`` relative to the repository (forward slashes) when inside it, else absolute."""
    a = os.path.abspath(path)
    try:
        r = os.path.relpath(a, REPO)
    except ValueError:
        return a.replace(os.sep, "/")
    return a.replace(os.sep, "/") if r.startswith("..") else r.replace(os.sep, "/")


READ, WRITTEN = {}, {}


def note(registry, path):
    """sha256 of ``path``, recorded in ``registry`` (READ or WRITTEN) with its size."""
    h = v15.sha256_file(path)
    registry[rel(path)] = {"bytes": os.path.getsize(path), "sha256": h}
    return h


def git(*args, raw=False):
    """Output of ``git -C REPO args`` (bytes with ``raw``), or None when git is unavailable or fails."""
    try:
        out = subprocess.run(["git", "-C", REPO, *args], capture_output=True, check=True).stdout
    except Exception:                            # noqa: BLE001
        return None
    return out if raw else out.decode("utf-8", "replace").strip()


print(f"repository : {REPO}")
print(f"code root  : {CODE_ROOT}" + ("  (the repository)" if same_path(CODE_ROOT, REPO) else "  (a copy)"))
print(f"v14        : {v14.__file__}")
print(f"v15        : {v15.__file__}")
print(f"started    : {T_START.isoformat(timespec='seconds')}")'''

MD_PARSERS = """\
The parsers of `RESUMEN_marea_vs_nidem.md` used in sections 1, 4, 5 and 8 (copied from
`notebooks/validation/_build_03_scene_count_curves.py`, where `tests/test_notebook_03.py` checks them)."""

MD_SETTINGS = """\
## 1. Settings and provenance

### 1.1 Inputs
Each site's files are looked for first in the server's own output folder (`results/scene_count_curves/`), then
in the laptop copies: MAREA in `server_2026-10-20/`; NIDEM of Villaviciosa and Westerschelde in
`server_2026-10-20/`, of Wadden and Ems in `server_2026-10-26/` (the complete runs of 2026-10-26). The partial
Wadden/Ems NIDEM files left in `server_2026-10-20/` are never candidates. Section 1.2 pins every file by sha256."""

CODE_INPUTS = '''\
OUT = v14.OUT                                    # results/scene_count_curves
SITES = list(v14.SITE_ORDER)                     # villaviciosa, escalda, wadden, ems
NAMES = v15.TXT["en"]["names"]
CANDIDATES = {
    "marea": {s: [OUT, f"{OUT}/server_2026-10-20"] for s in SITES},
    "nidem": {"villaviciosa": [OUT, f"{OUT}/server_2026-10-20"], "escalda": [OUT, f"{OUT}/server_2026-10-20"],
              "wadden": [OUT, f"{OUT}/server_2026-10-26"], "ems": [OUT, f"{OUT}/server_2026-10-26"]},
}
SUFFIX = {"marea": ("_refit.csv", "_refit_settings.json"), "nidem": ("_nidem_all.csv", "_nidem_all_settings.json")}
RESUMEN_CANDIDATES = [os.environ.get("CURVES_RESUMEN") or "", f"{OUT}/server_2026-10-26/RESUMEN_marea_vs_nidem.md",
                      f"{OUT}/RESUMEN_marea_vs_nidem.md"]

INPUTS, _rows = {}, []
for meth in ("marea", "nidem"):
    for s in SITES:
        d = v15.find_inputs(s, CANDIDATES[meth][s], SUFFIX[meth])
        assert d is not None, f"{s} {meth}: none of {CANDIDATES[meth][s]} holds {s}{SUFFIX[meth]}"
        INPUTS[(s, meth)] = {"dir": d, "csv": os.path.join(d, s + SUFFIX[meth][0]),
                             "settings": os.path.join(d, s + SUFFIX[meth][1])}
        for kind in ("csv", "settings"):
            _rows.append({"site": s, "method": meth.upper(), "file": kind, "path": rel(INPUTS[(s, meth)][kind])})
RESUMEN = next((p for p in RESUMEN_CANDIDATES if p and os.path.isfile(p)), None)
assert RESUMEN is not None, f"RESUMEN_marea_vs_nidem.md not found in {RESUMEN_CANDIDATES} (set CURVES_RESUMEN)"
display(pd.DataFrame(_rows).set_index(["site", "method", "file"]))
print("RESUMEN:", rel(RESUMEN))'''

MD_HASHES = """\
### 1.2 Hashes
The MAREA files must have the sha256 of `v15.MAREA_SHA256` (the copies the NIDEM runs were gated against), the
NIDEM files those listed in the RESUMEN, and every NIDEM settings file must name the MAREA CSV it was gated
against (`compat.marea_csv_sha256`)."""

CODE_HASHES = '''\
#: sha256 of the NIDEM CSVs and settings used here (RESUMEN_marea_vs_nidem.md, "Ficheros")
NIDEM_SHA256 = {
    "villaviciosa": ("5897045b8686cb128fb5503c58b456d6f88c8dab533feaff9eaf845f3d4bf536",
                     "ee8fbc910d2f3f01995aa8011b48b3562073ce89cbbd53764a2e3b14498fe42a"),
    "escalda": ("fcf6b1100a3b23ba8726976f428212c511a7720358c0d71ecd9c1f2d5e059a4d",
                "e485f54b47320bdac00c77b632a21fdc399d7b7422a5699e6540d80e7c56fe53"),
    "wadden": ("3e80a2dacaa6e5f4a18220d65a8d92595d506e5328af8004da36670d33fb7a69",
               "3d5774b532bf4b4ffad1976a6c3a7eaa0664fd9345aa08eb113a83ee0ee74ced"),
    "ems": ("715a5c7ec5de1a29987338fee3ad03d3b8be2e9fa453481eaa1203c6f9f348b5",
            "0cc954933ccce09f10623e7c13628b12137a238eed38619e03b06929919e4aee"),
}
with open(RESUMEN, encoding="utf-8") as _f:
    RES_TEXT = _f.read()
RESUMEN_SHA256 = note(READ, RESUMEN)
RES_HASHES = parse_resumen_hashes(RES_TEXT)

_rows = []
for (s, meth), f in INPUTS.items():
    want = (v15.MAREA_SHA256 if meth == "marea" else NIDEM_SHA256)[s]
    for i, kind in enumerate(("csv", "settings")):
        got = note(READ, f[kind])
        _rows.append({"site": s, "method": meth.upper(), "file": os.path.basename(f[kind]), "sha256": got,
                      "expected": got == want[i], "listed in RESUMEN": RES_HASHES.get(s, {}).get(SUFFIX[meth][i]) == got})
HASHES = pd.DataFrame(_rows)
display(HASHES.set_index(["site", "method", "file"]))
assert HASHES["expected"].all(), "an input file is not the expected copy"
assert HASHES["listed in RESUMEN"].all(), "an input file is not the one listed in the RESUMEN"

SET = {}
for key, f in INPUTS.items():
    with open(f["settings"], encoding="utf-8") as _f:
        SET[key] = json.load(_f)
for s in SITES:
    assert SET[(s, "nidem")]["compat"]["marea_csv_sha256"] == v15.MAREA_SHA256[s][0], s
print(f"{len(HASHES)} input files as expected; every NIDEM run was gated against the MAREA CSV used here. "
      f"RESUMEN sha256 {RESUMEN_SHA256}")'''

MD_CODE = """\
### 1.3 Code
The git state, the sha256 of the imported v14 and v15, and whether each function or constant this notebook uses
is the same text as at commit 82121c9 (the commit of the runs of 2026-10-26 and of the published figures). v14
must be the script that wrote the MAREA rows. Then the run records: which command and which v15 wrote which
rows."""

CODE_CODE = '''\
REF_COMMIT = "82121c935bdc81df0d70b5a9f9c4fb911c46cc02"
V15_KNOWN = {
    "370ebf8117b196b0c964a64b14ef2f4df705d9778bab8bfb1f98b07dc44c1475":
        "committed in 82121c9; drew the published figures and the NIDEM rows of 2026-10-26",
    "6d682cbaf0891e15d2c8f137111c316c373a3b08e93d8e0d0adc3639e5f22584":
        "uncommitted v15 of 2026-10-20; the earlier NIDEM rows",
}
print("HEAD    :", git("rev-parse", "HEAD"), "| branch", git("rev-parse", "--abbrev-ref", "HEAD"))
print("commit  :", git("log", "-1", "--format=%h %ad %s", "--date=iso"))


def blob_sha256(rev, path):
    b = git("cat-file", "blob", f"{rev}:{path}", raw=True)
    return hashlib.sha256(b).hexdigest() if b is not None else None


_rows, SOURCES = [], {}
for name, mod, path in (("v14", v14, "experiments/v14_scene_count_curves.py"),
                        ("v15", v15, "experiments/v15_nidem_scene_curves.py")):
    got = note(READ, mod.__file__)
    with open(mod.__file__, encoding="utf-8") as _f:
        SOURCES[name] = _f.read()
    st = git("status", "--porcelain", "--", path)
    _rows.append({"module": name, "imported from": rel(mod.__file__), "sha256": got,
                  "= HEAD": got == blob_sha256("HEAD", path), "= 82121c9": got == blob_sha256(REF_COMMIT, path),
                  "repository working copy": "unavailable" if st is None else (st or "clean")})
display(pd.DataFrame(_rows).set_index("module"))
assert _rows[0]["sha256"] == v15.V14_SHA256, "v14 is not the script that wrote the MAREA rows"
print("v15:", V15_KNOWN.get(_rows[1]["sha256"], "not a v15 that wrote rows; see the function table below"))


def top_level_sources(src):
    """{name: source text} of the top-level functions, classes and assignments of a module source."""
    tree, out = ast.parse(src), {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    out[t.id] = ast.get_source_segment(src, node)
    return out


USED = {"v14": ["OUT", "SITE_ORDER", "POOL_IDS", "LARGE_N", "PAL", "LOW_COVERAGE", "site_id", "draw", "draw_sha1",
                "reps_for", "make_jobs", "summarize", "guard_out", "_style_axes", "_log_n_axis"],
        "v15": ["V14_SHA256", "MAREA_SHA256", "NOT_COMPUTABLE", "STYLE", "TXT", "LAYOUT", "sha256_file",
                "find_inputs", "marea_rows", "plan_from_settings", "_pick_marea_csv", "plot_inputs", "_fmt_int",
                "_draw_site", "figure"]}
_rows = []
for name, path in (("v14", "experiments/v14_scene_count_curves.py"), ("v15", "experiments/v15_nidem_scene_curves.py")):
    mine = top_level_sources(SOURCES[name])
    b = git("cat-file", "blob", f"{REF_COMMIT}:{path}", raw=True)
    ref = top_level_sources(b.decode("utf-8")) if b is not None else None
    for obj in USED[name]:
        assert obj in mine, f"{name}.{obj} not found"
        _rows.append({"module": name, "name": obj, "same text as 82121c9":
                      "unknown (no git)" if ref is None else mine[obj] == ref.get(obj)})
FUNCS = pd.DataFrame(_rows)
_same = FUNCS["same text as 82121c9"].map(lambda v: v is True)
print(f"functions and constants used here with the same text as at 82121c9: {int(_same.sum())} of {len(FUNCS)}")
if not _same.all():
    display(FUNCS[~_same].set_index(["module", "name"]))

# the run records: MAREA (v14) and NIDEM (v15)
_rows = []
for s in SITES:
    ms, ns = SET[(s, "marea")], SET[(s, "nidem")]
    for r in ms.get("runs", []):
        _rows.append({"site": s, "method": "MAREA", "started": r.get("started"), "finished": r.get("finished"),
                      "status": r.get("status"), "argv": " ".join(r.get("argv", [])), "git": ms.get("git_head", "")[:7],
                      "script sha256": ms["script"]["sha256"][:12]})
    runs = ns.get("runs", [])
    for i, r in enumerate(runs):
        sha = r.get("script_sha256") or (ns["script"]["sha256"] if len(runs) == 1 else "not recorded")
        _rows.append({"site": s, "method": "NIDEM", "started": r.get("started"), "finished": r.get("finished"),
                      "status": r.get("status"), "argv": " ".join(r.get("argv", [])),
                      "git": ns.get("git_head", "")[:7] if i == len(runs) - 1 else "",
                      "script sha256": sha[:12]})
RUNS = pd.DataFrame(_rows)
display(RUNS.set_index(["site", "method"]))
for s in SITES:
    p = SET[(s, "nidem")]["progress"]
    print(f"{s:12s} NIDEM rows {p['rows']} of {p['jobs']} jobs; appended by the last run: {p['done_this_run']}")'''

MD_ENV = """\
### 1.4 Environment"""

CODE_ENV = '''\
def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:                            # noqa: BLE001
        return None


ENV = {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
       "executable": sys.executable, "numpy": np.__version__, "pandas": pd.__version__,
       "matplotlib": matplotlib.__version__, "backend": matplotlib.get_backend(),
       "nbformat": _version("nbformat"), "nbclient": _version("nbclient"), "ipykernel": _version("ipykernel"),
       "repository": REPO, "code root": CODE_ROOT, "working directory": os.getcwd(),
       "started (UTC)": T_START.isoformat(timespec="seconds")}
ENV.update({k: v for k, v in os.environ.items() if k.startswith("CURVES_")})
for k, v in ENV.items():
    print(f"{k:>18}: {v}")
print("\\nversions recorded by the NIDEM runs (server):", SET[("wadden", "nidem")].get("versions"))'''

MD_GATES = """\
## 2. Gates
Every gate in the eight settings files, with its status. A failed stopping gate would have stopped its run
before any curve row; `not checked` gates carry their reason (at Ems the primary NIDEM product does not exist).
`bitwise` is reported by the product gates (`False` means passed within the stated tolerance)."""

CODE_GATES = '''\
_SKIP = {"ok", "path", "policy", "rule", "expected", "found", "source", "subset"}


def gate_status(g):
    ok = g.get("ok") if isinstance(g, dict) else g
    if ok is True:
        return "PASS"
    if ok is None and isinstance(g, dict) and g.get("checked") is False:
        return "not checked"
    if ok is False and isinstance(g, dict) and g.get("stop_if_failed") is False:
        return "FAIL (reported, not a stop)"
    return "FAIL" if ok is False else f"unknown ({ok!r})"


def gate_detail(g):
    if not isinstance(g, dict):
        return ""
    parts = []
    if isinstance(g.get("found"), str) and isinstance(g.get("expected"), str):
        parts.append(f"sha256 {g['found'][:12]}... {'=' if g['found'] == g['expected'] else '!='} expected")
    for k, v in g.items():
        if isinstance(v, dict) and isinstance(v.get("found"), str) and isinstance(v.get("expected"), str):
            parts.append(f"{k} {v['found'][:12]}... {'=' if v['found'] == v['expected'] else '!='} expected")
    if isinstance(g.get("mine"), dict) and isinstance(g.get("v14"), dict):
        parts.append(f"mine = v14: {g['mine'] == g['v14']}")
    for k, v in g.items():
        if k in _SKIP or isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    tol = g.get("tolerance")
    if isinstance(tol, dict):
        parts += [f"{k}={v}" for k, v in tol.items() if k.startswith(("filtered_max", "R_"))]
    subs = g.get("subsets")
    if isinstance(subs, dict):
        parts += [f"{k}: {v.get('differing_px')} px differ" for k, v in subs.items() if isinstance(v, dict)]
    return "; ".join(parts)[:240]


_rows = []
for s in SITES:
    for meth, label in (("marea", "MAREA (v14)"), ("nidem", "NIDEM (v15)")):
        st = SET[(s, meth)]
        for name, g in st["gates"].items():
            _rows.append({"site": s, "method": label, "gate": name, "status": gate_status(g),
                          "bitwise": g.get("bitwise", "") if isinstance(g, dict) else "", "detail": gate_detail(g)})
GATES = pd.DataFrame(_rows)
display(GATES.set_index(["site", "method", "gate"]))
print(GATES.groupby(["method", "status"]).size().to_string())
RUN_STATUS = {f"{s} {m}": SET[(s, m)]["status"] for s in SITES for m in ("marea", "nidem")}
print("run status:", RUN_STATUS)
assert GATES["status"].isin(["PASS", "not checked"]).all(), "a gate did not pass"
assert set(RUN_STATUS.values()) == {"complete"}, "a run is not complete"'''

MD_DRAWS = """\
### Draw identity
For every (N, replicate), the NIDEM `draw_sha1` must equal MAREA's, and both must equal the draw recomputed
with v14's seed rule (`v14.draw(seed, site, 'all', N, rep, pool_size)`)."""

CODE_DRAWS = '''\
M = {s: v15.marea_rows(INPUTS[(s, "marea")]["csv"]) for s in SITES}       # pool 'all', mb 0: the MAREA curve
NI = {s: pd.read_csv(INPUTS[(s, "nidem")]["csv"], dtype={"draw_sha1": str}) for s in SITES}
_rows = []
for s in SITES:
    m, n = M[s], NI[s]
    assert not m.duplicated(["N", "replicate"]).any() and not n.duplicated(["N", "replicate"]).any(), s
    seed, size = int(SET[(s, "marea")]["seed"]), int(m["pool_size"].max())
    assert seed == int(SET[(s, "nidem")]["compat"]["seed"]) and size == int(n["pool_size"].max()), s
    km = dict(zip(zip(m["N"].astype(int), m["replicate"].astype(int)), m["draw_sha1"].astype(str)))
    kn = dict(zip(zip(n["N"].astype(int), n["replicate"].astype(int)), n["draw_sha1"].astype(str)))
    redraw = {k: v14.draw_sha1(v14.draw(seed, s, "all", k[0], k[1], size)) for k in km}
    _rows.append({"site": s, "seed": seed, "pool size": size, "MAREA draws": len(km), "NIDEM draws": len(kn),
                  "same (N, rep)": set(km) == set(kn), "NIDEM = MAREA": sum(kn.get(k) == v for k, v in km.items()),
                  "MAREA = v14.draw": sum(redraw[k] == v for k, v in km.items())})
DRAWS = pd.DataFrame(_rows).set_index("site")
display(DRAWS)
assert DRAWS["same (N, rep)"].all()
assert (DRAWS["NIDEM = MAREA"] == DRAWS["MAREA draws"]).all() and (DRAWS["MAREA = v14.draw"] == DRAWS["MAREA draws"]).all()
N_PAIRS = int(DRAWS["MAREA draws"].sum())
assert N_PAIRS == 224, N_PAIRS                    # the RESUMEN: the same draw_sha1 in the 224 pairs
print(f"{N_PAIRS} draw pairs: the same draws for both methods, and v14's seeded draws")'''

MD_PLAN = """\
## 3. The plan
v14's plan for the pool `all`, as recorded in each MAREA settings file (`v15.plan_from_settings`): the N grid
(the last N is the whole archive, one deterministic run), the replicates per N, and any shrink of the plan. The
NIDEM run used the same plan, and both CSVs hold one row per job."""

CODE_PLAN = '''\
_rows = []
for s in SITES:
    grid, reps, reps_large, size, jobs = v15.plan_from_settings(SET[(s, "marea")])
    nc = SET[(s, "nidem")]["compat"]
    assert [int(x) for x in nc["grid"]] == grid and int(nc["reps"]) == reps and int(nc["reps_large"]) == reps_large, s
    assert int(nc["jobs"]) == len(jobs) == len(M[s]) == len(NI[s]), s
    K = M[s].groupby("N").size()
    assert all(int(K[N]) == v14.reps_for(N, size, reps, reps_large) for N in grid), s
    _rows.append({"site": s, "pool size": size, "N grid": ", ".join(map(str, grid)),
                  f"draws per N (N <= {v14.LARGE_N})": reps, f"draws per N (N > {v14.LARGE_N})": reps_large,
                  "jobs": len(jobs),
                  "shrink notes": " | ".join(SET[(s, "marea")]["plan"].get("shrink_notes") or []) or "none"})
display(pd.DataFrame(_rows).set_index("site"))'''

MD_TABLES = """\
## 4. Per-site tables at the RESUMEN's N
Per site and N, for each method: **comp./K (RMSE)** = draws computable of the K at that N and, in brackets,
those with an RMSE (MAREA has no not-computable state: its failures are per pixel and show as coverage 0 or no
RMSE); **cov.** = median coverage over the K draws (not-computable draws at 0 %); **RMSEc** = median of the
median-centred RMSE over the draws with an RMSE (each on the part its method covers); **bias** = median of
median(p − t) over the same draws. **†** = the figure does not draw that median (an RMSE in under half of the K
draws, or median coverage under 50 %); **—** = no draw with an RMSE; **all** = the single run with the whole
archive (its values, not medians). The counts and the coverage come from `v14.summarize` (through
`v15.plot_inputs`); the medians of the † cells, which `v14.summarize` withholds, from the same rows. At Ems the
grid was reduced for time: the row for 30 uses N = 20 and the rows for 60 and 100 use N = 80."""

CODE_TABLES = '''\
#: the N of the RESUMEN's per-site tables (the last is the whole archive)
TABLE_N = {"villaviciosa": [30, 40, 60, 80, 100, 150, 465], "escalda": [30, 40, 60, 100, 150, 464],
           "wadden": [30, 40, 60, 100, 150, 462], "ems": [20, 40, 80, 150, 300, 462]}
TABLE_NOTE = {("villaviciosa", 80): "not requested; first N with NIDEM drawn", ("ems", 20): "instead of 30",
              ("ems", 80): "instead of 60 and 100", ("ems", 300): "not requested"}

D = {}
for s in SITES:
    D[s] = v15.plot_inputs(s, [INPUTS[(s, "marea")]["dir"]], [INPUTS[(s, "nidem")]["dir"]])
    assert same_path(D[s]["marea_src"], INPUTS[(s, "marea")]["csv"]), s
    assert same_path(D[s]["nidem_src"], INPUTS[(s, "nidem")]["csv"]), s
    assert D[s]["n_draws_marea"] == D[s]["n_draws_nidem"] == len(M[s]) and not D[s]["notes"], s


def method_cells(s, method, N):
    """One method at one N: counts, median coverage (%), medians of RMSE and bias over the draws with an RMSE,
    and whether the figure draws the median (v15._draw_site's rule on v14.summarize)."""
    rows = M[s] if method == "marea" else NI[s]
    g = rows[rows["N"] == N]
    K = len(g)
    calc = K if method == "marea" else int((~g["status"].isin(v15.NOT_COMPUTABLE)).sum())
    r = g["rmse"].to_numpy(float)
    fin = np.isfinite(r)
    k_rmse = int(fin.sum())
    rm = float(np.median(r[fin])) if k_rmse else float("nan")
    bi = float(np.median(g["bias"].to_numpy(float)[fin])) if k_rmse else float("nan")
    cv = D[s][method]
    srow = cv[cv["N"] == N].iloc[0]
    # the same numbers through v14.summarize, the function behind the figure
    assert int(srow["k"]) == K and int(srow["k_valid"]) == k_rmse, (s, method, N)
    assert float(srow["cov"]) == float(np.median(g["coverage"].to_numpy(float))), (s, method, N)
    assert not np.isfinite(srow["rmse"]) or float(srow["rmse"]) == rm, (s, method, N)
    if method == "nidem":
        assert D[s]["nc"][int(N)] == (K - calc, K), (s, N)
    drawn = bool(np.isfinite(srow["rmse"]) and float(srow["cov"]) >= v14.LOW_COVERAGE)
    return {"N": int(N), "calc": calc, "K": K, "k_rmse": k_rmse, "cov": 100.0 * float(srow["cov"]), "rmse": rm,
            "bias": bi, "drawn": drawn, "dagger": bool(k_rmse and not drawn)}


TABLES = {s: [{"N": N, "all": N == D[s]["pool_size"], "marea": method_cells(s, "marea", N),
               "nidem": method_cells(s, "nidem", N)} for N in TABLE_N[s]] for s in SITES}


def table_frame(s):
    def val(x, dag, signed=False):
        if not math.isfinite(x):
            return "—"
        return (f"{x:+.3f}" if signed else f"{x:.3f}") + ("†" if dag else "")

    recs = []
    for r in TABLES[s]:
        lab = f"all ({r['N']})" if r["all"] else str(r["N"])
        if (s, r["N"]) in TABLE_NOTE:
            lab += f" ({TABLE_NOTE[(s, r['N'])]})"
        rec = {"N": lab}
        for meth in ("marea", "nidem"):
            c, M_ = r[meth], meth.upper()
            rec[f"{M_} comp./K (RMSE)"] = f"{c['calc']}/{c['K']} ({c['k_rmse']})"
            rec[f"{M_} cov. %"] = f"{c['cov']:.2f}" if r["all"] else f"{c['cov']:.1f}"
            rec[f"{M_} RMSEc (m)"] = val(c["rmse"], c["dagger"])
            rec[f"{M_} bias (m)"] = val(c["bias"], c["dagger"], signed=True)
        recs.append(rec)
    return pd.DataFrame(recs).set_index("N")


for s in SITES:
    fx = SET[(s, "marea")]["fixed_set"]
    unit = "RTK points" if s == "villaviciosa" else "Vaklodingen pixels (fixed subsample)"
    p = SET[(s, "marea")]["plan"]
    display(Markdown(f"**{NAMES[s]}** — {fx['n_units_scored']:,} {unit}; archive of {D[s]['pool_size']} scenes; "
                     f"K = {p['reps']} up to N = {v14.LARGE_N}, {p['reps_large']} above, 1 with all"))
    display(table_frame(s))'''

MD_ASSERT = """\
The same tables, cell by cell, against those of `RESUMEN_marea_vs_nidem.md` (counts equal; values within the
rounding of the RESUMEN cell; the same † and —)."""

CODE_ASSERT = '''\
PARSED = parse_resumen_tables(RES_TEXT)
assert sorted(PARSED) == sorted(SITES), sorted(PARSED)
n_cells, mismatches = 0, []
for s in SITES:
    assert [r["N"] for r in PARSED[s]] == TABLE_N[s], (s, [r["N"] for r in PARSED[s]])
    n, bad = compare_rows(s, TABLES[s], PARSED[s])
    n_cells += n
    mismatches += bad
print(f"RESUMEN per-site tables: {n_cells} cells compared, {len(mismatches)} mismatches")
for b in mismatches:
    print("   ", b)
assert not mismatches, "the per-site tables differ from RESUMEN_marea_vs_nidem.md"
RESUMEN_TABLES_OK = True'''

MD_FIRST = """\
## 5. First-computable N per site and method
The first N of the grid at which at least half of the draws meet each criterion, with the count at that N and at
the previous N. *Drawn* adds the figure's coverage rule (median coverage ≥ 50 %). A `*` marks a criterion that
fails again at some larger N. MAREA has no not-computable state, so "computable" is trivially met at N = 10; the
fair comparison is "with RMSE". The leading N of every cell must equal the RESUMEN's table."""

CODE_FIRST = '''\
CRITERIA = [("MAREA: >= 1/2 with RMSE", "marea", "k_rmse"), ("MAREA: drawn", "marea", "drawn"),
            ("NIDEM: >= 1/2 computable", "nidem", "calc"), ("NIDEM: >= 1/2 with RMSE", "nidem", "k_rmse"),
            ("NIDEM: drawn", "nidem", "drawn")]


def meets(c, what):
    return c["drawn"] if what == "drawn" else 2 * c[what] >= c["K"]


FIRST, _rows = {}, []
for s in SITES:
    rec, FIRST[s] = {"site": NAMES[s]}, []
    for label, meth, what in CRITERIA:
        cells = [method_cells(s, meth, int(N)) for N in D[s][meth]["N"]]
        i = next((j for j, c in enumerate(cells) if meets(c, what)), None)
        if i is None:
            rec[label] = "never"
            FIRST[s].append(None)
            continue
        c, prev = cells[i], (cells[i - 1] if i else None)
        if what == "drawn":
            txt = f"{c['N']} ({c['cov']:.1f} %" + (f"; N = {prev['N']}: {prev['cov']:.1f} %" if prev else "") + ")"
        else:
            txt = f"{c['N']} ({c[what]}/{c['K']}" + (f"; N = {prev['N']}: {prev[what]}/{prev['K']}" if prev else "") + ")"
        rec[label] = txt + ("" if all(meets(x, what) for x in cells[i:]) else " *")
        FIRST[s].append(c["N"])
    _rows.append(rec)
display(pd.DataFrame(_rows).set_index("site"))
PARSED_FIRST = parse_first_n_table(RES_TEXT)
assert {s: PARSED_FIRST.get(s) for s in SITES} == FIRST, (PARSED_FIRST, FIRST)
print("first-computable N equal to the RESUMEN's table:", FIRST)'''

MD_FIGURE = """\
## 6. The published figure
The curves behind the figure (`v14.summarize` through `v15.plot_inputs`, every N), then the figure redrawn with
v15's own `figure` and `LAYOUT` (English poster version and Spanish note version, 220 dpi, the rcParams of
`v15.plot`) into the figure folder, and compared with the PNGs drawn on the server on 2026-10-26 at 10:28 with
v15 370ebf81 (found by their sha256, listed in the RESUMEN)."""

CODE_CURVES = '''\
def curve_frame(s):
    cm, cn = D[s]["marea"].set_index("N"), D[s]["nidem"].set_index("N")
    out = pd.DataFrame(index=cm.index)
    for tag, cv in (("MAREA", cm), ("NIDEM", cn)):
        drawn = np.isfinite(cv["rmse"]) & (cv["cov"] >= v14.LOW_COVERAGE)
        out[f"{tag} K"] = cv["k"]
        if tag == "NIDEM":
            out["NIDEM not comp."] = [D[s]["nc"][int(N)][0] for N in cv.index]
        out[f"{tag} with RMSE"] = cv["k_valid"]
        out[f"{tag} RMSEc"] = cv["rmse"].round(3)
        out[f"{tag} 10-90 %"] = [f"{a:.3f}-{b:.3f}" if np.isfinite(a) else "" for a, b in zip(cv["rmse_lo"], cv["rmse_hi"])]
        out[f"{tag} cov. %"] = (100 * cv["cov"]).round(1)
        out[f"{tag} drawn"] = drawn
    return out


for s in SITES:
    display(Markdown(f"**{NAMES[s]}**"))
    display(curve_frame(s))'''

CODE_FIGURE = '''\
FIG_DIR = os.environ.get("CURVES_FIG_DIR") or f"{OUT}/figs_{T_START.astimezone().date().isoformat()}"
v14.guard_out(FIG_DIR)
os.makedirs(FIG_DIR, exist_ok=True)
#: the published figures (server, 2026-10-26 10:28, v15 370ebf81): file stem, sha256 of the PNG, where to look
PUBLISHED = {
    "en": ("marea_vs_nidem_all_sites", "b729652e258f9197819c59673206ba1892e0aaedfd113ff23a7bd667c783fca1",
           [f"{OUT}/server_2026-10-26/marea_vs_nidem_all_sites.png", f"{OUT}/marea_vs_nidem_all_sites.png"]),
    "es": ("fig_marea_nidem_escenas", "06c288078242bcc15b6f2091b8a547a8cb21340f58af419e1c01b7ad317f8ba2",
           [f"{OUT}/server_2026-10-26/fig_marea_nidem_escenas.png", "docs/informe_fer/fig_marea_nidem_escenas.png"]),
}
V15_RC = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.titlepad": 6}     # as v15.plot


def png_compare(new, ref):
    import matplotlib.image as mpimg
    a, b = mpimg.imread(new), mpimg.imread(ref)
    if a.shape != b.shape:
        return {"same size": False, "new": a.shape, "published": b.shape}
    d = np.abs(a.astype(float) - b.astype(float)).max(axis=-1)
    return {"same size": True, "pixels": int(d.size), "pixels differing": int((d > 0).sum()),
            "max channel difference": float(d.max())}


Ds = [D[s] for s in SITES]
_rows = []
for lang in ("en", "es"):
    stem_name, want, cands = PUBLISHED[lang]
    lay = v15.LAYOUT[lang]
    stem = os.path.join(FIG_DIR, stem_name)
    with plt.style.context("default"), plt.rc_context(V15_RC):
        fig = v15.figure(Ds, lang, lay["fs"], lay["figsize"], lay["legend_y"], lay["foot_y"], lay["margins"])
        fig.savefig(stem + ".png", dpi=220)
        fig.savefig(stem + ".pdf")
    plt.close(fig)
    got = note(WRITTEN, stem + ".png")
    note(WRITTEN, stem + ".pdf")
    ref = next((p for p in cands if os.path.isfile(p) and v15.sha256_file(p) == want), None)
    rec = {"figure": f"{stem_name} ({lang})", "written": rel(stem + ".png"), "sha256": got}
    if ref is None:
        rec.update({"published copy": f"not found (sha256 {want[:12]}...)"})
    else:
        note(READ, ref)
        rec.update({"published copy": rel(ref), "bytes identical": got == want, **png_compare(stem + ".png", ref)})
    _rows.append(rec)
FIGCHECK = pd.DataFrame(_rows).set_index("figure")
display(FIGCHECK.T)
display(Image(filename=os.path.join(FIG_DIR, PUBLISHED["en"][0] + ".png"), width=900))'''

MD_BIAS = """\
## 7. Bias against N
The figure plots the median-centred RMSE, which removes median(p) − median(t), so the bias does not show. Here,
per site, the median over the draws of the bias median(p − t), over the same draws as the RMSE medians, with the
10–90 % range of those draws (3 or more draws) as a band and a whisker. Filled markers and the line: the N at
which the main figure draws the RMSE median; open markers: medians it does not draw (the † cells of section 4).
The value at the whole archive (a single run) is labelled."""

CODE_BIAS = '''\
def bias_curve(s, method):
    rows, cv = (M[s] if method == "marea" else NI[s]), D[s][method]
    out = []
    for N, rm, cov in zip(cv["N"], cv["rmse"], cv["cov"]):
        g = rows[rows["N"] == N]
        b = g["bias"].to_numpy(float)[np.isfinite(g["rmse"].to_numpy(float))]
        out.append({"N": int(N), "draws with bias": len(b), "median": float(np.median(b)) if len(b) else np.nan,
                    "lo": float(np.quantile(b, 0.1)) if len(b) >= 3 else np.nan,
                    "hi": float(np.quantile(b, 0.9)) if len(b) >= 3 else np.nan,
                    "drawn": bool(np.isfinite(rm) and cov >= v14.LOW_COVERAGE)})
    return pd.DataFrame(out)


BIAS = {(s, m): bias_curve(s, m) for s in SITES for m in ("marea", "nidem")}
PAL = v14.PAL
with plt.style.context("default"), plt.rc_context(V15_RC):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), sharey=False)
    for ax, s in zip(axes.ravel(), SITES):
        v14._style_axes(ax)
        ax.axhline(0.0, color=PAL["ink"], lw=0.9, zorder=2)
        ends = []
        for meth in ("marea", "nidem"):
            b, st = BIAS[(s, meth)], v15.STYLE[meth]
            x, y, dr = b["N"].to_numpy(float), b["median"].to_numpy(float), b["drawn"].to_numpy(bool)
            lo, hi = b["lo"].to_numpy(float), b["hi"].to_numpy(float)
            okb = dr & np.isfinite(lo) & np.isfinite(hi)
            if okb.any():
                ax.fill_between(x, np.where(okb, lo, 0.0), np.where(okb, hi, 0.0), where=okb, color=st["color"],
                                alpha=0.14, lw=0, zorder=1)
                ax.vlines(x[okb], lo[okb], hi[okb], color=st["color"], lw=1.0, alpha=0.6, zorder=2)
            ax.plot(x, np.where(dr, y, np.nan), color=st["color"], lw=2.0, zorder=3)
            ax.plot(x[dr], y[dr], ls="none", marker=st["marker"], ms=6.5, color=st["color"], mec="white", mew=1.2,
                    zorder=4)
            op = ~dr & np.isfinite(y)
            ax.plot(x[op], y[op], ls="none", marker=st["marker"], ms=6.5, mfc="white", mec=st["color"], mew=1.5,
                    zorder=4)
            if len(x) and int(x[-1]) == D[s]["pool_size"] and np.isfinite(y[-1]):
                ends.append((y[-1], x[-1]))
        # the whole-archive values, in ink; the lower one below its marker when there are two
        for i, (yv, xv) in enumerate(sorted(ends)):
            below = len(ends) == 2 and i == 0
            ax.annotate(f"{yv:+.3f}", (xv, yv), xytext=(7, -7 if below else (7 if len(ends) == 2 else 0)),
                        textcoords="offset points", fontsize=9.5, color=PAL["ink"], ha="left",
                        va="top" if below else ("bottom" if len(ends) == 2 else "center"))
        lo_y, hi_y = ax.get_ylim()                 # room for the end labels inside the axes
        ax.set_ylim(lo_y - 0.08 * (hi_y - lo_y), hi_y + 0.08 * (hi_y - lo_y))
        v14._log_n_axis(ax, D[s]["pool_size"])
        ax.set_xlim(ax.get_xlim()[0], D[s]["pool_size"] * 1.7)
        ax.set_title(NAMES[s], loc="left", fontsize=13, fontweight="bold", color=PAL["deep"], pad=18)
        n_u = SET[(s, "marea")]["fixed_set"]["n_units_scored"]
        ax.text(0.0, 1.012, (f"RTK, {n_u} points" if s == "villaviciosa" else f"Vaklodingen, {n_u:,} pixels")
                + f"; archive of {D[s]['pool_size']} scenes", transform=ax.transAxes, fontsize=9, color=PAL["slate"],
                ha="left", va="bottom")
        ax.set_ylabel("Bias, median(p − t) (m)", fontsize=10.5, color=PAL["ink"])
        ax.set_xlabel("Scenes used, N (log scale)", fontsize=10.5, color=PAL["ink"])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=v15.STYLE["marea"]["color"], lw=2, marker="s", ms=6.5, mec="white", label="MAREA"),
               Line2D([], [], color=v15.STYLE["nidem"]["color"], lw=2, marker="o", ms=6.5, mec="white", label="NIDEM"),
               Patch(facecolor=PAL["slate"], alpha=0.18, lw=0, label="10–90 % of the draws with an RMSE (3 or more)"),
               Line2D([], [], ls="none", marker="o", ms=6.5, mfc="white", mec=PAL["slate"], mew=1.5,
                      label="median the main figure does not draw (†)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=2.2)
    BIAS_STEM = os.path.join(FIG_DIR, "bias_vs_scenes_marea_nidem")
    fig.savefig(BIAS_STEM + ".png", dpi=200)
    fig.savefig(BIAS_STEM + ".pdf")
    plt.close(fig)
note(WRITTEN, BIAS_STEM + ".png")
note(WRITTEN, BIAS_STEM + ".pdf")
display(Image(filename=BIAS_STEM + ".png", width=900))

# the same numbers as a table: median bias (m) per N; † = a median the main figure does not draw
_t = {}
for s in SITES:
    for meth in ("marea", "nidem"):
        b = BIAS[(s, meth)].set_index("N")
        _t[(NAMES[s], meth.upper())] = pd.Series(
            ["—" if not np.isfinite(v) else f"{v:+.3f}" + ("" if d else "†") for v, d in zip(b["median"], b["drawn"])],
            index=b.index)
display(pd.DataFrame(_t).rename_axis("N").fillna(""))'''

MD_CAVEATS = """\
## 8. Caveats
From `RESUMEN_marea_vs_nidem.md` ("Salvedades"), in English; the original text is printed below.

- **The bias is not in the figure.** The plotted RMSE removes median(p) − median(t); the bias (median of p − t) is
  in the CSVs (section 7 draws it). Both methods read the same h_B and are scored against the same truth, so the
  difference in bias is an effect of the method: with the whole archive, NIDEM −0.543 m at Wadden and MAREA
  +0.208 m at Westerschelde.
- **No common set.** Each RMSE is over the part its method covers. Neither v14 nor v15 stores per-unit values, so
  the common part cannot be scored from these files.
- **Unequal filtering.** NIDEM is scored filtered (R 1 to 8 and mean NDWI standard deviation ≤ 0.25); MAREA without
  a quality rule (mb 0). MAREA's current rule (mb 3) is in the CSV but not drawn, and unfiltered NIDEM was not
  saved.
- **Conditional NIDEM medians.** Only computable draws with an RMSE enter. A draw scored on very little of the set
  enters with full weight (in drawn medians, draws with 1.2–2.5 % coverage), because the 50 % rule looks only at
  the median coverage.
- **A set made from MAREA.** Only units inside MAREA's keep mask and with a value in the published MAREA enter.
  So MAREA's coverage tends to 100 % by construction, and keep leaves out 76 of the 270 development RTK points.
- **Pool, not window.** The scenes are drawn from the unscreened archive (NIDEM's record). The author's design
  (the n most recent screened scenes, ending at the truth date) was not tested. N counts scenes of a fixed
  3-year archive, not the length of the window.
- **MAREA with the whole archive is not the published MAREA.** The published one (97 / 68 / 71 / 54 screened
  scenes) gives 0.192 / 0.411 / 0.408 / 0.347 m on the same units. NIDEM with the whole archive is its published
  product.
- **Truth.** Villaviciosa is scored with RTK, which the author sets aside for the method tables in favour of the
  PNOA LiDAR. The Dutch truth mixes surveys from 2019 to 2025; at Ems it is mostly from 2020.
- **Few draws.** 10 / 5 / 5 / 3 per N up to N = 100, 5 / 3 / 3 / 2 above, and 1 with the whole archive. The Ems
  grid was reduced for time. The 10–90 % bands come from 3 to 10 values (3 to 5 at the Dutch sites), and there are
  no significance tests.
- **NIDEM's stops are this study's.** "Empty interval" and the 1 % pre-flight are not NIDEM rules. What they cost
  is bounded only at Wadden (≤ 0.82 % coverage) and Ems (≤ 0.68 %).
- **Provenance.** The NIDEM rows of Villaviciosa and Westerschelde, and 26 of Wadden and 13 of Ems, come from an
  earlier uncommitted v15 (sha256 `6d682cba…`, 2026-10-20). The committed v15 added 27 Wadden rows and 4 Ems rows
  and drew both figures. The 39 old rows are identical in today's complete files, and the checks were bitwise in
  both runs.

*Added by this notebook, not in the RESUMEN:* the two NIDEM product gates of Villaviciosa (run of 2026-10-20) passed
within tolerance, not bitwise (filtered max |dz| ≤ 1.2e-7 m on keep, 0 at the scored units; section 2)."""

CODE_CAVEATS = '''\
SALVEDADES = resumen_section(RES_TEXT, "Salvedades")
assert SALVEDADES, "no 'Salvedades' section in the RESUMEN"
print(SALVEDADES)'''

MD_INTEGRITY = """\
## 9. Integrity
The sha256 of every file read (hashed again now: none may have changed during the run) and written, and the end
time."""

CODE_INTEGRITY = '''\
changed = [p for p, v in READ.items() if v15.sha256_file(os.path.join(REPO, p) if not os.path.isabs(p) else p)
           != v["sha256"]]
assert not changed, f"inputs changed during the run: {changed}"
T_END = dt.datetime.now(dt.timezone.utc)
_rows = ([{"role": "read", "path": p, **v} for p, v in sorted(READ.items())]
         + [{"role": "written", "path": p, **v} for p, v in sorted(WRITTEN.items())])
display(pd.DataFrame(_rows).set_index(["role", "path"]))
print(f"started {T_START.isoformat(timespec='seconds')}  ended {T_END.isoformat(timespec='seconds')}  "
      f"({(T_END - T_START).total_seconds():.1f} s)")
print(f"host {platform.node()}; code root {CODE_ROOT}; figures in {rel(FIG_DIR)}")
print("RESUMEN per-site tables reproduced:", RESUMEN_TABLES_OK, "| first-computable N reproduced: True")'''


def parser_cell_source():
    """The source of the RESUMEN parsers, as copied into the notebook."""
    return "\n\n".join(inspect.getsource(f).rstrip() for f in PARSERS)


def cells():
    return [
        ("md", TITLE),
        ("md", MD_HEADER),
        ("code", CODE_SETUP),
        ("md", MD_PARSERS),
        ("code", parser_cell_source()),
        ("md", MD_SETTINGS),
        ("code", CODE_INPUTS),
        ("md", MD_HASHES),
        ("code", CODE_HASHES),
        ("md", MD_CODE),
        ("code", CODE_CODE),
        ("md", MD_ENV),
        ("code", CODE_ENV),
        ("md", MD_GATES),
        ("code", CODE_GATES),
        ("md", MD_DRAWS),
        ("code", CODE_DRAWS),
        ("md", MD_PLAN),
        ("code", CODE_PLAN),
        ("md", MD_TABLES),
        ("code", CODE_TABLES),
        ("md", MD_ASSERT),
        ("code", CODE_ASSERT),
        ("md", MD_FIRST),
        ("code", CODE_FIRST),
        ("md", MD_FIGURE),
        ("code", CODE_CURVES),
        ("code", CODE_FIGURE),
        ("md", MD_BIAS),
        ("code", CODE_BIAS),
        ("md", MD_CAVEATS),
        ("code", CODE_CAVEATS),
        ("md", MD_INTEGRITY),
        ("code", CODE_INTEGRITY),
    ]


def build():
    """The notebook (no outputs), with deterministic cell ids."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for i, (kind, src) in enumerate(cells()):
        c = nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        c["id"] = f"{CELL_ID_PREFIX}-{i:02d}"
        nb.cells.append(c)
    return nb


def write(nb, path):
    """Write with LF line endings (stable bytes on every OS)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(nbf.writes(nb))
        f.write("\n")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def execute(out_path, timeout=900, kernel="python3", html=False):
    """Execute the notebook as it is on disk (NB_PATH) with nbclient, working
    directory = repository root, into ``out_path`` (+ .html). A failing run is
    written to <out>.failed.ipynb and exits 1."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbf.read(NB_PATH, as_version=4)
    nbf.validate(nb)
    if nbf.writes(nb) != nbf.writes(build()):
        print(f"NOTE: {NB_PATH} differs from a fresh build of this script (executing the file on disk)")
    started = _dt.datetime.now(_dt.timezone.utc)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel, resources={"metadata": {"path": ROOT}})
    failed = None
    try:
        client.execute()
    except CellExecutionError as e:              # the outputs up to the failing cell are kept
        failed = e
    nb.metadata["execution_record"] = {
        "source_notebook": os.path.relpath(NB_PATH, ROOT).replace(os.sep, "/"),
        "source_notebook_sha256": _sha256(NB_PATH),
        "build_script_sha256": _sha256(os.path.abspath(__file__)),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(), "python": sys.executable, "kernel": kernel, "cwd": ROOT,
        "env": {k: v for k, v in os.environ.items() if k.startswith("CURVES_")},
        "status": "failed" if failed else "ok",
    }
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if failed is not None:
        bad = os.path.splitext(out_path)[0] + ".failed.ipynb"
        write(nb, bad)
        print(f"EXECUTION FAILED; partial notebook -> {bad}\n{str(failed)[-3000:]}")
        sys.exit(1)
    write(nb, out_path)
    print(f"executed -> {out_path}  sha256 {_sha256(out_path)}")
    if html:
        try:
            from nbconvert import HTMLExporter
        except ImportError:
            sys.exit("--html needs nbconvert, which is not installed in this environment")
        body, _ = HTMLExporter().from_notebook_node(nb)
        hp = os.path.splitext(out_path)[0] + ".html"
        with open(hp, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        print(f"html     -> {hp}  sha256 {_sha256(hp)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--execute", metavar="OUT.ipynb", default=None,
                    help="execute the notebook on disk with nbclient into this path (does not rebuild it)")
    ap.add_argument("--html", action="store_true", help="with --execute: also an HTML export (nbconvert)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    ap.add_argument("--kernel", default="python3")
    args = ap.parse_args(argv)
    if args.execute:
        execute(args.execute, timeout=args.timeout, kernel=args.kernel, html=args.html)
        return
    nb = build()
    nbf.validate(nb)
    write(nb, NB_PATH)
    print(f"written  {NB_PATH}  sha256 {_sha256(NB_PATH)}")


if __name__ == "__main__":
    main()
