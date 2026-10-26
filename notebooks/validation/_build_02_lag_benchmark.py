# -*- coding: utf-8 -*-
"""Builds notebooks/validation/02_lag_benchmark.ipynb, the traceable record of
validation v2, Stage L (the lag benchmark run of 2026-10-26 on nserver1).

The notebook is generated from this file (nbformat, fixed cell ids) so that
it stays diffable: edit this file, never the .ipynb. The repository keeps the
notebook UNEXECUTED; executed copies go next to the run's outputs.

The notebook reads only the score step's outputs (``<LAG_DIR>/scores``,
``logs``, ``status``), recomputes the first lag table from lag_points.csv with
the benchmark's own scoring functions (experiments.benchmark.scoring,
lagtable, inference, judges) and asserts it against lag_table.md. Every
number it shows is computed from those files or from the code.

Run (repository root):
  python -X utf8 notebooks/validation/_build_02_lag_benchmark.py
      writes notebooks/validation/02_lag_benchmark.ipynb (unexecuted)
  python -X utf8 notebooks/validation/_build_02_lag_benchmark.py --no-write \\
      --lag-dir results/validation_v2 \\
      --execute results/validation_v2/scores/02_lag_benchmark.executed.ipynb \\
      --html results/validation_v2/scores/02_lag_benchmark.executed.html
      executes a copy with nbclient (and exports HTML: nbconvert when it is
      installed, else the small built-in exporter below)
  --check   exit 1 when the notebook on disk differs from the one this file builds
"""
from __future__ import annotations

import argparse
import base64
import html
import os
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NB_PATH = HERE / "02_lag_benchmark.ipynb"

# ─────────────────────────────────────────────────────────────────────────────
#  cells
# ─────────────────────────────────────────────────────────────────────────────

MD_TITLE = r"""# 02 · Lag benchmark (validation v2, Stage L): the pre-lock run of 2026-10-26

> **PRE-LOCK, descriptive; not the pre-registered analysis.** The confirmatory window W3 is
> BLINDED at every site. Unblinding W3 and lock-A are the user's decisions; this notebook
> takes neither and implies neither."""

MD_S0 = r"""## 0 · What this notebook is

**What the benchmark tests.** How close each lag method's read-out at the in-estuary tide
gauges (the *judges*) is to the lag the gauges themselves show, i.e. how much later the local
water moves than the boundary tide h_B (spec §6). MAREA's lag (L01, the m2a band lags) is set
against zero lag (L17), the global tide model at the gauges (L14), shallow-water celerity
(L18), the gauge-network leave-one-out reference (L20, never ranked) and the other imagery
lag methods (L02-L12). Three co-primary truths: (a) the best constant lag at the scene
instants, tau_ab (the same with a free gain) and (b) the M2 phase lag.

**Why it matters for the paper.** The lag is MAREA's modular tidal correction. This is the
first scored result of validation v2: it says where MAREA's band lags agree with the gauges,
where they do not, and how they compare with what a user would otherwise apply (no lag, a
global model), before any elevation result is read.

**How to read it.**
- *Pre-lock and descriptive.* `score --pre-lock`: every output carries the banner, no verdict
  is issued, family L-P1 is shown on each site's first available window ("mixed windows"),
  and W3 stays BLINDED (no truth read, no product scored).
- *Home advantage.* MAREA's internal parameters were developed on these sites, judges and
  truths (P0 is its development setting); Harlingen is a development judge.
- *Gauge agreement is weak evidence about MAREA's lag.* The lag absorbs wet/dry departures
  (ebb retention, drainage, high flats lagging low flats) and truth (a) is close to MAREA's own
  estimand. The decisive, non-circular test is the elevation contrast MAREA − no delay against
  external heights (family P1).
- *Catalogue instants.* Scene instants are the catalogue (datatake-start) times (user decision
  2026-10-14). The tile is sensed some minutes later, so every imagery lag is expected to read
  low by that offset (about 6.5 min at the Ems, the scorer's own note, printed from the code in
  §6). **Nothing is corrected**: the user's rule is to attribute errors before correcting
  anything. §6 compares imagery and non-imagery rows for that attribution.
- *Not validated (shown).* The imagery rows passed the positive control (§2) but gate
  G-record has not run, so they print as "not validated (shown)", only because of
  `--show-unvalidated`.

**How to reproduce.**
1. On nserver1, from the project root: `bash experiments/runs/run_v2_lag_2026-10-26.sh`, the
   exact script of this run (fetch, positive control, `plan --design lag`, 9 workers, score).
2. The score step alone (it reads the finished products and appends to the pre-lock ledger):
   `"$PY" -X utf8 -m experiments.validation_v2 score --pre-lock --show-unvalidated`.
3. Execute this notebook with nbclient:
   `"$PY" -X utf8 notebooks/validation/_build_02_lag_benchmark.py --no-write --execute
   results/validation_v2/scores/02_lag_benchmark.executed.ipynb --html
   results/validation_v2/scores/02_lag_benchmark.executed.html`.
   On a copy of the outputs set `V2_LAG_DIR` (or `--lag-dir`), e.g.
   `results/server_logs_2026-10-26/v2_lag`.

The notebook is generated by `notebooks/validation/_build_02_lag_benchmark.py`; edit that
file, not the .ipynb."""

CODE_PARAMS = r'''# ── parameters ───────────────────────────────────────────────────────────────
import os

# The run's output root (scores/, logs/, status/): results/validation_v2 on the server.
# The environment variable V2_LAG_DIR points it at a copy, e.g. the laptop's
# results/server_logs_2026-10-26/v2_lag. Relative paths start at the repository root.
LAG_DIR = os.environ.get("V2_LAG_DIR", "results/validation_v2")
# The archived run script: the exact script of the run, with a provenance header.
RUN_SCRIPT = "experiments/runs/run_v2_lag_2026-10-26.sh"'''

CODE_SETUP = r'''import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T_START = dt.datetime.now(dt.timezone.utc)
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "experiments" / "benchmark" / "scoring.py").is_file())
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark import designs, inference, judges, lagtable, scoring, settings  # noqa: E402

LAG = Path(LAG_DIR) if Path(LAG_DIR).is_absolute() else REPO / LAG_DIR
SCORES, LOGS, STATUS = LAG / "scores", LAG / "logs", LAG / "status"
assert SCORES.is_dir(), f"{SCORES} is missing: run the score step or set V2_LAG_DIR"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.max_rows", 400)

READ = {}          # every file read: relative path -> sha256 at first read (section 10 re-checks)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def use(path):
    """Record a file's sha256 the first time it is read; return its path."""
    p = Path(path)
    READ.setdefault(rel(p), sha256(p))
    return p


def read_text(path):
    return use(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read_text(path))


def read_lag_csv(name, usecols=None):
    """A lag score CSV with only ``usecols`` (lagtable.read_csv's rule: a first
    line '# <banner>' is skipped)."""
    p = use(SCORES / name)
    with open(p, encoding="utf-8") as f:
        first = f.readline()
    return pd.read_csv(p, skiprows=1 if first.startswith("# ") else 0, usecols=usecols)


# figure style; colours from the validated reference palette: blue = MAREA,
# orange = the best competitor, dark grey = zero lag, light grey = the rest
C_MAREA, C_COMP, C_ZERO, C_OTHER, C_BAND = "#2a78d6", "#eb6834", "#52514e", "#b4b3ad", "#dcdbd5"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9,
                     "axes.titlesize": 9.5, "legend.frameon": False, "lines.linewidth": 2,
                     "figure.facecolor": "white"})
print("repository:", REPO)
print("LAG_DIR   :", rel(LAG))'''

MD_S1 = r"""## 1 · Settings and provenance

`LAG_DIR` (parameters cell) is `results/validation_v2` on the server; `V2_LAG_DIR` points it
at a copy. The run is identified by the last `BEGIN lag benchmark` block of `logs/STATUS_lag`
and by the pre-lock ledger entry whose `points_sha256` equals the sha256 of `lag_points.csv`.
Then: the archived run script, the sha256 of every input, the benchmark code now against the
run's commit, the versions and the host."""

CODE_S1_RUN = r'''status_lines = [ln for ln in read_text(LOGS / "STATUS_lag").splitlines() if ln.strip()]
begins = [i for i, ln in enumerate(status_lines) if " BEGIN lag benchmark " in ln]
assert begins, "STATUS_lag has no 'BEGIN lag benchmark' line"
run_block = status_lines[begins[-1]:]                 # the server appends: the last run
ends = [i for i, ln in enumerate(run_block) if ln.endswith("DONE lag benchmark")]
run_block = run_block[: ends[0] + 1] if ends else run_block
events = pd.DataFrame([ln.split(" ", 1) for ln in run_block], columns=["utc", "event"])
events["utc"] = pd.to_datetime(events["utc"], utc=True)
RUN_COMMIT_SHORT = re.search(r"\(git ([0-9a-f]{7,40}) ", run_block[0]).group(1)
T_BEGIN, T_DONE = events["utc"].iloc[0], events["utc"].iloc[-1]
N_WORKERS = int(events["event"].str.startswith("END worker").sum())

meta = read_json(SCORES / lagtable.FILES["meta"])
ledger = read_json(SCORES / scoring.PRELOCK_LEDGER)
POINTS_SHA = sha256(use(SCORES / lagtable.FILES["points"]))
entries = [e for e in ledger["entries"] if e.get("points_sha256") == POINTS_SHA]
assert entries, "no pre-lock ledger entry matches lag_points.csv: these are not the ledgered scores"
entry = entries[-1]
RUN_COMMIT = entry["git"]["head"]
assert RUN_COMMIT.startswith(RUN_COMMIT_SHORT), (RUN_COMMIT, RUN_COMMIT_SHORT)
assert meta["pre_lock"] is True and not meta["unblinded"]

print(f"run            : {T_BEGIN:%Y-%m-%d %H:%M:%S} -> {T_DONE:%H:%M:%S} UTC "
      f"({(T_DONE - T_BEGIN).total_seconds() / 60:.0f} min; {N_WORKERS} workers ended)")
print(f"run commit     : {RUN_COMMIT} (STATUS_lag says {RUN_COMMIT_SHORT}; "
      f"ledger dirty = {entry['git']['dirty']})")
print(f"scores         : {meta['format']}, created {meta['created_utc']}, pre_lock = {meta['pre_lock']}, "
      f"banner '{meta['banner']}'")
print(f"ledger entry   : {entry['utc']}, '{entry['invocation']}' "
      f"(entry {ledger['entries'].index(entry) + 1} of {len(ledger['entries'])})")
print(f"lag_points.csv : sha256 {POINTS_SHA} = the entry's points_sha256")'''

CODE_S1_SCRIPT = r'''use(REPO / RUN_SCRIPT)
script_lines = (REPO / RUN_SCRIPT).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
h0, h1 = script_lines.index("# >>> archive header"), script_lines.index("# <<< archive header")
original = "\n".join(script_lines[:h0] + script_lines[h1 + 1:])
stated = re.search(r"sha256 of the original file: ([0-9a-f]{64})", "\n".join(script_lines)).group(1)
got = hashlib.sha256(original.encode("utf-8")).hexdigest()
print("\n".join(script_lines[h0:h1 + 1]))
print(f"\noriginal script (header removed, LF) sha256: {got}")
print(f"stated in the header                       : {stated}  -> equal: {got == stated}")
assert got == stated
called = re.findall(r"\bV2 (\w+[^>|]*?)\s*>", original)
print("\nvalidation_v2 commands in the script:")
for c in called:
    print("   ", c.strip())'''

CODE_S1_INPUTS = r'''INPUTS = ([SCORES / f for f in lagtable.FILES.values()] + [SCORES / scoring.PRELOCK_LEDGER]
          + [LOGS / n for n in ("STATUS_lag", "control.log", "status_before.log", "status_after.log",
                                "plan.log", "fetch_stac_meta.log", "fetch_emodnet.log",
                                "fetch_tides.log", "score_lag.log")]
          + sorted(LOGS.glob("lag_w*.log")) + [STATUS / "lag.csv"])
inputs = pd.DataFrame([{"file": rel(p), "bytes": p.stat().st_size if p.exists() else None,
                        "sha256": sha256(use(p)) if p.exists() else "MISSING"} for p in INPUTS])
print(inputs.to_string(index=False))
assert (inputs["sha256"] != "MISSING").all(), "input files are missing"
n_wlogs = len(list(LOGS.glob("lag_w*.log")))
print(f"\nworker logs present: {n_wlogs} of the {N_WORKERS} workers of the run "
      "(a copy of the outputs may hold fewer)")'''

CODE_S1_CODE = r'''def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


CODE = sorted(p for p in (REPO / "experiments" / "benchmark").rglob("*.py")
              if "__pycache__" not in p.parts) + [REPO / "experiments" / "validation_v2.py"]
code = pd.DataFrame({"file": [rel(p) for p in CODE]})
code["sha256_now"] = [sha256(p) for p in CODE]
head_now = (git("rev-parse", "HEAD") or "").strip() or None
branch = (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
run_tree = git("ls-tree", "-r", RUN_COMMIT, "--", "experiments/benchmark", "experiments/validation_v2.py")
run_blobs = {}
for line in (run_tree or "").splitlines():
    info, path = line.split("\t", 1)
    if path.endswith(".py"):
        run_blobs[path] = info.split()[2]
now_out = git("hash-object", "--", *code["file"])          # applies the checkout filters (CRLF)
now_blobs = dict(zip(code["file"], now_out.split())) if now_out else {}
if run_blobs and now_blobs:
    code["equal_to_run"] = [("new since the run" if f not in run_blobs else now_blobs[f] == run_blobs[f])
                            for f in code["file"]]
else:
    code["equal_to_run"] = "unknown (git or the run commit is not available)"
gone = sorted(set(run_blobs) - set(code["file"]))
porcelain = git("status", "--porcelain", "--", "experiments/benchmark", "experiments/validation_v2.py")
print(f"git HEAD now: {head_now} (branch {branch}); run commit {RUN_COMMIT}")
print(f"working-tree changes under experiments/benchmark and validation_v2.py: "
      f"{(porcelain or '').strip() or 'none'}")
print(code.to_string(index=False))
same = code["equal_to_run"].map(lambda v: v is True).sum()
print(f"\n{same} of {len(code)} source files equal the run commit's (git blob of the checked-out file "
      f"against the commit's; sha256_now is of the working-tree bytes, CRLF on a Windows checkout); "
      f"removed since the run: {gone or 'none'}")
CODE_EQUAL_TO_RUN = bool(run_blobs) and same == len(code) and not gone
print("benchmark code now == the run's code:", CODE_EQUAL_TO_RUN)'''

CODE_S1_VERSIONS = r'''versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__}
for mod in ("nbformat", "nbclient", "ipykernel"):
    try:
        versions[mod] = __import__(mod).__version__
    except ImportError:
        versions[mod] = "not installed"
print("versions:", ", ".join(f"{k} {v}" for k, v in versions.items()))
print(f"host    : {platform.node()} ({platform.platform()}); python {sys.executable}")
print(f"started : {T_START:%Y-%m-%d %H:%M:%S} UTC")

POINT_COLS = ["site", "window", "judge", "location", "s_g_km", "mouth", "development", "product",
              "method", "variant", "family", "fidelity", "lag_kind", "status", "validated",
              "readout_step", "n_readout_px", "edge", "tau_m", "truth", "tau_true", "I_lo", "I_hi",
              "e", "abs_e", "dE_cm", "dE_all_cm", "LLE_cm", "AGF", "explained", "IS", "covered",
              "flags"]
points = read_lag_csv(lagtable.FILES["points"], usecols=POINT_COLS)
print(f"\nlag_points.csv: {len(points):,} rows; {len(POINT_COLS)} of "
      f"{len(lagtable.POINTS_COLUMNS)} columns read ({points.memory_usage(deep=True).sum() / 1e6:.1f} MB)")'''

MD_S2 = r"""## 2 · The run

What the script did, in order (from `STATUS_lag` and the step logs): the free inputs, the
positive control, the plan, the workers and the score. Then the unit states per site
(`status/lag.csv`) and what was unavailable and why."""

CODE_S2_TIMELINE = r'''events["min_since_begin"] = ((events["utc"] - T_BEGIN).dt.total_seconds() / 60).round(1)
print(events.assign(utc=events["utc"].dt.strftime("%H:%M:%S")).to_string(index=False))

plan_txt = read_text(LOGS / "plan.log")
print("\nplan :", re.search(r"^\[plan\] lag: .*$", plan_txt, re.M).group(0))
score_txt = read_text(LOGS / "score_lag.log")
m = re.search(r"\[v2\] score finished in (\d+) s \(exit (\d+)\)", score_txt)
print(f"score: finished in {m.group(1)} s, exit {m.group(2)}; "
      f"{len(re.findall('BLINDED before lock-A', score_txt))} (site, window) pairs BLINDED")

tides = re.search(r"installed tide models: (\[.*?\])", read_text(LOGS / "fetch_tides.log")).group(1)
print("tide models installed:", tides)
stac = pd.DataFrame(re.findall(r"^\[fetch stac-meta\] (\w+): .* written: (\d+) dates over \[([\d, ]+)\]; "
                               r"relative orbits (.*)$", read_text(LOGS / "fetch_stac_meta.log"), re.M),
                    columns=["site", "dates", "years", "relative orbits"])
emod = set(re.findall(r"^\[fetch emodnet\] (\w+): .*dtm\.tif", read_text(LOGS / "fetch_emodnet.log"), re.M))
stac["EMODnet depths fetched"] = stac["site"].isin(emod)
print(stac.to_string(index=False))'''

CODE_S2_CONTROL = r'''ctl = read_text(LOGS / "control.log")
pat = re.compile(r"^\[control\] (\S+) / (\S+) (\S+): (PASS|control mismatch|FAIL \(fault\)|not applicable) "
                 r"\(median \|err\| (\S+) min over (\d+) bands\)$", re.M)
control = pd.DataFrame(pat.findall(ctl), columns=["world", "scenario", "product", "verdict",
                                                  "median_abs_err_min", "n_bands"])
control["median_abs_err_min"] = pd.to_numeric(control["median_abs_err_min"], errors="coerce").round(2)
control["n_bands"] = control["n_bands"].astype(int)
ctab = control.pivot_table(index="product", columns="scenario",
                           values=["verdict", "median_abs_err_min", "n_bands"], aggfunc="first")
ctab.columns = [f"{s} {v}" for v, s in ctab.columns]
ctab = ctab[sorted(ctab.columns, key=lambda c: (c.split()[0], c.split()[1] != "verdict"))]
print(f"world(s): {sorted(set(control['world']))}; scenarios: {sorted(set(control['scenario']))}")
print(ctab.to_string())

IMAGERY = list(scoring.SCORING_V2["printing"]["imagery"])
TOL = scoring.SCORING_V2["control"]["tol_min"]
im = control[control["product"].isin(IMAGERY)]
n_pass = im.groupby("product")["verdict"].apply(lambda v: (v == "PASS").all()).sum()
print(f"\nimagery products PASS in every scenario: {n_pass} of {im['product'].nunique()} "
      f"(PASS = median |err| over bands <= {TOL:g} min)")
print("G-levels:", sorted(set(re.findall(r"G-levels (PASS|FAIL)", ctl))),
      f"({len(re.findall(r'G-levels', ctl))} checks)")
m = re.search(r"\[v2\] control finished in (\d+) s \(exit (\d+)\)", ctl)
print(f"control finished in {m.group(1)} s, exit {m.group(2)}; E0a/E0b are the elevation products "
      "of the MAREA unit, not lag products ('not applicable' over 0 bands)")

ok_im = points[(points["status"] == "ok") & (points["truth"] == "a")
               & points["product"].str.split("@").str[0].isin(IMAGERY)]
n_unval = int((~ok_im["validated"].astype(bool)).sum())
n_grec = int(ok_im["flags"].fillna("").str.contains("G-record NOT RUN").sum())
print(f"\nyet {n_unval} of {len(ok_im)} ok imagery read-outs (truth a) are validated = False and "
      f"{n_grec} carry the flag 'G-record NOT RUN': the printing rule also needs the gates "
      f"{scoring.SCORING_V2['printing']['required_gates']['L01']} (L01), which have not run")'''

CODE_S2_UNITS = r'''units = pd.read_csv(use(STATUS / "lag.csv"),
                    usecols=["unit_id", "site", "window", "method", "variant", "state", "reason"])
print(f"status/lag.csv: {len(units)} units")
print(pd.crosstab(units["site"], units["state"], margins=True).to_string())
print()
print(pd.crosstab([units["site"], units["window"]], units["state"]).to_string())'''

CODE_S2_UNAVAILABLE = r'''after = read_text(LOGS / "status_after.log")
pkg = pd.DataFrame(re.findall(r"^\[status\] (\w+): record T = (\d+) over \[([^\]]*)\]", after, re.M),
                   columns=["site", "scenes T", "package years"])
pkg["package years"] = pkg["package years"].str.replace("'", "")
pkg["one-year package"] = pkg["package years"].str.count(",") == 0
print(pkg.to_string(index=False))

un = units[units["state"] == "unavailable"].copy()
rx = un["reason"].str.extract(r"store years \[([^\]]*)\] of window (\S+) are not in the package "
                              r"\(years \[([^\]]*)\]\)")
un["missing store years"], un["package years"] = rx[0], rx[2]
print("\nunavailable units: the window needs store years the package does not hold")
print(un.groupby(["site", "window", "missing store years", "package years"]).size()
        .rename("units").reset_index().to_string(index=False))
print("one-year sites:", ", ".join(f"{r['site']} ({r['package years']})"
                                   for _, r in pkg[pkg["one-year package"]].iterrows()))

other = units[units["state"].isin(["not_computable", "not_applicable"])].copy()
other["reason"] = other["reason"].str.slice(0, 110)
print("\nnot computable / not applicable units:")
print(other.groupby(["state", "method", "reason"])["site"]
        .apply(lambda s: ", ".join(sorted(set(s)))).rename("sites").reset_index().to_string(index=False))'''

MD_S3 = r"""## 3 · Sites, judges (gauges) and windows

The lag windows of the design (`designs.LAG_WINDOWS`, defined in `settings.window`), what each
(site, window) became in this run, and the judges: each gauge's location (the Terneuzen pair
is one location), along-channel distance s_g, mouth flag (s_g < 3 km), development flag and
truths. Development judges are those whose comparison influenced a MAREA parameter
(`judges.DEVELOPMENT_JUDGES`); the candidates (`judges.DEVELOPMENT_CANDIDATES`) are for the
authors to confirm before lock-A."""

CODE_S3_WINDOWS = r'''wins = pd.DataFrame([{"window": w, "start": settings.window(w).start,
                      "end (exclusive)": settings.window(w).end, "rule": settings.window(w).rule}
                     for w in designs.LAG_WINDOWS])
print(wins.to_string(index=False))


def sw_state(g):
    st = g["status"].astype(str)
    if st.str.startswith("BLINDED").any():
        return "BLINDED (pre-lock)"
    if st.str.startswith("NOT AVAILABLE: store years").any():
        return "not available (store years)"
    inc = g["judge"].notna() & ~st.str.startswith("judge excluded")
    return "scored against judges" if inc.any() else "ran; no judge (truth-free rows only)"


SITES = list(dict.fromkeys(points["site"]))
sw = points.groupby(["site", "window"], sort=False).apply(sw_state, include_groups=False)
sw = sw.unstack("window").reindex(index=SITES, columns=list(designs.LAG_WINDOWS))
# a judge-free site has no lag_points rows outside W3: its done units feed lag_repro.csv only
ran = units[units["state"] == "done"].groupby(["site", "window"]).size()
for (s, w), n in ran.items():
    if s in sw.index and w in sw.columns and pd.isna(sw.loc[s, w]):
        sw.loc[s, w] = f"ran ({n} units); no judge: lag_repro.csv only"
print()
print(sw.to_string())'''

CODE_S3_JUDGES = r'''jr = points[(points["product"] == "L17") & points["judge"].notna()
            & ~points["status"].astype(str).str.startswith(("BLINDED", "NOT AVAILABLE: store years"))]
tau = jr.pivot_table(index=["site", "window", "judge"], columns="truth", values="tau_true", aggfunc="first")
ia = jr[jr["truth"] == "a"].set_index(["site", "window", "judge"])
cand = {c for c, _ in judges.DEVELOPMENT_CANDIDATES}
jt = ia[["location", "s_g_km", "mouth", "development"]].copy()
jt["dev. candidate"] = [j in cand for j in jt.index.get_level_values("judge")]
jt["included"] = np.where(ia["status"].astype(str).str.startswith("judge excluded"), ia["status"], "yes")
jt["tau_a [I_a]"] = [("–" if not np.isfinite(t) else f"{t:.0f} [{lo:.0f}, {hi:.0f}]")
                     for t, lo, hi in zip(ia["tau_true"], ia["I_lo"], ia["I_hi"])]
for T in ("ab", "b", "hwlw"):
    jt[f"tau_{T}"] = tau.reindex(jt.index)[T].round(1) if T in tau else np.nan
jt["s_g_km"] = jt["s_g_km"].round(2)
print(jt.reset_index().to_string(index=False))

print("\ndevelopment judges (code):", [(c, why) for c, why in judges.DEVELOPMENT_JUDGES])
print("development flag in lag_points:", sorted(set(jr.loc[jr["development"].astype(bool), "judge"])),
      "| judges.development_list():", judges.development_list())
print("development candidates (to be confirmed before lock-A):",
      [c for c, _ in judges.DEVELOPMENT_CANDIDATES])

inc = jr[~jr["status"].astype(str).str.startswith("judge excluded")]
chk = []
for (s, w), g in inc.groupby(["site", "window"], sort=False):
    got = set(g["location"])
    chk.append({"site": s, "window": w, "included locations": len(got),
                "= judges.HEADLINE[site]": got == set(judges.HEADLINE.get(s, ()))})
print()
print(pd.DataFrame(chk).to_string(index=False))'''

MD_S4 = r"""## 4 · The first lag table, recomputed

The rule is the scorer's (`scoring.summarise`, `_location_frame`, `_summ_row`): per (site,
window, product, truth, judge set) the location-level means of the ok read-outs (the Terneuzen
pair is one location), then over locations MAE = mean |e|, bias = mean e, hit10 = share with
|e| <= `HIT_MIN`, RMS dE, RMS LLE and the mean AGF; pooled = the mean of the site values
(`scoring.pooled_rows`). The helper cell below applies these functions to `lag_points.csv`.

The CIs come from the truth-side month-block bootstrap draws, which the score step does not
write out, so they cannot be recomputed here: they are taken from `lag_summary.csv` and checked
against the markdown. Everything else is recomputed and asserted against `lag_summary.csv` (full
precision) and `lag_table.md` (within the printed rounding)."""

CODE_S4_HELPERS = r'''# ── section-4 helpers: the benchmark's own scoring rule applied to lag_points.csv ──
import re

import numpy as np
import pandas as pd

from experiments.benchmark import inference, judges, lagtable, scoring

JUDGE_SETS = tuple(scoring.SCORING_V2["judge_sets"])      # headline, without_mouth, development_free
JUDGED_SITES = tuple(inference.INFERENCE_V2["lp1"]["judged_sites"])
SUMMARY_KEY = ["site", "window", "product", "truth", "judge_set"]
#: how lagtable._first_table prints each quantity: (summary column, scale, decimals)
PRINTED = {"n": ("n_locations", 1, 0), "MAE": ("MAE", 1, 1), "bias": ("bias", 1, 1),
           "hit10": ("hit10", 100, 0), "rms_dE": ("rms_dE_cm", 1, 2),
           "rms_LLE": ("rms_LLE_cm", 1, 1), "mean_AGF": ("mean_AGF", 100, 0)}
#: the CI columns printed next to MAE and RMS dE
CI_OF = {"MAE": ("MAE_lo", "MAE_hi", 1), "rms_dE": ("rms_dE_lo_cm", "rms_dE_hi_cm", 2)}
DASH = "–"
_NUM = re.compile(r"^([+-]?\d+(?:\.\d+)?)(?: \[([+-]?\d+(?:\.\d+)?), ([+-]?\d+(?:\.\d+)?)\])?$")


def judge_sets_from_points(points):
    """({(site, window): judges.judge_sets(...)}, scored (site, window) set) from
    the judge rows of lag_points: a judge is included where its rows are not
    'judge excluded (...)', the development codes are the rows' own flag, and
    a window is scored where it has an included judge and is neither BLINDED
    nor outside the package years."""
    sets_by_sw, scored = {}, set()
    for (site, w), g in points.groupby(["site", "window"], sort=False):
        st = g["status"].astype(str)
        if st.str.startswith("BLINDED").any() or st.str.startswith("NOT AVAILABLE: store years").any():
            continue
        j = g[g["judge"].notna()].drop_duplicates("judge")
        table = [{"code": r.judge, "location": r.location, "mouth": bool(r.mouth),
                  "included": not str(r.status).startswith("judge excluded")}
                 for r in j.itertuples(index=False)]
        dev = sorted(set(j.loc[j["development"].astype(bool), "judge"]))
        if any(t["included"] for t in table):
            sets_by_sw[(site, w)] = judges.judge_sets(table, development=dev)
            scored.add((site, w))
    return sets_by_sw, scored


def recompute_summary(points):
    """lag_summary rows recomputed with scoring.summarise and scoring.pooled_rows
    (B = 0: the bootstrap draws are not written out, so the CI columns are NaN).
    Returns (summary, scored, lp1 windows, judge sets)."""
    sets_by_sw, scored = judge_sets_from_points(points)
    pts = points.reset_index(drop=True)
    label = scoring.SCORING_V2["banner_pre_lock"]
    summ, store = scoring.summarise(pts, [None] * len(pts), sets_by_sw, JUDGE_SETS, 0, label=label)
    lpw = scoring.lp1_windows(scored, JUDGED_SITES)
    pooled = scoring.pooled_rows(store, lpw, JUDGED_SITES, label=label)
    if pooled:
        summ = pd.concat([summ, pd.DataFrame(pooled)], ignore_index=True)
    for c in lagtable.SUMMARY_COLUMNS:
        if c not in summ.columns:
            summ[c] = np.nan
    return summ, scored, lpw, sets_by_sw


def attach_ci(summ, ref):
    """The recomputed summary with the CI columns of lag_summary.csv (not recomputable)."""
    ci = [c for c in ref.columns if c.endswith(("_lo", "_hi", "_lo_cm", "_hi_cm"))]
    out = summ.drop(columns=[c for c in ci if c in summ.columns])
    return out.merge(ref[SUMMARY_KEY + ci], on=SUMMARY_KEY, how="left")


def check_against_summary(summ, ref, metrics=None):
    """Per metric: rows compared and the max |recomputed - lag_summary.csv|."""
    metrics = metrics or ["n_locations", "MAE", "median_abs_e", "bias", "max_abs_e", "hit10",
                          "rms_dE_cm", "rms_dE_all_cm", "rms_LLE_cm", "mean_AGF", "mean_explained",
                          "edge_share", "slope", "ccc", "shape_err", "mono_violations",
                          "celerity_m_min_per_km", "celerity_true_min_per_km"]
    m = summ.merge(ref, on=SUMMARY_KEY, how="outer", suffixes=("", "_ref"), indicator=True)
    rows = [{"metric": "(rows)", "compared": int((m["_merge"] == "both").sum()),
             "only recomputed": int((m["_merge"] == "left_only").sum()),
             "only lag_summary": int((m["_merge"] == "right_only").sum()), "max_abs_diff": 0.0}]
    both = m[m["_merge"] == "both"]
    rows.append({"metric": "status", "compared": len(both),
                 "max_abs_diff": float((both["status"].astype(str) != both["status_ref"].astype(str)).sum())})
    for c in metrics:
        a = pd.to_numeric(both[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(both[c + "_ref"], errors="coerce").to_numpy(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        rows.append({"metric": c, "compared": int((fa & fb).sum()),
                     "finite only one side": int((fa ^ fb).sum()),
                     "max_abs_diff": float(np.max(np.abs(a[fa & fb] - b[fa & fb]))) if (fa & fb).any() else 0.0})
    return pd.DataFrame(rows)


def parse_cell(text):
    """'9.5 [6.5, 12.0]' -> (9.5, 6.5, 12.0); '–' -> NaNs; any other text -> None."""
    t = str(text).strip()
    if t == DASH:
        return (np.nan, np.nan, np.nan)
    m = _NUM.match(t)
    if not m:
        return None
    return tuple(np.nan if v is None else float(v) for v in m.groups())


def first_table_lines(md):
    """The lines lagtable._first_table wrote into lag_table.md (the section
    '## First lag table' without its heading and the blank lines around it)."""
    L = md.split("\n")
    i = L.index("## First lag table")
    j = next(k for k in range(i + 1, len(L)) if L[k].startswith("## "))
    return L[i + 2:j - 1]


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_first_table(md):
    """The printed first lag table as text cells: the per-site blocks (one row per
    site, window, product and co-primary truth) and the cross-site blocks (one row
    per window, truth, product and site, 'pooled' included)."""
    names = {v: k for k, v in lagtable.CO_PRIMARY_NAMES.items()}
    site_rows, cross_rows, block, hdr = [], [], None, None
    for line in first_table_lines(md):
        m = re.fullmatch(r"\*\*(\S+) · (\S+)\*\*", line)
        if m:
            block, hdr = ("site", m.group(1), m.group(2)), None
            continue
        m = re.fullmatch(r"\*(Y\d{4})\* — MAE \(min\) against truth (.+)", line)
        if m:
            block, hdr = ("cross", m.group(1), names[m.group(2)]), None
            continue
        if block is None or not line.startswith("|") or set(line) <= set("|-"):
            continue
        cells = _cells(line)
        if hdr is None:
            hdr = cells
            continue
        if block[0] == "site":
            base = {"site": block[1], "window": block[2], "product": cells[0].split()[0],
                    "status": cells[1], "n": cells[2]}
            for k, T in enumerate(lagtable.CO_PRIMARY):
                r = dict(base, truth=T, MAE=cells[3 + 3 * k], bias=cells[4 + 3 * k],
                         hit10=cells[5 + 3 * k])
                if T == "a":         # printed once, relative to tau_a for every truth
                    r.update(rms_dE=cells[12], rms_LLE=cells[13], mean_AGF=cells[14])
                site_rows.append(r)
        else:
            for site, c in zip(hdr[1:], cells[1:]):
                cross_rows.append({"site": site, "window": block[1], "truth": block[2],
                                   "product": cells[0], "MAE": c})
    return pd.DataFrame(site_rows), pd.DataFrame(cross_rows)


def expected_status(r):
    """What lagtable._first_table prints in a per-site block's status column."""
    st, show = lagtable._row_status(r, True)
    n = lagtable._num(r.get("n_locations"), "{:.0f}")
    if not show or n in (DASH, "0"):
        return str(st).replace("|", "/")
    return "ok" if st == "ok" else "not validated (shown)"


def compare_first_table(summ, md):
    """Every printed cell of the first lag table against the recomputed summary:
    a number agrees when |printed - recomputed| <= half a unit of its last printed
    digit; '–' agrees with NaN; a status agrees when it is the text the renderer
    prints for the recomputed row."""
    site_df, cross_df = parse_first_table(md)
    head = summ[summ["judge_set"] == "headline"].drop_duplicates(SUMMARY_KEY[:4])
    head = head.set_index(SUMMARY_KEY[:4])
    out = []

    def num(where, qty, text, value, scale, dec):
        p = parse_cell(text)
        v = np.nan if value is None else float(value) * scale
        tol = 0.5 * 10 ** -dec + 1e-9
        if p is None:
            ok = False
        elif np.isnan(p[0]):
            ok = not np.isfinite(v)
        else:
            ok = np.isfinite(v) and abs(p[0] - v) <= tol
        out.append(dict(where, quantity=qty, printed=text, recomputed=v, tol=tol, ok=bool(ok)))

    for r in site_df.to_dict("records"):
        key = (r["site"], r["window"], r["product"], r["truth"])
        where = {"block": "site", **{k: r[k] for k in ("site", "window", "product", "truth")}}
        if key not in head.index:
            out.append(dict(where, quantity="row", printed="present", recomputed=np.nan, tol=0, ok=False))
            continue
        s = head.loc[key]
        if r["truth"] == "a":
            exp = expected_status(s)
            out.append(dict(where, quantity="status", printed=r["status"], recomputed=exp, tol=0,
                            ok=r["status"] == exp))
            num(where, "n", r["n"], s["n_locations"], 1, 0)
        shown = r["status"] in ("ok", "not validated (shown)")
        qtys = ["MAE", "bias", "hit10"] + (["rms_dE", "rms_LLE", "mean_AGF"] if r["truth"] == "a" else [])
        for q in qtys:
            col, scale, dec = PRINTED[q]
            n_t = s["n_locations"]
            value = s[col] if (shown and np.isfinite(n_t) and n_t > 0) else np.nan
            num(where, q, r[q], value, scale, dec)
    for r in cross_df.to_dict("records"):
        key = (r["site"], r["window"], r["product"], r["truth"])
        where = {"block": "cross-site", **{k: r[k] for k in ("site", "window", "product", "truth")}}
        if key not in head.index:
            out.append(dict(where, quantity="MAE", printed=r["MAE"], recomputed=np.nan, tol=0,
                            ok=r["MAE"] == DASH))
            continue
        s = head.loc[key]
        if parse_cell(r["MAE"]) is None:                         # a status text
            st, show = lagtable._row_status(s, True)
            out.append(dict(where, quantity="status", printed=r["MAE"], recomputed=st, tol=0,
                            ok=(not show) and r["MAE"] == st))
        else:
            num(where, "MAE", r["MAE"], s["MAE"], 1, 1)
    return pd.DataFrame(out)


def compare_ci(md, ref):
    """The printed CIs of the first lag table against lag_summary.csv's CI columns."""
    site_df, _ = parse_first_table(md)
    r_ = ref[ref["judge_set"] == "headline"].drop_duplicates(SUMMARY_KEY[:4]).set_index(SUMMARY_KEY[:4])
    out = []
    for r in site_df.to_dict("records"):
        key = (r["site"], r["window"], r["product"], r["truth"])
        for q, (lo_c, hi_c, dec) in CI_OF.items():
            if q not in r or not isinstance(r[q], str):
                continue
            p = parse_cell(r[q])
            if p is None or np.isnan(p[1]):
                continue
            s = r_.loc[key]
            tol = 0.5 * 10 ** -dec + 1e-9
            out.append({"site": r["site"], "window": r["window"], "product": r["product"],
                        "truth": r["truth"], "quantity": q, "printed": r[q],
                        "csv_lo": s[lo_c], "csv_hi": s[hi_c],
                        "ok": abs(p[1] - s[lo_c]) <= tol and abs(p[2] - s[hi_c]) <= tol})
    return pd.DataFrame(out)'''

CODE_S4_SUMMARY = r'''summ, scored, lpw, sets_by_sw = recompute_summary(points)
print("descriptive L-P1 windows (scoring.lp1_windows):", lpw)
assert lpw == meta["lp1"]["windows"], (lpw, meta["lp1"]["windows"])
print("judge sets per scored (site, window) (judges.judge_sets):")
print(pd.DataFrame([{"site": s, "window": w, **{k: len(v) for k, v in d.items()},
                     "development_free drops": sorted(set(d["headline"]) - set(d["development_free"])),
                     "without_mouth drops": sorted(set(d["headline"]) - set(d["without_mouth"]))}
                    for (s, w), d in sets_by_sw.items()]).to_string(index=False))

SUMMARY_COLS = (lagtable.SUMMARY_KEYS + ("n_locations", "MAE", "MAE_lo", "MAE_hi", "median_abs_e",
                "bias", "bias_lo", "bias_hi", "max_abs_e", "hit10", "rms_dE_cm", "rms_dE_lo_cm",
                "rms_dE_hi_cm", "rms_dE_all_cm", "rms_LLE_cm", "rms_LLE_lo_cm", "rms_LLE_hi_cm",
                "mean_AGF", "mean_AGF_lo", "mean_AGF_hi", "mean_explained", "edge_share", "slope",
                "ccc", "shape_err", "mono_violations", "celerity_m_min_per_km",
                "celerity_true_min_per_km"))
ref_summary = read_lag_csv(lagtable.FILES["summary"], usecols=list(SUMMARY_COLS))
chk = check_against_summary(summ, ref_summary)
print("\nrecomputed summary against lag_summary.csv:")
print(chk.to_string(index=False))
assert chk.loc[0, "only recomputed"] == 0 and chk.loc[0, "only lag_summary"] == 0
assert (chk["max_abs_diff"].fillna(0) < 1e-9).all()
assert chk.get("finite only one side", pd.Series(dtype=float)).fillna(0).eq(0).all()
print("=> every summary row and value is reproduced from lag_points.csv by the scorer's functions")'''

CODE_S4_MD = r'''md = read_text(SCORES / lagtable.FILES["table_md"])
cmp_first = compare_first_table(summ, md)
print(f"printed cells compared: {len(cmp_first)} "
      f"({', '.join(f'{k} {v}' for k, v in cmp_first['block'].value_counts().items())}); "
      f"disagreeing: {int((~cmp_first['ok']).sum())}")
print(cmp_first.groupby("quantity")["ok"].agg(["size", "sum"]).rename(columns={"size": "cells", "sum": "agree"}).T.to_string())
bad = cmp_first[~cmp_first["ok"]]
assert bad.empty, bad
ci = compare_ci(md, ref_summary)
print(f"\nprinted CIs against lag_summary.csv: {len(ci)} intervals, disagreeing {int((~ci['ok']).sum())}")
assert ci["ok"].all(), ci[~ci["ok"]]

# the benchmark's renderer on the recomputed values (CIs from lag_summary.csv)
summ_ci = attach_ci(summ, ref_summary)
order = {k: i for i, k in enumerate(lagtable.CATALOGUE)}       # the plan's product order is not in the copy
rendered = lagtable._first_table(summ_ci, {"lp1_windows": lpw}, order, True)
printed = first_table_lines(md)
print(f"lagtable._first_table(recomputed) == the printed section: {rendered == printed} "
      f"({len(printed)} lines)")
print("=> ASSERTION PASSED: the first lag table of lag_table.md equals the recomputation")'''

CODE_S4_DISPLAY = r'''DISPLAY = ["L01", "L17", "L14", "L18", "L20", "L02", "L03", "L04w", "L04i", "L07", "L08", "L09",
           "L10", "L11", "L12"]
ha = summ_ci[summ_ci["judge_set"] == "headline"].set_index(["site", "window", "product", "truth"]).sort_index()


def cell(site, w, p, T="a", ci=True):
    k = (site, w, p, T)
    if k not in ha.index:
        return "–"
    r = ha.loc[k]
    if not (r["n_locations"] > 0):
        return str(r["status"]).split(" (")[0]
    s = f"{r['MAE']:.1f}"
    if ci and np.isfinite(r["MAE_lo"]):
        s += f" [{r['MAE_lo']:.1f}, {r['MAE_hi']:.1f}]"
    return s + f" · {r['bias']:+.1f}"


names = {p: lagtable.CATALOGUE[p][0] for p in DISPLAY}          # the spec §3.3 short names
tab = pd.DataFrame({f"{s} {w}": [cell(s, w, p) for p in DISPLAY] for s, w in lpw.items()},
                   index=pd.Index([f"{p} {names[p]}" for p in DISPLAY], name="product"))
print("MAE against truth (a) [CI, truth uncertainty only] · bias (a), min; headline judges; "
      "L-P1 windows; 'not validated (shown)' for every imagery row")
print(tab.to_string())

mtab = pd.DataFrame({(f"{s} {w}", T): [cell(s, w, p, T, ci=False).split(" ·")[0] for p in DISPLAY]
                     for s, w in lpw.items() for T in lagtable.CO_PRIMARY}, index=DISPLAY)
print("\nMAE (min) under the three co-primary truths (a), tau_ab, (b):")
print(mtab.to_string())

nloc = sum(len(sets_by_sw[(s, w)]["headline"]) for s, w in lpw.items())
print(f"\nheadline locations at the L-P1 windows: {nloc}")'''

CODE_S4_FAMILY = r'''fam, fam_info = scoring.lp1_family(points, scored, lock_doc=None, pre_lock=True, sites=SITES)
FAMILY_READ = ["truth", "comparison", "n", "k_zero", "S", "p", "p_min", "holm_p", "site_signs",
               "agree_sites", "label"]
ref_fam = read_lag_csv(lagtable.FILES["family"], usecols=FAMILY_READ)
assert len(fam) == len(ref_fam)
f2 = fam.reset_index(drop=True)
dS = max(float(np.nanmax(np.abs(f2[c].astype(float) - ref_fam[c].astype(float))))
         for c in ("n", "k_zero", "S", "p", "p_min"))
same_txt = all((f2[c].fillna("").astype(str) == ref_fam[c].fillna("").astype(str)).all()
               for c in ("truth", "comparison", "site_signs", "label"))
print(f"scoring.lp1_family against lag_family.csv: max |diff| {dS:.2e}; texts equal: {same_txt}")
assert dS < 1e-9 and same_txt
print("comparisons:", fam_info["comparisons"], "| confirmatory:", fam_info["confirmatory"],
      "| claims:", fam_info["claims"] or "none")
show = fam.assign(kind=np.where(fam["label"].str.contains("pseudo-replicated"), "pooled (pseudo-rep.)",
                                "site-weighted"))
print(show[["kind", "truth", "comparison", "n", "k_zero", "S", "p", "p_min", "holm_p", "site_signs",
            "agree_sites"]].round(4).to_string(index=False))

# the site-mean d behind S (inference.location_abs_e and inference.sign_flip, as family_lp1)
ok_lp = points[(points["status"] == "ok") & points["site"].isin(list(lpw))]
ok_lp = ok_lp[[lpw.get(s) == w for s, w in zip(ok_lp["site"], ok_lp["window"])]]
rows = []
for T in inference.INFERENCE_V2["lp1"]["truths"]:
    loc = inference.location_abs_e(ok_lp[ok_lp["truth"] == T])
    for X in fam_info["comparisons"]:
        d = {}
        for s in JUDGED_SITES:
            a = loc[(loc["site"] == s) & (loc["product"] == "L01")].set_index("location")["abs_e"]
            b = loc[(loc["site"] == s) & (loc["product"] == X)].set_index("location")["abs_e"]
            dd = (a - b).dropna().to_numpy(float)
            if len(dd):
                d[s] = dd
        sf = inference.sign_flip(d)
        rows.append({"truth": T, "comparison": X, **{s: sf["site_means"].get(s, np.nan) for s in JUDGED_SITES},
                     "S": sf["stat"]})
site_d = pd.DataFrame(rows)
print("\nsite-mean d = |e_L01| - |e_X| (min; < 0: MAREA closer), per judged site:")
print(site_d.round(2).to_string(index=False))'''

MD_S5 = r"""## 5 · Figures

(a) MAE against truth (a) per method, per site and window (headline judges; every scored window,
not only the L-P1 ones); MAREA in blue, zero lag in dark grey. Bars are the CIs from
`lag_summary.csv` (truth uncertainty only). (b) The signed error at each judge for MAREA, zero
lag and the best competitor of that window: the ranked method (imagery L02-L12, L14, L18; not
zero lag, not the reference L20, not the descriptive `@` rows) with the lowest MAE (a). The grey
bar is the truth interval I_a around zero."""

CODE_S5_MAE = r'''FIG_PRODUCTS = ["L01", "L02", "L03", "L04w", "L04i", "L07", "L08", "L09", "L10", "L11", "L12",
                "L14", "L17", "L18", "L20"]
W_ORDER = {w: i for i, w in enumerate(inference.INFERENCE_V2["lp1"]["descriptive_windows"])}
PANELS = sorted(scored, key=lambda sw: (list(JUDGED_SITES).index(sw[0]), W_ORDER.get(sw[1], 99)))
head_a = summ_ci[(summ_ci["judge_set"] == "headline") & (summ_ci["truth"] == "a")
                 & (summ_ci["site"] != "pooled")].set_index(["site", "window", "product"]).sort_index()
ypos = {p: i for i, p in enumerate(FIG_PRODUCTS)}
fig, axes = plt.subplots(2, 4, figsize=(13, 7.4), sharey=True, constrained_layout=True)
for ax, (site, w) in zip(axes.flat, PANELS):
    for p in FIG_PRODUCTS:
        y, k = ypos[p], (site, w, p)
        if k not in head_a.index or not head_a.loc[k, "n_locations"] > 0:
            ax.text(0.01, y, "–", va="center", ha="left", color="#8a8984", fontsize=8,
                    transform=ax.get_yaxis_transform())
            continue
        r = head_a.loc[k]
        col, mk, ms, z = ((C_MAREA, "o", 7, 4) if p == "L01" else (C_ZERO, "s", 6, 3) if p == "L17"
                          else (C_OTHER, "o", 5, 2))
        ax.plot([r["MAE_lo"], r["MAE_hi"]], [y, y], color=col, lw=1.6, zorder=z, solid_capstyle="round")
        ax.plot(r["MAE"], y, marker=mk, ms=ms, color=col, mfc="white" if p == "L17" else col,
                mew=1.6, ls="none", zorder=z + 1)
        if p in ("L01", "L17"):
            ax.annotate(f"{r['MAE']:.1f}", (r["MAE"], y), xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color="#2b2b2a")
    ax.set_title(f"{site} · {w}" + ("  (L-P1 window)" if lpw.get(site) == w else ""))
    ax.set_xlim(left=0)
    ax.set_xlabel("MAE against truth (a) (min)")
axes.flat[0].set_yticks(range(len(FIG_PRODUCTS)), FIG_PRODUCTS)
axes.flat[0].invert_yaxis()
for ax in list(axes.flat)[len(PANELS):]:
    ax.set_axis_off()
    ax.plot([], [], "o", color=C_MAREA, label="MAREA L01")
    ax.plot([], [], "s", color=C_ZERO, mfc="white", mew=1.6, label="zero lag L17")
    ax.plot([], [], "o", color=C_OTHER, label="other methods")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0, 0.45, "bars: CI, truth uncertainty only\n(lag_summary.csv)\n"
            "–: no value (status in lag_table.md)\nimagery rows: not validated (shown)\n"
            "PRE-LOCK, descriptive; W3 BLINDED", transform=ax.transAxes, fontsize=8, va="top",
            color="#52514e")
fig.suptitle("Lag benchmark (pre-lock): MAE per method, site and window, headline judges", fontsize=10.5)
plt.show()'''

CODE_S5_JUDGE = r'''RANKED = [p for p in FIG_PRODUCTS if p not in ("L01", "L17", "L20")]
best = {}
for site, w in PANELS:
    sub = head_a.loc[(site, w)]
    cand = sub.loc[[p for p in RANKED if p in sub.index]]
    cand = cand[cand["n_locations"] > 0]
    best[(site, w)] = cand["MAE"].idxmin()
best_tab = pd.DataFrame([{"site": s, "window": w, "MAREA L01": head_a.loc[(s, w, "L01"), "MAE"],
                          "zero lag L17": head_a.loc[(s, w, "L17"), "MAE"], "best competitor": best[(s, w)],
                          "its MAE": head_a.loc[(s, w, best[(s, w)]), "MAE"]} for s, w in PANELS])
print("MAE (a), min, headline judges:")
print(best_tab.round(1).to_string(index=False))

ok_a = points[(points["status"] == "ok") & (points["truth"] == "a")]
LAB = {f"{s} · {w}": (s, w) for s, w in PANELS}
labs = list(LAB)
mosaic = ([labs[:4], [labs[4], labs[5], labs[5], labs[6]]] if len(labs) == 7
          else [labs[i:i + 4] + ["."] * (4 - len(labs[i:i + 4])) for i in range(0, len(labs), 4)])
fig, axd = plt.subplot_mosaic(mosaic, figsize=(13, 7.6), constrained_layout=True, empty_sentinel=".")
for lab, ax in axd.items():
    site, w = LAB[lab]
    g = ok_a[(ok_a["site"] == site) & (ok_a["window"] == w)]
    jj = g[g["product"] == "L17"].drop_duplicates("judge").sort_values("s_g_km")
    xs = {j: i for i, j in enumerate(jj["judge"])}
    for i, r in enumerate(jj.itertuples(index=False)):
        ax.plot([i, i], [r.I_lo - r.tau_true, r.I_hi - r.tau_true], color=C_BAND, lw=9,
                solid_capstyle="butt", zorder=1)
    for p, col, mk, dx, lab in (("L01", C_MAREA, "o", -0.16, "MAREA L01"),
                                ("L17", C_ZERO, "s", 0.0, "zero lag L17"),
                                (best[(site, w)], C_COMP, "D", 0.16, f"best: {best[(site, w)]}")):
        gp = g[(g["product"] == p) & g["judge"].isin(list(xs))]
        ax.plot([xs[j] + dx for j in gp["judge"]], gp["e"], marker=mk, ms=6.5, ls="none", color=col,
                mfc="white" if p == "L17" else col, mew=1.5, label=lab, zorder=3)
    ax.axhline(0, color="#8a8984", lw=0.8, zorder=2)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_xticks(range(len(xs)), [f"{j[:16]}\n{s:.1f} km" for j, s in zip(jj["judge"], jj["s_g_km"])],
                  fontsize=7, rotation=0 if len(xs) <= 3 else 35, ha="center" if len(xs) <= 3 else "right")
    ax.set_title(f"{site} · {w}" + ("  (L-P1 window)" if lpw.get(site) == w else ""))
    ax.set_ylabel("e = tau_m - tau_a (min)")
    ax.legend(fontsize=7.5, loc="best")
fig.suptitle("Signed error per judge (truth a); grey: truth interval I_a around zero; "
             "judges ordered by s_g", fontsize=10.5)
plt.show()'''

MD_S6 = r"""## 6 · Imagery against non-imagery rows (the catalogue-instant offset)

`scoring.offset_lines`: the median signed e (truth a, ok rows, base products) of the imagery rows
(L01-L12) and of the lag estimates that read no imagery (L14, L16, L18; zero lag is a null
baseline). If the catalogue instants were the only difference, the imagery median would sit
below the other by about the datatake-to-sensing offset. Descriptive, for attribution; nothing is
corrected. The sensing-offset column needs the optional `diag time` output, which lives on the
server; it is read here from `lag_table.md`."""

CODE_S6 = r'''note = scoring.SENSING_OFFSET_NOTE
m = re.search(r"sensed ([\d.]+)-([\d.]+) min later \(median ([\d.]+) min at the (\w+)\)", note)
EXP_LO, EXP_HI, EXP_MED, EXP_SITE = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
print("the scorer's note (scoring.SENSING_OFFSET_NOTE):\n ", note)
print(f"=> expected: imagery lags read {EXP_LO:g}-{EXP_HI:g} min low (median {EXP_MED:g} at the "
      f"{EXP_SITE}); an expectation from the code, not a measurement of this run")

off = pd.DataFrame(scoring.offset_lines(points, str(LAG), [])).drop(columns="sensing")
off["imagery - non-imagery"] = off["imagery_median_e"] - off["non_imagery_median_e"]

sec = md.split("## Imagery against non-imagery rows (catalogue instants)")[1].split("\n## ")[0]
prow = [_cells(ln) for ln in sec.split("\n") if ln.startswith("|") and not set(ln) <= set("|-")][1:]
printed_off = pd.DataFrame(prow, columns=["site", "window", "imagery", "non_imagery", "sensing offset"])
mm = off.merge(printed_off, on=["site", "window"], how="outer", indicator=True)
assert (mm["_merge"] == "both").all()
for col, med, n in (("imagery", "imagery_median_e", "n_imagery"),
                    ("non_imagery", "non_imagery_median_e", "n_non_imagery")):
    v = mm[col].str.extract(r"^([+-]?[\d.]+) \((\d+)\)$").astype(float)
    assert ((v[0] - mm[med]).abs() <= 0.05 + 1e-9).all() and (v[1] == mm[n]).all(), col
print(f"\nscoring.offset_lines == the printed section: True ({len(mm)} rows); sensing-offset column: "
      f"{sorted(set(mm['sensing offset']))}")
print(off.round(1).to_string(index=False))

ctx = points[(points["status"] == "ok") & (points["truth"] == "a")]
per = ctx[ctx["product"].isin(["L01", "L17", "L14", "L18"])].groupby(["site", "window", "product"])["e"] \
    .median().unstack("product")[["L01", "L17", "L14", "L18"]]
print("\nmedian signed e (truth a) per product, min (gauge rows):")
print(per.round(1).to_string())

base = ctx[~ctx["product"].str.contains("@")]
fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
keys = list(zip(off["site"], off["window"]))
rng = np.random.default_rng(0)                 # jitter only (display)
for i, (s, w) in enumerate(keys):
    g = base[(base["site"] == s) & (base["window"] == w)]
    for grp, col, dx in ((scoring._IMAGERY, C_MAREA, -0.17), (scoring._NON_IMAGERY_ESTIMATES, C_COMP, 0.17)):
        e = g.loc[g["product"].isin(grp), "e"].dropna()
        ax.plot(i + dx + rng.uniform(-0.07, 0.07, len(e)), e, "o", ms=4, color=col, alpha=0.55, mew=0)
        if len(e):
            ax.plot([i + dx - 0.13, i + dx + 0.13], [e.median()] * 2, color="#2b2b2a", lw=2)
ax.axhline(0, color="#8a8984", lw=0.8)
ax.set_xticks(range(len(keys)), [f"{s}\n{w}" for s, w in keys])
ax.set_ylabel("e against truth (a) (min)")
ax.plot([], [], "o", color=C_MAREA, label="imagery rows (L01-L12)")
ax.plot([], [], "o", color=C_COMP, label="non-imagery estimates (L14, L16, L18)")
ax.plot([], [], color="#2b2b2a", label="group median")
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3)
ax.set_title("Imagery against non-imagery rows, per site and window (gauge rows; nothing corrected)",
             pad=24)
plt.show()'''

MD_S7 = r"""## 7 · Villaviciosa: truth-free coherence, and reproducibility

`lag_repro.csv` (no truth enters it): band lags across calendar years (Y2025 = W1) and across
relative orbits (L01v), as the median |dtau| per band and ICC(2,1) over bands; at Villaviciosa,
which has no judge, the method × band table and the spread across methods per band ("lag
coherence only"). The summary rows (band "all") and the spreads are recomputed from the per-band
rows with the scorer's helpers (`scoring._pair_stats`, `inference.icc21`); the per-band values are
printed with two decimals, so the check allows that rounding."""

CODE_S7 = r'''repro = read_lag_csv(lagtable.FILES["repro"])
print(f"lag_repro.csv: {len(repro)} rows;", repro["kind"].value_counts().to_dict())


def vals(s):
    return [float(x) for x in str(s).split(";")]


rows = []
for (site, w, kind, pid), g in repro[repro["kind"].isin(["years", "orbits"])].groupby(
        ["site", "window", "kind", "product"], sort=False):
    lk = lagtable.product_info(pid)["lag_kind"]
    bands = g[g["band"].astype(str) != "all"]
    allr = g[g["band"].astype(str) == "all"].iloc[0]
    M = np.array([vals(v) for v in bands["values"]])
    per_band = np.array([scoring._pair_stats(r, lk) for r in M])
    csv_band = bands["median_abs_diff"].astype(float).to_numpy()
    assert (np.isfinite(per_band) == np.isfinite(csv_band)).all(), (site, w, kind, pid)
    rec_med = float(np.nanmedian(bands["median_abs_diff"].astype(float))) if np.isfinite(
        bands["median_abs_diff"].astype(float)).any() else np.nan
    rec_icc = inference.icc21(M) if len(M) else np.nan
    rows.append({"site": site, "window": w, "kind": kind, "product": pid, "bands": len(bands),
                 "groups": allr["groups"], "median |dtau|": allr["median_abs_diff"],
                 "recomputed": rec_med, "ICC(2,1)": allr["icc21"], "ICC recomputed": rec_icc,
                 "per-band max diff": float(np.nanmax(np.abs(per_band - bands["median_abs_diff"].astype(float))))
                 if np.isfinite(per_band).any() else 0.0})
rep = pd.DataFrame(rows)
d_med = np.nanmax(np.abs(rep["median |dtau|"] - rep["recomputed"]))
d_icc = np.nanmax(np.abs(rep["ICC(2,1)"] - rep["ICC recomputed"]))
print(f"recomputed from the per-band rows: max |diff| median |dtau| {d_med:.2e}, "
      f"per band {rep['per-band max diff'].max():.3f} (2-decimal values), ICC {d_icc:.4f}")
assert d_med < 1e-9 and rep["per-band max diff"].max() <= 0.011 and d_icc <= 0.005
print(rep.drop(columns=["recomputed", "ICC recomputed", "per-band max diff"]).round(3).to_string(index=False))

coh = repro[repro["kind"] == "coherence"].copy()
long = []
for r in coh.itertuples(index=False):
    for kv in str(r.values).split(";"):
        p, v = kv.split("=")
        long.append({"window": r.window, "band": int(r.band), "centre_km": r.centre_km, "product": p,
                     "tau": float(v)})
long = pd.DataFrame(long)
def _sd(v):
    v = v.dropna()
    return float(np.std(v)) if len(v) > 1 else np.nan


def _range(v):
    v = v.dropna()
    return float(np.ptp(v)) if len(v) > 1 else np.nan


coh["band"] = coh["band"].astype(int)
spread = long.groupby(["window", "band"])["tau"].agg([_sd, _range])
coh["sd_rec"] = [spread.loc[(r.window, r.band), "_sd"] for r in coh.itertuples(index=False)]
coh["range_rec"] = [spread.loc[(r.window, r.band), "_range"] for r in coh.itertuples(index=False)]
d_sd = np.nanmax(np.abs(coh["sd_rec"] - coh["spread_sd"]))
d_rg = np.nanmax(np.abs(coh["range_rec"] - coh["spread_range"]))
print(f"\ncoherence spreads recomputed: max |diff| SD {d_sd:.3f}, range {d_rg:.3f} (2-decimal values)")
assert d_sd <= 0.011 and d_rg <= 0.011
wide = long.pivot_table(index=["window", "band", "centre_km"], columns="product", values="tau")
others = wide.drop(columns="L01")
cs = coh.set_index(["window", "band"])
wide["SD"] = [cs.loc[(w, b), "spread_sd"] for w, b, _ in wide.index]
wide["range"] = [cs.loc[(w, b), "spread_range"] for w, b, _ in wide.index]
wide["L01 - median(others)"] = wide["L01"] - others.median(axis=1)
print("\nVillaviciosa band lags (min) per method; spread across methods; MAREA against the others:")
print(wide.round(1).to_string())
l01_b0 = long[(long["product"] == "L01") & (long["band"] == 0)]["tau"]
orb0 = repro[(repro["kind"] == "orbits") & (repro["band"].astype(str) == "0")]
print(f"\nMAREA band 0 (pinned at tau = 0 by m2a): L01 {sorted(set(l01_b0))} at {len(l01_b0)} windows; "
      f"L01v band 0 all zero at {int(sum(np.allclose(vals(v), 0) for v in orb0['values']))} of "
      f"{len(orb0)} (site, window) orbit rows")

wins_v = list(dict.fromkeys(long["window"]))
fig, axes = plt.subplots(1, len(wins_v), figsize=(13, 3.4), sharey=True, constrained_layout=True)
for ax, w in zip(np.atleast_1d(axes), wins_v):
    g = long[long["window"] == w]
    for p, gp in g.groupby("product"):
        is_m = p == "L01"
        ax.plot(gp["centre_km"], gp["tau"], marker="o", ms=5 if is_m else 3.5, lw=2.2 if is_m else 1.1,
                color=C_MAREA if is_m else C_OTHER, zorder=3 if is_m else 2,
                label="MAREA L01" if is_m else None)
    ax.plot([], [], color=C_OTHER, lw=1.1, label="other methods")
    ax.set_title(f"villaviciosa · {w}")
    ax.set_xlabel("band centre s (km)")
np.atleast_1d(axes)[0].set_ylabel("band lag (min)")
np.atleast_1d(axes)[0].legend(fontsize=8)
fig.suptitle("Villaviciosa (no judge): band lags per method, lag coherence only", fontsize=10.5)
plt.show()'''

MD_S8 = r"""## 8 · Compute time per method

`lag_cost.csv`: medians over instances of each method's own stages (contended production runs,
good for budgeting only; spec §8.2 keeps comparisons for the dedicated Stage T runs, not part of
this run). L01's row is the whole MAREA unit (E0: both inversions and the lag). The fallback-0
share is recomputed from `lag_points.csv`; the per-unit times come from the worker logs that are
present."""

CODE_S8 = r'''cost = read_lag_csv(lagtable.FILES["cost"])
f0 = {}
okc = points[(points["status"] == "ok") & (points["truth"] == "a")]
for pid in cost["product"]:
    st = okc.loc[okc["product"] == pid, "readout_step"].map(str)
    f0[pid] = float(np.mean(st == "3")) if len(st) else np.nan
cost["fallback0 recomputed"] = cost["product"].map(f0)
d_f0 = np.nanmax(np.abs(cost["fallback0 recomputed"] - cost["share_fallback0"]))
print(f"label: {cost['label'].iloc[0]}")
print(f"share_fallback0 recomputed from lag_points.csv: max |diff| {d_f0:.2e}")
assert d_f0 < 1e-12
print(cost.drop(columns=["label", "fallback0 recomputed"]).round(3).to_string(index=False))

upat = re.compile(r"^\[unit\] (\S+) (\S+) (\S+) (\S+) (\S+) (\S+): (.+) in ([\d.]+) s \(cpu ([\d.]+) s, "
                  r"peak (\d+) MB\)$", re.M)
ulog = []
for p in sorted(LOGS.glob("lag_w*.log")):
    txt = read_text(p)
    t0 = re.search(r"\[v2\] run on \S+ \(pid \d+\), (.+)$", txt, re.M)
    t1 = re.search(r"\[v2\] run finished in (\d+) s \(exit (\d+)\)", txt)
    for u in upat.findall(txt):
        ulog.append({"log": p.name, "site": u[0], "window": u[1], "unit": u[3], "products": u[6],
                     "wall_s": float(u[7]), "cpu_s": float(u[8]), "peak_MB": int(u[9])})
    print(f"{p.name}: started {t0.group(1) if t0 else '?'}, finished in "
          f"{t1.group(1) if t1 else '?'} s (exit {t1.group(2) if t1 else '?'})")
ulog = pd.DataFrame(ulog)
if len(ulog):
    print("\nper-unit wall time (s) from the worker logs present (W3 runs are timing only: blinded):")
    print(ulog.pivot_table(index="unit", columns=["site", "window"], values="wall_s", aggfunc="first")
          .round(1).to_string())

c2 = cost.dropna(subset=["wall_s_median"]).sort_values("wall_s_median")
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
ax.barh(c2["product"] + " (" + c2["method"] + ")", c2["wall_s_median"], height=0.6,
        color=[C_MAREA if p == "L01" else C_OTHER for p in c2["product"]])
ax.set_xscale("log")
ax.set_xlabel("median wall time of the method's own stages (s, log scale)")
for y, v in enumerate(c2["wall_s_median"]):
    ax.text(v * 1.08, y, f"{v:.3g}", va="center", fontsize=7.5, color="#2b2b2a")
ax.set_title(f"Compute time per method, median over instances; MAREA (whole E0 unit) in blue; "
             f"contended runs")
ax.grid(axis="y", visible=False)
plt.show()'''

MD_S9 = r"""## 9 · What is blinded or not computed, and the ledger

W3 is the confirmatory window of family L-P1 (spec §6.5, §9.9): pre-lock it stays BLINDED, no
truth is read and no product is scored there, even where its units ran. The score step lists
the spec §6.4 / §6.5 items milestone V2 does not compute. Every pre-lock look at real 10 m lag
scores is appended to `scores/prelock_ledger.json`, which lock-A copies into the forking-paths
disclosure. Unblinding W3 and lock-A are the user's decisions."""

CODE_S9 = r'''bl = points[points["status"].astype(str).str.startswith("BLINDED")]
print("blinded (lag_meta.json):", [f"{b['site']} {b['window']}" for b in meta["blinded"]],
      "| unblinded:", meta["unblinded"] or "none")
print(bl.groupby(["site", "window"]).size().rename("BLINDED status rows").to_string())
num_cols = ["tau_m", "tau_true", "e", "abs_e", "dE_cm", "LLE_cm", "AGF"]
assert bl[num_cols].isna().all().all(), "a BLINDED row carries a number"
print(f"numbers in the BLINDED rows: none ({len(bl)} rows x {len(num_cols)} columns all empty)")
w3 = units[units["window"] == "W3"].groupby(["site", "state"]).size().unstack(fill_value=0)
print("\nW3 units in status/lag.csv (products that ran there are not scored):")
print(w3.to_string())

sec9 = md.split("## Not computed in milestone V2 (spec §6.4, §6.5)")[1].split("\n## ")[0].split("\n> ")[0]
printed_items = [ln[2:] for ln in sec9.split("\n") if ln.startswith("- ")]
print("\nnot computed (scoring.DEFERRED_ITEMS == the printed list:",
      list(scoring.DEFERRED_ITEMS) == printed_items, "):")
for it in scoring.DEFERRED_ITEMS:
    print("  -", it)
assert list(scoring.DEFERRED_ITEMS) == printed_items

print("\nalso absent from these outputs (counted here):")
print(f"  - interval score IS / truth coverage: {int(points['IS'].notna().sum())} / "
      f"{int(points['covered'].notna().sum())} values in lag_points.csv (they come from the L01 boot30 and "
      "L08 refit runs, which ran only at the blinded W3)")
for pid in ("L15", "L16"):
    st = points.loc[points["product"] == pid, "status"].astype(str).str.split(" \\(").str[0]
    print(f"  - {pid}: {int(points.loc[points['product'] == pid, 'e'].notna().sum())} finite e; statuses "
          f"{sorted(set(st))}")
print(f"  - diag time (sensing offset): {sorted(set(mm['sensing offset']))} at every scored window")
print("  - the dedicated Stage T timing runs (spec §8.2): not part of this run")

lg = pd.DataFrame([{"utc": e["utc"], "git": e["git"]["head"][:12], "dirty": e["git"]["dirty"],
                    "invocation": e["invocation"], "windows_scored": e["windows_scored"],
                    "blinded": len(e["blinded"]), "unblinded": e["unblinded"] or "none",
                    "products": len(e["products_scored"]), "points_sha256": e["points_sha256"][:16]}
                   for e in ledger["entries"]])
print(f"\nprelock_ledger.json ({ledger['format']}):")
print(lg.to_string(index=False))
print("lag_meta.json: lp1.confirmatory =", meta["lp1"]["confirmatory"], "| lp1_claims =",
      meta["lp1_claims"] or "{}", "| warnings =", meta["warnings"] or "none", "| B =", meta["B"])
lock = LAG / "LOCK.json"
print(f"LOCK.json in LAG_DIR: {'present' if lock.exists() else 'absent'} "
      "(lock-A has not been taken: the scores are pre-lock)")
print("\nUnblinding W3 and lock-A are the user's decisions; this notebook does neither.")'''

MD_S10 = r"""## 10 · Integrity

The sha256 of every file this notebook read, re-hashed now: nothing changed while it ran."""

CODE_S10 = r'''integ = pd.DataFrame([{"file": f, "sha256": h, "unchanged": sha256(REPO / f) == h
                       if not Path(f).is_absolute() else sha256(f) == h} for f, h in sorted(READ.items())])
print(integ.to_string(index=False))
assert integ["unchanged"].all()
T_END = dt.datetime.now(dt.timezone.utc)
print(f"\n{len(integ)} files read, all unchanged; run commit {RUN_COMMIT[:12]}; code equal to the run's: "
      f"{CODE_EQUAL_TO_RUN}")
print(f"ended {T_END:%Y-%m-%d %H:%M:%S} UTC ({(T_END - T_START).total_seconds():.0f} s)")'''

#: (kind, cell id, source, tags)
CELLS = [
    ("md", "title", MD_TITLE, ()),
    ("md", "s00", MD_S0, ()),
    ("code", "parameters", CODE_PARAMS, ("parameters",)),
    ("code", "setup", CODE_SETUP, ()),
    ("md", "s01", MD_S1, ()),
    ("code", "s01-run", CODE_S1_RUN, ()),
    ("code", "s01-script", CODE_S1_SCRIPT, ()),
    ("code", "s01-inputs", CODE_S1_INPUTS, ()),
    ("code", "s01-code", CODE_S1_CODE, ()),
    ("code", "s01-versions", CODE_S1_VERSIONS, ()),
    ("md", "s02", MD_S2, ()),
    ("code", "s02-timeline", CODE_S2_TIMELINE, ()),
    ("code", "s02-control", CODE_S2_CONTROL, ()),
    ("code", "s02-units", CODE_S2_UNITS, ()),
    ("code", "s02-unavailable", CODE_S2_UNAVAILABLE, ()),
    ("md", "s03", MD_S3, ()),
    ("code", "s03-windows", CODE_S3_WINDOWS, ()),
    ("code", "s03-judges", CODE_S3_JUDGES, ()),
    ("md", "s04", MD_S4, ()),
    ("code", "s04-helpers", CODE_S4_HELPERS, ("section4-helpers",)),
    ("code", "s04-summary", CODE_S4_SUMMARY, ()),
    ("code", "s04-markdown", CODE_S4_MD, ()),
    ("code", "s04-display", CODE_S4_DISPLAY, ()),
    ("code", "s04-family", CODE_S4_FAMILY, ()),
    ("md", "s05", MD_S5, ()),
    ("code", "s05-mae", CODE_S5_MAE, ()),
    ("code", "s05-judges", CODE_S5_JUDGE, ()),
    ("md", "s06", MD_S6, ()),
    ("code", "s06-offset", CODE_S6, ()),
    ("md", "s07", MD_S7, ()),
    ("code", "s07-repro", CODE_S7, ()),
    ("md", "s08", MD_S8, ()),
    ("code", "s08-cost", CODE_S8, ()),
    ("md", "s09", MD_S9, ()),
    ("code", "s09-blinded", CODE_S9, ()),
    ("md", "s10", MD_S10, ()),
    ("code", "s10-integrity", CODE_S10, ()),
]


def build():
    """The notebook (nbformat v4, fixed cell ids, no outputs)."""
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    for kind, cid, src, tags in CELLS:
        meta = {"tags": list(tags)} if tags else {}
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(src, id=cid, metadata=meta))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(src, id=cid, metadata=meta))
    nbformat.validate(nb)
    return nb


def text(nb=None):
    return nbformat.writes(nb or build()) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
#  execution and HTML
# ─────────────────────────────────────────────────────────────────────────────

def execute(nb, out_path, timeout=1800):
    """Execute ``nb`` with nbclient from the repository root and write it to
    ``out_path`` (also when a cell fails, so the failure can be read)."""
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    err = None
    try:
        client.execute()
    except Exception as e:          # CellExecutionError and kernel failures
        err = e
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out_path))
    return err


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _md_inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def _md_block(src):
    """A small markdown subset (headings, paragraphs, '-' and '1.' lists, quotes)."""
    out, para, lst = [], [], None

    def flush():
        nonlocal para, lst
        if para:
            out.append(f"<p>{_md_inline(' '.join(para))}</p>")
            para = []
        if lst:
            tag = lst[0]
            out.append(f"<{tag}>" + "".join(f"<li>{_md_inline(i)}</li>" for i in lst[1]) + f"</{tag}>")
            lst = None

    for line in src.split("\n"):
        m = re.match(r"^(#{1,4}) (.*)$", line)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{_md_inline(m.group(2))}</h{n}>")
            continue
        m = re.match(r"^(\s*)(-|\d+\.) (.*)$", line)
        if m and not m.group(1):
            if para:
                flush()
            tag = "ul" if m.group(2) == "-" else "ol"
            if lst is None or lst[0] != tag:
                flush()
                lst = (tag, [])
            lst[1].append(m.group(3))
            continue
        if line.startswith(">"):
            flush()
            out.append(f"<blockquote>{_md_inline(line.lstrip('> '))}</blockquote>")
            continue
        if not line.strip():
            flush()
            continue
        if lst is not None and line.startswith("  "):
            lst[1][-1] += " " + line.strip()
            continue
        para.append(line.strip())
    flush()
    return "\n".join(out)


def to_html(nb):
    """HTML of an executed notebook: nbconvert's when it is installed, else a
    small self-contained export (markdown subset, code, text, tables, PNGs)."""
    try:
        from nbconvert import HTMLExporter

        body, _ = HTMLExporter().from_notebook_node(nb)
        return body
    except ImportError:
        pass
    parts = []
    for c in nb.cells:
        if c.cell_type == "markdown":
            parts.append(f'<section class="md">{_md_block(c.source)}</section>')
            continue
        parts.append(f'<pre class="src">{html.escape(c.source)}</pre>')
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                parts.append(f'<pre class="out">{html.escape(o.text)}</pre>')
            elif o.output_type in ("execute_result", "display_data"):
                d = o.get("data", {})
                if "image/png" in d:
                    png = d["image/png"]
                    png = png if isinstance(png, str) else base64.b64encode(png).decode()
                    parts.append(f'<img alt="figure" src="data:image/png;base64,{png.strip()}">')
                elif "text/html" in d:
                    parts.append(f'<div class="html">{d["text/html"]}</div>')
                elif "text/plain" in d:
                    parts.append(f'<pre class="out">{html.escape(d["text/plain"])}</pre>')
            elif o.output_type == "error":
                tb = _ANSI.sub("", "\n".join(o.get("traceback", [])))
                parts.append(f'<pre class="err">{html.escape(tb)}</pre>')
    css = ("body{font:14px/1.5 system-ui,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;"
           "color:#1d1d1c;background:#fcfcfb}pre{font:12px/1.4 ui-monospace,Consolas,monospace;"
           "overflow-x:auto;padding:8px 10px;border-radius:4px}pre.src{background:#f1f0ec}"
           "pre.out{background:#fff;border:1px solid #e4e3df}pre.err{background:#fdeceb}"
           "img{max-width:100%}code{background:#f1f0ec;padding:0 3px;border-radius:3px}"
           "blockquote{border-left:3px solid #2a78d6;margin:8px 0;padding:2px 12px;color:#3a3a38}")
    title = html.escape(nb.cells[0].source.split("\n")[0].lstrip("# ")) if nb.cells else "notebook"
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{css}</style>"
            f"</head><body>{''.join(parts)}</body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true", help="do not write the repository notebook")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the notebook on disk differs from the built one")
    ap.add_argument("--execute", metavar="OUT.ipynb", help="execute a copy into OUT.ipynb (nbclient)")
    ap.add_argument("--html", metavar="OUT.html", help="with --execute: also an HTML export")
    ap.add_argument("--lag-dir", help="sets V2_LAG_DIR for the executed copy")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout (s)")
    a = ap.parse_args(argv)
    nb = build()
    if a.check:
        same = NB_PATH.is_file() and NB_PATH.read_text(encoding="utf-8") == text(nb)
        print(f"{NB_PATH}: {'up to date' if same else 'DIFFERS from the build script'}")
        return 0 if same else 1
    if not a.no_write:
        NB_PATH.write_text(text(nb), encoding="utf-8", newline="\n")
        print(f"written  {NB_PATH}")
    if a.execute:
        if a.lag_dir:
            os.environ["V2_LAG_DIR"] = a.lag_dir
        print(f"executing (V2_LAG_DIR={os.environ.get('V2_LAG_DIR', '<unset: results/validation_v2>')}) ...")
        err = execute(nb, a.execute, timeout=a.timeout)
        print(f"executed {a.execute}" + (f"  FAILED: {str(err)[:400]}" if err else ""))
        if a.html:
            Path(a.html).parent.mkdir(parents=True, exist_ok=True)
            Path(a.html).write_text(to_html(nb), encoding="utf-8", newline="\n")
            print(f"html     {a.html}")
        return 1 if err else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
