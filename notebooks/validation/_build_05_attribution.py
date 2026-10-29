# -*- coding: utf-8 -*-
"""Builds notebooks/validation/05_error_attribution_site.ipynb (unexecuted, fixed cell ids).

Edit this file, never the .ipynb: the cells below are the notebook, in order. Run from the repository root:

    python -X utf8 notebooks/validation/_build_05_attribution.py          # writes the notebook
    python -X utf8 notebooks/validation/_build_05_attribution.py --check  # exit 1 when the notebook on disk differs

The notebook is executed on the server by run_05_on_server.sh.
"""
import os
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "05_error_attribution_site.ipynb")

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# 05 · Error attribution, one site: why the methods without a delay come closer to the bed than MAREA

**The question.** In the n-tables (`01_fixed_config_n_tables_site.ipynb`) the methods that read the boundary level
without any tidal delay (LogiTide2DEM, sampled exceedance, NIDEM, DEA Intertidal, Flood2Topo) reach lower elevation
errors than MAREA on the same images, pixels and water level. This notebook takes those windows apart pixel by
pixel: how much of each method's error is offset, compressed relief or scatter (section 3); what each ingredient of
MAREA (the read-off point of its fit, fitting the NDWI magnitude instead of the wet/dry record, the delay) and of
LogiTide2DEM changes, one step at a time (section 4); where MAREA loses or wins: by delay, distance from the mouth,
elevation, the fit's σ and dry level, the number of looks and the bed change (sections 5 and 6); and what the
worst and best pixels look like (section 7).

**The construction is notebook 01's**, its code unchanged: the production screening, boundary and extraction; the
truth (Vaklodingen cells surveyed in 2023–2025, the most recent survey per cell, dated to the end of its year);
the pixels F; the windows of the n most recent screened images on or before the truth date, by default n = 60 and
50 (Ems: 50 and 40) on the 2025 row, the development windows; `NB_NS=125,100,75` runs the confirmatory windows,
which hold the pre-2023 images (their water level and NIDEM's frames as notebook 01 reads them). Every estimate that
notebook 01 also computes must give its rows bit for bit (section 9).

| id | estimate | |
|---|---|---|
| V0 | MAREA | as published, notebook 01's MAREA: the reference |
| V1 | MAREA-0 | V0's fit read where it crosses NDWI = 0 (notebook 01's *MAREA, z at NDWI = 0*) |
| V2 | MAREA-B | V0's delay; the probit likelihood of the wet/dry record (new here) |
| V10 | MAREA-B with lapse rates | V2 with P(wet) = e + (1 − 2e)Φ((h − z)/σ), e ∈ {0, 0.02, 0.05, 0.10} (new here) |
| V2d | MAREA-B, no delay | V2 with every delay 0 (new here) |
| | MAREA, no delay | V0 with every delay 0 (notebook 01's diagnostic) |
| | LogiTide2DEM, and + MAREA delay | E4, and E4 with each pixel's levels read MAREA's delay earlier (notebook 01's) |
| | sampled exceedance, DEA Intertidal, NIDEM, Flood2Topo | E5c, DEA 2.5.3, NIDEM, E5a as notebook 01 runs them |
| V11, V11d | MAREA-M | V0's delay (V11d: none); the isotonic wet-probability read-off, no curve shape (addendum) |
| V12 | MAREA-P | V0's delay and inputs; NDWI = a + Φ(u)(b + c(h − z)), the rise with depth c ≥ 0 (addendum) |
| V4, V4-0 | MAREA-H | V0 and V1 with a Huber loss (δ = 0.10) and weight 0.5 for SCL 7 (declared 2026-10-27) |
| V13a | MAREA-B, abstaining | V2 with no value where its 95 % likelihood interval is not bracketed (addendum) |
| V14 | + Otsu | wet = NDWI > the scene's Otsu threshold, on MAREA-B, MAREA-BL, MAREA-M, E4, E5c, E12 (addendum) |
| | Tseng, Coast X-Ray, wet/dry bracket | E5b, E7, E12 as notebook 01 runs them (for the error-coverage curves) |

V1, V2, V2d, V9 (σ as an uncertainty, a table for V0 and V1 in section 3) and V10 are declared in
`experiments/variants_preregistered_2026-10-27.md`, V4 too; V11-V14 in `experiments/variants_addendum_2026-10-28.md`,
before any window was computed with them (they run when `NB_METHODS` names them). On the development windows
**everything after V0 that is not one of notebook 01's methods (V1, V2, V10, V2d, V4, V11-V14, MAREA no delay,
LogiTide2DEM + MAREA delay) is exploratory, a diagnostic**, and never replaces a method; on the confirmatory windows
the variants are judged as those files say. The notebook reports observations in tables and figures; it draws no
conclusions.

**Outputs** (`OUT`): `<site>_scores.csv` (every window and estimate: seconds and scores on F),
`<site>_decomposition.csv`, `<site>_sigma.csv`, `<site>_ladder.csv`, `<site>_ladder_steps.csv`,
`<site>_ladder_cases.csv`, `<site>_rivals.csv`, `<site>_strata.csv`, `<site>_delay_summary.csv`,
`<site>_delay_bins.csv`, `<site>_gallery.csv`, with the addendum's estimates `<site>_interval.csv`,
`<site>_coverage.csv` and `<site>_bootstrap.csv` (section 8), the per-pixel arrays `<site>_pixels.npz` and the
figures (PNG, 200 dpi).
""")

md(r"""
## Parameters

Everything that changes between runs lives in the next cell. The environment variables `NB_SITE`, `NB_NS`,
`NB_METHODS`, `NB_EXTRA`, `NB_OUT`, `NB_NTABLES`, `NB_FIRST`, `NB_BED` and `NB_ROW_YEARS` override them (the server
script sets `NB_SITE`). The confirmatory windows: `NB_NS=125,100,75`, with `NB_OUT` a folder of their own and
`NB_NTABLES` notebook 01's run of the same windows. The constants of the last block are notebook 01's.
""")

code(r'''
import os

SITE    = os.environ.get("NB_SITE", "escalda")       # escalda | wadden | ems
NS      = os.environ.get("NB_NS")                    # the windows, e.g. "60,50" or "125,100,75"; unset: the site's
                                                     # development pair (next cell)
METHODS = os.environ.get("NB_METHODS", "MAREA,MAREA_ndwi0,MAREA_B,MAREA_BL,MAREA_B_nodelay,MAREA_nodelay,"
                                       "E4,E4_delay,E5c,DEA,NIDEM,E5a").split(",")
EXTRA   = os.environ.get("NB_EXTRA", "auto")         # pre-2023 scenes, as notebook 01 reads them ("auto", "none", a path)
OUT     = os.environ.get("NB_OUT", "results/attribution_nb")        # the CSVs, the npz and the figures go here
NTABLES = os.environ.get("NB_NTABLES", "results/fixedcfg_nb_diag")  # notebook 01 with its variants: the last check
FIRST   = os.environ.get("NB_FIRST", "results/fixedcfg_nb")         # ... else notebook 01's first run (nine methods)
BED     = os.environ.get("NB_BED", "results/bed_change_nb")         # notebook 04's change class per pixel (optional)
ROW_YEARS = [int(v) for v in os.environ.get("NB_ROW_YEARS", "2025").split(",")]   # the survey-year rows analysed
ONE_ROW = os.environ.get("NB_ONE_ROW", "1") == "1"   # 1: one row per site (the Wadden's 2023-2025 surveys together)

# ── The analysis ────────────────────────────────────────────────────────
Q_BINS      = 5           # quantile bins of the strata (section 5); the positive delays get 4 (section 6)
GALLERY     = (12, 4)     # the pixels where V0 errs most / least relative to LogiTide2DEM (section 7)
B_MIN_LOOKS = 8           # MAREA-B: at least 8 clear looks, one wet and one dry (the variants file)
B_TIE_TOL   = 1e-9        # MAREA-B: z levels whose profile log-likelihood is within this of the maximum are tied
B_LAPSES    = (0.0, 0.02, 0.05, 0.10)   # V10: the lapse rates e, maximised with z and sigma (the variants file)

# ── Fixed by notebook 01 ────────────────────────────────────────────────
MAX_PX            = 200_000   # a larger F is v19's seeded random subsample
SEED              = 20261019
RANGE_PAD_M       = 0.25      # truth kept inside the production no-delay range ± 0.25 m
MAREA_MIN_BRACKET = 0         # the rule of the published 10 m products (notebook 01's MAREA)
DEA_TMP = os.environ.get("MAREA_DEA_TMP")             # DEA's temp root, outside the repository (default below)
''')

code(r'''
# ── Each site as notebook 01 runs it ────────────────────────────────────
def dutch(site, n_gauges, judges, mouth_side, truth_years, ns):
    d, tag = f"products_{site}_10m", f"eot20_g{n_gauges}"
    return dict(cache=f"ndwi_cube_{site}_2023-2025_10m.nc", extract=f"{d}/marea_extract.npz",
                overpass=f"{d}/overpass_times.json", marea=f"{d}/marea_{tag}",
                nodelay=(f"{d}/comparison_{tag}/products.npz", "z_plain"),
                period=("2023-01-01", "2025-12-31"), n_gauges=n_gauges, judges=judges,
                mouth_side=mouth_side, truth_years=truth_years, ns=ns)


SITES = {
    "escalda": dutch("escalda", 2, ["trnz"], "west", [2025], (60, 50)),
    "wadden":  dutch("wadden", 1, ["harl"], "north+west", [2025, 2024, 2023], (60, 50)),
    "ems":     dutch("ems", 1, ["delf"], "north+west", [2025], (50, 40)),    # 54 screened scenes: 60 is not reachable
}
P = SITES[SITE]
NS = [int(v) for v in NS.split(",")] if NS else list(P["ns"])
print(f"{SITE}: windows n = {NS} on the rows of {ROW_YEARS}; estimates {METHODS}")
print(f"  cube {P['cache']} · extraction {P['extract']} · boundary EOT20 + {P['n_gauges'] or 0} gauge(s), "
      f"held out {P['judges']}")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 0 · Setup

As notebook 01: every path is relative to the repository root (opened from `notebooks/validation/`, the first lines
move there); no Copernicus connection is needed; DEA's temp root defaults to `v19_tmp` next to the repository. The
second cell fixes the labels, the roles and the figure style (the muted palette; MAREA in the accent colour).
""")

code(r'''
import sys
while not os.path.isdir("pyintertidal") and os.path.dirname(os.getcwd()) != os.getcwd():
    os.chdir("..")                                    # from notebooks/validation/ to the repository root
sys.path.insert(0, os.getcwd())

import glob, json, time, types, zlib
from collections import Counter
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import log_ndtr, ndtr, ndtri      # log Φ, Φ and Φ⁻¹ of the standard normal
from scipy.stats import spearmanr

import pyintertidal as pit
from pyintertidal import SentinelCube, marea
from pyintertidal import tide_estimators as te
from pyintertidal import waterline_contour as wc
from pyintertidal.elevation import _fit_block           # the per-pixel fit inside marea.invert_series
from pyintertidal.boundary import make_boundary, nearest_gauges
from experiments.fixedcfg import methods as fm
from experiments.fixedcfg import dea_intertidal as dea
from experiments.fixedcfg.elev_base import NotComputable
from experiments.fixedcfg.level_series import LevelSeries
from experiments import light_to_extraction as lte     # the production rule of the boundary before 2023
from experiments import zarr_cubes as zc               # the band-subset Zarr stores of the pre-2023 years

pit.net.use_system_certificates()
os.makedirs(OUT, exist_ok=True)
DEA_TMP = DEA_TMP or os.path.join(os.path.dirname(os.getcwd()), "v19_tmp")
print("repository", os.getcwd(), "· pyintertidal", pit.__version__, "· DEA temp root", DEA_TMP)
''')

code(r'''
LABELS = {"MAREA": "V0 MAREA", "MAREA_ndwi0": "V1 MAREA-0", "MAREA_B": "V2 MAREA-B",
          "MAREA_BL": "V10 MAREA-B, lapse rates", "MAREA_B_nodelay": "V2d MAREA-B, no delay",
          "MAREA_nodelay": "MAREA, no delay", "E4": "LogiTide2DEM", "E4_delay": "LogiTide2DEM + MAREA delay",
          "E5c": "sampled exceedance", "DEA": "DEA Intertidal", "NIDEM": "NIDEM", "E5a": "Flood2Topo",
          # the addendum of 2026-10-28 (run when NB_METHODS names them) and the competitors its curves need
          "MAREA_M": "V11 MAREA-M", "MAREA_M_nodelay": "V11d MAREA-M, no delay", "MAREA_P": "V12 MAREA-P",
          "MAREA_H": "V4 MAREA-H (Huber)", "MAREA_H0": "V4-0 MAREA-H, NDWI = 0",
          "MAREA_B_abst": "V13a MAREA-B, abstaining", "MAREA_B_otsu": "V2 MAREA-B + Otsu",
          "MAREA_BL_otsu": "V10 MAREA-BL + Otsu", "MAREA_M_otsu": "V11 MAREA-M + Otsu",
          "E4_otsu": "LogiTide2DEM + Otsu", "E5c_otsu": "sampled exceedance + Otsu", "E12_otsu": "wet/dry bracket + Otsu",
          "E5b": "Tseng linear scaling", "E7": "Coast X-Ray", "E12": "wet/dry bracket"}
SHORT = {"MAREA": "V0", "MAREA_ndwi0": "V1", "MAREA_B": "V2", "MAREA_BL": "V10", "MAREA_B_nodelay": "V2d",
         "MAREA_nodelay": "V0 τ=0", "E4": "LT2D", "E4_delay": "LT2D+τ", "E5c": "exceed.", "DEA": "DEA", "NIDEM": "NIDEM",
         "E5a": "F2T", "MAREA_M": "V11", "MAREA_M_nodelay": "V11d", "MAREA_P": "V12", "MAREA_H": "V4", "MAREA_H0": "V4-0",
         "MAREA_B_abst": "V13a", "MAREA_B_otsu": "V2+O", "MAREA_BL_otsu": "V10+O", "MAREA_M_otsu": "V11+O",
         "E4_otsu": "LT2D+O", "E5c_otsu": "exceed.+O", "E12_otsu": "WDB+O", "E5b": "Tseng", "E7": "CXR", "E12": "WDB"}
ROLE = {m: "diagnostic" for m in ("MAREA_ndwi0", "MAREA_B", "MAREA_BL", "MAREA_B_nodelay", "MAREA_nodelay",
                                  "E4_delay", "MAREA_M", "MAREA_M_nodelay", "MAREA_P", "MAREA_H", "MAREA_H0",
                                  "MAREA_B_abst", "MAREA_B_otsu", "MAREA_BL_otsu", "MAREA_M_otsu", "E4_otsu",
                                  "E5c_otsu", "E12_otsu")}
ROLE.update({"MAREA": "reference", **{m: "method" for m in ("E4", "E5c", "DEA", "NIDEM", "E5a", "E5b", "E7", "E12")}})
ORDER = [m for m in LABELS if m in METHODS]
NOT_COMMON = {"MAREA_B_abst"}      # an abstaining estimate never narrows the common pixels of the others

PAL = dict(deep="#2F5D7C", shallow="#8DB3C7", sand="#C9A66B", dark_sand="#8C6D46", seagrass="#5B7F3A",
           grey="#B8B2A3", accent="#B5523B", ink="#2B2B2B", secondary="#6B6B6B")
COLORS = {"MAREA": PAL["accent"], "MAREA_nodelay": PAL["accent"], "MAREA_ndwi0": PAL["sand"],
          "MAREA_B": PAL["dark_sand"], "MAREA_BL": PAL["dark_sand"], "MAREA_B_nodelay": PAL["dark_sand"],
          "E4": PAL["deep"], "E4_delay": PAL["deep"], "E5c": PAL["seagrass"], "DEA": PAL["shallow"], "NIDEM": PAL["grey"],
          "E5a": PAL["secondary"], "MAREA_M": PAL["accent"], "MAREA_M_nodelay": PAL["accent"], "MAREA_P": PAL["accent"],
          "MAREA_H": PAL["sand"], "MAREA_H0": PAL["sand"], "MAREA_B_abst": PAL["dark_sand"],
          "MAREA_B_otsu": PAL["dark_sand"], "MAREA_BL_otsu": PAL["dark_sand"], "MAREA_M_otsu": PAL["accent"],
          "E4_otsu": PAL["deep"], "E5c_otsu": PAL["seagrass"], "E12_otsu": PAL["ink"], "E5b": PAL["grey"],
          "E7": PAL["secondary"], "E12": PAL["ink"]}
MARKERS = {"MAREA": "o", "MAREA_nodelay": "o", "MAREA_ndwi0": "D", "MAREA_B": "s", "MAREA_BL": "h",
           "MAREA_B_nodelay": "s", "E4": "^", "E4_delay": "^", "E5c": "v", "DEA": "P", "NIDEM": "X", "E5a": "<",
           "MAREA_M": "*", "MAREA_M_nodelay": "*", "MAREA_P": "p", "MAREA_H": "d", "MAREA_H0": "d", "MAREA_B_abst": "s",
           "MAREA_B_otsu": "s", "MAREA_BL_otsu": "h", "MAREA_M_otsu": "*", "E4_otsu": "^", "E5c_otsu": "v",
           "E12_otsu": "x", "E5b": ">", "E7": "8", "E12": "x"}
WSTYLE = [("-", "o", PAL["ink"]), ("--", "s", PAL["secondary"]), (":", "^", PAL["grey"])]   # one per window
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
                     "axes.edgecolor": PAL["secondary"], "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"],
                     "xtick.color": PAL["secondary"], "ytick.color": PAL["secondary"], "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": PAL["grey"], "grid.alpha": 0.35, "grid.linewidth": 0.5, "legend.frameon": False,
                     "legend.fontsize": 7, "lines.linewidth": 1.3, "lines.markersize": 4.5, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "figure.dpi": 100})
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 1 · The site, the screened scenes, the truth and the windows

Notebook 01's sections 1 to 8, cell for cell: the polygon and the gauges (the N nearest IOC gauges not held out as
judges), the rows (one per survey year), the cached cube, the production screening, the boundary and the screened
list S (checked against the production MAREA record), the optional pre-2023 scenes, the extraction, the mouth distance
and the 3-km bands, the truth and F, the windows, the levels h_B(t − τ) and the 1-min level series of Flood2Topo.
Only the windows of the rows in `ROW_YEARS` are run: at the Wadden the 2024 and 2023 survey groups hold 37 and 20
screened images from 2023–2025, so no window of 50 images reaches them without pre-2023 scenes. F is built from every
row, as in notebook 01, so the pixels are its pixels.
""")

code(r'''
aoi = pit.sites.get(SITE)
lat_c, lon_c = aoi.centroid
print(aoi, f"· centroid {lat_c:.3f}, {lon_c:.3f}")
near = nearest_gauges(lat_c, lon_c, 6)
members = [g[0] for g in near if g[0] not in P["judges"]][:(P["n_gauges"] or 0)]
for code, name, km in near:
    role = "boundary" if code in members else "judge (held out)" if code in P["judges"] else "unused"
    print(f"  {code:5s} {name:28s} {km:5.1f} km   {role}")
''')

code(r'''
years = P["truth_years"]
if ONE_ROW:      # one row per site: every cell with its most recent survey of these years, one window to the last
    ROWS = [dict(site=SITE, year=years[0], end=f"{years[0]}-12-31")]
else:
    ROWS = [dict(site=f"{SITE}-{y}" if len(years) > 1 else SITE, year=y, end=f"{y}-12-31") for y in years]
for row in ROWS:
    print(f"row {row['site']:12s} truth dated {row['end']}: its windows end there"
          + ("" if row["year"] in ROW_YEARS else " (not analysed here)"))
''')

code(r'''
cube = SentinelCube(aoi, P["period"], water="ndwi", cache_path=P["cache"], resolution=10)
cube.ensure(None)                                      # cached: no connection needed
transform, crs = cube.grid
H, W = cube.shape
print(f"{len(cube.dates)} scenes {cube.dates[0]} → {cube.dates[-1]} · grid {H}×{W} @ {abs(transform.a):g} m")
''')

md(r"""
### The screening and the screened list S

The production screening (reference map voted by the clean scenes, then each scene's cloud share over the
transition zone; usable at most 10 % cloudy), the production boundary, S (usable, with an overpass time and a finite
level), and the optional pre-2023 scenes of `experiments/light_to_extraction.py`, put before S. With them, as in
notebook 01, `boundary` becomes a `SplitBoundary`: before 2023-01-01 the production rule of those years,
`pre2023_boundary` (production's object with its gauge members extended by the IOC records of those years, demeaned
by production's constants), from then on production's object, chosen instant by instant; every level is read from it.
""")

code(r'''
t0 = time.perf_counter()
reference_map, cloud_pct = pit.reference_and_clouds(
    cube, threshold=0.0, bad_classes=pit.water.BAD_CLASSES, clean_scene_max_bad=0.05,
    stable_threshold=0.95, transition_buffer_px=10)
usable = pit.filter_dates(cloud_pct, cloud_threshold=0.10)
for code, name in [(0, "transition"), (1, "stable water"), (2, "stable land")]:
    print(f"  class {code} ({name:13s}): {int((reference_map == code).sum()):>10,} px")
print(f"{len(usable)} of {len(cloud_pct)} scenes usable  [{time.perf_counter() - t0:.0f} s]")
''')

code(r'''
boundary = make_boundary(aoi, tide_model="EOT20", n_gauges=P["n_gauges"], period=P["period"],
                         exclude=P["judges"])
print(f"boundary: {boundary.name}")
overpass = {d: pd.Timestamp(v) for d, v in json.load(open(P["overpass"])).items()}
cand = [d for d in usable if d in overpass]
t_cand = pd.DatetimeIndex([overpass[d] for d in cand]).tz_localize(None)
ok = np.isfinite(np.asarray(boundary.levels(t_cand), float))
S_dates, S_t = np.array(cand)[ok], t_cand[ok]
print(f"S: {len(S_dates)} screened scenes {S_dates[0]} → {S_dates[-1]} ({len(usable) - len(cand)} usable "
      f"scenes without an overpass time, {int((~ok).sum())} without a finite level)")

record = [d[:10] for d in json.load(open(f"{P['marea']}/result.json"))["record_dates"]]
part = [d for d in S_dates if "2023-01-01" <= d <= "2025-12-31"]
print(f"S in 2023-2025 is the production MAREA record ({len(record)} scenes): {part == record}")
''')

code(r'''
class SplitBoundary:
    """A boundary that reads `before` at the instants before `cut` and `after` from `cut` on, instant by instant.
    Each is called once, on its own instants in the order given (sorted in, sorted out, as production's object
    needs)."""

    def __init__(self, before, after, cut):
        self.before, self.after, self.cut = before, after, pd.Timestamp(cut)
        self.name = f"{after.name} (before {self.cut.date()}: {before.name})"

    def levels(self, times):
        t = pd.DatetimeIndex(times)
        if t.tz is not None:
            t = t.tz_localize(None)
        early = np.asarray(t < self.cut)
        h = np.full(len(t), np.nan)
        if (~early).any():
            h[~early] = np.asarray(self.after.levels(t[~early]), float)
        if early.any():
            h[early] = np.asarray(self.before.levels(t[early]), float)
        return h
''')

code(r'''
found = sorted(glob.glob(f"results/fixedcfg/extra/{SITE}_extract_*.npz"))
EXTRA_PATH = (found[-1] if found else None) if EXTRA == "auto" else (None if EXTRA == "none" else EXTRA)
x_dates = np.array([], str)
boundary_prod = getattr(boundary, "after", boundary)   # production's boundary object (also when the cell runs again)
if EXTRA_PATH:
    side = json.load(open(EXTRA_PATH[:-4] + ".json", encoding="utf-8"))
    boundary_pre, pre_info = lte.pre2023_boundary(SITE, boundary_prod, lte.PRODUCTION_GAUGE_CACHE,
                                                  lte.PRE2023_GAUGE_CACHE)
    boundary = SplitBoundary(boundary_pre, boundary_prod, P["period"][0])
    print(f"boundary before {P['period'][0]}: {boundary_pre.name}; gauge weights {pre_info['weights']}")
    cloud_x = {d: p for d, p in side["cloud_pct"].items() if p is not None and d < P["period"][0]}
    cand_x = [d for d in pit.filter_dates(cloud_x, cloud_threshold=0.10) if side["overpass"].get(d)]
    t_x = pd.DatetimeIndex([pd.Timestamp(side["overpass"][d]) for d in cand_x]).tz_localize(None)
    ok_x = np.isfinite(np.asarray(boundary.levels(t_x), float))
    x_dates = np.array(cand_x)[ok_x]
    S_dates, S_t = np.concatenate([x_dates, S_dates]), t_x[ok_x].append(S_t)
    print(f"{EXTRA_PATH}: {len(x_dates)} screened scenes before {P['period'][0]} put before S "
          f"-> {len(S_dates)} scenes {S_dates[0]} → {S_dates[-1]}")
else:
    print("no extra extraction: the windows hold the cube's scenes only; every level is production's boundary")
''')

md(r"""
### The extraction, the mouth distance and the bands

`marea_extract.npz`: the production pixels, the NDWI of every scene (NaN where undefined) and the clear mask
C = SCL in {4, 5, 6, 7}; only the rows of S stay in memory. The mouth distance s and MAREA's 3-km bands, as
`marea.reconstruct` computes them.
""")

code(r'''
ex = np.load(P["extract"])
keep = ex["keep"]                                      # flat indices of the pixels in the H x W frame
pos = {d: i for i, d in enumerate(ex["dates"].astype(str))}
print(f"{len(keep):,} pixels · {len(pos)} scenes · the cube's dates: {list(pos) == list(cube.dates)}")
main = [pos[d] for d in S_dates[len(x_dates):]]
Y_S, C_S = ex["Y"][main], ex["C"][main]                # (scenes of S, pixels): NDWI and clear
if len(x_dates):
    X = np.load(EXTRA_PATH)
    assert np.array_equal(X["keep"], keep), "the extra extraction is on other pixels"
    xpos = {d: i for i, d in enumerate(X["dates"].astype(str))}
    rows_x = [xpos[d] for d in x_dates]
    Y_S, C_S = np.concatenate([X["Y"][rows_x], Y_S]), np.concatenate([X["C"][rows_x], C_S])
print(f"Y and C of S: {Y_S.shape[0]} scenes × {Y_S.shape[1]:,} pixels")
''')

code(r'''
seeds = pit.geometry.mouth_seeds(ex["sea"], side=P["mouth_side"])
s_km = pit.geometry.along_distance(ex["sea"] | ex["inter"], seeds, 10).ravel()[keep] / 1000.0
BAND_KM = 3.0
_, centers, band_of = te.make_bands_fixed(s_km, BAND_KM, 500)
production = np.load(f"{P['marea']}/marea.npz")
print(f"{len(centers)} bands of {BAND_KM:g} km, centres {np.round(centers, 1)} km; "
      f"the production product's bands: {np.array_equal(band_of, production['band'])}")
''')

md(r"""
### The truth and the pixels F

Every pixel reads the 20 m Vaklodingen cell under it (`sample_truth`), year by year from 2025 down to 2023: the most
recent survey with a value gives its truth and its survey year. The truth is kept inside the production no-delay
range ± 0.25 m; F is those pixels in the survey years of the rows; above 200,000 pixels, v19's seeded subsample.
""")

code(r'''
import pyproj
from experiments.v0_vaklodingen_fetch import sheets_for
from experiments.v1_vaklodingen_validate import sample_truth


def vaklodingen(year):
    """{sheet: (x, y, z, year)} of the cells surveyed in `year` (a later survey of that year wins)."""
    cells = {}
    for name, _, _ in sheets_for(SITE):
        path = f"data_v4/truth/vaklodingen/{name}"
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            k = np.flatnonzero(ds["time"].dt.year.values == year)
            if not len(k):
                continue
            z, x, y = ds["z"].isel(time=k).values, ds["x"].values, ds["y"].values
        zt = np.full(z.shape[1:], np.nan)
        for zk in z:
            zt[np.isfinite(zk)] = zk[np.isfinite(zk)]
        cells[name] = (x, y, zt, np.full(zt.shape, year))
    return cells
''')

code(r'''
with xr.open_dataset(P["cache"]) as ds:
    xs, ys = ds["x"].values, ds["y"].values
to_rd = pyproj.Transformer.from_crs(crs, "EPSG:28992", always_xy=True)
xq, yq = to_rd.transform(xs[keep % W], ys[keep // W])
truth, group = np.full(len(keep), np.nan), np.zeros(len(keep), int)
for year in (2025, 2024, 2023):                        # the most recent survey with a value wins
    v, _ = sample_truth(vaklodingen(year), xq, yq)
    new = np.isfinite(v) & (group == 0)
    truth[new], group[new] = v[new], year
survey = group.copy()                                  # the year of each cell's survey
if ONE_ROW:
    group[np.isin(group, P["truth_years"])] = ROWS[0]["year"]   # the site's truth years in its one row; a cell
                                                       # surveyed in another year stays out, as per year
''')

code(r'''
path, key = P["nodelay"]
z_nodelay = np.load(path)[key]
lo, hi = np.nanmin(z_nodelay) - RANGE_PAD_M, np.nanmax(z_nodelay) + RANGE_PAD_M
in_range = (truth >= lo) & (truth <= hi)
row_years = [row["year"] for row in ROWS]
for g in np.unique(group[np.isfinite(truth)]):
    print(f"  survey {g or 'LiDAR'}: {int((group == g).sum()):,} px with a value, "
          f"{int(((group == g) & in_range).sum()):,} inside [{lo:+.2f}, {hi:+.2f}] m"
          f" ({'a row' if g in row_years else 'left out'})")

F = np.flatnonzero(in_range & np.isin(group, row_years))
if len(F) > MAX_PX:                                    # v19's seeded subsample: the same pixels as v19
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(SITE.encode()) & 0x7FFFFFFF, 7]))
    F = F[np.sort(rng.permutation(len(F))[:MAX_PX])]
for row in ROWS:
    row["px"] = F[group[F] == row["year"]]
    print(f"row {row['site']:12s}: F = {len(row['px']):,} pixels")
''')

md(r"""
### The windows and the water levels

Window(n) is the n most recent images of S on or before the row's truth date. The levels h_B(t − τ) are read once for
every scene of S and every whole minute τ from −45 to 180 (MAREA's lag bank and its per-pixel delays; τ = 0 is what
every method without a delay reads); `make_obs` gives a method its inputs. Flood2Topo also gets the 1-min level series.
""")

code(r'''
windows = []
for row in ROWS:
    if row["year"] not in ROW_YEARS:
        continue
    upto = np.flatnonzero(S_dates <= row["end"])
    for n in NS:
        win = dict(site=row["site"], year=row["year"], end=row["end"], n=n, px=row["px"], available=len(upto),
                   scenes=upto[-n:] if len(upto) >= n else None)
        windows.append(win)
        if win["scenes"] is None:
            print(f"{row['site']:12s} n={n:3d}: not reachable, {len(upto)} screened images on or before {row['end']}")
        else:
            d = S_dates[win["scenes"]]
            per_year = dict(sorted(Counter(str(x)[:4] for x in d).items()))
            print(f"{row['site']:12s} n={n:3d}: {d[0]} → {d[-1]}  images per year {per_year}")
''')

code(r'''
shifted = {tau: S_t - pd.Timedelta(minutes=tau) for tau in range(-45, 181)}
inst = pd.DatetimeIndex(np.unique(np.concatenate([s.values for s in shifted.values()])))
lev = np.asarray(boundary.levels(inst), float)
LEV = {tau: lev[np.searchsorted(inst.values, s.values)] for tau, s in shifted.items()}
h_direct = np.asarray(boundary.levels(S_t), float)
print(f"h_B(t - tau) at {len(S_t)} scenes, tau = -45..180 min; at tau = 0 it is the level read directly "
      f"(max |diff| {np.max(np.abs(LEV[0] - h_direct)):.1e} m); range [{LEV[0].min():+.2f}, {LEV[0].max():+.2f}] m")


def level_at(tau):
    """h_B(t - tau) at every scene of S; a delay outside the table is read from the boundary itself."""
    if tau not in LEV:
        LEV[tau] = np.asarray(boundary.levels(S_t - pd.Timedelta(minutes=tau)), float)
    return LEV[tau]


def make_obs(win, px, tau=0):
    """The common inputs of a window at the pixels px (positions in keep), with the levels read tau minutes
    earlier (0 for every method; the '+ MAREA delay' variants pass MAREA's delay)."""
    w = win["scenes"]
    Y, C = Y_S[np.ix_(w, px)], C_S[np.ix_(w, px)]
    with np.errstate(invalid="ignore"):
        wet = C & (Y > 0)                              # an undefined NDWI is never wet
    return types.SimpleNamespace(Y=Y, C=C, W=wet, h=level_at(tau)[w], t=S_t[w], keep=keep[px],
                                 shape=(H, W), pixel_m=10.0, site=SITE)
''')

code(r'''
LS = None
reach = [w for w in windows if w["scenes"] is not None]
if reach and any("level_series" in fm.METHODS[m][2].get("needs", []) for m in METHODS if m in fm.METHODS):
    t0 = time.perf_counter()
    first = min(S_t[w["scenes"][0]] for w in reach).normalize()
    last = pd.Timestamp(max(w["end"] for w in reach)) + pd.Timedelta(days=1)
    LS = LevelSeries.from_boundary(boundary, first, last)
    print(f"1-min level series {first.date()} → {last.date()}: {len(LS.h):,} levels  [{time.perf_counter() - t0:.0f} s]")
''')

md(r"""
### The water level before 2023: the checks

With pre-2023 scenes: (1) the levels of the 2023–2025 scenes, at every delay of the table, are those production's
boundary gives in a run without pre-2023 scenes (one call over the same instants), bit for bit; (2) the levels of
the pre-2023 scenes without delay are the extraction's record (`levels_pre2023`, written by
`light_to_extraction.py build` with the same `pre2023_boundary`), bit for bit, and the record's boundary, tide
member, gauge weights and demeaning constants are those built here; (3) the 1-min level series switches from
`pre2023_boundary` to production's boundary at 2023-01-01 (the two days around it, bit for bit).
""")

code(r'''
if len(x_dates):
    cut = boundary.cut
    post = np.flatnonzero(S_t >= cut)                  # the 2023-2025 scenes of S
    sh = {tau: S_t[post] - pd.Timedelta(minutes=tau) for tau in range(-45, 181)}
    inst_p = pd.DatetimeIndex(np.unique(np.concatenate([s.values for s in sh.values()])))
    lev_p = np.asarray(boundary_prod.levels(inst_p), float)          # as a run without pre-2023 scenes reads them
    assert all(np.array_equal(LEV[tau][post], lev_p[np.searchsorted(inst_p.values, s.values)])
               for tau, s in sh.items()), "the 2023-2025 levels are not production's"
    print(f"2023-2025: the levels of its {len(post)} scenes at tau = -45..180 min are production's, bit for bit")

    rec = side["levels_pre2023"]
    for k in ("boundary", "production_boundary", "tide_member", "weights"):
        assert rec[k] == pre_info[k], f"the extraction's {k} is {rec[k]!r}, here {pre_info[k]!r}"
    for g, v in pre_info["gauges"].items():
        for k in ("production_median_m", "pre2023_median_m", "samples"):
            assert rec["gauges"][g][k] == v[k], f"gauge {g}: the extraction's {k} is {rec['gauges'][g][k]}, here {v[k]}"
    have = [i for i, d in enumerate(x_dates) if (rec["per_date"].get(d) or {}).get("h_B") is not None]
    h_rec = np.array([rec["per_date"][x_dates[i]]["h_B"] for i in have], float)
    assert np.array_equal(LEV[0][have], h_rec), \
        f"the pre-2023 levels are not the extraction's record (max |diff| {np.max(np.abs(LEV[0][have] - h_rec)):.1e} m)"
    kinds = Counter(rec["per_date"][x_dates[i]]["members"] for i in have)
    print(f"before 2023: the levels of {len(have)} of {len(x_dates)} scenes are the extraction's record "
          f"(levels_pre2023), bit for bit, with its boundary, tide member, gauge weights and demeaning constants; "
          f"members at those scenes {dict(kinds)}")

    if LS is not None and LS.t[0] < cut:
        a, b = LS.t.searchsorted(cut - pd.Timedelta(days=1)), LS.t.searchsorted(cut + pd.Timedelta(days=1))
        t = LS.t[a:b]
        h = np.concatenate([boundary_pre.levels(t[t < cut]), boundary_prod.levels(t[t >= cut])])
        assert np.array_equal(LS.h[a:b], h), "the 1-min series does not switch boundary at the cut"
        print(f"1-min series: its {b - a:,} levels around {cut.date()} are pre2023_boundary's before it and "
              "production's from it, bit for bit")
else:
    print("no pre-2023 scene: every level is production's boundary")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 2 · The estimates, per pixel

**V0 MAREA**, as notebook 01 runs it: the lag of every 3-km band refitted on the window's images by the profiled
Bernoulli likelihood (`te.m2a_rasch`, production's settings from `configs/m2.yaml`, its seeded draw), a negative band
lag set to 0, the profile smoothed along the mouth distance and rounded to the minute (τ_px); every pixel inverted
with the level read τ_px minutes earlier (`marea.invert_series`: the least-squares fit NDWI = a + bΦ((h − z)/σ) on a
grid, elevation = the midpoint z), between the window's own lowest and highest level. `marea_delays` is notebook
01's function; here it runs once per window (`window_delays`, the same seeded fit) and every delayed estimate reuses
it, so its seconds are a row of their own (`delay_fit`).
""")

code(r'''
M2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
FIT_KW = dict(sigma0=M2["sigma0_m"], sg_grid=tuple(M2["sigma_perfil_m"]), tau_bounds=marea.TAU_BOUNDS,
              z_points=M2["z_puntos"], max_px_band=M2["max_px_banda"]["m2a"], monotone=True)


def marea_delays(win):
    """MAREA's band lags on the window's images, and the delay of every pixel of keep to the minute."""
    w = win["scenes"]
    bank = te.ShiftBank(marea.BANK_TAUS, [LEV[int(tau)][w] for tau in marea.BANK_TAUS])
    with np.errstate(invalid="ignore"):
        wet = C_S[w] & (Y_S[w] > 0)
    fit = te.m2a_rasch(wet, C_S[w], bank, band_of, centers, len(centers),
                       rng=np.random.default_rng(M2["seed"] + 31), **FIT_KW)
    tau_bands = np.clip(np.asarray(fit["tau"], float), 0, None)       # causality: a negative band lag is 0
    tau_px = te.smooth_lag_profile(centers, tau_bands, s_km, bandwidth_km=BAND_KM)
    return tau_bands, np.where(np.isfinite(tau_px), np.round(tau_px), 0.0)


def lags_text(tau_bands):
    return "band lags " + "/".join(f"{v:.0f}" for v in tau_bands) + " min"


DELAYS = {}


def window_delays(win):
    """marea_delays of the window, fitted once (the fit is seeded: every call would give the same delays)."""
    key = (win["site"], win["n"])
    if key not in DELAYS:
        DELAYS[key] = marea_delays(win)
    return DELAYS[key]


def run_marea(win, px):
    """V0: notebook 01's run_marea (its value and its note), with the window's delays fitted once."""
    w = win["scenes"]
    tau_bands, tau_px = window_delays(win)
    tau_px = tau_px[px]
    h0 = LEV[0][w]
    Y0, C0 = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C_S[np.ix_(w, px)]
    z, sigma = np.full(len(px), np.nan), np.full(len(px), np.nan)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols], sigma[cols] = marea.invert_series(Y0[:, cols], C0[:, cols], LEV[int(tau)][w], h0.min(), h0.max(),
                                                   min_bracket=MAREA_MIN_BRACKET)
    return z, lags_text(tau_bands), None, {"sigma": sigma}
''')

md(r"""
**V1 MAREA-0** (diagnostic): V0's fit, read where the fitted curve crosses NDWI = 0: h0 = z + σΦ⁻¹(−a/b), no value
when 0 is not strictly between a and a + b. `marea_fit` is notebook 01's `invert_ndwi0` keeping the whole fit (a, b,
σ before the guards, and the midpoint), so that V0's a, b and σ are saved per pixel; every window asserts that its
midpoint is V0's z bit for bit.
""")

code(r'''
def marea_fit(Y, C, h, lo, hi, min_bracket=0):
    """Notebook 01's invert_ndwi0, keeping the fit: dict(z0, z_mid, a, b, sigma). The fit is marea.invert_series'
    (_fit_block with its z grid, sigma grid, blocks of 4,000 pixels and guards): z_mid is its midpoint (NaN where
    the guards reject), z0 the level where the fitted NDWI = a + b Phi((h - z)/sigma) crosses 0 (NaN when 0 is not
    strictly between a and a + b); a, b and sigma are the fit's, before the guards."""
    out = {k: np.full(Y.shape[1], np.nan) for k in ("z0", "z_mid", "a", "b", "sigma")}
    grid = np.linspace(lo, hi, 50)
    for j in range(0, Y.shape[1], 4000):
        s = slice(j, min(j + 4000, Y.shape[1]))
        a, b, mu, sg, _, N = _fit_block(np.asarray(Y[:, s], np.float64), np.asarray(C[:, s], np.float64),
                                        h, grid, marea.SG_GRID)
        ok = (N >= 8) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)          # invert_series' guards
        if min_bracket:
            clear, hh = np.asarray(C[:, s]) > 0, np.asarray(h, float)[:, None]
            below, above = (clear & (hh < mu)).sum(axis=0), (clear & (hh > mu)).sum(axis=0)
            ok &= (below >= min_bracket) & (above >= min_bracket)
        cross = ok & (a < 0) & (a + b > 0)
        out["z_mid"][s] = np.where(ok, mu, np.nan)
        out["z0"][s] = np.where(cross, mu + sg * ndtri(np.where(cross, -a / np.where(cross, b, 1.0), 0.5)), np.nan)
        out["a"][s], out["b"][s], out["sigma"][s] = a, b, sg
    return out


def run_marea_ndwi0(win, px):
    """V1: notebook 01's run_marea_ndwi0 (its value and its note); the fit's a, b, sigma and midpoint as extras."""
    w = win["scenes"]
    tau_bands, tau_px = window_delays(win)
    tau_px = tau_px[px]
    h0 = LEV[0][w]
    Y0, C0 = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C_S[np.ix_(w, px)]
    fit = {k: np.full(len(px), np.nan) for k in ("z0", "z_mid", "a", "b", "sigma")}
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        part = marea_fit(Y0[:, cols], C0[:, cols], level_at(int(tau))[w], h0.min(), h0.max(),
                         min_bracket=MAREA_MIN_BRACKET)
        for k, v in part.items():
            fit[k][cols] = v
    fitted, crossed = int(np.isfinite(fit["z_mid"]).sum()), int(np.isfinite(fit["z0"]).sum())
    note = f"{lags_text(tau_bands)}; {fitted - crossed:,} of {fitted:,} fitted px do not cross NDWI = 0"
    return fit.pop("z0"), note, None, fit
''')

md(r"""
**V2 MAREA-B** (diagnostic, new): V0's delay; per pixel, the probit likelihood of its wet/dry record on its clear
looks, ℓ(z, σ) = Σ_wet log Φ((h − z)/σ) + Σ_dry log Φ(−(h − z)/σ) with h = h_B(t − τ_px) and wet = clear and
NDWI > 0 (a clear look with an undefined NDWI is dry, as in MAREA's lag fit), maximised on `invert_series`' grid: 50
levels between the window's lowest and highest level, σ in `marea.SG_GRID`. The elevation is z, where P(wet) = 0.5.
Under complete separation ℓ has a plateau instead of a maximum: the z levels whose profile log-likelihood (the
maximum over σ) lies within `B_TIE_TOL` of the largest are tied, and z is their median (`B_TIE_TOL` is this notebook's
reading of *tied*). At least 8 clear looks, with at least one wet and one dry. **V2d** is V2 with every delay 0.
`invert_bernoulli` works on blocks of 4,000 pixels, as `_fit_block`; the next cell checks it against plain loops.
""")

code(r'''
def invert_bernoulli(Y, C, h, lo, hi, min_looks=B_MIN_LOOKS, tie_tol=B_TIE_TOL, mu_points=50, chunk=4000):
    """MAREA-B: per pixel, the probit likelihood of the wet/dry record on the clear looks against the levels h,
    P(wet) = Phi((h - z)/sigma), maximised on invert_series' grid (mu_points levels in [lo, hi], sigma in
    marea.SG_GRID). z = the median of the tied z levels (profile log-likelihood within tie_tol of its maximum);
    sigma = the best sigma at the tied level nearest z. NaN with fewer than min_looks clear looks or without a wet
    and a dry one. Returns (z, sigma, the number of tied levels)."""
    grid, sgs = np.linspace(lo, hi, mu_points), np.asarray(marea.SG_GRID, float)
    u = (np.asarray(h, float)[:, None, None] - grid[None, :, None]) / sgs[None, None, :]
    lw, ld = log_ndtr(u).reshape(len(h), -1), log_ndtr(-u).reshape(len(h), -1)   # (looks, levels x sigmas)
    P = Y.shape[1]
    z, sg, n_tied = np.full(P, np.nan), np.full(P, np.nan), np.zeros(P, int)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        with np.errstate(invalid="ignore"):
            wet = clear & (np.asarray(Y[:, s]) > 0)     # an undefined NDWI is never wet
        dry = clear & ~wet
        ll = (wet.T.astype(float) @ lw + dry.T.astype(float) @ ld).reshape(-1, len(grid), len(sgs))
        prof = ll.max(axis=2)                           # the profile over sigma: (pixels, levels)
        tied = prof >= prof.max(axis=1, keepdims=True) - tie_tol
        zb = np.nanmedian(np.where(tied, grid[None, :], np.nan), axis=1)
        near = np.argmin(np.where(tied, np.abs(grid[None, :] - zb[:, None]), np.inf), axis=1)
        sb = sgs[ll[np.arange(len(near)), near].argmax(axis=1)]
        ok = (clear.sum(axis=0) >= min_looks) & wet.any(axis=0) & dry.any(axis=0)
        z[s], sg[s], n_tied[s] = np.where(ok, zb, np.nan), np.where(ok, sb, np.nan), np.where(ok, tied.sum(1), 0)
    return z, sg, n_tied


def run_marea_b(win, px, delay=True):
    """V2 (delay=True, V0's delays) and V2d (every delay 0), between the window's lowest and highest level."""
    w = win["scenes"]
    tau_bands, tau_px = window_delays(win) if delay else (None, None)
    tau_px = tau_px[px] if delay else np.zeros(len(px))
    h0 = LEV[0][w]
    Y, C = Y_S[np.ix_(w, px)], C_S[np.ix_(w, px)]
    z, sigma, tied = np.full(len(px), np.nan), np.full(len(px), np.nan), np.zeros(len(px), int)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols], sigma[cols], tied[cols] = invert_bernoulli(Y[:, cols], C[:, cols], level_at(int(tau))[w],
                                                            h0.min(), h0.max())
    fin = np.isfinite(z)
    note = (f"{lags_text(tau_bands)}; " if delay else "every delay 0; ") + \
        f"{int((tied[fin] > 1).sum()):,} of {int(fin.sum()):,} px on a tied plateau"
    return z, note, None, {"sigma": sigma, "n_tied": tied}
''')

code(r'''
def brute_force_b(wet, clear, h, lo, hi, tie_tol=B_TIE_TOL):
    """MAREA-B of one pixel by plain loops over the grid (the check of invert_bernoulli)."""
    wet = wet & clear
    if clear.sum() < B_MIN_LOOKS or not wet.any() or not (clear & ~wet).any():
        return np.nan
    grid, prof = np.linspace(lo, hi, 50), []
    for zk in grid:
        best = -np.inf
        for sk in marea.SG_GRID:
            ll = 0.0
            for k in range(len(h)):
                if clear[k]:
                    u = (h[k] - zk) / sk
                    ll += float(log_ndtr(u if wet[k] else -u))
            best = max(best, ll)
        prof.append(best)
    prof = np.array(prof)
    return float(np.median(grid[prof >= prof.max() - tie_tol]))


rng = np.random.default_rng(5)
hc = rng.uniform(-2.0, 2.0, 60)                        # 60 looks, as in the longest window
Yc = np.stack([np.where(rng.random(60) < ndtr((hc - 0.3) / 0.25), 0.4, -0.3),   # a noisy pixel, z = 0.3
               np.where(hc > -0.4, 0.5, -0.2),                                  # complete separation at -0.4
               np.where(hc > 0.8, 0.5, np.nan),                                 # ... at 0.8, dry = undefined NDWI
               np.where(hc > 0.0, 0.5, -0.2)], axis=1)                          # 5 clear looks: no value
Cc = rng.random((60, 4)) > 0.1
Cc[:, 3] = np.arange(60) < 5
zv, sv, nt = invert_bernoulli(Yc, Cc, hc, hc.min(), hc.max())
with np.errstate(invalid="ignore"):
    zb = [brute_force_b(Yc[:, j] > 0, Cc[:, j], hc, hc.min(), hc.max()) for j in range(4)]
assert np.allclose(zv, zb, rtol=0, atol=1e-12, equal_nan=True) and np.isnan(zv[3]), (zv, zb)
print(f"invert_bernoulli = the brute-force maximum on 4 synthetic pixels: z = {np.round(zv, 3)} "
      f"(tied levels {nt.tolist()})")
''')

md(r"""
**V10 MAREA-B with lapse rates** (diagnostic, declared on 2026-10-27 before any confirmatory window): V2 with a
contamination term, P(wet) = e + (1 − 2e)Φ((h − z)/σ), so a few spurious wet looks at low tide or dry looks at high
tide (a missed cloud, a wet patch left on the flat) do not drag z. The lapse rate e is maximised with z and σ on the
grid `B_LAPSES` = {0, 0.02, 0.05, 0.10}; the rest is V2's: V0's delay, `invert_series`' z and σ grids between the
window's lowest and highest level, the median of the tied z levels (the profile now over σ and e), at least 8 clear
looks with a wet and a dry one. With e = 0 the log-likelihood is V2's (log Φ), so `invert_bernoulli_lapse` with
e = 0 alone is V2 bit for bit; the next cell checks that, and checks it against plain loops.
""")

code(r'''
def invert_bernoulli_lapse(Y, C, h, lo, hi, lapses=B_LAPSES, min_looks=B_MIN_LOOKS, tie_tol=B_TIE_TOL, mu_points=50,
                           chunk=4000):
    """V10: per pixel, the likelihood of its wet/dry record on the clear looks with P(wet) = e + (1 - 2e)
    Phi((h - z)/sigma), maximised on invert_series' grid (mu_points levels in [lo, hi], sigma in marea.SG_GRID) and
    over the lapse rates e in `lapses` (e = 0: log Phi, invert_bernoulli's own terms). z = the median of the tied z
    levels (profile over sigma and e within tie_tol of its maximum); sigma and e = the best pair at the tied level
    nearest z. NaN with fewer than min_looks clear looks or without a wet and a dry one. Returns (z, sigma, e, the
    number of tied levels)."""
    grid, sgs, es = np.linspace(lo, hi, mu_points), np.asarray(marea.SG_GRID, float), np.asarray(lapses, float)
    u = (np.asarray(h, float)[:, None, None] - grid[None, :, None]) / sgs[None, None, :]

    def log_p(v):                                     # log(e + (1 - 2e) Phi(v)) for every e: (..., lapses)
        return np.stack([log_ndtr(v) if e == 0 else np.log(e + (1 - 2 * e) * ndtr(v)) for e in es], axis=-1)

    lw, ld = log_p(u).reshape(len(h), -1), log_p(-u).reshape(len(h), -1)   # (looks, levels x sigmas x lapses)
    P, K = Y.shape[1], len(sgs) * len(es)
    z, sg, lapse, n_tied = np.full(P, np.nan), np.full(P, np.nan), np.full(P, np.nan), np.zeros(P, int)
    for j in range(0, P, chunk):
        s = slice(j, min(j + chunk, P))
        clear = np.asarray(C[:, s], bool)
        with np.errstate(invalid="ignore"):
            wet = clear & (np.asarray(Y[:, s]) > 0)     # an undefined NDWI is never wet
        dry = clear & ~wet
        ll = (wet.T.astype(float) @ lw + dry.T.astype(float) @ ld).reshape(-1, len(grid), K)
        prof = ll.max(axis=2)                           # the profile over sigma and e: (pixels, levels)
        tied = prof >= prof.max(axis=1, keepdims=True) - tie_tol
        zb = np.nanmedian(np.where(tied, grid[None, :], np.nan), axis=1)
        near = np.argmin(np.where(tied, np.abs(grid[None, :] - zb[:, None]), np.inf), axis=1)
        best = ll[np.arange(len(near)), near].argmax(axis=1)   # sigma index * len(es) + e index
        ok = (clear.sum(axis=0) >= min_looks) & wet.any(axis=0) & dry.any(axis=0)
        z[s], n_tied[s] = np.where(ok, zb, np.nan), np.where(ok, tied.sum(1), 0)
        sg[s] = np.where(ok, sgs[best // len(es)], np.nan)
        lapse[s] = np.where(ok, es[best % len(es)], np.nan)
    return z, sg, lapse, n_tied


def run_marea_bl(win, px):
    """V10: V2 (V0's delays, the same grid and guards) with the lapse rates B_LAPSES."""
    w = win["scenes"]
    tau_bands, tau_px = window_delays(win)
    tau_px = tau_px[px]
    h0 = LEV[0][w]
    Y, C = Y_S[np.ix_(w, px)], C_S[np.ix_(w, px)]
    z, sigma, lapse, tied = (np.full(len(px), np.nan), np.full(len(px), np.nan), np.full(len(px), np.nan),
                             np.zeros(len(px), int))
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols], sigma[cols], lapse[cols], tied[cols] = invert_bernoulli_lapse(Y[:, cols], C[:, cols],
                                                                               level_at(int(tau))[w], h0.min(), h0.max())
    fin = np.isfinite(z)
    share = ", ".join(f"{np.mean(lapse[fin] == e):.0%} e={e:g}" for e in B_LAPSES) if fin.any() else "no value"
    note = f"{lags_text(tau_bands)}; {share}; {int((tied[fin] > 1).sum()):,} of {int(fin.sum()):,} px on a tied plateau"
    return z, note, None, {"sigma": sigma, "lapse": lapse, "n_tied": tied}
''')

code(r'''
def brute_force_bl(wet, clear, h, lo, hi, lapses=B_LAPSES, tie_tol=B_TIE_TOL):
    """V10 of one pixel by plain loops over the grid and the lapse rates (the check of invert_bernoulli_lapse)."""
    wet = wet & clear
    if clear.sum() < B_MIN_LOOKS or not wet.any() or not (clear & ~wet).any():
        return np.nan
    grid, prof = np.linspace(lo, hi, 50), []
    for zk in grid:
        best = -np.inf
        for sk in marea.SG_GRID:
            for e in lapses:
                ll = 0.0
                for k in range(len(h)):
                    if clear[k]:
                        u = (h[k] - zk) / sk if wet[k] else -(h[k] - zk) / sk
                        ll += float(log_ndtr(u)) if e == 0 else float(np.log(e + (1 - 2 * e) * ndtr(u)))
                best = max(best, ll)
        prof.append(best)
    prof = np.array(prof)
    return float(np.median(grid[prof >= prof.max() - tie_tol]))


rng = np.random.default_rng(6)
hl = rng.uniform(-2.0, 2.0, 60)                        # 60 looks
low = np.argsort(hl)[:3]                               # the 3 lowest levels
Yl = np.stack([np.where(rng.random(60) < ndtr((hl - 0.3) / 0.25), 0.4, -0.3),    # a noisy pixel, z = 0.3
               np.where(hl > 0.5, 0.5, -0.2),                                   # complete separation at 0.5 ...
               np.where(hl > 0.5, 0.5, -0.2),                                   # ... and with 3 wet looks at low tide
               np.where(hl > 0.0, 0.5, -0.2)], axis=1)                          # 5 clear looks: no value
Yl[low, 2] = 0.3
Cl = rng.random((60, 4)) > 0.1
Cl[low, 2] = True
Cl[:, 3] = np.arange(60) < 5
zl, sl, el, ntl = invert_bernoulli_lapse(Yl, Cl, hl, hl.min(), hl.max())
with np.errstate(invalid="ignore"):
    zbl = [brute_force_bl(Yl[:, j] > 0, Cl[:, j], hl, hl.min(), hl.max()) for j in range(4)]
assert np.allclose(zl, zbl, rtol=0, atol=1e-12, equal_nan=True) and np.isnan(zl[3]), (zl, zbl)
z2, s2, _ = invert_bernoulli(Yl, Cl, hl, hl.min(), hl.max())
z0l, s0l, e0l, _ = invert_bernoulli_lapse(Yl, Cl, hl, hl.min(), hl.max(), lapses=(0.0,))
assert np.array_equal(z0l, z2, equal_nan=True) and np.array_equal(s0l, s2, equal_nan=True), "e = 0 alone is not V2"
print(f"invert_bernoulli_lapse = the brute-force maximum on 4 synthetic pixels: z = {np.round(zl, 3)}, "
      f"e = {el.tolist()}; with e = 0 alone it is invert_bernoulli (V2) bit for bit. The pixel with 3 wet looks at "
      f"low tide: V2 z = {z2[2]:.3f} m, V10 z = {zl[2]:.3f} m (e = {el[2]:g}); without them both give {zl[1]:.3f} m")
''')

md(r"""
**The addendum's estimates** (`experiments/variants_addendum_2026-10-28.md`, declared before any window was computed
with them; diagnostics, run when `NB_METHODS` names them), each on V0's delay unless said otherwise:
**V11 MAREA-M**, the isotonic wet-probability read-off, no curve shape (`experiments/variants/monotone.py`), and
**V11d** with every delay 0; **V12 MAREA-P**, the depth-aware model NDWI = a + Φ(u)(b + c(h − z)), c ≥ 0, on V0's
inputs (`depth_model.py`); **V4 MAREA-H** and **V4-0**, the Huber fit (δ = 0.10) with weight 0.5 for the looks of SCL
class 7, its midpoint and its NDWI = 0 crossing (`huber.py`; SCL read once at the keep pixels of every scene of S, the
cube's for its scenes and the band-subset stores' before 2023); **V13a**, MAREA-B with no value where its 95 %
profile-likelihood interval is not bracketed by the grid (`interval.py`; its z is V2's, asserted bit for bit every
window); **V14**, the per-scene Otsu water threshold (`otsu.py`) as a factor on every wet/dry estimator: MAREA-B,
MAREA-BL and MAREA-M read NDWI − t_s, and LogiTide2DEM, sampled exceedance and the wet/dry bracket get it through
their inputs (`make_obs_otsu`), so that their own rule wet = NDWI > 0 is wet = NDWI > t_s. Section 8 has V13's interval
checks, the error-coverage curves of every estimate and the bootstrap of the judging rule.
""")

code(r'''
from experiments.variants import depth_model as v12, huber as v4, interval as v13, monotone as v11, otsu as v14

ADD = {}            # the site's Otsu thresholds and shifted NDWI, and its SCL 7 looks: computed on first use


def otsu_ndwi():
    """V14: (Y_S - t_s per scene, t_s, the scenes that fell back to 0), once per site."""
    if "otsu" not in ADD:
        t, fell = v14.scene_thresholds(Y_S, C_S)
        ADD["otsu"] = (v14.shift_for_threshold(Y_S, t), np.asarray(t, float), np.asarray(fell, bool))
        print(f"V14: Otsu thresholds of the {len(t)} scenes of S: median {np.median(t):+.3f}, range "
              f"[{np.min(t):+.3f}, {np.max(t):+.3f}], {int(np.sum(fell))} fell back to 0", flush=True)
    return ADD["otsu"]


def scl7_at_keep():
    """V4: SCL == 7 at the keep pixels of every scene of S (scenes x keep, bool): the cube's SCL for its scenes, the
    band-subset stores' before 2023 (read as zarr_frames reads them)."""
    if "scl7" not in ADD:
        out = np.zeros((len(S_dates), len(keep)), bool)
        ds, bands, t_dim = pit.cube.open_cube(P["cache"])
        try:
            for k, d in enumerate(S_dates):
                if d in cube_pos:
                    out[k] = np.asarray(bands["SCL"].isel({t_dim: cube_pos[d]}).values).ravel()[keep] == 7
        finally:
            ds.close()
        old = [k for k, d in enumerate(S_dates) if d not in cube_pos]
        for year in sorted({S_dates[k][:4] for k in old}):
            ks = [k for k in old if S_dates[k][:4] == year]
            z = zc.open_store(zc.store_path(SITE, year, lte.LIGHT_ROOT), bands=["SCL"])
            try:
                zpos = {d: i for i, d in enumerate(zc.dates(z))}
                for a in range(0, len(ks), 16):
                    part = ks[a:a + 16]
                    scl = np.asarray(z["SCL"].isel({"t": [zpos[S_dates[k]] for k in part]}).values)
                    for j, k in enumerate(part):
                        out[k] = scl[j].ravel()[keep] == 7
            finally:
                z.close()
        ADD["scl7"] = out
        print(f"V4: SCL 7 on {np.mean(out[C_S]):.2%} of the clear looks of S at the keep pixels", flush=True)
    return ADD["scl7"]


def delay_groups(win, px, delay=True):
    """(the window's scenes, its band lags or None, the delay of each pixel of px, the window's lowest and highest
    level without delay)."""
    w = win["scenes"]
    tau_bands, tau_px = window_delays(win) if delay else (None, np.zeros(len(keep)))
    h0 = LEV[0][w]
    return w, tau_bands, tau_px[px], h0.min(), h0.max()


def run_marea_m(win, px, delay=True, Ysrc=None):
    """V11 (V0's delays), V11d (every delay 0) and, with the shifted NDWI, V11 + Otsu."""
    w, tau_bands, tau_px, lo, hi = delay_groups(win, px, delay)
    Y, C = (Y_S if Ysrc is None else Ysrc)[np.ix_(w, px)], C_S[np.ix_(w, px)]
    z, cens = np.full(len(px), np.nan), np.zeros(len(px), int)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols], cens[cols], _, _ = v11.invert_monotone(Y[:, cols], C[:, cols], level_at(int(tau))[w])
    fin = np.isfinite(z)
    note = (f"{lags_text(tau_bands)}; " if delay else "every delay 0; ") + \
        f"{int((cens[fin] != 0).sum()):,} of {int(fin.sum()):,} px censored"
    return z, note, None, {"censored": cens}


def run_marea_p(win, px):
    """V12 on V0's delays and inputs."""
    w, tau_bands, tau_px, lo, hi = delay_groups(win, px)
    Y0, C0 = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C_S[np.ix_(w, px)]
    z, c = np.full(len(px), np.nan), np.full(len(px), np.nan)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols], _, c[cols] = v12.invert_depth(Y0[:, cols], C0[:, cols], level_at(int(tau))[w], lo, hi)
    fin = np.isfinite(z) & np.isfinite(c)
    note = (f"{lags_text(tau_bands)}; depth slope c: median {np.median(c[fin]):.3f} per m, "
            f"{np.mean(c[fin] > 0):.0%} of px > 0" if fin.any() else f"{lags_text(tau_bands)}; no value")
    return z, note, None, {"c": c}


HUBER = {}


def huber_fit(win, px):
    """V4's fit of the window at px, computed once: (midpoint z, NDWI = 0 crossing z0, iterations, note)."""
    key = (win["site"], win["n"])
    if key not in HUBER:
        w, tau_bands, tau_px, lo, hi = delay_groups(win, px)
        Y0, C0 = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C_S[np.ix_(w, px)]
        s7 = scl7_at_keep()[np.ix_(w, px)]
        z, z0, it = np.full(len(px), np.nan), np.full(len(px), np.nan), np.zeros(len(px), int)
        for tau in np.unique(tau_px):
            cols = np.flatnonzero(tau_px == tau)
            z[cols], z0[cols], _, _, _, it[cols] = v4.invert_huber(Y0[:, cols], C0[:, cols], level_at(int(tau))[w],
                                                                    lo, hi, scl7=s7[:, cols])
        fin = np.isfinite(z)
        HUBER[key] = (z, z0, it, f"{lags_text(tau_bands)}; iterations median "
                                 f"{int(np.median(it[fin])) if fin.any() else 0}, max {int(it.max()) if len(it) else 0}; "
                                 f"SCL 7 on {np.mean(s7[C0]):.1%} of the clear looks")
    return HUBER[key]


def run_marea_h(win, px, crossing=False):
    """V4 (the Huber fit's midpoint) and V4-0 (its NDWI = 0 crossing)."""
    z, z0, it, note = huber_fit(win, px)
    return (z0 if crossing else z), note, None, {"n_iter": it}


BINT = {}           # per window: MAREA-B's z and its 95 % interval at F (section 8)


def run_marea_b_abst(win, px):
    """V13a: MAREA-B with no value where its 95 % profile-likelihood interval is not bracketed by the grid."""
    w, tau_bands, tau_px, lo, hi = delay_groups(win, px)
    Y, C = Y_S[np.ix_(w, px)], C_S[np.ix_(w, px)]
    b = {k: np.full(len(px), np.nan) for k in ("z", "z_lo", "z_hi")}
    br = np.zeros(len(px), bool)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z, _, zl, zh, ok = v13.bernoulli_interval(Y[:, cols], C[:, cols], level_at(int(tau))[w], lo, hi)
        b["z"][cols], b["z_lo"][cols], b["z_hi"][cols], br[cols] = z, zl, zh, ok
    BINT[(win["site"], win["n"])] = dict(b, bracketed=br)
    fin = np.isfinite(b["z"])
    note = f"{lags_text(tau_bands)}; {int((fin & br).sum()):,} of {int(fin.sum()):,} px with a bracketed interval"
    return (np.where(br, b["z"], np.nan), note, None,
            {"z_all": b["z"], "z_lo": b["z_lo"], "z_hi": b["z_hi"], "bracketed": br.astype(np.int8)})


def run_marea_b_otsu(win, px, lapse=False):
    """V14 on V2 (and on V10): their likelihoods on the wet/dry record wet = NDWI > t_s."""
    w, tau_bands, tau_px, lo, hi = delay_groups(win, px)
    Y, C = otsu_ndwi()[0][np.ix_(w, px)], C_S[np.ix_(w, px)]
    z = np.full(len(px), np.nan)
    fit = invert_bernoulli_lapse if lapse else invert_bernoulli
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols] = fit(Y[:, cols], C[:, cols], level_at(int(tau))[w], lo, hi)[0]
    return z, f"{lags_text(tau_bands)}; wet = NDWI > t_s (Otsu per scene)", None, {}


def make_obs_otsu(win, px, tau=0):
    """make_obs with the shifted NDWI (V14): wet = C & (Y - t_s > 0), NDWI > t_s, by the competitors' own rule."""
    w = win["scenes"]
    Y, C = otsu_ndwi()[0][np.ix_(w, px)], C_S[np.ix_(w, px)]
    with np.errstate(invalid="ignore"):
        wet = C & (Y > 0)
    return types.SimpleNamespace(Y=Y, C=C, W=wet, h=level_at(tau)[w], t=S_t[w], keep=keep[px],
                                 shape=(H, W), pixel_m=10.0, site=SITE)


def run_competitor_otsu(mid, win, px):
    """V14 on a wet/dry competitor (E4, E5c, E12): run_competitor's call with make_obs_otsu."""
    label, run, card = fm.METHODS[mid]
    cfg = {}
    if "level_series" in card.get("needs", []):
        start = S_t[win["scenes"][0]].normalize()
        cfg["level_series"] = LS.slice(start, pd.Timestamp(win["end"]) + pd.Timedelta(days=1))
    out = run(make_obs_otsu(win, px), cfg)
    return out["z"], (out["meta"].get("reason", "") + "; wet = NDWI > t_s (Otsu per scene)").lstrip("; "), None, {}
''')

md(r"""
**The other estimates**, as notebook 01 runs them: *MAREA, no delay* (V0 with every delay 0, diagnostic); NIDEM
(`waterline_contour`, whole frames read from the cube, and before 2023 from the band-subset Zarr stores
`cubes_light/<site>/<year>.zarr`, not timed); DEA Intertidal 2.5.3 (DEA's own code in its side
interpreter, on every pixel of the extraction); LogiTide2DEM (E4), sampled exceedance (E5c) and Flood2Topo (E5a) of
`experiments/fixedcfg/methods.py`, per pixel on F; *LogiTide2DEM + MAREA delay* (diagnostic: one E4 call per delay
group, each with its levels read τ_px earlier). E4's per-pixel extras (its case: separated or fitted, its MSR) are
kept.
""")

code(r'''
def run_marea_nodelay(win, px):
    w = win["scenes"]
    h0 = LEV[0][w]
    Y0, C0 = np.nan_to_num(Y_S[np.ix_(w, px)], nan=0.0), C_S[np.ix_(w, px)]
    z, _ = marea.invert_series(Y0, C0, h0, h0.min(), h0.max(), min_bracket=MAREA_MIN_BRACKET)
    return z, "every delay 0 (no lag fit)", None


def run_delayed(mid, win, px):
    """A per-pixel competitor with every pixel's levels read MAREA's delay earlier: one call per delay
    (notebook 01's run_delayed, with the window's delays fitted once)."""
    tau_bands, tau_px = window_delays(win)
    tau_px = tau_px[px]
    z = np.full(len(px), np.nan)
    for tau in np.unique(tau_px):
        cols = np.flatnonzero(tau_px == tau)
        z[cols] = fm.METHODS[mid][1](make_obs(win, px[cols], tau=int(tau)), {})["z"]
    return z, f"{lags_text(tau_bands)}; {len(np.unique(tau_px))} delay groups", None


cube_pos = {d: i for i, d in enumerate(cube.dates)}


STATIC = None       # the legacy cube's static no-data mask (H, W), set with the pre-2023 scenes (the frames checks)


def read_frames(dates):
    """NDWI and the common clear mask of whole scenes, (T, H, W): a scene of the cube is read from the cube, one
    scene at a time; a scene before the cube (pre-2023) from the band-subset Zarr store of its year (zarr_frames)."""
    ndwi, clear = np.empty((len(dates), H, W), np.float32), np.empty((len(dates), H, W), bool)
    ds, bands, t_dim = pit.cube.open_cube(P["cache"])
    try:
        for k, d in enumerate(dates):
            if d in cube_pos:
                at = {t_dim: cube_pos[d]}
                scl = bands["SCL"].isel(at).values
                ndwi[k], _ = wc.scene_ndwi(bands["B03"].isel(at).values, bands["B08"].isel(at).values, scl)
                clear[k] = np.isin(scl, wc.CLEAR)
    finally:
        ds.close()
    old = [k for k, d in enumerate(dates) if d not in cube_pos]
    if old:
        zarr_frames([dates[k] for k in old], ndwi, clear, old)
    return ndwi, clear


def zarr_frames(dates, ndwi, clear, at, root=lte.LIGHT_ROOT):
    """NDWI and the common clear mask of `dates` from the yearly Zarr stores <root>/<site>/<year>.zarr (B03, B08 and
    SCL as the backend delivered them, int16, on the legacy cube's frame), written into ndwi[at[i]], clear[at[i]].
    The fill -32768 is read as NaN, as xarray decodes the cube; then the cube's own scene_ndwi and clear mask. The
    pixels of STATIC (no data on every scene of the cube, data in the stores) get no data, the converter's rule.
    At most 16 scenes are read at a time (one chunk of the stores in time)."""
    for year in sorted({d[:4] for d in dates}):
        ks = [i for i, d in enumerate(dates) if d[:4] == year]
        z = zc.open_store(zc.store_path(SITE, year, root), bands=["B03", "B08", "SCL"])
        try:
            assert (z.sizes["y"], z.sizes["x"]) == (H, W), f"{root}/{SITE}/{year}.zarr is not on the cube's frame"
            zpos = {d: i for i, d in enumerate(zc.dates(z))}
            for a in range(0, len(ks), 16):
                part = ks[a:a + 16]
                sel = {"t": [zpos[dates[i]] for i in part]}
                b03, b08, scl = (np.asarray(z[b].isel(sel).values) for b in ("B03", "B08", "SCL"))
                for j, i in enumerate(part):
                    g, n = (np.where(v[j] == zc.FILL, np.float32(np.nan), v[j].astype(np.float32)) for v in (b03, b08))
                    ndwi[at[i]], _ = wc.scene_ndwi(g, n, scl[j])
                    clear[at[i]] = np.isin(scl[j], wc.CLEAR)
                    if STATIC is not None:
                        ndwi[at[i]][STATIC] = np.nan
                        clear[at[i]][STATIC] = False
        finally:
            z.close()


def run_nidem(win, px, frames):
    w = win["scenes"]
    j, edges, Z, U, sizes = wc.interval_assignment(LEV[0][w])
    wc.interval_coverage(wc.clear_counts_by_interval(C_S[w], Y_S[w], j)[0])   # pre-flight, on the pixels
    res = wc.nidem(wc.composites_from_arrays(*frames, j), Z, U)
    return res["filtered"].ravel()[keep[px]], f"{res['n_vertices']:,} contour vertices", None


def run_dea(win, px):
    out = dea.run(make_obs(win, np.arange(len(keep))), {"dea_tmp": DEA_TMP})
    return out["z"][px], f"{out['meta']['n_candidates']:,} candidate pixels", out["meta"].get("seconds_estimator")


def run_competitor(mid, win, px):
    """Notebook 01's run_competitor; the method's per-pixel extras (E4: msr, case, n_valid) are kept."""
    label, run, card = fm.METHODS[mid]
    cfg = {}
    if "level_series" in card.get("needs", []):
        start = S_t[win["scenes"][0]].normalize()
        cfg["level_series"] = LS.slice(start, pd.Timestamp(win["end"]) + pd.Timedelta(days=1))
    out = run(make_obs(win, px), cfg)
    extra = {k: v for k, v in out["meta"].get("extra", {}).items()
             if isinstance(v, np.ndarray) and v.shape == (len(px),)}
    return out["z"], out["meta"].get("reason", ""), None, extra


def run_method(mid, win, px, frames=None):
    """(z, note, the estimator's own seconds or None, per-pixel extras) of one estimate at the pixels px."""
    if mid == "MAREA":
        return run_marea(win, px)
    if mid == "MAREA_ndwi0":
        return run_marea_ndwi0(win, px)
    if mid == "MAREA_B":
        return run_marea_b(win, px)
    if mid == "MAREA_BL":
        return run_marea_bl(win, px)
    if mid == "MAREA_B_nodelay":
        return run_marea_b(win, px, delay=False)
    if mid == "MAREA_nodelay":
        return run_marea_nodelay(win, px) + ({},)
    if mid == "E4_delay":
        return run_delayed("E4", win, px) + ({},)
    if mid == "NIDEM":
        return run_nidem(win, px, frames) + ({},)
    if mid == "DEA":
        return run_dea(win, px) + ({},)
    if mid == "MAREA_M":                               # the addendum of 2026-10-28
        return run_marea_m(win, px)
    if mid == "MAREA_M_nodelay":
        return run_marea_m(win, px, delay=False)
    if mid == "MAREA_M_otsu":
        z, note, own, more = run_marea_m(win, px, Ysrc=otsu_ndwi()[0])
        return z, note + "; wet = NDWI > t_s (Otsu per scene)", own, more
    if mid == "MAREA_P":
        return run_marea_p(win, px)
    if mid in ("MAREA_H", "MAREA_H0"):
        return run_marea_h(win, px, crossing=mid == "MAREA_H0")
    if mid == "MAREA_B_abst":
        return run_marea_b_abst(win, px)
    if mid in ("MAREA_B_otsu", "MAREA_BL_otsu"):
        return run_marea_b_otsu(win, px, lapse=mid == "MAREA_BL_otsu")
    if mid in ("E4_otsu", "E5c_otsu", "E12_otsu"):
        return run_competitor_otsu(mid.removesuffix("_otsu"), win, px)
    return run_competitor(mid, win, px)
''')

md(r"""
**The pre-2023 frames: the checks.** The legacy cube has a static no-data block (no data on every one of its
scenes; `static_nodata` of the production reference map the extraction's JSON names, never a keep pixel) where the
Zarr stores have data: as the converter's screening does (its pre-registered rule), the Zarr frames get no data
there (`STATIC`), so NIDEM sees no observation the cube could not have. Then every Zarr store a window reads is on
the cube's frame (its x and y are the cube's, so the pixels come in the same order), and the reader gives the
cube's own arrays where both exist: on the screened scenes of 2023, the 2023 store (`cubes_light/<site>/2023.zarr`,
else the store the extraction's 2023 gate compared, `gate_2023` of its JSON) must give the clear mask of whole
frames, and their NDWI (NaN included) outside the static block, bit for bit as read from the cube, as that gate
found at the keep pixels on every 2023 scene. A difference stops the notebook unless the first author accepted the
2023 gate's differences (then it is reported).
""")

code(r'''
if len(x_dates):
    ref = side["reference_map"]                        # the production reference map and the static no-data mask
    assert lte.sha256_file(ref["path"]) == ref["sha256"], f"{ref['path']} is not the extraction's reference"
    with np.load(ref["path"]) as z:
        STATIC = z["static_nodata"].astype(bool)
    assert STATIC.shape == (H, W) and not STATIC.ravel()[keep].any()
    print(f"the legacy cube's static no-data ({ref['path']}): {int(STATIC.sum()):,} px, no keep pixel among them; "
          "the Zarr frames get no data there")
    with xr.open_dataset(P["cache"]) as ds:
        cube_x, cube_y = ds["x"].values, ds["y"].values
    gate = side["gate_2023"]
    light23 = zc.store_path(SITE, 2023, lte.LIGHT_ROOT)
    root23 = lte.LIGHT_ROOT if zc.is_complete(light23) else os.path.dirname(os.path.dirname(gate["store"]["path"]))
    years = sorted({d[:4] for d in x_dates})
    for year, root in [(y, lte.LIGHT_ROOT) for y in years] + [("2023", root23)]:
        with zc.open_store(zc.store_path(SITE, year, root), bands=["SCL"]) as z:
            assert np.array_equal(z["x"].values, cube_x) and np.array_equal(z["y"].values, cube_y), \
                f"{root}/{SITE}/{year}.zarr is not on the cube's frame"
    print(f"{lte.LIGHT_ROOT}/{SITE}/<year>.zarr of {years} and {root23}/{SITE}/2023.zarr: the cube's x and y")

    d23 = [d for d in S_dates if d[:4] == "2023"]      # the screened 2023 scenes: in the cube and in the store
    diff = Counter()
    for a in range(0, len(d23), 8):
        block = d23[a:a + 8]
        n_cube, c_cube = read_frames(block)            # from the cube
        n_zarr, c_zarr = np.empty_like(n_cube), np.empty_like(c_cube)
        zarr_frames(block, n_zarr, c_zarr, list(range(len(block))), root=root23)
        nan_cube, nan_zarr, out = np.isnan(n_cube), np.isnan(n_zarr), ~STATIC[None]   # NDWI: outside the static block
        diff["pixel-scenes"] += n_cube.size
        diff["NaN differs"] += int(((nan_cube != nan_zarr) & out).sum())
        diff["NDWI differs"] += int((~nan_cube & ~nan_zarr & out & (n_cube.view(np.uint32) != n_zarr.view(np.uint32)))
                                    .sum())
        diff["clear differs"] += int((c_cube != c_zarr).sum())                        # clear: the whole frame
    same = diff["NaN differs"] + diff["NDWI differs"] + diff["clear differs"] == 0
    print(f"{len(d23)} screened 2023 scenes, whole frames, {root23}/{SITE}/2023.zarr against the cube: "
          + ("the clear mask, and the NDWI (NaN included) outside the static block, bit for bit" if same
             else f"DIFFERENT {dict(diff)}"))
    print(f"the extraction's 2023 gate ({gate['store']['path']}, every 2023 scene at the keep pixels): "
          + ("passed" if gate.get("strict_ok") else f"differences accepted by {(gate.get('accepted') or {}).get('by')}"))
    if not same:
        assert gate.get("accepted"), "the Zarr frames are not the cube's, and no 2023 gate difference was accepted"
        print("  the 2023 gate's differences were accepted by the first author: NIDEM reads these frames (reported)")
else:
    print("no pre-2023 scene: NIDEM reads the cube only")
''')

md(r"""
**Every window.** The window's delays are fitted first and timed as `delay_fit`; then every estimate is called and
timed with `time.perf_counter` (a failure is recorded with its reason and the others go on) and scored on the row's
pixels F by notebook 01's `scores` (RMSE, bias = median(p − t), centred RMSE, OLS slope of p on t, r, coverage). The
per-pixel estimates, extras, delays τ_px and clear-look counts are kept for the next sections. The CSV is rewritten
after every window.
""")

code(r'''
def scores(p, t):
    """RMSE, bias, centred RMSE, slope, r, the pixels scored and the coverage of p against t (same pixels)."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    out = dict(n_px=int(ok.sum()), coverage=float(ok.mean()) if len(ok) else np.nan,
               rmse=np.nan, bias=np.nan, crmse=np.nan, slope=np.nan, r=np.nan)
    if ok.sum() >= 30:
        d = p[ok] - t[ok]
        pc, tc = p[ok] - np.median(p[ok]), t[ok] - np.median(t[ok])
        out.update(rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(np.median(d)),
                   crmse=float(np.sqrt(np.mean((pc - tc) ** 2))), slope=float(np.polyfit(tc, pc, 1)[0]),
                   r=float(np.corrcoef(tc, pc)[0, 1]))
    return out


def run_window(win):
    """Every estimate on the window's images: the CSV rows, {id: elevation on the row's pixels F} and
    {id: its per-pixel extras}."""
    dates = S_dates[win["scenes"]]
    base = dict(site=win["site"], survey_year=win["year"], truth_date=win["end"], n=win["n"],
                available=win["available"], first_date=dates[0], last_date=dates[-1], n_F=len(win["px"]))
    t0 = time.perf_counter()
    tau_bands, _ = window_delays(win)
    rows = [dict(base, method="delay_fit", label="MAREA's delay fit (shared)", role="", status="ok",
                 reason=lags_text(tau_bands), seconds=time.perf_counter() - t0, seconds_own=np.nan)]
    print(f"{win['site']} n={win['n']} delay fit [{rows[0]['seconds']:.1f} s] {rows[0]['reason']}", flush=True)
    est, extra = {}, {}
    for mid in METHODS:
        r = dict(base, method=mid, label=LABELS.get(mid, mid), role=ROLE.get(mid, ""), status="ok", reason="",
                 seconds=np.nan, seconds_own=np.nan)
        try:
            frames = None
            if mid == "NIDEM":
                frames = read_frames(dates)            # the cube (before 2023 the Zarr stores) read, not timed
            t0 = time.perf_counter()
            z, note, own, more = run_method(mid, win, win["px"], frames)
            r.update(seconds=time.perf_counter() - t0, reason=note, seconds_own=np.nan if own is None else own)
            z = np.asarray(z, float)
            if np.isfinite(z).any():
                est[mid], extra[mid] = z, more
            else:
                r["status"] = "not computable"
        except (NotComputable, wc.EmptyIntervalError) as e:
            r.update(status="not computable", reason=str(e)[:300])
        except Exception as e:
            r.update(status="error", reason=repr(e)[:300])
        frames = None
        print(f"{win['site']} n={win['n']} {mid:15s} {r['status']} [{r['seconds']:.1f} s] {r['reason'][:90]}", flush=True)
        rows.append(r)
    if "MAREA" in est and "MAREA_ndwi0" in est:
        assert np.array_equal(extra["MAREA_ndwi0"]["z_mid"], est["MAREA"], equal_nan=True), "V1's fit is not V0's"
    if "MAREA_B" in est and "MAREA_B_abst" in est:
        assert np.array_equal(extra["MAREA_B_abst"]["z_all"], est["MAREA_B"], equal_nan=True), "V13's z is not V2's"
    return rows, est, extra
''')

code(r'''
SCORES, EST, EXTRA_PX, TAU, LOOKS = [], {}, {}, {}, {}
for win in windows:
    key = (win["site"], win["n"])
    if win["scenes"] is None:
        SCORES += [dict(site=win["site"], survey_year=win["year"], truth_date=win["end"], n=win["n"],
                        available=win["available"], method=m, label=LABELS.get(m, m), role=ROLE.get(m, ""),
                        status="not reachable", reason=f"{win['available']} screened images on or before {win['end']}")
                   for m in METHODS]
        continue
    rows, EST[key], EXTRA_PX[key] = run_window(win)
    t_true = truth[win["px"]]
    for r in rows:
        if r["method"] in EST[key]:
            r.update(scores(EST[key][r["method"]], t_true))
    SCORES += rows
    w, px = win["scenes"], win["px"]
    TAU[key] = window_delays(win)[1][px]
    C_w = C_S[np.ix_(w, px)]
    with np.errstate(invalid="ignore"):
        LOOKS[key] = dict(n_clear=C_w.sum(axis=0), n_wet=(C_w & (Y_S[np.ix_(w, px)] > 0)).sum(axis=0))
    pd.DataFrame(SCORES).to_csv(f"{OUT}/{SITE}_scores.csv", index=False)
KEYS = list(EST)
print(f"-> {OUT}/{SITE}_scores.csv ({len(SCORES)} rows); windows computed: {KEYS}")
sc = pd.DataFrame(SCORES)
display(sc[sc.status == "ok"].pivot_table(index=["site", "n"], columns="method", values="rmse")
        .reindex(columns=[m for m in ORDER if m in set(sc.method)]).round(3))
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 3 · Error decomposition (exploratory)

For every estimate, on its own pixels of F and on the **common pixels** (F where every estimate of the window has a
value), notebook 01's scores and an additive split of the mean squared error, with d = p − t and the OLS fit of p on
t (slope β, residual e, population moments):

MSE = mean(d)² + ((1 − β) sd(t))² + sd(e)²

*offset* = |mean(d)|, *relief* = |1 − β| sd(t) (the relief a slope below 1 compresses), *scatter* = sd(e) (what the
linear fit leaves). The figure stacks the three squared terms on the common pixels; the number is the RMSE.
""")

code(r'''
def window_of(key):
    return next(w for w in windows if (w["site"], w["n"]) == key)


def decompose(p, t):
    """scores(p, t) plus the additive split MSE = mean_bias² + relief² + scatter² (relief = |1 − β| sd(t),
    scatter = sd of the residual of the OLS fit of p on t)."""
    out = dict(scores(p, t), mse=np.nan, mean_bias=np.nan, relief=np.nan, scatter=np.nan)
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() >= 30:
        d = p[ok] - t[ok]
        tc, pc = t[ok] - t[ok].mean(), p[ok] - p[ok].mean()
        beta = np.mean(tc * pc) / np.mean(tc * tc)
        out.update(mse=float(np.mean(d ** 2)), mean_bias=float(d.mean()), relief=float(abs(1 - beta) * tc.std()),
                   scatter=float((pc - beta * tc).std()))
    return out


rows = []
for key in KEYS:
    t, est = truth[window_of(key)["px"]], EST[key]
    common = np.isfinite(t)
    for mid, z in est.items():
        if mid not in NOT_COMMON:
            common &= np.isfinite(z)
    for mid in ORDER:
        if mid in est:
            for pixels, m in (("own", np.ones(len(t), bool)), ("common", common)):
                rows.append(dict(site=key[0], n=key[1], method=mid, label=LABELS[mid], role=ROLE[mid], pixels=pixels,
                                 **decompose(est[mid][m], t[m])))
decomp = pd.DataFrame(rows)
decomp.to_csv(f"{OUT}/{SITE}_decomposition.csv", index=False)
for pixels in ("common", "own"):
    print(f"{pixels} pixels")
    display(decomp[decomp.pixels == pixels].set_index(["site", "n", "label"])
            [["n_px", "coverage", "rmse", "crmse", "bias", "slope", "mean_bias", "relief", "scatter"]].round(3))
''')

code(r'''
COMP = [("offset²", "mean_bias", PAL["sand"]), ("relief²", "relief", PAL["deep"]), ("scatter²", "scatter", PAL["shallow"])]
fig, axes = plt.subplots(1, len(KEYS), figsize=(3.3 * len(KEYS), 0.26 * len(ORDER) + 1.3), squeeze=False)
for ax, key in zip(axes[0], KEYS):
    d = decomp[(decomp.site == key[0]) & (decomp.n == key[1]) & (decomp.pixels == "common")].set_index("method")
    ms = [m for m in ORDER if m in d.index and np.isfinite(d.loc[m, "mse"])][::-1]
    left = np.zeros(len(ms))
    for name, col, color in COMP:
        v = d.loc[ms, col].to_numpy() ** 2
        ax.barh(np.arange(len(ms)), v, left=left, height=0.72, color=color, edgecolor="white", linewidth=1, label=name)
        left += v
    for i, m in enumerate(ms):
        ax.text(left[i], i, f" {d.loc[m, 'rmse']:.2f}", va="center", fontsize=6.5, color=PAL["secondary"])
    ax.set_yticks(np.arange(len(ms)), [SHORT[m] for m in ms])
    ax.set_xlabel("MSE (m²)")
    ax.set_title(f"{key[0]} n={key[1]} · {int(d.n_px.max()) if len(d) else 0:,} px", loc="left")
    ax.grid(axis="y", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{OUT}/{SITE}_decomposition.png")
''')

md(r"""
### V9 · σ as an uncertainty (exploratory)

Not an estimator (the variants file): for V0 and V1, whose fit gives every pixel a σ (the width of its NDWI
transition, `invert_series`' σ), on each one's own pixels of F with a value, the share of pixels whose error
|p − t| is at most σ and at most 2σ, and Spearman's correlation of |p − t| with σ (σ takes the 8 values of
`marea.SG_GRID`, so the ranks are tied). A σ that were a standard error of a normal error would give about 68 % and
95 %. `<site>_sigma.csv`.
""")

code(r'''
def sigma_check(p, t, sigma):
    """V9 on the pixels where p, t and sigma are finite: the share with |p - t| <= sigma and <= 2 sigma, Spearman's
    correlation of |p - t| with sigma (and its p-value), the median sigma and the median |p - t|."""
    p, t, sigma = np.asarray(p, float), np.asarray(t, float), np.asarray(sigma, float)
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(sigma)
    out = dict(n_px=int(ok.sum()), within_1sigma=np.nan, within_2sigma=np.nan, spearman_rho=np.nan,
               spearman_p=np.nan, median_sigma=np.nan, median_abs_error=np.nan)
    if ok.sum() >= 30:
        e, s = np.abs(p[ok] - t[ok]), sigma[ok]
        rho, pval = spearmanr(e, s)
        out.update(within_1sigma=float(np.mean(e <= s)), within_2sigma=float(np.mean(e <= 2 * s)),
                   spearman_rho=float(rho), spearman_p=float(pval), median_sigma=float(np.median(s)),
                   median_abs_error=float(np.median(e)))
    return out


rows = []
for key in KEYS:
    t = truth[window_of(key)["px"]]
    for mid in ("MAREA", "MAREA_ndwi0"):
        if mid in EST[key] and "sigma" in EXTRA_PX[key].get(mid, {}):
            rows.append(dict(site=key[0], n=key[1], method=mid, label=LABELS[mid],
                             **sigma_check(EST[key][mid], t, EXTRA_PX[key][mid]["sigma"])))
sigma_tab = pd.DataFrame(rows)
sigma_tab.to_csv(f"{OUT}/{SITE}_sigma.csv", index=False)
if len(sigma_tab):
    display(sigma_tab.set_index(["site", "n", "label"]).round(3))
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 4 · The ladder (exploratory)

From MAREA to LogiTide2DEM one ingredient at a time, on the **common pixels of every rung** (F where V0, V1, V2, V2d,
LogiTide2DEM and LogiTide2DEM + MAREA delay all have a value): V0 → V1 changes the read-off point (midpoint → NDWI
= 0); V1 → V2 replaces the least-squares fit of the NDWI magnitude by the probit likelihood of the wet/dry record;
V2 → V2d removes the delay; V2d → LogiTide2DEM replaces the probit grid search by LogiTide2DEM (logistic IRLS, its
separation rule, its filters MSR ≥ 0.6 and z inside the pixel's own range, looks with a defined NDWI only); and
LogiTide2DEM → + MAREA delay adds the delay back. The coverage of F says what each rung's own filters keep. The last
table splits the V2d → LogiTide2DEM step by LogiTide2DEM's case (separated: its gap midpoint; fitted: the logistic).
""")

code(r'''
RUNGS = ["MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_B_nodelay", "E4", "E4_delay"]
STEPS = [("MAREA", "MAREA_ndwi0", "read-off: midpoint → NDWI = 0"),
         ("MAREA_ndwi0", "MAREA_B", "NDWI magnitude (least squares) → wet/dry (probit likelihood)"),
         ("MAREA_B", "MAREA_B_nodelay", "MAREA's delay → none"),
         ("MAREA_B_nodelay", "E4", "probit on a grid → LogiTide2DEM (logistic, its separation rule and filters)"),
         ("E4", "E4_delay", "LogiTide2DEM: no delay → MAREA's delay"),
         ("MAREA_B", "E4_delay", "with MAREA's delay: probit on a grid → LogiTide2DEM")]
E4_CASES = {1: "separated (land below water)", 2: "separated, reversed", 3: "fitted (not separable)"}

lad, steps, cases = [], [], []
for key in KEYS:
    t, est = truth[window_of(key)["px"]], EST[key]
    rungs = [m for m in RUNGS if m in est]
    common = np.isfinite(t)
    for m in rungs:
        common &= np.isfinite(est[m])
    d = {}
    for m in rungs:
        d[m] = dict(site=key[0], n=key[1], method=m, label=LABELS[m], coverage_F=float(np.isfinite(est[m]).mean()),
                    **scores(est[m][common], t[common]))
        lad.append(d[m])
    for a, b, what in STEPS:
        if a in d and b in d:
            steps.append(dict(site=key[0], n=key[1], step=f"{SHORT[a]} → {SHORT[b]}", ingredient=what,
                              n_px=d[a]["n_px"], **{f"d_{k}": d[b][k] - d[a][k] for k in ("rmse", "crmse", "bias", "slope")}))
    if {"MAREA_B_nodelay", "E4"} <= set(d) and "case" in EXTRA_PX[key].get("E4", {}):
        case = EXTRA_PX[key]["E4"]["case"][common]
        tied = EXTRA_PX[key]["MAREA_B_nodelay"]["n_tied"][common] > 1
        a, b, tc = est["MAREA_B_nodelay"][common], est["E4"][common], t[common]
        for c, name in E4_CASES.items():
            m = case == c
            if m.any():
                cases.append(dict(site=key[0], n=key[1], e4_case=name, n_px=int(m.sum()), share=float(m.mean()),
                                  v2d_tied_share=float(tied[m].mean()), rmse_V2d=float(np.sqrt(np.mean((a[m] - tc[m]) ** 2))),
                                  rmse_E4=float(np.sqrt(np.mean((b[m] - tc[m]) ** 2))),
                                  median_E4_minus_V2d=float(np.median(b[m] - a[m])),
                                  mean_abs_E4_minus_V2d=float(np.mean(np.abs(b[m] - a[m])))))
lad, steps, cases = pd.DataFrame(lad), pd.DataFrame(steps), pd.DataFrame(cases)
lad.to_csv(f"{OUT}/{SITE}_ladder.csv", index=False)
steps.to_csv(f"{OUT}/{SITE}_ladder_steps.csv", index=False)
cases.to_csv(f"{OUT}/{SITE}_ladder_cases.csv", index=False)
if len(lad):
    display(lad.set_index(["site", "n", "label"])[["n_px", "coverage_F", "rmse", "crmse", "bias", "slope", "r"]].round(3))
    display(steps.set_index(["site", "n", "step"]).round(3))
    display(cases.set_index(["site", "n", "e4_case"]).round(3))
''')

code(r'''
XR = ["MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_B_nodelay", "E4"]
PANELS = [("rmse", "RMSE (m)"), ("crmse", "centred RMSE (m)"), ("bias", "bias (m)"), ("slope", "slope of p on t")]
fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.6))
for ax, (col, title) in zip(axes, PANELS):
    for i, (key, d) in enumerate(lad.groupby(["site", "n"], sort=False) if len(lad) else []):
        d = d.set_index("method")
        ls, mk, color = WSTYLE[i % len(WSTYLE)]
        xs = [j for j, m in enumerate(XR) if m in d.index]
        ax.plot(xs, [d.loc[XR[j], col] for j in xs], ls=ls, marker=mk, color=color, label=f"{key[0]} n={key[1]}")
        if {"E4", "E4_delay"} <= set(d.index):
            ax.plot([4, 5], [d.loc["E4", col], d.loc["E4_delay", col]], ls=":", color=color, lw=1)
            ax.plot([5], [d.loc["E4_delay", col]], marker=mk, mfc="white", color=color, ls="none")
    ax.set_xticks(range(6), ["V0", "V1", "V2", "V2d", "LT2D", "LT2D+τ"])
    ax.set_title(title, loc="left")
    if col == "bias":
        ax.axhline(0, color=PAL["secondary"], lw=0.7)
    if col == "slope":
        ax.axhline(1, color=PAL["secondary"], lw=0.7)
axes[0].legend(loc="best")
fig.tight_layout()
fig.savefig(f"{OUT}/{SITE}_ladder.png")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5 · Where MAREA loses and where it wins (exploratory)

The **best rival** of a window is the method without a delay (LogiTide2DEM, sampled exceedance, DEA Intertidal,
NIDEM, Flood2Topo) with the largest RMSE gain over V0 on the pixels where both have a value (`<site>_rivals.csv`).
On the pixels where V0, V1, V2, LogiTide2DEM and the best rival all have a value, the errors e = p − t are stratified
by the pixel's delay τ_px and its mouth distance s, its truth elevation, V0's σ and dry NDWI level a, its number of
clear looks (quantile bins, `Q_BINS`; a variable with at most that many values gets one bin per value) and, where
notebook 04's `<site>_pixels.npz` is present, its bed-change class. Per bin: the RMSE and the median error of each
estimate, its RMS centred error (centred with the medians over all these pixels, so the bins' mean squares, weighted
by their shares, give the centred RMSE²) and the paired differences V0 − best rival and V2 − LogiTide2DEM: the RMSE
difference in the bin and its share of the overall MSE difference (`d_mse_share_*`, which add up over the bins).
""")

code(r'''
RIVALS = ["E4", "E5c", "DEA", "NIDEM", "E5a"]
SHOW = ["MAREA", "MAREA_ndwi0", "MAREA_B", "E4"]

bed_path = f"{BED}/{SITE}_pixels.npz"
BED_CLASS = None
if os.path.exists(bed_path):
    with np.load(bed_path) as b:
        BED_CLASS = pd.Series(b["change_class"].astype(str), index=b["keep"])
    print(f"bed change class from {bed_path} ({len(BED_CLASS):,} pixels)")
else:
    print(f"no {bed_path}: no bed-change stratum")


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def best_rival(est, t):
    """{rival: its paired comparison with V0} and the rival with the largest RMSE gain over V0."""
    gains = {}
    for r in RIVALS:
        if r in est and "MAREA" in est:
            ok = np.isfinite(est["MAREA"]) & np.isfinite(est[r]) & np.isfinite(t)
            if ok.sum() >= 30:
                e0, er = est["MAREA"][ok] - t[ok], est[r][ok] - t[ok]
                gains[r] = dict(n_px=int(ok.sum()), rmse_V0=rmse(e0), rmse_rival=rmse(er), gain=rmse(e0) - rmse(er))
    return (max(gains, key=lambda r: gains[r]["gain"]) if gains else None), gains


def bins_of(x, q=Q_BINS):
    """[(label, mask)]: the quantile bins of a numeric x (one bin per value when it has at most q values), or the
    classes of a categorical one."""
    x = np.asarray(x)
    if x.dtype.kind not in "fiu":
        return [(c, x == c) for c in sorted(set(x.tolist()))]
    vals = np.unique(x)
    if len(vals) <= q:
        return [(f"{v:g}", x == v) for v in vals]
    cut = pd.qcut(x, q, duplicates="drop")
    return [(str(c), np.asarray(cut == c)) for c in cut.categories]


def strata(err, cerr, x, pairs, q=Q_BINS):
    """Per bin of x: the pixels, every estimate's RMSE, median error and RMS centred error, and for each pair
    (a, b) the RMSE difference a − b and its share of the overall MSE difference."""
    numeric = np.asarray(x).dtype.kind in "fiu"
    rows = []
    for name, m in bins_of(x, q):
        if not m.any():
            continue
        r = dict(bin=name, x_median=float(np.median(x[m])) if numeric else np.nan, n_px=int(m.sum()),
                 share=float(m.mean()))
        for k, e in err.items():
            r[f"rmse_{k}"], r[f"bias_{k}"], r[f"crms_{k}"] = rmse(e[m]), float(np.median(e[m])), rmse(cerr[k][m])
        for tag, (a, b) in pairs.items():
            r[f"d_rmse_{tag}"] = r[f"rmse_{a}"] - r[f"rmse_{b}"]
            r[f"d_mse_share_{tag}"] = float(m.mean() * (np.mean(err[a][m] ** 2) - np.mean(err[b][m] ** 2)))
        rows.append(r)
    return rows


def stratifiers(key):
    """{name: the value of every pixel of the row's F}: delay, mouth distance, truth, V0's sigma and a, clear
    looks, bed-change class."""
    px = window_of(key)["px"]
    fit = EXTRA_PX[key].get("MAREA_ndwi0", {})
    out = {"delay τ (min)": TAU[key], "mouth distance (km)": s_km[px], "truth z (m)": truth[px]}
    if "sigma" in fit:
        out["V0 σ (m)"], out["V0 dry NDWI a"] = fit["sigma"], fit["a"]
    out["clear looks"] = LOOKS[key]["n_clear"].astype(float)
    if BED_CLASS is not None:
        out["bed change"] = BED_CLASS.reindex(keep[px]).fillna("not in notebook 04").to_numpy(str)
    return out


rivals, strata_rows, BEST = [], [], {}
for key in KEYS:
    t, est = truth[window_of(key)["px"]], EST[key]
    BEST[key], gains = best_rival(est, t)
    rivals += [dict(site=key[0], n=key[1], rival=r, label=LABELS[r], best=(r == BEST[key]), **g) for r, g in gains.items()]
    show = [m for m in dict.fromkeys(SHOW + [BEST[key]]) if m in est]
    S = np.isfinite(t)
    for m in show:
        S &= np.isfinite(est[m])
    err = {m: est[m][S] - t[S] for m in show}
    cerr = {m: (est[m][S] - np.median(est[m][S])) - (t[S] - np.median(t[S])) for m in show}
    pairs = {}
    if BEST[key] and "MAREA" in show:
        pairs["V0_best"] = ("MAREA", BEST[key])
    if {"MAREA_B", "E4"} <= set(show):
        pairs["V2_E4"] = ("MAREA_B", "E4")
    for name, x in stratifiers(key).items():
        x = np.asarray(x)[S]
        sel = np.isfinite(x) if x.dtype.kind in "fiu" else np.ones(len(x), bool)
        for r in strata({m: e[sel] for m, e in err.items()}, {m: c[sel] for m, c in cerr.items()}, x[sel], pairs):
            strata_rows.append(dict(site=key[0], n=key[1], stratum=name, best_rival=BEST[key], n_set=int(S.sum()), **r))
rivals, strat = pd.DataFrame(rivals), pd.DataFrame(strata_rows)
rivals.to_csv(f"{OUT}/{SITE}_rivals.csv", index=False)
strat.to_csv(f"{OUT}/{SITE}_strata.csv", index=False)
if len(rivals):
    display(rivals.set_index(["site", "n", "label"]).round(3))
for key in KEYS:
    d = strat[(strat.site == key[0]) & (strat.n == key[1])].dropna(axis=1, how="all") if len(strat) else strat
    cols = ["n_px"] + [c for c in d.columns if c.startswith(("rmse_", "d_rmse_", "d_mse_share_"))]
    print(f"{key[0]} n={key[1]}: best rival {LABELS.get(BEST[key], BEST[key])}")
    if len(d):
        display(d.set_index(["stratum", "bin"])[cols].round(3))
''')

code(r'''
for key in KEYS:
    d = strat[(strat.site == key[0]) & (strat.n == key[1])]
    names = list(dict.fromkeys(d.stratum))
    if not names:
        continue
    show = [m for m in dict.fromkeys(SHOW + [BEST[key]]) if m and f"rmse_{m}" in d.columns and d[f"rmse_{m}"].notna().any()]
    fig, axes = plt.subplots(2, len(names), figsize=(2.15 * len(names), 4.3), squeeze=False, sharey="row")
    for j, name in enumerate(names):
        e = d[d.stratum == name]
        x = np.arange(len(e))
        ticks = [f"{v:.3g}" for v in e.x_median] if e.x_median.notna().all() else [str(b)[:10] for b in e["bin"]]
        for m in show:
            axes[0, j].plot(x, e[f"rmse_{m}"], marker=MARKERS[m], color=COLORS[m], label=SHORT[m],
                            lw=1.8 if m == "MAREA" else 1.1)
        for k, (tag, color, lab) in enumerate((("V0_best", PAL["accent"], f"V0 − {SHORT.get(BEST[key], '')}"),
                                               ("V2_E4", PAL["dark_sand"], "V2 − LT2D"))):
            if f"d_rmse_{tag}" in e:
                axes[1, j].bar(x + (k - 0.5) * 0.38, e[f"d_rmse_{tag}"], 0.36, color=color, label=lab)
        axes[1, j].axhline(0, color=PAL["ink"], lw=0.7)
        axes[0, j].set_title(name, loc="left")
        for ax in axes[:, j]:
            ax.set_xticks(x, ticks, rotation=35, ha="right")
    axes[0, 0].set_ylabel("RMSE (m)")
    axes[1, 0].set_ylabel("paired ΔRMSE (m)")
    axes[0, -1].legend(loc="best")
    axes[1, -1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{key[0]}_n{key[1]}_strata.png")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 6 · The delay as a source of error (exploratory)

MAREA's delay τ_px of each window (no gauge-derived delay here): the band lags, the share of F and of all the
extraction's pixels with τ_px > 0 and its quantiles on F. Then three pairs, each estimate with MAREA's delay against
itself with zero delay: V0 against *MAREA, no delay*, V2 against V2d, and *LogiTide2DEM + MAREA delay* against
LogiTide2DEM, on the pixels where all six have a value, in bins of τ_px (τ = 0, then quartiles of τ > 0): the RMSE,
the median error and the RMS centred error (centred as in section 5) of each.
""")

code(r'''
DPAIRS = [("MAREA", "MAREA_nodelay", "V0"), ("MAREA_B", "MAREA_B_nodelay", "V2"), ("E4_delay", "E4", "LT2D")]
dsum, dbins = [], []
for key in KEYS:
    win, est = window_of(key), EST[key]
    t, tau = truth[win["px"]], TAU[key]
    tau_bands, tau_all = window_delays(win)
    q = np.percentile(tau, [25, 50, 75])
    dsum.append(dict(site=key[0], n=key[1], band_lags=lags_text(tau_bands), share_F_delayed=float(np.mean(tau > 0)),
                     share_keep_delayed=float(np.mean(tau_all > 0)), tau_F_p25=q[0], tau_F_median=q[1], tau_F_p75=q[2],
                     tau_F_max=float(tau.max())))
    pairs = [p for p in DPAIRS if p[0] in est and p[1] in est]
    S = np.isfinite(t)
    for a, b, _ in pairs:
        S &= np.isfinite(est[a]) & np.isfinite(est[b])
    if not pairs or S.sum() < 30:
        continue
    ts, taus = t[S], tau[S]
    groups = [("τ = 0", taus == 0)]
    pos = np.flatnonzero(taus > 0)
    for lab, m in (bins_of(taus[pos], 4) if len(pos) else []):
        g = np.zeros(len(taus), bool)
        g[pos[m]] = True
        groups.append((f"τ > 0: {lab}", g))
    for lab, g in groups:
        if not g.any():
            continue
        r = dict(site=key[0], n=key[1], tau_bin=lab, n_px=int(g.sum()), share=float(g.mean()),
                 tau_median=float(np.median(taus[g])))
        for a, b, name in pairs:
            for which, mid in (("delay", a), ("none", b)):
                e = est[mid][S] - ts
                c = (est[mid][S] - np.median(est[mid][S])) - (ts - np.median(ts))
                r[f"rmse_{name}_{which}"], r[f"bias_{name}_{which}"], r[f"crms_{name}_{which}"] = \
                    rmse(e[g]), float(np.median(e[g])), rmse(c[g])
            r[f"d_rmse_{name}"] = r[f"rmse_{name}_delay"] - r[f"rmse_{name}_none"]
        dbins.append(r)
dsum, dbins = pd.DataFrame(dsum), pd.DataFrame(dbins)
dsum.to_csv(f"{OUT}/{SITE}_delay_summary.csv", index=False)
dbins.to_csv(f"{OUT}/{SITE}_delay_bins.csv", index=False)
display(dsum.set_index(["site", "n"]).round(3))
if len(dbins):
    display(dbins.set_index(["site", "n", "tau_bin"]).round(3))
''')

code(r'''
for key in KEYS:
    d = dbins[(dbins.site == key[0]) & (dbins.n == key[1])] if len(dbins) else dbins
    names = [p for p in DPAIRS if len(d) and f"rmse_{p[2]}_delay" in d.columns]
    if not names:
        continue
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 4.2), squeeze=False, sharex=True)
    x = np.arange(len(d))
    for j, (a, b, name) in enumerate(names):
        for row, stat, ylab in ((0, "rmse", "RMSE (m)"), (1, "bias", "median error (m)")):
            ax = axes[row, j]
            ax.plot(x, d[f"{stat}_{name}_delay"], marker=MARKERS[a], color=COLORS[a], label=f"{SHORT[a]} (MAREA's τ)")
            ax.plot(x, d[f"{stat}_{name}_none"], marker=MARKERS[b], color=COLORS[b], ls="--", mfc="white",
                    label=f"{SHORT[b]} (τ = 0)")
            if row == 1:
                ax.axhline(0, color=PAL["secondary"], lw=0.7)
            if j == 0:
                ax.set_ylabel(ylab)
        axes[0, j].set_title(name, loc="left")
        axes[0, j].legend(loc="best")
        axes[1, j].set_xticks(x, [f"{v:.0f}" for v in d.tau_median])
        axes[1, j].set_xlabel("delay τ, bin median (min)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{key[0]}_n{key[1]}_delay.png")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 7 · Gallery (exploratory)

The 12 pixels of each window where V0 errs most relative to LogiTide2DEM (largest |e_V0| − |e_LT2D|) and the 4 where
it errs least. Each panel is a pixel's record on its clear looks, NDWI against the level read at its delay,
h_B(t − τ_px) (an undefined NDWI is drawn at 0, as MAREA reads it): wet looks (NDWI > 0) and dry looks, V0's fitted
curve with its midpoint (filled) and V1's crossing of NDWI = 0 (open). The strip below holds the same looks as wet (1)
or dry (0) and V2's probit curve P(wet) with its z. Vertical lines: LogiTide2DEM's elevation (dashed) and the truth.
""")

code(r'''
def pick_gallery(d, worst, best):
    """Positions of the `worst` largest and the `best` smallest values of d (ties in order)."""
    order = np.argsort(-d, kind="stable")
    return order[:worst], np.argsort(d, kind="stable")[:best]


def draw_pixel(ax, strip, key, j, title):
    win, est, fit = window_of(key), EST[key], EXTRA_PX[key]["MAREA_ndwi0"]
    w, p = win["scenes"], win["px"][j]
    h = level_at(int(TAU[key][j]))[w]
    c, y = C_S[w, p], Y_S[w, p]
    with np.errstate(invalid="ignore"):
        wet = c & (y > 0)
    dry, y0 = c & ~wet, np.nan_to_num(y, nan=0.0)
    x = np.linspace(min(LEV[0][w].min(), h.min()), max(LEV[0][w].max(), h.max()), 200)
    ax.scatter(h[wet], y0[wet], s=8, color=PAL["deep"], lw=0, zorder=3)
    ax.scatter(h[dry], y0[dry], s=8, color=PAL["sand"], lw=0, zorder=3)
    a, b, z, sg = fit["a"][j], fit["b"][j], est["MAREA"][j], fit["sigma"][j]
    ax.plot(x, a + b * ndtr((x - z) / sg), color=PAL["accent"], lw=1.4)
    ax.plot([z], [a + b / 2], "o", color=PAL["accent"], ms=4.5, zorder=4)
    z0 = est["MAREA_ndwi0"][j] if "MAREA_ndwi0" in est else np.nan
    if np.isfinite(z0):
        ax.plot([z0], [0], "D", mfc="white", mec=PAL["accent"], ms=4.5, zorder=4)
    ax.axhline(0, color=PAL["grey"], lw=0.6)
    strip.scatter(h[c], wet[c].astype(float), s=6, c=np.where(wet[c], PAL["deep"], PAL["sand"]).tolist(), lw=0)
    if "MAREA_B" in est and np.isfinite(est["MAREA_B"][j]):
        zb, sb = est["MAREA_B"][j], EXTRA_PX[key]["MAREA_B"]["sigma"][j]
        strip.plot(x, ndtr((x - zb) / sb), color=PAL["dark_sand"], lw=1.2)
        strip.plot([zb], [0.5], "s", color=PAL["dark_sand"], ms=3.5)
    for axis in (ax, strip):
        axis.axvline(truth[p], color=PAL["ink"], lw=1.1)
        if np.isfinite(est["E4"][j]):
            axis.axvline(est["E4"][j], color=PAL["deep"], lw=1.1, ls="--")
    strip.set_ylim(-0.25, 1.25)
    strip.set_yticks([0, 1], ["dry", "wet"])
    ax.tick_params(labelbottom=False)
    ax.set_title(title, loc="left", fontsize=7.5)


gal = []
for key in KEYS:
    est = EST[key]
    if not {"MAREA", "E4", "MAREA_ndwi0"} <= set(est):
        print(f"{key}: V0, V1 or LogiTide2DEM missing, no gallery")
        continue
    win = window_of(key)
    t = truth[win["px"]]
    cand = np.flatnonzero(np.isfinite(est["MAREA"]) & np.isfinite(est["E4"]) & np.isfinite(t))
    d = np.abs(est["MAREA"][cand] - t[cand]) - np.abs(est["E4"][cand] - t[cand])
    worst, best = pick_gallery(d, *GALLERY)
    picks = [("worst", k + 1, cand[i], d[i]) for k, i in enumerate(worst)] + \
            [("best", k + 1, cand[i], d[i]) for k, i in enumerate(best)]
    n_rows = int(np.ceil(len(picks) / 4))
    fig = plt.figure(figsize=(12.5, 2.75 * n_rows + 0.5))
    outer = fig.add_gridspec(n_rows, 4, hspace=0.5, wspace=0.28, top=1 - 0.55 / (2.75 * n_rows + 0.5),
                             bottom=0.5 / (2.75 * n_rows + 0.5))
    for k, (kind, rank, j, dj) in enumerate(picks):
        inner = outer[k].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.06)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=ax)
        draw_pixel(ax, strip, key, j, f"{kind} {rank} · Δ|e| {dj:+.2f} m · τ {TAU[key][j]:.0f} min")
        gal.append(dict(site=key[0], n=key[1], kind=kind, rank=rank, keep=int(keep[win["px"][j]]), d_abs_error=dj,
                        truth=t[j], tau_px=TAU[key][j], n_clear=int(LOOKS[key]["n_clear"][j]),
                        n_wet=int(LOOKS[key]["n_wet"][j]),
                        **{f"fit_{v}": EXTRA_PX[key]["MAREA_ndwi0"][v][j] for v in ("a", "b", "sigma")},
                        **{m: est[m][j] for m in ORDER if m in est}))
    handles = [Line2D([], [], marker="o", ls="none", color=PAL["deep"], label="wet look"),
               Line2D([], [], marker="o", ls="none", color=PAL["sand"], label="dry look"),
               Line2D([], [], color=PAL["accent"], marker="o", label="V0 fit, midpoint"),
               Line2D([], [], ls="none", marker="D", mfc="white", mec=PAL["accent"], label="V1 crossing"),
               Line2D([], [], color=PAL["dark_sand"], marker="s", label="V2 P(wet)"),
               Line2D([], [], color=PAL["deep"], ls="--", label="LogiTide2DEM"),
               Line2D([], [], color=PAL["ink"], label="truth")]
    fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("boundary level at the pixel's delay, h_B(t − τ_px) (m)", fontsize=8, y=0.0)
    fig.savefig(f"{OUT}/{key[0]}_n{key[1]}_gallery.png")
gal = pd.DataFrame(gal)
gal.to_csv(f"{OUT}/{SITE}_gallery.csv", index=False)
if len(gal):
    display(gal.set_index(["site", "n", "kind", "rank"]).round(3))
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 8 · The addendum: V13's interval, the error-coverage curves and the bootstrap (exploratory)

Computed for the windows where the estimates exist (`NB_METHODS` names them). **V13's interval:** the share of the
pixels of F whose 95 % interval for MAREA-B is bracketed by the grid (V13a's coverage) and the share whose error
|MAREA-B − t| lies within the interval's half-width (nominal 0.95, on all pixels and on the bracketed ones).
**Error-coverage curves:** every estimate (V13a aside, which abstains by itself), on a reliability common to all and
blind to every method, r = min(wet looks, dry looks) of the window's clear looks (wet = NDWI > 0): at 100, 95, 90, 85
and 80 % of F, the pixels of largest r (ties by pixel index), the scores on each estimate's pixels with a value and on
the pixels common to all (`v13.error_coverage`). **Bootstrap:** every diagnostic against V0 on the pixels both have,
the paired difference of RMSE and centred RMSE with its 95 % interval, blocks of 32 × 32 pixels of the frame, 1,000
resamples, seed 20261028 (`v13.block_bootstrap`). The judging rule needs the three sites and is applied in the report.
""")

code(r'''
cal, cov, boot = [], [], []
for key in KEYS:
    win, est = window_of(key), EST[key]
    t = truth[win["px"]]
    if key in BINT:
        b = BINT[key]
        fin = np.isfinite(b["z"]) & np.isfinite(t)
        half, err = (b["z_hi"] - b["z_lo"]) / 2, np.abs(b["z"] - t)
        inb = fin & b["bracketed"]
        cal.append(dict(site=key[0], n=key[1], n_px=int(fin.sum()), bracketed=float(b["bracketed"][fin].mean()),
                        within_halfwidth=float((err[fin] <= half[fin]).mean()),
                        within_halfwidth_bracketed=float((err[inb] <= half[inb]).mean()) if inb.any() else np.nan,
                        median_halfwidth_m=float(np.median(half[fin]))))
    looks = LOOKS[key]
    r = np.minimum(looks["n_wet"], looks["n_clear"] - looks["n_wet"])
    ec = v13.error_coverage(t, {m: est[m] for m in ORDER if m in est and m not in NOT_COMMON}, r)
    cov.append(ec.assign(site=key[0], n=key[1]))
    rows_px, cols_px = np.divmod(keep[win["px"]], W)
    if "MAREA" in est:
        for m in ORDER:
            if m in est and ROLE.get(m) == "diagnostic":
                boot.append(dict(site=key[0], n=key[1], method=m, label=LABELS[m],
                                 **v13.block_bootstrap(t, est[m], est["MAREA"], rows_px, cols_px)))
cal, cov, boot = pd.DataFrame(cal), (pd.concat(cov, ignore_index=True) if cov else pd.DataFrame()), pd.DataFrame(boot)
for name, df in (("interval", cal), ("coverage", cov), ("bootstrap", boot)):
    df.to_csv(f"{OUT}/{SITE}_{name}.csv", index=False)
    print(f"-> {OUT}/{SITE}_{name}.csv ({len(df)} rows)")
if len(cal):
    display(cal.round(3))
if len(boot):
    display(boot.round(4))

if len(cov):                                           # RMSE on the common pixels against the coverage of F
    mine = [m for m in ORDER if m.startswith("MAREA")]
    fig, axes = plt.subplots(1, len(KEYS), figsize=(3.4 * len(KEYS), 3.4), squeeze=False)
    for ax, key in zip(axes[0], KEYS):
        d = cov[(cov.site == key[0]) & (cov.n == key[1])]
        for m in d.method.unique():
            e = d[d.method == m].sort_values("coverage")
            ax.plot(100 * e.coverage, e.rmse_common, marker=MARKERS.get(m, "o"), ms=3.5 if m in mine else 2.5,
                    lw=1.8 if m in mine else 0.8, color=COLORS.get(m, PAL["grey"]) if m in mine else PAL["grey"],
                    alpha=1.0 if m in mine else 0.8, zorder=3 if m in mine else 2, label=SHORT.get(m, m))
        ax.invert_xaxis()
        ax.set(title=f"{key[0]} n={key[1]}", xlabel="coverage of F (%), most reliable pixels kept",
               ylabel="RMSE on the common pixels (m)")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=2)
    fig.savefig(f"{OUT}/{SITE}_coverage.png")
    plt.show()
''')

md(r"""
## 9 · Outputs, provenance and the check against notebook 01

The per-pixel arrays: for every analysed row its pixels of F (`<row>_keep`, flat indices in the frame), their truth
and mouth distance; for every window (`<row>_n<n>_`) every estimate, the extras (V0's fit a, b, σ and z as `fit_*`,
V2's σ and tied levels, V10's σ, lapse rate and tied levels, LogiTide2DEM's MSR, case and valid looks, sampled
exceedance's frequency), τ_px and the clear and wet looks. Then the provenance, and the check: every row this notebook
shares with notebook 01 (its run of the same windows, `NTABLES`, else its first run, `FIRST`) must be that row bit for
bit: the status, the note (MAREA's band lags among them), the window's dates, F and the scores on F. A window that
notebook 01's run could not reach (it had no pre-2023 scenes) is not compared.
""")

code(r'''
arrays = {}
for row in ROWS:
    if row["year"] in ROW_YEARS:
        arrays.update({f"{row['site']}_keep": keep[row["px"]], f"{row['site']}_truth": truth[row["px"]],
                       f"{row['site']}_s_km": s_km[row["px"]]})
for key in KEYS:
    tag = f"{key[0]}_n{key[1]}"
    arrays[f"{tag}_tau_px"] = TAU[key]
    arrays.update({f"{tag}_{k}": v for k, v in LOOKS[key].items()})
    for mid, z in EST[key].items():
        arrays[f"{tag}_{mid}"] = z
        for k, v in EXTRA_PX[key].get(mid, {}).items():
            name = f"{tag}_fit_{k}" if mid == "MAREA_ndwi0" else f"{tag}_{mid}_{k}"
            arrays[name.replace("fit_z_mid", "fit_z")] = v
np.savez_compressed(f"{OUT}/{SITE}_pixels.npz", **arrays)
print(f"-> {OUT}/{SITE}_pixels.npz: {len(arrays)} arrays")
print("written:", sorted(os.path.basename(f) for f in glob.glob(f"{OUT}/{SITE}*")))
''')

code(r'''
import platform, socket, subprocess
import scipy
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.rstrip()
dirty = [l for l in git("status", "--porcelain").splitlines() if not l.startswith("??")]
print("commit  ", git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD"))
print(f"modified {len(dirty)} tracked file(s)" + "".join(f"\n  {l}" for l in dirty[:15]))
print("host    ", socket.gethostname(), "· OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
print("versions", f"python {platform.python_version()} · numpy {np.__version__} · pandas {pd.__version__} · "
      f"xarray {xr.__version__} · scipy {scipy.__version__} · pyintertidal {pit.__version__}")
''')

code(r'''
ref = next((f"{d}/{SITE}.csv" for d in (NTABLES, FIRST) if os.path.exists(f"{d}/{SITE}.csv")), None)
assert ref, f"neither {NTABLES}/{SITE}.csv nor {FIRST}/{SITE}.csv: notebook 01's rows cannot be checked"
first = pd.read_csv(ref, float_precision="round_trip")
new = pd.read_csv(f"{OUT}/{SITE}_scores.csv", float_precision="round_trip")
key, same = ["site", "n", "method"], ["status", "reason", "first_date", "last_date", "n_F", "n_px", "coverage",
                                      "rmse", "bias", "crmse", "slope", "r"]
rows_here = {w["site"] for w in windows}
done = first.method.isin(METHODS) & first.n.isin(NS) & first.site.isin(rows_here) & (first.status != "not reachable")
a = first[done].reindex(columns=key + same).set_index(key)            # the windows notebook 01's run computed
b = new[new.method.isin(set(first.method))].reindex(columns=key + same).set_index(key)
both = a.index.intersection(b.index)
assert len(both), f"no window of n = {NS} computed in {ref}: run notebook 01 on these windows (NB_NTABLES)"
pd.testing.assert_frame_equal(a.loc[both].sort_index(), b.loc[both].sort_index(), check_exact=True, check_dtype=False)
print(f"{len(both)} rows bit for bit those of {ref}: " +
      ", ".join(f"{m} ×{k}" for m, k in Counter(i[2] for i in both).items()))
missing = sorted(set(a.index) - set(both))
print("rows of notebook 01 not computed here:", missing or "none")
''')


# ─────────────────────────────────────────────────────────────────────────────
def build():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    for k, (kind, src) in enumerate(CELLS):
        cell = nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        cell["id"] = f"nb05-{k:02d}"
        nb.cells.append(cell)
    nbformat.validate(nb)
    return nb


if __name__ == "__main__":
    text = nbformat.writes(build()) + "\n"
    if "--check" in sys.argv:
        on_disk = open(NB_PATH, encoding="utf-8").read() if os.path.exists(NB_PATH) else ""
        print("up to date" if on_disk == text else f"{NB_PATH} differs from the builder")
        sys.exit(0 if on_disk == text else 1)
    with open(NB_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")
