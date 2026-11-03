# -*- coding: utf-8 -*-
"""Builds notebooks/validation/v3_production_site.ipynb, the production of one validation v3 cube.

The production is Ferrol's (the design's amendment of 2026-11-03): this builder takes the code cells of
``tide_boundary_comparison_ferrol.ipynb`` that make it (sections 0-8: setup, area, cube, threshold, screening and
intertidal zone, boundary, levels at the scenes, MAREA's lag fit, the plain and MAREA inversions) and changes only
what the v3 cubes need, each change checked against the template (a template that no longer holds the text stops the
build): the parameters come from ``experiments/v3/sites.py`` (``NB_V3`` names the truth), the AOI is the download box
instead of a registry polygon, and the overpass times are the downloader's file. Ferrol's sections 9-10b (Granadeiro,
the gauge pair, the external reference) need instruments these sites do not have and are left out. The last cell
writes the products where notebooks 01 and 05 read them and a provenance record.

    python notebooks/validation/_build_v3_production.py
    NB_V3=foz_2016 jupyter nbconvert --execute notebooks/validation/v3_production_site.ipynb   (or the run script)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "tide_boundary_comparison_ferrol.ipynb")
OUT = os.path.join(ROOT, "notebooks", "validation", "v3_production_site.ipynb")


def source(nb, i):
    return "".join(nb["cells"][i]["source"])


def swap(src, old, new):
    """``src`` with ``old`` replaced once; the template must hold it exactly once."""
    if src.count(old) != 1:
        raise SystemExit(f"template changed: {old!r} found {src.count(old)} times")
    return src.replace(old, new)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


PARAMS = r'''
import os, sys

sys.path.insert(0, os.getcwd())
from experiments.v3.sites import NEW, V3

KEY = os.environ.get("NB_V3", "foz_2016")          # the truth whose cube this run produces
assert KEY in NEW, f"{KEY!r} is not a new v3 cube; one of {NEW}"
V3C = V3[KEY]                                      # (not C: the template's cells use C for the clear mask)

# ── Which area, and over what period (experiments/v3/sites.py) ───────────
SITE         = V3C["site"]
TIME_EXTENT  = V3C["period"]
RESOLUTION_M = 10
CACHE        = V3C["cube"]
OUT_DIR      = V3C["out"]
OVERPASS_CACHE = V3C["overpass"]                     # the downloader's overpass times of every cube date
MIN_SCENES   = V3C["min_scenes"]

# ── The boundary: two switches ──────────────────────────────────────────
TIDE_MODEL   = V3C["tide_model"]
N_GAUGES     = V3C["n_gauges"]
JUDGE_GAUGES = V3C["judges"]
MOUTH_SIDE   = V3C["mouth_side"]

# ── Derived ─────────────────────────────────────────────────────────────
BOUNDARY_TAG = f"{(TIDE_MODEL or 'nomodel').lower()}_g{N_GAUGES or 0}"
print(f"v3 truth {KEY}: {SITE} · {TIME_EXTENT[0]} → {TIME_EXTENT[1]} @ {RESOLUTION_M} m")
print(f"  boundary  model={TIDE_MODEL}  gauges={N_GAUGES} (expected {V3C['expected_gauge']})  "
      f"held out={JUDGE_GAUGES}  → tag {BOUNDARY_TAG}")
print(f"  cache     {CACHE}")
print(f"  outputs   {OUT_DIR}")
'''

SAVE = r'''
import subprocess

prod_dir = f"{OUT_DIR}/comparison_{BOUNDARY_TAG}"
os.makedirs(prod_dir, exist_ok=True)
np.savez_compressed(f"{prod_dir}/products.npz",
                    keep=keep, shape=np.array([H, W]), s_km=s_km, band=band_of,
                    z_plain=z_plain, z_marea=z_marea, z_marea_fit=z_marea_fit,
                    record_dates=np.asarray(ex["dates"][have])[okh], level_range=np.array([lo, hi]))
band_table.to_csv(f"{prod_dir}/marea_effect_by_band.csv", index=False)
summary = pd.DataFrame([{"product": name, "px mapped": int(np.isfinite(z).sum()),
                         "median z (m)": float(np.nanmedian(z)),
                         "median |Δz| vs plain (m)": float(np.nanmedian(np.abs(z - z_plain)))}
                        for name, z in (("no delay", z_plain), ("MAREA", z_marea))])
display(summary.round(3))


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


record = {
    "key": KEY, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in V3C.items()},
    "design": "experiments/validation_v3_cantabrian_2026-10-29.md (amendment of 2026-11-03)",
    "template": "tide_boundary_comparison_ferrol.ipynb (sections 0-8, built by notebooks/validation/_build_v3_production.py)",
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "cube": {"path": CACHE, "bytes": os.path.getsize(CACHE), "scenes": len(cube.dates), "shape": [H, W]},
    "usable_dates": len(usable_dates), "scene_rule": SCENE_RULE, "ndwi_threshold": NDWI_THRESHOLD,
    "boundary": boundary.name, "boundary_members": member_codes,
    "marea": {"record_scenes": int(marea_res["n_escenas"]), "pixels": int(marea_res["n_px"]),
              "pixels_with_z": int(marea_res["n_px_cota"]), "band_centres_km": [float(c) for c in centres],
              "tau_fit_min": [float(t) for t in tau_fit], "tau_used_min": [float(t) for t in tau_used]},
    "extraction": {"path": EXTRACT, "keep_px": int(len(keep)), "dates": int(len(ex["dates"]))},
    "products": f"{prod_dir}/products.npz", "marea_dir": f"{OUT_DIR}/marea_{BOUNDARY_TAG}",
    "pyintertidal": pit.__version__,
}
json.dump(record, open(f"{OUT_DIR}/v3_production.json", "w", encoding="utf-8"), indent=1, default=str)
assert V3C["expected_gauge"] in member_codes, f"boundary gauge {member_codes}, expected {V3C['expected_gauge']}"
print(f"-> {prod_dir}/products.npz, {OUT_DIR}/marea_{BOUNDARY_TAG}/ and {OUT_DIR}/v3_production.json")
'''


def build(template=TEMPLATE, out=OUT):
    nb = json.load(open(template, encoding="utf-8"))
    setup = swap(source(nb, 4), 'log = explain.OpLog(f"Escalda {TIME_EXTENT[0][:4]}–{TIME_EXTENT[1][:4]}")',
                 'log = explain.OpLog(f"{SITE} v3 {TIME_EXTENT[0][:4]}–{TIME_EXTENT[1][:4]}")')
    area = swap(source(nb, 6), "aoi = pit.sites.get(SITE)",
                'from pyintertidal.aoi import AOI\n\naoi = AOI.from_bbox(**V3C["box"], name=f"{SITE} (v3 box)")')
    levels = swap(source(nb, 17), 'OVERPASS_CACHE = f"{OUT_DIR}/overpass_times.json"\n',
                  'assert os.path.exists(OVERPASS_CACHE), f"{OVERPASS_CACHE}: the downloader writes it"\n')
    cells = [
        md("""
# Validation v3 · the production of one cube

The production that every v3 method reads (the design `experiments/validation_v3_cantabrian_2026-10-29.md`, its
amendment of 2026-11-03): Ferrol's production notebook (`tide_boundary_comparison_ferrol.ipynb`), sections 0-8,
on the cube of one truth. `NB_V3` names it (`experiments/v3/sites.py`): `urdaibai_2017`, `urdaibai_2024`,
`foz_2016` or `eo_2020`. It writes the production extraction (`marea_extract.npz`: the intertidal pixels, their
wet/dry record and their distance from the mouth), MAREA's product (`marea_<tag>/`) and the no-delay and MAREA
inversions (`comparison_<tag>/products.npz`), where notebooks 01 and 05 read them. Built by
`notebooks/validation/_build_v3_production.py`; the code cells are the template's, changed only where noted.
"""),
        md("""
## Parameters

From `experiments/v3/sites.py`: the cube, the period, the boundary (EOT20 + the nearest IOC gauge, none held out),
the sea's edge and the record floor of the lag fit. Changed from the template: everything comes from the truth's
entry, and the overpass times are the downloader's file (`products_<site>_10m/overpass_times.json`).
"""),
        code(PARAMS),
        md("## 0 · Setup\n\nNo Copernicus connection is needed: the cube is cached."),
        code(setup),
        md("""
## 1 · Study area and its instruments

Changed from the template: the AOI is the download box of `experiments/download_cube_v3.py` (the registry polygons
leave part of these estuaries out). The gauges come from the IOC station list: the nearest one forms the boundary.
"""),
        code(area),
        md("## 2 · The datacube\n\nThe merged cube of the truth's years (B03, B08, SCL at 10 m). Every later step streams it."),
        code(source(nb, 8)),
        md("## 3 · Water detection threshold\n\nOtsu over the clearest scenes is printed; the adopted value is NDWI = 0."),
        code(source(nb, 10)),
        md("## 4 · Coastal stability, cloud screening, water frequency, intertidal zone\n\nThe production parameters."),
        code(source(nb, 12)),
        code(source(nb, 13)),
        md("## 5 · The boundary\n\nEOT20 and the nearest gauge, and their consensus, which is the boundary."),
        code(source(nb, 15)),
        md("## 6 · Tide at each scene, and is the site mappable?"),
        code(levels),
        md(source(nb, 18)),
        code(source(nb, 19)),
        md(source(nb, 20)),
        code(source(nb, 21)),
        code(source(nb, 22)),
        md("""
## 9 · The products and their provenance

What notebooks 01 and 05 read: `comparison_<tag>/products.npz` (the no-delay inversion `z_plain` gives the range
the truth is kept in), and `v3_production.json` (the inputs, the boundary, the record and the lags).
"""),
        code(SAVE),
    ]
    out_nb = {"cells": cells, "metadata": {"kernelspec": nb["metadata"]["kernelspec"],
                                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(out_nb["cells"]):
        c["id"] = f"v3prod-{i:02d}"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out_nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return out


if __name__ == "__main__":
    print("wrote", build(*sys.argv[1:3]))
