"""Insert the diagnostic panels and rework the tide cells."""
import json, os, shutil

NB = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\intertidal_topography_villaviciosa.ipynb"
shutil.copy(NB, NB + ".bak")
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]


def idx(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i
    raise KeyError(cell_id)


def code(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.rstrip("\n").split("\n"), "outputs": [],
            "execution_count": None}


def fix(cell):
    """Notebook JSON wants a list of lines, each keeping its newline."""
    s = cell["source"]
    cell["source"] = [l + "\n" for l in s[:-1]] + [s[-1]]
    return cell


NEW = [
    # (after this cell id, new cell)
    ("3efb0f37", code("rpt-dates", """\
# What the archive actually offers here. The revisit statistics matter more
# than the raw count: a decade with one long gap is not the same archive as a
# decade sampled evenly, and those gaps are what later decide which epochs can
# be fitted at all. Open "Every date" for the full listing.
explain.dates_available(cube.dates, TIME_EXTENT)""")),

    ("9a865471", code("rpt-clouds", """\
# Which dates survived and, more to the point, WHY. Three outcomes:
#   [ref]  clean over the whole scene, so it votes in the reference map
#   [✓]    too cloudy globally but clear over the transition zone — an
#          observation a whole-scene filter would have thrown away
#   [✗]    cloudy where it matters
explain.cloud_screening(cloud_pct, usable_dates,
                        clean_scene_max_bad=CLEAN_SCENE_MAX_BAD,
                        cloud_threshold=CLOUD_THRESHOLD)""")),

    ("ff0e1029", code("rpt-wf", """\
# The product itself: shape, grid, valid pixels and range, on the same panel
# the cube gets. `Valid` is the number to read first — a map can look complete
# and still be mostly NaN.
explain.summarize(water_freq, name="water frequency", transform=transform,
                  crs=crs, long_name="Fraction of clear observations wet")""")),

    ("54a81380", code("rpt-inter", """\
# Where every pixel of the candidate zone ended up. The transition class is
# only a CANDIDATE; the water-frequency window then splits it into ground
# almost never wet, ground almost always wet, and the intertidal proper.
# Reporting the three together is what makes the window a defensible choice
# instead of a magic number — widen it and you see exactly what you gained.
explain.intertidal_breakdown(reference_map, water_freq, intertidal,
                             WF_LOW, WF_HIGH, transform)""")),
]

for after, cell in NEW:
    if any(c.get("id") == cell["id"] for c in cells):
        continue
    cells.insert(idx(after) + 1, fix(cell))
    print("insertada", cell["id"], "tras", after)

# ── Tides: the continuous series over the WHOLE period ───────────────────
c = cells[idx("556def2c")]
src = "".join(c["source"])
src = src.replace(
    '''series_times, series_heights = tides.series(aoi, "2024-01-01", "2024-12-31",
                                            every_hours=1)''',
    '''# The full study period, not one year: the tidal frame is a decadal
# quantity. The 18.6-year nodal cycle plus the annual and semiannual terms
# all move the extremes, so a single year understates the frame our
# elevations are referenced to. About a minute of pyTMD for ten years.
series_times, series_heights = tides.series(aoi, TIME_EXTENT[0],
                                            TIME_EXTENT[1], every_hours=1)''')
c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print("seccion 6: serie continua sobre TIME_EXTENT completo")

# ── Tides: the three-panel record ────────────────────────────────────────
new_plot = """\
# Three views of the tidal frame over the whole period:
#   left    every modelled tide height — the envelope the estuary experiences
#   centre  only the heights the satellite caught
#   right   those same heights against the true extremes, so the band no
#           image reaches is a distance you can read in metres
sampled_times = pd.to_datetime(sorted(tide_heights))
sampled = np.array([tide_heights[d] for d in sorted(tide_heights)])

viz.plot_tide_record(series_times, series_heights, sampled_times, sampled,
                     model_name=TIDE_MODEL, location=tide_point);"""
c = cells[idx("3b3a63d2")]
c["source"] = [l + "\n" for l in new_plot.split("\n")[:-1]] + [new_plot.split("\n")[-1]]
c["outputs"] = []
c["execution_count"] = None
print("seccion 7: plot_tide_record de 3 paneles")

# `pandas` se usaba a partir de esa celda; subirlo al setup
c = cells[idx("ad216632")]
s = "".join(c["source"])
if "import pandas" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport pandas as pd\n")
    c["source"] = [l + "\n" for l in s.split("\n")[:-1]] + [s.split("\n")[-1]]
    print("setup: anadido import pandas")

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nOK -> {len(cells)} celdas, copia de seguridad en .bak")
