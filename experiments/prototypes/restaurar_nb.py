"""Restore the three cells VS Code overwrote, with the new auto binning."""
import json, shutil

NB = (r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
      r"\intertidal_topography_villaviciosa.ipynb")
shutil.copy(NB, NB + ".bak2")
nb = json.load(open(NB, encoding="utf-8"))


def cell(cid):
    return next(c for c in nb["cells"] if c.get("id") == cid)


def put(cid, src):
    c = cell(cid)
    lines = src.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c["outputs"] = []
    c["execution_count"] = None


# ── section 5: the wider water-frequency window ──────────────────────────
put("54a81380", '''\
# ── Parameters ──────────────────────────────────────────────────────────
transition = reference_map == 0
otsu_low, otsu_high = pit.multiotsu_window(water_freq, transition)
print(f"multi-Otsu suggests the intertidal window "
      f"[{otsu_low:.2f}, {otsu_high:.2f}]")

# Adopted wider than the suggestion, on purpose. Multi-Otsu splits the
# transition histogram into three populations and hands back the two valleys,
# which cuts off the ends of the flat: the lowest ground, wet on almost every
# pass, and the highest, dry on almost every pass. Those ends are real
# intertidal and the fit can resolve them, so the window is opened and the
# per-pixel quality control is left to decide instead.
WF_LOW, WF_HIGH = 0.05, 0.95

# ── Intertidal mask (also clipped to the study polygon) ─────────────────
polygon_mask   = aoi.raster_mask(transform, crs, reference_map.shape)
intertidal     = pit.intertidal_mask(water_freq, reference_map,
                                     WF_LOW, WF_HIGH, aoi_mask=polygon_mask)

area_km2 = intertidal.sum() * px_km2
narrow = pit.intertidal_mask(water_freq, reference_map, otsu_low, otsu_high,
                             aoi_mask=polygon_mask)
print(f"intertidal zone: {int(intertidal.sum()):,} px = {area_km2:.2f} km\\u00b2 "
      f"(the multi-Otsu window would give {int(narrow.sum()):,} px = "
      f"{narrow.sum() * px_km2:.2f} km\\u00b2)")

log.record("intertidal mask", kind="mask",
           params={"wf_low": WF_LOW, "wf_high": WF_HIGH},
           outputs=["intertidal mask"], note=f"{area_km2:.2f} km\\u00b2")''')

# ── section 8: the fit, with automatic binning and despeckling ───────────
put("30fe0e1f", '''\
# \\u2500\\u2500 Parameters \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500
EPOCH_YEARS     = 3       # morphology is stable enough within ~3 years
SCALE           = 4       # super-resolution factor: 10 m \\u2192 2.5 m
MAX_UNCERTAINTY = 0.10    # drop pixels whose \\u03c3(\\u03bc) exceeds 10 cm
SIGMA_FLOOR     = 0.05    # noise floor of \\u03c3, not injected as relief
TIDE_BINS       = "auto"  # from this archive\\u0027s own tide sampling
MIN_REGION_PX   = 25      # blank isolated specks (25 px at 10 m = 0.25 ha)

epoch_list = pit.epochs(sorted(tide_heights), epoch_years=EPOCH_YEARS)
for e in epoch_list:
    print(f"  epoch {e[\\'label\\']}: {len(e[\\'dates\\'])} dates")

latest = epoch_list[-1]
print(f"\\\\nfitting the most recent epoch: {latest[\\'label\\']}")

# \\u2500\\u2500 HSR fit (local: no cloud job) \\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500
# `tide_bins` averages every observation sharing a tide level before fitting:
# scene noise is independent of tide height while the signal is a function of
# it, so averaging cancels one and keeps the other.
#
# The COUNT is not a free choice. Too few bins quantise the tide axis until a
# pixel\\u0027s whole transition fits inside one bin and its elevation stops being
# identifiable there; too many average nothing. Measured on the Tagus against
# a real bathymetric survey: 40 bins gave RMSE 0.339 m with 39 % of pixels
# pinned at the bottom of the \\u03c3 grid, no binning gave 0.289 m, and the optimum
# near 150 bins gave 0.275 m. "auto" derives the count from the tide sampling
# of this archive, which is what sets that balance.
dem = pit.fit_hsr(
    cube, tide_heights,
    dates=latest["dates"],
    scale=SCALE,
    sigma_floor=SIGMA_FLOOR,
    max_uncertainty=MAX_UNCERTAINTY,
    clip_mask=intertidal,
    epoch_label=latest["label"],
    tide_bins=TIDE_BINS,
)

# A per-pixel fit makes per-pixel decisions, so a few isolated pixels far
# from the estuary can pass on their own. Real flats are contiguous.
before = int(np.isfinite(dem.mu).sum())
dem.mu = pit.terrain.drop_small_regions(dem.mu, min_region_px=MIN_REGION_PX)
print(f"\\\\ndespeckled: {before:,} \\u2192 {int(np.isfinite(dem.mu).sum()):,} px")
dem.summary()

log.record("HSR elevation", kind="apply",
           params={"epoch": latest["label"], "scale": SCALE,
                   "tide_bins": TIDE_BINS,
                   "max_uncertainty": MAX_UNCERTAINTY},
           inputs=[f"{len(latest[\\'dates\\'])} dates", "tide heights"],
           outputs=["elevation", "sub-pixel relief", "uncertainty"])''')

# ── the step baseline, despeckled the same way ───────────────────────────
put("0a6b8498", '''\
dem_step = pit.fit_step(
    cube, tide_heights, dates=latest["dates"],
    threshold=NDWI_THRESHOLD, min_obs=5,
    clip_mask=intertidal, epoch_label=latest["label"],
)
dem_step.mu = pit.terrain.drop_small_regions(dem_step.mu,
                                             min_region_px=MIN_REGION_PX)
dem_step.summary()

print(f"\\\\ncoverage: HSR {int(np.isfinite(dem.mu).sum()):,} px vs "
      f"step {int(np.isfinite(dem_step.mu).sum()):,} px")''')

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

import nbformat
nbformat.validate(nbformat.read(NB, as_version=4))
code = "\n".join("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code")
for frag, desc in [('TIDE_BINS       = "auto"', "binning automatico"),
                   ("drop_small_regions", "despeckle"),
                   ("WF_LOW, WF_HIGH = 0.05", "ventana ancha")]:
    print(f"  {desc:22s} {'OK' if frag in code else '** FALTA **'}")
print("notebook valido,", len(nb["cells"]), "celdas")
