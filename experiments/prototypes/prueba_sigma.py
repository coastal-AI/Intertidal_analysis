"""Does raising the sigma floor to the tide sampling kill the band?

The sigmoid cannot be narrower than the spacing of the tide observations
that define it: below that, mu slides freely inside one interval and the fit
snaps to an arbitrary value. So the floor of the sigma grid is not a free
parameter — it is set by how finely the tide was sampled.
"""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import pyintertidal as pit
from pyintertidal.elevation import hsr_stage1, epochs
from pyintertidal import SentinelCube, terrain

CACHE = "ndwi_cube_tejo_2019_2021.nc"
aoi = pit.sites.get("tejo")
cube = SentinelCube(aoi, ("2019-01-01", "2021-12-31"), water="ndwi",
                    cache_path=CACHE, resolution=10)
tides = json.load(open("products_tejo/tides.json"))
ep = epochs(sorted(tides), 3)[-1]

BINS = 40
t = np.array([tides[d] for d in ep["dates"] if d in tides])
width = (t.max() - t.min()) / BINS
print(f"rango {t.min():+.2f}..{t.max():+.2f}, {BINS} bins -> "
      f"{width:.4f} m por bin")
print(f"suelo teorico de sigma (mitad del bin): {width/2:.3f} m\n")


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape


inter, TR, CRS, SH = load("products_tejo/intertidal_mask.tif")
inter = inter > 0
bathy = pit.reproject_to_grid("products_tejo/bathy_tagus_native.tif",
                              TR, CRS, SH)
step = load("products_tejo/step_elevation.tif")[0]

FULL = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
GRIDS = {
    "actual (desde 0.03)": FULL,
    "suelo en 0.06": tuple(s for s in FULL if s >= 0.06),
    "suelo en 0.10": tuple(s for s in FULL if s >= 0.10),
}

results = {}
for label, grid in GRIDS.items():
    s1 = hsr_stage1(CACHE, cube.dates, tides, valid_dates=ep["dates"],
                    tide_bins=BINS, sg_grid=grid)
    valid = s1["valid"] & inter
    valid &= np.isfinite(s1["sigma_mu"]) & (s1["sigma_mu"] <= 0.10)
    mu = np.where(valid, s1["mu"], np.nan).astype("float32")
    mu = terrain.drop_small_regions(mu, min_region_px=25)
    m = np.isfinite(mu) & np.isfinite(bathy) & inter
    b = np.median(mu[m] - bathy[m])
    rmse = np.sqrt(np.mean((mu[m] - b - bathy[m]) ** 2))
    r = np.corrcoef(mu[m], bathy[m])[0, 1]
    sg = s1["sigma"][valid]
    at_floor = 100 * np.mean(np.isclose(sg, grid[0]))
    results[label] = (m.sum(), rmse, r, at_floor, mu)
    print(f"{label:22s} n={m.sum():7,d}  RMSE={rmse:.3f}  r={r:+.3f}  "
          f"en el suelo de la rejilla: {at_floor:4.1f} %")

# fair comparison against step on common pixels
print("\nsobre los pixeles que el escalon tambien resuelve:")
for label, (_, _, _, _, mu) in results.items():
    common = inter & np.isfinite(bathy) & np.isfinite(mu) & np.isfinite(step)
    rows = pit.compare_dems({"HSR": mu, "step": step}, bathy, mask=common)
    h, s = rows[0], rows[1]
    print(f"  {label:22s} n={h['n']:7,d}  HSR {h['rmse_m']:.3f}  "
          f"vs escalon {s['rmse_m']:.3f}")
