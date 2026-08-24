"""Is the tide binning throwing away the resolution the method lives on?

The raw archive samples the tide with median gaps of 0.4 cm. Binning to 40
levels quantises that axis to 7 cm — a factor of 17 coarser than the data.
If the band is the price of that, more bins (or none) should dissolve it.
"""
import os, sys, json, time
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

print(f"{'configuracion':22s} {'min':>5s} {'n':>9s} {'RMSE':>7s} {'r':>7s} "
      f"{'suelo sigma':>12s}")
print("-" * 68)
for label, bins in [("40 bins (actual)", 40), ("150 bins", 150),
                    ("sin binning", None)]:
    t0 = time.time()
    s1 = hsr_stage1(CACHE, cube.dates, tides, valid_dates=ep["dates"],
                    tide_bins=bins)
    valid = (s1["valid"] & inter & np.isfinite(s1["sigma_mu"])
             & (s1["sigma_mu"] <= 0.10))
    mu = np.where(valid, s1["mu"], np.nan).astype("float32")
    mu = terrain.drop_small_regions(mu, min_region_px=25)
    m = np.isfinite(mu) & np.isfinite(bathy) & inter
    b = np.median(mu[m] - bathy[m])
    rmse = np.sqrt(np.mean((mu[m] - b - bathy[m]) ** 2))
    r = np.corrcoef(mu[m], bathy[m])[0, 1]
    floor = 100 * np.mean(s1["sigma"][valid] <= 0.031)
    print(f"{label:22s} {(time.time()-t0)/60:5.1f} {m.sum():9,d} "
          f"{rmse:7.3f} {r:+7.3f} {floor:11.1f} %", flush=True)

    common = inter & np.isfinite(bathy) & np.isfinite(mu) & np.isfinite(step)
    rows = pit.compare_dems({"HSR": mu, "step": step}, bathy, mask=common)
    print(f"{'':22s} sobre los {rows[0]['n']:,} comunes: HSR "
          f"{rows[0]['rmse_m']:.3f} vs escalon {rows[1]['rmse_m']:.3f}",
          flush=True)
    np.save(f"products_tejo/mu_bins_{bins}.npy", mu)
