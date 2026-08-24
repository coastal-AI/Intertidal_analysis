"""What are the pixels in the horizontal band, and why do they all get 1.45 m?"""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
import pyintertidal as pit
from pyintertidal.elevation import hsr_stage1, epochs
from pyintertidal import SentinelCube

CACHE = "ndwi_cube_tejo_2019_2021.nc"
aoi = pit.sites.get("tejo")
cube = SentinelCube(aoi, ("2019-01-01", "2021-12-31"), water="ndwi",
                    cache_path=CACHE, resolution=10)
tides = json.load(open("products_tejo/tides.json"))
ep = epochs(sorted(tides), 3)[-1]

print("re-ajustando y guardando TODOS los parametros...", flush=True)
s1 = hsr_stage1(CACHE, cube.dates, tides, valid_dates=ep["dates"],
                tide_bins=40)
np.savez_compressed("products_tejo/stage1.npz",
                    **{k: s1[k] for k in ("a", "b", "mu", "sigma", "rmse",
                                          "n_obs", "sigma_mu", "valid")})
tmin, tmax = s1["tide_range"]
print(f"rango de marea: {tmin:+.2f} .. {tmax:+.2f} m, centro {(tmin+tmax)/2:+.3f}")

mu_grid = np.linspace(tmin, tmax, 60)
print(f"rejilla de mu: paso {mu_grid[1]-mu_grid[0]:.4f} m, "
      f"valor central {mu_grid[29]:+.4f} / {mu_grid[30]:+.4f}")

def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs, s.shape

hsr, TR, CRS, SH = load("products_tejo/hsr_elevation.tif")
inter = load("products_tejo/intertidal_mask.tif")[0] > 0
bathy = pit.reproject_to_grid("products_tejo/bathy_tagus_native.tif",
                              TR, CRS, SH)
m = inter & np.isfinite(bathy) & np.isfinite(hsr)
bias = np.median(hsr[m] - bathy[m])
print(f"\nsesgo de datum: {bias:+.3f} m")

# The band sits at h2 = hsr - bias ~= 1.45, i.e. raw mu ~= 1.45 + bias
band_raw = 1.45 + bias
print(f"la banda esta en mu crudo ~= {band_raw:+.3f} m")

mu = s1["mu"]
inband = m & (np.abs(mu - band_raw) < 0.05)
outband = m & (np.abs(mu - band_raw) >= 0.15)
print(f"\npixeles en la banda: {inband.sum():,} "
      f"({100*inband.sum()/m.sum():.1f} % de los validados)")

print(f"\n{'':22s} {'EN la banda':>14s} {'fuera':>14s}")
for name, arr in [("contraste b", s1["b"]), ("sigma", s1["sigma"]),
                  ("rmse del ajuste", s1["rmse"]),
                  ("n observaciones", s1["n_obs"]),
                  ("incertidumbre (cm)", s1["sigma_mu"] * 100)]:
    a1 = np.nanmedian(arr[inband]); a2 = np.nanmedian(arr[outband])
    print(f"  {name:20s} {a1:14.3f} {a2:14.3f}")

# how far is mu from the grid centre, for band pixels?
centre = 0.5 * (tmin + tmax)
print(f"\ncentro del rango de marea: {centre:+.3f} m")
print(f"  mediana de mu en la banda: {np.median(mu[inband]):+.3f} m")
print(f"  |mu - centro| en la banda: {np.median(np.abs(mu[inband]-centre)):.3f} m")
print(f"  |mu - centro| fuera:       {np.median(np.abs(mu[outband]-centre)):.3f} m")

# error against the reference
for nm, sel in (("banda", inband), ("fuera", outband)):
    d = (hsr[sel] - bias) - bathy[sel]
    print(f"\n{nm}: RMSE {np.sqrt(np.mean(d**2)):.3f} m, "
          f"r {np.corrcoef(hsr[sel], bathy[sel])[0,1]:+.3f}")
