import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np
from intertidal.notebook_compat import analyze_ndwi_cube_openeo
from intertidal.overpass import get_overpass_times
from intertidal.tidemodel import PyTMDTideModel
from intertidal.hsr import reconstruct_hsr
NC=r"C:\Users\Jorge\AppData\Local\Temp\ndwi_santona_1yr.nc"
bbox={"west":-3.50,"south":43.40,"east":-3.42,"north":43.46}
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

# 1) analysis desde cache (sin descargar)
a=analyze_ndwi_cube_openeo(conn=None,bbox=None,time_extent=None,ndwi_threshold=0.0,
                           cache_nc=NC,verbose=False,plot=False)
log(f"analysis: {len(a.dates)} fechas, valid {len(a.valid_dates)}")

# 2) mareas pyTMD reales en las horas de paso
ov=get_overpass_times(bbox,["2020-01-01","2020-12-31"])
lat,lon=43.43,-3.46  # centro marismas
tm=PyTMDTideModel(model_name="GOT4.10",directory="./tide_models",box_size=2.0,resolution=0.05)
dts=[ov[d] for d in a.valid_dates if d in ov]
dd=[d for d in a.valid_dates if d in ov]
hh=tm.get_tide_heights_batch(lat,lon,dts)
tide={d:h for d,h in zip(dd,hh) if h is not None and np.isfinite(h)}
th=np.array(list(tide.values()))
log(f"mareas pyTMD: {len(tide)} fechas | rango [{th.min():.2f},{th.max():.2f}] m")

# 3) HSR
r=reconstruct_hsr(a,tide,scale=4,sigma_floor=0.05)
log(f"HSR: {int(r.valid.sum())} px intermareales ajustados ({100*r.valid.mean():.1f}%)")
log(f"  mu    rango [{np.nanmin(r.mu):.2f},{np.nanmax(r.mu):.2f}] m")
log(f"  sigma (relieve subpixel) mediana {np.nanmedian(r.sigma)*100:.0f} cm | p90 {np.nanpercentile(r.sigma,90)*100:.0f} cm")
log(f"  incertidumbre cota: mediana {np.nanmedian(r.sigma_mu[r.valid])*100:.1f} cm")
log(f"  DEM fino {r.z_fine.shape} a 2.5 m")
r.save("bathymetry_hsr_santona")
log("guardado en bathymetry_hsr_santona/")
print("HSR SANTONA DONE")
