import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, rasterio, xarray as xr
from intertidal.notebook_compat import (
    _grid_from_dataset, build_reference_map_from_cube,
    compute_water_frequency_from_cube,
)
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"
t0=time.time()
ds=xr.open_dataset(NC)
da=ds["SCL"]; tdim="t" if "t" in da.dims else da.dims[0]
scl=np.asarray(da.transpose(tdim,...).values).astype(np.int16)
import re
dates=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[tdim].values)]
transform,crs=_grid_from_dataset(ds); ds.close()
print(f"[{time.time()-t0:.1f}s] cubo {scl.shape} crs {crs} origin {(transform.c,transform.f)} res {(transform.a,transform.e)}")

# ── grid vs GeoTIFFs del servidor ──
with rasterio.open("reference_map.tif") as s:
    rs=s.read(1); print("ref server grid origin",(s.transform.c,s.transform.f),"res",(s.transform.a,s.transform.e))
print("grid local == server:", (round(transform.c)==round(303890) and round(transform.f)==round(4822280) and transform.a==10 and transform.e==-10))

# ── reference map local ──
t1=time.time()
rl=build_reference_map_from_cube(scl,bad_classes=[3,8,9,10],bad_fraction_threshold=0.05,stable_threshold=0.95,transition_buffer_pixels=10)
print(f"[refmap {time.time()-t1:.1f}s] shape {rl.shape} clases {np.unique(rl,return_counts=True)}")

# ── water frequency local (all dates) vs water_frequency.tif del servidor ──
t2=time.time()
wf=compute_water_frequency_from_cube(scl,dates,valid_dates=None,water_class=(6,12),clear_classes=(4,5,6,12),min_obs=8)
print(f"[wf {time.time()-t2:.1f}s] shape {wf.shape} rango [{np.nanmin(wf):.3f},{np.nanmax(wf):.3f}] NaN {100*np.isnan(wf).mean():.1f}%")
with rasterio.open("water_frequency.tif") as s: ws=s.read(1)
# recortar a region comun (415 vs 416)
h=min(wf.shape[0],ws.shape[0]); w=min(wf.shape[1],ws.shape[1])
a=wf[:h,:w]; b=ws[:h,:w]
both=np.isfinite(a)&np.isfinite(b)
if both.sum():
    diff=np.abs(a[both]-b[both])
    from scipy.stats import pearsonr
    print(f"WF local-vs-server: pixeles validos comunes {both.sum()} | MAE {diff.mean():.4f} | max {diff.max():.4f} | corr {pearsonr(a[both],b[both])[0]:.4f}")
    print(f"  (nota: el WF del servidor uso valid_dates del filtro de nubes; este local usa TODAS las fechas, por eso no es identico)")
print(f"[TOTAL local {time.time()-t0:.1f}s] TEST DONE")
