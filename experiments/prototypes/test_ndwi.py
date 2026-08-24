import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, xarray as xr, openeo, re
from intertidal.notebook_compat import (
    analyze_ndwi_cube_openeo, _open_ndwi_cube, _ndwi, _grid_from_dataset,
)
NC=r"C:\Users\Jorge\AppData\Local\Temp\ndwi_villaviciosa_1yr.nc"
bbox={"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

if not os.path.exists(NC):
    conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
    log("bajando B03/B08/SCL 2020 (10m)...")
    cube=conn.load_collection("SENTINEL2_L2A",spatial_extent=bbox,temporal_extent=["2020-01-01","2020-12-31"],bands=["B03","B08","SCL"],max_cloud_cover=100)
    cube=cube.resample_spatial(resolution=10, method="near")
    job=cube.save_result(format="netCDF").create_job(title="ndwi_test"); job.start_and_wait()
    job.get_results().get_assets()[0].download(NC)
    log("descargado -> "+NC)
else:
    log("cubo ya en disco")

# inspección
ds=xr.open_dataset(NC)
print("data_vars:", list(ds.data_vars), "| dims:", dict(ds.sizes))
tr,crs=_grid_from_dataset(ds)
print(f"resolución: {abs(tr.a)} m | crs {crs}")
ds.close()

# analyze NDWI (streaming) via cache
a=analyze_ndwi_cube_openeo(conn=None, bbox=None, time_extent=None, cache_nc=NC, ndwi_threshold=0.1, verbose=False, plot=False)
wf_s,_,_=a.water_frequency(save=False)
log(f"analyze NDWI: refmap {a.reference_map.shape} clases {np.unique(a.reference_map,return_counts=True)}")
log(f"  WF rango [{np.nanmin(wf_s):.3f},{np.nanmax(wf_s):.3f}] NaN {100*np.isnan(wf_s).mean():.1f}% | valid {len(a.valid_dates)}")

# referencia in-memory
ds,b03,b08,scl,td=_open_ndwi_cube(NC)
B03=np.asarray(b03.values); B08=np.asarray(b08.values); SCL=np.asarray(scl.values).astype(np.int16)
dates=[re.search(r'\d{4}-\d{2}-\d{2}',str(v)).group(0) for v in np.asarray(scl[td].values)]; ds.close()
ndwi=_ndwi(B03,B08); clear=np.isin(SCL,[4,5,6,12])
water=clear&np.isfinite(ndwi)&(ndwi>0.1)
vd=set(a.valid_dates); keep=[i for i,d in enumerate(dates) if d in vd]
wv=water[keep].sum(0).astype(np.float32); cv=clear[keep].sum(0).astype(np.float32)
safe=np.where(cv==0,1,cv); wf_m=(wv/safe).astype(np.float32); wf_m[cv<8]=np.nan

both=np.isfinite(wf_s)&np.isfinite(wf_m)
log(f"STREAMING vs in-memory NDWI: NaN iguales {bool(np.array_equal(np.isnan(wf_s),np.isnan(wf_m)))} | max|diff| {np.abs(wf_s[both]-wf_m[both]).max():.2e}")
print("TEST NDWI DONE")
