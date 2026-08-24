import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, openeo
from intertidal.notebook_compat import analyze_ndwi_cube_openeo
bbox={"west":-5.46,"south":43.47,"east":-5.35,"north":43.55}  # Villaviciosa GRANDE
CACHE=os.path.join(PROJ,"ndwi_cube_villaviciosa_grande_10y.nc")
t0=time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]",*a,flush=True)
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
log("descargando B03/B08/SCL 2016-2025 AOI grande...")
a=analyze_ndwi_cube_openeo(conn=conn,bbox=bbox,time_extent=["2016-01-01","2025-12-31"],
                           ndwi_threshold=0.0,cache_nc=CACHE,verbose=False,plot=False)
log(f"OK: {len(a.dates)} fechas | grid {a.reference_map.shape} | valid {len(a.valid_dates)}")
u,c=np.unique(a.reference_map,return_counts=True)
for ui,ci in zip(u,c): log(f"  clase {ui}: {ci*100/1e6:.2f} km2")
log(f"cache: {CACHE} ({os.path.getsize(CACHE)/1e6:.0f} MB)")
print("VILLAVICIOSA 10Y DOWNLOAD DONE")
