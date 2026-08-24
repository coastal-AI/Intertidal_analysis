import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, openeo
from intertidal.notebook_compat import analyze_ndwi_cube_openeo, _grid_from_dataset
NC=r"C:\Users\Jorge\AppData\Local\Temp\ndwi_santona_1yr.nc"
bbox={"west":-3.50,"south":43.40,"east":-3.42,"north":43.46}  # Marismas de Santona
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

conn=None
if not os.path.exists(NC):
    conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
    log("bajando B03/B08/SCL Santona 2020 (10m)...")

a=analyze_ndwi_cube_openeo(conn=conn, bbox=bbox, time_extent=["2020-01-01","2020-12-31"],
                           ndwi_threshold=0.0, cache_nc=NC, verbose=False, plot=False)
log(f"AOI Santona OK: {len(a.dates)} fechas | refmap {a.reference_map.shape} | crs {a.crs}")
u,c=np.unique(a.reference_map,return_counts=True)
km2={0:'transicion',1:'agua',2:'tierra'}
for ui,ci in zip(u,c): log(f"  clase {ui} ({km2.get(int(ui))}): {ci} px ({ci*100/1e6:.2f} km2)")
log(f"  valid dates (filtro nubes): {len(a.valid_dates)} / {len(a.dates)}")
wf,_,_=a.water_frequency(save=False)
log(f"  WF rango [{np.nanmin(wf):.2f},{np.nanmax(wf):.2f}] | NaN {100*np.isnan(wf).mean():.1f}%")
print("SANTONA AOI TEST DONE")
