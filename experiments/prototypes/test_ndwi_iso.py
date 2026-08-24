import sys, os, time, math, re
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, xarray as xr, openeo
from intertidal.bathymetry import BathymetryReconstructor
NC=r"C:\Users\Jorge\AppData\Local\Temp\ndwi_villaviciosa_1yr.nc"
bbox={"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

# fechas del cubo + mareas SINTÉTICAS (solo para probar el UDF, no es ciencia real)
ds=xr.open_dataset(NC); scl=ds["SCL"]; td="t" if "t" in scl.dims else scl.dims[0]
dates=sorted({re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(scl[td].values)}); ds.close()
tide={d: round(2.0*math.sin(i*0.9)+0.6*math.sin(i*0.27),3) for i,d in enumerate(dates)}
log(f"{len(dates)} fechas | mareas sintéticas [{min(tide.values())},{max(tide.values())}] m")

conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
rec=BathymetryReconstructor(conn)
log("reconstruct_isolines NDWI (server-side, baja B03/B08/SCL + UDF)...")
res=rec.reconstruct_isolines(
    bbox=bbox, time_extent=["2020-01-01","2020-12-31"],
    tide_heights=tide, valid_dates=list(tide.keys()),
    out_dir="bench/ndwi_smoke_iso", force=True, water_source="ndwi",
)
el=res.elevation
log(f"DONE | elevation {el.shape} | rango [{np.nanmin(el):.2f},{np.nanmax(el):.2f}] m | "
    f"px reconstruidos {int(np.isfinite(el).sum()):,} ({100*np.isfinite(el).mean():.1f}%) | "
    f"conf media {np.nanmean(res.confidence):.3f}")
print("SMOKE NDWI BATHY OK")
