import time, tempfile, sys, os
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
t0=time.time()
import truststore; truststore.inject_into_ssl()
import numpy as np, rasterio, xarray as xr, openeo
from intertidal.notebook_compat import build_reference_map_from_cube
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
bbox={"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}  # pentagono Villaviciosa
te=["2016-01-01","2025-12-31"]; bad=[3,8,9,10]
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
log("conectado; bajando cubo SCL (pentagono)...")
cube=conn.load_collection("SENTINEL2_L2A",spatial_extent=bbox,temporal_extent=te,bands=["SCL"],max_cloud_cover=100)
tmp=tempfile.mktemp(suffix=".nc")
job=cube.save_result(format="netCDF").create_job(title="scl_verify"); job.start_and_wait()
job.get_results().get_assets()[0].download(tmp)
log("descargado -> "+tmp)
ds=xr.open_dataset(tmp); var="SCL" if "SCL" in ds.data_vars else list(ds.data_vars)[0]
da=ds[var]; tdim="t" if "t" in da.dims else da.dims[0]
scl=np.asarray(da.transpose(tdim,...).values).astype(np.int16)
log(f"cubo {scl.shape}")
rl=build_reference_map_from_cube(scl,bad_classes=bad,bad_fraction_threshold=0.05,stable_threshold=0.95,transition_buffer_pixels=10)
with rasterio.open("reference_map.tif") as s: rs=s.read(1)
log(f"local {rl.shape} vs server {rs.shape}")
h=min(rl.shape[0],rs.shape[0]); w=min(rl.shape[1],rs.shape[1]); a=rl[:h,:w]; b=rs[:h,:w]
tl=(a==0); ts=(b==0)
log(f"TRANSICION IoU local-vs-server = {(tl&ts).sum()/max((tl|ts).sum(),1):.3f} | acuerdo clases {100*(a==b).mean():.1f}%")
log(f"transicion px: local {int(tl.sum())} vs server {int(ts.sum())}")
log("VERIFY2 DONE")
