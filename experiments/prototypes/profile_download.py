import sys, os, time, json, tempfile
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import openeo, xarray as xr, numpy as np
t0=time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]",*a,flush=True)
bbox={"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
OUT=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\profile_download.json"
res={}
# 10 años ya medido: 1379 fechas, 41.8 min = 2509 s (job). Lo añadimos como punto conocido.
res["2016-01-01..2025-12-31"]={"years":10,"job_s":2509,"n_dates":1379,"note":"medido antes"}
for label,te in [("1yr",["2020-01-01","2020-12-31"]),("3yr",["2018-01-01","2020-12-31"])]:
    log(f"== {label} {te} ==")
    cube=conn.load_collection("SENTINEL2_L2A",spatial_extent=bbox,temporal_extent=te,bands=["SCL"],max_cloud_cover=100)
    tmp=tempfile.mktemp(suffix=".nc")
    tj=time.time()
    job=cube.save_result(format="netCDF").create_job(title=f"prof_{label}"); job.start_and_wait()
    job_s=time.time()-tj
    job.get_results().get_assets()[0].download(tmp)
    ds=xr.open_dataset(tmp); da=ds["SCL"]; nd=da.shape[da.dims.index("t") if "t" in da.dims else 0]; ds.close()
    try: os.remove(tmp)
    except OSError: pass
    res[f"{te[0]}..{te[1]}"]={"years":{"1yr":1,"3yr":3}[label],"job_s":round(job_s,1),"n_dates":int(nd)}
    log(f"   {label}: job {job_s:.1f}s, {nd} fechas")
    json.dump(res,open(OUT,"w"),indent=2)
log("DOWNLOAD PROFILE DONE -> "+OUT)
print(json.dumps(res,indent=2))
