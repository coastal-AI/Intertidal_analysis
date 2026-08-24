import sys, os, time, json, tempfile
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import openeo, xarray as xr
t0=time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]",*a,flush=True)
# pentagono bbox (centro para escalar)
W,S,E,N=-5.4256,43.4977,-5.3755,43.5277
cx,cy=(W+E)/2,(S+N)/2; hw,hh=(E-W)/2,(N-S)/2
TE=["2020-01-01","2020-12-31"]  # 1 año fijo -> barato, aisla el efecto AREA
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
OUT=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\profile_download_area.json"
res={}
# punto FULL 1yr ya medido en el sweep anterior: 336.5 s, 415x346 px = 14.36 km2
res["1.00"]={"scale":1.0,"km2":14.36,"job_s":336.5,"n_dates":144,"note":"medido antes (full)"}
for sc in [0.5,0.75]:
    bbox={"west":cx-hw*sc,"south":cy-hh*sc,"east":cx+hw*sc,"north":cy+hh*sc}
    log(f"== scale {sc} bbox {bbox} ==")
    cube=conn.load_collection("SENTINEL2_L2A",spatial_extent=bbox,temporal_extent=TE,bands=["SCL"],max_cloud_cover=100)
    tmp=tempfile.mktemp(suffix=".nc")
    tj=time.time()
    job=cube.save_result(format="netCDF").create_job(title=f"prof_area_{sc}"); job.start_and_wait()
    job_s=time.time()-tj
    job.get_results().get_assets()[0].download(tmp)
    ds=xr.open_dataset(tmp); da=ds["SCL"]
    dims={k:v for k,v in zip(da.dims,da.shape)}
    h=dims.get("y"); w=dims.get("x"); nd=dims.get("t")
    ds.close()
    try: os.remove(tmp)
    except OSError: pass
    km2=w*h*100/1e6
    res[f"{sc:.2f}"]={"scale":sc,"km2":round(km2,2),"px":f"{h}x{w}","job_s":round(job_s,1),"n_dates":int(nd)}
    log(f"   scale {sc}: {km2:.2f} km2 ({h}x{w}), job {job_s:.1f}s, {nd} fechas")
    json.dump(res,open(OUT,"w"),indent=2)
log("AREA PROFILE DONE -> "+OUT)
print(json.dumps(res,indent=2))
