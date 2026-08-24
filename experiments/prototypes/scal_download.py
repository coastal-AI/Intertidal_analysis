import sys, os, time, json, tempfile, math
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import openeo, xarray as xr
t0=time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]",*a,flush=True)
cx,cy=-5.40055,43.5127  # centro Villaviciosa
TE=["2020-01-01","2020-12-31"]  # 1 año fijo -> aisla el efecto AREA
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
OUT=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\scal_download.json"
res={}
# puntos pequeños ya medidos hoy (area sweep 1yr): 3.6/8.1/14.4 km2 ~ 337 s
res["14.4"]={"km2":14.4,"job_s":336.5,"n_dates":144,"note":"medido antes","ok":True}
def bbox_for(area_km2):
    half=math.sqrt(area_km2)/2.0
    dlat=half/111.32; dlon=half/(111.32*math.cos(math.radians(cy)))
    return {"west":cx-dlon,"south":cy-dlat,"east":cx+dlon,"north":cy+dlat}
for A in [50,100,200,400]:
    bbox=bbox_for(A)
    log(f"== target {A} km2 ==")
    try:
        cube=conn.load_collection("SENTINEL2_L2A",spatial_extent=bbox,temporal_extent=TE,bands=["SCL"],max_cloud_cover=100)
        tmp=tempfile.mktemp(suffix=".nc"); tj=time.time()
        job=cube.save_result(format="netCDF").create_job(title=f"scal_{A}"); job.start_and_wait()
        job_s=time.time()-tj
        job.get_results().get_assets()[0].download(tmp)
        mb=os.path.getsize(tmp)/1e6
        ds=xr.open_dataset(tmp); da=ds["SCL"]; dims={k:v for k,v in zip(da.dims,da.shape)}
        h,w,nd=dims.get("y"),dims.get("x"),dims.get("t"); ds.close()
        try: os.remove(tmp)
        except OSError: pass
        km2=w*h*100/1e6
        res[f"{km2:.1f}"]={"km2":round(km2,1),"px":f"{h}x{w}","n_pixels":h*w,"job_s":round(job_s,1),
                           "nc_mb":round(mb,1),"n_dates":int(nd),"ok":True}
        log(f"   {A} km2 real {km2:.1f} ({h}x{w}) job {job_s:.1f}s nc {mb:.1f}MB {nd} fechas")
    except Exception as e:
        res[f"target_{A}"]={"km2_target":A,"ok":False,"error":str(e)[:300]}
        log(f"   {A} km2 FALLO: {str(e)[:200]}")
    json.dump(res,open(OUT,"w"),indent=2)
log("SCAL DOWNLOAD DONE -> "+OUT)
