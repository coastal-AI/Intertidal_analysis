import sys, os, time, json, threading
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, xarray as xr, re
from intertidal.notebook_compat import build_reference_map_from_cube, compute_water_frequency_from_cube
try:
    import psutil; P=psutil.Process(); HAVE=True
except Exception: HAVE=False
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"

def rss_mb(): return P.memory_info().rss/1e6 if HAVE else float("nan")

# ── Sampler de pico de RAM en un hilo ──
peak={"v":0.0}; stop={"s":False}
def sampler():
    while not stop["s"]:
        if HAVE: peak["v"]=max(peak["v"],rss_mb())
        time.sleep(0.02)

base=rss_mb()
ds=xr.open_dataset(NC); da=ds["SCL"]; td="t" if "t" in da.dims else da.dims[0]
scl=np.asarray(da.transpose(td,...).values).astype(np.int16)
dates=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[td].values)]
ds.close()
T,H,W=scl.shape; px=H*W; km2=px*100/1e6
cube_mb=scl.nbytes/1e6
after_load=rss_mb()
print(f"cubo {scl.shape} = {km2:.1f} km2, {T} fechas")
print(f"cube.nbytes = {cube_mb:.0f} MB (int16)")
print(f"RSS base {base:.0f} MB -> tras cargar cubo {after_load:.0f} MB (delta {after_load-base:.0f} MB)")

th=threading.Thread(target=sampler); th.start()
t=time.perf_counter()
rm=build_reference_map_from_cube(scl,bad_classes=[3,8,9,10],bad_fraction_threshold=0.05,stable_threshold=0.95,transition_buffer_pixels=10)
tmask=rm==0; nt=max(int(tmask.sum()),1)
for i in range(T):
    b=np.isin(scl[i],[3,8,9,10]); gf=b.sum()/b.size
    _=(gf*100) if gf<=0.05 else float(b[tmask].sum())/nt*100
wf=compute_water_frequency_from_cube(scl,dates,valid_dates=None,min_obs=8)
dt=time.perf_counter()-t
stop["s"]=True; th.join()
print(f"compute local (refmap+nubes+WF) {dt:.1f}s | PICO RSS {peak['v']:.0f} MB (delta pico {peak['v']-base:.0f} MB)")
mult=(peak['v']-base)/cube_mb if cube_mb else float('nan')
print(f"pico/cube = x{mult:.1f}  (cuánta RAM extra sobre el propio cubo)")

# ── Proyección RAM del cubo a km2 grandes (10 años ~ 1379 fechas) ──
DATES_10Y=1379
print("\n== PROYECCIÓN RAM del cubo en RAM (10 años, ~1379 fechas) ==")
print(f"{'km2':>7} | {'pixeles':>10} | {'cube RAM':>9} | {'pico est. x'+f'{mult:.1f}':>12}")
proj=[]
for A in [14,50,100,200,500,1000,2000]:
    p=int(A*1e4); cube=p*DATES_10Y*2/1e9  # GB
    peakest=cube*mult if not np.isnan(mult) else float('nan')
    proj.append({"km2":A,"pixels":p,"cube_gb":round(cube,2),"peak_est_gb":round(peakest,2)})
    print(f"{A:7d} | {p:10d} | {cube:6.2f} GB | {peakest:8.2f} GB")

out=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\scal_local.json"
json.dump({"measured":{"km2":round(km2,1),"n_dates":T,"cube_mb":round(cube_mb,1),
           "compute_s":round(dt,1),"peak_rss_mb":round(peak['v'],1),"base_mb":round(base,1),
           "peak_over_cube":round(mult,2)},"projection_10y":proj},open(out,"w"),indent=2)
print("\nguardado:",out)
