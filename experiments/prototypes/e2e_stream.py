import sys, os, time, threading
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, xarray as xr, re
from intertidal.notebook_compat import (
    analyze_scl_cube_openeo, build_reference_map_from_cube, compute_water_frequency_from_cube,
)
try: import psutil; P=psutil.Process(); HAVE=True
except Exception: HAVE=False
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"
def rss(): return P.memory_info().rss/1e6 if HAVE else float('nan')
pk={'v':0}; stop={'s':False}
def watch():
    while not stop['s']: pk['v']=max(pk['v'],rss()); time.sleep(0.02)
base=rss(); th=threading.Thread(target=watch); th.start()

# ── analyze refactorizado (STREAMING) via cache_nc, sin conn ──
t=time.perf_counter()
a=analyze_scl_cube_openeo(conn=None, bbox=None, time_extent=None, cache_nc=NC,
                          verbose=False, plot=False)
wf_s,_,_=a.water_frequency(save=False)
dt=time.perf_counter()-t
stop['s']=True; th.join()
print(f"analyze+WF STREAMING: {dt:.1f}s | pico RSS {pk['v']:.0f} MB (delta {pk['v']-base:.0f} MB)")
print(f"  reference_map {a.reference_map.shape} | valid {len(a.valid_dates)} | ref {len(a.reference_dates)}")
print(f"  nc conservado: {os.path.exists(a.nc_path)} | owns_nc={a.owns_nc} (cache usuario -> no borra)")

# ── referencia in-memory ──
ds=xr.open_dataset(NC); da=ds['SCL']; td='t' if 't' in da.dims else da.dims[0]
scl=np.asarray(da.transpose(td,'y','x').values).astype(np.int16)
dates=[re.search(r'\d{4}-\d{2}-\d{2}',str(v)).group(0) for v in np.asarray(da[td].values)]; ds.close()
rm_m=build_reference_map_from_cube(scl,bad_classes=[3,8,9,10],bad_fraction_threshold=0.05,stable_threshold=0.95,transition_buffer_pixels=10)
tmask=rm_m==0; nt=max(int(tmask.sum()),1)
stats_m={}
for i,d in enumerate(dates):
    b=np.isin(scl[i],[3,8,9,10]); gf=b.sum()/b.size
    stats_m[d]=round((gf*100) if gf<=0.05 else float(b[tmask].sum())/nt*100,4)
vd_m=sorted(d for d,p in stats_m.items() if p/100.0<=0.1)
wf_m=compute_water_frequency_from_cube(scl,dates,valid_dates=vd_m,min_obs=8)

print("\n== EXACTITUD analyze(streaming) vs in-memory ==")
print("reference_map idéntico:", bool(np.array_equal(a.reference_map,rm_m)))
print("valid_dates idénticas:", sorted(a.valid_dates)==vd_m)
both=np.isfinite(wf_s)&np.isfinite(wf_m)
print("WF NaN iguales:", bool(np.array_equal(np.isnan(wf_s),np.isnan(wf_m))),
      "| max|diff|:", f"{np.abs(wf_s[both]-wf_m[both]).max():.2e}")
