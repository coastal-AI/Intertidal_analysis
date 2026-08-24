import sys, os, time, json
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, xarray as xr, re
from intertidal.notebook_compat import (
    build_reference_map_from_cube, compute_water_frequency_from_cube,
)
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"
ds=xr.open_dataset(NC); da=ds["SCL"]; td="t" if "t" in da.dims else da.dims[0]
scl_full=np.asarray(da.transpose(td,...).values).astype(np.int16)
dates_full=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[td].values)]
ds.close()
BAD=[3,8,9,10]; RES=10.0
T,H,W=scl_full.shape
print(f"cubo completo: {scl_full.shape}  ({W*H*RES*RES/1e6:.1f} km2, {T} fechas)")

def timeit(fn,rep=3):
    best=1e9
    for _ in range(rep):
        t0=time.perf_counter(); fn(); best=min(best,time.perf_counter()-t0)
    return best

def full_local(scl,dates):
    rm=build_reference_map_from_cube(scl,bad_classes=BAD,bad_fraction_threshold=0.05,stable_threshold=0.95,transition_buffer_pixels=10)
    tmask=rm==0; nt=max(int(tmask.sum()),1)
    for i in range(scl.shape[0]):
        b=np.isin(scl[i],BAD); gf=b.sum()/b.size
        _=(gf*100) if gf<=0.05 else float(b[tmask].sum())/nt*100
    wf=compute_water_frequency_from_cube(scl,dates,valid_dates=None,min_obs=8)
    return rm,wf

# ── Escalado por nº de fechas (área fija = total) ──
print("\n== LOCAL vs nº fechas (área completa) ==")
by_dates=[]
for k in [100,300,600,1000,T]:
    scl=scl_full[:k]; dts=dates_full[:k]
    dt=timeit(lambda: full_local(scl,dts),rep=2)
    by_dates.append((k,dt)); print(f"  {k:5d} fechas -> {dt*1000:7.1f} ms")

# ── Escalado por área/km² (fechas fijas = total) ──
print("\n== LOCAL vs área/km2 (todas las fechas) ==")
by_area=[]
for frac in [0.25,0.5,0.75,1.0]:
    h=max(2,int(H*frac)); w=max(2,int(W*frac))
    scl=scl_full[:,:h,:w]
    km2=w*h*RES*RES/1e6
    dt=timeit(lambda: full_local(scl,dates_full),rep=2)
    by_area.append((km2,h*w,dt)); print(f"  {km2:5.1f} km2 ({h}x{w}) -> {dt*1000:7.1f} ms")

json.dump({"by_dates":by_dates,"by_area":by_area,"shape":[T,H,W],"res":RES},
          open(os.path.join(os.path.dirname(NC.replace("tmpswzoon9o.nc","")),"profile_local.json"),"w"))
# guardar en scratchpad
out=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad\profile_local.json"
json.dump({"by_dates":by_dates,"by_area":by_area,"shape":[T,H,W],"res":RES}, open(out,"w"))
print("\nguardado:",out)
print("LOCAL PROFILE DONE")
