import sys, os, time, threading
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, xarray as xr, re
from scipy.ndimage import binary_dilation
from intertidal.notebook_compat import build_reference_map_from_cube, compute_water_frequency_from_cube
try: import psutil; P=psutil.Process(); HAVE=True
except Exception: HAVE=False
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"
BAD=[3,8,9,10]; RT=0.05; CT=0.1; STAB=0.95; BUF=10; WCL=[6,12]; CCL=[4,5,6,12]; MINOBS=8

def rss(): return P.memory_info().rss/1e6 if HAVE else float('nan')
def peakwatch():
    pk={'v':0.0}; stop={'s':False}
    def run():
        while not stop['s']:
            pk['v']=max(pk['v'],rss()); time.sleep(0.02)
    th=threading.Thread(target=run); th.start()
    return pk,stop,th

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING: 3 pasadas sobre el netCDF, acumulando arrays (H,W). Nunca carga (T,H,W).
# ─────────────────────────────────────────────────────────────────────────────
def analyze_streaming(nc_path, chunk=64):
    ds=xr.open_dataset(nc_path); da=ds["SCL"]
    td="t" if "t" in da.dims else da.dims[0]
    da=da.transpose(td,"y","x")
    T=da.sizes[td]; H=da.sizes["y"]; W=da.sizes["x"]
    dates=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[td].values)]
    # PASS 1: votos reference map (solo fechas limpias) + fracción global por fecha
    water=np.zeros((H,W),np.int32); land=np.zeros((H,W),np.int32)
    gfrac=np.zeros(T,np.float32)
    for a in range(0,T,chunk):
        blk=np.asarray(da.isel({td:slice(a,a+chunk)}).values).astype(np.int16)  # (k,H,W)
        bad=np.isin(blk,BAD)
        gf=bad.reshape(bad.shape[0],-1).mean(axis=1)
        gfrac[a:a+blk.shape[0]]=gf
        clean=gf<=RT
        if clean.any():
            cb=blk[clean]
            water+=(cb==6).sum(0).astype(np.int32)
            land+=np.isin(cb,[4,5]).sum(0).astype(np.int32)
    valid=water+land; valid[valid==0]=1
    ws=(water/valid)>=STAB; ls=(land/valid)>=STAB
    trans=~(ws|ls)
    if BUF>0: trans=binary_dilation(trans,iterations=BUF)
    refmap=np.zeros((H,W),np.uint8); refmap[ws]=1; refmap[ls]=2; refmap[trans]=0
    tmask=refmap==0; nt=max(int(tmask.sum()),1)
    # PASS 2: cloud stats (transición para fechas sucias; global para limpias)
    stats={}
    for a in range(0,T,chunk):
        gfc=gfrac[a:a+chunk]
        dirty=np.where(gfc>RT)[0]
        if len(dirty):
            blk=np.asarray(da.isel({td:slice(a,a+chunk)}).values).astype(np.int16)
        for j,gf in enumerate(gfc):
            date=dates[a+j]
            if gf<=RT: stats[date]=round(float(gf)*100,4)
            else:
                bad=np.isin(blk[j],BAD)
                stats[date]=round(float(bad[tmask].sum())/nt*100,4)
    valid_dates=sorted(d for d,p in stats.items() if p/100.0<=CT)
    # PASS 3: water frequency sobre valid_dates
    vd=set(valid_dates)
    wv=np.zeros((H,W),np.int32); cv=np.zeros((H,W),np.int32)
    for a in range(0,T,chunk):
        keep=[j for j in range(min(chunk,T-a)) if dates[a+j] in vd]
        if not keep: continue
        blk=np.asarray(da.isel({td:slice(a,a+chunk)}).values).astype(np.int16)
        sub=blk[keep]
        wv+=np.isin(sub,WCL).sum(0).astype(np.int32)
        cv+=np.isin(sub,CCL).sum(0).astype(np.int32)
    safe=np.where(cv==0,1,cv).astype(np.float32)
    wf=(wv/safe).astype(np.float32); wf[cv<max(1,MINOBS)]=np.nan
    ds.close()
    return refmap,stats,valid_dates,wf

# ── correr STREAMING con medición de pico ──
pk,stop,th=peakwatch(); base=rss()
t=time.perf_counter()
rm_s,stats_s,vd_s,wf_s=analyze_streaming(NC,chunk=64)
dt_s=time.perf_counter()-t
stop['s']=True; th.join()
print(f"STREAMING: {dt_s:.1f}s | pico RSS {pk['v']:.0f} MB (delta {pk['v']-base:.0f} MB) | valid {len(vd_s)}")

# ── in-memory de referencia (para comparar exactitud) ──
ds=xr.open_dataset(NC); da=ds["SCL"]; td="t" if "t" in da.dims else da.dims[0]
scl=np.asarray(da.transpose(td,"y","x").values).astype(np.int16)
dates=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[td].values)]; ds.close()
rm_m=build_reference_map_from_cube(scl,bad_classes=BAD,bad_fraction_threshold=RT,stable_threshold=STAB,transition_buffer_pixels=BUF)
tmask=rm_m==0; nt=max(int(tmask.sum()),1)
stats_m={}
for i,d in enumerate(dates):
    b=np.isin(scl[i],BAD); gf=b.sum()/b.size
    stats_m[d]=round((gf*100) if gf<=RT else float(b[tmask].sum())/nt*100,4)
vd_m=sorted(d for d,p in stats_m.items() if p/100.0<=CT)
wf_m=compute_water_frequency_from_cube(scl,dates,valid_dates=vd_m,water_class=WCL,clear_classes=CCL,min_obs=MINOBS)

# ── comparación exacta ──
print("\n== EXACTITUD streaming vs in-memory ==")
print("reference_map idéntico:", bool(np.array_equal(rm_s,rm_m)))
print("valid_dates idénticas:", vd_s==vd_m)
sd=[d for d in stats_s if abs(stats_s[d]-stats_m.get(d,-9))>1e-6]
print("cloud stats: fechas que difieren:", len(sd))
both=np.isfinite(wf_s)&np.isfinite(wf_m)
print(f"WF: NaN iguales: {bool(np.array_equal(np.isnan(wf_s),np.isnan(wf_m)))} | max|diff| {np.abs(wf_s[both]-wf_m[both]).max():.2e}")
