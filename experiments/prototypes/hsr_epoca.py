# -*- coding: utf-8 -*-
"""Experimento de ÉPOCAS: HSR ajustado solo con 2022-2025 (mismo cubo 10y).
Hipótesis: menos cambio morfológico dentro de la época -> sigma se desatura ->
el DEM fino (2.5 m) pasa a batir al baseline contra el LiDAR reciente."""
import sys, os, time, json
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import numpy as np, rasterio
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr
from intertidal.notebook_compat import analyze_ndwi_cube_openeo
from intertidal.overpass import get_overpass_times
from intertidal.tidemodel import PyTMDTideModel
from intertidal.hsr import reconstruct_hsr
from intertidal.validation import reproject_to_grid
S=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]",*a,flush=True)

bbox={"west":-5.46,"south":43.47,"east":-5.35,"north":43.55}
CACHE=os.path.join(PROJ,"ndwi_cube_villaviciosa_grande_10y.nc")

# 1) analysis desde cache
a=analyze_ndwi_cube_openeo(conn=None,bbox=None,time_extent=None,ndwi_threshold=0.0,
                           cache_nc=CACHE,verbose=False,plot=False)
epoch=[d for d in a.valid_dates if d>="2022-01-01"]
log(f"época 2022-2025: {len(epoch)} fechas válidas (de {len(a.valid_dates)} en 10 años)")

# 2) mareas pyTMD en horas de paso (centroide del AOI grande)
ov=get_overpass_times(bbox,["2022-01-01","2025-12-31"])
tm=PyTMDTideModel(model_name="GOT4.10",directory="./tide_models",box_size=2.0,resolution=0.05)
dd=[d for d in epoch if d in ov]
hh=tm.get_tide_heights_batch(43.51,-5.405,[ov[d] for d in dd])
tide={d:h for d,h in zip(dd,hh) if h is not None and np.isfinite(h)}
th=np.array(list(tide.values()))
log(f"mareas: {len(tide)} fechas | rango [{th.min():.2f},{th.max():.2f}] m")

# 3) HSR con la época
r=reconstruct_hsr(a,tide,valid_dates=list(tide.keys()),scale=4,sigma_floor=0.05)
imask_arr=rasterio.open("intertidal_mask.tif").read(1).astype(bool)
r.clip_to(imask_arr)
bad=~np.isfinite(r.sigma_mu)|(r.sigma_mu>0.10)
r.mu[bad]=np.nan
valid=np.isfinite(r.mu)
sat=(r.sigma>=0.60)&valid
log(f"HSR época: {int(valid.sum())} px | sigma saturada: {100*sat.sum()/max(valid.sum(),1):.0f}% (10 años era 87%)")
log(f"  sigma mediana {np.nanmedian(r.sigma[valid])*100:.0f} cm | incert mediana {np.nanmedian(r.sigma_mu[valid])*100:.1f} cm")

# re-stage2 con mu limpio
from intertidal.hsr import hsr_stage2
zf=hsr_stage2(np.nan_to_num(np.where(valid,r.mu,np.nan),nan=0.0),
              np.where(valid,r.sigma,0.0), valid, scale=4, sigma_floor=0.05)
from intertidal.notebook_compat import _write_geotiff
_write_geotiff("bathymetry_hsr/hsr_mu_epoca.tif", np.where(valid,r.mu,np.nan), r.transform, r.crs, "float32", np.nan)
_write_geotiff("bathymetry_hsr/hsr_dem_fine_epoca.tif", zf, r.transform_fine, r.crs, "float32", np.nan)

# 4) validación LiDAR
def grid(p):
    with rasterio.open(p) as s: return s.transform, s.crs, s.shape
tf10,crs10,shp10=grid("bathymetry_hsr/hsr_mu_epoca.tif")
lid10=reproject_to_grid("mdt5_villaviciosa_grande.tif",tf10,crs10,shp10)
tf25,crs25,shp25=grid("bathymetry_hsr/hsr_dem_fine_epoca.tif")
lid25=reproject_to_grid("mdt5_villaviciosa_grande.tif",tf25,crs25,shp25)

def stats(pred,ref,mask,label):
    m=mask&np.isfinite(pred)&np.isfinite(ref); n=int(m.sum())
    if n<50: return {"metodo":label,"n":n}
    d=pred[m]-ref[m]; bias=float(np.median(d)); dd=d-bias
    return {"metodo":label,"n":n,"bias_m":round(bias,3),
            "rmse_debias_m":round(float(np.sqrt((dd**2).mean())),3),
            "mae_debias_m":round(float(np.abs(dd).mean()),3),
            "pearson_r":round(float(pearsonr(pred[m],ref[m])[0]),3),
            "spearman":round(float(spearmanr(pred[m],ref[m])[0]),3)}

mu_c=np.where(valid,r.mu,np.nan)
base25=zoom(np.nan_to_num(mu_c,nan=0.0),4,order=1)
bm=zoom(valid.astype(float),4,order=1)>0.99
base25=np.where(bm,base25,np.nan)
res=[stats(mu_c,lid10,valid,"HSR mu ÉPOCA (10 m)"),
     stats(zf,lid25,np.isfinite(zf),"HSR fino ÉPOCA (2.5 m)"),
     stats(base25,lid25,np.isfinite(zf),"baseline ÉPOCA (interp. mu)")]
print()
print(f"{'método':32s} {'n':>7s} {'bias':>7s} {'RMSE*':>6s} {'MAE*':>6s} {'r':>6s} {'rho':>6s}")
for x in res:
    print(f"{x['metodo']:32s} {x['n']:>7d} {x['bias_m']:>7.2f} {x['rmse_debias_m']:>6.2f} "
          f"{x['mae_debias_m']:>6.2f} {x['pearson_r']:>6.3f} {x['spearman']:>6.3f}")
json.dump(res,open(os.path.join(S,"hsr_epoca.json"),"w"),indent=2,ensure_ascii=False)
print("EPOCA DONE")
