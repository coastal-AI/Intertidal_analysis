import sys, os
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import matplotlib; matplotlib.use("Agg")  # sin ventana
import numpy as np, xarray as xr, re
from intertidal.notebook_compat import (
    _grid_from_dataset, build_reference_map_from_cube, SclCubeAnalysis,
)
NC=r"C:\Users\Jorge\AppData\Local\Temp\tmpswzoon9o.nc"
ds=xr.open_dataset(NC); da=ds["SCL"]; tdim="t" if "t" in da.dims else da.dims[0]
scl=np.asarray(da.transpose(tdim,...).values).astype(np.int16)
dates=[re.search(r"\d{4}-\d{2}-\d{2}",str(v)).group(0) for v in np.asarray(da[tdim].values)]
transform,crs=_grid_from_dataset(ds); ds.close()

# Reproducir el bloque local de analyze (reference map + nubes) sin re-descargar
bad=[3,8,9,10]; rt=0.05; ct=0.1
rm=build_reference_map_from_cube(scl,bad_classes=bad,bad_fraction_threshold=rt,stable_threshold=0.95,transition_buffer_pixels=10)
tmask=rm==0; n_trans=max(int(tmask.sum()),1)
tstats={}
for i,d in enumerate(dates):
    b=np.isin(scl[i],bad); gf=float(b.sum())/max(b.size,1)
    tstats[d]=round((gf*100) if gf<=rt else float(b[tmask].sum())/n_trans*100,4)

a=SclCubeAnalysis(scl_stack=scl,dates=dates,transform=transform,crs=crs,
    reference_map=rm,transition_pct=tstats,bad_classes=bad,
    ref_bad_fraction_threshold=rt,transition_cloud_threshold=ct)
print("="*70)
print("REPORTE (plot=False para no volcar imagen aqui):")
print("="*70)
a.report(plot=False)
print("="*70)
print("DOWNSTREAM disponibles para celdas WF/intertidal:")
print("  valid_dates:", len(a.valid_dates), "| reference_dates:", len(a.reference_dates),
      "| newly_valid:", len(a.newly_valid))
print("  reference_map.shape:", a.reference_map.shape, "| crs:", a.crs)
