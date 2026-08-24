import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
from intertidal.pipeline import run_pipeline, validate_against_lidar
t0=time.time()
result = run_pipeline(
    connection=None,                       # cube ya cacheado
    site="Villaviciosa",
    bbox={"west":-5.46,"south":43.47,"east":-5.35,"north":43.55},
    time_extent=("2016-01-01","2025-12-31"),
    cache_nc="ndwi_cube_villaviciosa_grande_10y.nc",
    chunk=16,                              # convivir con el kernel del usuario
)
print(f"\n[pipeline en {(time.time()-t0)/60:.1f} min]")
result.save("products_villaviciosa")
print("saved -> products_villaviciosa/")
metrics = validate_against_lidar(result, mdt_path="mdt5_villaviciosa_grande.tif")
print("SDK SMOKE DONE")
