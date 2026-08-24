import sys, os, time
PROJ=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import truststore; truststore.inject_into_ssl()
import openeo
from intertidal.benchmark import run_benchmark
t0=time.time()
print("conectando...", flush=True)
conn=openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
bbox={"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}  # pentagono Villaviciosa
conditions=[
    ("1yr",  ["2020-01-01","2020-12-31"]),
    ("3yr",  ["2018-01-01","2020-12-31"]),
    ("10yr", ["2016-01-01","2025-12-31"]),
]
res=run_benchmark(conn, bbox, conditions,
                  out_json=os.path.join(PROJ,"benchmark_results.json"),
                  outdir=os.path.join(PROJ,"bench"))
print(f"\nBENCHMARK DONE en {(time.time()-t0)/60:.1f} min, {len(res)} condiciones", flush=True)
