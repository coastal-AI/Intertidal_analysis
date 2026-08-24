"""Fetch EOT20 so both methods can run on the same, better tide.

GOT4.10 is on a 0.5 degree grid: it has no ocean cell within 32 km of the
Ria de Villaviciosa, so every elevation we have used a tide brought in from
open water 32 km away. EOT20 is 1/8 degree — sixteen times finer — and free
of registration, which makes it the cheapest way to stop the tide being the
confound in the HSR-versus-DEA comparison.

Downloaded to a temporary name and only renamed once complete: a truncated
archive that looks finished has already cost us two cubes today.
"""
import os, sys, time, zipfile
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

import requests

URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"
DEST = "tide_models/EOT20.zip"
os.makedirs("tide_models", exist_ok=True)
tmp = DEST + ".part"

if not os.path.exists(DEST):
    with requests.get(URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        last = time.time()
        t0 = last
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 20:
                    last = time.time()
                    mb = done / 1e6
                    rate = mb / (last - t0)
                    eta = (total / 1e6 - mb) / max(rate, 1e-6) / 60
                    print(f"  {mb:7.0f}/{total/1e6:.0f} MB "
                          f"({100*done/max(total,1):4.1f} %) "
                          f"{rate:5.1f} MB/s  faltan {eta:4.1f} min",
                          flush=True)
    if total and abs(os.path.getsize(tmp) - total) > 1024:
        raise RuntimeError(f"descarga incompleta: {os.path.getsize(tmp)} "
                           f"de {total}")
    os.replace(tmp, DEST)
    print(f"descargado {os.path.getsize(DEST)/1e6:.0f} MB", flush=True)
else:
    print(f"ya estaba: {os.path.getsize(DEST)/1e6:.0f} MB", flush=True)

with zipfile.ZipFile(DEST) as z:
    names = z.namelist()
    print(f"{len(names)} entradas en el zip; muestra:", flush=True)
    for n in names[:6]:
        print("   ", n)
    ocean = [n for n in names if "ocean_tides" in n and n.endswith(".nc")]
    print(f"  {len(ocean)} ficheros de ocean_tides", flush=True)
    z.extractall("tide_models")

# pyTMD expects tide_models/EOT20/ocean_tides/<C>_ocean_eot20.nc
import glob
found = glob.glob("tide_models/**/ocean_tides/*_ocean_eot20.nc",
                  recursive=True)
print(f"\n{len(found)} constituyentes en disco")
if found:
    d = os.path.dirname(os.path.dirname(found[0]))
    print(f"  raiz detectada: {d}")
    if os.path.basename(d) != "EOT20":
        target = "tide_models/EOT20"
        if not os.path.exists(target):
            os.rename(d, target)
            print(f"  renombrado a {target}")
print("hecho", flush=True)
