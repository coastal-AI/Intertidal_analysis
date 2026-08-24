"""GOT4.10 (harmonic, 55 km cells) vs CMEMS IBI (total sea level, 2.2 km)."""
import os, sys, json, ssl
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

_ORIG_CTX = ssl.SSLContext
import truststore
truststore.inject_into_ssl()
import botocore.httpsession as _bh
_orig = _bh.create_urllib3_context


def _safe(*a, **k):
    cur = ssl.SSLContext
    ssl.SSLContext = _ORIG_CTX
    try:
        return _orig(*a, **k)
    finally:
        ssl.SSLContext = cur


_bh.create_urllib3_context = _safe

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import copernicusmarine

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
got = json.load(open(os.path.join(SC, "tides.json")))     # GOT4.10, por fecha

# La epoca 2023-2025, que es la que ajustamos
dates = sorted(d for d in got if d >= "2023-01-01")
print(f"{len(dates)} fechas de la epoca con marea GOT4.10")

# Hora de paso real de Sentinel-2 sobre la ria (~11:00 UTC)
from pyintertidal.overpass import get_overpass_times
AOI = {"west": -5.455, "south": 43.470, "east": -5.375, "north": 43.548}
over = get_overpass_times(AOI, [min(dates), max(dates)])
have = [d for d in dates if d in over]
print(f"{len(have)} con hora de paso conocida")

times = pd.to_datetime([over[d] for d in have]).tz_localize(None)
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m",
    minimum_longitude=-5.55, maximum_longitude=-5.25,
    minimum_latitude=43.45, maximum_latitude=43.65,
    start_datetime=str(times.min().date()),
    end_datetime=str((times.max() + pd.Timedelta(days=1)).date()),
    variables=["zos"])

# celda valida mas cercana al centroide
lo, la = np.meshgrid(ds.longitude.values, ds.latitude.values)
first = ds["zos"].isel(time=0).values
d = np.hypot((la - 43.509) * 111.32, (lo + 5.416) * 111.32 * np.cos(np.radians(43.5)))
i = np.unravel_index(np.argmin(np.where(np.isfinite(first), d, np.inf)), d.shape)
print(f"celda IBI usada: ({la[i]:.3f}, {lo[i]:.3f}) a {d[i]:.1f} km del centroide")

series = ds["zos"].isel(latitude=i[0], longitude=i[1])
ibi = series.interp(time=times.values).values
g = np.array([got[x] for x in have])
ok = np.isfinite(ibi) & np.isfinite(g)
a, b = g[ok] - np.mean(g[ok]), ibi[ok] - np.mean(ibi[ok])

print(f"\nn = {ok.sum()} fechas comparadas")
print(f"  GOT4.10 (armonico, celda a 32 km): {a.min():+.2f} .. {a.max():+.2f} m")
print(f"  IBI (nivel total, celda a 6.8 km): {b.min():+.2f} .. {b.max():+.2f} m")
print(f"  correlacion  {np.corrcoef(a, b)[0, 1]:.4f}")
print(f"  RMS de la diferencia  {np.sqrt(np.mean((a - b) ** 2)) * 100:.1f} cm")
print(f"  diferencia mediana |.| {np.median(np.abs(a - b)) * 100:.1f} cm")
print(f"  p95 |.|                {np.percentile(np.abs(a - b), 95) * 100:.1f} cm")
np.savez(os.path.join(SC, "marea_cmp.npz"), dates=np.array(have)[ok],
         got=a, ibi=b)
