"""Which tide source actually has data OVER the ria?"""
import os, sys
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import ssl
_ORIG_CTX = ssl.SSLContext            # antes de que truststore lo sustituya

import truststore
truststore.inject_into_ssl()

# botocore construye su contexto con `context.options |= ...`; con la clase
# de truststore en medio ese setter recursiona hasta agotar la pila. Le
# devolvemos la clase original solo dentro de esa funcion.
import botocore.httpsession as _bh
_orig_ctx_fn = _bh.create_urllib3_context


def _safe_ctx(*a, **k):
    cur = ssl.SSLContext
    ssl.SSLContext = _ORIG_CTX
    try:
        return _orig_ctx_fn(*a, **k)
    finally:
        ssl.SSLContext = cur


_bh.create_urllib3_context = _safe_ctx
import copernicusmarine
print("parche botocore aplicado")

AOI = dict(w=-5.455, s=43.470, e=-5.375, n=43.548)
DS = "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m"

ds = copernicusmarine.open_dataset(
    dataset_id=DS,
    minimum_longitude=AOI["w"] - 0.15, maximum_longitude=AOI["e"] + 0.15,
    minimum_latitude=AOI["s"] - 0.15, maximum_latitude=AOI["n"] + 0.15,
    start_datetime="2023-06-01", end_datetime="2023-06-03",
    variables=["zos"])
print("malla IBI:", dict(ds.sizes))
lon, lat = ds.longitude.values, ds.latitude.values
print(f"  paso {abs(lon[1]-lon[0]):.4f} deg lon = "
      f"{abs(lon[1]-lon[0])*111.32*np.cos(np.radians(43.5)):.2f} km")

first = ds["zos"].isel(time=0).values
LO, LA = np.meshgrid(lon, lat)
valid = np.isfinite(first)
print(f"  celdas con dato en la caja: {valid.sum()}/{valid.size}")

inside = ((LO >= AOI["w"]) & (LO <= AOI["e"])
          & (LA >= AOI["s"]) & (LA <= AOI["n"]))
print(f"\nCELDAS DENTRO DEL AOI: {inside.sum()}  "
      f"de las cuales con dato: {(inside & valid).sum()}")

if valid.any():
    clat, clon = 43.509, -5.416          # centroide del AOI
    d = np.hypot((LA - clat) * 111.32,
                 (LO - clon) * 111.32 * np.cos(np.radians(43.5)))
    d_valid = np.where(valid, d, np.inf)
    i = np.unravel_index(np.argmin(d_valid), d.shape)
    print(f"celda valida mas cercana al centroide: "
          f"({LA[i]:.3f}, {LO[i]:.3f}) a {d[i]:.1f} km")
    print(f"  (con GOT4.10 la mas cercana estaba a 32.3 km)")

    # cuanto varia el nivel entre celdas dentro/junto al AOI?
    near = valid & (d < 12)
    print(f"\ncon {near.sum()} celdas validas a menos de 12 km:")
    z = ds["zos"].values                              # (t, y, x)
    zz = z[:, near]
    sp = np.nanmax(zz, axis=1) - np.nanmin(zz, axis=1)
    print(f"  diferencia de nivel entre ellas, mismo instante: "
          f"mediana {np.nanmedian(sp)*100:.1f} cm, max {np.nanmax(sp)*100:.1f} cm")
    print(f"  rango del nivel en las 48 h: "
          f"{np.nanmin(zz):+.2f} .. {np.nanmax(zz):+.2f} m")
