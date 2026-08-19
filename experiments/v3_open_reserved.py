"""V3 — abrir el RTK reservado. SOLO LO EJECUTA EL HUMANO. UNA VEZ.

Este es el único módulo del repositorio autorizado a pasar
``allow_reserved=True`` (regla R1; el guard dinámico de rtk.py verifica el
nombre de este archivo y tests/test_r1_guard.py escanea que nadie más lo
haga). Correrlo consume el conjunto reservado para siempre: después de mirar,
ningún número sacado de él vuelve a ser una validación.

ACTA VIGENTE (rtk.py, 2026-08-18): la partición reservada actual fue abierta
varias veces ese día ANTES de adoptarse la spec v4, de modo que está QUEMADA
para afirmaciones de nivel publicación. Los veredictos V3 de esta campaña se
trasladan a los bloques de una campaña futura. Este script queda como la
herramienta correcta para ese día.

Uso (humano, deliberado):

    V3_CONFIRMO=SI python -m experiments.v3_open_reserved <producto.tif>

Sin la variable de entorno, imprime el acta y sale sin tocar nada.
"""
import json
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np


def main():
    if os.environ.get("V3_CONFIRMO") != "SI":
        print(__doc__)
        print("V3_CONFIRMO no es 'SI': no se abre nada. (Correcto por "
              "defecto.)")
        return

    import rasterio
    from pyintertidal import rtk

    product = sys.argv[1] if len(sys.argv) > 1 else rtk.DEFAULT_GRID
    _dev, res = rtk.load_rtk(allow_reserved=True)
    with rasterio.open(product) as s:
        arr = s.read(1)
    z = arr[res["row"], res["col"]]
    ok = np.isfinite(z)
    slope = float(np.polyfit(res["elev"][ok], z[ok], 1)[0])
    rmse = float(np.sqrt(np.mean((z[ok] - res["elev"][ok]) ** 2)))
    rec = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"),
           "producto": product, "n": int(ok.sum()),
           "pendiente": slope, "rmse": rmse,
           "nota": "apertura V3 registrada; el reservado queda consumido"}
    log = os.path.join("results", "v3_aperturas.jsonl")
    os.makedirs("results", exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
