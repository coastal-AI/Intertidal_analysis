"""Collect every tide model reachable without credentials.

The ensemble the DEA paper uses wants nine models. Five of them (FES2014,
FES2022, TPXO8/9/10) sit behind registration, so the honest maximum we can
assemble unattended is whatever GSFC and the free archives hand over. Each
one we add is one more member of the ensemble and one less thing to
apologise for later.
"""
import os, sys, time

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

from pyintertidal.net import use_system_certificates
use_system_certificates()

from pyTMD.datasets import fetch_gsfc_got

for model in ("GOT5.6", "GOT5.5", "GOT4.8"):
    t0 = time.time()
    try:
        fetch_gsfc_got(model=model, directory="tide_models",
                       format="netcdf", compressed=False)
        print(f"  {model:8s} OK ({(time.time()-t0)/60:.1f} min)", flush=True)
    except Exception as e:
        print(f"  {model:8s} FALLA {type(e).__name__}: {str(e)[:120]}",
              flush=True)

from eo_tides.utils import list_models
have, _ = list_models(directory="tide_models", show_available=False,
                      show_supported=False)
print(f"\ndisponibles ahora: {have}", flush=True)
