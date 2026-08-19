"""Gate M1 (plan v4): geometry invariants + the visual artefact exists.

Invariants, from configs/m1.yaml:
  * width(s, h) non-decreasing in h in >= the configured fraction of bands
    (physics: more water cannot wet less estuary);
  * the along-mouth coordinate agrees with the 2026-08-18 channel distance
    (different definitions, so consistency not identity is demanded);
  * hydraulically isolated pixels below the configured fraction;
  * the visual gate artefact (figure.png) was produced.

Run:  python -m tests.test_gate_m1
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

CFG = yaml.safe_load(open("configs/m1.yaml", encoding="utf-8"))


def test_gate_m1():
    p = "results/m1_geometry/result.json"
    assert os.path.exists(p), "ejecuta antes: python -m experiments.m1_geometry"
    r = json.load(open(p, encoding="utf-8"))
    g = CFG["puerta"]
    assert r["frac_bandas_monotonas"] >= g["min_frac_bandas_monotonas"], \
        f"ancho(s,h) decrece con h en demasiadas bandas: {r['frac_bandas_monotonas']:.2f}"
    assert r["corr_s_canal"] >= g["min_corr_s_canal"], \
        f"s_boca no es consistente con s_canal: r={r['corr_s_canal']:.3f}"
    frac_iso = r["n_isolated"] / 39951.0
    assert frac_iso <= g["max_aislados_frac"], \
        f"demasiados pixeles inalcanzables: {100*frac_iso:.1f} %"
    assert os.path.exists("results/m1_geometry/figure.png"), \
        "falta el artefacto visual de la puerta"
    z = np.load("results/m1_geometry/geometry.npz")
    st = z["s_mouth"][z["thalweg"]]
    st = st[np.isfinite(st)]
    assert st.max() > 3000, "el thalweg no penetra en la ria"


if __name__ == "__main__":
    test_gate_m1()
    print("PUERTA M1: VERDE")
