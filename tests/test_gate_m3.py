"""Gate M3: a planted boundary error lands in gamma-hat, not in T.

Reads results/m3_gate_sim/result.json (python -m experiments.m3_gate_sim)
and asserts against configs/m3.yaml: the mouth judge recovered the planted
(gamma, dt) within tolerance AND the interior operator stayed flat once the
boundary was corrected — the error did not leak into fictitious estuarine
physics.

Run:  python -m tests.test_gate_m3
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

CFG = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))


def test_m3_gate():
    p = "results/m3_gate_sim/result.json"
    assert os.path.exists(p), "ejecuta antes: python -m experiments.m3_gate_sim"
    r = json.load(open(p, encoding="utf-8"))
    assert r.get("version_puerta") == 3, "resultado antiguo: re-ejecuta"
    dt_pl = CFG["plantado"]["dt_m2_min"]
    assert r["plantado"]["dt_m2_min"] == dt_pl
    # v3: la fase del error de contorno se recupera; gamma no se corrige
    # (teorema afin) y su presencia plantada no debe falsear fase
    assert abs(r["recuperado"]["dt_min"] - dt_pl) \
        <= CFG["puerta"]["tol_dt_min"]
    assert r["recuperado"]["gamma"] == 0.0, "v3 no corrige ganancia"
    assert r["recuperado"]["adopted"], \
        "el juez OOS no adopto una correccion que era real"
    assert r["residuales"]["rms_tau_min"] \
        <= CFG["puerta"]["max_tau_residual_rms_min"], "fuga al operador"
    assert r["T_sin_corregir_fuga"]["max_min"] \
        >= CFG["puerta"]["min_fuga_sin_corregir_min"], \
        "la fuga no aparece sin corregir: la puerta no prueba nada"
    assert r["puerta"]["PASA"]
    assert os.path.exists("results/m3_gate_sim/figure.png")


if __name__ == "__main__":
    test_m3_gate()
    print("OK  puerta M3 (v3): la FASE del error de contorno acaba en dt-hat,"
          " no en T; la ganancia queda con el prior y no falsea fase")
    print("PUERTA M3: VERDE")
