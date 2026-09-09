"""Gate M2 (plan v4): identifiability of the interior tide, judged in
simulation with real sampling, real clouds and a planted truth.

Reads results/m2_gate_sim/result.json (produced by
``python -m experiments.m2_gate_sim``) and asserts, against configs/m2.yaml:

* the alias decision exists and froze S2 (results/m2_alias/result.json);
* v2 criteria (PHASE_LOG 2026-08-19; the v1 alpha-recovery demand
  contradicted the project's own affine theorem and no implementation can
  pass it): the NLL(alpha) profile with (z, sigma) profiled is FLAT — the
  degeneracy exposed, not papered over;
* the best new estimator beats the literature baseline (M2d) in phase RMSE.

If any of this fails the spec says STOP — the phase does not proceed on
vibes, and neither does this test suite.

Run:  python -m tests.test_gate_m2
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

CFG = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))


def test_alias_decision():
    p = "results/m2_alias/result.json"
    assert os.path.exists(p), "run first: python -m experiments.m2_alias"
    r = json.load(open(p, encoding="utf-8"))
    assert "S2" in r["clavados_al_prior_del_contorno"]
    assert not r["tabla"]["S2"]["estimable"]
    assert {"M2", "N2", "O1"} <= set(r["estimables"])


def test_identifiability_gate():
    p = "results/m2_gate_sim/result.json"
    assert os.path.exists(p), \
        "run first: python -m experiments.m2_gate_sim"
    r = json.load(open(p, encoding="utf-8"))

    assert r.get("version_puerta") == 2, "v1 gate result: re-run"

    # the planted truth is what the config promised
    a_lo, a_hi = CFG["plantado"]["alpha_boca_a_fondo"]
    assert abs(r["alpha_true"][0] - a_lo) < 0.06
    assert r["alpha_true"][-1] > a_hi - 0.1

    # amplitude: the affine degeneracy must be EXPOSED (flat NLL profile)
    rango = float(r["perfil_nll_alpha"]["recorrido"])
    assert rango < CFG["puerta"]["max_recorrido_nll_alpha"], \
        f"NLL(alpha) has curvature {rango:.4f}: unexplained non-affine channel"

    # phase: the best new estimator beats the literature baseline
    rmse = r["rmse_fase_min"]
    best = r["mejor_nuevo"]
    assert rmse[best] < rmse["m2d"], \
        f"the M2d bar ({rmse['m2d']:.1f} min) was not beaten " \
        f"(best new {best}: {rmse[best]:.1f} min)"

    # and the verdict recorded is the one the numbers imply
    assert r["puerta"]["PASA"] == (r["puerta"]["alpha_plano_ok"]
                                   and r["puerta"]["fase_ok"])
    assert r["puerta"]["PASA"], "the gate came out RED: check the report"

    # visual artifact exists (part of the no-black-box contract)
    assert os.path.exists("results/m2_gate_sim/figure.png")


if __name__ == "__main__":
    test_alias_decision()
    print("OK  gate M2.0: alias table froze S2 and kept M2/N2/O1")
    test_identifiability_gate()
    print("OK  gate M2.1 (v2): affine degeneracy exposed (flat NLL) and "
          "the M2d bar beaten in phase")
    print("GATE M2: GREEN")
