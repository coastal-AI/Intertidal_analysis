"""Gate M0 (plan v4): the baseline measures the structure, the simulator's
null reproduces the archive's marginals. Fails loudly.

Two halves, per the spec:

1. BASELINE — reads results/m0_baseline/result.json (produced by
   ``python -m experiments.m0_baseline``) and asserts, against configs/m0.yaml:
   pooled dev slope inside the historical CI AND within the self-consistency
   tolerance of the value fixed at adoption; enough blocks; the per-block
   SPREAD (the structure the plan exists to explain) above threshold.

2. SIMULATOR NULL — calibrates the simulator from the sealed store, plants
   the template's own elevations with all mechanisms OFF, refits with the
   standard estimator, and asserts the fitted marginals (a, b, sigma
   quartiles, usable-observation rate) match the archive within tolerance.

Run:  python -m tests.test_gate_m0
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

CFG = yaml.safe_load(open("configs/m0.yaml", encoding="utf-8"))


def test_baseline_structure():
    p = "results/m0_baseline/result.json"
    assert os.path.exists(p), "run first: python -m experiments.m0_baseline"
    r = json.load(open(p, encoding="utf-8"))
    cfg = CFG["baseline"]
    lo, hi = cfg["ci_historico"]
    s = r["pooled_dev_slope"]
    assert lo <= s <= hi, f"dev slope {s:.3f} outside the historical CI"
    ref = cfg["pooled_dev_slope_adopcion"]
    assert abs(s - ref) <= cfg["tolerancia_autoconsistencia"], \
        f"dev slope {s:.3f} drifted from the reference {ref:.3f}"
    assert len(r["blocks"]) >= cfg["min_bloques_con_8pts"]
    assert r["spread"] >= cfg["min_recorrido_pendientes"], \
        "the per-block STRUCTURE (the plan's target) does not appear"
    assert r["n_reserved_hidden"] > 0, \
        "the reserved set must exist and never be read"


def test_simulator_null_marginals():
    from pyintertidal.simulator import calibrate, simulate
    from pyintertidal.elevation import _fit_block

    cfg = CFG["simulador"]
    tide = np.load(CFG["datos"]["mareas"])["tide"]
    d = np.load(CFG["datos"]["store"], allow_pickle=True)
    dates = np.array([str(s) for s in d["dates"]])
    ep = np.array([int(s[:4]) >= 2023 for s in dates]) & np.isfinite(tide)
    t = tide[ep]

    tpl = calibrate(CFG["datos"]["store"], "data_v4/store/lamina_base.npz",
                    t, epoch_years=2023,
                    sd_level=cfg.get("sd_nivel_m", 0.13))
    assert tpl["n_template"] > 20000

    rng = np.random.default_rng(cfg["seed"])
    sub = rng.choice(tpl["n_template"], 4000, replace=False)
    z_true = tpl["z"][sub]
    # the REAL cloud mask of the sampled pixels: an iid mask is too kind
    # (see simulator.calibrate) and shrank fitted sigma by ~30 %
    C_real = (d["C"][ep][:, tpl["idx"][sub]] > 0)
    Y, C = simulate(tpl, z_true, t, seed=cfg["seed"], C_real=C_real,
                    template_rows=sub)                  # mechanisms OFF

    grid = np.linspace(t.min(), t.max(), 50)
    a, b, mu, sg, rm, N = _fit_block(Y.astype(np.float64),
                                     C.astype(np.float64), t, grid,
                                     (0.03, 0.06, 0.1, 0.15, 0.22, 0.32,
                                      0.45, 0.65))
    ok = (N >= 8) & (b > 0.15) & (b < 2.5) & (np.abs(a) < 2.5)
    tol = cfg["tolerancia_relativa"]

    for q in (25, 50, 75):
        s_ = np.percentile(b[ok], q)
        r_ = np.percentile(tpl["b"], q)
        assert abs(s_ - r_) <= tol * max(abs(r_), 0.05), \
            f"marginal b p{q}: sim {s_:.3f} vs archive {r_:.3f}"
    # sigma lives on an 8-value GRID, so sample quantiles are quantised and
    # teeter between neighbouring atoms (0.15 vs 0.22 is a 32 % jump with
    # nothing in between — the first gate version failed on exactly that).
    # The stable comparison is the DISTRIBUTION over the grid atoms: total
    # variation distance between the simulated refit and the archive.
    sg_grid = np.array([0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65])

    def atom_fracs(v):
        return np.array([(np.abs(v - g0) < 1e-9).mean() for g0 in sg_grid])

    tv = 0.5 * np.abs(atom_fracs(sg[ok]) - atom_fracs(tpl["sg"])).sum()
    tol_sg = cfg.get("tolerancia_tv_sigma", tol)
    assert tv <= tol_sg, f"sigma distribution: TV {tv:.3f} > {tol_sg}"
    # usable-observation rate
    rate_sim = float((C > 0).mean())
    assert abs(rate_sim - tpl["obs_frac"]) <= 0.02
    # and the planted truth comes back — IN THE INTERIOR of the tide window.
    # The estimator is measurably biased at the window edges (the sesgo.py
    # curve: under 3 cm of bias with >0.5 m of headroom, metres beyond the
    # edges), and with the level jitter the null now REPRODUCES that edge
    # clipping — full-range slope 0.870 at adoption, which is the simulator
    # behaving like the real thing, not a defect. The unbiasedness claim,
    # and hence this regression test, applies to the interior (the
    # 2026-08-18 master-null result: slope 1.019 on the surveyed range).
    inner = (z_true > t.min() + 0.5) & (z_true < t.max() - 0.5)
    g = ok & np.isfinite(mu) & inner
    sl = np.polyfit(z_true[g], mu[g], 1)[0]
    assert 0.93 <= sl <= 1.07, \
        f"the null in the INTERIOR returns slope {sl:.3f} != 1"


if __name__ == "__main__":
    test_baseline_structure()
    print("OK  gate M0.1: baseline structure reproduced (dev only)")
    test_simulator_null_marginals()
    print("OK  gate M0.2: the simulator null reproduces the marginals and "
          "returns slope ~1")
    print("GATE M0: GREEN")
