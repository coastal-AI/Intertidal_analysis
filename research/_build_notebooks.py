"""SUPERSEDED — do not run.

This script generated the FIRST versions of the research notebooks. The
committed notebooks have since been extended by hand (investigation
narrative, mathematics, concept boxes, notebook 08); running this would
overwrite them with the old thin versions. Kept for the record only.
"""
raise SystemExit("research/_build_notebooks.py is superseded: the committed "
                 "notebooks are newer than what this script generates. "
                 "See the module docstring.")

"""Builds the research notebooks from the stored experiment results.

Each notebook reproduces one line of reasoning developed during the MAREA
project, reading ONLY the sealed/stored artifacts under results/ — nothing
heavy is recomputed, so the whole folder executes in seconds and the
outputs shown are exactly the numbers the conclusions were drawn from.

Run:  python research/_build_notebooks.py     (writes + executes all)
"""
import glob
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))

PRELUDE = """\
import json, os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(globals().get("_ROOT", ".."))   # repo root
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)
"""


def nb(title, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                       else nbf.v4.new_code_cell(src))
    return n


NOTEBOOKS = {}

# ── 01 · interior tide: what is measurable ───────────────────────────────
NOTEBOOKS["01_interior_tide_identifiability"] = nb(
    "Interior tide from wet/dry imagery: what is measurable, and what is not",
    [
     ("md", """\
**The claim.** Every published intertidal-DEM method assumes one uniform
water level per scene, taken from an ocean model at the mouth. MAREA instead
*measures* the interior tide from the archive itself: each pixel is a binary
threshold sensor, and thousands of them jointly date the tide per along-mouth
band. This notebook reproduces the three results that establish what such an
estimator can and cannot do."""),
     ("code", PRELUDE),
     ("md", """\
## 1. The alias table — what a sun-synchronous archive can even see
S2's constituent period divides the solar day: it is frozen forever. K1/P1
alias onto the annual cycle. The estimable set (M2, N2, O1, Q1, M4, MN4, MS4,
M6) is arithmetic, not opinion."""),
     ("code", """\
r = load("results/m2_alias/result.json")
print(f"{r['n_escenas']} real overpasses, mean hour {r['hora_media_utc']:.2f} UTC")
for k, v in sorted(r["tabla"].items(), key=lambda kv: kv[1]["period_h"]):
    al = v["alias_days"]
    tag = "ESTIMABLE" if v["estimable"] else "pinned to boundary prior"
    al_s = "inf" if not np.isfinite(al) else f"{al:6.1f} d"
    print(f"  {k:>3}: alias {al_s} - {tag}")"""),
     ("md", """\
## 2. The identifiability gate (simulation with planted truth)
A lag profile 0-40 min and gain 1.0-1.3 were planted with the REAL temporal
sampling, real cloud masks, calibrated noise and the measured scene-level
tide error. Verdicts: **phase is recoverable** (MAREA's likelihood engine,
4.7 min RMSE, beats the literature baseline M2d at 5.4) and **gain is not**
— the Bernoulli likelihood is exactly invariant under
(alpha, z, sigma) -> (c alpha, c z, c sigma), and the NLL(alpha) profile
with properly scaled grids is flat to machine precision. The affine theorem,
made executable."""),
     ("code", """\
r = load("results/m2_gate_sim/result.json")
print("phase RMSE vs planted truth (min):",
      {k: round(v, 2) for k, v in r["rmse_fase_min"].items()})
p = r["perfil_nll_alpha"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(r["centros_km"], r["tau_true_min"], "k-", lw=2, label="planted")
for k, c in (("m2a", "C0"), ("m2d", "C3")):
    ax[0].plot(r["centros_km"], r["estimadores"][k]["tau"], "o-", color=c,
               label=f"{k} ({r['rmse_fase_min'][k]:.1f} min)")
ax[0].set_xlabel("s from mouth (km)"); ax[0].set_ylabel("lag (min)")
ax[0].legend(); ax[0].set_title("phase: recovered")
ax[1].plot(p["malla"], p["nll"], "o-")
ax[1].set_xlabel("imposed gain alpha"); ax[1].set_ylabel("profiled NLL")
ax[1].set_title(f"gain: NLL(alpha) flat (range {p['recorrido']:.2e})")
plt.tight_layout()"""),
     ("md", """\
## 3. The real archive, judged against a matched null
The same four estimators on real Villaviciosa, with a null band from 5
uniform-tide re-simulations of the very same pixels and clouds. Mid-estuary
lags sit INSIDE the null; only the head (7.7 km) leaves it: **+26 min, with
3 of 4 estimators agreeing**. The operator carries exactly that and nothing
else — it never invents signal the null can explain."""),
     ("code", """\
r = load("results/m2_real/result.json")
for k, v in r["veredicto"].items():
    print(f"{k:8s} real={np.round(v['real'],1)}")
    print(f"         outside null: {v['fuera_del_nulo']}")"""),
    ])

# ── 02 · boundary audit ──────────────────────────────────────────────────
NOTEBOOKS["02_boundary_audit"] = nb(
    "Auditing the ocean model at the mouth — before it fakes estuarine physics",
    [
     ("md", """\
**The danger this gate exists for**: a phase error of the boundary model
masquerades as distributed "estuarine transfer". We planted exactly that
(+12 min on M2, +8 % gain, NO interior transfer) and demanded the machinery
sort it out. Findings: the phase error is recoverable from the mouth pixels
(dt-hat within 4 min, adopted by an out-of-sample judge); the planted GAIN
is NOT auditable from binary data (affine theorem again — it lands with the
ensemble prior); and the demonstrated leak (up to +21 min of fictitious
transfer without correction) disappears once the phase is corrected."""),
     ("code", PRELUDE),
     ("code", """\
r = load("results/m3_gate_sim/result.json")
print("planted:", r["plantado"])
print("recovered:", {k: r["recuperado"][k] for k in ("dt_min", "adopted")})
print("leak without correction (min):",
      np.round(r["T_sin_corregir_fuga"]["tau"], 1))
print("residual with correction (min):",
      np.round(r["T_corregido"]["tau"], 1),
      f"-> RMS {r['residuales']['rms_tau_min']:.1f}")
print("gate:", r["puerta"])"""),
    ])

# ── 03 · operator vs gauges ──────────────────────────────────────────────
NOTEBOOKS["03_operator_and_gauges"] = nb(
    "The distributed tide gauge, scored against real gauges",
    [
     ("md", """\
Three independent scoreboards. (1) The contraction gate: applied to a
phase-corrupted synthetic world, the operator recovers the planted damage in
elevation and is a fixed point of its own estimation. (2) Terneuzen (a gauge
never used in calibration): MAREA's zero-instrument correction beats every
ocean model, their ensemble, and the earlier v3 adapter, capturing 92 % of
what any pure lag can give. (3) The Scheldt axis: the blind imagery gradient
(0.72 min/km) matches the gauge-measured one (0.9)."""),
     ("code", PRELUDE),
     ("code", """\
r = load("results/m4_gate_sim/result.json")
print("elevation RMSE per band, uniform -> operator:")
for c, u, o in zip(r["centros_km"], r["rmse_z"]["uniforme"],
                   r["rmse_z"]["operador"]):
    print(f"  s={c:5.2f} km: {u:.3f} -> {o:.3f}")
print("damage recovered:", round(r["fraccion_dano_recuperado"], 2),
      "| gate:", r["puerta"])"""),
     ("code", """\
r = load("results/p6_comparativa/result.json")
for k, v in sorted(r["rmse_m"].items(), key=lambda kv: -kv[1]):
    print(f"  {v:.4f} m  {k}")
print("gauge-calibrated reference (needs 2 gauges):",
      r["referencia_con_mareografos"])"""),
     ("code", """\
r = load("results/p4_scheldt/result.json")
print(f"Scheldt blind gradient: {r['gradiente_m2a_min_km']:.2f} min/km "
      f"vs gauges {r['gradiente_mareografos_min_km']} "
      f"-> {r['validacion']}")
t = load("results/p7_gauge_table/result.json")["tabla"]
print("\\nlevel RMSE at every studied gauge (models as-is):")
for k, v in t.items():
    if "rmse_m" in v:
        m = v["rmse_m"]
        print(f"  {k:46s} EOT20 {m['EOT20']:.3f} | ens {m['ensemble_media3']:.3f}")"""),
    ])

# ── 04 · bathymetry validation ───────────────────────────────────────────
NOTEBOOKS["04_bathymetry_validation"] = nb(
    "Elevation against external truth: where MAREA wins, ties, and declines",
    [
     ("md", """\
The dominance case. Against EMODnet surveys of the flats themselves:
**Tagus** (interior estuary, ~45 min lag detected blind) improves −11 % of
RMSE over the uniform-level assumption every other method makes; **Aveiro**
(branched coastal lagoon, lags 41→56 min) improves +24..+53 mm exactly in
the bands where the lag grows, while the truth-broken inlet band ties;
**Vadehavet** (open Wadden flats) measures an apparent lag, finds it does
not help, and correctly declines to correct. Against our own RTK
(Villaviciosa, short deep ria) the method ties the best of the family with
zero labels — and the point-vs-pixel decomposition shows ~0.21 m of the
observed error is the yardstick, not the method."""),
     ("code", PRELUDE),
     ("code", """\
for site in ("tejo", "vadehavet", "aveiro"):
    r = load(f"results/p5_{site}/result.json")
    res = r["contra_levantamiento"]
    uni = res.get("uniforme") or res["todos"]["uniforme"]
    op = res.get("operador") or res["todos"]["operador"]
    print(f"{site:10s} tau applied {np.round(r['tau_aplicado_min'],0)}")
    print(f"           uniform  slope {uni['pendiente']:.3f} RMSE {uni['rmse_centrado']:.3f}")
    print(f"           operator slope {op['pendiente']:.3f} RMSE {op['rmse_centrado']:.3f}")"""),
     ("code", """\
# Aveiro per-band anatomy: improvement grows with the detected lag
Z = np.load("results/p5_aveiro/z_scores.npz")
zu, zo, zr, s = Z["z_uni"], Z["z_op"], Z["z_ref"], Z["s_km"]
r = load("results/p5_aveiro/result.json")
ok = np.isfinite(zu) & np.isfinite(zo) & np.isfinite(zr) & np.isfinite(s)
qs = np.nanquantile(s[ok], np.linspace(0, 1, 7)); qs[0] -= 1e-9
rows = []
for k, (a, b) in enumerate(zip(qs, qs[1:])):
    m = ok & (s > a) & (s <= b)
    eu, eo = zu[m]-zr[m], zo[m]-zr[m]
    ru = np.sqrt(np.mean((eu-np.median(eu))**2))
    ro = np.sqrt(np.mean((eo-np.median(eo))**2))
    rows.append((r["centros_km"][k], r["tau_aplicado_min"][k], ru, ro))
    print(f"band {k} (s~{rows[-1][0]:.1f} km, tau {rows[-1][1]:+.0f} min): "
          f"{ru:.3f} -> {ro:.3f}  ({1000*(ru-ro):+.0f} mm)")"""),
     ("code", """\
r = load("results/b1_bathymetry/result.json")
d = r["diagnostico_dev_rtk_DECLARADO"]
print("Villaviciosa dev-RTK declared diagnostic (both variants):", d)
print("NOTE: RTK sits where the operator is identity by measurement;")
print("~0.21 m of sampling error (point vs 10 m pixel median) is the")
print("yardstick's, leaving ~0.13-0.15 m as the method's own error.")"""),
    ])

# ── 05 · uncertainty + hydraulics ────────────────────────────────────────
NOTEBOOKS["05_uncertainty_hydraulics"] = nb(
    "A DEM that knows what it knows: calibrated sigma, sigma decomposition, ponding",
    [
     ("md", """\
Three product layers no published intertidal DEM carries. (1) The fitted
transition width decomposes into sub-pixel relief and the scene-level tide
error (median 0.22 -> 0.177 m of true relief; 35 % of the variance was
level). (2) A per-pixel sigma_z tabulated from planted recoveries in the
calibrated digital twin — conservative by measurement (83 % coverage at the
68 % nominal). (3) Hydraulic layers: spill elevation, ponding depth (14 % of
the intertidal sits behind a barrier), with the flag validated against the
real per-pixel ponding signature and its completeness honestly bounded —
censoring hides itself, so absence of the flag is not proof of terrain."""),
     ("code", PRELUDE),
     ("code", """\
print("B2:", {k: round(v, 3) if isinstance(v, float) else v
              for k, v in load("results/b2_sigma/result.json").items()
              if k != "inputs_sha"})"""),
     ("code", """\
r = load("results/b7_incertidumbre/result.json")
print("sigma_z table (headroom x n_obs), metres:")
print(np.round(np.asarray(r["tabla_sigma_z"], float), 3))
print(f"dev coverage at 68% nominal: {100*r['cobertura_dev']:.0f}% "
      f"(conservative)")"""),
     ("code", """\
r = load("results/b6_hydraulic_dem/result.json")
print(f"ponded pixels: {r['n_px_charco']:,} ({100*r['frac_charco']:.1f} %), "
      f"median depth {r['profundidad_mediana_m']:.2f} m")
print("real-signature judge:", r["juez_real"])
print("simulation judge (self-hiding censoring):", r["juez_simulacion"])"""),
    ])

# ── 06 · negative results ────────────────────────────────────────────────
NOTEBOOKS["06_negative_results"] = nb(
    "What the gates killed — three demonstrated limits worth publishing",
    [
     ("md", """\
1. **Gain is invisible to binary data** (affine theorem, twice: per-band
   alpha and the boundary's dominant-constituent gain). The pipeline
   exposes the degeneracy instead of printing a number.
2. **Per-limb clocks are not adoptable at 465 scenes.** Three versions,
   three distinct failure modes caught: likelihood runaway (limb separation
   predicts wetness), multiple-comparisons adoption under an OOS judge with
   no margin, and finally insufficient power under the matched-null
   adoption threshold. The negative is recorded with a clean control.
3. **Censoring hides itself.** A fully censored pixel refits AT its spill
   level, erasing the geometric evidence of its own pit — hard bound on any
   detector built from the refit map (recall ~0.4-0.5)."""),
     ("code", PRELUDE),
     ("code", """\
r = load("results/b5_gate_sim/veredicto_v3.json")
print("B5 v3 (null-thresholded adoption):", r["puerta"])
print(f"null adoption threshold (OOS improvement): "
      f"{r['umbral_nulo_delta_oos']:.4f}")
print("improvements measured with hysteresis planted:",
      np.round(r["mejoras_oos_con_histeresis"], 4))"""),
     ("code", """\
g = load("results/m2_gate_sim/result.json")["perfil_nll_alpha"]
print(f"NLL(alpha) range with scale-invariant grids: {g['recorrido']:.2e} "
      f"(machine-flat: the theorem, executable)")
b6 = load("results/b6_hydraulic_dem/result.json")["juez_simulacion"]
print(f"censoring self-hiding: precision {b6['precision']:.2f}, "
      f"recall {b6['exhaustividad']:.2f} against planted censoring")"""),
    ])

# ── 07 · north coast census ──────────────────────────────────────────────
NOTEBOOKS["07_north_coast_census"] = nb(
    "The north-coast campaign: 116 cells, one interior-tide census",
    [
     ("md", """\
Every cell of the northern Spanish coast runs the identical MAREA pipeline
(same configs, same seeds, zero per-site tuning). This notebook summarises
whatever the campaign has produced so far: per-cell hypsometric integrals
with uncertainty, and the first interior-tide census of the coast — which
rias run on the ocean model's clock and which run late. Re-run it any time;
it reads runs/*/marea/result.json progressively."""),
     ("code", PRELUDE),
     ("code", """\
import glob
rows = []
for p in sorted(glob.glob("runs/north_coast_cell*/marea/result.json")):
    r = load(p)
    if r.get("estado") != "ok":
        rows.append((r.get("sitio", p), r.get("estado"), None, None, None))
        continue
    tmax = float(np.nanmax(np.abs(r["tau_usado_min"]))) if r["tau_usado_min"] else 0.0
    rows.append((r["sitio"], "ok", r["n_px_cota"], tmax,
                 r["hipsometria"]["integral"]))
print(f"cells processed so far: {len(rows)}")
print(f"{'cell':24s} {'state':10s} {'px':>8s} {'max tau':>8s} {'HI':>5s}")
for n, st, npx, tmax, hi in rows:
    print(f"{n:24s} {st:10s} "
          f"{npx if npx is not None else '-':>8} "
          f"{f'{tmax:+.0f} min' if tmax is not None else '-':>8} "
          f"{f'{hi:.2f}' if hi is not None else '-':>5}")
ok_rows = [r for r in rows if r[1] == 'ok']
if ok_rows:
    with_op = sum(1 for r in ok_rows if r[3] and r[3] > 0)
    print(f"\\ninterior tide detected (operator active) in "
          f"{with_op}/{len(ok_rows)} cells so far")"""),
    ])


def main():
    execute = "--no-exec" not in sys.argv
    for name, notebook in NOTEBOOKS.items():
        # inject repo root for the prelude
        for c in notebook.cells:
            if c.cell_type == "code" and "_ROOT" in c.source:
                c.source = c.source.replace('globals().get("_ROOT", "..")',
                                            repr(ROOT))
        path = os.path.join(OUT, name + ".ipynb")
        if execute:
            client = NotebookClient(notebook, timeout=300,
                                    kernel_name="python3",
                                    resources={"metadata":
                                               {"path": ROOT}})
            try:
                client.execute()
                print(f"executed {name}")
            except Exception as e:
                print(f"EXEC ERROR {name}: {str(e)[:200]}")
        nbf.write(notebook, path)
        print(f"written  {path}")


if __name__ == "__main__":
    main()
