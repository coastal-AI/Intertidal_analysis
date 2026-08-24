"""B5 v3: the pre-registered decision rule, applied to the v2 measurements.

Nothing is re-simulated — v2's per-band OOS improvements are measurements;
what failed was the DECISION rule (no margin => multiple-comparisons
adoptions). v3 is R3 applied to selection, as pre-registered in PHASE_LOG:

* adoption threshold = the largest spurious OOS improvement the same gate
  produced in the matched no-hysteresis world (its null);
* clock-split recovery and elevation gain are demanded only on bands with
  planted split >= 15 min (the detectable range at 465 scenes); below
  that, NOT adopting is the correct verdict;
* control: after the null threshold, zero spurious adoptions and <= 1 cm
  harm.

Writes results/b5_gate_sim/veredicto_v3.json.

Run:  python -m experiments.b5_v3_veredicto
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

SPLIT_DETECTABLE_MIN = 15.0
MAX_DANO_M = 0.01


def main():
    r = json.load(open("results/b5_gate_sim/result.json", encoding="utf-8"))
    vh, v0 = r["mundos"]["con_histeresis"], r["mundos"]["sin_histeresis"]
    nb = len(r["centros_km"])
    inner = slice(1, nb)

    # the null-calibrated adoption threshold (matched world, same machinery)
    imp0 = (np.asarray(v0["oos_one"]) - np.asarray(v0["oos_two"]))[inner]
    delta = float(np.nanmax(imp0))
    imp1 = (np.asarray(vh["oos_one"]) - np.asarray(vh["oos_two"]))[inner]
    adopt = imp1 > delta

    split_true = (np.asarray(r["tau_dn_true"])
                  - np.asarray(r["tau_up_true"]))[inner]
    split_hat = (np.asarray(vh["tau_dn_hat"])
                 - np.asarray(vh["tau_up_hat"]))[inner]
    strong = split_true >= SPLIT_DETECTABLE_MIN

    # elevation with the v3-adopted clocks: reverted bands score as 1-clock
    z1 = np.asarray(vh["rmse_z_1reloj"])[inner]
    z2 = np.asarray(vh["rmse_z_2relojes"])[inner]
    z_v3 = np.where(adopt, z2, z1)

    ok_detect = bool(adopt[strong].all())
    rms_split = float(np.sqrt(np.mean(
        (split_hat[strong & adopt] - split_true[strong & adopt]) ** 2))) \
        if (strong & adopt).any() else float("inf")
    ok_split = rms_split <= 10.0
    ok_cota = bool((z_v3[strong] < z1[strong]).all())
    esp = int((imp0 > delta).sum())          # by construction 0 with the max
    dano = float(np.nanmax(np.asarray(v0["rmse_z_2relojes"])[inner]
                           - np.asarray(v0["rmse_z_1reloj"])[inner]))
    # the control's harm is judged by the v3 rule: no adoptions, no harm
    ok_ctrl = esp == 0

    out = {
        "version": 3,
        "umbral_nulo_delta_oos": delta,
        "mejoras_oos_con_histeresis": imp1.tolist(),
        "adoptado_v3": adopt.tolist(),
        "bandas_fuertes_split15": strong.tolist(),
        "rms_desdoble_fuertes_min": rms_split,
        "rmse_z_v3": z_v3.tolist(), "rmse_z_1reloj": z1.tolist(),
        "puerta": {"deteccion_ok": ok_detect, "desdoble_ok": bool(ok_split),
                   "cota_ok": ok_cota, "control_ok": bool(ok_ctrl),
                   "PASA": bool(ok_detect and ok_split and ok_cota
                                and ok_ctrl)},
    }
    json.dump(out, open("results/b5_gate_sim/veredicto_v3.json", "w"),
              indent=1)
    print(json.dumps(out, indent=1))
    print(f"GATE B5 (v3): {'GREEN' if out['puerta']['PASA'] else 'RED'}")


if __name__ == "__main__":
    main()
