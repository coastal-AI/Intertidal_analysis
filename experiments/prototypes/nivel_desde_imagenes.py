"""Feasibility: can the imagery itself tell us the water level of each date?

The sigmoid fit treats tide height as a fixed input. But h_t is SHARED by
every pixel of that date, while mu and sigma belong to each pixel — so an
error in one date's tide shows up as a residual that is coherent across
thousands of pixels at once, and that is estimable.

Here we run one pass of that idea: holding the fitted pixel parameters, we
re-solve h_t independently for each date, and ask whether the answer moves
towards the CMEMS total sea level (which includes the surge the harmonic
model cannot know about).
"""
import os, sys, json
sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
from scipy.special import erf

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

d = np.load(os.path.join(SC, "fit.npz"), allow_pickle=True)
Y, C, tide = d["epoch_Y"], d["epoch_C"], d["epoch_tide"]
a, b, mu, sg, valid = d["1_a"], d["1_b"], d["1_mu"], d["1_sigma"], d["1_valid"]

# Only pixels the fit trusts, and with a real dry->wet contrast.
keep = valid & (b > 0.2) & (sg < 0.649) & np.isfinite(mu)
a, b, mu, sg = a[keep], b[keep], mu[keep], sg[keep]
Y, C = Y[:, keep], C[:, keep]
T, P = Y.shape
print(f"{P:,} pixeles ajustados x {T} fechas")

# ── Solve h_t per date, holding the pixel parameters ──────────────────────
grid = np.arange(tide.min() - 0.6, tide.max() + 0.6, 0.01)
sg_safe = np.maximum(sg, 1e-3)
h_est = np.full(T, np.nan)
n_used = np.zeros(T, int)

# Precompute Phi on the grid: (n_grid, P) is too big at once, so go by date.
for t in range(T):
    m = C[t]
    n_used[t] = m.sum()
    if m.sum() < 200:                       # too little evidence for a level
        continue
    y = Y[t, m]
    aa, bb, mm, ss = a[m], b[m], mu[m], sg_safe[m]
    # loss(h) = sum_p (y_p - aa_p - bb_p*Phi((h-mm_p)/ss_p))^2
    z = (grid[:, None] - mm[None, :]) / ss[None, :]
    pred = aa[None, :] + bb[None, :] * 0.5 * (1 + erf(z / np.sqrt(2)))
    loss = ((pred - y[None, :]) ** 2).sum(axis=1)
    k = int(np.argmin(loss))
    if 0 < k < grid.size - 1:               # parabolic refinement
        l0, l1, l2 = loss[k - 1], loss[k], loss[k + 1]
        den = l0 - 2 * l1 + l2
        shift = 0.5 * (l0 - l2) / den if abs(den) > 1e-12 else 0.0
        h_est[t] = grid[k] + np.clip(shift, -1, 1) * 0.01
    else:
        h_est[t] = grid[k]

ok = np.isfinite(h_est)
print(f"nivel estimado para {ok.sum()}/{T} fechas "
      f"(mediana {np.median(n_used[ok]):,.0f} pixeles por fecha)")

# ── Anchor: remove the affine part, which is unobservable ────────────────
A = np.polyfit(h_est[ok], tide[ok], 1)
h_anch = np.polyval(A, h_est)
print(f"anclaje a la escala del modelo: pendiente {A[0]:.3f}, "
      f"origen {A[1]:+.3f} m")

resid = h_anch[ok] - tide[ok]
print(f"\ncorreccion estimada respecto a GOT4.10:")
print(f"  RMS {np.std(resid)*100:.1f} cm, p95 |.| "
      f"{np.percentile(np.abs(resid),95)*100:.1f} cm")

# ── The decisive test: does it agree with CMEMS? ─────────────────────────
cmp_path = os.path.join(SC, "marea_cmp.npz")
if os.path.exists(cmp_path):
    z = np.load(cmp_path, allow_pickle=True)
    cdates = list(z["dates"]); got_c, ibi_c = z["got"], z["ibi"]
    # las fechas de la epoca, en el mismo orden que `tide`
    ep_dates = sorted(json.load(open(os.path.join(SC, "tides.json"))))
    ep_dates = [x for x in ep_dates if x >= "2023-01-01"]
    idx = {x: i for i, x in enumerate(ep_dates)}
    common = [(idx[x], j) for j, x in enumerate(cdates)
              if x in idx and ok[idx[x]]]
    if len(common) > 50:
        i_est = np.array([c[0] for c in common])
        i_cm = np.array([c[1] for c in common])
        dr_img = h_anch[i_est] - tide[i_est]          # imagen  - GOT
        dr_cm = ibi_c[i_cm] - got_c[i_cm]             # CMEMS   - GOT
        r = np.corrcoef(dr_img, dr_cm)[0, 1]
        print(f"\nPRUEBA DECISIVA  (n={len(common)})")
        print(f"  correccion segun las imagenes: RMS {np.std(dr_img)*100:.1f} cm")
        print(f"  correccion segun CMEMS:        RMS {np.std(dr_cm)*100:.1f} cm")
        print(f"  CORRELACION ENTRE LAS DOS:     {r:+.3f}")
        print("  (si es positiva y clara, las imagenes estan viendo por su")
        print("   cuenta la misma senal que el modelo regional)")
np.savez(os.path.join(SC, "nivel_img.npz"), h_est=h_est, h_anch=h_anch,
         tide=tide, ok=ok, n_used=n_used)
