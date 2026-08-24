"""How much can a sloping water surface possibly fix? Measure the ceiling first.

The new method adds one spatial degree of freedom: the water level varies
along the channel. Whatever the solver ends up doing, its effect on any pixel
is a shift proportional to that pixel's channel distance s.

So there is a hard ceiling on what it can achieve against the survey, and it
can be read off directly, before writing a single line of solver: take HSR's
residual against the RTK GNSS and ask how much of it a term linear in s can
remove. If the answer is "almost none", the method cannot help here no matter
how well it is implemented, and it is far better to learn that in one minute
than after fitting.

This matters especially because the survey turns out to span only 141-887 m
of channel distance, in a ria whose intertidal reaches 3.5 km. The lever arm
is short.

Two ceilings are reported:

  * the BEST CASE — fit the linear-in-s term on the same points it is scored
    on. Nothing the real method does can beat this; it is an upper bound, not
    a result.
  * the same thing on the held-out blocks, fitted on the training half only,
    which is what an honest version would actually deliver.

Neither is the method. Both bound it.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy import stats

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
BLOCK_M = 150.0
HOLDOUT_FRACTION = 0.35
SEED = 20260817


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    field_flat, gnss = d["field_flat"], d["gnss"]
    SH = tuple(int(v) for v in d["shape"])
    ch = np.load(os.path.join(SC, "canal.npz"))
    geo = ch["geo"]

    rr, cc = field_flat // SH[1], field_flat % SH[1]
    with rasterio.open("products_villaviciosa/hsr_eot20_2023-2025.tif") as s:
        hsr = s.read(1)[rr, cc]
        tr = s.transform
    with rasterio.open("products_villaviciosa/dea_eot20_2023-2025.tif") as s:
        dea = s.read(1)[rr, cc]
    s_m = geo[rr, cc].astype(float)

    ok = (np.isfinite(hsr) & np.isfinite(dea) & np.isfinite(gnss)
          & np.isfinite(s_m))
    y, hsr, dea, s_m, rr, cc = (gnss[ok], hsr[ok], dea[ok], s_m[ok],
                                rr[ok], cc[ok])
    print(f"{len(y)} puntos · canal {s_m.min():.0f}-{s_m.max():.0f} m "
          f"(brazo {s_m.max()-s_m.min():.0f} m)\n")

    # ── is there a trend at all? ─────────────────────────────────────────
    print(f"{'metodo':10s} {'RMSE':>7s} {'pend':>7s} "
          f"{'r(residuo, s)':>14s} {'p':>9s} {'m/km':>8s}")
    print("-" * 60)
    fits = {}
    for name, v in (("HSR", hsr), ("DEA", dea)):
        res = y - v
        res = res - np.median(res)
        r = stats.pearsonr(s_m, res)
        sl = np.polyfit(s_m, res, 1)[0] * 1000.0
        fits[name] = (v, res, sl)
        print(f"{name:10s} {np.sqrt(np.mean(res**2)):7.3f} "
              f"{np.polyfit(y, v, 1)[0]:7.3f} {r.statistic:+14.3f} "
              f"{r.pvalue:9.2g} {sl:+8.3f}")

    print("\nr = correlacion del residuo con la distancia por el canal.")
    print("Si es ~0, una lamina inclinada no puede corregir nada aqui.\n")

    # ── ceiling 1: best possible, fitted where it is scored ──────────────
    print(f"{'techo (optimista, se ajusta donde se puntua)':46s}")
    print("-" * 60)
    for name, (v, res, sl) in fits.items():
        A = np.column_stack([np.ones_like(s_m), s_m])
        coef, *_ = np.linalg.lstsq(A, res, rcond=None)
        corr = v + A @ coef
        r2 = y - corr
        r2 = r2 - np.median(r2)
        print(f"  {name:8s} {np.sqrt(np.mean(res**2)):.3f} -> "
              f"{np.sqrt(np.mean(r2**2)):.3f} m   "
              f"pendiente {np.polyfit(y, v, 1)[0]:.3f} -> "
              f"{np.polyfit(y, corr, 1)[0]:.3f}")

    # ── ceiling 2: the same, but honest about where it was fitted ────────
    east = tr.c + (cc + .5) * tr.a
    north = tr.f + (rr + .5) * tr.e
    block = (((east - east.min()) // BLOCK_M).astype(int) * 1000
             + ((north - north.min()) // BLOCK_M).astype(int))
    blocks = np.unique(block)
    rng = np.random.default_rng(SEED)
    hold = np.isin(block, rng.choice(
        blocks, max(2, int(round(HOLDOUT_FRACTION * len(blocks)))),
        replace=False))
    print(f"\n{'techo honesto (ajustado solo en entrenamiento)':46s}")
    print(f"  entrenamiento {int((~hold).sum())} px · "
          f"reservado {int(hold.sum())} px")
    print("-" * 60)
    for name, (v, _, _) in fits.items():
        off = np.median(y[~hold] - v[~hold])
        base = y[hold] - v[hold] - off
        A = np.column_stack([np.ones_like(s_m), s_m])
        coef, *_ = np.linalg.lstsq(A[~hold], (y - v)[~hold], rcond=None)
        corr = v[hold] + A[hold] @ coef
        print(f"  {name:8s} {np.sqrt(np.mean(base**2)):.3f} -> "
              f"{np.sqrt(np.mean((y[hold]-corr)**2)):.3f} m   "
              f"pendiente {np.polyfit(y[hold], v[hold], 1)[0]:.3f} -> "
              f"{np.polyfit(y[hold], corr, 1)[0]:.3f}")

    # ── how far does the survey reach into the ria? ──────────────────────
    inter_s = geo.ravel()[d["keep"]]
    inter_s = inter_s[np.isfinite(inter_s)]
    covered = ((inter_s >= s_m.min()) & (inter_s <= s_m.max())).mean()
    arm = float(s_m.max() - s_m.min())
    print(f"\nel GNSS cubre {100*covered:.0f} % del intermareal por distancia "
          f"de canal;\nel resto llega hasta {np.nanmax(inter_s):.0f} m, "
          f"donde el brazo es {np.nanmax(inter_s)/arm:.1f}x mayor.")
    print("\nEste techo acota SOLO la parte estatica (un sesgo fijo lineal en s).")
    print("La parte que produce la compresion es que gamma cambie con la marea,")
    print("y esa no la acota este test — se mide en desdoble.py.")


if __name__ == "__main__":
    main()
