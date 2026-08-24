"""B6: the hydraulically-aware DEM — terrain vs spill elevation, declared.

Flooding needs a path (101 of 361 RTK points never once seen wet). For a
pixel behind a barrier, the wet/dry archive observes the SPILL level (the
minimax sill on the way from the sea), not the terrain: everything below
the sill is censored. This experiment turns that physics into product
layers and puts the claim through two judges:

(a) SIMULATION: plant connectivity censoring with the map's own REAL sills
    (simulator.ConnectivityCensoring), refit, recompute sills on the refit,
    flag — precision and recall of the flag against the planted censored
    set must clear 0.7 (configs/b6.yaml);
(b) REAL: flagged pixels must concentrate the per-pixel ponding signature
    (charcos, sealed: wet-more-when-falling at matched levels) against a
    null that permutes flags within s-band x elevation-quintile cells —
    the flag has to know something the elevation and position alone do not.

Outputs results/b6_hydraulic_dem/{capas.npz, result.json, figure.png}:
spill elevation, ponding depth (spill - z), and the censored flag.

Run:  python -m experiments.b6_dem_hidraulico    (~10 min)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from pyintertidal.net import use_system_certificates
from pyintertidal import simulator, seal, geometry
from pyintertidal.marea import invert_series
import pyintertidal as pit

CFG = yaml.safe_load(open("configs/b6.yaml", encoding="utf-8"))
CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "b6_dem_hidraulico")


def refit(Y, C, h):
    """Thin wrapper over the canonical pyintertidal.marea.invert_series."""
    z, _ = invert_series(Y, C, h)
    return z


def spill_of(z_keep, keep, SH, sea):
    zmap = np.full(SH[0] * SH[1], np.nan, float)
    zmap[keep] = z_keep
    zmap = zmap.reshape(SH)
    valid = np.isfinite(zmap) | sea
    zmap2 = np.where(sea & ~np.isfinite(zmap), np.nanmin(zmap) - 1.0, zmap)
    sp = geometry.flood_threshold(zmap2, sea, valid)
    return sp.ravel()[keep]


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    d = np.load(CFG["datos"]["store"], allow_pickle=True)
    dates = np.array([str(s) for s in d["dates"]])
    SH = tuple(int(x) for x in d["shape"])
    keep = d["keep"]
    sea = np.load(CFG["datos"]["canal"])["sea"]
    B = np.load(CFG["datos"]["bathy"])
    z_op = B["z_operador"].astype(float)

    # ── the layers on the real product ───────────────────────────────────
    spill = spill_of(z_op, keep, SH, sea)
    depth = spill - z_op
    flag = np.isfinite(depth) & (depth > CFG["umbral_charco_m"])
    print(f"spill computed: {int(flag.sum()):,} ponded px "
          f"({100*np.nanmean(flag):.1f} %) · median depth "
          f"{np.nanmedian(depth[flag]):.2f} m ({time.time()-t0:.0f} s)",
          flush=True)

    # ── judge (a): simulation with the real sills planted ────────────────
    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([(s in times) and (int(s[:4]) >= CFG2["epoca_min"])
                     for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    h0 = model_tides(x=[lon_c], y=[lat_c], time=t_real, model="EOT20",
                     directory="tide_models", crs="EPSG:4326",
                     extrapolate=True, cutoff=np.inf,
                     parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    tpl = simulator.calibrate(CFG["datos"]["store"], CFG["datos"]["base"],
                              h0, epoch_years=CFG2["epoca_min"],
                              sd_level=CFG2["sd_nivel_m"])
    # template pixels with a REAL sill from the product map
    z_true = tpl["z"]
    sp_t = spill[tpl["idx"]]
    thr = np.where(np.isfinite(sp_t), np.maximum(sp_t, z_true), z_true)
    planted = (thr - z_true) > CFG["umbral_charco_m"]
    C_real = (d["C"][have][:, tpl["idx"]] > 0)
    Y, C = simulator.simulate(
        tpl, z_true, h0, seed=CFG["puerta"]["seed"],
        mechanisms=(simulator.ConnectivityCensoring(thr),),
        C_real=C_real, template_rows=np.arange(tpl["n_template"]))
    z_hat = refit(Y, C, h0)
    sp_hat = spill_of(z_hat, keep[tpl["idx"]], SH, sea)
    flag_hat = np.isfinite(sp_hat) & ((sp_hat - z_hat)
                                      > CFG["umbral_charco_m"])
    both = np.isfinite(z_hat) & np.isfinite(z_true)
    tp = int((flag_hat & planted & both).sum())
    prec = tp / max(int((flag_hat & both).sum()), 1)
    rec = tp / max(int((planted & both).sum()), 1)
    ok_sim = (prec >= CFG["puerta"]["min_precision"]
              and rec >= CFG["puerta"]["min_exhaustividad"])
    print(f"judge (a) simulation: precision {prec:.2f} recall "
          f"{rec:.2f} (planted {int(planted.sum()):,}) "
          f"({time.time()-t0:.0f} s)", flush=True)

    # ── judge (b): the real ponding signature, against a matched null ────
    ch = np.load(CFG["datos"]["charcos"])
    pond = ch["pond"]
    geo = np.load(CFG["datos"]["geometria"])
    s_keep = geo["s_keep"].astype(float)
    okp = np.isfinite(pond) & np.isfinite(z_op) & np.isfinite(s_keep)
    sb = np.clip(np.digitize(s_keep, np.nanquantile(
        s_keep[okp], np.linspace(0, 1, 7)[1:-1])), 0, 5)
    zq = np.clip(np.digitize(z_op, np.nanquantile(
        z_op[okp], np.linspace(0, 1, 6)[1:-1])), 0, 4)
    cell = sb * 8 + zq
    obs = float(np.nanmean(pond[okp & flag])
                - np.nanmean(pond[okp & ~flag]))
    rng = np.random.default_rng(CFG["puerta"]["seed"])
    null = []
    fl = flag[okp]
    pv = pond[okp]
    cl = cell[okp]
    for _ in range(CFG["puerta"]["n_permutaciones"]):
        f2 = fl.copy()
        for c in np.unique(cl):
            m = cl == c
            f2[m] = rng.permutation(f2[m])
        null.append(np.nanmean(pv[f2]) - np.nanmean(pv[~f2]))
    null = np.asarray(null)
    p95 = float(np.percentile(null, 95))
    ok_real = obs > p95
    print(f"judge (b) real: ponding signature {obs:+.4f} vs null p95 "
          f"{p95:+.4f} (null mean {null.mean():+.4f})", flush=True)

    result = {
        "n_px_charco": int(flag.sum()),
        "frac_charco": float(np.nanmean(flag)),
        "profundidad_mediana_m": float(np.nanmedian(depth[flag])),
        "juez_simulacion": {"precision": float(prec),
                            "exhaustividad": float(rec),
                            "n_plantados": int(planted.sum())},
        "juez_real": {"firma_observada": obs, "nulo_p95": p95,
                      "nulo_media": float(null.mean())},
        "puerta": {"sim_ok": bool(ok_sim), "real_ok": bool(ok_real),
                   "PASA": bool(ok_sim and ok_real)},
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "capas.npz"),
                        spill=spill.astype(np.float32),
                        depth=depth.astype(np.float32),
                        flag=flag, keep=keep, shape=d["shape"])
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    img = np.full(SH[0] * SH[1], np.nan, np.float32)
    img[keep] = np.where(flag, depth, np.nan)
    im = ax[0].imshow(img.reshape(SH), cmap="Blues", vmax=1.0)
    ax[0].set_title("ponding depth (m): spill elevation − terrain")
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].hist(null, bins=30, alpha=0.6, label="matched null")
    ax[1].axvline(obs, color="r", lw=2, label=f"observed {obs:+.3f}")
    ax[1].set_xlabel("ponding signature (ponded − not ponded)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("the flag knows something elevation and position do not")
    fig.suptitle(f"Gate B6 — "
                 f"{'GREEN' if result['puerta']['PASA'] else 'RED'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(f"GATE B6: {'GREEN' if result['puerta']['PASA'] else 'RED'}",
          flush=True)


if __name__ == "__main__":
    main()
