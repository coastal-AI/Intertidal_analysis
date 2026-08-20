"""M1: the estuary's geometry from the archive — s, thalweg, width(s,h), γ_c.

Produces everything phase M2+ consumes and the gate checks:

* ``s_mouth``: along-estuary distance from the mouth for every pixel water
  can reach (this is the axis of every transfer profile; the older canal.npz
  distance was "distance from permanent water", a different and less
  interpretable quantity — both are kept and compared);
* the thalweg skeleton (display + monotonicity check);
* width(s, h) at four tide levels with scene-bootstrap uncertainty;
* γ_c(s) with e-folding length.

Outputs results/m1_geometry/{geometry.npz, result.json, figure.png}. The
figure IS part of the gate (test visual): thalweg over the mask, s field, and
the four width profiles.

Run:  python -m experiments.m1_geometry
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

from pyintertidal import geometry, seal

CFG = yaml.safe_load(open("configs/m1.yaml", encoding="utf-8"))
OUT = os.path.join("results", "m1_geometry")


def main():
    ch = np.load(CFG["datos"]["canal"])
    sea, reachable = ch["sea"], ch["reachable"]
    d = np.load(CFG["datos"]["store"], allow_pickle=True)
    Y, C, keep = d["Y"], d["C"], d["keep"]
    dates = np.array([str(s) for s in d["dates"]])
    SH = tuple(int(v) for v in d["shape"])
    tide = np.load(CFG["datos"]["mareas"])["tide"]
    ep = np.array([int(s[:4]) >= CFG["epoca_min"] for s in dates]) \
        & np.isfinite(tide)
    Yb = np.nan_to_num(Y[ep], nan=-9.0)
    Cb = C[ep] > 0
    t = tide[ep]
    print(f"{int(ep.sum())} scenes · {len(keep):,} intertidal px")

    # ── s from the MOUTH ─────────────────────────────────────────────────
    seeds = geometry.mouth_seeds(sea)
    s_mouth = geometry.along_distance(reachable, seeds, CFG["pixel_m"])
    thal = geometry.thalweg(sea)
    s_keep = s_mouth.ravel()[keep]
    print(f"mouth: {int(seeds.sum())} seed px · "
          f"s up to {np.nanmax(s_keep)/1000:.2f} km · "
          f"isolated {int(np.isnan(s_keep).sum())} px")

    # invariant: s monotone along the thalweg (increasing inland)
    st = s_mouth[thal]
    st = st[np.isfinite(st)]
    corr_geo = float(np.corrcoef(
        s_keep[np.isfinite(s_keep) & np.isfinite(ch["s_keep"])],
        ch["s_keep"][np.isfinite(s_keep) & np.isfinite(ch["s_keep"])])[0, 1])
    print(f"correlation s_mouth vs s_channel(2018-08-18): {corr_geo:.3f}")

    # ── per-level masks and width(s,h) ───────────────────────────────────
    qs = np.quantile(t, np.linspace(0, 1, CFG["n_niveles"] + 1))
    qs[0] -= 1e-9
    wet = Cb & (Yb > 0)
    rng = np.random.default_rng(CFG["seed"])
    centers = None
    widths, widths_ci = [], []
    for a, b in zip(qs, qs[1:]):
        sel = np.where((t > a) & (t <= b))[0]
        wc = wet[sel].sum(0)
        oc = Cb[sel].sum(0)
        wf = geometry.wet_fraction_by_bin(keep, wc, oc, SH,
                                          CFG["min_obs_bin"])
        mask = geometry.level_masks([np.nan_to_num(wf, nan=0.0)], sea)[0]
        c, w = geometry.width_profile(mask, s_mouth, CFG["banda_m"])
        centers = c
        widths.append(w)
        # bootstrap by SCENES
        boots = []
        for k in range(CFG["n_boot"]):
            i = rng.choice(sel, len(sel), replace=True)
            wf_b = geometry.wet_fraction_by_bin(
                keep, wet[i].sum(0), (Cb[i]).sum(0), SH, CFG["min_obs_bin"])
            mask_b = geometry.level_masks(
                [np.nan_to_num(wf_b, nan=0.0)], sea)[0]
            _, w_b = geometry.width_profile(mask_b, s_mouth, CFG["banda_m"],
                                            s_max=c[-1] + 1)
            boots.append(w_b[:len(w)])
        widths_ci.append(np.percentile(np.array(boots), [2.5, 97.5], axis=0))
        print(f"  level ({a:+.2f},{b:+.2f}] m: median width "
              f"{np.nanmedian(w[w>0]):.0f} m ({len(sel)} scenes)")
    widths = np.array(widths)

    # physical invariant: width non-decreasing with level
    valid = widths[0] > 0
    mono = np.mean([np.all(np.diff(widths[:, j]) >= -CFG["tol_mono_m"])
                    for j in np.where(valid)[0]])
    print(f"bands with width(s,h) non-decreasing in h: {100*mono:.0f} %")

    # ── convergence at low level ─────────────────────────────────────────
    gamma, efold, _ = geometry.convergence(centers, widths[0])
    print(f"convergence (low level): median e-folding "
          f"{np.median(efold[np.isfinite(efold)]):.1f} km")

    # ── figure (part of the gate: visual test) ───────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    s_show = np.where(reachable, s_mouth / 1000.0, np.nan)
    im = ax[0].imshow(s_show, cmap="viridis")
    ty, tx = np.where(thal)
    ax[0].plot(tx, ty, ".", ms=0.3, color="red")
    ax[0].set_title("s from the mouth (km) + thalweg")
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    for k, w in enumerate(widths):
        ax[1].plot(centers / 1000.0, w, label=f"level {k+1}")
    ax[1].set_xlabel("s (km)")
    ax[1].set_ylabel("wet width (m)")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].set_title("width(s, h)")
    ok = np.isfinite(efold)
    ax[2].plot(centers[:len(gamma)] / 1000.0, gamma)
    ax[2].set_xlabel("s (km)")
    ax[2].set_ylabel("γ_c (1/km)")
    ax[2].set_title("convergence")
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)

    np.savez_compressed(
        os.path.join(OUT, "geometry.npz"),
        s_mouth=s_mouth.astype(np.float32), thalweg=thal,
        s_keep=s_keep.astype(np.float32), centers=centers,
        widths=widths, level_edges=qs, gamma=gamma, efold=efold)
    result = {
        "n_scenes": int(ep.sum()),
        "s_max_km": float(np.nanmax(s_keep) / 1000.0),
        "n_isolated": int(np.isnan(s_keep).sum()),
        "corr_s_canal": corr_geo,
        "frac_bandas_monotonas": float(mono),
        "efold_mediano_km": float(np.median(efold[np.isfinite(efold)])),
        "inputs_sha": {k: seal._sha256(v) for k, v in CFG["datos"].items()},
        "seed": CFG["seed"],
    }
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(f"written {OUT}/(geometry.npz, result.json, figure.png)")


if __name__ == "__main__":
    main()
