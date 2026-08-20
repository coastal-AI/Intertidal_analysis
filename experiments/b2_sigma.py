"""B2: the sigma decomposition — how much fitted width is TOPOGRAPHY.

The insight that surfaced while calibrating the M0 null and belongs to the
product line: the per-pixel transition width sigma the estimator fits is NOT
sub-pixel relief alone — the scene-level water-level error (sd_level,
measured 0.13 m on 2026-08-18) is convolved into it, because the fit absorbs
scene-wide level noise by widening the transition. Deconvolving,

    sigma_topo = sqrt(max(sigma^2 - sd_level^2, floor^2)),

turns the archive's sigma field into two named, physical numbers per pixel:
the relief inside the pixel and the level uncertainty of the epoch. This
script materialises that decomposition as a product layer with summary
statistics — the roughness map is free science the archive was hiding.

Run:  python -m experiments.b2_sigma        (~1 min, no network)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml

from pyintertidal import seal

CFG0 = yaml.safe_load(open("configs/m0.yaml", encoding="utf-8"))
OUT = os.path.join("results", "b2_sigma")
FLOOR = 0.03      # the smallest sigma atom of the fit grid: below it the
                  # archive cannot distinguish relief from zero


def main():
    B = np.load("data_v4/store/lamina_base.npz")
    sg, good = B["sg"], B["good"]
    sd_level = float(CFG0["simulador"]["sd_nivel_m"])
    sg_topo = np.sqrt(np.maximum(sg ** 2 - sd_level ** 2, FLOOR ** 2))
    share_level = np.clip(sd_level ** 2 / np.maximum(sg, 1e-6) ** 2, 0, 1)

    d = np.load("data_v4/store/ndwi_intermareal.npz", allow_pickle=True)
    keep, SH = d["keep"], tuple(int(x) for x in d["shape"])

    g = good & np.isfinite(sg)
    result = {
        "sd_level_m": sd_level,
        "sigma_ajustada_mediana": float(np.median(sg[g])),
        "sigma_topo_mediana": float(np.median(sg_topo[g])),
        "fraccion_varianza_nivel_mediana": float(np.median(share_level[g])),
        "px_dominados_por_nivel": int((share_level[g] > 0.5).sum()),
        "n_px": int(g.sum()),
        "inputs_sha": {
            "base": seal._sha256("data_v4/store/lamina_base.npz"),
            "store": seal._sha256("data_v4/store/ndwi_intermareal.npz")},
    }
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "sigma_decomp.npz"),
                        sg=sg.astype(np.float32),
                        sg_topo=sg_topo.astype(np.float32),
                        share_level=share_level.astype(np.float32),
                        keep=keep, shape=d["shape"], sd_level=sd_level)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    img = np.full(SH[0] * SH[1], np.nan, np.float32)
    img.ravel()[keep] = np.where(g, sg_topo, np.nan)
    im = ax[0].imshow(img.reshape(SH), cmap="magma", vmax=0.5)
    ax[0].set_title("σ_topo: sub-pixel relief (m)")
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].hist(sg[g], bins=40, alpha=0.55, label="fitted σ")
    ax[1].hist(sg_topo[g], bins=40, alpha=0.55, label="σ_topo (deconv.)")
    ax[1].axvline(sd_level, color="k", ls="--",
                  label=f"sd_level = {sd_level} m")
    ax[1].set_xlabel("σ (m)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("the fitted width carries the level inside")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
