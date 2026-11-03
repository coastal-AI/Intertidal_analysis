# -*- coding: utf-8 -*-
"""Builds notebooks/validation/v3_maps_site.ipynb: the elevation maps of validation v3's maps-only sites.

Santander and Santoña (the design's amendment of 2026-11-03: their LiDAR was flown at high water) get maps, no score:
the elevations of the design's methods on the window of the 100 screened scenes nearest 2024-07-15, on every pixel
the production's no-delay inversion maps. The notebook is notebook 05's (``_build_05_attribution.py``) from its
parameters to its estimates, cell for cell (with ``NB_V3`` naming a maps-only key, its cells take the maps-only
branches: no truth, no class mask), and one cell of maps: a GeoTIFF per method on the cube's grid, one figure with a
map per method on one colour scale, and a table of their ranges. Nothing after notebook 05's estimates (the scores,
the decomposition, the diagnostics) is run.

    python notebooks/validation/_build_v3_maps.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nbformat  # noqa: E402

import _build_05_attribution as b05  # noqa: E402

OUT = os.path.join(HERE, "v3_maps_site.ipynb")

INTRO = r"""
# Validation v3 · the elevation maps of a maps-only site

Santander and Santoña (the design `experiments/validation_v3_cantabrian_2026-10-29.md`, its amendment of 2026-11-03):
their PNOA LiDAR was flown at high water, so they get elevation maps and no score. The cells up to the estimates are
notebook 05's, cell for cell (`NB_V3` names `santander_maps` or `santona_maps`: no truth, no class mask, the window of
the 100 screened scenes nearest 2024-07-15, every pixel of the production's no-delay product); the last cells write
the maps. The elevations are on the boundary's datum (EOT20 + the nearest IOC gauge), as every product of the project.
"""

MAPS = r'''
import rasterio

show = [m for m in ORDER if m in V3_REPORT]       # the original MAREA and the n-tables' methods
MAPS_DIR = f"{OUT}/{TAG}_maps"
os.makedirs(MAPS_DIR, exist_ok=True)
summary = []
for key in KEYS:
    win = next(w for w in windows if (w["site"], w["n"]) == key)
    px = win["px"]
    est = {m: EST[key][m] for m in show if m in EST[key]}
    vals = np.concatenate([z[np.isfinite(z)] for z in est.values()])
    lo, hi = np.percentile(vals, [2, 98])
    rows_, cols_ = np.divmod(keep[px], W)
    r0, r1, c0, c1 = rows_.min(), rows_.max() + 1, cols_.min(), cols_.max() + 1
    ncol = min(5, len(est))
    nrow = int(np.ceil(len(est) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow * (r1 - r0) / (c1 - c0) + 0.9),
                             squeeze=False)
    for ax, (m, z) in zip(axes.flat, est.items()):
        r = np.full(H * W, np.nan, np.float32)
        r[keep[px]] = z
        with rasterio.open(f"{MAPS_DIR}/{key[0]}_n{key[1]}_{m}.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                           compress="deflate") as dst:
            dst.write(r.reshape(H, W), 1)
        im = ax.imshow(r.reshape(H, W)[r0:r1, c0:c1], cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        fin = np.isfinite(z)
        ax.set_title(f"{LABELS.get(m, m)}\n{int(fin.sum()):,} px · median {np.nanmedian(z):+.2f} m", fontsize=7,
                     color=PAL["accent"] if m.startswith("MAREA") else PAL["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        q = np.percentile(z[fin], [5, 50, 95]) if fin.any() else [np.nan] * 3
        summary.append(dict(site=key[0], n=key[1], method=m, label=LABELS.get(m, m), pixels=len(px),
                            with_value=int(fin.sum()), p5=q[0], median=q[1], p95=q[2]))
    for ax in list(axes.flat)[len(est):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="elevation (m, the boundary's datum)")
    d = S_dates[win["scenes"]]
    fig.suptitle(f"{V3KEY}: the elevation of each method, n = {key[1]} ({d[0]} → {d[-1]})", fontsize=9)
    fig.savefig(f"{OUT}/{TAG}_n{key[1]}_elevation_maps.png")
    plt.show()
summary = pd.DataFrame(summary)
summary.to_csv(f"{OUT}/{TAG}_elevation_summary.csv", index=False)
display(summary.set_index(["n", "label"])[["with_value", "p5", "median", "p95"]].round(3))
print(f"-> {MAPS_DIR}/ (one GeoTIFF per method), {OUT}/{TAG}_n*_elevation_maps.png, {OUT}/{TAG}_elevation_summary.csv")
'''


def build(out=OUT):
    cells = list(b05.CELLS)
    stop = next(i for i, (kind, src) in enumerate(cells)
                if kind == "code" and src.startswith("SCORES, EST, EXTRA_PX, TAU, LOOKS"))
    prov = next(src for kind, src in cells if kind == "code" and src.startswith("import platform, socket"))
    chosen = [("markdown", INTRO.strip("\n"))] + cells[1:stop + 1] + [
        ("markdown", "## The maps\n\nOne GeoTIFF per method on the cube's grid; one figure, one colour scale "
                     "(the 2nd to 98th percentile of every method's elevations)."),
        ("code", MAPS.strip("\n")), ("markdown", "## Provenance"), ("code", prov)]
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    for k, (kind, src) in enumerate(chosen):
        cell = nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        cell["id"] = f"v3maps-{k:02d}"
        nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    print("wrote", build(*sys.argv[1:2]))
