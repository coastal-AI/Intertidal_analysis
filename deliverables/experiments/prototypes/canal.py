"""Distance along the channel, measured the way water actually travels.

The new method writes the water level of each date as a profile,

    h_t(p) = h0_t + gamma_t * s_p

so everything depends on s_p meaning something. The obvious choice — Euclidean
distance to permanent water, which ml_honesto.py used — is wrong here: it cuts
straight across land, so two pixels on opposite banks of a headland get the
same s even though the tide reaches them minutes apart. What the tide follows
is the channel.

So s_p is a GEODESIC distance: shortest path from permanent water, constrained
to the surface water can actually occupy (permanent water plus the intertidal
transition zone). Same machinery the connectivity term needs later.

Also, before anything else, this checks the grid. The duplicate-cube incident
left products on two different grids and a silent index mismatch is what
invalidated this afternoon's ML numbers. Nothing here runs unless the raster
grid and the npz agree.

Why the sea comes from water_frequency and not from reference_map: the
reference map on disk dates from 12 August and contains ZERO pixels of the
stable-water class — 132 177 transition and 705 048 land, nothing else, so
the open sea itself is labelled land. Seeding a flood from it is impossible.
water_frequency.tif is current, on the right grid, and is what every result
today actually rests on, so the classes are derived from it:

    sea        wf >= 0.95, largest connected component (48 432 px, and it
               reaches the image edge, so it is the open sea and not a pond)
    reachable  wf >= 0.02 or in the intertidal mask — anywhere water has been
               seen, which is exactly where a flood path may run
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import rasterio
from scipy import ndimage
from skimage.graph import MCP_Geometric

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")

SEA_WF = 0.95      # wet this often = permanent water
REACH_WF = 0.02    # ever seen wet = water may pass through


def check_grid(shape_npz):
    """Every raster this method touches must be on the npz's grid."""
    ok = True
    for name in ("reference_map", "water_frequency", "hsr_eot20_2023-2025"):
        path = f"products_villaviciosa/{name}.tif"
        if not os.path.exists(path):
            print(f"  {name:24s} AUSENTE")
            ok = False
            continue
        with rasterio.open(path) as s:
            same = s.shape == shape_npz
            print(f"  {name:24s} {str(s.shape):12s} "
                  f"{'coincide' if same else '*** NO COINCIDE ***'}")
            ok &= same
    return ok


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def classes(wf, inter):
    """Sea and reachable set, both taken as single connected components.

    Keeping only the largest component matters: 1 310 separate blobs pass the
    wf >= 0.95 test, and the small ones are inland ponds and shadows that
    would seed a flood from the wrong place entirely.
    """
    sea = largest_component(wf >= SEA_WF)
    reachable = largest_component((wf >= REACH_WF) | inter | sea)
    return sea, reachable


def channel_distance(sea, reachable, pixel_m):
    """Geodesic distance from the open sea, through water-reachable ground.

    Returns metres, NaN where water cannot get at all — which is itself a
    result rather than a nuisance: those pixels are hydraulically isolated,
    and the bathtub model that every published method uses has no way to
    say so.
    """
    costs = np.where(reachable, 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=True)
    seeds = [tuple(s) for s in np.argwhere(sea & reachable)]
    dist, _ = mcp.find_costs(starts=seeds)
    dist = np.asarray(dist, dtype=np.float64) * pixel_m
    dist[~np.isfinite(dist)] = np.nan
    return dist


def main():
    d = np.load(os.path.join(SC, "ndwi_intermareal.npz"), allow_pickle=True)
    SH = tuple(int(v) for v in d["shape"])
    keep, field_flat, gnss = d["keep"], d["field_flat"], d["gnss"]
    print(f"npz: rejilla {SH}, {len(keep):,} px intermareales, "
          f"{len(field_flat)} puntos de campo\n")

    print("comprobacion de rejilla:")
    if not check_grid(SH):
        sys.exit("\nABORTADO: hay rasteres en otra rejilla. "
                 "Es el fallo que invalido los numeros de ML.")

    with rasterio.open("products_villaviciosa/water_frequency.tif") as s:
        wf = np.nan_to_num(s.read(1), nan=0.0)
        pixel_m = abs(s.transform.a)

    inter = np.zeros(SH, bool)
    inter.ravel()[keep] = True
    sea, reachable = classes(wf, inter)
    print(f"\nmar abierto {int(sea.sum()):,} px  ·  alcanzable "
          f"{int(reachable.sum()):,} px  ·  pixel {pixel_m:.0f} m")

    # ── the two distances, side by side ──────────────────────────────────
    eucl = ndimage.distance_transform_edt(~sea) * pixel_m
    geo = channel_distance(sea, reachable, pixel_m)

    unreachable = inter & ~np.isfinite(geo)
    n_t = int(inter.sum())
    print(f"px de transicion que el agua NO alcanza: "
          f"{int(unreachable.sum()):,} ({100*unreachable.sum()/max(n_t,1):.1f} %)")

    e_i, g_i = eucl[inter], geo[inter]
    fin = np.isfinite(g_i)
    print(f"\n{'':14s} {'mediana':>9s} {'p90':>9s} {'max':>9s}")
    print(f"{'euclidea':14s} {np.median(e_i):9.0f} "
          f"{np.percentile(e_i,90):9.0f} {e_i.max():9.0f}   m")
    print(f"{'geodesica':14s} {np.median(g_i[fin]):9.0f} "
          f"{np.percentile(g_i[fin],90):9.0f} {g_i[fin].max():9.0f}   m")
    ratio = g_i[fin] / np.maximum(e_i[fin], pixel_m)
    print(f"razon geo/eucl: mediana {np.median(ratio):.2f}, "
          f"p99 {np.percentile(ratio,99):.2f}  "
          f"(1.0 seria que dan lo mismo)")

    # ── does it order the survey the way the ria runs? ───────────────────
    rr, cc = field_flat // SH[1], field_flat % SH[1]
    s_field = geo[rr, cc]
    e_field = eucl[rr, cc]
    ok = np.isfinite(s_field) & np.isfinite(gnss)
    print(f"\npuntos de campo con distancia por canal: "
          f"{int(ok.sum())} de {len(field_flat)}")
    if ok.sum() > 10:
        from scipy.stats import spearmanr
        print(f"  recorrido rio arriba: "
              f"{s_field[ok].min():.0f} a {s_field[ok].max():.0f} m")
        rg = spearmanr(s_field[ok], gnss[ok])
        re = spearmanr(e_field[ok], gnss[ok])
        print(f"  Spearman cota-vs-distancia   geodesica {rg.statistic:+.3f}"
              f"   euclidea {re.statistic:+.3f}")
        print("  (se espera positiva: el intermareal sube rio arriba)")

    np.savez_compressed(os.path.join(SC, "canal.npz"),
                        geo=geo.astype(np.float32),
                        eucl=eucl.astype(np.float32),
                        s_keep=geo.ravel()[keep].astype(np.float32),
                        sea=sea, reachable=reachable,
                        shape=np.array(SH), pixel_m=pixel_m)
    print(f"\nguardado canal.npz")


if __name__ == "__main__":
    main()
