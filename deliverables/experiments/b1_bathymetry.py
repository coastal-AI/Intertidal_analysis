"""B1: the bathymetry product, inverted with the distributed tide gauge.

The payoff step: invert every intertidal pixel of Villaviciosa twice —

  * ``uniforme``: the boundary tide as-is (what every published per-pixel
    method assumes: one level for the whole scene);
  * ``operador``: each pixel's level series from the InteriorTide operator
    T (phase profile measured by M2a on the REAL archive in m2_real, only
    where it left the uniform-tide null band; bands inside the null keep
    tau = 0 — the operator never invents signal).

Declared diagnostic (R2): the dev RTK split (rtk.py, reserved never opened)
scores both variants — slope and RMSE per block and pooled. This is a
DIAGNOSTIC, not a tuning target: nothing upstream saw a label.

Outputs results/b1_bathymetry/{bathy.npz, result.json, figure.png}.

Run:  python -m experiments.b1_bathymetry      (~15 min)
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
from pyintertidal import seal, rtk
from pyintertidal import tide_estimators as te
from pyintertidal.marea import invert_series
from pyintertidal.operator import InteriorTide
import pyintertidal as pit

CFG2 = yaml.safe_load(open("configs/m2.yaml", encoding="utf-8"))
OUT = os.path.join("results", "b1_bathymetry")


def invert(Y, C, h, lo, hi):
    """Thin wrapper over the canonical pyintertidal.marea.invert_series."""
    z, _ = invert_series(Y, C, h, lo, hi)
    return z


def main():
    t0 = time.time()
    use_system_certificates()
    from eo_tides.model import model_tides

    mr = json.load(open("results/m2_real/result.json", encoding="utf-8"))
    centers = np.asarray(mr["centros_km"], float)
    v = mr["veredicto"]["m2a_tau"]
    tau_prof = np.asarray(v["real"], float)
    outside = np.asarray(v["fuera_del_nulo"], bool)
    # the operator only carries what beat the null; elsewhere tau = 0
    tau_used = np.where(outside, tau_prof, 0.0)
    tau_used[0] = 0.0                       # the mouth anchor, always

    d = np.load(CFG2["datos"]["store"], allow_pickle=True)
    dates = np.array([str(s) for s in d["dates"]])
    aoi = pit.sites.get("villaviciosa")
    lat_c, lon_c = aoi.centroid
    times = pit.overpass.get_overpass_times(
        aoi.bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([(s in times) and (int(s[:4]) >= CFG2["epoca_min"])
                     for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)

    def tide_at(t):
        return model_tides(x=[lon_c], y=[lat_c], time=t, model="EOT20",
                           directory="tide_models", crs="EPSG:4326",
                           extrapolate=True, cutoff=np.inf,
                           parallel=False).reset_index().sort_values(
            "time")["tide_height"].to_numpy(float)

    bank_taus = [float(x) for x in CFG2["taus_banco_min"]]
    bank = te.ShiftBank(bank_taus, [
        tide_at(t_real - pd.Timedelta(minutes=tv)) for tv in bank_taus])
    h0 = bank.at(0.0)
    T = InteriorTide(bank, centers, np.ones_like(tau_used), tau_used,
                     meta={"boundary": "EOT20 @ real overpass hour (STAC)",
                           "correction": "M3 per results/m3_real",
                           "source": "results/m2_real/result.json",
                           "hashes": mr["inputs_sha"]})
    print(T.describe(), flush=True)

    geo = np.load(CFG2["datos"]["geometria"])
    s_keep = geo["s_keep"].astype(float) / 1000.0
    keep = d["keep"]
    Y = np.nan_to_num(d["Y"][have], nan=0.0).astype(np.float64)
    C = (d["C"][have] > 0).astype(np.float64)
    print(f"{Y.shape[0]} scenes x {Y.shape[1]:,} px "
          f"({time.time()-t0:.0f} s)", flush=True)

    # bands of the FULL pixel set, from the same quantile edges as m2_real
    nb = len(centers)
    edges = np.nanquantile(s_keep, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-9
    band_full = np.full(len(s_keep), -1, int)
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        band_full[np.isfinite(s_keep) & (s_keep > a) & (s_keep <= b)] = k
    # isolated pixels (no s) invert with the uniform boundary
    band_full[band_full < 0] = 0

    lo, hi = float(h0.min()), float(h0.max())
    z_uni = invert(Y, C, h0, lo, hi)
    print(f"uniform inversion done ({time.time()-t0:.0f} s)", flush=True)
    z_op = np.full_like(z_uni, np.nan)
    for k in range(nb):
        cols = np.where(band_full == k)[0]
        h_k = T.level(float(centers[k]))
        z_op[cols] = invert(Y[:, cols], C[:, cols], h_k, lo, hi)
        print(f"  band {k}: tau {tau_used[k]:+.1f} min, "
              f"{len(cols):,} px ({time.time()-t0:.0f} s)", flush=True)

    # ── declared diagnostic on dev RTK (R2; reserved stays hidden) ───────
    SH = tuple(int(x) for x in d["shape"])
    rk = rtk.load_rtk()
    flat_rtk = rk["row"].astype(np.int64) * SH[1] + rk["col"]
    pos = np.searchsorted(keep, flat_rtk)
    valid = (pos < len(keep)) & (keep[np.minimum(pos, len(keep) - 1)]
                                 == flat_rtk)
    # RTK heights are ellipsoidal, the product lives on the tide datum: the
    # ~52 m offset between frames is a CONSTANT, so the honest comparables
    # are the slope (scale-free) and the median-centred RMSE. Note the RTK
    # lies in the mouth bands where the operator is identity by measurement
    # (tau inside the null), so both variants MUST score the same here —
    # the operator's effect lives exactly where no ground truth exists.
    diag = {}
    for name, zz in (("uniforme", z_uni), ("operador", z_op)):
        zi = np.where(valid, zz[np.minimum(pos, len(zz) - 1)], np.nan)
        ok = np.isfinite(zi)
        if ok.sum() >= 30:
            sl = float(np.polyfit(rk["elev"][ok], zi[ok], 1)[0])
            e = zi[ok] - rk["elev"][ok]
            rmse = float(np.sqrt(np.mean((e - np.median(e)) ** 2)))
        else:
            sl, rmse = float("nan"), float("nan")
        diag[name] = {"n": int(ok.sum()), "slope_dev": sl,
                      "rmse_dev_centrado": rmse}
        print(f"dev diagnostic [{name}]: n={int(ok.sum())} "
              f"slope={sl:.3f} centred RMSE={rmse:.3f}", flush=True)

    both = np.isfinite(z_uni) & np.isfinite(z_op)
    delta = z_op - z_uni
    result = {
        "operador": {"centros_km": centers.tolist(),
                     "tau_usado_min": tau_used.tolist(),
                     "fuera_del_nulo": outside.tolist()},
        "n_px_invertidos": {"uniforme": int(np.isfinite(z_uni).sum()),
                            "operador": int(np.isfinite(z_op).sum())},
        "delta_op_uni_m": {
            "mediana_por_banda": [
                float(np.nanmedian(delta[band_full == k])) for k in range(nb)],
            "p95_abs": float(np.nanpercentile(np.abs(delta[both]), 95))},
        "diagnostico_dev_rtk_DECLARADO": diag,
        "inputs_sha": {"store": seal._sha256(CFG2["datos"]["store"]),
                       "m2_real": seal._sha256(
                           "results/m2_real/result.json")},
        "duracion_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "bathy.npz"),
                        z_uniforme=z_uni.astype(np.float32),
                        z_operador=z_op.astype(np.float32),
                        keep=keep, shape=d["shape"],
                        band=band_full.astype(np.int8),
                        tau_usado_min=tau_used)
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    # B4: the shippable product, on the canonical grid (georef from the
    # existing HSR product — same 915x915 grid, verified by shape assert)
    import rasterio
    from pyintertidal.raster import write_geotiff
    with rasterio.open(rtk.DEFAULT_GRID) as src:
        assert src.shape == SH, f"grid {src.shape} != store {SH}"
        tr, crs = src.transform, src.crs
    for name, zz in (("hsr_v4_operador", z_op),
                     ("hsr_v4_uniforme", z_uni)):
        img = np.full(SH[0] * SH[1], np.nan, np.float32)
        img.ravel()[keep] = zz
        write_geotiff(os.path.join("products_villaviciosa",
                                   f"{name}_2023-2025.tif"),
                      img.reshape(SH), tr, crs)
    print("GeoTIFF products written to products_villaviciosa/", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SH = tuple(int(x) for x in d["shape"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, (zz, tt) in enumerate(((z_op, "elevation (operator T)"),
                                  (delta, "operator − uniform (m)"))):
        img = np.full(SH[0] * SH[1], np.nan, np.float32)
        img.ravel()[keep] = zz
        im = ax[i].imshow(img.reshape(SH),
                          cmap="terrain" if i == 0 else "coolwarm",
                          vmin=(None if i == 0 else -0.25),
                          vmax=(None if i == 0 else 0.25))
        ax[i].set_title(tt)
        plt.colorbar(im, ax=ax[i], shrink=0.8)
    ax[2].plot(centers, tau_used, "o-", color="C0")
    ax[2].set_xlabel("s from the mouth (km)")
    ax[2].set_ylabel("operator τ (min)")
    ax[2].set_title("the distributed tide gauge, applied")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=130)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
