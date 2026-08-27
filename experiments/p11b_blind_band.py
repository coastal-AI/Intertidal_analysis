"""P11b — the blind-band test: self-calibrated depth from inverted NDWI.

The follow-up p11 registered: the extraction masks out never-dry pixels by
construction, so the blind test had no subjects. Here they are: SEA-class
pixels of Aveiro whose LiDAR elevation lies just BELOW the lowest tide the
satellite ever sampled — the exact analogue of Santander's central banks,
unreachable by any wet/dry method.

For each such pixel, every clear observation whose NDWI falls inside the
monotone 0-1 m window of the p11 master curve is inverted into a depth
d_hat, giving one sounding z = tide(t) - d_hat; the pixel's elevation is
the median of its soundings. The master curve was built WITHOUT these
pixels (calibration = sometimes-exposed pixels only), and the LiDAR never
enters the estimation — it only grades it.

Also scored: the trivial baseline (assign every pixel the band's middle
elevation), so the skill is read against "knowing nothing".

Declared limitation: selecting the test band uses the LiDAR (we test
accuracy IN the band, not the ability to find the band blindly); the
blind band-finding problem — e.g. flagging non-invertible turbid-channel
pixels from their own NDWI statistics — is registered as the next rung.

Run:  python -m experiments.p11b_blind_band      (~10 min)
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

from pyintertidal.net import use_system_certificates
from pyintertidal import seal
import pyintertidal as pit
from experiments.p4_site import extract

OUT = os.path.join("results", "p11b_blind_band")
CUBE = "ndwi_cube_aveiro_2023-2025.nc"
BAND_BELOW_M = 0.6          # how far below the lowest sampled tide to test
MIN_SOUNDINGS = 4


def main():
    t0 = time.time()
    use_system_certificates()
    import xarray as xr
    import pyproj
    from eo_tides.model import model_tides

    p11 = json.load(open("results/p11_ndwi_depth/result.json",
                         encoding="utf-8"))
    datum = p11["datum_m"]
    dcent = np.asarray(p11["curva"]["d_m"], float)
    ndwi_med = np.asarray(p11["curva"]["ndwi_mediana"], float)
    win = np.isfinite(ndwi_med) & (dcent > -0.05) & (dcent < 1.0)
    xc, yc = dcent[win], ndwi_med[win]
    order = np.argsort(yc)
    yc_s, xc_s = yc[order], xc[order]
    print(f"inversion window: d {xc_s.min():.2f}..{xc_s.max():.2f} m, "
          f"NDWI {yc_s.min():+.2f}..{yc_s.max():+.2f}")

    # sea mask + scene dates from the standard extraction
    Y_i, C_i, keep, dates, SH, sea, inter, bbox = extract(CUBE)
    del Y_i, C_i
    H, W = SH
    times = pit.overpass.get_overpass_times(
        bbox, ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    lat_c = 0.5 * (bbox["south"] + bbox["north"])
    lon_c = 0.5 * (bbox["west"] + bbox["east"])
    h = model_tides(x=[lon_c], y=[lat_c], time=t_real, model="EOT20",
                    directory="tide_models", crs="EPSG:4326",
                    extrapolate=True, cutoff=np.inf,
                    parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)
    h_min = float(h.min())
    print(f"lowest sampled tide: {h_min:+.2f} m "
          f"({time.time()-t0:.0f} s)", flush=True)

    # LiDAR at the sea pixels
    ds = xr.open_dataset(CUBE)
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(crs_cube, 4326, always_xy=True)
    sea_flat = np.where(sea.ravel())[0]
    rows, cols_px = sea_flat // W, sea_flat % W
    lon_px, lat_px = tr.transform(ds["x"].values[cols_px],
                                  ds["y"].values[rows])
    ref = xr.open_dataset("bathy_candidatos/590_HR_Lidar_Norte.nc")
    z_ref = ref["elevation"].interp(
        lon=xr.DataArray(lon_px), lat=xr.DataArray(lat_px),
        method="nearest").values
    z_t = z_ref - datum
    blind = np.isfinite(z_t) & (z_t < h_min - 0.02) \
        & (z_t > h_min - BAND_BELOW_M)
    idx = sea_flat[blind]
    z_true = z_t[blind]
    print(f"blind-band sea pixels: {len(idx):,} "
          f"(z in [{h_min-BAND_BELOW_M:+.2f}, {h_min-0.02:+.2f}] m)",
          flush=True)

    # stream their NDWI series
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    T = ds.sizes[t_dim]
    rr, cc = idx // W, idx % W
    Yb = np.full((int(have.sum()), len(idx)), np.nan, np.float32)
    hpos = 0
    for i0 in range(0, T, 40):
        i1 = min(i0 + 40, T)
        sub = np.where(have[i0:i1])[0]
        if not len(sub):
            continue
        g = ds["B03"].isel({t_dim: slice(i0, i1)}).values.astype(np.float32)
        n = ds["B08"].isel({t_dim: slice(i0, i1)}).values.astype(np.float32)
        scl = ds["SCL"].isel({t_dim: slice(i0, i1)}).values
        clear = np.isin(scl, (4, 5, 6, 7))
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
        ndwi = np.where(clear, ndwi, np.nan)
        Yb[hpos:hpos + len(sub)] = ndwi[sub][:, rr, cc]
        hpos += len(sub)
    print(f"series streamed ({time.time()-t0:.0f} s)", flush=True)

    # invert: one sounding per usable observation, median per pixel
    np.savez_compressed(os.path.join("results", "p11b_series.npz"),
                        Yb=Yb, h=h, idx=idx, z_true=z_true)
    # soundings ONLY from low-tide scenes: the pixel must plausibly be
    # inside the invertible 0-1 m window when observed
    low_scene = h < (h_min + 0.7)
    print(f"low-tide scenes usable: {int(low_scene.sum())}", flush=True)
    z_hat = np.full(len(idx), np.nan)
    n_used = np.zeros(len(idx), np.int32)
    for j in range(len(idx)):
        y = np.where(low_scene, Yb[:, j], np.nan)
        m = np.isfinite(y) & (y > yc_s[0]) & (y < yc_s[-1])
        if m.sum() < MIN_SOUNDINGS:
            continue
        d_est = np.interp(y[m], yc_s, xc_s)
        z_hat[j] = float(np.median(h[m] - d_est))
        n_used[j] = int(m.sum())
    got = np.isfinite(z_hat)
    e = z_hat[got] - z_true[got]
    bias = float(np.median(e))
    ec = e - bias
    rmse = float(np.sqrt(np.mean(ec ** 2)))
    slope = float(np.polyfit(z_true[got], z_hat[got], 1)[0])
    r = float(np.corrcoef(z_true[got], z_hat[got])[0, 1])
    # trivial baseline: everyone gets the band middle
    e0 = np.median(z_true[got]) - z_true[got]
    rmse0 = float(np.sqrt(np.mean((e0 - np.median(e0)) ** 2)))
    print(f"BLIND BAND: n={got.sum():,}/{len(idx):,} px · "
          f"RMSE {rmse:.3f} m (trivial {rmse0:.3f}) · slope {slope:.3f} · "
          f"r {r:.3f} · bias {bias:+.2f} · "
          f"soundings median {np.median(n_used[got]):.0f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump({
        "n_px_banda": int(len(idx)), "n_px_estimados": int(got.sum()),
        "rmse_centrado": rmse, "rmse_trivial": rmse0,
        "pendiente": slope, "pearson": r, "sesgo_m": bias,
        "sondas_por_px_mediana": float(np.median(n_used[got])),
        "banda_m": [h_min - BAND_BELOW_M, h_min - 0.02],
        "marea_minima_muestreada_m": h_min,
        "ventana_inversion_d_m": [float(xc_s.min()), float(xc_s.max())],
        "inputs_sha": {"cube": seal._sha256(CUBE)},
        "duracion_s": round(time.time() - t0, 1),
    }, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    hb = ax.hexbin(z_true[got], z_hat[got] - bias, gridsize=40,
                   cmap="viridis", mincnt=3)
    lims = [np.percentile(z_true[got], 1), np.percentile(z_true[got], 99)]
    ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
    ax.set_xlabel("LiDAR z (tide frame, m)")
    ax.set_ylabel("z from inverted NDWI soundings (m)")
    ax.set_title(f"never-exposed band, {got.sum():,} px:\n"
                 f"RMSE {rmse:.2f} m (trivial {rmse0:.2f}) · "
                 f"slope {slope:.2f} · r {r:.2f}")
    ax.legend(fontsize=9)
    fig.colorbar(hb, ax=ax, label="px")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"))
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
