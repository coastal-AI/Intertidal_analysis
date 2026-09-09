"""
marea.py — MAREA end to end for any cube (campaign engine)
==========================================================

*Mudflat Altimetry via Remotely-Estimated tides from the Archive.*

One call takes a raw-band NDWI cube and returns the full MAREA product:

1. intertidal mask and per-pixel series from the water-frequency recipe;
2. archive-derived geometry (mouth = permanent sea touching the border,
   geodesic along-water distance s);
3. the interior-tide profile tau(s) by profiled Bernoulli likelihood (M2a),
   anchored at the mouth, applied ONLY beyond the demonstrated detection
   threshold (campaign mode: the per-site matched-null band is replaced by
   the 10-minute threshold measured against nulls at the reference sites —
   declared, and conservative);
4. elevation inversion per band with the corrected clock, standard guards;
5. hypsometric curve with an uncertainty band, excluding nothing silently.

Everything a cell writes is self-describing: result.json carries the tau
profile, the applied clocks, counts, and input hashes.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import geometry, seal
from . import tide_estimators as te
from .elevation import _fit_block

SG_GRID = (0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65)
CLEAR = (4, 5, 6, 7)
T_CHUNK = 40
TAU_DETECT_MIN = 10.0     # detection threshold demonstrated against the
                          # matched nulls of Villaviciosa/Scheldt; below it
                          # the operator stays at identity (conservative)


def extract(cube_path):
    """Streaming intertidal extraction from a raw-band cube (B03/B08/SCL)."""
    import xarray as xr

    ds = xr.open_dataset(cube_path)
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    T = ds.sizes[t_dim]
    H, W = ds.sizes["y"], ds.sizes["x"]
    dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])

    wet_n = np.zeros((H, W), np.int32)
    clr_n = np.zeros((H, W), np.int32)
    for i0 in range(0, T, T_CHUNK):
        sl = {t_dim: slice(i0, min(i0 + T_CHUNK, T))}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            wet = clear & ((g - n) / np.maximum(g + n, 1) > 0)
        wet_n += wet.sum(axis=0, dtype=np.int32)
        clr_n += clear.sum(axis=0, dtype=np.int32)
    wf = np.where(clr_n > 30, wet_n / np.maximum(clr_n, 1), np.nan)
    inter = np.isfinite(wf) & (wf > 0.10) & (wf < 0.90)
    sea = np.isfinite(wf) & (wf >= 0.90)
    keep = np.where(inter.ravel())[0]
    rows, cols = keep // W, keep % W

    Y = np.full((T, len(keep)), np.nan, np.float32)
    C = np.zeros((T, len(keep)), bool)
    for i0 in range(0, T, T_CHUNK):
        i1 = min(i0 + T_CHUNK, T)
        sl = {t_dim: slice(i0, i1)}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = np.where(g + n != 0, (g - n) / (g + n), np.nan)
        Y[i0:i1] = ndwi[:, rows, cols]
        C[i0:i1] = clear[:, rows, cols]

    import pyproj
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(crs_cube, 4326, always_xy=True)
    xs, ys = ds["x"].values, ds["y"].values
    lons, lats = tr.transform([xs.min(), xs.max()], [ys.min(), ys.max()])
    bbox = {"west": float(min(lons)), "south": float(min(lats)),
            "east": float(max(lons)), "north": float(max(lats)),
            "crs": "EPSG:4326"}
    return dict(Y=Y, C=C, keep=keep, dates=dates, shape=(H, W), sea=sea,
                inter=inter, bbox=bbox, crs_wkt=str(crs_cube.to_wkt()))


def invert_series(Y, C, h, lo=None, hi=None, mu_points=50, chunk=4000,
                  min_obs=8, min_b=0.15, max_b=2.5, max_a=2.5):
    """Per-pixel elevation from NDWI series under a level series ``h``.

    THE canonical inversion of the package: closed-form (a, b) sweep over a
    mu grid with the standard guards (adopted at prototype time; unbounded
    amplitudes exploded in frozen prediction). Every experiment and the
    campaign call this one function — five near-copies were folded into it
    at delivery time. Returns ``(z, sigma)``, NaN where the guards reject.
    """
    lo = float(np.min(h)) if lo is None else lo
    hi = float(np.max(h)) if hi is None else hi
    z = np.full(Y.shape[1], np.nan)
    sg_out = np.full(Y.shape[1], np.nan)
    grid = np.linspace(lo, hi, mu_points)
    for j in range(0, Y.shape[1], chunk):
        s = slice(j, min(j + chunk, Y.shape[1]))
        a, b, mu, sg, _, N = _fit_block(np.asarray(Y[:, s], np.float64),
                                        np.asarray(C[:, s], np.float64),
                                        h, grid, SG_GRID)
        ok = ((N >= min_obs) & (b > min_b) & (b < max_b)
              & (np.abs(a) < max_a))
        z[s] = np.where(ok, mu, np.nan)
        sg_out[s] = np.where(ok, sg, np.nan)
    return z, sg_out


_invert = invert_series          # backward-compatible internal alias


def build_bank(lat, lon, times, bank_taus=None, model="EOT20",
               tide_dir="tide_models"):
    """Shift bank + limb flags in ONE tide-model call.

    Every consumer used to make one ``model_tides`` call per shift (13+),
    each re-reading the constituent grids from disk — measured as ~80 % of
    a cell's wall time. All shifted instants go into a single prediction
    over the sorted unique times; each series is then a positional lookup.
    Returns ``(bank, rising)``.
    """
    import pandas as pd
    from eo_tides.model import model_tides
    from . import tide_estimators as te

    bank_taus = list(BANK_TAUS if bank_taus is None else bank_taus)
    shifted = [times - pd.Timedelta(minutes=float(tv)) for tv in bank_taus]
    limb = [times - pd.Timedelta(minutes=30),
            times + pd.Timedelta(minutes=30)]
    all_t = pd.DatetimeIndex(np.unique(np.concatenate(
        [s.values for s in shifted + limb])))
    h_all = model_tides(x=[lon], y=[lat], time=all_t, model=model,
                        directory=tide_dir, crs="EPSG:4326",
                        extrapolate=True, cutoff=np.inf,
                        parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)

    def at(tt):
        pos = np.searchsorted(all_t.values, tt.values)
        return h_all[np.minimum(pos, len(h_all) - 1)]

    bank = te.ShiftBank(bank_taus, [at(s) for s in shifted])
    rising = (at(limb[1]) - at(limb[0])) > 0
    return bank, rising


BANK_TAUS = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0,
             75.0, 90.0, 105.0, 120.0]


def reconstruct(cube_path, out_dir, name=None, n_bands=6, pixel_m=10.0,
                tau_detect_min=TAU_DETECT_MIN, model="EOT20",
                tide_dir="tide_models", verbose=True):
    """The full MAREA product for one cube. Writes ``out_dir``/{marea.npz,
    result.json, figure.png} and returns the result dict."""
    import pandas as pd
    from .net import use_system_certificates
    from . import overpass

    t0 = time.time()
    name = name or os.path.splitext(os.path.basename(cube_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    use_system_certificates()

    ex = extract(cube_path)
    H, W = ex["shape"]
    keep, sea, inter = ex["keep"], ex["sea"], ex["inter"]
    if verbose:
        print(f"[{name}] {len(keep):,} intertidal px", flush=True)
    if len(keep) < 3000:
        json.dump({"sitio": name, "estado": "sin_intermareal",
                   "n_px": int(len(keep))},
                  open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return None

    # geometry; a cell whose sea does not touch the border gets no operator
    try:
        seeds = geometry.mouth_seeds(sea)
        s_m = geometry.along_distance(sea | inter, seeds, pixel_m)
        s_km = s_m.ravel()[keep] / 1000.0
    except ValueError:
        s_km = np.full(len(keep), np.nan)

    times = overpass.get_overpass_times(
        ex["bbox"], ("2016-01-01", "2025-12-31"), verbose=False)
    have = np.array([s in times for s in ex["dates"]])
    if have.sum() < 80:
        json.dump({"sitio": name, "estado": "pocas_escenas",
                   "n_escenas": int(have.sum())},
                  open(os.path.join(out_dir, "result.json"), "w"), indent=1)
        return None
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in ex["dates"][have]])).tz_localize(None)
    Y = np.nan_to_num(ex["Y"][have], nan=0.0).astype(np.float64)
    C = (ex["C"][have]).astype(np.float64)
    wet = (C > 0) & (Y > 0)
    lat_c = 0.5 * (ex["bbox"]["south"] + ex["bbox"]["north"])
    lon_c = 0.5 * (ex["bbox"]["west"] + ex["bbox"]["east"])

    bank, _rising = build_bank(lat_c, lon_c, t_real, model=model,
                               tide_dir=tide_dir)
    h0 = bank.at(0.0)

    # interior tide (only with a usable mouth anchor)
    if np.isfinite(s_km).sum() > 5000:
        edges, centers, band_of = te.make_bands(s_km, n_bands)
        import yaml
        cfg2 = yaml.safe_load(open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "m2.yaml"), encoding="utf-8"))
        r2a = te.m2a_rasch(wet, C > 0, bank, band_of, centers, n_bands,
                           sigma0=cfg2["sigma0_m"],
                           sg_grid=tuple(cfg2["sigma_perfil_m"]),
                           tau_bounds=(-45.0, 120.0),
                           z_points=cfg2["z_puntos"],
                           rng=np.random.default_rng(cfg2["seed"] + 31),
                           max_px_band=cfg2["max_px_banda"]["m2a"])
        tau = np.asarray(r2a["tau"], float)
    else:
        centers = np.array([0.0])
        band_of = np.zeros(len(keep), int)
        tau = np.array([0.0])
    tau_used = np.where(np.abs(tau) > tau_detect_min, tau, 0.0)
    if verbose:
        print(f"[{name}] tau={np.round(tau, 1)} -> "
              f"applied {np.round(tau_used, 1)}", flush=True)

    lo, hi = float(h0.min()), float(h0.max())
    z = np.full(len(keep), np.nan)
    sg = np.full(len(keep), np.nan)
    for k in range(len(centers)):
        cols = np.where(band_of == k)[0]
        if not len(cols):
            continue
        h_k = bank.at(float(tau_used[k])) if tau_used[k] else h0
        z[cols], sg[cols] = _invert(Y[:, cols], C[:, cols], h_k, lo, hi)

    # hypsometry with uncertainty (sigma as the per-pixel jitter)
    from .hypsometry import curve_with_uncertainty
    hyp = curve_with_uncertainty(z, pixel_m=pixel_m, sigma_z=sg)

    np.savez_compressed(
        os.path.join(out_dir, "marea.npz"),
        z=z.astype(np.float32), sigma=sg.astype(np.float32),
        s_km=s_km.astype(np.float32), band=band_of.astype(np.int16),
        keep=keep, shape=np.array([H, W]),
        tau_min=tau, tau_usado_min=tau_used,
        hyp_z=hyp["z"], hyp_area=hyp["area_km2"],
        hyp_lo=hyp["area_lo"], hyp_hi=hyp["area_hi"])
    result = {
        "sitio": name, "estado": "ok",
        "n_escenas": int(have.sum()), "n_px": int(len(keep)),
        "n_px_cota": int(np.isfinite(z).sum()),
        "s_max_km": (float(np.nanmax(s_km))
                     if np.isfinite(s_km).any() else None),
        "centros_km": np.asarray(centers).tolist(),
        "tau_min": np.asarray(tau).tolist(),
        "tau_usado_min": np.asarray(tau_used).tolist(),
        "con_operador": bool((tau_used != 0).any()),
        "hipsometria": {"integral": hyp["integral"],
                        "area_total_km2": hyp["area_total_km2"]},
        "rango_marea_m": [lo, hi],
        "inputs_sha": {"cube": seal._sha256(cube_path)},
        "duracion_s": round(time.time() - t0, 1),
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"),
              indent=1)

    import matplotlib
    # headless campaigns need Agg, but inside a notebook this call would
    # kill the inline backend for every later cell — switch only if no
    # interactive backend is already live
    if "inline" not in matplotlib.get_backend():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    img = np.full(H * W, np.nan, np.float32)
    img[keep] = z
    im = ax[0].imshow(img.reshape(H, W), cmap="viridis")
    ax[0].set_title(f"{name}: cota MAREA (m)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    plt.colorbar(im, ax=ax[0], shrink=0.8)
    ax[1].plot(centers, tau, "o-", color="C0", label="medido")
    ax[1].plot(centers, tau_used, "s--", color="C2", label="applied")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("s (km)"); ax[1].set_ylabel("tau (min)")
    ax[1].legend(fontsize=8); ax[1].set_title("marea interior")
    ax[2].fill_betweenx(hyp["z"], hyp["area_lo"], hyp["area_hi"],
                        alpha=0.25, color="C0")
    ax[2].plot(hyp["area_km2"], hyp["z"], color="C0")
    ax[2].set_xlabel("area acumulada (km2)"); ax[2].set_ylabel("cota (m)")
    ax[2].set_title(f"hipsometria (HI={hyp['integral']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure.png"), dpi=130)
    plt.close(fig)
    if verbose:
        print(f"[{name}] listo en {time.time()-t0:.0f} s -> {out_dir}",
              flush=True)
    return result
