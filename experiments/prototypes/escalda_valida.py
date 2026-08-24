"""The imagery-lag adapter gets its RMSE — scored against a gauge it never saw.

Everything before this was indirect. The adapter — tidal lag from flooded-area
ranks, no instruments anywhere in its calibration — was built at Villaviciosa,
where no gauge exists to score it in metres. The Westerschelde closes that:
vast flats between Vlissingen and Terneuzen, an archive over them, and an
inner gauge whose record the adapter never touches until scoring.

Protocol, in the order the data is allowed to flow:

  1. stream the cube once to find the intertidal pixels (wet sometimes, not
     always), and once more for the flooded fraction per scene per band of
     the along-estuary coordinate;
  2. estimate the lag per band by rank correlation against EOT20 at the
     MOUTH, exactly the Villaviciosa recipe — imagery plus ocean model, no
     gauges;
  3. read the lag the profile implies at Terneuzen's position;
  4. only then open the Terneuzen record, and score three predictions on the
     same test window every other row of the comparison used:

        EOT20(t)            what the pipeline does today        (was 0.414)
        EOT20(t - tau_img)  the adapter, calibrated from imagery
        EOT20(t - tau_gau)  the ceiling: same correction with the lag the
                            gauges measured (+22.8 min) — the best a
                            lag-only fix can possibly do here

The two questions, in order of importance: does tau_img land near the gauge
truth of +22.8 min, and does the adapter close a useful part of the gap
between EOT20 and the lag-only ceiling. It cannot reach the full operator
(0.418→0.112 needed the gauge's surge and gains); beating plain EOT20 with
zero instruments is the claim on trial.
"""
import os
import sys
import json

sys.path.insert(0, r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from pyintertidal.net import use_system_certificates
import pyintertidal as pit

SC = (r"C:\Users\Jorge\AppData\Local\Temp\claude"
      r"\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8"
      r"\scratchpad")
CUBE = "ndwi_cube_escalda_2023-2025_20m.nc"
BBOX = ("2023-01-01", "2025-12-31")
AOI = {"west": 3.55, "south": 51.33, "east": 3.85, "north": 51.46}
MOUTH = (51.443, 3.597)          # Vlissingen
INNER = (51.336, 3.820)          # Terneuzen
CLEAR = (4, 5, 6, 7)
TAU_GRID = np.arange(-60, 91, 5)
MIN_CLEAR = 0.25
N_BANDS = 5
T_CHUNK = 40
TAU_GAUGE = 22.8                 # measured Vlissingen->Terneuzen, M2


def tide_at(lat, lon, t):
    from eo_tides.model import model_tides
    return model_tides(x=[lon], y=[lat], time=t, model="EOT20",
                       directory="tide_models", crs="EPSG:4326",
                       extrapolate=True, cutoff=np.inf,
                       parallel=False).reset_index().sort_values(
        "time")["tide_height"].to_numpy(float)


def lag_for(frac, h_cols):
    rho = np.array([stats.spearmanr(frac, h).statistic for h in h_cols])
    j = int(np.nanargmax(rho))
    if 0 < j < len(TAU_GRID) - 1:
        y0, y1, y2 = rho[j - 1], rho[j], rho[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        return float(TAU_GRID[j] + np.clip(off, -1, 1) * 5.0), float(rho[j])
    return float(TAU_GRID[j]), float(rho[j])


def load_gauge(code):
    import glob
    raw = []
    for h in sorted(glob.glob(os.path.join(SC, f"ioc_{code}_*.json"))):
        raw += json.load(open(h))
    df = pd.DataFrame(raw)
    if "sensor" in df:
        df = df[df["sensor"] == df["sensor"].value_counts().idxmax()]
    return (pd.DataFrame({"time": pd.to_datetime(df["stime"]),
                          "level_m": pd.to_numeric(df["slevel"],
                                                   errors="coerce")})
            .dropna().sort_values("time").drop_duplicates("time"))


def main():
    use_system_certificates()
    ds = xr.open_dataset(CUBE)
    t_dim = [k for k in ds["B03"].dims if k not in ("x", "y")][0]
    T = ds.sizes[t_dim]
    H, W = ds.sizes["y"], ds.sizes["x"]
    print(f"cubo: {T} escenas, {H}x{W} a 20 m", flush=True)

    dates = np.array([str(np.datetime64(v, "D")) for v in ds[t_dim].values])
    times = pit.overpass.get_overpass_times(
        {**AOI, "crs": "EPSG:4326"}, BBOX, verbose=False)
    have = np.array([s in times for s in dates])
    t_real = pd.DatetimeIndex(pd.to_datetime(
        [times[s] for s in dates[have]])).tz_localize(None)
    print(f"{int(have.sum())} escenas con hora real", flush=True)

    # ── pass 1: who is intertidal ────────────────────────────────────────
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
    print(f"pixeles intermareales: {int(inter.sum()):,} "
          f"({100*inter.mean():.0f} % del recorte)", flush=True)

    # along-estuary coordinate: easting, west (mouth) to east (inner)
    xs = ds["x"].values
    x2d = np.broadcast_to(xs[None, :], (H, W))
    import pyproj
    crs_cube = pyproj.CRS.from_wkt(ds["crs"].attrs.get("crs_wkt")
                                   or ds["crs"].attrs.get("spatial_ref"))
    tr = pyproj.Transformer.from_crs(4326, crs_cube, always_xy=True)
    x_mouth = tr.transform(MOUTH[1], MOUTH[0])[0]
    x_inner = tr.transform(INNER[1], INNER[0])[0]
    s_px = (x2d - x_mouth) / 1000.0            # km east of Vlissingen
    s_inner = (x_inner - x_mouth) / 1000.0
    print(f"Terneuzen esta a s = {s_inner:.1f} km de Vlissingen en este eje",
          flush=True)

    # ── pass 2: flooded fraction per scene per band ──────────────────────
    s_vals = s_px[inter]
    qs = np.quantile(s_vals, np.linspace(0, 1, N_BANDS + 1))
    qs[0] -= 1e-9
    band_of = np.full((H, W), -1, np.int8)
    for k, (a, b) in enumerate(zip(qs, qs[1:])):
        band_of[inter & (s_px > a) & (s_px <= b)] = k

    wet_b = np.zeros((T, N_BANDS), np.int32)
    clr_b = np.zeros((T, N_BANDS), np.int32)
    for i0 in range(0, T, T_CHUNK):
        i1 = min(i0 + T_CHUNK, T)
        sl = {t_dim: slice(i0, i1)}
        g = ds["B03"].isel(**sl).values.astype(np.float32)
        n = ds["B08"].isel(**sl).values.astype(np.float32)
        scl = ds["SCL"].isel(**sl).values
        clear = np.isin(scl, CLEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            wet = clear & ((g - n) / np.maximum(g + n, 1) > 0)
        for k in range(N_BANDS):
            m = band_of == k
            wet_b[i0:i1, k] = wet[:, m].sum(axis=1)
            clr_b[i0:i1, k] = clear[:, m].sum(axis=1)
    n_band = np.array([(band_of == k).sum() for k in range(N_BANDS)])
    frac = np.where(clr_b > MIN_CLEAR * n_band[None, :],
                    wet_b / np.maximum(clr_b, 1), np.nan)
    frac = frac[have]

    # ── 2. the lag profile, imagery + mouth model only ───────────────────
    print("\nbarrido de tau…", flush=True)
    h_by_tau = np.array([
        tide_at(MOUTH[0], MOUTH[1],
                t_real - pd.Timedelta(minutes=float(tv)))
        for tv in TAU_GRID])
    print(f"\n{'banda (km este)':18s} {'px':>8s} {'esc':>5s} "
          f"{'tau (min)':>10s} {'rho':>7s}")
    print("-" * 54)
    centers, taus, ws = [], [], []
    for k, (a, b) in enumerate(zip(qs, qs[1:])):
        fk = frac[:, k]
        m = np.isfinite(fk)
        if m.sum() < 60:
            continue
        tk, rk = lag_for(fk[m], h_by_tau[:, m])
        print(f"{f'{a:.1f} a {b:.1f}':18s} {n_band[k]:8,d} "
              f"{int(m.sum()):5d} {tk:+10.1f} {rk:7.3f}")
        centers.append(0.5 * (a + b))
        taus.append(tk)
        ws.append(m.sum())
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=90.0,
                             out_of_bounds="clip")
    iso.fit(centers, taus, sample_weight=np.sqrt(ws))
    tau_img = float(iso.predict([s_inner])[0])
    print(f"\ntau de IMAGEN en la posicion de Terneuzen: {tau_img:+.1f} min")
    print(f"tau medido con MAREOGRAFOS (M2):            +{TAU_GAUGE:.1f} min")

    # ── 3. the gauge, opened only now ────────────────────────────────────
    g = load_gauge("trnz")
    lo, hi = g["time"].min(), g["time"].max()
    split = lo + (hi - lo) * 0.65
    tt = pd.date_range(split, hi, freq="10min")
    x = g["time"].values.astype("int64").astype(float)
    truth = np.interp(tt.values.astype("int64").astype(float), x,
                      g["level_m"].to_numpy(float),
                      left=np.nan, right=np.nan)

    def rmse(pred):
        m = np.isfinite(pred) & np.isfinite(truth)
        d = pred[m] - truth[m]
        return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))

    rows = {}
    for lab, tau in (("EOT20 tal cual (pipeline hoy)", 0.0),
                     (f"ADAPTADOR DE IMAGEN (tau={tau_img:.0f}m)", tau_img),
                     (f"techo: retardo de mareografos ({TAU_GAUGE:.0f}m)",
                      TAU_GAUGE)):
        p = tide_at(INNER[0], INNER[1],
                    tt - pd.Timedelta(minutes=float(tau)))
        rows[lab] = rmse(p)
    print(f"\nRMSE EN TERNEUZEN, ventana de prueba "
          f"({split.date()} a {hi.date()}):")
    for lab, v in rows.items():
        print(f"  {lab:42s} {v:.4f} m")
    base = rows["EOT20 tal cual (pipeline hoy)"]
    adap = [v for k, v in rows.items() if "ADAPTADOR" in k][0]
    ceil = [v for k, v in rows.items() if "techo" in k][0]
    print(f"\n  mejora del adaptador: {base-adap:+.4f} m "
          f"(el techo de solo-retardo da {base-ceil:+.4f})")
    if base - adap > 0.02 and abs(tau_img - TAU_GAUGE) < 15:
        print("\nVALIDADO: el adaptador acierta el retardo desde la imagen y")
        print("mejora el nivel sin ningun instrumento.")
    elif abs(tau_img - TAU_GAUGE) < 15:
        print("\nEl retardo se acierta pero no rinde en RMSE aqui.")
    else:
        print("\nEl retardo de imagen NO coincide con el de mareografos:")
        print("la calibracion desde imagen no vale en este estuario tal cual.")

    json.dump({"tau_imagen": tau_img, "tau_gauge": TAU_GAUGE,
               "bands": {"centers": centers, "taus": taus},
               "rmse": rows, "s_inner_km": float(s_inner)},
              open(os.path.join(SC, "escalda_valida.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
