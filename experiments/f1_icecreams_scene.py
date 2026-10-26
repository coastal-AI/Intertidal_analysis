"""Apply the (reproduced) ICE CREAMS model to one low-tide scene of a site.

Pipeline-native end to end:

1. the scene is chosen the way the paper canvas is chosen — every date
   scored by SCL bad-pixel fraction, only near-cloud-free ones kept, the
   lowest EOT20 tide among them wins (emerged flats are the model's stated
   domain);
2. the 12 L2A bands come from ONE idempotent openEO batch job at 10 m;
3. features are built exactly as ICE_CREAMS/apply_ICECREAMS.py builds them:
   raw reflectances, the same 12 min-max standardised PER PIXEL across the
   spectrum, NDVI and NDWI — with the post-2022 processing-baseline offset
   (-1000) detected from the data rather than assumed;
4. output: ``products_{site}/vegetation_{date}.tif`` (class, probability,
   seagrass % cover) and a legend figure.

Run:  python -m experiments.f1_icecreams_scene --site arousa [--date YYYY-MM-DD]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BANDS12 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
           "B8A", "B09", "B11", "B12")
CLASSES = {1: "Bare Sand", 2: "Bare Sediment", 3: "Chlorophyta",
           4: "Magnoliopsida (seagrass)", 5: "Microphytobenthos",
           6: "Phaeophyceae", 7: "Rhodophyta", 8: "Water",
           9: "Xanthophyceae"}
PALETTE = {1: "#e8d8a0", 2: "#b08968", 3: "#7fc97f", 4: "#1b7837",
           5: "#c8a24b", 6: "#8c510a", 7: "#c51b7d", 8: "#2a6f97",
           9: "#a6d854"}
KNOWN_CACHES = {"villaviciosa": "ndwi_cube_villaviciosa_grande_10y.nc",
                "santander": "ndwi_cube_santander_2023_2025.nc",
                "arousa": "ndwi_cube_arousa_2023-2025_20m.nc"}


def pick_low_tide_clear_date(cube, aoi, max_bad=0.01, months=None):
    """Same rule as viz.scene_canvas — clear and covered first, then tide —
    plus an optional growing-season filter: Zostera noltei dies back in
    winter (Davies et al. 2024 show Bourgneuf at near-zero every winter,
    with northern maxima late Aug - mid Sep), so a vegetation scene picked
    on tide alone lands in February and maps the die-back, not the meadow.
    """
    from pyintertidal.cube import open_cube
    from pyintertidal.water import BAD_CLASSES
    import pyintertidal as pit

    ds, _, tname = open_cube(cube.cache_path, cube.bands)
    try:
        n = ds.sizes[tname]
        bad = np.empty(n, np.float32)
        nodata = np.empty(n, np.float32)
        for i in range(n):
            tile = np.nan_to_num(
                ds["SCL"].isel({tname: i}).values[::6, ::6], nan=0.0)
            bad[i] = np.isin(tile, BAD_CLASSES).mean()
            nodata[i] = (tile == 0).mean()
    finally:
        ds.close()
    # a swath-edge acquisition looks "clear" because nodata is not cloud —
    # the 2023-08-07 Arousa pick covered only the NE corner of the frame.
    # Demand the scene actually covers the AOI before judging its clouds.
    ok = np.flatnonzero((bad <= max_bad) & (nodata <= 0.02))
    if not len(ok):
        ok = np.argsort(bad)[:5]
    dates = [cube.dates[int(i)] for i in ok]
    if months:
        seasonal = [d for d in dates if int(d[5:7]) in months]
        if seasonal:
            dates = seasonal
        else:
            print("no clear scene inside the growing season — "
                  "falling back to the whole year")
    lon, lat = aoi.centroid
    tides = pit.TideService(model="EOT20", directory="./tide_models",
                            location=(lat, lon))
    h = tides.heights_for(aoi, dates)
    dates = [d for d in dates if np.isfinite(h.get(d, np.nan))]
    date = min(dates, key=lambda d: h[d])
    print(f"chosen scene: {date} (tide {h[date]:+.2f} m, "
          f"{len(ok)} near-cloud-free candidates)")
    return date


def download_bands(conn, aoi, date, out_path):
    """The 12 bands of one date at 10 m — one idempotent batch job."""
    if os.path.exists(out_path):
        print(f"bands cached: {out_path}")
        return out_path
    cube = conn.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi.bbox,
        temporal_extent=[date, date], bands=list(BANDS12),
        max_cloud_cover=100)
    cube = cube.resample_spatial(resolution=10, method="near")
    job = cube.save_result(format="GTiff").create_job(
        title=f"icecreams_bands_{date}")
    job.start_and_wait()
    assets = job.get_results().get_assets()
    if not assets:
        raise RuntimeError("no assets")
    assets[0].download(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="arousa")
    ap.add_argument("--date", default=None)
    ap.add_argument("--months", type=int, nargs="+", default=None,
                    help="growing-season months for the scene pick, "
                         "e.g. --months 7 8 9 10")
    ap.add_argument("--model",
                    default=os.path.join("checkpoints", "icecreams",
                                         "ICECREAMS_repro.pkl"))
    ap.add_argument("--mask_tif", default=None,
                    help="GeoTIFF whose FINITE pixels define the intertidal "
                         "domain (e.g. an HSR elevation product), used when "
                         "the site has no marea.npz")
    args = ap.parse_args()

    import rasterio
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors, patches as mpatches
    from fastai.tabular.all import load_learner

    import pyintertidal as pit
    from pyintertidal.cube import SentinelCube

    aoi = pit.sites.get(args.site)
    cube = SentinelCube(aoi, ("2016-01-01", "2026-01-01"), water="ndwi",
                        cache_path=KNOWN_CACHES[args.site])
    date = args.date or pick_low_tide_clear_date(cube, aoi,
                                                 months=args.months)

    os.makedirs("scenes_veg", exist_ok=True)
    tif = os.path.join("scenes_veg", f"bands12_{args.site}_{date}.tif")
    conn = pit.scenes.connect() if not os.path.exists(tif) else None
    download_bands(conn, aoi, date, tif)

    with rasterio.open(tif) as src:
        data = np.ma.filled(src.read(masked=True).astype(np.float32),
                            np.nan)
        transform, crs = src.transform, src.crs
    H, W = data.shape[1:]
    print(f"bands raster: {data.shape}")

    # processing-baseline offset: post-2022 products carry +1000 in the DN.
    # Water NIR reflectance is a few hundredths at most, so its low
    # percentile sits near 100 without the offset and near 1100 with it.
    b08 = data[BANDS12.index("B08")]
    p1 = float(np.nanpercentile(b08, 1))
    if p1 > 600:
        data = data - 1000.0
        print(f"baseline offset detected (B08 p1={p1:.0f}) -> -1000")
    else:
        print(f"no baseline offset (B08 p1={p1:.0f})")

    flat = data.reshape(len(BANDS12), -1)
    valid = np.isfinite(flat).all(axis=0)
    N = flat.shape[1]
    learn = load_learner(args.model)

    # ICE CREAMS is an EMERGED-INTERTIDAL model: on dry land every
    # vegetated pixel reads "seagrass". So the model is applied ONLY to
    # the intertidal domain the pipeline itself measured — the indices in
    # marea.npz, reprojected from the cube grid to this scene's grid —
    # exactly as the authors mask their SAFE scenes to a study vector.
    # (This also cuts the work from ~12 M pixels to a few hundred k.)
    marea_npz = os.path.join(f"products_{args.site}_marea", "marea.npz")
    if os.path.exists(marea_npz):
        from rasterio.warp import reproject, Resampling as _Rs
        mz = np.load(marea_npz)
        Hm, Wm = (int(v) for v in mz["shape"])
        inter = np.zeros(Hm * Wm, np.uint8)
        inter[mz["keep"]] = 1
        with rasterio.open(tif) as src10:
            inter10 = np.zeros((H, W), np.uint8)
            reproject(inter.reshape(Hm, Wm), inter10,
                      src_transform=cube.grid[0], src_crs=cube.grid[1],
                      dst_transform=src10.transform, dst_crs=src10.crs,
                      resampling=_Rs.nearest)
        domain = valid & (inter10.ravel() == 1)
        print(f"classifying the measured intertidal only: "
              f"{int(domain.sum()):,} of {N:,} px")
    elif args.mask_tif and os.path.exists(args.mask_tif):
        from rasterio.warp import reproject, Resampling as _Rs
        with rasterio.open(args.mask_tif) as ms:
            msk = np.isfinite(np.ma.filled(
                ms.read(1, masked=True).astype(np.float32),
                np.nan)).astype(np.uint8)
            mtr, mcrs = ms.transform, ms.crs
        with rasterio.open(tif) as src10:
            inter10 = np.zeros((H, W), np.uint8)
            reproject(msk, inter10, src_transform=mtr, src_crs=mcrs,
                      dst_transform=src10.transform, dst_crs=src10.crs,
                      resampling=_Rs.nearest)
        domain = valid & (inter10.ravel() == 1)
        print(f"classifying the {args.mask_tif} domain: "
              f"{int(domain.sum()):,} of {N:,} px")
    else:
        domain = valid
        print("no marea.npz found — classifying the WHOLE frame: "
              "land classes are out-of-domain noise")
    domain_idx = np.flatnonzero(domain)

    klass = np.full(N, -1, np.int16)
    prob = np.full(N, np.nan, np.float32)
    spc = np.zeros(N, np.float32)
    CH = 1_500_000
    for a in range(0, len(domain_idx), CH):
        sl = domain_idx[a:a + CH]
        blk = flat[:, sl]
        cols = {f"Reflectance_{b}": blk[i]
                for i, b in enumerate(BANDS12)}
        df = pd.DataFrame(cols)
        # per-pixel min-max across the spectrum: the spectral SHAPE, free
        # of illumination — the trick that lets one tabular model travel
        vmin, vmax = blk.min(axis=0), blk.max(axis=0)
        span = np.maximum(vmax - vmin, 1e-6)
        for i, b in enumerate(BANDS12):
            df[f"Reflectance_Stan_{b}"] = (blk[i] - vmin) / span
        red, nir, grn = (blk[BANDS12.index(k)]
                         for k in ("B04", "B08", "B03"))
        with np.errstate(invalid="ignore", divide="ignore"):
            df["NDVI"] = (nir - red) / (nir + red)
            df["NDWI"] = (grn - nir) / (grn + nir)
        df = df.fillna(0)
        dl = learn.dls.test_dl(df, bs=4096)
        preds, _ = learn.get_preds(dl=dl)
        klass[sl] = preds.argmax(axis=1).numpy().astype(np.int16) + 1
        prob[sl] = preds.max(axis=1).values.numpy().astype(np.float32)
        ndvi = df["NDVI"].to_numpy()
        s = np.clip(172.06 * ndvi - 22.18, 0, 100).astype(np.float32)
        s[klass[sl] != 4] = 0.0    # seagrass % cover only where seagrass
        spc[sl] = s
        print(f"  classified {min(a + CH, len(domain_idx)):,}"
              f"/{len(domain_idx):,} px", flush=True)

    klass[~valid] = -1
    prob[~valid] = np.nan
    out_dir = f"products_{args.site}"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"vegetation_{date}.tif")
    with rasterio.open(
            out, "w", driver="GTiff", height=H, width=W, count=3,
            dtype="float32", transform=transform, crs=crs,
            compress="deflate") as dst:
        dst.write(klass.reshape(H, W).astype(np.float32), 1)
        dst.write(prob.reshape(H, W), 2)
        dst.write(spc.reshape(H, W), 3)
        dst.set_band_description(1, "ICE CREAMS class")
        dst.set_band_description(2, "class probability")
        dst.set_band_description(3, "seagrass cover %")

    km = klass.reshape(H, W).astype(float)
    km[km < 1] = np.nan
    # context: the scene's own water painted as the Water class, so the
    # classified flats read against the ría instead of floating on white
    grn = flat[BANDS12.index("B03")].reshape(H, W)
    nir = flat[BANDS12.index("B08")].reshape(H, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        wat = (grn - nir) / (grn + nir) > 0
    km = np.where(np.isnan(km) & wat & valid.reshape(H, W), 8.0, km)
    cmap = mcolors.ListedColormap([PALETTE[k] for k in sorted(PALETTE)])
    norm = mcolors.BoundaryNorm(np.arange(0.5, 10.5), cmap.N)
    fig, ax = plt.subplots(figsize=(12, 11), dpi=150)
    ax.imshow(km, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"ICE CREAMS · {aoi.name} · {date}", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[mpatches.Patch(color=PALETTE[k], label=CLASSES[k])
                       for k in sorted(CLASSES)],
              loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    png = os.path.join(out_dir, f"vegetation_{date}.png")
    fig.savefig(png, bbox_inches="tight")
    counts = pd.Series(klass[klass > 0]).map(CLASSES).value_counts()
    print("class counts (classified intertidal px at 10 m):")
    print(counts.to_string())

    # ── the paper's own figure: Seagrass Cover (%) over true colour ──────
    # (Davies et al. 2024, Fig. 5): SC = 172.06*NDVI - 22.18 on seagrass
    # pixels, only SC > 20 % kept; extent = sum(SC * 100 m2); uncertainty
    # tau = (1 - mean p) / Ga with Ga = 0.815 from their binary validation.
    rgb = np.dstack([flat[BANDS12.index(b)].reshape(H, W)
                     for b in ("B04", "B03", "B02")])
    lo_, hi_ = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo_) / max(hi_ - lo_, 1e-6), 0, 1) ** (1 / 1.15)
    rgb = np.nan_to_num(rgb, nan=1.0)

    sc = spc.reshape(H, W).astype(float)
    sc[(klass.reshape(H, W) != 4) | (sc <= 20.0)] = np.nan
    # each pixel is 100 m2; SC% / 100 * 100 m2 = covered m2 -> just sum SC
    extent_km2 = float(np.nansum(sc)) / 1e6
    p_mean = float(np.nanmean(prob[klass == 4])) if (klass == 4).any() \
        else float("nan")
    tau_unc = (1.0 - p_mean) / 0.815

    fig2, ax2 = plt.subplots(figsize=(12, 11), dpi=150)
    ax2.imshow(rgb)
    im2 = ax2.imshow(sc, cmap="viridis", vmin=0, vmax=100,
                     interpolation="nearest")
    cb = fig2.colorbar(im2, ax=ax2, shrink=0.7, pad=0.015)
    cb.set_label("Seagrass Cover (%)", fontsize=10)
    ax2.set_title(f"Seagrass cover · {aoi.name} · {date} · "
                  f"{extent_km2:.3f} km² (τ = {tau_unc:.2f})", loc="left")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig2.tight_layout()
    png2 = os.path.join(out_dir, f"seagrass_cover_{date}.png")
    fig2.savefig(png2, bbox_inches="tight")
    print(f"seagrass extent (SC>20%): {extent_km2:.3f} km² · "
          f"mean p {p_mean:.3f} · tau {tau_unc:.2f}")
    print(f"-> {out}\n-> {png}\n-> {png2}")


if __name__ == "__main__":
    main()
