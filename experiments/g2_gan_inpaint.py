"""Experiment: cloud inpainting with TideGAN, flagged as synthetic.

THE RULE THIS RESPECTS: generated pixels never enter MAREA estimation or
validation. Inpainting serves the generative dataset and visualisation, and
its coverage gain is reported SEPARATELY from real coverage — a filled
product carries a synthetic-fraction flag, so nobody can mistake imputed
pixels for measured ones.

Protocol, pre-registered
------------------------
1. **Hold-out QC first, filling second.** Take the archive's clear scenes
   (clear fraction > 0.95). For each, borrow a REAL cloud mask from another
   date (matched by cloud fraction 0.2-0.7), pretend those pixels are
   missing, generate the scene at its own tide from the fixed reference, and
   score ONLY the hidden-but-actually-observed pixels:

   * MAE in the trained channel(s);
   * wet/dry agreement (NDWI > 0) — the number MAREA would care about;
   * waterline displacement: median distance (m) between the observed and
     generated NDWI zero-contour inside the hidden region.

   The BASELINE to beat is climatology: filling the hidden pixels with the
   per-pixel median NDWI of the 5 clear scenes nearest in tide. A generator
   that cannot beat a tide-binned median has learned nothing useful.

2. **Gate.** Only if the generator beats the climatology baseline on wet/dry
   agreement does the filling pass below get produced at all.

3. **Filling.** For every usable scene, composite: real clear pixels kept
   verbatim, cloudy pixels replaced by the generation at that scene's tide;
   write per-scene synthetic fraction and report coverage before/after
   against the archive's geometric ceiling.

Run:  python -m experiments.g2_gan_inpaint --checkpoint checkpoints/....pth
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from scipy import ndimage

from experiments.tidegan import Archive, Generator


def pad16(a):
    """Reflect-pad (C, H, W) so H and W are divisible by 16."""
    _, H, W = a.shape
    ph, pw = (-H) % 16, (-W) % 16
    return np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="reflect"), H, W


def generate_full(G, archive, channels, ref_idx, tide_m, device):
    """One full-frame generation at the given tide, from the fixed ref."""
    ref = np.stack([np.nan_to_num(
        np.asarray(archive.channels[c][ref_idx], np.float32), nan=0.0)
        for c in channels])
    ref, H, W = pad16(ref)
    cond = torch.from_numpy(archive.condition(tide_m)).unsqueeze(0).to(device)
    x = torch.from_numpy(ref).unsqueeze(0).to(device)
    with torch.no_grad():
        out = G(x, cond)[0].cpu().numpy()
    return out[:, :H, :W]


def waterline_displacement_m(obs_ndwi, gen_ndwi, hidden, pixel_m=10.0):
    """Median distance from generated to observed waterline, hidden px only."""
    obs_wet = obs_ndwi > 0
    gen_wet = gen_ndwi > 0
    obs_edge = obs_wet ^ ndimage.binary_erosion(obs_wet)
    gen_edge = gen_wet ^ ndimage.binary_erosion(gen_wet)
    if not (obs_edge & hidden).any() or not gen_edge.any():
        return float("nan")
    d_to_gen = ndimage.distance_transform_edt(~gen_edge)
    return float(np.median(d_to_gen[obs_edge & hidden]) * pixel_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--site", default="villaviciosa")
    ap.add_argument("--n_holdout", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    channels = ck["channels"]
    G = Generator(in_channels=len(channels)).to(device)
    G.load_state_dict(ck["generator_state"])
    G.eval()
    print(f"checkpoint {args.checkpoint} · channels {channels}")

    archive = Archive(f"gan_archive_{args.site}")
    cf = archive.clear_fraction
    tides = archive.tides
    ok = np.isfinite(tides)

    clear_scenes = [i for i in range(len(cf)) if cf[i] > 0.95 and ok[i]]
    donor_masks = [i for i in range(len(cf)) if 0.3 < cf[i] < 0.8 and ok[i]]
    holdout = rng.choice(clear_scenes,
                         min(args.n_holdout, len(clear_scenes)),
                         replace=False)
    print(f"{len(clear_scenes)} clear scenes, {len(donor_masks)} mask "
          f"donors, {len(holdout)} in the hold-out")

    # fixed reference: same rule as training (clearest of low-tide quartile)
    q = np.quantile(tides[ok], 0.25)
    low = [i for i in np.flatnonzero(ok) if tides[i] <= q]
    ref_idx = max(low, key=lambda i: cf[i])
    print(f"reference scene: {archive.dates[ref_idx]} "
          f"(tide {tides[ref_idx]:+.2f} m, clear {cf[ref_idx]:.2f})")

    ndwi_ch = channels.index("ndwi") if "ndwi" in channels else None
    rows = []
    for k, t in enumerate(holdout):
        donor = int(rng.choice(donor_masks))
        cloudy_donor = np.asarray(archive.clear[donor]) == 0
        hidden = cloudy_donor & (np.asarray(archive.clear[t]) == 1)
        if hidden.mean() < 0.02:
            continue
        gen = generate_full(G, archive, channels, ref_idx,
                            float(tides[t]), device)
        obs = np.stack([np.nan_to_num(
            np.asarray(archive.channels[c][t], np.float32), nan=0.0)
            for c in channels])
        mae = float(np.abs(gen - obs)[:, hidden].mean())
        row = {"date": archive.dates[t], "tide_m": round(float(tides[t]), 3),
               "hidden_pct": round(100 * float(hidden.mean()), 1),
               "mae": round(mae, 4)}
        if ndwi_ch is not None:
            g_n, o_n = gen[ndwi_ch], obs[ndwi_ch]
            row["wetdry"] = round(float(
                ((g_n > 0) == (o_n > 0))[hidden].mean()), 4)
            # climatology baseline: median NDWI of 5 clear scenes nearest
            # in tide (excluding the scene itself)
            others = [i for i in clear_scenes if i != t]
            near = sorted(others,
                          key=lambda i: abs(tides[i] - tides[t]))[:5]
            clim = np.median(np.stack(
                [np.nan_to_num(np.asarray(archive.channels["ndwi"][i],
                                          np.float32), nan=0.0)
                 for i in near]), axis=0)
            row["wetdry_climatology"] = round(float(
                ((clim > 0) == (o_n > 0))[hidden].mean()), 4)
            row["waterline_disp_m"] = round(
                waterline_displacement_m(o_n, g_n, hidden), 1)
        rows.append(row)
        print(f"  [{k + 1}/{len(holdout)}] {row}", flush=True)

    out = {"checkpoint": args.checkpoint, "channels": channels,
           "reference": archive.dates[ref_idx], "holdout": rows}
    if rows and ndwi_ch is not None:
        wd = np.array([r["wetdry"] for r in rows])
        wc = np.array([r["wetdry_climatology"] for r in rows])
        out["summary"] = {
            "median_wetdry_gan": round(float(np.median(wd)), 4),
            "median_wetdry_climatology": round(float(np.median(wc)), 4),
            "gan_beats_climatology": bool(np.median(wd) > np.median(wc)),
            "median_waterline_disp_m": round(float(np.nanmedian(
                [r["waterline_disp_m"] for r in rows])), 1),
        }
        print("\nGATE:", out["summary"])
    tag = "-".join(channels)
    path = os.path.join("results", f"gan_inpaint_qc_{args.site}_{tag}.json")
    os.makedirs("results", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
