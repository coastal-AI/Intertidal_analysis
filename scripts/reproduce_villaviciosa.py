# -*- coding: utf-8 -*-
"""Reproduce the published Villaviciosa product (products_marea/) and check it.

Route A, offline (no account, no cube, no tide model): the replication kit
in replication/ holds the record and the EOT20 levels; steps 6-11 are rerun
and must give arrays identical to products_marea/marea.npz.

    python -X utf8 scripts/reproduce_villaviciosa.py --from-kit replication

Route B, from the authors' cube (3.09 GB, not public) and the EOT20 files:
steps 3-11 are rerun.

    python -X utf8 scripts/reproduce_villaviciosa.py --from-cube ndwi_cube_villaviciosa_grande_10y.nc --tide-dir tide_models

The settings are read from scripts/sites/villaviciosa.yaml (min_scenes 40,
quality_min_bracket 0, the tide at the registry centroid). The run writes
marea.npz, result.json, figure.png, record.json and provenance.json into
--out (default runs/villaviciosa_repro), never into products_marea/, and
ends with a check table against products_marea/. The exit code is 0 only
if every check passes. These checks are about code behaviour: they compare
a run with the published product, not with any measurement.
"""
import argparse
import datetime as _dt
import json
import sys
import time
import warnings
from pathlib import Path

# s_km is NaN where water cannot reach; tide_estimators.make_bands_fixed casts
# it to int before masking those pixels to band -1, which numpy reports as an
# invalid cast. Harmless; silenced so the console shows only the run.
warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                        category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import marea_tools as mt  # noqa: E402

PROD = ROOT / "products_marea"
SITE_YAML = ROOT / "scripts" / "sites" / "villaviciosa.yaml"
KIT_RECORD = "villaviciosa_record_2023-2025.npz"
KIT_LEVELS = "villaviciosa_eot20_levels.npz"

#: Route B counts that no file in the repository stores (the screening of the
#: authors' cube, run with the settings of villaviciosa.yaml; Windows 10
#: laptop, 2026-10-27). Everything else is read from products_marea/.
EXPECTED_B = {"cube_scenes": 1379, "frame": (915, 915),
              "classes": {0: 116_011, 1: 24_924, 2: 696_290},
              "usable": 305}
CLASS_NAMES = {0: "transition", 1: "stable water", 2: "stable land"}


def say(k, text, expected=None, ok=None):
    """Print one step line: '(k) <step name>: text  [expected ...] ok'."""
    tail = ""
    if expected is not None:
        tail = f"  (expected {expected})"
    if ok is not None:
        tail += "  ok" if ok else "  MISMATCH"
    print(f"{mt.step(k)}: {text}{tail}", flush=True)
    return ok


def print_checks(rows, verdict=True):
    if not verdict:
        rows = [(*r[:4], {"PASS": "same", "FAIL": "differs", "SKIP": "-"}[r[4]])
                for r in rows]
    print(mt.format_table(rows, ("check", "expected", "found", "tolerance", "status")))


def reference():
    res = json.loads((PROD / "result.json").read_text(encoding="utf-8"))
    with np.load(PROD / "marea.npz") as d:
        prod = {k: d[k] for k in ("keep", "shape", "tau_px_min")}
    return res, prod


def run_reconstruct(cube_path, out, boundary, ex_rec, times, cfg, n_boot):
    """Steps 7-11 (and the hypsometry) through pyintertidal.marea.reconstruct."""
    from pyintertidal import marea

    m = cfg["marea"]
    t0 = time.time()
    res = marea.reconstruct(
        mt.short_path(cube_path), mt.short_path(out), name="Villaviciosa", boundary=boundary,
        extraction=ex_rec, overpass_times=times, mouth_side=m["mouth_side"],
        band_km=float(m["band_km"]), min_px_band=int(m["min_px_band"]),
        n_boot=int(n_boot), min_scenes=int(m["min_scenes"]),
        quality_min_bracket=int(m["quality_min_bracket"]))
    return res, time.time() - t0


def report_steps(res, ref, out, n_boot):
    """Lines for steps 7-11 from the run's own result.json."""
    with np.load(Path(out) / "marea.npz") as d:
        tau_px = d["tau_px_min"]
    say(7, f"{len(res['centros_km'])} sections centred at "
           + ", ".join(f"{c:.2f}" for c in res["centros_km"]) + " km",
        expected=", ".join(f"{c:.2f}" for c in ref["centros_km"]),
        ok=np.allclose(res["centros_km"], ref["centros_km"], rtol=0, atol=1e-9))
    iv = ("" if not n_boot else "; 95 % interval " + ", ".join(
        f"[{a:.1f}, {b:.1f}]" for a, b in zip(res["tau_lo_min"], res["tau_hi_min"])))
    say(8, "delays " + ", ".join(f"{t:.2f}" for t in res["tau_usado_min"]) + " min" + iv,
        expected=", ".join(f"{t:.2f}" for t in ref["tau_usado_min"]),
        ok=np.allclose(res["tau_usado_min"], ref["tau_usado_min"], rtol=0, atol=1e-6))
    say(9, f"per-pixel delay median {res['tau_px_median_min']:.0f} min, max "
           f"{res['tau_px_max_min']:.0f} min, {int((tau_px > 0).sum()):,} px shifted",
        expected=f"{ref['tau_px_median_min']:.0f}, {ref['tau_px_max_min']:.0f}",
        ok=(res["tau_px_median_min"], res["tau_px_max_min"])
        == (ref["tau_px_median_min"], ref["tau_px_max_min"]))
    say(10, f"{res['n_px_cota']:,} of {res['n_px']:,} px with an elevation "
            f"(quality_min_bracket {res.get('quality_min_bracket')})",
        expected=f"{ref['n_px_cota']:,}", ok=res["n_px_cota"] == ref["n_px_cota"])
    say(11, f"hypsometry over {res['hipsometria']['area_total_km2']:.4f} km2; "
            f"marea.npz, result.json, figure.png, record.json, provenance.json in {mt.rel(out)}")


def route_a(kit, out, cfg, n_boot, started):
    from pyintertidal import sites

    kit = Path(kit)
    ref, prod = reference()
    manifest = json.loads((kit / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = []
    for name, f in manifest["files"].items():
        got = mt.sha256(kit / name)
        if got != f["sha256"]:
            bad.append(name)
    if bad:
        raise SystemExit(f"kit files differ from MANIFEST.json: {bad}; rebuild or "
                         f"download the kit again")
    rec_sha = manifest["files"][KIT_RECORD]["sha256"]
    times = mt.load_overpass(mt.resolve(cfg["overpass_times"]))
    ex = mt.load_extraction(kit / KIT_RECORD)

    say(1, f"not needed (the kit holds the record); source cube sha256 "
           f"{ex['source_cube_sha256'][:12]}...",
        expected=ref["inputs_sha"]["cube"][:12] + "...",
        ok=ex["source_cube_sha256"] == ref["inputs_sha"]["cube"])
    say(2, f"{len(times):,} acquisition times in {cfg['overpass_times']}")
    say(3, "done when the kit was built (replication/MANIFEST.json); every kit "
           "file matches its sha256")
    say(4, f"{ex['keep'].size:,} intertidal px on a {ex['shape'][0]} x {ex['shape'][1]} frame",
        expected=f"{ref['n_px']:,}",
        ok=ex["keep"].size == ref["n_px"] and np.array_equal(ex["keep"], prod["keep"]))
    rec = [str(d) for d in ex["dates"]]
    say(5, f"{len(rec)} scenes, {rec[0]} .. {rec[-1]}",
        expected=len(ref["record_dates"]), ok=rec == ref["record_dates"])
    tb = mt.TableBoundary(kit / KIT_LEVELS)
    lat, lon = (float(v) for v in sites.get(cfg["key"]).centroid)
    import pandas as pd
    t_rec = pd.DatetimeIndex(pd.to_datetime([times[d] for d in rec])).tz_localize(None)
    h0 = tb.levels(t_rec)
    say(6, f"{tb.name} tabulated at ({tb.lat:.6f}, {tb.lon:.6f}), {len(tb.times_ns)} instants "
           f"x {len(tb.shifts_min)} shifts; level {h0.min():+.3f} .. {h0.max():+.3f} m",
        expected=f"{ref['rango_marea_m'][0]:+.3f} .. {ref['rango_marea_m'][1]:+.3f} m at "
                 f"({lat:.6f}, {lon:.6f})",
        ok=[float(h0.min()), float(h0.max())] == ref["rango_marea_m"]
        and (tb.lat, tb.lon) == (lat, lon))
    print(f"steps 7-11: pyintertidal.marea.reconstruct (n_boot {n_boot})", flush=True)
    res, dt = run_reconstruct(kit / KIT_RECORD, out, tb, ex, times, cfg, n_boot)
    if res is None:
        raise SystemExit(f"reconstruct refused the record: see {mt.rel(out)}/result.json")
    print(f"reconstruct took {dt:.0f} s", flush=True)
    mt.write_json(out / "record.json", {"dates": rec, "epoch": "2023-2025",
                                         "boundary": tb.name})
    report_steps(res, ref, out, n_boot)
    mt.provenance(out, {"kit_record": {"path": kit / KIT_RECORD, "sha256": rec_sha},
                        "kit_levels": kit / KIT_LEVELS,
                        "kit_manifest": kit / "MANIFEST.json",
                        "overpass_times": mt.resolve(cfg["overpass_times"]),
                        "site_yaml": SITE_YAML},
                  {"route": "A (offline kit)", "n_boot": n_boot,
                   **{k: cfg["marea"][k] for k in ("mouth_side", "min_scenes",
                                                   "quality_min_bracket", "band_km",
                                                   "min_px_band")},
                   "boundary": f"TableBoundary({KIT_LEVELS})"},
                  started=started)
    return rec_sha


def route_b(cube, tide_dir, out, cfg, n_boot, started):
    import pandas as pd
    from pyintertidal import marea, sites
    from pyintertidal.boundary import PyTMDBoundary

    ref, prod = reference()
    cube = Path(cube)
    t0 = time.time()
    info = mt.cube_info(cube)
    cube_sha = mt.sha256(cube)
    say(1, f"{cube.name}: {len(info['dates'])} scenes {info['dates'][0]} .. "
           f"{info['dates'][-1]}, {info['shape'][0]} x {info['shape'][1]} px; sha256 "
           f"{cube_sha[:12]}... [{time.time() - t0:.0f} s]",
        expected=f"{EXPECTED_B['cube_scenes']} scenes, sha256 {ref['inputs_sha']['cube'][:12]}...",
        ok=(len(info["dates"]) == EXPECTED_B["cube_scenes"]
            and tuple(info["shape"]) == EXPECTED_B["frame"]
            and cube_sha == ref["inputs_sha"]["cube"]))
    times = mt.load_overpass(mt.resolve(cfg["overpass_times"]))
    missing = [d for d in info["dates"] if d not in times]
    say(2, f"{len(times):,} acquisition times; {len(missing)} cube dates without one",
        expected="0 without", ok=not missing)
    if missing:
        raise SystemExit(f"cube dates without an acquisition time: {missing[:5]} ...")
    s = cfg["screening"]
    t0 = time.time()
    usable, refmap, cloud_pct = mt.screen(
        cube, threshold=s["threshold"], bad_classes=tuple(s["bad_classes"]),
        clean_scene_max_bad=s["clean_scene_max_bad"],
        stable_threshold=s["stable_threshold"],
        transition_buffer_px=s["transition_buffer_px"],
        cloud_threshold=s["cloud_threshold"])
    counts = {c: int((refmap == c).sum()) for c in (0, 1, 2)}
    say(3, "classes " + " / ".join(f"{counts[c]:,} {CLASS_NAMES[c]}" for c in (0, 1, 2))
           + f" px; {len(usable)} usable scenes [{time.time() - t0:.0f} s]",
        expected=" / ".join(f"{EXPECTED_B['classes'][c]:,}" for c in (0, 1, 2))
                 + f"; {EXPECTED_B['usable']}",
        ok=counts == EXPECTED_B["classes"] and len(usable) == EXPECTED_B["usable"])
    t0 = time.time()
    ex = marea.extract(str(cube))
    say(4, f"{ex['keep'].size:,} intertidal px [{time.time() - t0:.0f} s]",
        expected=f"{ref['n_px']:,}", ok=np.array_equal(ex["keep"], prod["keep"]))
    lat, lon = (float(v) for v in sites.get(cfg["key"]).centroid)
    bnd = PyTMDBoundary(cfg["tide"]["model"], lat, lon, directory=str(tide_dir))
    t0 = time.time()
    ex_rec, rec = mt.build_record(ex, usable, times, bnd,
                                  epoch_years=cfg["record"]["epoch_years"],
                                  which=cfg["record"]["which"])
    del ex
    say(5, f"{len(rec)} scenes in the epoch {ex_rec['record_epoch']}, {rec[0]} .. {rec[-1]} "
           f"[{time.time() - t0:.0f} s]",
        expected=len(ref["record_dates"]), ok=rec == ref["record_dates"])
    # the production call passed the registry polygon's box with the record
    ex_rec["bbox"] = {**sites.get(cfg["key"]).bbox, "crs": "EPSG:4326"}
    t_rec = pd.DatetimeIndex(pd.to_datetime([times[d] for d in rec])).tz_localize(None)
    h0 = np.asarray(bnd.levels(t_rec), float)
    say(6, f"{bnd.name} at ({lat:.6f}, {lon:.6f}); level {h0.min():+.3f} .. {h0.max():+.3f} m",
        expected=f"{ref['rango_marea_m'][0]:+.3f} .. {ref['rango_marea_m'][1]:+.3f} m",
        ok=[float(h0.min()), float(h0.max())] == ref["rango_marea_m"])
    print(f"steps 7-11: pyintertidal.marea.reconstruct (n_boot {n_boot})", flush=True)
    res, dt = run_reconstruct(cube, out, bnd, ex_rec, times, cfg, n_boot)
    if res is None:
        raise SystemExit(f"reconstruct refused the record: see {mt.rel(out)}/result.json")
    print(f"reconstruct took {dt:.0f} s", flush=True)
    mt.write_json(out / "record.json", {"dates": rec, "epoch": ex_rec["record_epoch"],
                                         "boundary": bnd.name})
    report_steps(res, ref, out, n_boot)
    mt.provenance(out, {"cube": {"path": cube, "sha256": cube_sha},
                        "overpass_times": mt.resolve(cfg["overpass_times"]),
                        "site_yaml": SITE_YAML},
                  {"route": "B (authors' cube)", "n_boot": n_boot,
                   "tide": f"PyTMDBoundary({cfg['tide']['model']!r}, {lat}, {lon}, "
                           f"directory={mt.rel(tide_dir)!r})",
                   "screening": s, "record": cfg["record"],
                   **{k: cfg["marea"][k] for k in ("mouth_side", "min_scenes",
                                                   "quality_min_bracket", "band_km",
                                                   "min_px_band")}},
                  started=started)
    return cube_sha


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproduce products_marea/ (Ría de Villaviciosa) and check the run "
                    "against it. Route A: --from-kit; Route B: --from-cube.",
        epilog="Steps: " + "; ".join(mt.step(k) for k in range(1, 12)))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-env", action="store_true",
                   help="print installed against pinned package versions and stop")
    g.add_argument("--from-kit", metavar="DIR", help="Route A: the replication kit folder")
    g.add_argument("--from-cube", metavar="PATH",
                   help="Route B: ndwi_cube_villaviciosa_grande_10y.nc")
    g.add_argument("--compare-only", metavar="DIR",
                   help="compare an existing run folder with products_marea/ "
                        "(facts only, no verdict; e.g. a run from a re-downloaded cube)")
    ap.add_argument("--tide-dir", default="tide_models",
                    help="Route B: folder holding EOT20/ocean_tides/ (default tide_models)")
    ap.add_argument("--fast", action="store_true",
                    help="n_boot 0: skip the 30 bootstrap fits; the intervals are not checked")
    ap.add_argument("--out", default=None,
                    help="output folder (default: out_dir of scripts/sites/villaviciosa.yaml, "
                         "runs/villaviciosa_repro)")
    a = ap.parse_args(argv)

    if a.check_env:
        rows, ok = mt.check_env()
        print(mt.format_table(rows, ("package", "required", "installed", "status")))
        print("environment ok" if ok else "a core package is MISSING")
        return 0 if ok else 1

    if a.compare_only:
        rows, _ = mt.compare_products(a.compare_only, PROD)
        print(f"comparison of {a.compare_only} with products_marea/ (no verdict)")
        print_checks(rows, verdict=False)
        ov = mt.overlap_summary(a.compare_only, PROD)
        print(f"pixels: run {ov['pixels_run']:,}, product {ov['pixels_reference']:,}, "
              f"common {ov['pixels_common']:,}; z finite in both {ov['z_both_finite']:,}")
        print(f"identical z: {ov['z_identical_share']:.1%}; largest |dz| "
              f"{ov['z_max_abs_diff_m']:.3f} m; median |dz| {ov['z_median_abs_diff_m']:.3f} m")
        return 0

    cfg = mt.load_site(SITE_YAML)
    out = mt.guard_out_dir(mt.resolve(a.out or cfg["out_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    n_boot = 0 if a.fast else int(cfg["marea"]["n_boot"])
    started = _dt.datetime.now(_dt.timezone.utc)
    t_all = time.time()
    route = "A, offline kit" if a.from_kit else "B, authors' cube"
    print(f"MAREA: reproduce products_marea/ (Ría de Villaviciosa); route {route}")
    print(f"settings from {mt.rel(SITE_YAML)}: min_scenes {cfg['marea']['min_scenes']}, "
          f"quality_min_bracket {cfg['marea']['quality_min_bracket']}, mouth_side "
          f"{cfg['marea']['mouth_side']}, band_km {cfg['marea']['band_km']}, n_boot {n_boot}; "
          f"output {mt.rel(out)}", flush=True)
    if a.from_kit:
        sha = route_a(mt.resolve(a.from_kit), out, cfg, n_boot, started)
    else:
        sha = route_b(mt.resolve(a.from_cube), mt.resolve(a.tide_dir), out, cfg, n_boot, started)

    rows, passed = mt.compare_products(out, PROD, expected_cube_sha=sha, fast=bool(a.fast))
    print()
    print(f"check of {mt.rel(out)} against products_marea/ (code behaviour, not a validation)")
    print_checks(rows)
    n_pass = sum(r[4] == "PASS" for r in rows)
    n_skip = sum(r[4] == "SKIP" for r in rows)
    print(f"{'PASS' if passed else 'FAIL'}: {n_pass} of {len(rows) - n_skip} checks passed"
          + (f", {n_skip} skipped (--fast)" if n_skip else "")
          + f"; total {time.time() - t_all:.0f} s")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
