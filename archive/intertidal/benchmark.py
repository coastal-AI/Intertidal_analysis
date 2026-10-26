"""Benchmark reproducible: pipeline VIEJO (3 jobs) vs NUEVO (1 descarga + local).

Mide tiempos reales de cada paso y la equivalencia de resultados (el nuevo debe
dar lo mismo que el viejo) para documentar la mejora de rendimiento sin datos
inventados. Cada condición se guarda a un JSON trazable.

VIEJO:  download_reference_map_openeo + evaluate_transition_cloud_coverage_openeo
        + compute_water_frequency_openeo   (baja el SCL 3 veces)
NUEVO:  analyze_scl_cube_openeo + SclCubeAnalysis.water_frequency
        (baja el SCL 1 vez, calcula el resto en local)
"""

from __future__ import annotations

import os
import time
import json

import numpy as np
import rasterio

from .notebook_compat import (
    download_reference_map_openeo,
    load_reference_map_tif,
    evaluate_transition_cloud_coverage_openeo,
    compute_water_frequency_openeo,
    analyze_scl_cube_openeo,
)


def _align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _refmap_equiv(new_rm, old_rm):
    a, b = _align(np.asarray(new_rm), np.asarray(old_rm))
    ta, tb = (a == 0), (b == 0)
    iou = float((ta & tb).sum()) / max(int((ta | tb).sum()), 1)
    return {
        "transition_iou": round(iou, 4),
        "class_agreement": round(float((a == b).mean()), 4),
    }


def _wf_equiv(new_wf, old_wf):
    a, b = _align(np.asarray(new_wf, float), np.asarray(old_wf, float))
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 10:
        return {"corr": None, "mae": None, "n_common": int(m.sum())}
    return {
        "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4),
        "mae": round(float(np.abs(a[m] - b[m]).mean()), 4),
        "n_common": int(m.sum()),
    }


def _area_km2(path):
    with rasterio.open(path) as s:
        px = abs(s.transform.a) * abs(s.transform.e)
        return round(s.width * s.height * px / 1e6, 2)


def benchmark_condition(
    conn,
    bbox,
    time_extent,
    *,
    label,
    outdir="bench",
    bad_classes=(3, 8, 9, 10),
    ref_bad_fraction_threshold=0.05,
    ref_stable_threshold=0.95,
    ref_coastal_buffer_pixels=10,
    transition_cloud_threshold=0.1,
    min_obs=8,
):
    """Corre viejo y nuevo para una condición y devuelve un dict con tiempos y
    equivalencia. Escribe los rasters intermedios en ``outdir`` (no reusa: cada
    paso se recomputa con force=True para medir de verdad)."""
    os.makedirs(outdir, exist_ok=True)
    res = {"label": label, "time_extent": list(time_extent)}

    # ══ PIPELINE VIEJO (3 jobs) ══════════════════════════════════════════════
    old_ref_path = os.path.join(outdir, f"old_refmap_{label}.tif")
    t = time.perf_counter()
    download_reference_map_openeo(
        conn=conn, bbox=bbox, time_extent=time_extent, out_path=old_ref_path,
        bad_classes=list(bad_classes), bad_fraction_threshold=ref_bad_fraction_threshold,
        stable_threshold=ref_stable_threshold,
        transition_buffer_pixels=ref_coastal_buffer_pixels, force=True,
    )
    t_ref = time.perf_counter() - t
    old_rm, _, _ = load_reference_map_tif(old_ref_path)

    t = time.perf_counter()
    old_stats = evaluate_transition_cloud_coverage_openeo(
        conn=conn, bbox=bbox, time_extent=time_extent, reference_map=old_rm,
        bad_classes=list(bad_classes), reference_dates=[],
        global_bad_fraction_threshold=ref_bad_fraction_threshold,
    )
    t_cloud = time.perf_counter() - t
    old_valid = sorted(
        d for d, pct in old_stats.items() if pct / 100.0 <= transition_cloud_threshold
    )

    old_wf_path = os.path.join(outdir, f"old_wf_{label}.tif")
    t = time.perf_counter()
    old_wf, _, _ = compute_water_frequency_openeo(
        conn=conn, bbox=bbox, time_extent=time_extent, valid_dates=old_valid,
        out_path=old_wf_path, min_obs=min_obs, force=True,
    )
    t_wf = time.perf_counter() - t
    old_total = t_ref + t_cloud + t_wf

    # ══ PIPELINE NUEVO (1 descarga + local) ══════════════════════════════════
    t = time.perf_counter()
    analysis = analyze_scl_cube_openeo(
        conn=conn, bbox=bbox, time_extent=time_extent, bad_classes=list(bad_classes),
        bad_fraction_threshold=ref_bad_fraction_threshold,
        stable_threshold=ref_stable_threshold,
        transition_buffer_pixels=ref_coastal_buffer_pixels,
        global_bad_fraction_threshold=ref_bad_fraction_threshold,
        transition_cloud_threshold=transition_cloud_threshold, reference_dates=[],
        verbose=False, plot=False,
    )
    t_analyze = time.perf_counter() - t

    new_wf_path = os.path.join(outdir, f"new_wf_{label}.tif")
    t = time.perf_counter()
    new_wf, _, _ = analysis.water_frequency(
        valid_dates=analysis.valid_dates, out_path=new_wf_path, min_obs=min_obs,
    )
    t_wf_local = time.perf_counter() - t
    new_total = t_analyze + t_wf_local

    res.update({
        "area_km2": _area_km2(old_ref_path),
        "n_dates": len(analysis.dates),
        "n_valid_new": len(analysis.valid_dates),
        "n_valid_old": len(old_valid),
        "old": {
            "ref_job": round(t_ref, 1), "cloud": round(t_cloud, 1),
            "wf_job": round(t_wf, 1), "total": round(old_total, 1),
        },
        "new": {
            "download_analyze": round(t_analyze, 1),
            "wf_local": round(t_wf_local, 3), "total": round(new_total, 1),
        },
        "speedup": round(old_total / new_total, 2) if new_total else None,
        "equiv_refmap": _refmap_equiv(analysis.reference_map, old_rm),
        "equiv_wf": _wf_equiv(new_wf, old_wf),
    })
    return res


def run_benchmark(conn, bbox, conditions, out_json="benchmark_results.json",
                  outdir="bench", **kwargs):
    """Corre varias condiciones y escribe el JSON de forma INCREMENTAL (así los
    resultados parciales sobreviven si algo se corta). ``conditions`` es una
    lista de (label, time_extent)."""
    results = []
    if os.path.exists(out_json):
        try:
            results = json.load(open(out_json, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            results = []
    done = {r["label"] for r in results}

    for label, te in conditions:
        if label in done:
            print(f"[benchmark] '{label}' ya estaba -> se salta")
            continue
        print(f"[benchmark] condición '{label}' {te} ...", flush=True)
        r = benchmark_condition(conn, bbox, te, label=label, outdir=outdir, **kwargs)
        results.append(r)
        json.dump(results, open(out_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[benchmark] '{label}' hecho: viejo {r['old']['total']/60:.1f} min "
              f"| nuevo {r['new']['total']/60:.1f} min | x{r['speedup']} | "
              f"IoU {r['equiv_refmap']['transition_iou']} | "
              f"WF corr {r['equiv_wf']['corr']}", flush=True)
    return results
