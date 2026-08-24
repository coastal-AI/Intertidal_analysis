# -*- coding: utf-8 -*-
import json, shutil, os, sys
NB=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\final_notebook.ipynb"
BK=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\final_notebook.ipynb.bak"
shutil.copyfile(NB, BK); print("backup ->", BK)
nb=json.load(open(NB, encoding="utf-8"))

def cell(src):  # helper: código con source como lista de líneas terminadas en \n
    lines=src.splitlines(keepends=True)
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":lines}

SEC4 = '''# ════════════════════════════════════════════════════════════════════════════
# Sección 4 · Reference map — DESCARGA ÚNICA del cubo SCL
# `analyze_scl_cube_openeo` baja el cubo SCL UNA sola vez y calcula en LOCAL el
# reference map, la cobertura de nubes (sección 5) y el water frequency (sección
# 6), en lugar de lanzar 3 jobs separados. La réplica local coincide con el UDF
# server-side (IoU transición 0.99, acuerdo de clases 99.7%).
# ════════════════════════════════════════════════════════════════════════════
from intertidal.notebook_compat import analyze_scl_cube_openeo

analysis = analyze_scl_cube_openeo(
    conn=conn,
    bbox=bbox,
    time_extent=time_extent,
    bad_classes=bad_scl_classes,
    bad_fraction_threshold=ref_bad_fraction_threshold,
    stable_threshold=ref_stable_threshold,
    transition_buffer_pixels=ref_coastal_buffer_pixels,
    global_bad_fraction_threshold=ref_bad_fraction_threshold,
    transition_cloud_threshold=transition_cloud_threshold,
    reference_dates=[],
    verbose=False, plot=False,        # el reporte se muestra por secciones (aquí y en la 5)
    # cache_nc="scl_cube_cache.nc",   # descomenta para REUSAR una descarga previa (mismo AOI)
)

# Variables del reference map (mismos nombres que antes)
reference_map_openeo   = analysis.reference_map
ref_transform, ref_crs = analysis.transform, analysis.crs
transition_mask        = (reference_map_openeo == 0)
analysis.save_reference_map(ref_map_path)        # guarda reference_map.tif

# Stats + visualización del reference map
analysis.report_reference_map()
'''

SEC5 = '''# ════════════════════════════════════════════════════════════════════════════
# Sección 5 · Recuperación de datos usables — nubes en la zona de transición
# Usa el MISMO cubo ya descargado en la sección 4 (no vuelve a bajar nada).
# ════════════════════════════════════════════════════════════════════════════
analysis.report_cloud_coverage()

# Variables aguas abajo (mismos nombres que antes)
transition_stats        = analysis.transition_stats
valid_dates_transition  = analysis.valid_dates
reference_dates         = analysis.reference_dates
newly_valid             = analysis.newly_valid
gain                    = analysis.gain
'''

SEC6 = '''# ════════════════════════════════════════════════════════════════════════════
# Sección 6 · Water frequency — desde el cubo ya descargado (sin nuevo job)
# ════════════════════════════════════════════════════════════════════════════
valid_dates_for_wf = sorted(valid_dates_transition)
print(f"Valid dates for water frequency computation: {len(valid_dates_for_wf)}")

if valid_dates_for_wf:
    print(f"  Range: {valid_dates_for_wf[0]} \u2192 {valid_dates_for_wf[-1]}")
else:
    raise ValueError("No valid dates available for water frequency computation.")

wf_out_out_path = "water_frequency.tif"

# Reusa el cubo SCL de `analysis` (cálculo local) en vez de lanzar otro batch job
water_freq, wf_transform, wf_crs = analysis.water_frequency(
    valid_dates=valid_dates_for_wf,
    out_path=wf_out_out_path,
    min_obs=8,
)

print("Water frequency computed:")
print(f"  Shape: {water_freq.shape}")
print(f"  CRS:   {wf_crs}")
print(f"  Values: [{np.nanmin(water_freq):.3f}, {np.nanmax(water_freq):.3f}]")
print(f"  Pixels with data (non-NaN): {np.isfinite(water_freq).sum():,}")

plot_water_frequency(
    water_freq,
    transform=wf_transform,
    crs=wf_crs,
    polygon=polygon,
    title="Water frequency map - Ria de Villaviciosa",
    min_water_patch_pixels=20,
    water_presence_threshold=0.15,
)
'''

new_cells=[]; actions=[]
for c in nb["cells"]:
    src="".join(c["source"])
    if c["cell_type"]=="code" and "download_reference_map_openeo(" in src:
        new_cells.append(cell(SEC4)); actions.append("REEMPLAZADA sección 4 (analyze + refmap)")
    elif c["cell_type"]=="code" and "load_reference_map_tif(" in src:
        actions.append("ELIMINADA celda antigua de stats reference map (fusionada en sección 4)")
        continue  # borrar
    elif c["cell_type"]=="code" and "evaluate_transition_cloud_coverage_openeo(" in src:
        new_cells.append(cell(SEC5)); actions.append("REEMPLAZADA sección 5 (report_cloud_coverage)")
    elif c["cell_type"]=="code" and "compute_water_frequency_openeo(" in src:
        new_cells.append(cell(SEC6)); actions.append("REEMPLAZADA sección 6 (analysis.water_frequency)")
    else:
        new_cells.append(c)

nb["cells"]=new_cells
json.dump(nb, open(NB,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(" - "+a for a in actions))
print(f"\nceldas: {len(new_cells)} (antes {len(new_cells)+1})")
