# -*- coding: utf-8 -*-
import json, shutil
NB=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\final_notebook.ipynb"
shutil.copyfile(NB, NB+".bak2"); print("backup -> final_notebook.ipynb.bak2")
nb=json.load(open(NB,encoding="utf-8"))
NL=chr(10)
actions=[]
for c in nb["cells"]:
    if c["cell_type"]!="code": continue
    src="".join(c["source"])

    # ── Cell params: añadir ndwi_threshold ──
    if "bad_scl_classes = [3, 8, 9, 10]" in src and "ndwi_threshold" not in src:
        anchor="bad_scl_classes = [3, 8, 9, 10] "
        add=(anchor+NL+NL
             +"# Umbral NDWI para detectar agua (agua = NDWI > umbral). El reference map,"+NL
             +"# el water frequency, el intertidal y la batimetria usan NDWI a 10 m REAL"+NL
             +"# (B03/B08). 0.0 = umbral clasico; en rias turbias conviene bajarlo. DEA usa 0.1."+NL
             +"ndwi_threshold = 0.0")
        src=src.replace(anchor, add, 1)
        actions.append("params: +ndwi_threshold")

    # ── Cell SEC4: analyze_scl -> analyze_ndwi + ndwi_threshold ──
    if "analyze_scl_cube_openeo" in src:
        src=src.replace("analyze_scl_cube_openeo","analyze_ndwi_cube_openeo")
        src=src.replace("Reference map — DESCARGA ÚNICA del cubo SCL",
                        "Reference map — DESCARGA ÚNICA (agua por NDWI a 10 m real)")
        src=src.replace("baja el cubo SCL UNA sola vez","baja B03/B08/SCL UNA sola vez")
        # insertar ndwi_threshold tras time_extent=
        src=src.replace("    time_extent=time_extent,"+NL,
                        "    time_extent=time_extent,"+NL+"    ndwi_threshold=ndwi_threshold,"+NL, 1)
        actions.append("SEC4: analyze_ndwi + ndwi_threshold")

    # ── Cells batimetría: reconstruct / pixels / isolines -> +ndwi_threshold ──
    if any(k in src for k in ("reconstructor.reconstruct(", "reconstruct_pixels(", "reconstruct_isolines(")) \
       and "ndwi_threshold" not in src:
        src=src.replace("        force=True,"+NL+"    )",
                        "        force=True,"+NL+"        ndwi_threshold=ndwi_threshold,"+NL+"    )")
        actions.append("batimetria: +ndwi_threshold")

    c["source"]=[l+NL for l in src.split(NL)]
    if c["source"] and c["source"][-1]==NL:
        c["source"][-1]=""
        c["source"]=[x for x in c["source"] if x!=""] or [""]
    # recomponer respetando el original: mejor split keepends
    c["source"]=src.splitlines(keepends=True)

json.dump(nb,open(NB,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(NL.join(" - "+a for a in actions))
