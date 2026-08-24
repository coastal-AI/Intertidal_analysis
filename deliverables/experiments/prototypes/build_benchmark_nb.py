# -*- coding: utf-8 -*-
import json
OUT=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\benchmark_optimizacion.ipynb"
def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def co(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}
cells=[]

cells.append(md('''# Benchmark — pipeline VIEJO vs NUEVO (datos medidos, sin inventar)

Compara el pipeline **viejo** (baja el SCL en 3 jobs: reference map, nubes,
water frequency) con el **nuevo** (`analyze_scl_cube_openeo`: baja el cubo una
vez y calcula el resto en local), sobre la **Ría de Villaviciosa**, variando el
**nº de años** (eje temporal).

Para cada condición se mide, de verdad:
- **tiempos reales** de cada paso (viejo y nuevo),
- **equivalencia** de resultados: el nuevo debe dar lo mismo que el viejo
  (IoU del reference map, correlación del water frequency).

Todo se guarda en `benchmark_results.json` y las gráficas se dibujan **desde ese
JSON** — ningún número escrito a mano.
'''))

cells.append(md('''## 1 · Cómo se generó (reproducible)

La celda siguiente está **desactivada por defecto** (`RUN = False`) porque corre
el pipeline viejo completo y tarda horas. Los resultados ya están en el JSON. Pon
`RUN = True` para regenerarlos desde cero.
'''))

cells.append(co('''RUN = False  # -> True para regenerar (corre los 3 jobs viejos + el nuevo por condición; ~2-3 h)

if RUN:
    import truststore; truststore.inject_into_ssl()
    import openeo
    from intertidal.benchmark import run_benchmark
    conn = openeo.connect("openeo.dataspace.copernicus.eu"); conn.authenticate_oidc()
    bbox = {"west":-5.4256,"south":43.4977,"east":-5.3755,"north":43.5277}  # pentágono Villaviciosa
    conditions = [
        ("1yr",  ["2020-01-01","2020-12-31"]),
        ("3yr",  ["2018-01-01","2020-12-31"]),
        ("10yr", ["2016-01-01","2025-12-31"]),
    ]
    run_benchmark(conn, bbox, conditions,
                  out_json="benchmark_results.json", outdir="bench")'''))

cells.append(md('## 2 · Resultados medidos'))

cells.append(co('''import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

with open("benchmark_results.json", encoding="utf-8") as f:
    R = json.load(f)
R = sorted(R, key=lambda r: r["n_dates"])   # por nº de fechas creciente
print(f"{len(R)} condiciones medidas")

rows=[]
for r in R:
    rows.append({
        "cond": r["label"], "años": len(r["time_extent"]) and round((int(r["time_extent"][1][:4])-int(r["time_extent"][0][:4]))+1),
        "fechas": r["n_dates"],
        "VIEJO (min)": round(r["old"]["total"]/60,1),
        "NUEVO (min)": round(r["new"]["total"]/60,1),
        "speedup": r["speedup"],
        "IoU refmap": r["equiv_refmap"]["transition_iou"],
        "acuerdo clases": r["equiv_refmap"]["class_agreement"],
        "corr WF": r["equiv_wf"]["corr"],
    })
df = pd.DataFrame(rows)
df'''))

cells.append(md('## 3 · Tiempo: viejo (3 jobs) vs nuevo (1 descarga + local)\n\nCada barra desglosa los pasos. El nuevo elimina los dos jobs de *reduce* (reference map y water frequency); lo que los sustituye (cálculo local) es la banda verde, casi invisible.'))

cells.append(co('''labels=[r["label"] for r in R]
x=np.arange(len(R)); w=0.38
fig,ax=plt.subplots(figsize=(1.8+2.2*len(R),5))
# VIEJO (apilado): ref_job + cloud + wf_job
ref=np.array([r["old"]["ref_job"]/60 for r in R])
cloud=np.array([r["old"]["cloud"]/60 for r in R])
wf=np.array([r["old"]["wf_job"]/60 for r in R])
ax.bar(x-w/2,ref,w,color="#ff9896",label="viejo · job reference map")
ax.bar(x-w/2,cloud,w,bottom=ref,color="#c5b0d5",label="viejo · descarga cubo (nubes)")
ax.bar(x-w/2,wf,w,bottom=ref+cloud,color="#ffbb78",label="viejo · job water frequency")
# NUEVO (apilado): download + local
dl=np.array([r["new"]["download_analyze"]/60 for r in R])
loc=np.array([r["new"]["wf_local"]/60 for r in R])
ax.bar(x+w/2,dl,w,color="#9ecae1",label="nuevo · descarga única")
ax.bar(x+w/2,loc,w,bottom=dl,color="#2ca02c",label="nuevo · cálculo local")
for i,r in enumerate(R):
    ax.text(x[i]-w/2,r["old"]["total"]/60,f'{r["old"]["total"]/60:.0f}',ha="center",va="bottom",fontsize=9)
    ax.text(x[i]+w/2,r["new"]["total"]/60,f'{r["new"]["total"]/60:.0f}',ha="center",va="bottom",fontsize=9,color="#2ca02c")
ax.set_xticks(x); ax.set_xticklabels([f'{r["label"]}\\n{r["n_dates"]} fechas' for r in R])
ax.set_ylabel("tiempo (min)"); ax.set_title("Pipeline viejo vs nuevo — Villaviciosa")
ax.legend(fontsize=8,ncol=2); ax.grid(axis="y",alpha=.3)
fig.tight_layout(); plt.show()'''))

cells.append(md('## 4 · Speedup por condición'))

cells.append(co('''fig,ax=plt.subplots(figsize=(1.8+1.6*len(R),4))
sp=[r["speedup"] for r in R]
b=ax.bar(x,sp,color="#2ca02c",width=0.5)
ax.axhline(1,color="gray",ls="--",lw=1)
for i,v in enumerate(sp): ax.text(i,v,f"x{v}",ha="center",va="bottom",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in R])
ax.set_ylabel("speedup (viejo/nuevo)"); ax.set_title("Cuántas veces más rápido es el nuevo")
ax.grid(axis="y",alpha=.3); fig.tight_layout(); plt.show()'''))

cells.append(md('## 5 · Equivalencia: el nuevo da lo mismo que el viejo\n\nSi la optimización fuera solo "más rápida pero distinta" no valdría. Aquí se comprueba que los resultados coinciden: IoU y acuerdo de clases del reference map, y correlación del water frequency (todo debería estar cerca de 1).'))

cells.append(co('''iou=[r["equiv_refmap"]["transition_iou"] for r in R]
agr=[r["equiv_refmap"]["class_agreement"] for r in R]
corr=[r["equiv_wf"]["corr"] if r["equiv_wf"]["corr"] is not None else np.nan for r in R]
ww=0.26
fig,ax=plt.subplots(figsize=(1.8+2.0*len(R),4.5))
ax.bar(x-ww,iou,ww,color="#0066ff",label="IoU zona transición")
ax.bar(x,agr,ww,color="#6baed6",label="acuerdo de clases")
ax.bar(x+ww,corr,ww,color="#2ca02c",label="corr water frequency")
ax.axhline(1,color="gray",ls="--",lw=1)
for i in range(len(R)):
    for dxp,val in [(-ww,iou[i]),(0,agr[i]),(ww,corr[i])]:
        if not np.isnan(val): ax.text(x[i]+dxp,val,f"{val:.3f}",ha="center",va="bottom",fontsize=8,rotation=90)
ax.set_ylim(0.9,1.01); ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in R])
ax.set_ylabel("coincidencia (1 = idéntico)"); ax.set_title("Equivalencia nuevo vs viejo")
ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3); fig.tight_layout(); plt.show()'''))

cells.append(md('''## 6 · Lectura

- El **nuevo** es más rápido en todas las condiciones porque quita los dos jobs de
  *reduce* del servidor; el cálculo local que los sustituye cuesta segundos.
- La ventaja **crece con los años**: a más fechas, más pesaban los jobs viejos.
- Y lo importante: **da el mismo resultado** (IoU ≈ 1, correlación ≈ 1), así que
  no es "más rápido a costa de peor" — es la misma ciencia, más barata.

*(Mañana: repetir el eje de escalabilidad — km²/área — para el paper.)*
'''))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
import nbformat
_,nb=nbformat.validator.normalize(nb) if hasattr(nbformat,"validator") else (0,nb)
json.dump(nb,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("escrito:",OUT,"|",len(cells),"celdas")
