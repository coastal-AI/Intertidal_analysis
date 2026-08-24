# -*- coding: utf-8 -*-
import json
OUT=r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\perfilado_optimizacion.ipynb"

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def co(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}

cells=[]

cells.append(md('''# Perfilado de la optimización — descarga única del cubo SCL

Este notebook resume, con datos reales medidos sobre la **Ría de Villaviciosa**
(AOI de 14.4 km², 10 años, 1379 fechas Sentinel-2), lo que ganamos al **bajar el
cubo SCL una sola vez** y calcular en local el *reference map*, la cobertura de
nubes y el *water frequency*, en lugar de lanzar tres jobs separados en el
servidor.

**Idea:** el paso caro es la descarga; los tres productos salen del mismo cubo,
así que se baja una vez y se calcula todo en local (réplica exacta de los UDF del
servidor).
'''))

cells.append(md('## 1 · Datos medidos\n\nTodos los números están embebidos (medidos en las corridas reales), así que el notebook se ejecuta sin volver a descargar nada.'))

cells.append(co('''import numpy as np
import matplotlib.pyplot as plt

# ── Cálculo LOCAL vs nº de fechas (área completa 14.4 km²) — [fechas, segundos] ──
LOCAL_BY_DATES = np.array([[100,0.501],[300,1.616],[600,4.771],[1000,6.966],[1379,9.299]])

# ── Cálculo LOCAL vs área (todas las fechas) — [km², segundos] ──
LOCAL_BY_AREA  = np.array([[0.9,0.519],[3.6,1.753],[8.1,3.529],[14.4,7.501]])

# ── DESCARGA del cubo vs nº de fechas (AOI 14.4 km²) — [fechas, seg_job, años] ──
DL_BY_DATES = np.array([[144,336.5,1],[434,578.6,3],[1379,2509.0,10]])

# ── DESCARGA vs área (periodo fijo 1 año) — [km², seg_job] ──
DL_BY_AREA  = np.array([[3.62,337.1],[8.11,336.9],[14.36,336.5]])

# ── Pipeline por sitio (10 años, 14.4 km²), segundos ──
REF_JOB, CLOUD_DL, WF_JOB = 390.0, 2509.0, 510.0   # 3 jobs (viejo)
LOCAL_ALL = 9.3                                     # cálculo local total (nuevo)
OLD = REF_JOB + CLOUD_DL + WF_JOB
NEW = CLOUD_DL + LOCAL_ALL
print(f"Pipeline viejo (3 jobs): {OLD/60:.1f} min")
print(f"Pipeline nuevo (1 job):  {NEW/60:.1f} min")
print(f"Ahorro: {(OLD-NEW)/60:.1f} min por sitio ({100*(OLD-NEW)/OLD:.0f}%)")'''))

cells.append(md('## 2 · El cálculo local escala lineal y es despreciable\n\nMover el reference map y el water frequency a cálculo local no añade coste real: el total (14.4 km² × 10 años) son ~9 s, frente a los ~42 min de la descarga.'))

cells.append(co('''fig,ax=plt.subplots(1,2,figsize=(11,4.2))
x,y=LOCAL_BY_DATES[:,0],LOCAL_BY_DATES[:,1]
ax[0].plot(x,y,"o-",color="#0066ff")
m=np.polyfit(x,y,1); ax[0].plot(x,np.polyval(m,x),"--",color="gray",lw=1,label=f"~{m[0]*1000:.2f} ms/fecha")
ax[0].set_xlabel("nº de fechas (14.4 km²)"); ax[0].set_ylabel("tiempo local (s)")
ax[0].set_title("Local vs nº de fechas"); ax[0].legend(); ax[0].grid(alpha=.3)
x2,y2=LOCAL_BY_AREA[:,0],LOCAL_BY_AREA[:,1]
ax[1].plot(x2,y2,"o-",color="#d2691e")
m2=np.polyfit(x2,y2,1); ax[1].plot(x2,np.polyval(m2,x2),"--",color="gray",lw=1,label=f"~{m2[0]:.2f} s/km²")
ax[1].set_xlabel("área (km², 1379 fechas)"); ax[1].set_ylabel("tiempo local (s)")
ax[1].set_title("Local vs área"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("Cálculo local (reference map + nubes + water frequency)",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); plt.show()'''))

cells.append(md('## 3 · La descarga manda: escala con las fechas, NO con el área\n\nLa descarga crece lineal con el número de fechas (~1.8 s/fecha), pero es **plana con el área**: bajar 3.6, 8.1 o 14.4 km² tarda lo mismo (~337 s a 1 año). Es decir, el coste lo fija el nº de escenas, no los km².'))

cells.append(co('''fig,ax=plt.subplots(1,2,figsize=(11,4.2))
# vs fechas
x,y,yr=DL_BY_DATES[:,0],DL_BY_DATES[:,1],DL_BY_DATES[:,2]
xs=np.linspace(0,1450,50); m=np.polyfit(x,y,1)
ax[0].plot(xs,np.polyval(m,xs),"--",color="gray",lw=1,label=f"~{m[0]:.1f} s/fecha")
ax[0].plot(x,y,"o",color="#0066ff",ms=9)
for xi,yi,a in zip(x,y,yr): ax[0].annotate(f"{int(a)} año(s)\\n{yi/60:.0f} min",(xi,yi),textcoords="offset points",xytext=(8,-6),fontsize=9)
ax[0].plot(x,np.full_like(x,LOCAL_ALL),"s-",color="#2ca02c",label="cálculo local")
ax[0].set_xlabel("nº de fechas"); ax[0].set_ylabel("tiempo (s)")
ax[0].set_title("Descarga vs nº de fechas"); ax[0].legend(); ax[0].grid(alpha=.3)
# vs área (plano)
xa,ya=DL_BY_AREA[:,0],DL_BY_AREA[:,1]
ax[1].plot(xa,ya,"o-",color="#c0392b",ms=9); ax[1].set_ylim(0,400)
ax[1].set_xlabel("área (km², 1 año fijo)"); ax[1].set_ylabel("tiempo descarga (s)")
ax[1].set_title("Descarga vs área  →  PLANA"); ax[1].grid(alpha=.3)
for xi,yi in zip(xa,ya): ax[1].annotate(f"{yi:.0f} s",(xi,yi),textcoords="offset points",xytext=(0,8),ha="center",fontsize=9)
fig.suptitle("Descarga del cubo SCL",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); plt.show()'''))

cells.append(md('## 4 · Pipeline viejo (3 jobs) vs nuevo (1 job)\n\nMismo sitio, 10 años, 14.4 km². Se eliminan los dos jobs de *reduce* (reference map y water frequency); el cálculo local que los sustituye cuesta ~9 s.'))

cells.append(co('''fig,ax=plt.subplots(figsize=(6,5))
ax.bar(0,REF_JOB,color="#ff9896",label="job reference map (~6.5 min)")
ax.bar(0,CLOUD_DL,bottom=REF_JOB,color="#c5b0d5",label="descarga cubo (41.8 min)")
ax.bar(0,WF_JOB,bottom=REF_JOB+CLOUD_DL,color="#ffbb78",label="job water frequency (~8.5 min)")
ax.bar(1,CLOUD_DL,color="#c5b0d5")
ax.bar(1,LOCAL_ALL,bottom=CLOUD_DL,color="#2ca02c",label="cálculo local (9 s)")
ax.set_xticks([0,1]); ax.set_xticklabels([f"VIEJO · 3 jobs\\n{OLD/60:.1f} min",f"NUEVO · 1 job\\n{NEW/60:.1f} min"])
ax.set_ylabel("tiempo total (s)"); ax.set_title("Pipeline por sitio (10 años, 14.4 km²)")
ax.legend(fontsize=8,loc="upper right")
ax.annotate(f"-{(OLD-NEW)/60:.0f} min ({100*(OLD-NEW)/OLD:.0f}%)",(1,NEW),textcoords="offset points",xytext=(15,10),fontsize=12,fontweight="bold",color="#2ca02c")
fig.tight_layout(); plt.show()'''))

cells.append(md('''## 5 · La réplica local coincide con el servidor

Antes de fiarnos del cálculo local hay que comprobar que da lo mismo que los UDF
del servidor. Comparación sobre el mismo cubo (Villaviciosa, 10 años):

| Producto | Local vs servidor |
|---|---|
| **Reference map** | IoU de la zona de transición **0.990**, acuerdo de clases **99.7 %** |
| **Water frequency** | correlación **0.9992**, error medio **0.003** |
| **Grid + CRS** | idénticos (origen, resolución 10 m, EPSG:32630) |

La pequeña diferencia (0.3 %) es una columna de borde del cubo; el resto coincide.
'''))

cells.append(md('''## 6 · Qué significa para escalar a la costa norte

- **El área es gratis (en tiempo de descarga).** Bajar 3.6 o 14.4 km² tarda lo
  mismo → conviene usar AOIs lo más grandes que permita el backend, cubriendo más
  ría por job.
- **El coste lo fijan los años (fechas).** ~1.8 s/fecha → ~42 min por sitio de
  10 años, independientemente del tamaño.
- **Una descarga en vez de tres.** Se ahorran ~15 min/sitio (−26 %) y se deja de
  bajar el SCL por triplicado.
- **El cálculo local no estorba.** Segundos incluso a 14 km² × 10 años, así que
  no se convierte en cuello de botella al crecer.
- **Paralelizable.** OpenEO admite varios jobs a la vez → se mapean muchas rías
  en paralelo; el ahorro del 26 % se multiplica por cada una.
'''))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("escrito:",OUT,"|",len(cells),"celdas")
