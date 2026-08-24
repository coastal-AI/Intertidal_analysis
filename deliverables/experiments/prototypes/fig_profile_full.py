import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad"
dl=json.load(open(S+r"\profile_download.json"))
pts=sorted([(v["n_dates"],v["job_s"],v["years"]) for v in dl.values()])
nd=np.array([p[0] for p in pts],float); js=np.array([p[1] for p in pts],float); yr=[p[2] for p in pts]
m=np.polyfit(nd,js,1); slope,inter=m
print(f"descarga ~ {slope:.2f} s/fecha + {inter:.0f} s  (fit)")

# tiempos aprox del pipeline VIEJO (3 jobs) para el sitio 10yr / 14.4 km2
ref_job=390.0   # ~6.5 min (job reduce reference map)
cloud_dl=2509.0 # descarga del cubo para nubes
wf_job=510.0    # ~8.5 min (job reduce water frequency)
old_total=ref_job+cloud_dl+wf_job
local_all=9.3   # calculo local total (medido)
new_total=cloud_dl+local_all

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
# ── Panel 1: descarga vs nº fechas ──
xs=np.linspace(0,1450,50)
ax[0].plot(xs,np.polyval(m,xs),"--",color="gray",lw=1,label=f"~{slope:.1f} s/fecha")
ax[0].plot(nd,js,"o",color="#0066ff",ms=9)
for x,y,a in zip(nd,js,yr): ax[0].annotate(f"{a} año{'s' if a>1 else ''}\n{y/60:.0f} min",(x,y),
    textcoords="offset points",xytext=(8,-4),fontsize=9)
ax[0].plot(nd, np.full_like(nd, local_all), "s-", color="#2ca02c", label="cálculo local (todo)")
ax[0].set_xlabel("nº de fechas (≈ años × 138)"); ax[0].set_ylabel("tiempo (s)")
ax[0].set_title("Descarga del cubo SCL vs nº de fechas\n(AOI 14.4 km²)"); ax[0].legend(); ax[0].grid(alpha=.3)

# ── Panel 2: pipeline viejo vs nuevo (10 años, 14.4 km²) ──
ax[1].bar(0,ref_job,color="#ff9896",label="job reference map (~6.5 min)")
ax[1].bar(0,cloud_dl,bottom=ref_job,color="#c5b0d5",label="descarga cubo nubes (41.8 min)")
ax[1].bar(0,wf_job,bottom=ref_job+cloud_dl,color="#ffbb78",label="job water frequency (~8.5 min)")
ax[1].bar(1,cloud_dl,color="#c5b0d5")
ax[1].bar(1,local_all,bottom=cloud_dl,color="#2ca02c",label="cálculo local (9 s)")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels([f"VIEJO\n3 jobs\n{old_total/60:.1f} min",f"NUEVO\n1 job\n{new_total/60:.1f} min"])
ax[1].set_ylabel("tiempo total (s)"); ax[1].set_title("Pipeline por sitio (10 años, 14.4 km²)")
ax[1].legend(fontsize=8,loc="upper right")
saving=old_total-new_total
ax[1].annotate(f"−{saving/60:.0f} min\n(−{100*saving/old_total:.0f}%)",(1,new_total),
    textcoords="offset points",xytext=(20,10),fontsize=11,fontweight="bold",color="#2ca02c")
fig.suptitle("Perfilado del pipeline intermareal — descarga única del cubo SCL",fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
out=S+r"\profile_full.png"; fig.savefig(out,dpi=130); print("saved",out)
print(f"OLD {old_total/60:.1f} min | NEW {new_total/60:.1f} min | saving {saving/60:.1f} min ({100*saving/old_total:.0f}%)")
