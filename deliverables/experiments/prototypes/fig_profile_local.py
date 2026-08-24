import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=r"C:\Users\Jorge\AppData\Local\Temp\claude\C--Users-Jorge-sketch-fitton\5d9205af-fca0-4694-94f5-29f7a56ab2b8\scratchpad"
d=json.load(open(S+r"\profile_local.json"))
bd=np.array(d["by_dates"],float); ba=np.array(d["by_area"],float)
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
# vs fechas
x,y=bd[:,0],bd[:,1]
ax[0].plot(x,y,"o-",color="#0066ff");
m=np.polyfit(x,y,1)
ax[0].plot(x,np.polyval(m,x),"--",color="gray",lw=1,label=f"~{m[0]*1000:.2f} ms/fecha")
ax[0].set_xlabel("nº de fechas (área 14.4 km²)"); ax[0].set_ylabel("tiempo cálculo local (s)")
ax[0].set_title("Local vs nº de fechas"); ax[0].legend(); ax[0].grid(alpha=.3)
# vs km2
x2,y2=ba[:,0],ba[:,2]
ax[1].plot(x2,y2,"o-",color="#d2691e")
m2=np.polyfit(x2,y2,1)
ax[1].plot(x2,np.polyval(m2,x2),"--",color="gray",lw=1,label=f"~{m2[0]:.2f} s/km²")
ax[1].set_xlabel("área (km², 1379 fechas)"); ax[1].set_ylabel("tiempo cálculo local (s)")
ax[1].set_title("Local vs área"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("Perfilado del cálculo LOCAL (reference map + nubes + water frequency)\n"
             "Total 14.4 km² × 10 años ≈ 9 s  vs  descarga del cubo ≈ 2509 s (41.8 min)",
             fontsize=11)
fig.tight_layout(rect=[0,0,1,0.92])
out=S+r"\profile_local.png"; fig.savefig(out,dpi=130); print("saved",out)
