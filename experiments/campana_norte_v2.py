"""Campana MAREA v2: encolar TODO en openEO, cosechar y procesar en paralelo.

The v1 runner serialized download->process per cell (~days). The insight the
user pushed for: openEO batch jobs run SERVER-side, so all cells can be
SUBMITTED at once (the backend parallelises and queues by account limits) —
the local machine only harvests finished jobs and processes them.

Architecture, all resumable through runs/campana_jobs.json:

* SUBMIT: for every cell without a product and without a cube, build the
  graph and ``create_job().start()`` — no waiting. Job ids to the manifest.
* HARVEST loop: poll pending jobs (one connection); each finished job is
  downloaded (sequentially — bandwidth and the OOM incident say one at a
  time) and handed to the processing pool.
* PROCESS pool: N_WORKERS subprocesses run pyintertidal.marea.reconstruct
  (now one tide-model call per cell instead of 13).
* PURGE: a successfully processed cell deletes its cube (71 GB free minus
  116 cubes would not survive the night). KEEP_CUBES=1 disables.

Run:  python -m experiments.campana_norte_v2
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

N_WORKERS = 4
KEEP_CUBES = os.environ.get("KEEP_CUBES", "0") == "1"
LOG = os.path.join("runs", "campana_marea.log")
MANIFEST = os.path.join("runs", "campana_jobs.json")
PY = sys.executable


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_manifest(m):
    json.dump(m, open(MANIFEST, "w", encoding="utf-8"), indent=1)


def main():
    os.makedirs("runs", exist_ok=True)
    from pyintertidal.net import use_system_certificates
    use_system_certificates()
    from pyintertidal.scenes import connect
    from pyintertidal.aoi import AOI

    cells = {c["name"]: c for c in
             json.load(open("north_coast_cells.json", encoding="utf-8"))}
    manifest = (json.load(open(MANIFEST, encoding="utf-8"))
                if os.path.exists(MANIFEST) else {})
    conn = connect(interactive=False)
    log(f"CAMPANA v2: {len(cells)} celdas, {N_WORKERS} workers, "
        f"purga={'no' if KEEP_CUBES else 'si'}")

    def done(name):
        return os.path.exists(f"runs/{name}/marea/result.json")

    def cube_path(name):
        return f"ndwi_cube_{name}.nc"

    # ── SUBMIT continuo: CDSE limita a 30 jobs concurrentes por cuenta ───
    # (medido: [400] ConcurrentJobLimit). La cola de envio se vacia desde el
    # bucle de cosecha: cada job terminado libera un hueco.
    to_submit = [n for n, c in cells.items()
                 if not done(n) and not os.path.exists(cube_path(n))
                 and manifest.get(n, {}).get("estado")
                 not in ("enviado", "descargado")]
    limit_hit = [0.0]        # instante del ultimo ConcurrentJobLimit

    def try_submit_next():
        if not to_submit or time.time() - limit_hit[0] < 120:
            return
        name = to_submit[0]
        for intento in range(4):
            try:
                aoi = AOI.from_polygon(cells[name]["polygon"], name=name)
                cube = conn.load_collection(
                    "SENTINEL2_L2A", spatial_extent=aoi.bbox,
                    temporal_extent=["2023-01-01", "2025-12-31"],
                    bands=["B03", "B08", "SCL"], max_cloud_cover=100)
                try:
                    cube = cube.resample_spatial(resolution=10,
                                                 method="near")
                except Exception:
                    pass
                job = cube.save_result(format="netCDF").create_job(
                    title=f"marea_{name}")
                job.start()
                manifest[name] = {"job": job.job_id, "estado": "enviado"}
                save_manifest(manifest)
                log(f"ENVIADO {name} -> {job.job_id} "
                    f"(quedan {len(to_submit)-1} por encolar)")
                to_submit.pop(0)
                time.sleep(2.0)
                return
            except Exception as e:
                msg = str(e)[:160]
                if "ConcurrentJobLimit" in msg:
                    limit_hit[0] = time.time()
                    return               # sin hueco: se reintenta luego
                if "429" in msg:
                    time.sleep(20 * (intento + 1))
                    continue
                log(f"ERROR envio {name}: {msg}")
                to_submit.pop(0)         # celda problematica: no bloquear
                return

    for _ in range(len(to_submit)):      # llenado inicial hasta el limite
        n0 = len(to_submit)
        try_submit_next()
        if len(to_submit) == n0:         # limite alcanzado o pausa
            break

    # ── HARVEST + PROCESS ────────────────────────────────────────────────
    running = {}

    def reap():
        for n in [n for n, p in running.items() if p.poll() is not None]:
            rc = running.pop(n).returncode
            estado = "?"
            res = f"runs/{n}/marea/result.json"
            if os.path.exists(res):
                try:
                    r = json.load(open(res, encoding="utf-8"))
                    estado = r.get("estado", "?")
                    if estado == "ok":
                        estado += (" OPERADOR" if r.get("con_operador")
                                   else " identidad")
                except Exception:
                    pass
            log(f"FIN {n}: rc={rc} {estado}")
            if rc == 0 and estado.startswith("ok") and not KEEP_CUBES:
                try:
                    os.remove(cube_path(n))
                    log(f"PURGADO cubo {n}")
                except OSError:
                    pass

    def launch(name):
        out = f"runs/{name}/marea"
        code = (f"from pyintertidal import marea; "
                f"marea.reconstruct({cube_path(name)!r}, {out!r}, "
                f"name={name!r})")
        running[name] = subprocess.Popen(
            [PY, "-c", code], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        log(f"PROCESANDO {name}")

    # cubos ya en disco entran directos al pool
    pending_local = [n for n in cells
                     if os.path.exists(cube_path(n)) and not done(n)]

    while True:
        reap()
        try_submit_next()                # rellenar huecos del limite de 30
        while pending_local and len(running) < N_WORKERS:
            launch(pending_local.pop(0))
        pend = [n for n, m in manifest.items()
                if m.get("estado") == "enviado" and not done(n)]
        if not pend and not pending_local and not running and not to_submit:
            break
        for name in pend:
            if len(running) >= N_WORKERS:
                break
            try:
                job = conn.job(manifest[name]["job"])
                st = job.status()
            except Exception as e:
                log(f"ERROR estado {name}: {str(e)[:120]}")
                try:
                    conn = connect(interactive=False)
                except Exception:
                    time.sleep(60)
                break
            if st == "finished":
                try:
                    log(f"DESCARGANDO {name}...")
                    job.get_results().download_file(cube_path(name))
                    manifest[name]["estado"] = "descargado"
                    save_manifest(manifest)
                    launch(name)
                except Exception as e:
                    log(f"ERROR descarga {name}: {str(e)[:160]}")
            elif st in ("error", "canceled"):
                log(f"JOB FALLIDO {name}: {st}")
                manifest[name]["estado"] = f"job_{st}"
                save_manifest(manifest)
        time.sleep(45)
    log("CAMPANA v2 COMPLETA")


if __name__ == "__main__":
    main()
