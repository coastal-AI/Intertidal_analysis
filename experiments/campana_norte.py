"""Campana MAREA de la costa norte: todas las celdas, sin mirar ninguna a mano.

Reads north_coast_cells.json (the cells defined for the campaign), and for
each cell: ensures its cube (downloads are SERIAL — CDSE allows exactly one
connection; that incident is in the institutional memory), then processes it
with the full MAREA pipeline (pyintertidal.marea.reconstruct) in a WORKER
POOL of ``N_WORKERS`` subprocesses — download of cell k+1 overlaps the
processing of cell k.

Everything is resumable: a cell with runs/<name>/marea/result.json is
skipped, a cell whose cube exists skips the download. Failures are logged
and the campaign continues — one broken cell must never kill a night of
work. Progress goes to runs/campana_marea.log (one line per event) so a
Monitor can follow it.

Run:  python -m experiments.campana_norte            (all cells)
      MAX_CELLS=10 python -m experiments.campana_norte
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

N_WORKERS = 2          # 7.4 GB de RAM: cada worker pica ~1.5-2 GB
LOG = os.path.join("runs", "campana_marea.log")
PY = sys.executable


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    os.makedirs("runs", exist_ok=True)
    cells = json.load(open("north_coast_cells.json", encoding="utf-8"))
    mx = int(os.environ.get("MAX_CELLS", "0")) or len(cells)
    cells = cells[:mx]
    log(f"CAMPANA: {len(cells)} celdas, {N_WORKERS} workers")

    conn = None
    running = {}           # name -> Popen

    def reap(block=False):
        while True:
            done = [n for n, p in running.items() if p.poll() is not None]
            for n in done:
                rc = running.pop(n).returncode
                res = f"runs/{n}/marea/result.json"
                estado = "?"
                if os.path.exists(res):
                    try:
                        r = json.load(open(res, encoding="utf-8"))
                        estado = r.get("estado", "?")
                        if estado == "ok":
                            estado += (" OPERADOR" if r.get("con_operador")
                                       else " identidad")
                    except Exception:
                        pass
                log(f"FIN {n}: rc={rc} estado={estado}")
            if not block or len(running) < N_WORKERS:
                return
            time.sleep(20)

    for i, cell in enumerate(cells):
        name = cell["name"]
        out = os.path.join("runs", name, "marea")
        if os.path.exists(os.path.join(out, "result.json")):
            log(f"SKIP {name}: ya procesada")
            continue
        cube = f"ndwi_cube_{name}.nc"
        if not os.path.exists(cube):
            try:
                if conn is None:
                    from pyintertidal.scenes import connect
                    conn = connect(interactive=False)
                from pyintertidal.aoi import AOI
                from pyintertidal.cube import SentinelCube
                log(f"DESCARGA {name} ({cell.get('km2', '?')} km2)...")
                aoi = AOI.from_polygon(cell["polygon"], name=name)
                c = SentinelCube(aoi, ("2023-01-01", "2025-12-31"),
                                 water="ndwi", cache_path=cube,
                                 resolution=10)
                c.ensure(connection=conn)
                log(f"DESCARGADO {name}")
            except Exception as e:
                log(f"ERROR descarga {name}: {str(e)[:200]}")
                conn = None          # token/conexion pueden haber caducado
                continue
        reap(block=True)             # hueco en el pool antes de lanzar
        code = (f"from pyintertidal import marea; "
                f"marea.reconstruct({cube!r}, {out!r}, name={name!r})")
        running[name] = subprocess.Popen(
            [PY, "-c", code], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        log(f"PROCESANDO {name} ({i+1}/{len(cells)})")

    while running:
        reap(block=False)
        time.sleep(20)
    log("CAMPANA COMPLETA")


if __name__ == "__main__":
    main()
