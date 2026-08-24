"""Re-execute the study notebook so its outputs match its code.

The uncertainty map stored in the notebook covers the whole polygon, land
included. The code no longer does that — `fit_hsr` masks every per-pixel
product to the same valid set — so the figure is simply older than the fix.
A notebook whose saved outputs predate its own source is worse than one with
no outputs at all: it looks authoritative and is wrong.

Everything downstream of the tide model also has to be recomputed, because
the default moved from GOT4.10 to EOT20 today.

The cube is cached, so nothing is downloaded except the IGN LiDAR tile.
Executed in place: the notebook is the deliverable, not a copy of it.
"""
import io
import shutil
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis"
NB = ROOT + r"\intertidal_topography_villaviciosa.ipynb"
BACKUP = ROOT + r"\_superseded\before_rerun.ipynb"

shutil.copy(NB, BACKUP)
print(f"backup -> {BACKUP}", flush=True)

nb = nbformat.read(io.open(NB, encoding="utf-8"), as_version=4)
print(f"{len(nb.cells)} cells; executing…", flush=True)

t0 = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                        resources={"metadata": {"path": ROOT}},
                        allow_errors=True)   # keep going; report at the end
try:
    client.execute()
finally:
    nbformat.write(nb, io.open(NB, "w", encoding="utf-8"))
    print(f"\nwritten back after {(time.time() - t0) / 60:.1f} min", flush=True)

errors = []
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            errors.append((i, o.get("ename"), (o.get("evalue") or "")[:160]))
if errors:
    print(f"\n{len(errors)} cell(s) raised:")
    for i, name, val in errors:
        print(f"  cell {i}: {name}: {val}")
else:
    print("\nevery cell executed cleanly")
