#!/usr/bin/env bash
# Runs the validation v3 production notebook v3_production_site.ipynb on the server, one kernel per truth (its key
# passed as NB_V3; experiments/v3/sites.py), in parallel.
#
#   bash notebooks/validation/run_v3_production_on_server.sh                    # every new v3 cube
#   bash notebooks/validation/run_v3_production_on_server.sh foz_2016           # one
#
# Executed copies (and HTML when nbconvert can) go to results/v3_production/ (NB_OUT); logs and STATUS to
# logs/v2/claude/v3_production/ (NB_LOG); the products to each truth's products_v3_<key>_10m/.
set -u
REPO=/media/HDD/jalias/Intertidal_analysis
cd "$REPO" || exit 1
source env.sh > /dev/null

SITES=${*:-urdaibai_2017 urdaibai_2024 foz_2016 eo_2020}
NB=${NB_DIR:-notebooks/validation}
OUT=${NB_OUT:-results/v3_production}
LOG=${NB_LOG:-logs/v2/claude/v3_production}
export NB_OUT=$OUT                                                   # the notebook writes there
mkdir -p "$OUT" "$LOG"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
STATUS=$LOG/STATUS
echo "started $(date -Is) on $(hostname): sites $SITES, commit $(git rev-parse --short HEAD)" > "$STATUS"

# execute <notebook> <executed copy> <log>: runs every cell in a fresh kernel from the repository root,
# prints the cells' printed output to the log as it comes, and saves the executed copy even when a cell fails
execute() {
  "$PY" -X utf8 - "$1" "$2" > "$3" 2>&1 <<'EOF'
import sys, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class LoggedClient(NotebookClient):
    def output(self, outs, msg, display_id, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().output(outs, msg, display_id, cell_index)


src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
t0, ok = time.time(), True
try:
    LoggedClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
except CellExecutionError as e:
    ok = False
    print(e)
finally:
    nbformat.write(nb, dst)
print(f"\n{'OK' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min -> {dst}")
sys.exit(0 if ok else 1)
EOF
}

to_html() {
  "$PY" -m nbconvert --to html --output-dir "$OUT" "$1" >> "$2" 2>&1 || echo "html: nbconvert failed for $1" >> "$2"
}

for s in $SITES; do
  (
    t0=$(date +%s)
    NB_V3=$s execute "$NB/v3_production_site.ipynb" "$OUT/v3_production_$s.executed.ipynb" "$LOG/v3_production_$s.log"
    rc=$?
    to_html "$OUT/v3_production_$s.executed.ipynb" "$LOG/v3_production_$s.log"
    echo "$s $([ $rc -eq 0 ] && echo ok || echo FAILED) ($(( ($(date +%s) - t0) / 60 )) min) $(date -Is)" >> "$STATUS"
  ) &
done
wait
echo "finished $(date -Is)" >> "$STATUS"
