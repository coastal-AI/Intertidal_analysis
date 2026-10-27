#!/usr/bin/env bash
# Runs the attribution notebook 05_error_attribution_site.ipynb on the server for every site in parallel (one
# kernel each, the site passed as NB_SITE).
#
#   bash notebooks/validation/run_05_on_server.sh                 # escalda wadden ems
#   bash notebooks/validation/run_05_on_server.sh escalda         # one site
#
# Passed through to the notebook when set: NB_NS, NB_METHODS, NB_EXTRA, NB_NTABLES, NB_FIRST, NB_BED, NB_ROW_YEARS,
# MAREA_DEA_TMP, OMP_NUM_THREADS. Executed copies (and HTML when nbconvert can) go to results/attribution_nb/ (NB_OUT),
# with the CSVs, the per-pixel npz and the figures; logs and STATUS to logs/v2/claude/nb05/ (NB_LOG); the notebook is
# read from notebooks/validation/ (NB_DIR). Its last cell checks the rows it shares with notebook 01 against
# results/fixedcfg_nb_diag/<site>.csv (else results/fixedcfg_nb/<site>.csv), bit for bit.
set -u
REPO=/media/HDD/jalias/Intertidal_analysis
cd "$REPO" || exit 1
source env.sh > /dev/null

SITES=${*:-escalda wadden ems}
NB=${NB_DIR:-notebooks/validation}
OUT=${NB_OUT:-results/attribution_nb}
LOG=${NB_LOG:-logs/v2/claude/nb05}
export NB_OUT=$OUT                                                   # the notebook writes there
mkdir -p "$OUT" "$LOG"
export MAREA_DEA_TMP=${MAREA_DEA_TMP:-/media/HDD/jalias/v19_tmp}     # DEA refuses a temp root inside the repo
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
    NB_SITE=$s execute "$NB/05_error_attribution_site.ipynb" "$OUT/05_$s.executed.ipynb" "$LOG/05_$s.log"
    rc=$?
    to_html "$OUT/05_$s.executed.ipynb" "$LOG/05_$s.log"
    echo "$s $([ $rc -eq 0 ] && echo ok || echo FAILED) ($(( ($(date +%s) - t0) / 60 )) min) $(date -Is)" >> "$STATUS"
  ) &
done
wait
echo "finished $(date -Is)" >> "$STATUS"
