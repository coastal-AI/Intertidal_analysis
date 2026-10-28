#!/usr/bin/env bash
# Runs the n-tables notebooks on the server: 01_fixed_config_n_tables_site.ipynb for every site in
# parallel (one kernel each, the site passed as NB_SITE), then 01b_fixed_config_n_tables_summary.ipynb.
#
#   bash notebooks/validation/run_01_on_server.sh                 # escalda wadden ems
#   bash notebooks/validation/run_01_on_server.sh escalda         # one site
#
# The confirmatory windows (n = 125, 100, 75, with the pre-2023 images) in a folder of their own, checked against
# the diagnostic run on the windows that run computed:
#   NB_NS=125,100,75 NB_OUT=results/fixedcfg_nb_conf NB_FIRST=results/fixedcfg_nb_diag NB_LOG=logs/v2/claude/nb01_conf \
#     bash notebooks/validation/run_01_on_server.sh escalda wadden
#
# The Spanish sites, with their PNOA-LiDAR truth (experiments/pnoa3_truth.py; the date defaults to the truth_date of
# the GeoTIFF's JSON), one site per call since each has its own GeoTIFF:
#   NB_LIDAR=data_v4/truth/pnoa/ferrol_pnoa3_10m.tif bash notebooks/validation/run_01_on_server.sh ferrol
#
# Passed through to the notebook when set: NB_NS, NB_METHODS, NB_VARIANTS, NB_EXTRA, NB_FIRST, NB_LIDAR,
# NB_LIDAR_DATE, MAREA_DEA_TMP, OMP_NUM_THREADS. Executed copies (and HTML when nbconvert can) go to
# results/fixedcfg_nb_diag/ (NB_OUT; the first run's results/fixedcfg_nb/ is kept, and the notebook checks its nine methods against it), logs and
# STATUS to logs/v2/claude/nb01_diag/ (NB_LOG); the notebooks are read from notebooks/validation/ (NB_DIR).
set -u
REPO=/media/HDD/jalias/Intertidal_analysis
cd "$REPO" || exit 1
source env.sh > /dev/null

SITES=${*:-escalda wadden ems}
NB=${NB_DIR:-notebooks/validation}
OUT=${NB_OUT:-results/fixedcfg_nb_diag}
LOG=${NB_LOG:-logs/v2/claude/nb01_diag}
export NB_OUT=$OUT                                                   # the notebooks write there
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
    NB_SITE=$s execute "$NB/01_fixed_config_n_tables_site.ipynb" "$OUT/01_$s.executed.ipynb" "$LOG/01_$s.log"
    rc=$?
    to_html "$OUT/01_$s.executed.ipynb" "$LOG/01_$s.log"
    echo "$s $([ $rc -eq 0 ] && echo ok || echo FAILED) $(date -Is)" >> "$STATUS"
  ) &
done
wait

execute "$NB/01b_fixed_config_n_tables_summary.ipynb" "$OUT/01b_summary.executed.ipynb" "$LOG/01b_summary.log"
rc=$?
to_html "$OUT/01b_summary.executed.ipynb" "$LOG/01b_summary.log"
echo "summary $([ $rc -eq 0 ] && echo ok || echo FAILED) $(date -Is)" >> "$STATUS"
echo "finished $(date -Is)" >> "$STATUS"
