#!/bin/bash
# Build the manuscript. Compiles into build/ (so an open main.pdf does not
# block the run), then copies the result to main.pdf and verifies the copy.
# If main.pdf is held open by a viewer (Windows locks it), the fresh copy
# goes to main_latest.pdf instead and the message says so.
# MiKTeX ships a sample "references.bib" that shadows any local file of that
# name when BibTeX runs from another directory: our file is marea.bib and
# BIBINPUTS points BibTeX back at this folder.
set -e
cd "$(dirname "$0")"
mkdir -p build
BIBINPUTS="$(pwd -W 2>/dev/null || pwd);" \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
if grep -q "Warning--" build/main.blg; then grep "Warning--" build/main.blg; fi
echo "unresolved citations: $(pdftotext build/main.pdf - | grep -c '(?)' || true)"

PY=/c/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe
check() { "$PY" -c "import pypdf,sys; r=pypdf.PdfReader(sys.argv[1], strict=True); print(sys.argv[1], 'ok,', len(r.pages), 'pages')" "$1"; }

if rm -f main.pdf 2>/dev/null && cp build/main.pdf main.pdf 2>/dev/null && check main.pdf; then
  rm -f main_latest.pdf
else
  cp build/main.pdf main_latest.pdf && check main_latest.pdf
  echo "main.pdf is open in a viewer and could not be replaced; open main_latest.pdf"
fi
