# -*- coding: utf-8 -*-
"""CARTOGRASS 2026 poster, Methods: the MAREA pipeline in five steps.

Compiles the TikZ source poster/tools/cg/cg_method_pipeline.tex (next to this
script), drawn at print size, and writes a vector PDF (the poster's Type 1
fonts) and a 300 dpi PNG:

    --preset slot (default)  391 x 104 mm, the slot \\figPipeW x \\figPipeH of
                             poster/cartograss2026.tex
                             -> poster/figures/cg/method_pipeline.{pdf,png}
    --preset wide            520 x 121 mm, the width first planned for the row
                             -> poster/figures/cg/method_pipeline_wide.{pdf,png}

Type at print size: titles 22 pt (wide: 26), step text 20 pt (wide: 22),
labels in the icons 20 pt. No data is read: the icons are schematics.

Run from the repository root (a few seconds):

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/fig_method_pipeline.py [--preset wide]

pdflatex runs from poster/ (the source inputs preamble/colors.tex) with the
flags tools/build.ps1 documents as load-bearing plus -disable-installer, so a
missing package fails instead of opening a MiKTeX dialog. Intermediate files
go to poster/build/cg/ (gitignored). The PNG is rasterised by pdftocairo:
poppler's Splash backend (pdftoppm) leaves a 2-px seam on one fill edge.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.getcwd()
if not os.path.isfile(os.path.join(ROOT, "poster", "tools", "cg", "cg_method_pipeline.tex")):
    sys.exit("run this script from the repository root")

MIKTEX = r"C:\Users\Jorge\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
POSTER = os.path.join(ROOT, "poster")
SRC = "tools/cg/cg_method_pipeline.tex"       # relative to poster/ (kept out of
                                              # tikz/, which tools/build.ps1 compiles whole)
BUILD = os.path.join("build", "cg")           # relative to poster/
OUT_DIR = os.path.join(POSTER, "figures", "cg")
PRESETS = {"slot": ("method_pipeline", 391.0, 104.0),
           "wide": ("method_pipeline_wide", 520.0, 121.0)}


def main():
    """Compile the TikZ source at the size of ``--preset`` and write the PDF
    and the 300 dpi PNG (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="slot")
    args = ap.parse_args()
    stem, want_w, want_h = PRESETS[args.preset]
    job = "cg_" + stem

    os.makedirs(os.path.join(POSTER, BUILD), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    tex_arg = r"\def\cgPreset{%s}\input{%s}" % (args.preset, SRC)
    cmd = [os.path.join(MIKTEX, "pdflatex.exe"), "-interaction=nonstopmode",
           "-halt-on-error", "-file-line-error", "-disable-installer",
           f"-output-directory={BUILD}", f"-jobname={job}", tex_arg]
    run = subprocess.run(cmd, cwd=POSTER, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    log = os.path.join(POSTER, BUILD, job + ".log")
    if run.returncode != 0:
        errs = [ln for ln in open(log, encoding="utf-8", errors="replace")
                if re.match(r"^.*:\d+:", ln) or ln.startswith("!")]
        sys.exit("pdflatex failed:\n" + "".join(errs[:12]))
    text = open(log, encoding="utf-8", errors="replace").read()
    for pat in ("Overfull", "Underfull", "Font Warning", "Missing character"):
        n = text.count(pat)
        if n:
            print(f"warning: {n} x '{pat}' in {log}")

    pdf = os.path.join(OUT_DIR, stem + ".pdf")
    shutil.copyfile(os.path.join(POSTER, BUILD, job + ".pdf"), pdf)
    info = subprocess.run([os.path.join(MIKTEX, "pdfinfo.exe"), pdf],
                          capture_output=True, text=True).stdout
    w_pt, h_pt = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)",
                                      info).groups())
    w_mm, h_mm = w_pt / 72 * 25.4, h_pt / 72 * 25.4
    print(f"page {w_mm:.1f} x {h_mm:.1f} mm")
    if abs(w_mm - want_w) > 0.5 or abs(h_mm - want_h) > 0.5:
        sys.exit(f"page is {w_mm:.1f} x {h_mm:.1f} mm, not "
                 f"{want_w:.0f} x {want_h:.0f} mm")

    subprocess.run([os.path.join(MIKTEX, "pdftocairo.exe"), "-r", "300",
                    "-png", "-singlefile", pdf, os.path.join(OUT_DIR, stem)],
                   check=True, stderr=subprocess.DEVNULL)
    print(f"wrote {pdf} and {os.path.join(OUT_DIR, stem + '.png')}")


if __name__ == "__main__":
    main()
