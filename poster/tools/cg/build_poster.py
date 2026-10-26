# -*- coding: utf-8 -*-
"""Build the CARTOGRASS 2026 poster (poster/cartograss2026.tex) and check it.

Run from the repository root:

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 poster/tools/cg/build_poster.py
        [--dpi 25] [--no-copy] [--quiet] [--final] [--print]

What it does
    1. pdflatex twice (every panel is placed with `remember picture', so the
       first pass only records positions), output in poster/build/.
    2. Reads the log lines written by content/cg/cg_macros.tex and prints the
       geometry of every panel (CGPANEL), the fill of every row (CGROW), the
       figure slots (CGFIG), the placeholders still on the sheet (CGRESERVED,
       CGPHOTO, CGLOGO) and which texts came from content/cg/text_*.tex
       (CGTEXT).
    3. Counts as a problem: any OVERFLOW; an overfull or underfull \\hbox or
       \\vbox; a LaTeX, font or package warning; a missing character (a glyph
       the font lacks vanishes silently); a PDF that is not one A0 page.
       With --final, also every figure that is MISSING or a DRAFT, every
       text that fell back to its draft, and every placeholder (reserved
       slot, photo or logo frame, QR frame): the print build must have none.
    4. Only when there is no problem: copies the PDF to
       poster/CARTOGRASS2026_poster_draft.pdf (unless --no-copy). A failed
       build never overwrites the shareable draft.
    5. Renders poster/build/cartograss2026_preview.png at --dpi (25 by
       default) with pdftocairo (poppler's cairo backend; its Splash backend,
       pdftoppm, draws zipper edges on dense curves, drops dashes and doubles
       some labels, none of which is in the PDF).
    6. With --print: a Ghostscript pass (MiKTeX's mgs) writes
       poster/build/cartograss2026_print.pdf. matplotlib names every subset of
       a font after a hash of the whole font, so the figures embed several
       different subsets under one name (e.g. BEATAY+FiraSans-Book), which a
       RIP that caches fonts by name may mix up; Ghostscript re-subsets with a
       unique tag each. Images are passed through losslessly. The script then
       checks the page size, the tags and the fonts of the print file and,
       when the whole build is clean, copies it to
       poster/CARTOGRASS2026_poster_print.pdf: THE FILE TO SEND TO THE PRINT
       SHOP (the draft PDF is the pdfLaTeX output, with the shared subset
       names). The page is the trim size (A0, no bleed): ask the shop whether
       the full-bleed bands (navy header, pale footer) need 3 mm of bleed.
       Ghostscript writes to build/cartograss2026_print.tmp.pdf; only a
       complete file (exit 0, a PDF trailer at its end) replaces the print
       file, a failed run is retried once and its output shown, and the
       previous good print file is never overwritten by a partial one.

Nothing here touches data; it needs MiKTeX only (pdflatex, pdfinfo,
pdftocairo, pdffonts, mgs).
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys

MIK = r"C:\Users\Jorge\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
POSTER = os.path.join("poster")
JOB = "cartograss2026"
# -max-print-line keeps every \typeout report on one log line (MiKTeX option).
FLAGS = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
         "-disable-installer", "-max-print-line=1000"]
# Log patterns that are problems in any build.
PROBLEMS = {
    "overfull hbox": r"Overfull \\hbox \(([\d.]+)pt too wide\)",
    "overfull vbox": r"Overfull \\vbox \(([\d.]+)pt too high\)",
    "underfull hbox": r"Underfull \\hbox",
    "underfull vbox": r"Underfull \\vbox",
    "LaTeX warning": r"LaTeX Warning:",
    "font warning": r"LaTeX Font Warning:",
    "package warning": r"Package \S+ Warning:",
    "missing character": r"Missing character:",
}


def locked(path: str) -> bool:
    """True when another program holds the file open for writing (a PDF viewer
    on Windows keeps the file it shows locked, and pdflatex then cannot write
    it)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r+b"):
            return False
    except OSError:
        return True


def run_tex(passno: int, job: str) -> None:
    """One pdflatex pass of the sheet into poster/build/; on failure print the
    error lines and exit."""
    exe = os.path.join(MIK, "pdflatex.exe")
    cmd = [exe, *FLAGS, "-output-directory=build", f"-jobname={job}", JOB + ".tex"]
    p = subprocess.run(cmd, cwd=POSTER, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        log = os.path.join(POSTER, "build", job + ".log")
        print(f"pdflatex pass {passno} FAILED (exit {p.returncode}); see {log}")
        for line in p.stdout.splitlines():
            if re.match(r"^.*:\d+:", line) or line.startswith("!"):
                print("  " + line)
        sys.exit(1)


def report(log_text: str, quiet: bool, final: bool) -> int:
    """Print the geometry the macros logged (CGPANEL, CGROW, CGFIG, CGTEXT and
    the placeholders) and return the number of problems: overflows, and with
    ``final`` also drafts, missing figures and placeholders left on the sheet."""
    fails = 0
    panels = re.findall(r"CGPANEL: (\S+) x=([\d.]+) y=([\d.]+)\s+w=([\d.]+) "
                        r"h=([\d.]+) natural=([\d.]+) \(mm\) (\w+)", log_text)
    rows = re.findall(r"CGROW: (\S+) uses ([\d.]+) of\s+([\d.]+) mm \(top at "
                      r"([\d.]+) mm\) (\w+)", log_text)
    figs = re.findall(r"CGFIG: (\S+) slot=([\d.]+) x ([\d.]+) mm found=(\w+)",
                      log_text)
    texts = re.findall(r"CGTEXT: (\S+) (from text file|FALLBACK)", log_text)
    if not quiet:
        print(f"{'panel':18s} {'x':>6s} {'y':>7s} {'w':>6s} {'h':>6s} "
              f"{'used':>6s}  fill")
    for pid, x, y, w, h, n, st in panels:
        fill = 100 * float(n) / float(h) if float(h) else 0
        if st != "ok":
            fails += 1
        if not quiet or st != "ok":
            print(f"{pid:18s} {x:>6s} {y:>7s} {w:>6s} {h:>6s} {n:>6s} "
                  f"{fill:5.0f}%  {'' if st == 'ok' else st}")
    if not quiet:
        print()
    for name, used, budget, top, st in rows:
        if st != "ok":
            fails += 1
        if not quiet or st != "ok":
            print(f"row {name:12s} top {top:>6s} mm  uses {used:>6s} of "
                  f"{budget:>6s} mm  {st}")
    seen = {}
    for stem, w, h, found in figs:
        seen.setdefault(stem, (w, h, found))
    if not quiet:
        print()
        for stem, (w, h, found) in seen.items():
            print(f"figure {stem:24s} slot {w:>6s} x {h:>6s} mm  {found}")
    fb = sorted({t for t, src in texts if src == "FALLBACK"})
    fromfile = sorted({t for t, src in texts if src != "FALLBACK"})
    missing = [s for s, (_, _, f) in seen.items() if f == "MISSING"]
    draft = [s for s, (_, _, f) in seen.items() if f == "DRAFT"]
    reserved = re.findall(r"CGRESERVED: slot=([\d.]+) x ([\d.]+) mm", log_text)
    photos = re.findall(r"CGPHOTO: (\S+) slot=([\d.]+) x ([\d.]+) mm", log_text)
    logos = re.findall(r"CGLOGO: (\S+) slot=", log_text)
    other = re.findall(r"CGPLACEHOLDER: ([^\n]*)", log_text)
    # placeholders: what each pass logs once (the log holds the last pass)
    have_photo = [p for p in photos if not any(
        os.path.exists(os.path.join(POSTER, "figures", "cg", p[0] + e)) for e in (".jpg", ".png", ".pdf"))]
    have_logo = [g for g in logos if not any(
        os.path.exists(os.path.join(POSTER, "figures", "cg", g + e)) for e in (".png", ".pdf"))]
    print(f"\ntexts from content/cg/text_*.tex: {len(fromfile)}; drafts (macro missing): "
          f"{len(fb)} {' '.join(fb)}")
    print(f"figures MISSING: {len(missing)} {' '.join(missing)}; DRAFT: {len(draft)} {' '.join(draft)}")
    print(f"placeholders: {len(reserved)} reserved slot(s) "
          f"[{', '.join(f'{w} x {h}' for w, h in reserved)} mm], "
          f"{len(have_photo)} photo slot(s) [{', '.join(f'{s} {w} x {h}' for s, w, h in have_photo)} mm], "
          f"{len(set(have_logo))} logo frame(s) [{', '.join(sorted(set(have_logo)))}], "
          f"{len(other)} other [{'; '.join(o.strip() for o in other)}]")
    if final:
        n = (len(fb) + len(missing) + len(draft) + len(reserved) + len(have_photo)
             + len(set(have_logo)) + len(other))
        if n:
            print(f"--final: {n} draft(s) or placeholder(s) left on the sheet")
            fails += n
    m = re.search(r"CGSUMMARY: (\d+) overflow", log_text)
    if m and int(m.group(1)) and not fails:
        fails = int(m.group(1))
    return fails


def log_problems(log_text: str) -> int:
    """Count and print the log lines matching :data:`PROBLEMS` (boxes, warnings,
    missing characters)."""
    fails = 0
    for name, pat in PROBLEMS.items():
        hits = re.findall(pat + r"[^\n]*", log_text)
        if hits:
            print(f"{len(hits)} x {name}: " + " | ".join(h.strip()[:90] for h in hits[:4]))
            fails += len(hits)
    if not fails:
        print("log: no overfull/underfull box, no LaTeX/font/package warning, no missing character")
    return fails


def pdf_page_ok(pdf: str) -> bool:
    """True when the PDF is exactly one A0 page (pdfinfo)."""
    info = subprocess.run([os.path.join(MIK, "pdfinfo.exe"), pdf],
                          capture_output=True, text=True).stdout
    size_ok = re.search(r"Page size:\s+2383\.9\d* x 3370\.3\d*", info)
    pages_ok = re.search(r"Pages:\s+1\b", info)
    print(f"PDF {os.path.basename(pdf)}: {'A0' if size_ok else 'NOT A0'}, "
          f"{'one page' if pages_ok else 'NOT one page'}, {os.path.getsize(pdf) / 1024:.0f} KB")
    return bool(size_ok and pages_ok)


def fonts(pdf: str):
    """The fonts of a PDF as (name, type, embedded, subset, object id) rows (pdffonts)."""
    out = subprocess.run([os.path.join(MIK, "pdffonts.exe"), pdf],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    rows = []
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 7:
            # name, type (may be two words), emb, sub, uni, object id
            name = parts[0]
            emb, sub = parts[-5], parts[-4]
            typ = " ".join(parts[1:-6])
            rows.append((name, typ, emb, sub, parts[-2]))
    return rows


def _complete_pdf(path: str) -> bool:
    """True when ``path`` exists and ends with a PDF trailer (%%EOF in its last
    1024 bytes): a Ghostscript run that dies half-way leaves a file cut off
    after its first content stream, with no trailer."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 1024))
            return b"%%EOF" in f.read()
    except OSError:
        return False


def print_pass(pdf: str, retries: int = 1) -> int:
    """Ghostscript pdfwrite pass for the print shop (see the module docstring).

    Ghostscript writes to a temporary name first; the result replaces
    build/cartograss2026_print.pdf only when Ghostscript exits 0 and the file
    is a complete PDF (a trailer at its end). On failure the partial file is
    deleted, the previous good print file is left as it was, and both of
    Ghostscript's streams are shown (mgs writes its errors to stdout). One
    failed run is retried once: a run of 2026-10-19 died with nothing on
    stderr and the same command then succeeded four times by hand (the
    system drive was nearly full)."""
    out = os.path.join(POSTER, "build", JOB + "_print.pdf")
    tmp = os.path.join(POSTER, "build", JOB + "_print.tmp.pdf")
    gs = os.path.join(MIK, "mgs.exe")
    cmd = [gs, "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dQUIET", "-sDEVICE=pdfwrite",
           "-dPDFSETTINGS=/prepress", "-dAutoFilterColorImages=false",
           "-dColorImageFilter=/FlateEncode", "-dDownsampleColorImages=false",
           "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/FlateEncode",
           "-dDownsampleGrayImages=false", "-dDownsampleMonoImages=false",
           f"-sOutputFile={tmp}", pdf]
    for attempt in range(1 + retries):
        if os.path.exists(tmp):
            os.remove(tmp)
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode == 0 and _complete_pdf(tmp):
            break
        tail = ((p.stdout or "") + (p.stderr or "")).strip()[-800:]
        size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        print(f"print pass FAILED (exit {p.returncode}, {size} bytes written, "
              f"{'complete' if _complete_pdf(tmp) else 'no PDF trailer'}): {tail or '(no output)'}")
        if os.path.exists(tmp):
            os.remove(tmp)
        if attempt < retries:
            print("print pass: retrying once")
    else:
        print(f"print pass: {out} left as it was")
        return 1
    if locked(out):
        os.remove(tmp)
        print(f"print pass: {out} is open in another program (locked): NOT updated")
        return 1
    os.replace(tmp, out)
    fails = 0 if pdf_page_ok(out) else 1
    rows = fonts(out)
    by_name = collections.defaultdict(set)
    for name, typ, emb, sub, obj in rows:
        by_name[name].add(obj)
    dup = {n: o for n, o in by_name.items() if len(o) > 1}
    t3 = [r for r in rows if "Type 3" in r[1]]
    unemb = [r for r in rows if r[2] != "yes"]
    print(f"print file {out}: {len(rows)} fonts, {len(dup)} name(s) shared by several "
          f"subsets, {len(t3)} Type 3, {len(unemb)} not embedded")
    fails += len(dup) + len(t3) + len(unemb)
    return fails


def main() -> None:
    """Build, check, preview and (with no problem) copy the poster; see the
    module docstring for the options."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=25)
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="also fail on drafts, missing figures and placeholders")
    ap.add_argument("--print", dest="print_pdf", action="store_true",
                    help="also write build/cartograss2026_print.pdf (Ghostscript pass)")
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(POSTER, JOB + ".tex")):
        sys.exit("run from the repository root (poster/cartograss2026.tex not found)")
    os.makedirs(os.path.join(POSTER, "build"), exist_ok=True)
    job = JOB
    if locked(os.path.join(POSTER, "build", JOB + ".pdf")):
        job = JOB + "-b"
        print(f"build/{JOB}.pdf is open in another program (locked): building as build/{job}.pdf")
    for k in (1, 2):
        run_tex(k, job)

    log_path = os.path.join(POSTER, "build", job + ".log")
    with open(log_path, encoding="latin-1") as f:
        log_text = f.read()
    fails = report(log_text, a.quiet, a.final)
    print()
    fails += log_problems(log_text)

    pdf = os.path.join(POSTER, "build", job + ".pdf")
    if not pdf_page_ok(pdf):
        fails += 1

    stem = os.path.join(POSTER, "build", JOB + "_preview")
    subprocess.run([os.path.join(MIK, "pdftocairo.exe"), "-r", str(a.dpi),
                    "-png", "-singlefile", pdf, stem],
                   capture_output=True, text=True)
    print(f"preview {stem}.png at {a.dpi} dpi (pdftocairo)")

    if a.print_pdf:
        fails += print_pass(pdf)

    if fails:
        print(f"\n{fails} problem(s); the shareable draft was NOT updated")
        sys.exit(1)
    if not a.no_copy:
        dst = os.path.join(POSTER, "CARTOGRASS2026_poster_draft.pdf")
        shutil.copyfile(pdf, dst)
        print(f"copied to {dst}")
        if a.print_pdf:
            # the print shop's file: the Ghostscript pass (unique font subsets)
            src = os.path.join(POSTER, "build", JOB + "_print.pdf")
            dst_p = os.path.join(POSTER, "CARTOGRASS2026_poster_print.pdf")
            if locked(dst_p):
                print(f"{dst_p} is open in another program (locked): NOT updated")
            else:
                shutil.copyfile(src, dst_p)
                print(f"print file copied to {dst_p} (send this one to the printer)")
    print("\nall panels within budget, no problem in the log")


if __name__ == "__main__":
    main()
