#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The validation v3 report: the Cantabrian estuaries against the PNOA LiDAR (the design
experiments/validation_v3_cantabrian_2026-10-29.md): per truth the error tables and curves against n, the
class-dominance sensitivity, and the per-pixel error maps with the decomposition of the error.

    C:/Users/Jorge/sketch_fitton/.venv/Scripts/python.exe -X utf8 reports/validacion_v3/build_v3.py
    ... build_v3.py --fetch        # first copies the server's results (results/v3_nb05/) under BASE
    ... build_v3.py --job NAME     # writes NAME.pdf (when a viewer locks the usual one)

What it does, in order:
  1. (--fetch) copies notebook 05's v3 outputs of every truth from the server into BASE (CSV and PNG);
  2. reads ONLY those files, with their sha256: <truth>_scores.csv (the scores at q = 0.10 on each method's pixels),
     <truth>_decomposition.csv (own and common pixels), <truth>_class_sensitivity.csv and the maps;
  3. writes per truth the tables of RMSE, bias and centred RMSE (rows n, columns the design's methods: the n-tables'
     ten, MAREA-B, MAREA-BL and MAREA-S) on each method's pixels and the RMSE on the common pixels, the curves against
     n (one figure, the MAREA family in colour), the sensitivity to q, and the value macros;
  4. compiles main_v3.tex with latexmk into validacion_v3.pdf.
The palette, the number formatting and the bold rule are build_report.py's (imported; its main() is not run).
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "resultados_2026-10"))
import build_report as br  # noqa: E402  (sets the matplotlib backend and style)
import matplotlib.pyplot as plt  # noqa: E402

REPO = br.REPO
FIG, TAB = HERE / "figures", HERE / "tables"
JOB = sys.argv[sys.argv.index("--job") + 1] if "--job" in sys.argv else "validacion_v3"
BASE = REPO / "results" / "server_logs_v3"                 # local copies of the server's results/v3_nb05/
SERVER = "jalias@156.35.105.23:/media/HDD/jalias/Intertidal_analysis/results/v3_nb05"
SERVER_MAPS = "jalias@156.35.105.23:/media/HDD/jalias/Intertidal_analysis/results/v3_maps"
MAP_SITES = {"santander_maps": "Santander", "santona_maps": "Santoña"}
SSH = ["-i", str(Path.home() / ".ssh" / "nserver1_claude"), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"]

TRUTHS = ["villaviciosa_2020", "eo_2020", "foz_2016", "urdaibai_2017", "urdaibai_2024"]
TRUTH_NAME = {"villaviciosa_2020": "Villaviciosa 2020", "eo_2020": "Eo 2020", "foz_2016": "Foz 2016",
              "urdaibai_2017": "Mundaka 2017", "urdaibai_2024": "Mundaka 2024"}
NS = [20, 40, 60, 80, 100, 120]
FAMILY = ["MAREA", "MAREA_ndwi0", "MAREA_B", "MAREA_BL", "MAREA_Ss"]
RIVALS = ["NIDEM", "DEA", "E4", "E5a", "E5b", "E5c", "E7", "E12"]
METHODS = FAMILY + RIVALS
ABBR = {**br.ABBR, "MAREA_B": "MAREA-B", "MAREA_BL": "MAREA-BL", "MAREA_Ss": "MAREA-S"}
SCORES = [("rmse", "RMSE", "min"), ("bias", "Sesgo", "absmin"), ("crmse", "RMSE centrado", "min")]
GREY = "#9C968B"
CURVE_STYLE = {"MAREA": (br.PAL["accent"], "o", "-", 2.3), "MAREA_ndwi0": ("#E3A33B", "D", ":", 1.9),
               "MAREA_B": ("#6B1F2A", "s", "--", 1.9), "MAREA_BL": ("#D8737F", "^", "-.", 1.9),
               "MAREA_Ss": ("#2F5D7C", "H", "-", 2.1),
               **{m: (GREY, br.STYLE[m][1], br.STYLE[m][2], 0.8) for m in RIVALS}}
READ = {}


def fetch():
    BASE.mkdir(parents=True, exist_ok=True)
    for pat in ("*_scores.csv", "*_decomposition.csv", "*_class_sensitivity.csv", "*_map_decomposition.csv",
                "*_error_maps.png", "*_map_decomposition.png", "*_uncertainty_maps.png", "*_uncertainty_vs_error.png",
                "*_pixels.npz", "*_delay_summary.csv"):
        subprocess.run(["scp", "-q", *SSH, f"{SERVER}/{pat}", str(BASE)], check=False)
    for pat in ("*_elevation_maps.png", "*_elevation_summary.csv"):
        subprocess.run(["scp", "-q", *SSH, f"{SERVER_MAPS}/{pat}", str(BASE)], check=False)


def read(name: str) -> pd.DataFrame | None:
    p = BASE / name
    if not p.exists():
        return None
    READ[name] = (br.sha256(p), None)
    d = pd.read_csv(p)
    READ[name] = (READ[name][0], len(d))
    return d


def load():
    """{truth: (scores, decomposition, sensitivity)} of the truths whose notebook 05 run has finished."""
    out = {}
    for t in TRUTHS:
        sc = read(f"{t}_scores.csv")
        if sc is None:
            continue
        sc = sc[sc.method.isin(METHODS)]
        dec, sens = read(f"{t}_decomposition.csv"), read(f"{t}_class_sensitivity.csv")
        dup = sc[sc.status == "ok"].duplicated(["n", "method"])
        assert not dup.any(), f"{t}: two computed rows of one window"
        out[t] = (sc, dec, sens)
    return out


def cell(d, n, m, score, digits=3):
    q = d[(d.n == n) & (d.method == m)]
    if q.empty:                                         # computed windows lack only the methods that gave no value
        return np.nan, br.DASH if (d.n == n).any() else "?"
    r = q.iloc[0]
    if r.status == "not reachable":
        return np.nan, "n.a."
    if r.status != "ok" or not np.isfinite(r[score]):
        return np.nan, br.DASH
    return float(r[score]), br.num(r[score], digits)


def table(t, d, score, rule, label, common=False):
    """Rows n, columns the methods; the best of a row in bold; n.a. where the window is not reachable."""
    head = " & ".join(ABBR[m] for m in METHODS)
    lines = [r"\begin{tabular}{r" + "r" * len(METHODS) + "}", r"\toprule",
             rf"$n$ & \multicolumn{{{len(FAMILY)}}}{{c}}{{MAREA y versiones}} & "
             rf"\multicolumn{{{len(RIVALS)}}}{{c}}{{otros métodos}}\\",
             rf"\cmidrule(lr){{2-{1 + len(FAMILY)}}}\cmidrule(lr){{{2 + len(FAMILY)}-{1 + len(METHODS)}}}",
             f" & {head}\\\\", r"\midrule"]
    for n in NS:
        raw, txt = zip(*(cell(d, n, m, score) for m in METHODS))
        if all(x in ("n.a.", "?") for x in txt):
            continue
        txt = br.bold_best(list(txt), list(raw), rule)
        lines.append(f"{n} & " + " & ".join(txt) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    name = f"{t}_{score}{'_common' if common else ''}.tex"
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def common_frame(dec):
    """The decomposition's common-pixel rows as a scores frame (status ok where the RMSE exists)."""
    c = dec[(dec.pixels == "common") & dec.method.isin(METHODS)].copy()
    c["status"] = np.where(np.isfinite(c.rmse), "ok", "not computable")
    return c


def fig_curves(data, name):
    """Rows the scores, columns the truths: each method's score against n (own pixels, q = 0.10)."""
    truths = [t for t in TRUTHS if t in data]
    fig, axes = plt.subplots(len(SCORES), len(truths), figsize=(1.9 * len(truths) + 1.2, 7.4), squeeze=False)
    for j, t in enumerate(truths):
        d = data[t][0]
        ok = d[d.status == "ok"]
        for i, (score, title, _) in enumerate(SCORES):
            ax = axes[i, j]
            worst = {m: float(ok[ok.method == m][score].abs().max()) for m in METHODS if (ok.method == m).any()}
            if not worst:
                ax.set_visible(False)
                continue
            cap = sorted(worst.values())[min(8, len(worst)) - 1] * 1.15
            shown = [m for m, v in worst.items() if v <= cap]
            vals = ok[ok.method.isin(shown)][score]
            lo, hi = float(vals.min()), float(vals.max())
            if score == "bias":
                lo, hi = min(lo, 0.0), max(hi, 0.0)
                ax.axhline(0, color="#8F897D", lw=0.6, zorder=1)
            pad = 0.06 * (hi - lo) or 0.01
            for m in METHODS:
                q = ok[ok.method == m].sort_values("n")
                if q.empty:
                    continue
                c, mk, ls, lw = CURVE_STYLE[m]
                mine = m in FAMILY
                ax.plot(q.n, q[score], color=c, marker=mk, ls=ls, lw=lw, ms=3.4 if mine else 2.4,
                        zorder=5 if mine else 2, label=ABBR[m])
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(12, 128)
            ax.set_xticks(NS, [str(n) for n in NS])
            off = [ABBR[m] for m in METHODS if m in worst and m not in shown]
            if off:
                ax.text(0.98, 0.97, "fuera de escala: " + ", ".join(off), transform=ax.transAxes, ha="right",
                        va="top", fontsize=5.0, color=br.PAL["secondary"],
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
            if i == 0:
                ax.set_title(TRUTH_NAME[t])
            if j == 0:
                ax.set_ylabel(f"{title} (m)")
            if i == len(SCORES) - 1:
                ax.set_xlabel("n (imágenes)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fam = [lab in [ABBR[m] for m in FAMILY] for lab in labels]
    for row, (mine, title) in enumerate(((True, "MAREA y sus versiones"), (False, "otros métodos"))):
        hs = [h for h, g in zip(handles, fam) if g == mine]
        ls_ = [lab for lab, g in zip(labels, fam) if g == mine]
        fig.legend(hs, ls_, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, 0.035 - 0.035 * row),
                   handlelength=2.6, columnspacing=1.2, title=title, title_fontsize=6.6, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.0, w_pad=0.8)
    fig.savefig(FIG / name, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(FIG / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def fig_sensitivity(data, name):
    """RMSE on the common pixels against q, at each truth's largest computed window."""
    truths = [t for t in TRUTHS if t in data and data[t][2] is not None and len(data[t][2])]
    if not truths:
        return None
    fig, axes = plt.subplots(1, len(truths), figsize=(1.9 * len(truths) + 1.0, 2.6), squeeze=False)
    for j, t in enumerate(truths):
        s = data[t][2]
        s = s[(s.pixels == "common") & s.method.isin(METHODS)]
        n = int(s.n.max())
        s = s[s.n == n]
        ax = axes[0, j]
        for m in METHODS:
            q = s[s.method == m].sort_values("q")
            if q.empty:
                continue
            c, mk, ls, lw = CURVE_STYLE[m]
            ax.plot(q.q, q.rmse, color=c, marker=mk, ls=ls, lw=lw, ms=3 if m in FAMILY else 2,
                    zorder=5 if m in FAMILY else 2, label=ABBR[m])
        ax.axvline(0.10, color="#8F897D", lw=0.6, ls=":")
        ax.set_title(f"{TRUTH_NAME[t]}, n = {n}")
        ax.set_xlabel("q (umbral de clase dominante)")
        if j == 0:
            ax.set_ylabel("RMSE, píxeles comunes (m)")
    fig.tight_layout()
    fig.savefig(FIG / name)
    fig.savefig(FIG / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return name


DIST_METHODS = [("MAREA", "MAREA_nodelay"), ("MAREA_B", "MAREA_B_nodelay"), ("E4_delay", "E4")]
DIST_RIVALS = ["DEA", "NIDEM", "E5c"]
DIST_LABEL = {"MAREA": "MAREA", "MAREA_nodelay": "MAREA sin retardo", "MAREA_B": "MAREA-B",
              "MAREA_B_nodelay": "MAREA-B sin retardo", "E4_delay": "LT2D con retardo", "E4": "LT2D",
              "DEA": "DEA", "NIDEM": "NIDEM", "E5c": "SE"}


def distance_frame(t):
    """The largest computed window of a truth, per pixel: its distance from the mouth, delay, truth and estimates
    (notebook 05's <truth>_pixels.npz). None without the npz."""
    p = BASE / f"{t}_pixels.npz"
    if not p.exists():
        return None, None
    READ[p.name] = (br.sha256(p), None)
    z = np.load(p)
    ns = sorted({int(k.split("_n")[-1].split("_")[0]) for k in z.files if k.startswith(f"{t}_n") and k.endswith("_keep")})
    if not ns:
        return None, None
    n = ns[-1]
    tag = f"{t}_n{n}"
    s_of = dict(zip(z[f"{t}_keep"].tolist(), z[f"{t}_s_km"].tolist()))
    keep = z[f"{tag}_keep"]
    out = pd.DataFrame({"s_km": [s_of.get(int(k), np.nan) for k in keep], "tau": z[f"{tag}_tau_px"],
                        "truth": z[f"{tag}_truth"]})
    for m in [m for pair in DIST_METHODS for m in pair] + DIST_RIVALS:
        if f"{tag}_{m}" in z.files:
            out[m] = z[f"{tag}_{m}"]
    return n, out


def fig_distance(data, name, bins=5):
    """Per truth, the RMSE and the median error against the distance from the mouth (equal-count bins), each method
    with MAREA's delay (solid) and without it (dashed), the rivals without delay, and the median delay of each bin."""
    truths = [t for t in TRUTHS if t in data]
    frames = {t: distance_frame(t) for t in truths}
    truths = [t for t in truths if frames[t][1] is not None]
    if not truths:
        return None
    fig, axes = plt.subplots(2, len(truths), figsize=(2.3 * len(truths) + 1.0, 5.2), squeeze=False)
    colours = {"MAREA": br.PAL["accent"], "MAREA_B": "#6B1F2A", "E4_delay": "#2F5D7C"}
    rows = []
    for j, t in enumerate(truths):
        n, d = frames[t]
        d = d[np.isfinite(d.s_km) & np.isfinite(d.truth)]
        edges = np.unique(np.percentile(d.s_km, np.linspace(0, 100, bins + 1)))
        d = d.assign(bin=np.clip(np.digitize(d.s_km, edges[1:-1]), 0, len(edges) - 2))
        g = d.groupby("bin")
        x = g.s_km.median().to_numpy()
        series = []
        for a, b in DIST_METHODS:
            series += [(a, colours[a], "-"), (b, colours[a], "--")]
        series += [(m, GREY, ":") for m in DIST_RIVALS]
        for m, c, ls in series:
            if m not in d:
                continue
            e = d[m] - d.truth
            rmse = e.pow(2).groupby(d.bin).mean().pow(0.5).to_numpy()
            med = e.groupby(d.bin).median().to_numpy()
            axes[0, j].plot(x, rmse, color=c, ls=ls, marker="o", ms=2.5, lw=1.4 if ls != ":" else 0.9,
                            label=DIST_LABEL[m])
            axes[1, j].plot(x, med, color=c, ls=ls, marker="o", ms=2.5, lw=1.4 if ls != ":" else 0.9)
            for k in range(len(x)):
                rows.append(dict(truth=t, n=n, bin=k, s_km=x[k], method=m, rmse=rmse[k], bias=med[k]))
        tw = axes[0, j].twinx()
        tw.bar(x, g.tau.median().to_numpy(), width=0.35 * (np.ptp(x) / max(len(x), 1) or 1), color="#D9D4C7",
               alpha=0.6, zorder=0)
        tw.set_ylabel("retardo mediano (min)" if j == len(truths) - 1 else "", fontsize=6.5)
        tw.tick_params(labelsize=6)
        axes[0, j].set_zorder(tw.get_zorder() + 1)
        axes[0, j].patch.set_visible(False)
        axes[0, j].set_title(f"{TRUTH_NAME[t]}, n = {n}")
        axes[1, j].axhline(0, color="#8F897D", lw=0.6)
        axes[1, j].set_xlabel("distancia a la boca (km)")
    axes[0, 0].set_ylabel("RMSE (m)")
    axes[1, 0].set_ylabel("error mediano (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / name, format="pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(FIG / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TAB / "distance.csv", index=False)
    return name


def maps_tex(data):
    """The \\includegraphics of each truth's error maps and decomposition bars (notebook 05's PNGs)."""
    lines = []
    for t in TRUTHS:
        if t not in data:
            continue
        for png in sorted(BASE.glob(f"{t}_n*_error_maps.png"), key=lambda p: int(p.name.split("_n")[1].split("_")[0])):
            n = png.name.split("_n")[1].split("_")[0]
            dst = FIG / png.name
            shutil.copyfile(png, dst)
            lines += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=\textwidth]{{figures/{png.name}}}"]
            bars = BASE / f"{t}_n{n}_map_decomposition.png"
            if bars.exists():
                shutil.copyfile(bars, FIG / bars.name)
                lines.append(rf"\\[4pt]\includegraphics[width=0.62\textwidth]{{figures/{bars.name}}}")
            lines += [rf"\caption{{{TRUTH_NAME[t]}, $n = {n}$: error por píxel $p - t$ (m) de cada método, misma "
                      r"escala para todos ($\pm 0{,}5$\,m; morado, por debajo del LiDAR; naranja, por encima; gris "
                      r"oscuro, fuera del intermareal puntuado), y descomposición del error cuadrático medio en "
                      r"desplazamiento, relieve y dispersión sobre los píxeles comunes.}", r"\end{figure}", ""]
            unc, vs = BASE / f"{t}_n{n}_uncertainty_maps.png", BASE / f"{t}_n{n}_uncertainty_vs_error.png"
            if unc.exists():
                shutil.copyfile(unc, FIG / unc.name)
                lines += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=\textwidth]{{figures/{unc.name}}}"]
                if vs.exists():
                    shutil.copyfile(vs, FIG / vs.name)
                    lines.append(rf"\\[6pt]\includegraphics[width=\textwidth]{{figures/{vs.name}}}")
                lines += [rf"\caption{{{TRUTH_NAME[t]}, $n = {n}$: la incertidumbre de la cota. Arriba, por píxel: "
                          r"el ancho del intervalo de confianza del 95\,\% de MAREA-B (verosimilitud perfilada), su "
                          r"$\sigma$ (lo gradual de la transición seco--mojado), la dispersión entre los métodos del "
                          r"diseño y, para comparar, el $|p - t|$ de MAREA-B. Abajo, el $|p - t|$ de MAREA-B por "
                          r"deciles de cada incertidumbre: si crece, la incertidumbre avisa del error.}",
                          r"\end{figure}", ""]
    (TAB / "maps.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def elevation_maps_tex():
    """The maps-only sites: notebook v3_maps_site's figure of each method's elevation."""
    lines = []
    for key, name in MAP_SITES.items():
        if (BASE / f"{key}_elevation_summary.csv").exists():
            read(f"{key}_elevation_summary.csv")
        for png in sorted(BASE.glob(f"{key}_n*_elevation_maps.png")):
            n = png.name.split("_n")[-1].split("_")[0]
            shutil.copyfile(png, FIG / png.name)
            lines += [r"\begin{figure}[p]\centering", rf"\includegraphics[width=\textwidth]{{figures/{png.name}}}",
                      rf"\caption{{{name}: la cota de cada método con la ventana de las {n} escenas cribadas más "
                      r"cercanas al 15 de julio de 2024 (sin verdad: el LiDAR de esta ría se voló en pleamar). Misma "
                      r"escala de color para todos los métodos; cotas en el datum del contorno de marea.}",
                      r"\end{figure}", ""]
    (TAB / "elevation_maps.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def windows_tex(t, d):
    """The windows of a truth: half-width T, scenes before / after the truth date, first and last scene, pixels."""
    w = d[d.method == "MAREA"].sort_values("n")
    lines = [r"\begin{tabular}{rrrrrllr}", r"\toprule",
             r"$n$ & $T$ (días) & centro desplazado & antes & después & primera escena & última escena & "
             r"píxeles $F$\\", r"\midrule"]
    for _, r in w.iterrows():
        if r.status == "not reachable":
            lines.append(rf"{int(r.n)} & \multicolumn{{7}}{{l}}{{no alcanzable: ninguna ventana con la fecha de "
                         r"verdad cabe en el cubo}\\")
        else:
            shift = int(r.shift_days) if "shift_days" in r.index and np.isfinite(r.shift_days) else 0
            moved = f"{shift:+d} d" if shift else "no"
            lines.append(f"{int(r.n)} & {r.half_width_days:.0f} & {moved} & {int(r.n_before)} & {int(r.n_after)} & "
                         f"{r.first_date} & {r.last_date} & {br.thousands(r.n_F)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{t}_windows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def body_tex(data):
    """One subsection per truth: its windows and its four tables."""
    out = []
    for t in TRUTHS:
        if t not in data:
            continue
        sc = data[t][0]
        date = sc.truth_date.dropna().iloc[0]
        out += [rf"\subsection*{{{TRUTH_NAME[t]} (fecha de verdad {date})}}",
                r"\begin{table}[H]\centering\small", rf"\input{{tables/{t}_windows.tex}}",
                rf"\caption{{{TRUTH_NAME[t]}: las ventanas, las $n$ escenas cribadas más cercanas a la fecha de verdad "
                r"$d$, simétricas en el tiempo ($[d-T, d+T]$); píxeles puntuados con $q = 0{,}10$.}", r"\end{table}"]
        for score, title, _ in SCORES + [("rmse_common", "RMSE en los píxeles comunes", "min")]:
            f = TAB / f"{t}_{score}.tex"
            if not f.exists():
                continue
            pix = "comunes a todos los métodos" if score.endswith("common") else "propios de cada método"
            out += [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
                    rf"\resizebox{{\textwidth}}{{!}}{{\input{{tables/{t}_{score}.tex}}}}",
                    rf"\caption{{{TRUTH_NAME[t]}: {title} (m) frente a $n$, píxeles {pix}; en negrita el mejor de "
                    r"cada fila; n.a.: ventana no alcanzable; ---: el método no da valor.}", r"\end{table}"]
        out.append(r"\clearpage")
    (TAB / "body.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def compile_pdf():
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={JOB}",
                        "main_v3.tex"], cwd=HERE, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    if r.returncode:
        print(r.stdout[-3000:])
        sys.exit(f"latexmk failed (exit {r.returncode}); see {JOB}.log")
    subprocess.run(["latexmk", "-c", f"-jobname={JOB}", "main_v3.tex"], cwd=HERE, capture_output=True)


def main():
    if "--fetch" in sys.argv:
        fetch()
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    data = load()
    assert data, f"no result in {BASE} (run with --fetch)"
    inputs = []
    for t, (sc, dec, _) in data.items():
        for score, title, rule in SCORES:
            inputs.append(table(t, sc, score, rule, title))
        if dec is not None:
            inputs.append(table(t, common_frame(dec), "rmse", "min", "RMSE", common=True))
        windows_tex(t, sc)
    body_tex(data)
    fig_curves(data, "curves_v3.pdf")
    fig_sensitivity(data, "sensitivity_v3.pdf")
    fig_distance(data, "distance_v3.pdf")
    maps_tex(data)
    elevation_maps_tex()
    vals = {"build:date": dt.datetime.now().astimezone().strftime("%Y-%m-%d"),
            "truths": ", ".join(TRUTH_NAME[t] for t in data)}
    (TAB / "macros.tex").write_text("% generated by build_v3.py\n" + "".join(
        rf"\valdef{{{k}}}{{{v}}}" + "\n" for k, v in vals.items()), encoding="utf-8")
    pd.DataFrame([dict(file=k, sha256=h, rows=n) for k, (h, n) in READ.items()]).to_csv(
        HERE / f"{JOB}_inputs.csv", index=False)
    compile_pdf()
    print(f"truths with results: {list(data)}; {len(READ)} files read -> {HERE / (JOB + '.pdf')}")


if __name__ == "__main__":
    main()
