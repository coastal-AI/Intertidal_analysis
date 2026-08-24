"""Strip siunitx out of the poster sources.

siunitx v3 here is out of step with the expl3 kernel MiKTeX ships — \num
fails inside the package's own internals — and fixing that means updating
l3kernel on the user's TeX installation over the network. A poster uses a
dozen numbers and half a dozen units; writing them out costs nothing and
removes a whole class of build fragility.

Thin spaces (\,) are used both for the number-unit gap and as the thousands
separator, which is the SI convention anyway.
"""
import re
import pathlib

ROOT = pathlib.Path(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\poster")

UNITS = {
    r"\metre": "m", r"\meter": "m",
    r"\centi\metre": "cm", r"\milli\metre": "mm",
    r"\kilo\metre": "km",
    r"\kilo\metre\squared": r"km\textsuperscript{2}",
    r"\percent": r"\%",
}


def group(n):
    """1379 -> 1\\,379, leaving decimals alone."""
    if "." in n or len(n) <= 4:
        return n
    out, s = [], n
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return r"\," .join(out)


def fix(text):
    # \SIrange{a}{b}{unit}
    def _range(m):
        u = UNITS.get(m.group(3), m.group(3).lstrip("\\"))
        return f"{group(m.group(1))}--{group(m.group(2))}\\,{u}"
    text = re.sub(r"\\SIrange\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", _range, text)

    # \SI{value}{unit}
    def _si(m):
        u = UNITS.get(m.group(2), m.group(2).lstrip("\\"))
        return f"{group(m.group(1))}\\,{u}"
    text = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", _si, text)

    # \num{value}
    text = re.sub(r"\\num\{([^}]*)\}", lambda m: group(m.group(1)), text)

    # siunitx S columns -> plain right-aligned; Fira's figures are tabular,
    # so decimal points still line up.
    text = re.sub(r"S\[table-format=[^\]]*\]", "r", text)
    return text


changed = []
for p in sorted(ROOT.rglob("*.tex")):
    s = p.read_text(encoding="utf-8")
    t = fix(s)
    if t != s:
        p.write_text(t, encoding="utf-8")
        changed.append(p.relative_to(ROOT))

# drop the package and its setup
pkg = ROOT / "preamble" / "packages.tex"
s = pkg.read_text(encoding="utf-8")
s = s.replace("\\usepackage{siunitx}\n", "")
pkg.write_text(s, encoding="utf-8")

tab = ROOT / "preamble" / "tables.tex"
s = tab.read_text(encoding="utf-8")
s = re.sub(r"\\sisetup\{[^}]*\}\n", "", s, flags=re.S)
tab.write_text(s, encoding="utf-8")

print(f"{len(changed)} ficheros reescritos:")
for c in changed:
    print("  ", c)
print("siunitx eliminado del preambulo")
