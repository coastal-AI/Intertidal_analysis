"""Undo the literal-escape corruption in the study notebook.

Some cells hold their source with escape sequences written out as text —
"\\u2500" as six characters rather than the box-drawing rule, "\\'" for a
quote, "\\\\n" for a newline. A cell in that state is not merely ugly: it is a
SyntaxError, so the notebook cannot run at all, and it predates today's edits
(an editor rewrote three cells at some point).

Decoding the whole source with unicode_escape would also mangle legitimate
backslashes, so only the sequences actually observed are translated, and the
result is re-parsed to prove it is valid Python before anything is written.
"""
import ast
import io
import re

import nbformat

NB = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\intertidal_topography_villaviciosa.ipynb"


def repair(src):
    out = re.sub(r"\\u([0-9a-fA-F]{4})",
                 lambda m: chr(int(m.group(1), 16)), src)
    out = out.replace("\\'", "'")
    out = out.replace("\\\\n", "\\n")
    return out


nb = nbformat.read(io.open(NB, encoding="utf-8"), as_version=4)

broken = []
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    try:
        ast.parse(c.source)
    except SyntaxError:
        broken.append(i)

print(f"celdas que no compilan: {broken or 'ninguna'}")

fixed, still = [], []
for i in broken:
    candidate = repair(nb.cells[i].source)
    try:
        ast.parse(candidate)
    except SyntaxError as e:
        still.append((i, str(e)[:70]))
        continue
    nb.cells[i].source = candidate
    fixed.append(i)

# Cells that compile but still carry the literal escapes are readable-but-wrong:
# they print "\u2500" instead of a rule. Repair those too, but only if the
# repair still parses.
cosmetic = []
for i, c in enumerate(nb.cells):
    if i in fixed or c.cell_type != "code":
        continue
    if re.search(r"\\u[0-9a-fA-F]{4}", c.source):
        candidate = repair(c.source)
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        c.source = candidate
        cosmetic.append(i)

print(f"reparadas (no compilaban): {fixed or 'ninguna'}")
print(f"reparadas (escapes visibles): {cosmetic or 'ninguna'}")
print(f"siguen rotas: {still or 'ninguna'}")

if fixed or cosmetic:
    nbformat.validate(nb)
    nbformat.write(nb, io.open(NB, "w", encoding="utf-8"))
    print("notebook reescrito y validado")
