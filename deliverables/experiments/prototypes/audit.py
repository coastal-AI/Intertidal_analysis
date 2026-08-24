"""Final package-wide audit of pyintertidal."""
import ast, glob, re, sys, os, importlib

os.chdir(r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis")
sys.path.insert(0, ".")

files = sorted(glob.glob("pyintertidal/**/*.py", recursive=True))
print(f"MODULES: {len(files)}")

spanish_markers = re.compile(
    r"\b(el|la|los|las|una|para|con|que|del|por|segun|funcion|calculo|pixel|"
    r"mascara|marea|mas)\b", re.I)
no_doc, spanish, total = [], [], 0
for f in files:
    src = open(f, encoding="utf-8").read()
    tree = ast.parse(src)
    if not ast.get_docstring(tree):
        no_doc.append(f"{f} (module)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and \
                not node.name.startswith("_"):
            total += 1
            doc = ast.get_docstring(node)
            if not doc:
                no_doc.append(f"{f}::{node.name}")
            elif len(spanish_markers.findall(doc)) >= 3:
                spanish.append(f"{f}::{node.name}")
print(f"PUBLIC FUNCTIONS/CLASSES: {total}")
print(f"  missing docstring: {len(no_doc)}"
      + (f" -> {no_doc[:5]}" if no_doc else "  OK"))
print(f"  docstrings looking Spanish: {len(spanish)}"
      + (f" -> {spanish[:5]}" if spanish else "  OK"))

leaks = []
for f in files:
    for i, line in enumerate(open(f, encoding="utf-8"), 1):
        if re.search(r"^\s*(from|import)\s+intertidal\b", line):
            leaks.append(f"{f}:{i}")
print(f"IMPORTS OF OLD PACKAGE: {len(leaks)}"
      + (f" -> {leaks}" if leaks else "  OK (self-contained)"))

bad = []
for f in files:
    norm = f.replace(os.sep, "/")
    if "/legacy/" in norm:
        continue
    src = open(f, encoding="utf-8").read()
    if re.search(r"from\s+\.legacy|from\s+\.\.legacy|import\s+legacy", src):
        bad.append(f)
print(f"CORE -> LEGACY IMPORTS: {len(bad)}"
      + (f" -> {bad}" if bad else "  OK (one-way dependency)"))

failed = []
for f in files:
    mod = (f.replace(os.sep, ".").replace("/", ".")
             .replace(".py", "").replace(".__init__", ""))
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append(f"{mod}: {type(e).__name__}: {e}")
print(f"IMPORTABLE: {len(files) - len(failed)}/{len(files)}"
      + (f" -> {failed}" if failed else "  OK"))

old = open("intertidal/bathymetry.py", encoding="utf-8").read()
new = open("pyintertidal/legacy/udf.py", encoding="utf-8").read()
exact = True
for o, n in (("_BATHYMETRY_UDF_SCL", "BATHYMETRY_UDF_SCL"),
             ("_BATHYMETRY_UDF_NDWI", "BATHYMETRY_UDF_NDWI")):
    a = re.search(re.escape(o) + r" = '''(.*?)'''", old, re.S).group(1)
    b = re.search(re.escape(n) + r" = '''(.*?)'''", new, re.S).group(1)
    exact &= (a == b)
print(f"LEGACY UDFs BIT-EXACT: {'OK' if exact else 'MISMATCH'}")

lines = sum(len(open(f, encoding='utf-8').readlines()) for f in files)
print(f"TOTAL LINES: {lines:,}")
