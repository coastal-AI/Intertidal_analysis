import json

PATH = "final_notebook.ipynb"
orig = open(PATH, encoding="utf-8").read()
nb = json.loads(orig)

c29 = nb["cells"][29]
assert c29["cell_type"] == "markdown"
assert "".join(c29["source"]).startswith("## MAREA")
c29["source"] = ["## MAREA — measured interior tide + hypsometry "
                 "with uncertainty (method v4)\n"]

c30 = nb["cells"][30]
assert c30["cell_type"] == "code"
src = c30["source"]
assert src[0].startswith("from pyintertidal import marea")
box = [ln for ln in src if ln.startswith("# ═")]
assert len(box) == 2, box
tail = [ln for ln in src if ln.startswith("CUBE_PATH")
        or ln.startswith("res_marea")
        or ln.lstrip().startswith("name=site_name")]
assert len(tail) == 3, tail
print_lines = [ln for ln in src if "print(f" in ln or "f\"operador" in ln
               or "HI = " in ln]

new_src = [
    "from pyintertidal import marea\n",
    box[0],
    "# MAREA — the complete v4 method (HSR + interior tide + "
    "hypsometry with\n",
    "# uncertainty). The difference from the previous cell: the water "
    "level is not\n",
    "# assumed uniform — the ria is SPLIT into bands of distance to "
    "the mouth and\n",
    "# each band's lag is measured from the wet/dry sequences themselves "
    "(M2a,\n",
    "# profiled likelihood, anchored at the mouth). The clock is only "
    "applied where\n",
    "# it exceeds the demonstrated detection threshold (10 min); in deep "
    "rias the\n",
    "# operator stays at identity, and that too is a result.\n",
    "# Input: the SAME raw netCDF from section 4 (downloads nothing).\n",
    box[1],
    tail[0],
    tail[1],
    tail[2],
    "print(f\"MAREA: {res_marea['n_px_cota']:,} px with elevation | \"\n",
    "      f\"operator {'ACTIVE' if res_marea['con_operador'] else "
    "'at identity'} | \"\n",
    "      f\"HI = {res_marea['hipsometria']['integral']:.2f}\")\n",
    "# the figure (DEM + lag profile + hypsometric curve with an "
    "uncertainty\n",
    "# band) lands in bathymetry/marea/figure.png; the JSON carries the "
    "taus,\n",
    "# input hashes and every count — nothing is left undeclared",
]
c30["source"] = new_src

out = json.dumps(nb, indent=1, ensure_ascii=True)
open(PATH, "w", encoding="utf-8", newline="").write(out)
print("written; cells 29/30 translated")
# sanity: still valid JSON and code compiles
nb2 = json.load(open(PATH, encoding="utf-8"))
compile("".join(nb2["cells"][30]["source"]), "<cell30>", "exec")
print("cell 30 compiles OK")
