import json

NB = r"C:\Users\Jorge\sketch_fitton\Intertidal_analysis\intertidal_topography_villaviciosa.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
c = next(x for x in nb["cells"] if x.get("id") == "8a598ee9")
src = "".join(c["source"]).replace(
    "# / plus a raster draw the trace on a map beside the profile —",
    "# `start`/`end` plus a raster draw the trace on a map beside the profile —")
c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("".join(c["source"]))
