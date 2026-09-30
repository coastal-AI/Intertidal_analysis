"""Reproduce the ICE CREAMS intertidal-vegetation model from its own data.

ICE CREAMS — *Intertidal Classification of Europe: Categorising Reflectance
of Emerged Areas of Marine vegetation using Sentinel-2* (Davies et al.,
Université de Nantes / BiCOME; github.com/BedeFfinian/ICE_CREAMS) — is a
per-pixel fastai tabular network (2 hidden layers, ~27 k parameters) that
labels emerged intertidal pixels into 9 classes: Bare Sand, Bare Sediment,
Chlorophyta, Magnoliopsida (seagrass), Microphytobenthos, Phaeophyceae,
Rhodophyta, Water, Xanthophyceae. Features are the 12 L2A bands raw, the
same 12 min-max standardised PER PIXEL across the spectrum (spectral shape,
illumination-free), plus NDVI and NDWI. Published accuracy: 82 % on
independent European seagrass validation.

The repository ships the full training tables but not the .pkl, so the
model is REPRODUCED here from their data with their exact recipe (their
training notebook: concat V1_1+V1_2+V1_3, FillMissing, default
tabular_learner, fine_tune(20)) — which also pins the random seed and
records the validation accuracy we obtained, instead of trusting an opaque
binary.

Run:  python -m experiments.f0_icecreams_train --data <ICE_CREAMS repo dir>
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="path to the cloned ICE_CREAMS repository")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd
    from fastai.tabular.all import (CategoryBlock, FillMissing,
                                    RandomSplitter, TabularPandas,
                                    cont_cat_split, range_of, set_seed,
                                    tabular_learner)

    set_seed(args.seed, reproducible=True)
    frames = []
    for v in ("V1_1", "V1_2", "V1_3"):
        files = glob.glob(os.path.join(args.data, "Data", "Input",
                                       "Training", v, "*.csv"))
        if not files:
            raise SystemExit(f"no training CSVs under {args.data} ({v})")
        frames.append(pd.concat((pd.read_csv(f) for f in files),
                                ignore_index=True))
    df = pd.concat(frames, ignore_index=True)
    print(f"training table: {len(df):,} px, "
          f"{df['True_Class'].nunique()} classes")
    print(df["True_Class"].value_counts().to_string())

    dep = "True_Class"
    splits = RandomSplitter(valid_pct=0.3, seed=args.seed)(range_of(df))
    cont, cat = cont_cat_split(df, dep_var=dep)
    to = TabularPandas(df, [FillMissing], cat, cont, splits=splits,
                       y_names=dep, y_block=CategoryBlock())
    dls = to.dataloaders(bs=1024)
    learn = tabular_learner(dls, n_out=9)
    learn.fine_tune(args.epochs)

    from fastai.metrics import error_rate
    preds, targs = learn.get_preds()
    acc = float((preds.argmax(1) == targs.squeeze()).float().mean())
    print(f"validation accuracy: {acc:.4f}")

    out_dir = os.path.join("checkpoints", "icecreams")
    os.makedirs(out_dir, exist_ok=True)
    pkl = os.path.abspath(os.path.join(out_dir, "ICECREAMS_repro.pkl"))
    learn.export(pkl)
    meta = {"source": "github.com/BedeFfinian/ICE_CREAMS",
            "recipe": "their training notebook, seeded",
            "n_train_px": int(len(df)), "epochs": args.epochs,
            "seed": args.seed, "valid_accuracy": round(acc, 4),
            "features": cont, "classes": sorted(df[dep].unique().tolist())}
    with open(os.path.join(out_dir, "meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"model -> {pkl}")


if __name__ == "__main__":
    main()
