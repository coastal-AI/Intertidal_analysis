"""Train TideGAN on physical channels (NDWI and/or NIR).

Two deliberate departures from the original training loop, both because the
consumer is MAREA rather than a human eye:

* **The L1 loss is masked to CLEAR pixels of the target.** The archive keeps
  cloudy scenes (they carry clear regions worth learning from), but a loss
  paid on cloudy pixels teaches the network to paint clouds — the exact
  artefact the inpainting experiment exists to remove.
* **A waterline diagnostic is logged**: the fraction of pixels whose wet/dry
  state (NDWI > 0) the generation gets right against the clear target. L1
  can look good while the waterline sits metres off; this number is the one
  the downstream experiments actually care about.

Run:
  python -m experiments.g1_gan_train --channels ndwi          (1-channel)
  python -m experiments.g1_gan_train --channels nir           (1-channel)
  python -m experiments.g1_gan_train --channels ndwi nir      (2-channel)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.tidegan import (Archive, Generator, PatchSampler,
                                 count_parameters, tide_spread_split)


def wetdry_agreement(fake, target, clear, ndwi_channel=0):
    """Fraction of clear pixels whose wet/dry state matches (NDWI > 0)."""
    f = fake[:, ndwi_channel] > 0
    t = target[:, ndwi_channel] > 0
    m = clear > 0.5
    if m.sum() == 0:
        return float("nan")
    return float((f.eq(t) & m).sum().item() / m.sum().item())


class MaskedSampler(PatchSampler):
    """PatchSampler that also returns the target's clear mask."""

    def __getitem__(self, idx):
        # re-draw through the parent by re-running its search, capturing the
        # crop: simplest is to sample here with the same rules
        p = self.patch
        T, H, W = self.a.shape
        for _ in range(400):
            t = self.rng.choice(self.pool)
            nt = self.a.norm_tide(self.a.tides[t])
            if abs(nt - self.ref_norm) <= self.min_tide_diff:
                continue
            y = self.rng.randint(0, H - p)
            x = self.rng.randint(0, W - p)
            if not (self._crop_ok(self.ref_idx, y, x)
                    and self._crop_ok(t, y, x)):
                continue
            ndwi_t = np.asarray(
                self.a.channels["ndwi"][t, y:y + p, x:x + p], np.float32)
            wet = ndwi_t > 0.0
            if not (0.02 < wet.mean() < 0.98):
                continue
            ref = np.stack([np.asarray(
                self.a.channels[c][self.ref_idx, y:y + p, x:x + p],
                np.float32) for c in self.channels])
            tgt = np.stack([np.asarray(
                self.a.channels[c][t, y:y + p, x:x + p], np.float32)
                for c in self.channels])
            clear = np.asarray(self.a.clear[t, y:y + p, x:x + p],
                               np.float32)
            ref = np.nan_to_num(ref, nan=0.0)
            tgt = np.nan_to_num(tgt, nan=0.0)
            if self.augment and self.rng.random() > 0.5:
                ref = np.ascontiguousarray(ref[:, :, ::-1])
                tgt = np.ascontiguousarray(tgt[:, :, ::-1])
                clear = np.ascontiguousarray(clear[:, ::-1])
            cond = self.a.condition(self.a.tides[t])
            return (torch.from_numpy(ref), torch.from_numpy(tgt),
                    torch.from_numpy(clear), torch.from_numpy(cond))
        raise RuntimeError("no valid patch found in 400 attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="villaviciosa")
    ap.add_argument("--channels", nargs="+", default=["ndwi"],
                    choices=["ndwi", "nir"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps_per_epoch", type=int, default=125)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "-".join(args.channels)
    save_dir = os.path.join("checkpoints", f"tidegan_{args.site}_{tag}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"device {device} · channels {args.channels} · save {save_dir}")

    archive = Archive(f"gan_archive_{args.site}")
    usable = [i for i in range(len(archive.dates))
              if np.isfinite(archive.tides[i])
              and archive.clear_fraction[i] > 0.3]
    train_idx, val_idx = tide_spread_split(archive.tides, usable,
                                           val_ratio=0.15)
    print(f"{len(usable)} usable dates -> {len(train_idx)} train / "
          f"{len(val_idx)} val (tide-spread split)")

    n = args.steps_per_epoch * args.batch_size
    train_ds = MaskedSampler(archive, channels=args.channels,
                             patch=args.patch_size, date_indices=train_idx,
                             length=n, augment=True, seed=args.seed)
    val_ds = MaskedSampler(archive, channels=args.channels,
                           patch=args.patch_size, date_indices=val_idx,
                           length=args.batch_size * 8, augment=False,
                           seed=args.seed + 1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    G = Generator(in_channels=len(args.channels)).to(device)
    print(f"generator parameters: {count_parameters(G):,}")
    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ck["generator_state"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        print(f"resumed from epoch {start_epoch}")

    history = []
    ndwi_ch = args.channels.index("ndwi") if "ndwi" in args.channels else None
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        G.train()
        losses = []
        for ref, tgt, clear, cond in train_dl:
            ref, tgt = ref.to(device), tgt.to(device)
            clear, cond = clear.to(device), cond.to(device)
            opt.zero_grad()
            fake = G(ref, cond)
            m = clear.unsqueeze(1)
            loss = (torch.abs(fake - tgt) * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        G.eval()
        vl, wd = [], []
        with torch.no_grad():
            for ref, tgt, clear, cond in val_dl:
                ref, tgt = ref.to(device), tgt.to(device)
                clear, cond = clear.to(device), cond.to(device)
                fake = G(ref, cond)
                m = clear.unsqueeze(1)
                vl.append(float(((torch.abs(fake - tgt) * m).sum()
                                 / m.sum().clamp(min=1)).item()))
                if ndwi_ch is not None:
                    wd.append(wetdry_agreement(fake, tgt, clear, ndwi_ch))
        row = {"epoch": epoch + 1,
               "train_l1": round(float(np.mean(losses)), 5),
               "val_l1": round(float(np.mean(vl)), 5),
               "val_wetdry": (round(float(np.nanmean(wd)), 4)
                              if wd else None),
               "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {row['epoch']:3d}/{args.epochs} · "
              f"L1 {row['train_l1']:.4f} · val {row['val_l1']:.4f} · "
              f"wet/dry {row['val_wetdry']} · {row['secs']}s", flush=True)
        with open(os.path.join(save_dir, "history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, indent=1)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save({"generator_state": G.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "epoch": epoch + 1,
                        "channels": args.channels,
                        "site": args.site},
                       os.path.join(save_dir, f"epoch_{epoch + 1:03d}.pth"))
    print(f"done -> {save_dir}")


if __name__ == "__main__":
    main()
