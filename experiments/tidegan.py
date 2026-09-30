"""TideGAN, adapted to physical channels (NDWI / NIR).

The architecture is Pablo's (pglez82/IntertidalGAN): a U-Net generator with
AdaIN condition injection, where the condition is the normalized target tide
plus a site one-hot, and the objective is plain L1 — supervised image
translation, no discriminator. Two changes, both motivated by how MAREA will
consume the outputs:

* **Channels are configurable** (``in_channels``): instead of 8-bit
  percentile-stretched RGB PNGs, the model trains on the physical rasters the
  estimation actually uses — NDWI (already in [-1, 1], the tanh range, with
  the waterline at a fixed physical value) and/or NIR reflectance. A model
  that must place the waterline in index space can be *judged* in index
  space.
* **The dataset is a memmap patch sampler** rather than PNGs loaded whole
  into RAM: the Villaviciosa archive is 1379 dates of 915x915 float16, which
  does not fit the 7 GB of this machine.

Everything else — patch size 256, cloud-aware crop search, fixed clear
reference scene, min tide separation between reference and target, the
condition vector layout — follows the original so a checkpoint here remains
comparable with Pablo's runs.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

SITES = ["Foz", "Santander", "Villaviciosa"]   # Pablo's vocabulary, frozen


# ── model (Generator of pglez82/IntertidalGAN, channel-parameterised) ──────
class ConditionEmbed(nn.Module):
    def __init__(self, n_sites=3, cond_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + n_sites, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, cond_dim))

    def forward(self, condition):
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        return self.mlp(condition)


class AdaIN(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.scale_shift = nn.Linear(cond_dim, in_channels * 2)

    def forward(self, x, cond_embed):
        stats = self.norm(x)
        params = self.scale_shift(cond_embed).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return stats * (1 + scale) + shift


class CondEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = AdaIN(out_ch, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, c):
        return self.act(self.norm(self.conv(x), c))


class CondDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2,
                                         padding=1)
        self.norm = AdaIN(out_ch, cond_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, c):
        return self.act(self.norm(self.deconv(x), c))


class CondConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = AdaIN(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = AdaIN(out_ch, cond_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):
        x = self.relu(self.norm1(self.conv1(x), c))
        return self.relu(self.norm2(self.conv2(x), c))


class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = AdaIN(channels, cond_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaIN(channels, cond_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):
        out = self.relu(self.norm1(self.conv1(x), c))
        out = self.norm2(self.conv2(out), c)
        return out + x


class Generator(nn.Module):
    """U-Net + AdaIN generator; ``in_channels`` = 1 (NDWI or NIR) or 2."""

    def __init__(self, in_channels=1, n_sites=3, cond_dim=16):
        super().__init__()
        self.in_channels = in_channels
        self.condition_embed = ConditionEmbed(n_sites, cond_dim)
        self.enc1 = CondEncoder(in_channels, 64, cond_dim)
        self.enc2 = CondEncoder(64, 128, cond_dim)
        self.enc3 = CondEncoder(128, 256, cond_dim)
        self.enc4 = CondEncoder(256, 512, cond_dim)
        self.bn1 = ResBlock(512, cond_dim)
        self.bn2 = ResBlock(512, cond_dim)
        self.bn3 = ResBlock(512, cond_dim)
        self.dec4 = CondDecoder(512, 256, cond_dim)
        self.dec4_block = CondConvBlock(512, 256, cond_dim)
        self.dec3 = CondDecoder(256, 128, cond_dim)
        self.dec3_block = CondConvBlock(256, 128, cond_dim)
        self.dec2 = CondDecoder(128, 64, cond_dim)
        self.dec2_block = CondConvBlock(128, 64, cond_dim)
        self.dec1 = CondDecoder(64, 64, cond_dim)
        self.dec1_block = CondConvBlock(64 + in_channels, 64, cond_dim)
        self.final = nn.Conv2d(64, in_channels, 1)

    def forward(self, ref, condition):
        c = self.condition_embed(condition)
        e1 = self.enc1(ref, c)
        e2 = self.enc2(e1, c)
        e3 = self.enc3(e2, c)
        e4 = self.enc4(e3, c)
        b = self.bn3(self.bn2(self.bn1(e4, c), c), c)
        d4 = self.dec4_block(torch.cat([self.dec4(b, c), e3], 1), c)
        d3 = self.dec3_block(torch.cat([self.dec3(d4, c), e2], 1), c)
        d2 = self.dec2_block(torch.cat([self.dec2(d3, c), e1], 1), c)
        d1 = self.dec1_block(torch.cat([self.dec1(d2, c), ref], 1), c)
        return torch.tanh(self.final(d1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── the memmap archive written by g0_gan_dataset.py ────────────────────────
class Archive:
    """Read-only view of one exported site archive.

    ``{root}/meta.json`` describes shape, dates, tides and normalisation;
    ``ndwi.f16``, ``nir.f16`` are (T, H, W) float16 memmaps already in
    [-1, 1]; ``clear.u8`` is 1 where SCL calls the pixel clear.
    """

    def __init__(self, root):
        self.root = root
        self.meta = json.load(open(os.path.join(root, "meta.json"),
                                   encoding="utf-8"))
        T, H, W = self.meta["shape"]
        self.shape = (T, H, W)
        self.dates = self.meta["dates"]
        self.tides = np.asarray([np.nan if v is None else v
                                 for v in self.meta["tides_m"]],
                                dtype=np.float32)
        self.site = self.meta["site"]
        self.channels = {}
        for name in ("ndwi", "nir"):
            p = os.path.join(root, f"{name}.f16")
            if os.path.exists(p):
                self.channels[name] = np.memmap(p, dtype=np.float16,
                                                mode="r", shape=(T, H, W))
        self.clear = np.memmap(os.path.join(root, "clear.u8"),
                               dtype=np.uint8, mode="r", shape=(T, H, W))
        self.clear_fraction = np.asarray(self.meta["clear_fraction"],
                                         dtype=np.float32)

    def norm_tide(self, h):
        lo, hi = self.meta["tide_range_m"]
        return 2.0 * (float(h) - lo) / (hi - lo) - 1.0

    def condition(self, h):
        one_hot = [1.0 if s == self.site else 0.0 for s in SITES]
        return np.asarray([self.norm_tide(h)] + one_hot, dtype=np.float32)


class PatchSampler(torch.utils.data.Dataset):
    """Cloud-aware (reference, target, condition) patch pairs from memmaps.

    Follows the original sampler's rules: one FIXED clear reference scene per
    site, a minimum normalized-tide separation between reference and target,
    crops rejected while either member has too many cloudy pixels, and a
    requirement that the target patch contains both wet and dry ground (the
    waterline must be in frame, or the pair teaches nothing about tide).
    """

    def __init__(self, archive, channels=("ndwi",), patch=256,
                 min_tide_diff=0.2, max_cloud=0.02, date_indices=None,
                 length=1000, augment=True, seed=42):
        self.a = archive
        self.channels = list(channels)
        self.patch = int(patch)
        self.min_tide_diff = float(min_tide_diff)
        self.max_cloud = float(max_cloud)
        self.length = int(length)
        self.augment = augment
        self.rng = random.Random(seed)

        idx = (list(range(len(archive.dates))) if date_indices is None
               else list(date_indices))
        # usable targets: some clear sky and a finite tide
        self.pool = [i for i in idx
                     if archive.clear_fraction[i] > 0.3
                     and np.isfinite(archive.tides[i])]
        if not self.pool:
            raise ValueError("no usable dates in this split")

        # fixed reference: the clearest scene in the LOWEST-tide quartile,
        # like the original (lowest clear tide shows the most ground)
        q = np.quantile(archive.tides[self.pool], 0.25)
        low = [i for i in self.pool if archive.tides[i] <= q]
        self.ref_idx = max(low or self.pool,
                           key=lambda i: archive.clear_fraction[i])
        self.ref_norm = archive.norm_tide(archive.tides[self.ref_idx])

    def __len__(self):
        return self.length

    def _crop_ok(self, t_idx, y, x):
        p = self.patch
        clear = self.a.clear[t_idx, y:y + p, x:x + p]
        return (1.0 - clear.mean()) <= self.max_cloud

    def __getitem__(self, _):
        p = self.patch
        T, H, W = self.a.shape
        for _attempt in range(400):
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
            if not (0.02 < wet.mean() < 0.98):     # waterline must be in frame
                continue
            ref = np.stack([np.asarray(
                self.a.channels[c][self.ref_idx, y:y + p, x:x + p],
                np.float32) for c in self.channels])
            tgt = np.stack([np.asarray(
                self.a.channels[c][t, y:y + p, x:x + p], np.float32)
                for c in self.channels])
            ref = np.nan_to_num(ref, nan=0.0)
            tgt = np.nan_to_num(tgt, nan=0.0)
            if self.augment and self.rng.random() > 0.5:
                ref = np.ascontiguousarray(ref[:, :, ::-1])
                tgt = np.ascontiguousarray(tgt[:, :, ::-1])
            cond = self.a.condition(self.a.tides[t])
            return (torch.from_numpy(ref), torch.from_numpy(tgt),
                    torch.from_numpy(cond))
        raise RuntimeError("no valid patch found in 400 attempts — "
                           "loosen max_cloud or check the archive")


def tide_spread_split(tides, indices, val_ratio=0.2):
    """Deterministic split: every k-th index of the tide-sorted list goes to
    val, so both splits cover the tide spectrum (the original's scheme)."""
    order = sorted(indices, key=lambda i: float(tides[i]))
    n_val = max(1, int(round(len(order) * val_ratio)))
    val_pos = set(int(v) for v in
                  np.unique(np.linspace(0, len(order) - 1, num=n_val,
                                        dtype=int)))
    train = [order[i] for i in range(len(order)) if i not in val_pos]
    val = [order[i] for i in range(len(order)) if i in val_pos]
    return train, val
