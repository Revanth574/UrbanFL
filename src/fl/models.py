"""
Models for UrbanFL federated experiments.
  Silo 1: MLP head over cached sentence embeddings.
  Silo 2: U-Net with a torchvision ResNet encoder (stage 1, building localisation from the pre image). The Siamese
          damage model (stage 2) will reuse the same encoder/decoder blocks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear over frozen sentence-transformer embeddings."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(train_labels: Sequence[int], n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights n / (k * count_c), normalised so a balanced set gets all-ones.
    Classes with zero training examples get weight 0 (they contribute no loss terms anyway).
    """
    return class_weights_from_counts(np.bincount(np.asarray(train_labels, dtype=int), minlength=n_classes))


def class_weights_from_counts(counts: Sequence[int]) -> torch.Tensor:
    """Same weighting as class_weights, from per-class counts (e.g. a federated sum of client label histograms)."""
    counts = np.asarray(counts, dtype=float)
    n_classes = len(counts)
    present = counts > 0
    weights = np.zeros(n_classes)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def weighted_ce(train_labels: Sequence[int], n_classes: int, device: str | torch.device = "cpu") -> nn.Module:
    """Class-weighted cross-entropy with weights from the given training label counts."""
    return nn.CrossEntropyLoss(weight=class_weights(train_labels, n_classes).to(device))


# ── Silo 2 ─────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """x2 bilinear upsample, concatenate the skip feature, two 3x3 conv-BN-ReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1) if skip is not None else x)


class UNetResNet(nn.Module):
    """
    U-Net with a torchvision ResNet18/34 encoder (ImageNet weights by default) and a 5-stage decoder back to full
    resolution. Input (B, 3, H, W) with H, W divisible by 32; output logits (B, out_channels, H, W).
    """

    ENCODERS = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights"}

    def __init__(self, encoder: str = "resnet18", pretrained: bool = True, out_channels: int = 1):
        super().__init__()
        import torchvision
        if encoder not in self.ENCODERS:
            raise ValueError(f"encoder must be one of {list(self.ENCODERS)}")
        weights = getattr(torchvision.models, self.ENCODERS[encoder]).IMAGENET1K_V1 if pretrained else None
        enc = getattr(torchvision.models, encoder)(weights=weights)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)        # /2,  64
        self.pool = enc.maxpool                                        # /4
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        # encoder channels at /4, /8, /16, /32 are 64, 128, 256, 512 for both resnet18 and resnet34
        self.dec4 = DecoderBlock(512, 256, 256)   # /32 -> /16
        self.dec3 = DecoderBlock(256, 128, 128)   # /16 -> /8
        self.dec2 = DecoderBlock(128, 64, 64)     # /8  -> /4
        self.dec1 = DecoderBlock(64, 64, 64)      # /4  -> /2  (stem skip)
        self.dec0 = DecoderBlock(64, 0, 32)       # /2  -> /1
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        return self.head(self.dec0(d, None))


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary segmentation loss: BCEWithLogits + (1 - soft Dice). logits (B,1,H,W) or (B,H,W); target in {0,1}."""
    logits = logits.squeeze(1) if logits.dim() == 4 else logits
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits.float())
    inter = (prob * target).sum(dim=(1, 2))
    dice = (2 * inter + smooth) / (prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return bce + (1 - dice).mean()
