"""Models for UrbanFL federated experiments. Silo 1: MLP head over cached sentence embeddings."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


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
