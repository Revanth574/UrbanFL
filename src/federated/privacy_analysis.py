"""
DEPRECATED — minor-project prototype. NOT a differential-privacy mechanism; do not cite any ε from it.

The original class decremented ε by hand (-0.05 per call) and scaled noise by 2/ε, which gives no formal guarantee
(docs/AUDIT.md §4A; CLAUDE.md rule 3: ε comes only from a real accountant). The ε bookkeeping has been removed; what
remains is per-tensor clipping plus Gaussian noise with a fixed multiplier, kept only so old imports resolve.
Client-level DP-FedAvg with an Opacus RDPAccountant belongs in src/fl/dp.py (Prompt 4).
"""

import logging

import torch

logger = logging.getLogger("urbanfl.federated.privacy")


class DifferentialPrivacyGaussianMechanism:
    """Clip each tensor to max_grad_norm and add N(0, (noise_multiplier * max_grad_norm)^2). No privacy accounting."""

    def __init__(self, max_grad_norm: float = 1.0, noise_multiplier: float = 0.5, **_ignored):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.epsilon = None   # not computed: there is no accountant here
        logger.info("Deprecated Gaussian noise helper (no ε accounting); see src/fl/dp.py")

    def apply_noise(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        noised = []
        for grad in gradients:
            clip_coef = min(1.0, (self.max_grad_norm / (torch.norm(grad) + 1e-6)).item())
            clipped = grad * clip_coef
            noised.append(clipped + torch.randn_like(clipped) * self.noise_multiplier * self.max_grad_norm)
        return noised
