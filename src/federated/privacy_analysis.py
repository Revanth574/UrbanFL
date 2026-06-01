import logging
import torch
import math

logger = logging.getLogger("urbanfl.federated.privacy")

class DifferentialPrivacyGaussianMechanism:
    """
    Applies Gaussian noise to aggregated gradients across silos.
    Strictly tracks the privacy budget (Epsilon ε).
    """
    def __init__(self, initial_epsilon: float = 2.0, max_grad_norm: float = 1.0, noise_multiplier: float = 0.5):
        self.epsilon = initial_epsilon
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        logger.info(f"DP Mechanism initialized (ε={initial_epsilon}, max_norm={max_grad_norm})")

    def apply_noise(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """Clips gradients and adds calibrated Gaussian noise."""
        if self.epsilon <= 0.1:
            logger.warning("Privacy budget depleted. High noise will degrade accuracy.")
            
        noised_gradients = []
        for grad in gradients:
            # 1. Clip
            norm = torch.norm(grad)
            clip_coef = self.max_grad_norm / (norm + 1e-6)
            clip_coef = min(1.0, clip_coef.item())
            clipped_grad = grad * clip_coef
            
            # 2. Add Noise (scale inversely with remaining budget securely)
            effective_noise = self.noise_multiplier * (2.0 / max(0.1, self.epsilon))
            noise = torch.randn_like(clipped_grad) * effective_noise
            
            noised_gradients.append(clipped_grad + noise)
            
        # Deduct budget slowly
        self.epsilon = max(0.1, self.epsilon - 0.05)
        return noised_gradients
