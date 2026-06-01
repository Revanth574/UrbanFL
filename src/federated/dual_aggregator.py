import logging
import torch
import math
from typing import Dict, Any, Optional

logger = logging.getLogger("urbanfl.federated.aggregator")

class DualAggregator:
    """
    Advanced aggregator that handles hierarchical model states.
    Supports 'island mode' for independent hemisphere training and
    controlled synchronization based on L2 divergence thresholds.
    """
    def __init__(self, island_mode: bool = False, adaptive_threshold: float = 0.4):
        self.island_mode = island_mode
        self.threshold = adaptive_threshold
        self.round_count = 0
        self.metrics = {"divergence": 0.0, "sync_events": 0}
        logger.info(f"DualAggregator initialized (island_mode={island_mode}, threshold={adaptive_threshold})")

    def compute_divergence(self, left_model: Dict[str, torch.Tensor], right_model: Dict[str, torch.Tensor]) -> float:
        """Compute normalized L2 divergence between two model states."""
        total_sq = 0.0
        total_params = 0
        
        for name in left_model:
            if name in right_model:
                diff = left_model[name] - right_model[name]
                total_sq += torch.sum(diff * diff).item()
                total_params += left_model[name].numel()
        
        div = math.sqrt(total_sq) / max(1, total_params) if total_params > 0 else 0.0
        self.metrics["divergence"] = div
        return div

    def sync_step(self, left_model: Dict[str, torch.Tensor], right_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Conditionally syncs models. In island mode, sync only if divergence exceeds threshold.
        Otherwise, returns the weighted average (FedAvg).
        """
        self.round_count += 1
        div = self.compute_divergence(left_model, right_model)
        
        if self.island_mode and div < self.threshold:
            logger.info(f"Round {self.round_count}: Island Mode Active (divergence {div:.4f} < {self.threshold}). Skipping sync.")
            return left_model # In island mode, we return primary branch
        
        logger.info(f"Round {self.round_count}: Syncing models (divergence {div:.4f}).")
        self.metrics["sync_events"] += 1
        return self._merge_models(left_model, right_model)

    def _merge_models(self, m1: Dict[str, torch.Tensor], m2: Dict[str, torch.Tensor], w1: float = 0.5) -> Dict[str, torch.Tensor]:
        """Merge two models via weighted average."""
        w2 = 1.0 - w1
        new_state = {}
        for name in m1:
            if name in m2:
                new_state[name] = m1[name] * w1 + m2[name] * w2
            else:
                new_state[name] = m1[name]
        return new_state

    def enter_island_mode(self):
        self.island_mode = True
        logger.info("Entering Island Mode: Hemispheres operating independently.")

    def exit_island_mode(self):
        self.island_mode = False
        logger.info("Exiting Island Mode: Global synchronization forced.")
