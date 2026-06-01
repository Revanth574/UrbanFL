import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger("urbanfl.federated.hetero")

class FedHetero:
    """
    Handles client heterogeneity in the federated network.
    Weights client updates based on localized performance metrics,
    system latency, and data quality (Heterogeneity Index).
    """
    def __init__(self, silos: List[str]):
        self.silos = silos
        self.hetero_scores = {s: 1.0 for s in silos}
        self.history = {s: [] for s in silos}
        logger.info(f"FedHetero initialized for silos: {silos}")

    def update_weights(self, metrics: Dict[str, Dict[str, float]]):
        """
        Update heterogeneity scores based on client performance metrics.
        metrics: {silo_id: {"accuracy": 0.8, "latency": 150, ...}}
        """
        for silo, m in metrics.items():
            if silo not in self.hetero_scores: continue
            
            # Hetero Index = (Accuracy * 0.7) + (1/Latency * 0.3)
            acc = m.get("accuracy", 0.5)
            lat = m.get("latency", 100)
            
            # Normalized score calculation
            score = (acc * 0.8) + (100 / max(1, lat) * 0.2)
            self.hetero_scores[silo] = score
            self.history[silo].append(score)
            
        # Normalize weights to sum to 1.0
        total = sum(self.hetero_scores.values())
        if total > 0:
            for silo in self.hetero_scores:
                self.hetero_scores[silo] /= total

        logger.info(f"Heterogeneity Weights Updated: {self.hetero_scores}")

    def get_weighted_average_params(self, silo_updates: Dict[str, float]) -> float:
        """Helper to compute a weighted average primitive (telemetry only)."""
        weighted_sum = 0.0
        for silo, val in silo_updates.items():
            weighted_sum += val * self.hetero_scores.get(silo, 1.0/len(self.silos))
        return weighted_sum

    def get_global_hetero_index(self) -> float:
        """Calculate the overall network heterogeneity (Variance of scores)."""
        scores = list(self.hetero_scores.values())
        if not scores: return 0.0
        return float(np.std(scores))
