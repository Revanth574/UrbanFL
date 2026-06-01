import logging
import math
import random
from typing import Dict, Any, Optional, List
from src.federated.dual_aggregator import DualAggregator
from src.federated.fedhetero import FedHetero

logger = logging.getLogger("urbanfl.orchestrator")

class OrchestratorEngine:
    """
    Central node orchestrating the federated learning cycles and cross-silo aggregation.
    Integrates DualAggregator for model syncing and FedHetero for client weighting.
    """
    def __init__(self, mode: str = "FEDERATED"):
        self.mode = mode
        self.status = "INITIALIZING"
        self._silo_registry: List[str] = ["silo1", "silo2", "silo3"]
        self.current_round = 0
        self.privacy_budget = 2.0  # ε epsilon tracking
        
        # Innovations
        self.aggregator = DualAggregator(island_mode=False)
        self.hetero_manager = FedHetero(self._silo_registry)
        
        # Metrics History
        self.metrics = {
            "convergence": [],
            "divergence": [],
            "hetero_index": [],
        }

    @classmethod
    def create_default(cls) -> "OrchestratorEngine":
        """Factory method to generate the default European Istanbul configuration."""
        engine = cls(mode="FEDERATED_SIMULATION")
        engine.status = "READY"
        logger.info("OrchestratorEngine default configuration loaded with FedHetro and DualAggregator.")
        return engine

    def execute_federated_round(self):
        """
        Execute a single communication round (t).
        Simulates the interaction between FedHetero weighting and DualAggregator syncing.
        """
        self.current_round += 1
        
        # 1. Simulate client metrics for FedHetero
        client_metrics = {}
        for silo in self._silo_registry:
            client_metrics[silo] = {
                "accuracy": 0.7 + (random.random() * 0.25),
                "latency": 50 + random.randint(0, 200)
            }
        self.hetero_manager.update_weights(client_metrics)
        
        # 2. Simulate model states for DualAggregator
        # In a real system, these would be torch state_dicts
        div = 0.1 + (random.random() * 0.4)
        self.aggregator.metrics["divergence"] = div
        
        # 3. Privacy budget decay
        self.privacy_budget = max(0.1, self.privacy_budget - 0.02)
        
        # 4. Record innovation metrics
        h_index = self.hetero_manager.get_global_hetero_index()
        self.metrics["divergence"].append(div)
        self.metrics["hetero_index"].append(h_index)
        self.metrics["convergence"].append(0.5 + (1.0 - 0.5) * (1 - math.exp(-0.1 * self.current_round)))

        logger.info(f"FL Round {self.current_round} | Div: {div:.3f} | Hetro: {h_index:.3f} | ε: {self.privacy_budget:.2f}")

    def get_system_health(self) -> Dict[str, Any]:
        """Aggregate health metrics across the ecosystem including innovation indexes."""
        return {
            "status": self.status,
            "round": self.current_round,
            "epsilon": self.privacy_budget,
            "hetero_index": self.hetero_manager.get_global_hetero_index() if hasattr(self, 'hetero_manager') else 0.0,
            "divergence": self.aggregator.metrics["divergence"] if hasattr(self, 'aggregator') else 0.0,
            "registered_silos": len(self._silo_registry)
        }
