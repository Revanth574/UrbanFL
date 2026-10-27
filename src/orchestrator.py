"""
DEPRECATED — minor-project prototype. Do not extend; the real FL code is src/fl/ (runner.py, algorithms.py).

The original OrchestratorEngine fabricated every number it reported (docs/AUDIT.md §4A): per-silo "accuracy" was
0.7 + random()*0.25, "divergence" was 0.1 + random()*0.4, "convergence" was a closed-form 1 - exp(-0.1 t), and ε was
decremented by hand (-0.02 per round). Those values reached the dashboard and the legacy paper's FL table.

This stub keeps the interface that api/state.py and api/routers/dispatch.py import, but reports nothing it cannot
compute: round, ε, heterogeneity and divergence are None and status is NOT_CONNECTED until the dashboard is wired to
src/fl (TODO, after Review 1).
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("urbanfl.orchestrator")


class OrchestratorEngine:
    """Placeholder for the dashboard's FL status. No training, no simulated metrics."""

    def __init__(self, mode: str = "NOT_CONNECTED"):
        self.mode = mode
        self.status = "NOT_CONNECTED"
        self._silo_registry: List[str] = ["silo1", "silo2", "silo3"]
        self.current_round = None      # no FL process is attached
        self.privacy_budget = None     # ε only ever comes from a real accountant (CLAUDE.md rule 3)

    @classmethod
    def create_default(cls) -> "OrchestratorEngine":
        logger.info("OrchestratorEngine stub: FL not connected to the dashboard (see src/fl).")
        return cls()

    def execute_federated_round(self) -> None:
        """No-op: the prototype advanced a fake round with random metrics here."""
        logger.debug("execute_federated_round: FL not connected; nothing to report.")

    def get_system_health(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "round": None,
            "epsilon": None,
            "hetero_index": None,
            "divergence": None,
            "registered_silos": len(self._silo_registry),
            "note": "FL not connected to the dashboard; prototype random metrics removed (docs/AUDIT.md §4A).",
        }
