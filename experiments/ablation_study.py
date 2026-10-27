"""
DEPRECATED / SYNTHETIC — minor-project prototype. Produces NO real results; do not run or cite.

Why (docs/AUDIT.md §4A): ablation accuracies are 1 - (1/(r+1) + hardcoded per-component penalties) + noise; no model is trained.
It also imports prototype modules that no longer exist. Real Silo 1 experiments: experiments/silo1_baselines.py and experiments/silo1_fl.py.

Original description:
UrbanFL — Ablation Study [Review §3.3]
Comprehensive ablation to measure each component's contribution.
"""

raise SystemExit("experiments/ablation_study.py is a DEPRECATED synthetic prototype script: its numbers are formulas, not experiments (see the module docstring and docs/AUDIT.md section 4A).")

import logging
import numpy as np
from typing import Any, Dict, List
from src.utils.core import MetricsTracker

logger = logging.getLogger("urbanfl.experiments.ablation")


ABLATION_CONFIGS = {
    "full_system": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "no_gradient_norm": {
        "gradient_normalisation": False,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "no_adaptive_weight": {
        "gradient_normalisation": True,
        "adaptive_weighting": False,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "no_semantic_bridge": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": False,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "no_byzantine": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": False,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "sum_credibility": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": False,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "no_gradient_decomp": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": False,
        "dual_aggregator": True,
        "per_silo_dp": True,
    },
    "single_aggregator": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": False,
        "per_silo_dp": True,
    },
    "uniform_dp": {
        "gradient_normalisation": True,
        "adaptive_weighting": True,
        "semantic_bridge": True,
        "byzantine_resilience": True,
        "credential_product": True,
        "gradient_decomposition": True,
        "dual_aggregator": True,
        "per_silo_dp": False,
    },
}


def run_ablation_study(n_rounds: int = 50) -> Dict[str, Dict[str, float]]:
    """Run all ablation configurations and collect metrics."""
    results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        logger.info(f"Running ablation: {config_name}")
        metrics = MetricsTracker()

        # Simulate FL training with this config
        np.random.seed(42)
        for r in range(n_rounds):
            base_loss = 1.0 / (r + 1)

            # Each disabled component adds a penalty
            penalty = 0.0
            if not config["gradient_normalisation"]:
                penalty += 0.08  # ~8% accuracy loss
            if not config["adaptive_weighting"]:
                penalty += 0.05
            if not config["semantic_bridge"]:
                penalty += 0.03
            if not config["byzantine_resilience"]:
                penalty += 0.02 * np.random.choice([0, 1], p=[0.9, 0.1])
            if not config["credential_product"]:
                penalty += 0.04
            if not config["gradient_decomposition"]:
                penalty += 0.06  # Privacy-utility tradeoff worse
            if not config["per_silo_dp"]:
                penalty += 0.07

            noise = np.random.normal(0, 0.02)
            loss = base_loss + penalty + noise
            accuracy = max(0, min(1, 1 - loss + 0.1))

            metrics.log("loss", loss)
            metrics.log("accuracy", accuracy)

        results[config_name] = {
            "final_loss": metrics.last("loss"),
            "final_accuracy": metrics.last("accuracy"),
            "mean_accuracy": metrics.mean("accuracy"),
            "disabled": [k for k, v in config.items() if not v],
        }
        logger.info(f"  → accuracy={results[config_name]['final_accuracy']:.4f}")

    return results


def format_ablation_table(results: Dict[str, Dict]) -> str:
    """Format ablation results as a markdown table."""
    lines = ["| Configuration | Disabled Component | Final Accuracy | Δ vs Full |",
             "|---|---|---|---|"]

    full_acc = results.get("full_system", {}).get("final_accuracy", 0)

    for name, data in sorted(results.items()):
        disabled = ", ".join(data.get("disabled", [])) or "None"
        acc = data.get("final_accuracy", 0)
        delta = acc - full_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"| {name} | {disabled} | {acc:.4f} | {delta_str} |")

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_ablation_study()
    print(format_ablation_table(results))
