"""
UrbanFL — End-to-End Pipeline
Runs the complete 3-silo + FedHetero-Robust + GeoPrivacyFL pipeline.
This is both the experiment driver and the smoke test.
"""

import time
import logging
import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from src.utils.core import (UrbanFLConfig, get_device, set_seed, setup_logging,
                             MetricsTracker, Timer, DisasterScenario, SiloClientUpdate)
from src.silo1.credibility_scorer import CredibilityScorer, IoTAnomalyAutoencoder, generate_synthetic_reports
from src.silo1.nlp_model import CrisisClassifier
from src.silo1.causal_verifier import CausalVerifier
from src.silo2.change_detection import SiameseChangeDetector
from src.silo2.aerial_mesh_sim import AerialMeshSimulator
from src.silo2.coverage_theorem import CoverageTheorems, TheoremValidator
from src.silo3.traffic_engine import TrafficStateEngine, SimulatedTrafficGenerator
from src.silo3.routing_optimizer import DynamicRoadGraph, ResponseOptimizer, ConsequencePropagationModel
from src.federated.fedhetero import FedHeteroRobust
from src.federated.privacy_analysis import GeoPrivacyFL
from src.federated.dual_aggregator import DualAggregator
from src.federated.communication import CommunicationManager
from src.federated.concept_drift import ConceptDriftDetector
from src.event_bus import EventBus, setup_default_subscriptions, Event, EventType


logger = logging.getLogger("urbanfl.pipeline")


def create_placeholder_model(in_features: int = 256, out_features: int = 8) -> nn.Module:
    """Create a small placeholder model for FL simulation."""
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, out_features),
    )


def run_full_pipeline(config_dir: str = "config", n_rounds: int = 20,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete UrbanFL pipeline end-to-end.
    Returns: comprehensive results dictionary.
    """
    results = {}

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    cfg = UrbanFLConfig(config_dir)
    set_seed(cfg.get("seed", 42))
    log = setup_logging("INFO", console=verbose)
    device = get_device(cfg.get("compute.device", "cpu"))
    metrics = MetricsTracker()

    log.info(f"UrbanFL Pipeline — Device: {device}")

    # -----------------------------------------------------------------------
    # Phase 1: Initialize Silos
    # -----------------------------------------------------------------------
    with Timer("silo_init", log):
        # Silo 1: Alert Ingestion
        scorer = CredibilityScorer(scoring_mode=cfg.get("credibility.scoring_mode", "weighted_product"))
        iot_ae = IoTAnomalyAutoencoder(n_timesteps=60, n_sensors=5)
        scorer.set_iot_autoencoder(iot_ae)
        verifier = CausalVerifier()

        # Silo 2: Aerial Intelligence
        mesh_sim = AerialMeshSimulator()
        mesh_sim.setup_default_mesh()

        # Silo 3: Traffic & Routing
        road_graph = DynamicRoadGraph.from_synthetic(n_nodes=150)
        engine = TrafficStateEngine(road_graph.graph)
        traffic_gen = SimulatedTrafficGenerator(engine)
        optimizer = ResponseOptimizer(road_graph)

        # Event Bus
        bus = EventBus(max_queue_size=500)
        setup_default_subscriptions(bus)

    log.info("All silos initialized")

    # -----------------------------------------------------------------------
    # Phase 2: Simulate Disaster Scenario
    # -----------------------------------------------------------------------
    scenario = DisasterScenario.earthquake()
    log.info(f"Scenario: {scenario.name} at ({scenario.center_lat}, {scenario.center_lon})")

    # Generate synthetic alerts
    reports = generate_synthetic_reports(
        scenario_type=scenario.incident_type,
        n_reports=60,
        center_lat=scenario.center_lat,
        center_lon=scenario.center_lon,
    )

    # Score alerts
    scores = scorer.score_batch(reports)
    escalated = [r for r in reports if scorer.should_escalate(r)]
    log.info(f"Generated {len(reports)} reports, {len(escalated)} escalated "
             f"(mean credibility: {np.mean(scores):.3f})")
    results["silo1_mean_credibility"] = float(np.mean(scores))
    results["silo1_escalated_count"] = len(escalated)

    # Publish verified alerts to event bus
    for r in escalated[:5]:
        bus.publish(Event(
            priority=1 - r.confidence,
            timestamp=r.timestamp,
            event_type=EventType.ALERT_VERIFIED,
            source_silo=0,
            data={"lat": r.latitude, "lon": r.longitude, "type": r.incident_type},
        ))

    # Aerial coverage
    coverage = mesh_sim.compute_coverage_mc(n_test_points=2000)
    response_times = mesh_sim.compute_response_time_mc(n_incidents=2000)
    log.info(f"Coverage: {coverage:.2%}, Mean response: {response_times['mean']:.1f}s")
    results["silo2_coverage"] = coverage
    results["silo2_mean_response_s"] = response_times["mean"]

    # Coverage theorems
    theorems = CoverageTheorems()
    theorem_results = theorems.compute_all()
    log.info(f"Theorem 2 bound: {theorem_results['theorem2_coverage']:.4f}")
    log.info(f"Theorem 2b (correlated): {theorem_results['theorem2b_correlated_coverage']:.4f}")
    results["theorems"] = theorem_results

    # Traffic
    traffic_gen.generate_traffic(hour=8)

    # -----------------------------------------------------------------------
    # Phase 3: Federated Learning with GeoPrivacyFL
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FEDERATED LEARNING — GeoPrivacyFL")
    log.info("=" * 60)

    # Create per-silo models (uniform architecture for FL aggregation;
    # data heterogeneity is in what each silo trains on, not model shape)
    silo_models = {
        0: create_placeholder_model(256, 8),   # Silo 1: NLP
        1: create_placeholder_model(256, 8),   # Silo 2: Vision
        2: create_placeholder_model(256, 8),   # Silo 3: Traffic
    }

    # Unified model state for FL (simplified: use silo 0's structure)
    global_state = {k: v.clone() for k, v in silo_models[0].state_dict().items()}

    # Initialize FL components
    fedhetero = FedHeteroRobust(
        global_model_state=global_state,
        n_silos=3,
        byzantine_method=cfg.get("fedhetero.byzantine.method", "coordinate_trimmed_mean"),
    )

    privacy_fl = GeoPrivacyFL(cfg.raw, n_silos=3)
    dual_agg = DualAggregator()
    comm_mgr = CommunicationManager(
        method=cfg.get("communication.method", "topk"),
        k_fraction=cfg.get("communication.topk_fraction", 0.1),
    )
    drift_detector = ConceptDriftDetector(n_silos=3)

    # FL Training Loop
    for fl_round in range(1, n_rounds + 1):
        round_start = time.perf_counter()

        silo_updates = []
        for silo_id in range(3):
            model = silo_models[silo_id]
            model.load_state_dict(global_state)

            # Simulate local training (generate synthetic gradients)
            grad = {k: torch.randn_like(v) * 0.01 for k, v in model.state_dict().items()}

            # GeoPrivacyFL pipeline
            sanitised, privacy_meta = privacy_fl.process_gradient(silo_id, grad, fl_round)

            # Communication compression
            compressed, comp_stats = comm_mgr.compress(sanitised)

            # Concept drift check
            fake_dist = np.random.dirichlet(np.ones(8))
            drift_detector.record_distribution(silo_id, fake_dist)

            # Build update
            new_weights = {k: global_state[k] + compressed.get(k, torch.zeros_like(v))
                          for k, v in global_state.items()}

            update = SiloClientUpdate(
                silo_id=silo_id,
                round_number=fl_round,
                weights=new_weights,
                dataset_size=1000 + silo_id * 500,
                train_loss=1.0 / (fl_round + 1) + np.random.normal(0, 0.05),
                train_metrics={"accuracy": min(0.95, 0.5 + fl_round * 0.03)},
                timestamp=time.time(),
                data_quality=scorer.get_fl_quality_signal() if silo_id == 0 else 0.7,
            )
            silo_updates.append(update)

        # FedHetero-Robust aggregation
        global_state = fedhetero.run_round(silo_updates)

        # Dual aggregator sync
        dual_agg.left.update_model(global_state)
        dual_agg.right.update_model(global_state)
        sync_result = dual_agg.sync_step()

        round_time = (time.perf_counter() - round_start) * 1000
        metrics.log("round_time_ms", round_time)

        if fl_round % 5 == 0 or fl_round == 1:
            log.info(f"Round {fl_round}/{n_rounds}: "
                     f"time={round_time:.0f}ms, "
                     f"sync={sync_result.get('action', 'N/A')}")

    # -----------------------------------------------------------------------
    # Phase 4: Privacy Analysis
    # -----------------------------------------------------------------------
    silo_eps = {i: privacy_fl.dp_engine.compute_silo_epsilon(i) for i in range(3)}
    weights = fedhetero.weighter.compute_weights(silo_updates, n_rounds)
    global_privacy = privacy_fl.dp_engine.compute_global_privacy(silo_eps, weights)

    log.info(f"\n{'='*60}")
    log.info(f"PRIVACY CERTIFICATE")
    log.info(f"{'='*60}")
    log.info(f"Per-silo ε: {silo_eps}")
    log.info(f"Silo weights: {weights}")
    log.info(f"Global ε (weight-aware RDP): {global_privacy['epsilon_total']:.4f}")
    log.info(f"Naive composition:           {global_privacy['naive_composition']:.4f}")
    log.info(f"Improvement factor:          {global_privacy['improvement_factor']:.2f}×")

    results["privacy"] = global_privacy
    results["silo_epsilons"] = silo_eps

    if privacy_fl.budget_manager:
        cert = privacy_fl.budget_manager.issue_certificate()
        log.info(f"Budget certificate: {cert}")
        results["budget_certificate"] = cert

    # -----------------------------------------------------------------------
    # Phase 5: Response Optimization Test
    # -----------------------------------------------------------------------
    incidents = [
        {"id": f"inc_{i}", "lat": scenario.center_lat + np.random.normal(0, 0.02),
         "lon": scenario.center_lon + np.random.normal(0, 0.02),
         "severity": np.random.uniform(0.5, 1.0), "population_density": 5000,
         "nearest_node": np.random.randint(0, 150), "casualties": np.random.randint(0, 20)}
        for i in range(10)
    ]
    vehicles = [
        {"id": f"veh_{i}", "nearest_node": np.random.randint(0, 150), "speed_mps": 15}
        for i in range(5)
    ]
    hospitals = [
        {"name": f"Hospital_{i}", "node": np.random.randint(0, 150)}
        for i in range(3)
    ]

    with Timer("response_optimization", log):
        assignments = optimizer.optimize_response(incidents, vehicles, hospitals)

    results["response_assignments"] = len(assignments)
    results["event_bus_stats"] = bus.stats()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Avg round time: {metrics.mean('round_time_ms'):.0f}ms")
    log.info(f"Events processed: {bus.stats()['total_events']}")
    log.info(f"Assignments: {len(assignments)}")

    results["avg_round_time_ms"] = metrics.mean("round_time_ms")
    return results


if __name__ == "__main__":
    results = run_full_pipeline(n_rounds=20, verbose=True)
    print(f"\nFinal Results: {results}")
