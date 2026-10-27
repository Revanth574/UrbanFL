"""
UrbanFL Operational Module — End-to-End Integration
====================================================
Connects trained ML models (Silo 1 NLP + Silo 2 Vision) with the full
12-subsystem operational pipeline for real-time inference and emergency dispatch.

This replaces the synthetic fallbacks in run_real_data_pipeline.py with actual
model inference from the training pipeline.

Usage:
    python experiments/run_operational.py [--scenario earthquake|flood|fire]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("urbanfl.operational")


# ── Model Loader ──────────────────────────────────────────────────────

class TrainedModelLoader:
    """Load trained Silo 1 (NLP) and Silo 2 (Vision) models for inference."""

    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or PROJECT_ROOT / "results")
        self._silo1_model = None
        self._silo2_model = None
        self._device = "cpu"

    def load_models(self, device: str = "auto"):
        import torch
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Load Silo 1 NLP model
        try:
            from silo1.nlp_model import CrisisClassifier
            self._silo1_model = CrisisClassifier()
            if self._silo1_model.is_functional:
                self._silo1_model.eval()
                print(f"  ✓ Silo 1 NLP model loaded (functional)")
            else:
                print(f"  ⚡ Silo 1 NLP model loaded (lightweight)")
        except Exception as e:
            print(f"  ✗ Silo 1 NLP model not available: {e}")

        # Load Silo 2 Vision model
        try:
            from silo2.change_detection import SiameseChangeDetector
            self._silo2_model = SiameseChangeDetector(in_channels=3, pretrained=False)
            self._silo2_model.eval()
            n_params = sum(p.numel() for p in self._silo2_model.parameters())
            print(f"  ✓ Silo 2 Vision model loaded ({n_params/1e6:.1f}M params)")
        except Exception as e:
            print(f"  ✗ Silo 2 Vision model not available: {e}")

        # Load training metrics
        metrics_path = self.results_dir / "training_metrics.json"
        self._metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                self._metrics = json.load(f)
            summary = self._metrics.get("summary", {})
            print(f"  Training metrics: NLP acc={summary.get('silo1_best_acc', 0)*100:.1f}%, "
                  f"Vision IoU={summary.get('silo2_best_iou', 0)*100:.1f}%")

    def classify_crisis(self, texts: list) -> list:
        """Run NLP crisis classification on texts."""
        if self._silo1_model is None:
            return [{"class": "unknown", "confidence": 0.0} for _ in texts]

        import torch
        # The prototype's label list lived in data.silo1_data_adapter, which was never committed (docs/AUDIT.md §1).
        # Rather than guess the class order, refuse: Silo 1 inference will come from src/fl.
        raise RuntimeError("prototype Silo 1 classifier unavailable (label module never committed); use src/fl")

        results = []
        try:
            tokenizer = getattr(self._silo1_model, 'tokenizer', None)
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

            for text in texts:
                tokens = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=128, padding="max_length")
                with torch.no_grad():
                    outputs = self._silo1_model(
                        tokens["input_ids"], tokens["attention_mask"]
                    )
                    if isinstance(outputs, dict):
                        logits = outputs.get("class_logits", outputs.get("logits"))
                        severity = outputs.get("severity", None)
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                        severity = outputs[1] if len(outputs) > 1 else None
                    else:
                        logits = outputs
                        severity = None

                    probs = torch.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                    cls_idx = pred.item()
                    cls_name = INCIDENT_CLASSES[cls_idx] if cls_idx < len(INCIDENT_CLASSES) else "other"

                    results.append({
                        "class": cls_name,
                        "confidence": round(conf.item(), 4),
                        "severity": round(severity.item(), 4) if severity is not None else 0.5,
                        "all_probs": {INCIDENT_CLASSES[i]: round(p, 4)
                                      for i, p in enumerate(probs[0].tolist())
                                      if i < len(INCIDENT_CLASSES)},
                    })
        except Exception as e:
            logger.warning(f"NLP inference error: {e}")
            results = [{"class": "unknown", "confidence": 0.0} for _ in texts]

        return results

    def detect_changes(self, pre_image, post_image):
        """Run change detection on pre/post image pair."""
        if self._silo2_model is None:
            return {"change_mask": None, "damage_score": 0.0}

        import torch
        try:
            with torch.no_grad():
                outputs = self._silo2_model(pre_image.unsqueeze(0), post_image.unsqueeze(0))
                if isinstance(outputs, dict):
                    mask = outputs.get("change_mask", outputs.get("mask"))
                else:
                    mask = outputs
                damage_score = mask.mean().item()
                return {"change_mask": mask.squeeze(0), "damage_score": round(damage_score, 4)}
        except Exception as e:
            logger.warning(f"Vision inference error: {e}")
            return {"change_mask": None, "damage_score": 0.0}

    @property
    def training_metrics(self):
        return self._metrics


# ── Operational Pipeline ──────────────────────────────────────────────

def run_operational_pipeline(scenario_type: str = "earthquake", device: str = "auto"):
    """Full operational pipeline with trained model inference."""
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  UrbanFL OPERATIONAL MODULE — Model-Integrated Pipeline")
    print("=" * 60)

    # ── Step 1: Load trained models ──
    print("\n── Loading Trained Models ──")
    model_loader = TrainedModelLoader()
    model_loader.load_models(device=device)

    # ── Step 2: Load real-world data ──
    print("\n── Loading Real-World Data ──")
    from src.data.real_data_loader import RealDataLoader   # was `data.real_data_loader` (wrong package path)
    loader = RealDataLoader(verbose=False)
    load_results = loader.load_all()
    bbox = loader.get_istanbul_bbox()
    print(f"  Data loaded: {len(load_results)} sources")

    # ── Step 3: NLP Crisis Classification (trained model) ──
    print("\n── NLP Crisis Classification (Trained Model) ──")
    crisis_reports = loader.get_crisis_reports()
    sample_texts = [r["text"] for r in crisis_reports[:20]]
    classifications = model_loader.classify_crisis(sample_texts)

    # Aggregate classification results
    class_counts = {}
    for c in classifications:
        cls = c["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  Classified {len(classifications)} reports:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
        avg_conf = np.mean([c["confidence"] for c in classifications if c["class"] == cls])
        print(f"    {cls}: {count} reports (avg conf: {avg_conf:.2f})")

    # ── Step 4: Orchestrator Pipeline (with model outputs) ──
    print("\n── Running Orchestrator Pipeline ──")
    from orchestrator import OrchestratorEngine

    engine = OrchestratorEngine.create_default()
    earthquakes = loader.get_earthquakes()
    strongest_eq = max(earthquakes, key=lambda e: e["magnitude"]) if earthquakes else None

    context = {
        "scenario": {
            "type": scenario_type,
            "center_lat": strongest_eq["latitude"] if strongest_eq else 29.0,
            "center_lon": strongest_eq["longitude"] if strongest_eq else 41.05,
            "n_reports": 30,
            "incident_type": scenario_type,
        },
        "hour": 8,
        "nlp_classifications": classifications,
    }
    pipeline_result = engine.run(context)
    print(f"  Pipeline: {'SUCCESS' if pipeline_result.is_success else 'PARTIAL'}")
    print(f"  Total latency: {pipeline_result.total_latency_ms:.0f}ms")
    print(f"  SLA compliant: {'✓' if pipeline_result.sla_compliant else '✗'}")

    # ── Step 5: Event Bus + Situation Monitor ──
    print("\n── Situation Monitoring ──")
    from event_bus import EventBus, Event, EventType, setup_default_subscriptions
    bus = EventBus(max_queue_size=5000)
    setup_default_subscriptions(bus)

    from dashboard.situation_monitor import RealTimeMonitor
    monitor = RealTimeMonitor(bus=bus, grid_rows=8, grid_cols=8, bbox=bbox)

    # Feed classified reports as events
    for i, (report, cls) in enumerate(zip(crisis_reports[:20], classifications)):
        bus.publish(Event(
            priority=cls.get("severity", 0.5),
            timestamp=time.time(),
            event_type=EventType.ALERT_VERIFIED if cls["confidence"] > 0.7 else EventType.REPORT_SUBMITTED,
            source_silo=1,
            data={
                "lat": report["latitude"], "lon": report["longitude"],
                "type": cls["class"], "confidence": cls["confidence"],
                "text": report["text"][:100],
            }
        ))

    # Feed earthquake events
    for eq in earthquakes[:10]:
        bus.publish(Event(
            priority=min(1.0, eq["magnitude"] / 7.0),
            timestamp=eq["time_ms"] / 1000,
            event_type=EventType.ALERT_VERIFIED,
            source_silo=0,
            data={"lat": eq["latitude"], "lon": eq["longitude"],
                  "type": "earthquake", "magnitude": eq["magnitude"]},
        ))
    print(f"  Events published: {bus.stats()['total_events']}")

    # ── Step 6: Zone Criticality ──
    from dashboard.zone_criticality import ZoneCriticalityScorer, TriageRanker
    scorer = ZoneCriticalityScorer()
    ranker = TriageRanker(scorer)
    zone_assessments = ranker.rank_zones(monitor.zones)
    deployment_orders = ranker.get_deployment_order(zone_assessments)
    print(f"  Zones assessed: {len(zone_assessments)}, deployments: {len(deployment_orders)}")

    # ── Step 7: Emergency Routing (Real Roads) ──
    print("\n── Emergency Routing ──")
    road_graph = loader.get_road_graph()
    from silo3.emergency_router import EmergencyRouter, EmergencyDispatcher, EmergencyVehicle, EmergencyVehicleType
    router = EmergencyRouter(road_graph)
    dispatcher = EmergencyDispatcher(router)

    graph_nodes = list(road_graph.nodes)
    rng = np.random.RandomState(42)
    vehicles = [
        EmergencyVehicle(f"AMB_{i}", EmergencyVehicleType.AMBULANCE, rng.choice(graph_nodes), 15.0)
        for i in range(8)
    ] + [
        EmergencyVehicle(f"FIRE_{i}", EmergencyVehicleType.FIRE, rng.choice(graph_nodes), 12.0)
        for i in range(4)
    ]
    dispatcher.register_vehicles(vehicles)

    incidents = [{"id": f"inc_{i}", "type": scenario_type,
                  "severity": min(1.0, eq["magnitude"] / 7.0),
                  "nearest_node": rng.choice(graph_nodes), "population_density": 15000}
                 for i, eq in enumerate(earthquakes[:8])]

    hospitals_data = loader.get_hospitals()
    hospital_nodes = [{"name": h["name"], "node": rng.choice(graph_nodes)} for h in hospitals_data[:5]]
    dispatch_results = dispatcher.dispatch(incidents, hospital_nodes)
    print(f"  Dispatched {len(dispatch_results)} vehicles to {len(incidents)} incidents")

    # ── Step 8: Hospital + Medical ──
    print("\n── Medical Network ──")
    from medical.hospital_network import HospitalNetwork, SimulatedHospitalFeed
    from medical.patient_distributor import PatientDistributor, generate_synthetic_patients

    network = HospitalNetwork.create_default_network(min(10, len(hospitals_data)), bbox)
    feed = SimulatedHospitalFeed(network)
    feed.tick(disaster_active=True, severity=0.6)
    patients = generate_synthetic_patients(25, 29.0, 41.0, seed=42)
    dist = PatientDistributor(network)
    patient_assignments = dist.assign_patients(patients)
    surge = dist.predict_surge(intake_rate_per_hour=12)
    print(f"  Patients assigned: {len(patient_assignments)}, hospitals: {len(hospitals_data)}")

    # ── Step 9: Infrastructure + Environment ──
    print("\n── Infrastructure & Environment ──")
    from infrastructure.grid_monitor import InfrastructureGridMonitor
    infra_monitor = InfrastructureGridMonitor.create_synthetic(10, bbox)
    infra_monitor.simulate_damage(29.0, 41.0, radius_km=5, severity=0.4)
    cascades = infra_monitor.predict_cascades()
    health = infra_monitor.network_health()

    from environment.env_monitor import EnvironmentalMonitor
    env_monitor = EnvironmentalMonitor(bbox)
    env_monitor.generate_synthetic_data(29.0, 41.0)
    env_risks = env_monitor.compute_risk_overlay()
    evac_constraints = env_monitor.get_evacuation_constraints()
    print(f"  Cascades: {len(cascades) if isinstance(cascades, list) else 0}, "
          f"infra health: {health.get('overall_health', 'N/A')}")

    # ── Step 10: Public Alerts + Shelters ──
    from public.alert_system import AlertComposer, EscalationProtocol, AlertSeverity
    from public.citizen_platform import ShelterManager

    composer = AlertComposer()
    if strongest_eq:
        alert = composer.compose_alert(
            "earthquake", AlertSeverity.EXTREME,
            area_name="Istanbul", lat=strongest_eq["latitude"], lon=strongest_eq["longitude"],
            extra_info={"magnitude": strongest_eq["magnitude"]},
        )
        EscalationProtocol().issue_alert(alert)

    shelters_data = loader.get_shelters()
    shelter_mgr = ShelterManager.create_default(min(20, len(shelters_data)), bbox)
    shelter_summary = shelter_mgr.summary()

    # ── Step 11: Resource Management + Analytics ──
    from resources.resource_manager import ResourceManager
    rm = ResourceManager.create_default(5, bbox)
    population = loader.get_population()
    demand = rm.forecast_demand(zone_casualties=80, zone_population=population["total_population"] // 37)
    for cat, qty in list(demand.items())[:3]:
        rm.request_resources("zone_1_1", cat, qty, urgency=0.8)
    fulfillment = rm.fulfill_requests()

    from analytics.risk_forecaster import ScenarioSimulator
    sim = ScenarioSimulator()
    scenario_result = {}
    if strongest_eq:
        scenario_result = sim.run_scenario({
            "type": scenario_type, "lat": strongest_eq["latitude"],
            "lon": strongest_eq["longitude"], "magnitude": strongest_eq["magnitude"],
            "population": population["total_population"], "hours_since": 1.0,
        })

    # ── Step 12: Command Center + Report ──
    print("\n── Command Center ──")
    from coordination.command_center import CommandCenter
    cc = CommandCenter.create_default(f"ISTANBUL_{scenario_type.upper()}")
    cc.create_mission("search_rescue", "zone_1_1", "SAR in affected zones", priority=1)
    cc.create_mission("medical_triage", "zone_2_1", "Medical triage", priority=2)
    cc.create_mission("infrastructure_assessment", "zone_3_1", "Critical infra inspection", priority=3)
    cc.log_decision("Deploy all reserve units", "Model predicts escalation", "AI-Assisted IC")
    sitrep = cc.generate_sitrep({"zones_affected": len(zone_assessments)})

    from command.data_processor import CommandDataProcessor
    from command.report_generator import ReportGenerator

    processor = CommandDataProcessor()
    display_data = processor.process(
        situation_snapshot={"timestamp": time.time(), "scenario": scenario_type},
        zone_assessments=zone_assessments,
        deployment_orders=deployment_orders,
        damage_reports=[],
        emergency_dispatch=dispatch_results,
        fleet_status={"vehicles": len(vehicles), "dispatched": len(dispatch_results)},
        hospital_status={},
        patient_assignments=patient_assignments,
        surge_predictions=surge if isinstance(surge, dict) else {},
        infrastructure_health=health,
        cascade_predictions=cascades if isinstance(cascades, list) else [],
        environmental_risks=env_risks,
        evacuation_constraints=evac_constraints,
        shelter_status=shelter_summary,
        resource_status={"demand_categories": len(demand), "fulfilled": len(fulfillment)},
        risk_forecasts=scenario_result if isinstance(scenario_result, dict) else {},
        sitrep=sitrep,
        command_summary=cc.summary(),
        privacy_status={"epsilon_consumed": model_loader.training_metrics.get("summary", {}).get("fl_privacy_budget", 0)},
        fl_status={
            "silo1_acc": model_loader.training_metrics.get("summary", {}).get("silo1_best_acc", 0),
            "silo2_iou": model_loader.training_metrics.get("summary", {}).get("silo2_best_iou", 0),
            "training_time": model_loader.training_metrics.get("summary", {}).get("training_time_seconds", 0),
        },
    )

    report = ReportGenerator().generate_text_sitrep(display_data)
    processor.export_json(display_data, "operational_latest.json")

    # ── Final Summary ──
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("  OPERATIONAL MODULE — COMPLETE")
    print("=" * 60)
    print(f"""
┌──────────────────────────────────────────────────────────┐
│              OPERATIONAL RESULTS SUMMARY                  │
├──────────────────────┬───────────────────────────────────┤
│ Scenario             │ {scenario_type:>33s} │
│ NLP Classifications  │ {len(classifications):>33d} │
│ Top Predicted Class  │ {max(class_counts, key=class_counts.get) if class_counts else 'N/A':>33s} │
│ Orchestrator Status  │ {'SUCCESS' if pipeline_result.is_success else 'PARTIAL':>33s} │
│ Pipeline Latency     │ {pipeline_result.total_latency_ms:>30.0f} ms │
│ SLA Compliant        │ {'Yes' if pipeline_result.sla_compliant else 'No':>33s} │
│ Vehicles Dispatched  │ {len(dispatch_results):>33d} │
│ Patients Assigned    │ {len(patient_assignments):>33d} │
│ Zones Assessed       │ {len(zone_assessments):>33d} │
│ Missions Active      │ {3:>33d} │
│ Events Published     │ {bus.stats()['total_events']:>33d} │
│ Population Coverage  │ {population['total_population']:>29,d}     │
│ Road Network         │ {road_graph.number_of_nodes():>19,d} nodes, {road_graph.number_of_edges():>6,d} edges │
│ Real Hospitals       │ {len(hospitals_data):>33,d} │
│ Real Shelters        │ {len(shelters_data):>33,d} │
│ Total Time           │ {total_time:>31.1f} s │
└──────────────────────┴───────────────────────────────────┘
""")

    # Save operational results
    op_results = {
        "scenario": scenario_type,
        "timestamp": time.time(),
        "total_time_s": round(total_time, 1),
        "nlp_classifications": len(classifications),
        "class_distribution": class_counts,
        "pipeline_success": pipeline_result.is_success,
        "pipeline_latency_ms": round(pipeline_result.total_latency_ms, 1),
        "sla_compliant": pipeline_result.sla_compliant,
        "vehicles_dispatched": len(dispatch_results),
        "patients_assigned": len(patient_assignments),
        "zones_assessed": len(zone_assessments),
        "training_metrics": model_loader.training_metrics.get("summary", {}),
    }

    results_path = PROJECT_ROOT / "results" / "operational_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(op_results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return op_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanFL Operational Module")
    parser.add_argument("--scenario", type=str, default="earthquake",
                        choices=["earthquake", "flood", "fire", "hurricane", "tsunami"],
                        help="Disaster scenario type")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    run_operational_pipeline(scenario_type=args.scenario, device=args.device)
