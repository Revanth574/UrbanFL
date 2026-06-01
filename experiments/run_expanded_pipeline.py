"""
UrbanCityOS — Expanded Pipeline
End-to-end demonstration running all 12 feature groups + command center.
Synthetic data throughout; no real APIs required.
"""

import sys
import time
import logging
import json
from pathlib import Path

# ── Ensure src/ is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx

# ── Core ─────────────────────────────────────────────────────────────
from src.event_bus import EventBus, setup_default_subscriptions
from src.utils.core import set_seed

# ── Group 1: Dashboard ───────────────────────────────────────────────
from src.dashboard.situation_monitor import RealTimeMonitor, SimulatedRealTimeFeed
from src.dashboard.zone_criticality import ZoneCriticalityScorer, TriageRanker
from src.dashboard.damage_assessment import DamageAssessor, DamageHeatmapGenerator, InfrastructureImpactAnalyzer

# ── Group 2: Emergency Routing ───────────────────────────────────────
from src.silo3.emergency_router import (
    EmergencyRouter, EmergencyDispatcher, EmergencyVehicle,
    EmergencyVehicleType, EmergencyRouteRequest
)
from src.silo3.aerial_nav_advisor import AerialNavAdvisor

# ── Group 3: Civilian Rerouting ──────────────────────────────────────
from src.silo3.civilian_rerouter import CivilianRerouteEngine

# ── Group 4: Hospital ────────────────────────────────────────────────
from src.medical.hospital_network import HospitalNetwork, SimulatedHospitalFeed
from src.medical.patient_distributor import PatientDistributor, generate_synthetic_patients

# ── Group 5: Infrastructure ──────────────────────────────────────────
from src.infrastructure.grid_monitor import InfrastructureGridMonitor

# ── Group 6: Environment ─────────────────────────────────────────────
from src.environment.env_monitor import EnvironmentalMonitor

# ── Group 7: Public ──────────────────────────────────────────────────
from src.public.alert_system import AlertComposer, EscalationProtocol, AlertSeverity
from src.public.citizen_platform import ShelterManager

# ── Group 8: Resources ───────────────────────────────────────────────
from src.resources.resource_manager import ResourceManager

# ── Group 9: Analytics ───────────────────────────────────────────────
from src.analytics.risk_forecaster import ScenarioSimulator

# ── Group 10: Coordination ───────────────────────────────────────────
from src.coordination.command_center import CommandCenter

# ── Group 11: Recovery ───────────────────────────────────────────────
from src.recovery.recovery_tracker import RecoveryTracker, RecoveryPhase

# ── Group 12: Command Data ───────────────────────────────────────────
from src.command.data_processor import CommandDataProcessor
from src.command.report_generator import ReportGenerator


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("urbanfl.expanded_pipeline")


def build_synthetic_graph(n_nodes: int = 200, seed: int = 42) -> nx.DiGraph:
    """Build a synthetic road network as a grid graph."""
    rng = np.random.RandomState(seed)
    g = nx.grid_2d_graph(int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)))
    dg = nx.DiGraph()
    for u, v in g.edges:
        uid = u[0] * int(np.sqrt(n_nodes)) + u[1]
        vid = v[0] * int(np.sqrt(n_nodes)) + v[1]
        length = rng.uniform(100, 500)
        speed = rng.choice([30, 40, 50, 60, 80])
        dg.add_edge(uid, vid, length=length, speed_kph=speed, lanes=rng.choice([1, 2, 3]))
        dg.add_edge(vid, uid, length=length, speed_kph=speed, lanes=rng.choice([1, 2, 3]))

    # Assign coordinates to nodes
    bbox = (28.9, 40.9, 29.3, 41.2)
    side = int(np.sqrt(n_nodes))
    for node in dg.nodes:
        row = node // side
        col = node % side
        dg.nodes[node]['y'] = bbox[1] + (row / side) * (bbox[3] - bbox[1])
        dg.nodes[node]['x'] = bbox[0] + (col / side) * (bbox[2] - bbox[0])
    return dg


def run_expanded_pipeline():
    """Run the full UrbanCityOS expanded pipeline with synthetic data."""

    set_seed(42)
    rng = np.random.RandomState(42)
    BBOX = (28.9, 40.9, 29.3, 41.2)
    DISASTER_LAT, DISASTER_LON = 29.1, 41.05

    logger.info("=" * 70)
    logger.info("  UrbanCityOS — Expanded Pipeline")
    logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # 1. INITIALISE ALL SYSTEMS
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[1/12] Initialising systems...")

    bus = EventBus()
    setup_default_subscriptions(bus)
    graph = build_synthetic_graph(196, seed=42)

    # Dashboard
    monitor = RealTimeMonitor(bus=bus, grid_rows=5, grid_cols=5, bbox=BBOX)
    scorer = ZoneCriticalityScorer()
    ranker = TriageRanker(scorer)
    damage_assessor = DamageAssessor()
    heatmap_gen = DamageHeatmapGenerator()
    infra_analyzer = InfrastructureImpactAnalyzer()

    # Routing
    emergency_router = EmergencyRouter(graph)
    aerial_nav = AerialNavAdvisor(graph)
    civilian_router = CivilianRerouteEngine(graph)
    dispatcher = EmergencyDispatcher(emergency_router)

    # Hospital
    hospital_network = HospitalNetwork.create_default_network(8, BBOX)
    hospital_feed = SimulatedHospitalFeed(hospital_network)
    patient_dist = PatientDistributor(hospital_network)

    # Infrastructure
    infra_monitor = InfrastructureGridMonitor.create_synthetic(15, BBOX)

    # Environment
    env_monitor = EnvironmentalMonitor(BBOX)

    # Public
    alert_composer = AlertComposer()
    escalation = EscalationProtocol()
    shelters = ShelterManager.create_default(10, BBOX)

    # Resources
    resources = ResourceManager.create_default(5, BBOX)

    # Analytics
    scenario_sim = ScenarioSimulator()

    # Coordination
    command = CommandCenter.create_default("EARTHQUAKE_RESPONSE")

    # Recovery
    recovery = RecoveryTracker("earthquake")

    # Command Data
    data_processor = CommandDataProcessor()
    report_gen = ReportGenerator()

    logger.info("All 12 subsystems initialised.")

    # ──────────────────────────────────────────────────────────────────
    # 2. SIMULATE DISASTER IMPACT
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[2/12] Simulating M6.5 earthquake...")

    # Generate synthetic damage
    damage_reports = damage_assessor.assess_synthetic(100, BBOX, seed=42)
    heatmap = heatmap_gen.generate(damage_reports, BBOX)
    infra_impact = infra_analyzer.analyze(damage_reports)
    logger.info(f"  Damage: {len(damage_reports)} cells assessed")
    logger.info(f"  Infrastructure impact: {infra_impact.get('systems_failed', 0)} systems failed")

    # Simulate infrastructure damage
    infra_monitor.simulate_damage(DISASTER_LAT, DISASTER_LON, radius_km=5, severity=0.6)

    # Generate environmental data
    env_monitor.generate_synthetic_data(DISASTER_LAT, DISASTER_LON)

    # Generate aerial clearance
    aerial_nav.generate_synthetic_clearance(damage_fraction=0.25)

    logger.info("  Disaster impact simulation complete.")

    # ──────────────────────────────────────────────────────────────────
    # 3. ZONE CRITICALITY & TRIAGE
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[3/12] Computing zone criticality...")

    # Simulate some incident activity in zones
    feed = SimulatedRealTimeFeed(monitor, rng_seed=42)
    for _ in range(10):
        feed.tick(1.0)

    assessments = ranker.rank_zones(monitor.zones)
    deployment_orders = ranker.get_deployment_order(assessments)

    p1 = sum(1 for a in assessments if a.criticality.value == "P1")
    p2 = sum(1 for a in assessments if a.criticality.value == "P2")
    logger.info(f"  Zones: {p1} CRITICAL (P1), {p2} SEVERE (P2)")
    logger.info(f"  Deployment orders: {len(deployment_orders)}")

    # ──────────────────────────────────────────────────────────────────
    # 4. EMERGENCY DISPATCH
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[4/12] Dispatching emergency vehicles...")

    # Create synthetic fleet
    vehicles = []
    for vtype in EmergencyVehicleType:
        for i in range(4):
            vehicles.append(EmergencyVehicle(
                vehicle_id=f"{vtype.value}_{i}",
                vehicle_type=vtype,
                nearest_node=rng.randint(0, len(graph.nodes)),
                speed_mps=15.0,
            ))
    dispatcher.register_vehicles(vehicles)

    # Build incidents from P1/P2 zones
    incidents = []
    for a in assessments[:5]:
        zone = next((z for z in monitor.zones if z.zone_id == a.zone_id), None)
        if zone:
            incidents.append({
                "id": a.zone_id,
                "type": "earthquake",
                "severity": a.composite_score,
                "nearest_node": rng.randint(0, len(graph.nodes)),
                "population_density": 5000,
            })

    hospitals_for_dispatch = [
        {"name": h.name, "node": h.nearest_node}
        for h in hospital_network.get_operational()
    ]
    dispatch_results = dispatcher.dispatch(incidents, hospitals_for_dispatch)
    logger.info(f"  Dispatched: {len(dispatch_results)} vehicles")

    # ──────────────────────────────────────────────────────────────────
    # 5. CIVILIAN REROUTING
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[5/12] Computing civilian reroutes...")

    # Close damaged roads
    impassable = list(aerial_nav.get_impassable_edges())[:20]
    civilian_router.close_roads(impassable)

    # Activate emergency corridor
    corridor_edges = list(graph.edges)[:10]
    civilian_router.activate_corridor(corridor_edges, ["ambulance", "fire_engine"])

    # Compute alternative routes for sample O-D pairs
    nodes = list(graph.nodes)
    civilian_routes = []
    for _ in range(3):
        o, d = rng.choice(nodes, 2, replace=False)
        routes = civilian_router.compute_alternative_routes(o, d, n_alternatives=2)
        civilian_routes.extend(routes)
    logger.info(f"  Civilian routes computed: {len(civilian_routes)}")

    traffic_advisories = civilian_router.generate_advisories()

    # ──────────────────────────────────────────────────────────────────
    # 6. HOSPITAL & PATIENT DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[6/12] Distributing patients to hospitals...")

    # Simulate disaster surge
    for _ in range(5):
        hospital_feed.tick(disaster_active=True, severity=0.7)

    patients = generate_synthetic_patients(50, DISASTER_LAT, DISASTER_LON, seed=42)
    assignments_med = patient_dist.assign_patients(patients)
    surge = patient_dist.predict_surge(intake_rate_per_hour=15)
    redistribution = patient_dist.check_redistribution_needed()

    logger.info(f"  Patients assigned: {len(assignments_med)}/50")
    logger.info(f"  Hospitals at risk of surge: {sum(1 for v in surge.values() if v.get('at_risk'))}")

    # ──────────────────────────────────────────────────────────────────
    # 7. INFRASTRUCTURE CASCADE PREDICTION
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[7/12] Predicting cascading failures...")

    cascades = infra_monitor.predict_cascades()
    infra_health = infra_monitor.network_health()
    logger.info(f"  Cascade scenarios: {len(cascades)}")
    for net, info in infra_health.items():
        logger.info(f"    {net}: {info['health_pct']}% operational, "
                     f"{info['failed']} failed")

    # ──────────────────────────────────────────────────────────────────
    # 8. ENVIRONMENTAL RISK
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[8/12] Computing environmental risks...")

    env_risks = env_monitor.compute_risk_overlay()
    env_constraints = env_monitor.get_evacuation_constraints()
    logger.info(f"  High-risk cells: {env_risks['high_risk_cells']}")
    logger.info(f"  Aftershock warning: {env_constraints.get('aftershock_warning', False)}")

    # ──────────────────────────────────────────────────────────────────
    # 9. PUBLIC ALERTS & SHELTERS
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[9/12] Issuing public alerts...")

    alert = alert_composer.compose_alert(
        "earthquake", AlertSeverity.EXTREME,
        area_name="Istanbul Central District",
        lat=DISASTER_LAT, lon=DISASTER_LON,
        extra_info={"magnitude": 6.5}
    )
    escalation.issue_alert(alert)

    # Shelter status
    for _ in range(20):
        nearest = shelters.find_nearest_available(
            DISASTER_LAT + rng.uniform(-0.03, 0.03),
            DISASTER_LON + rng.uniform(-0.03, 0.03), n=1)
        if nearest:
            shelters.check_in(nearest[0].shelter_id, count=rng.randint(1, 10))

    logger.info(f"  Alert issued: [{alert.severity.value}] {alert.title}")
    logger.info(f"  Shelter status: {shelters.summary()}")

    # ──────────────────────────────────────────────────────────────────
    # 10. RESOURCE & SUPPLY CHAIN
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[10/12] Managing resources...")

    demand = resources.forecast_demand(zone_casualties=150, zone_population=20000)
    # File resource requests
    for cat, qty in list(demand.items())[:4]:
        resources.request_resources("zone_2_2", cat, qty, urgency=0.8)
    fulfillment = resources.fulfill_requests()
    logger.info(f"  Resource fulfillments: {len(fulfillment)}")
    logger.info(f"  Mutual aid needed: {len(resources.check_mutual_aid_needed())}")

    # ──────────────────────────────────────────────────────────────────
    # 11. PREDICTIVE ANALYTICS
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[11/12] Running predictive scenarios...")

    earthquake_scenario = scenario_sim.run_scenario({
        "type": "earthquake", "lat": DISASTER_LAT, "lon": DISASTER_LON,
        "magnitude": 6.5, "population": 50000, "hours_since": 1.0,
    })
    fire_scenario = scenario_sim.run_scenario({
        "type": "fire", "lat": DISASTER_LAT + 0.02, "lon": DISASTER_LON,
        "wind_speed_mps": 12, "population": 10000, "damage_pct": 0.4,
    })
    comparison = scenario_sim.compare_scenarios([
        {"name": "M6.5 Earthquake", "type": "earthquake", "magnitude": 6.5, "population": 50000},
        {"name": "M7.0 Earthquake", "type": "earthquake", "magnitude": 7.0, "population": 50000},
    ])
    logger.info(f"  Earthquake scenario: "
                f"{earthquake_scenario['casualty_estimate']['total_casualties']} est. casualties")
    logger.info(f"  Aftershock safe? {earthquake_scenario['aftershock_forecast']['safe_for_entry']}")

    # ──────────────────────────────────────────────────────────────────
    # 12. COMMAND CENTER — AGGREGATE & REPORT
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n[12/12] Aggregating command center data...")

    # Create missions
    for order in deployment_orders[:3]:
        command.create_mission(
            "rescue", order["zone"],
            f"Respond to {order['criticality']} zone", priority=1
        )
    command.log_decision(
        "Prioritise zone_2_2 for search & rescue",
        "Highest composite criticality score (0.82), estimated 15 trapped",
        "Incident Commander"
    )

    # Generate situation snapshot
    snapshot = monitor.generate_snapshot()
    sitrep = command.generate_sitrep({"zones_affected": len(assessments)})

    # Aggregate ALL data into command display
    display_data = data_processor.process(
        situation_snapshot=snapshot,
        zone_assessments=assessments,
        deployment_orders=deployment_orders,
        damage_reports=damage_reports,
        damage_heatmap=heatmap,
        infrastructure_impact=infra_impact,
        emergency_dispatch=dispatch_results,
        fleet_status=dispatcher.fleet_status,
        traffic_advisories=traffic_advisories,
        civilian_routes=civilian_routes,
        hospital_status=hospital_network.network_summary(),
        patient_assignments=assignments_med,
        surge_predictions=surge,
        infrastructure_health=infra_health,
        cascade_predictions=cascades,
        environmental_risks=env_risks,
        evacuation_constraints=env_constraints,
        active_alerts=[alert],
        shelter_status=shelters.summary(),
        resource_status=resources.summary(),
        risk_forecasts=earthquake_scenario,
        scenario_analyses=[comparison],
        sitrep=sitrep,
        active_missions=command.get_active_missions(),
        command_summary=command.summary(),
        recovery_status=recovery.recovery_timeline(),
    )

    # Export
    json_path = data_processor.export_latest()
    metrics = data_processor.get_metrics_summary()

    # Generate reports
    sitrep_report = report_gen.generate_sitrep(display_data)
    exec_brief = report_gen.generate_executive_brief(display_data)
    text_sitrep = report_gen.generate_text_sitrep(display_data)

    report_gen.export_report(sitrep_report)
    report_gen.export_report(exec_brief)

    # ──────────────────────────────────────────────────────────────────
    # RESULTS SUMMARY
    # ──────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  UrbanCityOS — Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"  Data exported: {json_path}")
    logger.info(f"  Metrics: {json.dumps(metrics, indent=2, default=str)}")
    logger.info(f"\n{text_sitrep}")

    return {
        "display_data": display_data,
        "metrics": metrics,
        "sitrep": sitrep_report,
        "text_sitrep": text_sitrep,
    }


if __name__ == "__main__":
    results = run_expanded_pipeline()
