"""
UrbanCityOS — Integration Tests for All 12 Feature Groups
Quick smoke tests that import, instantiate, and exercise each subsystem.
Run: python -m experiments.test_expanded_features
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import networkx as nx

PASS = 0
FAIL = 0
BBOX = (28.9, 40.9, 29.3, 41.2)


def _build_graph(n: int = 100) -> nx.DiGraph:
    rng = np.random.RandomState(42)
    side = int(np.sqrt(n))
    g = nx.grid_2d_graph(side, side)
    dg = nx.DiGraph()
    for u, v in g.edges:
        uid, vid = u[0] * side + u[1], v[0] * side + v[1]
        dg.add_edge(uid, vid, length=rng.uniform(100, 500), speed_kph=50, lanes=2)
        dg.add_edge(vid, uid, length=rng.uniform(100, 500), speed_kph=50, lanes=2)
    for node in dg.nodes:
        dg.nodes[node]['y'] = BBOX[1] + (node // side) / side * (BBOX[3] - BBOX[1])
        dg.nodes[node]['x'] = BBOX[0] + (node % side) / side * (BBOX[2] - BBOX[0])
    return dg


def test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ─── Group 1: Dashboard & Situational Awareness ────────────────────────


def test_group1():
    from src.event_bus import EventBus, setup_default_subscriptions
    from src.dashboard.situation_monitor import RealTimeMonitor, SimulatedRealTimeFeed
    from src.dashboard.zone_criticality import ZoneCriticalityScorer, TriageRanker
    from src.dashboard.damage_assessment import (
        DamageAssessor, DamageHeatmapGenerator, InfrastructureImpactAnalyzer
    )

    bus = EventBus()
    setup_default_subscriptions(bus)
    monitor = RealTimeMonitor(bus=bus, grid_rows=3, grid_cols=3, bbox=BBOX)
    feed = SimulatedRealTimeFeed(monitor, rng_seed=42)
    for _ in range(5):
        feed.tick(1.0)
    assert len(monitor.zones) > 0, "No zones created"

    scorer = ZoneCriticalityScorer()
    ranker = TriageRanker(scorer)
    assessments = ranker.rank_zones(monitor.zones)
    assert len(assessments) > 0, "No assessments"
    orders = ranker.get_deployment_order(assessments)
    assert isinstance(orders, list)

    assessor = DamageAssessor()
    reports = assessor.assess_synthetic(20, BBOX, seed=42)
    heatmap = DamageHeatmapGenerator().generate(reports, BBOX)
    impact = InfrastructureImpactAnalyzer().analyze(reports)
    assert isinstance(heatmap, dict)


# ─── Group 2: Emergency Routing ────────────────────────────────────────


def test_group2():
    from src.silo3.emergency_router import (
        EmergencyRouter, EmergencyDispatcher, EmergencyVehicle,
        EmergencyVehicleType
    )
    from src.silo3.aerial_nav_advisor import AerialNavAdvisor

    g = _build_graph()
    router = EmergencyRouter(g)
    aerial = AerialNavAdvisor(g)
    aerial.generate_synthetic_clearance(damage_fraction=0.2)

    dispatcher = EmergencyDispatcher(router)
    vehicles = [
        EmergencyVehicle(f"v_{i}", EmergencyVehicleType.AMBULANCE, i * 5, 15.0)
        for i in range(3)
    ]
    dispatcher.register_vehicles(vehicles)
    dispatch = dispatcher.dispatch(
        [{"id": "inc_0", "type": "fire", "severity": 0.9, "nearest_node": 10,
          "population_density": 3000}],
        [{"name": "H1", "node": 50}]
    )
    assert len(dispatch) > 0, "No dispatch results"


# ─── Group 3: Civilian Rerouting ───────────────────────────────────────


def test_group3():
    from src.silo3.civilian_rerouter import CivilianRerouteEngine

    g = _build_graph()
    engine = CivilianRerouteEngine(g)
    edges = list(g.edges)[:5]
    engine.close_roads(edges)
    engine.activate_corridor(edges[:2], ["ambulance"])
    nodes = list(g.nodes)
    routes = engine.compute_alternative_routes(nodes[0], nodes[-1], n_alternatives=2)
    advisories = engine.generate_advisories()
    assert isinstance(advisories, list)


# ─── Group 4: Hospital & Medical ──────────────────────────────────────


def test_group4():
    from src.medical.hospital_network import HospitalNetwork, SimulatedHospitalFeed
    from src.medical.patient_distributor import PatientDistributor, generate_synthetic_patients

    network = HospitalNetwork.create_default_network(5, BBOX)
    feed = SimulatedHospitalFeed(network)
    feed.tick(disaster_active=True, severity=0.5)
    patients = generate_synthetic_patients(10, 29.1, 41.05, seed=42)
    dist = PatientDistributor(network)
    assignments = dist.assign_patients(patients)
    surge = dist.predict_surge(intake_rate_per_hour=10)
    assert isinstance(assignments, list)


# ─── Group 5: Infrastructure Grid ─────────────────────────────────────


def test_group5():
    from src.infrastructure.grid_monitor import InfrastructureGridMonitor

    monitor = InfrastructureGridMonitor.create_synthetic(10, BBOX)
    monitor.simulate_damage(29.1, 41.05, radius_km=3, severity=0.5)
    cascades = monitor.predict_cascades()
    health = monitor.network_health()
    assert isinstance(health, dict)
    assert len(health) > 0


# ─── Group 6: Environmental Monitoring ─────────────────────────────────


def test_group6():
    from src.environment.env_monitor import EnvironmentalMonitor

    monitor = EnvironmentalMonitor(BBOX)
    monitor.generate_synthetic_data(29.1, 41.05)
    risks = monitor.compute_risk_overlay()
    constraints = monitor.get_evacuation_constraints()
    assert "high_risk_cells" in risks


# ─── Group 7: Public Alert & Citizen Engagement ────────────────────────


def test_group7():
    from src.public.alert_system import AlertComposer, EscalationProtocol, AlertSeverity
    from src.public.citizen_platform import ShelterManager

    composer = AlertComposer()
    alert = composer.compose_alert(
        "earthquake", AlertSeverity.EXTREME,
        area_name="Test District", lat=29.1, lon=41.05,
        extra_info={"magnitude": 6.5}
    )
    escalation = EscalationProtocol()
    escalation.issue_alert(alert)
    assert alert.title

    shelters = ShelterManager.create_default(5, BBOX)
    nearest = shelters.find_nearest_available(29.1, 41.05, n=1)
    if nearest:
        shelters.check_in(nearest[0].shelter_id, count=5)
    summary = shelters.summary()
    assert isinstance(summary, dict)


# ─── Group 8: Resource Management ──────────────────────────────────────


def test_group8():
    from src.resources.resource_manager import ResourceManager

    rm = ResourceManager.create_default(3, BBOX)
    demand = rm.forecast_demand(zone_casualties=50, zone_population=10000)
    for cat, qty in list(demand.items())[:2]:
        rm.request_resources("zone_1_1", cat, qty, urgency=0.7)
    fulfillment = rm.fulfill_requests()
    mutual_aid = rm.check_mutual_aid_needed()
    assert isinstance(fulfillment, list)


# ─── Group 9: Predictive Analytics ─────────────────────────────────────


def test_group9():
    from src.analytics.risk_forecaster import ScenarioSimulator

    sim = ScenarioSimulator()
    result = sim.run_scenario({
        "type": "earthquake", "lat": 29.1, "lon": 41.05,
        "magnitude": 6.5, "population": 50000, "hours_since": 2.0
    })
    assert "casualty_estimate" in result
    comparison = sim.compare_scenarios([
        {"name": "M6.5", "type": "earthquake", "magnitude": 6.5, "population": 30000},
        {"name": "M7.0", "type": "earthquake", "magnitude": 7.0, "population": 30000},
    ])
    assert isinstance(comparison, dict)


# ─── Group 10: Command Center ──────────────────────────────────────────


def test_group10():
    from src.coordination.command_center import CommandCenter

    cc = CommandCenter.create_default("TEST_INCIDENT")
    cc.create_mission("rescue", "zone_1_1", "Test rescue mission", priority=1)
    cc.log_decision("Test decision", "Test rationale", "Incident Commander")
    sitrep = cc.generate_sitrep({"zones_affected": 5})
    summary = cc.summary()
    assert isinstance(sitrep, dict)
    assert isinstance(summary, dict)


# ─── Group 11: Recovery Tracker ────────────────────────────────────────


def test_group11():
    from src.recovery.recovery_tracker import RecoveryTracker, RecoveryPhase

    tracker = RecoveryTracker("earthquake")
    timeline = tracker.recovery_timeline()
    assert isinstance(timeline, dict)


# ─── Group 12: Command Data Processing ────────────────────────────────


def test_group12():
    from src.command.data_processor import CommandDataProcessor
    from src.command.report_generator import ReportGenerator

    processor = CommandDataProcessor()
    report_gen = ReportGenerator()

    # Minimal display data
    display = processor.process(
        situation_snapshot={"timestamp": time.time(), "zones": []},
        zone_assessments=[],
        deployment_orders=[],
        damage_reports=[],
        damage_heatmap={},
        infrastructure_impact={},
        emergency_dispatch=[],
        fleet_status={},
        traffic_advisories=[],
        civilian_routes=[],
        hospital_status={},
        patient_assignments=[],
        surge_predictions={},
        infrastructure_health={},
        cascade_predictions=[],
        environmental_risks={},
        evacuation_constraints={},
        active_alerts=[],
        shelter_status={},
        resource_status={},
        risk_forecasts={},
        scenario_analyses=[],
        sitrep={},
        active_missions=[],
        command_summary={},
        recovery_status={},
    )
    assert display is not None

    sitrep = report_gen.generate_sitrep(display)
    brief = report_gen.generate_executive_brief(display)
    text = report_gen.generate_text_sitrep(display)
    assert isinstance(text, str)


# ─── Event Bus ─────────────────────────────────────────────────────────


def test_event_bus():
    from src.event_bus import EventBus, Event, EventType, setup_default_subscriptions

    bus = EventBus(max_queue_size=100)
    setup_default_subscriptions(bus)

    bus.publish(Event(
        priority=0.5, timestamp=time.time(),
        event_type=EventType.ALERT_VERIFIED,
        source_silo=0,
        data={"lat": 29.1, "lon": 41.05, "type": "earthquake"}
    ))
    stats = bus.stats()
    assert stats["total_events"] >= 1


# ─── Main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 60)
    print("  UrbanCityOS — Integration Tests (12 Feature Groups)")
    print("=" * 60)

    groups = [
        ("Group  1: Dashboard & Situational Awareness", test_group1),
        ("Group  2: Emergency Routing", test_group2),
        ("Group  3: Civilian Rerouting", test_group3),
        ("Group  4: Hospital & Medical", test_group4),
        ("Group  5: Infrastructure Grid", test_group5),
        ("Group  6: Environmental Monitoring", test_group6),
        ("Group  7: Public Alert & Citizen", test_group7),
        ("Group  8: Resource Management", test_group8),
        ("Group  9: Predictive Analytics", test_group9),
        ("Group 10: Command Center", test_group10),
        ("Group 11: Recovery Tracker", test_group11),
        ("Group 12: Command Data Processing", test_group12),
        ("Event Bus", test_event_bus),
    ]

    for name, fn in groups:
        print(f"\n{name}")
        test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if FAIL == 0 else 1)
