"""
UrbanCityOS — New Module Integration Tests
Tests for: RAG Retrieval, Safety Guardrails, Agentic Orchestrator,
           Multi-Hazard Adapter, and Concurrent Hazard Events.

Run: python -m experiments.test_new_modules
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

PASS = 0
FAIL = 0


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


# ─── Test 1: RAG Historical Incident Store ─────────────────────────

def test_rag_store():
    from src.silo1.credibility_scorer import HistoricalIncidentStore

    store = HistoricalIncidentStore()

    # Add 20 historical incidents
    confirmed_types = {
        "earthquake": "Major earthquake shaking felt in downtown area",
        "flood": "Severe flooding in residential zone, roads submerged",
        "fire": "Large fire erupting in industrial area",
        "explosion": "Gas explosion reported at factory",
        "hazmat": "Chemical spill at port area storage facility",
    }

    for i in range(20):
        incident_type = list(confirmed_types.keys())[i % len(confirmed_types)]
        text = f"{confirmed_types[incident_type]} incident_{i}"
        was_confirmed = i % 3 != 0  # 2/3 confirmed
        store.add_incident(
            text=text,
            incident_type=incident_type,
            was_confirmed=was_confirmed,
            severity=np.random.uniform(0.3, 0.9),
            timestamp=time.time() - i * 3600,
        )

    assert store.size == 20, f"Expected 20 incidents, got {store.size}"

    # Query similar incidents
    results = store.query_similar("earthquake shaking in downtown", k=5)
    assert len(results) > 0, "No similar incidents found"
    assert "similarity" in results[0], "Missing similarity score"
    assert results[0]["similarity"] > 0, "Similarity should be positive"

    # Query corroboration score
    score, similar = store.query_corroboration_score(
        "earthquake shaking reported", incident_type="earthquake", k=5
    )
    assert 0 <= score <= 1, f"Score out of range: {score}"
    assert len(similar) > 0, "No similar incidents in corroboration query"

    # Test with empty store
    empty_store = HistoricalIncidentStore()
    empty_score, empty_similar = empty_store.query_corroboration_score("test")
    assert empty_score == 0.5, "Empty store should return 0.5 (neutral)"
    assert len(empty_similar) == 0

    print(f"    RAG Store: {store.size} incidents, FAISS={store.uses_faiss}, "
          f"ST={store.uses_sentence_transformers}")
    print(f"    Corroboration score: {score:.3f} (from {len(similar)} similar)")


# ─── Test 2: 7D Credibility Scorer (with RAG) ──────────────────────

def test_7d_credibility():
    from src.silo1.credibility_scorer import (
        CredibilityScorer, generate_synthetic_reports, DEFAULT_DIM_WEIGHTS
    )

    # Verify 7 dimensions
    assert len(DEFAULT_DIM_WEIGHTS) == 7, (
        f"Expected 7 dimensions, got {len(DEFAULT_DIM_WEIGHTS)}: {list(DEFAULT_DIM_WEIGHTS.keys())}"
    )
    assert "historical_corroboration" in DEFAULT_DIM_WEIGHTS, "Missing RAG dimension"

    # Verify weights sum to ~1.0
    weight_sum = sum(DEFAULT_DIM_WEIGHTS.values())
    assert 0.99 <= weight_sum <= 1.01, f"Weights sum to {weight_sum}, expected ~1.0"

    # Create scorer and score reports
    scorer = CredibilityScorer(enable_rag=True)

    # Add some historical data to the RAG store
    if scorer.historical_store is not None:
        for i in range(10):
            scorer.historical_store.add_incident(
                text=f"earthquake damage reported in zone_{i}",
                incident_type="earthquake",
                was_confirmed=True,
                severity=0.8,
            )

    reports = generate_synthetic_reports(
        scenario_type="earthquake", n_reports=10,
        center_lat=29.1, center_lon=41.05,
    )

    scores = scorer.score_batch(reports)
    assert len(scores) == 10, f"Expected 10 scores, got {len(scores)}"
    assert all(0 <= s <= 1 for s in scores), "Scores out of range"

    # Check that dim_scores have 7 dimensions
    dim_scores = scorer._recent_dim_scores
    if dim_scores:
        assert len(dim_scores[-1]) == 7, (
            f"Expected 7 dim scores, got {len(dim_scores[-1])}: {list(dim_scores[-1].keys())}"
        )

    mean_score = np.mean(scores)
    print(f"    7D scores: mean={mean_score:.3f}, min={min(scores):.3f}, max={max(scores):.3f}")
    print(f"    Dimensions: {list(dim_scores[-1].keys()) if dim_scores else 'N/A'}")


# ─── Test 3: Safety Guardrail Layer ────────────────────────────────

def test_guardrails():
    from src.federated.guardrails import (
        SafetyGuardrailLayer, GuardrailVerdict, GuardrailResult
    )

    guardrail = SafetyGuardrailLayer()

    # Test 1: SAFE recommendation
    safe_rec = {
        "id": "rec_safe",
        "route": [1, 2, 3, 4, 5],
        "eta_seconds": 300,
        "distance_m": 5000,
        "credibility_score": 0.6,
        "severity": 0.5,
    }
    result = guardrail.check(safe_rec)
    assert result.overall_verdict == GuardrailVerdict.SAFE, (
        f"Expected SAFE, got {result.overall_verdict}"
    )

    # Test 2: FLAG - human escalation (high cred + high severity)
    flag_rec = {
        "id": "rec_flag",
        "credibility_score": 0.90,
        "severity": 0.85,
    }
    result = guardrail.check(flag_rec)
    assert result.overall_verdict == GuardrailVerdict.FLAG, (
        f"Expected FLAG, got {result.overall_verdict}"
    )

    # Test 3: BLOCK - impossible speed
    block_rec = {
        "id": "rec_block",
        "route": [1, 2, 3, 4, 5],
        "eta_seconds": 10,
        "distance_m": 100000,  # 100km in 10 seconds = 36000 km/h
    }
    result = guardrail.check(block_rec)
    assert result.overall_verdict == GuardrailVerdict.BLOCK, (
        f"Expected BLOCK, got {result.overall_verdict}"
    )

    # Test 4: FLAG - NLP/IoT inconsistency
    inconsistent_rec = {
        "id": "rec_inconsistent",
        "incident_type_nlp": "earthquake",
        "iot_anomaly_type": "water_level",
    }
    result = guardrail.check(inconsistent_rec)
    has_flag = any(
        cr.check_name == "nlp_iot_consistency" and cr.verdict == GuardrailVerdict.FLAG
        for cr in result.check_results
    )
    assert has_flag, "Expected NLP↔IoT consistency FLAG"

    # Test 5: NLP/IoT consistent
    consistent_rec = {
        "id": "rec_consistent",
        "incident_type_nlp": "earthquake",
        "iot_anomaly_type": "seismic_vibration",
    }
    result = guardrail.check(consistent_rec)
    consistency_check = next(
        cr for cr in result.check_results if cr.check_name == "nlp_iot_consistency"
    )
    assert consistency_check.verdict == GuardrailVerdict.SAFE, (
        f"Expected SAFE for consistent NLP/IoT, got {consistency_check.verdict}"
    )

    # Test batch
    results = guardrail.check_batch([safe_rec, flag_rec, block_rec])
    assert len(results) == 3

    summary = guardrail.get_audit_summary()
    print(f"    Guardrail: {summary['total_checked']} checked, "
          f"{summary['safe']} safe, {summary['flagged']} flagged, "
          f"{summary['blocked']} blocked, avg={summary['avg_processing_ms']:.2f}ms")


# ─── Test 4: Agentic Orchestrator ──────────────────────────────────

def test_orchestrator():
    from src.orchestrator import (
        OrchestratorEngine, AgentGraph, Agent, AgentStatus,
        AlertIngestionAgent, CredibilityAgent, GuardrailAgent, DispatchAgent,
    )

    # Create default pipeline
    engine = OrchestratorEngine.create_default()

    # Run with a synthetic earthquake scenario
    result = engine.run(
        scenario={
            "type": "earthquake",
            "lat": 29.1,
            "lon": 41.05,
            "n_reports": 20,
            "incident_type": "earthquake",
            "center_lat": 29.1,
            "center_lon": 41.05,
        }
    )

    # Verify all agents ran
    assert len(result.agent_results) >= 5, (
        f"Expected >=5 agents, got {len(result.agent_results)}"
    )

    # Verify critical agents succeeded
    for critical in ["alert_ingestion", "credibility"]:
        if critical in result.agent_results:
            assert result.agent_results[critical].is_success, (
                f"Critical agent '{critical}' failed: "
                f"{result.agent_results[critical].error}"
            )

    # Check latency benchmarks
    benchmarks = engine.get_latency_benchmarks()
    assert len(benchmarks) > 0, "No latency benchmarks recorded"

    print(f"    Orchestrator: {len(result.agent_results)} agents, "
          f"total={result.total_latency_ms:.1f}ms, "
          f"speedup={result.parallel_speedup:.2f}x")
    print(f"    SLA compliant: {result.sla_compliant}")

    # Print per-agent latency
    for name, ar in sorted(result.agent_results.items()):
        status = "+" if ar.is_success else "x"
        print(f"      {status} {name:25s} {ar.latency_ms:7.1f}ms")


# ─── Test 5: Multi-Hazard Concurrent Scenarios ─────────────────────

def test_multi_hazard():
    from src.utils.core import (
        DisasterScenario, ConcurrentHazardEvent, HazardType,
        _compute_hazard_interactions,
    )

    # Test individual scenario types
    eq = DisasterScenario.earthquake()
    assert eq.incident_type == "earthquake"

    explosion = DisasterScenario.explosion()
    assert explosion.severity == 0.95

    hazmat = DisasterScenario.hazmat()
    assert hazmat.incident_type == "hazmat"

    tsunami = DisasterScenario.tsunami()
    assert tsunami.radius_m == 10000

    landslide = DisasterScenario.landslide()
    assert landslide.name == "landslide"

    # Test HazardType enum
    assert len(HazardType.ALL) == 10, f"Expected 10 hazard types, got {len(HazardType.ALL)}"
    assert "earthquake" in HazardType.ALL
    assert "tsunami" in HazardType.ALL

    # Test multi-hazard event (default: earthquake + fire + building_collapse)
    multi = DisasterScenario.multi_hazard()
    assert multi.n_hazards == 3, f"Expected 3 hazards, got {multi.n_hazards}"
    assert multi.compound_severity > 0, "Compound severity should be > 0"
    assert len(multi.interaction_effects) > 0, "Should have interaction effects"

    # Test interaction computation
    interactions = _compute_hazard_interactions([eq, explosion])
    assert len(interactions) == 1, f"Expected 1 interaction, got {len(interactions)}"

    # Test custom multi-hazard
    custom = DisasterScenario.multi_hazard([
        DisasterScenario.earthquake(),
        DisasterScenario.flood(),
        DisasterScenario.hazmat(),
        DisasterScenario.explosion(),
    ])
    assert custom.n_hazards == 4
    assert custom.total_estimated_casualties > 0
    priority = custom.get_priority_ordered()
    assert priority[0].severity >= priority[-1].severity

    print(f"    Multi-hazard: {multi.n_hazards} hazards, "
          f"compound_severity={multi.compound_severity:.2f}")
    print(f"    Interaction effects: {multi.interaction_effects}")
    print(f"    Custom: {custom.n_hazards} hazards, "
          f"casualties={custom.total_estimated_casualties}")


# ─── Test 6: Multi-Hazard Data Adapter ─────────────────────────────

def test_data_adapter():
    from src.data.multi_hazard_adapter import MultiHazardDataAdapter

    adapter = MultiHazardDataAdapter()
    summary = adapter.get_training_summary()

    assert "datasets_available" in summary, "Missing datasets_available"
    assert "datasets_missing" in summary, "Missing datasets_missing"
    assert summary["total_available"] >= 0

    # Test environmental context
    env = adapter.get_environmental_context()
    assert "environmental_risk_score" in env
    assert 0 <= env["environmental_risk_score"] <= 1

    print(f"    Data adapter: {summary['total_available']} available, "
          f"{summary['total_missing']} missing")
    print(f"    Environmental risk: {env['environmental_risk_score']:.3f}")

    for ds in summary["datasets_available"][:5]:
        print(f"      + {ds}")
    for ds in summary["datasets_missing"][:5]:
        print(f"      - {ds}")


# ─── Test 7: End-to-End Orchestrator + Guardrails ─────────────────

def test_e2e_pipeline():
    """Full end-to-end: orchestrator pipeline with guardrails active."""
    from src.orchestrator import OrchestratorEngine
    from src.federated.guardrails import SafetyGuardrailLayer

    guardrail = SafetyGuardrailLayer(
        escalation_credibility_threshold=0.85,
        escalation_severity_threshold=0.80,
    )

    engine = OrchestratorEngine.create_default(guardrail_layer=guardrail)

    result = engine.run(
        scenario={
            "type": "earthquake",
            "incident_type": "earthquake",
            "center_lat": 29.1,
            "center_lon": 41.05,
            "n_reports": 15,
        }
    )

    # Verify guardrail agent ran
    assert "guardrail" in result.agent_results, "Guardrail agent missing"
    if result.agent_results["guardrail"].is_success:
        gs = result.agent_results["guardrail"].output.get("guardrail_summary", {})
        print(f"    E2E: Guardrail -> {gs.get('safe', 0)} safe, "
              f"{gs.get('flagged', 0)} flagged, {gs.get('blocked', 0)} blocked")

    print(f"    E2E total: {result.total_latency_ms:.1f}ms, "
          f"success={result.is_success}")


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  UrbanCityOS — New Module Tests")
    print("=" * 60)

    groups = [
        ("RAG Historical Incident Store", test_rag_store),
        ("7D Credibility Scorer", test_7d_credibility),
        ("Safety Guardrail Layer", test_guardrails),
        ("Agentic Orchestrator", test_orchestrator),
        ("Multi-Hazard Concurrent Scenarios", test_multi_hazard),
        ("Multi-Hazard Data Adapter", test_data_adapter),
        ("E2E Pipeline + Guardrails", test_e2e_pipeline),
    ]

    for name, fn in groups:
        print(f"\n{name}")
        test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if FAIL == 0 else 1)
