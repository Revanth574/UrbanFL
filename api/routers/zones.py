"""Zones router — criticality assessments, triage ranking, and full enrichment."""

import random
import math
from fastapi import APIRouter, HTTPException
from api.state import app_state

router = APIRouter()


def _compute_zone_enrichment(zone, infra_health, hospital_summary, env_overlay, damage_stats):
    """
    Build the full dimension_scores dict and derived metrics for a zone.
    Cross-references all backend modules to produce realistic scores.
    """
    threat = getattr(zone, 'threat_level', 0.0)
    population = getattr(zone, 'population', 50000)
    pop_normalized = min(1.0, population / 500000.0)

    # Infrastructure loss — from real infra grid health
    avg_infra = 50.0
    if infra_health:
        pcts = [v.get("health_pct", 50) for v in infra_health.values()]
        avg_infra = sum(pcts) / max(len(pcts), 1)
    infra_loss = max(0.0, min(1.0, (100 - avg_infra) / 100.0))

    # Environmental risk — from env_monitor overlay
    env_risk = 0.0
    if env_overlay:
        env_risk = min(1.0, env_overlay.get("max_risk", 0.0))

    # Structural damage — from damage assessor stats
    total_buildings = max(1, sum(damage_stats.values())) if damage_stats else 1
    destroyed = damage_stats.get("destroyed", 0)
    major = damage_stats.get("major_damage", 0)
    struct_damage = min(1.0, (destroyed * 1.0 + major * 0.6) / max(total_buildings * 0.3, 1))

    # Medical proximity — inverse of hospital availability
    hospital_util = 0.5
    if hospital_summary:
        total_beds = hospital_summary.get("total_beds", 100)
        occupied = hospital_summary.get("total_occupied", 50)
        hospital_util = min(1.0, occupied / max(total_beds, 1))

    # Per-zone jitter to differentiate sectors (deterministic from zone_id hash)
    seed = hash(getattr(zone, 'zone_id', '')) % 1000
    rng = random.Random(seed)
    jitter = lambda base, spread=0.15: max(0.0, min(1.0, base + rng.uniform(-spread, spread)))

    dims = {
        "casualty_density": jitter(threat * 0.8 + struct_damage * 0.2),
        "structural_damage": jitter(struct_damage),
        "population_density": jitter(pop_normalized),
        "infrastructure_loss": jitter(infra_loss),
        "accessibility": jitter(max(0.0, 1.0 - infra_loss * 0.7 - threat * 0.3)),
        "medical_proximity": jitter(hospital_util),
        "secondary_hazard": jitter(threat * 0.5 + env_risk * 0.3),
        "time_without_contact": jitter(threat * 0.4),
        "environmental_risk": jitter(env_risk),
    }

    # Composite score — weighted average
    weights = {
        "casualty_density": 1.0, "structural_damage": 1.0,
        "population_density": 0.8, "infrastructure_loss": 0.7,
        "accessibility": 0.5, "medical_proximity": 0.6,
        "secondary_hazard": 0.4, "time_without_contact": 0.3,
        "environmental_risk": 0.3,
    }
    w_total = sum(weights.values())
    composite = sum(dims[k] * weights.get(k, 0.5) for k in dims) / w_total

    # Casualties & trapped — derived from composite + population
    base_casualties = int(composite * pop_normalized * 80 * rng.uniform(0.5, 1.5))
    base_trapped = int(composite * pop_normalized * 40 * rng.uniform(0.3, 1.2))

    # Cascading risks
    cascading = []
    if dims["infrastructure_loss"] > 0.5:
        cascading.append("Power grid failure may cascade — hospital backup generators at risk within 4-8hr")
    if dims["structural_damage"] > 0.7:
        cascading.append("Secondary collapse risk — aftershock P(M≥5.0) elevated")
    if dims["environmental_risk"] > 0.5:
        cascading.append("Gas line rupture risk — evacuate 500m perimeter before entry")
    if dims["medical_proximity"] > 0.7:
        cascading.append("Nearest hospitals at capacity — redirect casualties to field stations")

    # Resource allocation suggestion
    ambulances = max(1, int(composite * 5))
    fire_engines = max(1, int(dims["structural_damage"] * 4))
    police = max(1, int(dims["casualty_density"] * 3))
    search_rescue = max(1, int(base_trapped / 15))
    resources = {
        "ambulances": ambulances,
        "fire_engines": fire_engines,
        "police": police,
        "search_rescue": search_rescue,
    }

    # Accessibility percentage
    accessibility = max(0.0, min(1.0, 1.0 - dims["accessibility"]))  # invert so higher = more accessible

    return {
        "dimension_scores": dims,
        "composite_score": round(composite, 3),
        "estimated_casualties": base_casualties,
        "estimated_trapped": base_trapped,
        "population": population,
        "accessibility": round(accessibility, 3),
        "cascading_risks": cascading,
        "resources": resources,
    }


@router.get("/zones")
async def get_zones():
    """All zones with full enrichment — criticality, dimensions, casualties, resources."""
    monitor = app_state.monitor
    scorer = app_state.triage_ranker
    if not monitor or not scorer:
        return {"zones": [], "error": "Modules not initialised"}

    # Gather cross-module data for enrichment
    infra_health = app_state.infra_grid.network_health() if app_state.infra_grid else {}
    hospital_summary = app_state.hospital_network.network_summary() if app_state.hospital_network else {}
    env_overlay = app_state.env_monitor.compute_risk_overlay() if app_state.env_monitor else {}
    damage_stats = app_state.damage_assessor.stats if app_state.damage_assessor else {}

    assessments = scorer.rank_zones(monitor.zones)

    zones_out = []
    for a in assessments:
        zmatch = next((z for z in monitor.zones if z.zone_id == a.zone_id), None)
        if not zmatch:
            continue

        enrichment = _compute_zone_enrichment(zmatch, infra_health, hospital_summary, env_overlay, damage_stats)

        zones_out.append({
            "zone_id": a.zone_id,
            "criticality": a.criticality.value,
            "lat": zmatch.center_lat,
            "lon": zmatch.center_lon,
            **enrichment,
        })

    # Sort by composite descending (most critical first)
    zones_out.sort(key=lambda z: z["composite_score"], reverse=True)

    return {
        "zones": zones_out,
        "summary": {
            "total": len(zones_out),
            "P1": sum(1 for a in assessments if a.criticality.value == "P1"),
            "P2": sum(1 for a in assessments if a.criticality.value == "P2"),
            "P3": sum(1 for a in assessments if a.criticality.value == "P3"),
            "P4": sum(1 for a in assessments if a.criticality.value == "P4"),
            "P5": sum(1 for a in assessments if a.criticality.value == "P5"),
        },
    }


@router.get("/zones/triage")
async def get_triage():
    """Prioritised deployment orders from TriageRanker."""
    monitor = app_state.monitor
    ranker = app_state.triage_ranker
    if not monitor or not ranker:
        return {"orders": []}

    assessments = ranker.rank_zones(monitor.zones)
    orders = ranker.get_deployment_order(assessments)
    return {"orders": orders, "total_actionable": len(orders)}


@router.get("/zones/{zone_id}")
async def get_zone_detail(zone_id: str):
    """Detailed assessment for a single zone."""
    monitor = app_state.monitor
    scorer = app_state.zone_scorer
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialised")

    zone = next((z for z in monitor.zones if z.zone_id == zone_id), None)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    # Build enrichment
    infra_health = app_state.infra_grid.network_health() if app_state.infra_grid else {}
    hospital_summary = app_state.hospital_network.network_summary() if app_state.hospital_network else {}
    env_overlay = app_state.env_monitor.compute_risk_overlay() if app_state.env_monitor else {}
    damage_stats = app_state.damage_assessor.stats if app_state.damage_assessor else {}

    enrichment = _compute_zone_enrichment(zone, infra_health, hospital_summary, env_overlay, damage_stats)

    # Assessment
    assessment_data = {"zone_id": zone.zone_id}
    if scorer:
        try:
            a = scorer.score_zone(zone)
            assessment_data = a.to_dict()
        except Exception:
            pass

    return {
        "assessment": assessment_data,
        **enrichment,
        "zone_geometry": {
            "center_lat": zone.center_lat,
            "center_lon": zone.center_lon,
        },
    }
