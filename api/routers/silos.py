"""Silo-specific endpoints — NLP feed, Vision assessment, Infra status."""

import time
import numpy as np
from fastapi import APIRouter
from api.state import app_state

router = APIRouter()


# ── Silo Status ──────────────────────────────────────────────────────

@router.get("/silos/status")
async def silos_status():
    """Health/status for all 3 silos."""
    snap = app_state.get_latest_snapshot() or {}
    infra_h = app_state.infra_grid.network_health() if app_state.infra_grid else {}
    pcts = [v.get("health_pct", 50) for v in infra_h.values()]
    infra_pct = round(sum(pcts) / max(len(pcts), 1), 1) if pcts else 50

    return {
        "silos": [
            {
                "id": 1, "name": "NLP — Crisis Classification",
                "model": "XLM-RoBERTa",
                "health_pct": min(98, 70 + (snap.get("alerts", {}).get("mean_credibility", 0.7) * 40)),
                "status": "ONLINE",
                "last_update": time.time(),
                "metrics": snap.get("alerts", {}),
            },
            {
                "id": 2, "name": "Vision — Damage Detection",
                "model": "SiameseChangeDetector",
                "health_pct": min(95, 50 + snap.get("aerial", {}).get("coverage_pct", 50)),
                "status": "PROCESSING",
                "last_update": time.time(),
                "metrics": snap.get("aerial", {}),
            },
            {
                "id": 3, "name": "Infrastructure — Geospatial Routing",
                "model": "OSRM + BPR",
                "health_pct": infra_pct,
                "status": "DEGRADED" if infra_pct < 60 else "ONLINE",
                "last_update": time.time(),
                "metrics": snap.get("traffic", {}),
            },
        ],
        "health_history": app_state.get_silo_health_history(),
    }


# ── Silo 1: NLP Feed ────────────────────────────────────────────────

@router.get("/silos/nlp/feed")
async def nlp_feed(limit: int = 30):
    """Real-time NLP alert feed with new reports pulled from the event queue."""
    reports = list(app_state._recent_reports)[:limit] if hasattr(app_state, "_recent_reports") else []

    stats = {
        "total": len(reports),
        "critical": sum(1 for r in reports if r.get("severity") == "CRITICAL"),
        "verified": sum(1 for r in reports if r.get("verified")),
        "avg_credibility": round(sum(r.get("credibility", 0) for r in reports) / max(len(reports), 1), 3),
    }
    return {"reports": reports, "stats": stats}


# ── Silo 2: Vision Assessment ────────────────────────────────────────

@router.get("/silos/vision/assessment")
async def vision_assessment():
    """Damage assessment from the change detection pipeline with per-stage playback."""
    assessor = app_state.damage_assessor
    if not assessor:
        return {"error": "Damage assessor not initialised"}

    active_region = getattr(app_state, "_active_disaster_name", "UNKNOWN")
    
    # Filter reports for the active region
    all_reports = assessor.all_reports
    active_reports = [r for r in all_reports if r.disaster == active_region]
    
    # If active region is empty, fall back to all (for startup period)
    source_reports = active_reports if active_reports else all_reports
    
    by_class = assessor.stats

    # Build cells with lat/lon and image links
    cells = []
    # Return top 500 samples for mapping performance
    for r in source_reports[:500]:
        d = r.to_dict()
        
        # Build URLs for the static files
        base_url = "/api/v1/vision/files"
        d["imagery"] = {
            "pre": f"{base_url}/{r.split}/images/{r.img_prefix}_pre_disaster.png",
            "post": f"{base_url}/{r.split}/images/{r.img_prefix}_post_disaster.png"
        }
        cells.append(d)

    return {
        "stage": active_region,
        "total_cells": len(all_reports),
        "active_cells": len(active_reports),
        "by_class": by_class,
        "destroyed": by_class.get("destroyed", 0),
        "major_damage": by_class.get("major_damage", 0),
        "minor_damage": by_class.get("minor_damage", 0),
        "intact": by_class.get("intact", 0),
        "cells": cells,
    }


# ── Silo 3: Infrastructure Status ────────────────────────────────────

@router.get("/silos/infra/status")
async def infra_status():
    """Infrastructure grid health + cascade predictions."""
    grid = app_state.infra_grid
    if not grid:
        return {"error": "Infrastructure grid not initialised"}

    health = grid.network_health()
    cascades = grid.predict_cascades()

    env = app_state.env_monitor
    env_risks = env.compute_risk_overlay() if env else {}

    return {
        "network_health": health,
        "total_nodes": len(grid.nodes),
        "failed_nodes": sum(1 for n in grid.nodes.values() if n.status.value == "failed"),
        "degraded_nodes": sum(1 for n in grid.nodes.values() if n.status.value == "degraded"),
        "cascade_predictions": [c.to_dict() for c in cascades[:10]],
        "nodes": [n.to_dict() for n in list(grid.nodes.values())[:80]],
        "environmental_overlay": env_risks,
    }


@router.get("/silos/infra/traffic")
async def infra_traffic():
    """Real TomTom traffic flow segments for Istanbul road network."""
    import json
    from pathlib import Path

    traffic_path = Path(__file__).resolve().parent.parent.parent / "data" / "real" / "istanbul_traffic.json"
    if not traffic_path.exists():
        return {"segments": [], "total": 0}

    raw = json.loads(traffic_path.read_text(encoding="utf-8"))
    segments = raw.get("segments", [])

    result = []
    for seg in segments:
        coords = seg.get("coordinates", [])
        if len(coords) < 2:
            continue

        # Downsample to max ~60 points per segment for performance
        if len(coords) > 60:
            step = max(1, len(coords) // 60)
            coords = coords[::step]
            if coords[-1] != seg["coordinates"][-1]:
                coords.append(seg["coordinates"][-1])

        current = seg.get("current_speed_kmh", 0)
        freeflow = seg.get("free_flow_speed_kmh", 1)
        congestion = max(0, 1.0 - (current / max(freeflow, 1)))

        result.append({
            "name": seg.get("sample_name", "Unknown"),
            "coords": [[c["latitude"], c["longitude"]] for c in coords],
            "current_speed": current,
            "free_flow_speed": freeflow,
            "congestion": round(congestion, 3),
            "closure": seg.get("road_closure", False),
            "frc": seg.get("frc", ""),
        })

    return {"segments": result, "total": len(result)}
