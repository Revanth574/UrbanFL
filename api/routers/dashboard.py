"""Dashboard router — situation snapshot + history."""

from fastapi import APIRouter
from api.state import app_state

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard():
    """
    Returns the latest SituationSnapshot plus enriched data from
    all backend modules — the single endpoint for the main dashboard page.
    """
    snap = app_state.get_latest_snapshot() or {}

    # Enrich with cross-module data
    infra_health = app_state.infra_grid.network_health() if app_state.infra_grid else {}
    hospital_summary = app_state.hospital_network.network_summary() if app_state.hospital_network else {}
    resource_summary = app_state.resource_manager.summary() if app_state.resource_manager else {}
    env_status = app_state.env_monitor.status_summary() if app_state.env_monitor else {}
    command_summary = app_state.command_center.summary() if app_state.command_center else {}

    # Real damage stats from the damage assessor
    damage_stats = app_state.damage_assessor.stats if app_state.damage_assessor else {"destroyed": 0, "major_damage": 0, "minor_damage": 0, "intact": 0}

    # Silo health (live-computed)
    infra_pct = _avg_infra_health(infra_health)
    silo_health = {
        "silo1_nlp": {
            "name": "SILO 1 (NLP)",
            "health_pct": min(98, 70 + (snap.get("alerts", {}).get("mean_credibility", 0.7) * 40)),
            "status": "ONLINE" if snap.get("alerts", {}).get("total", 0) > 0 else "STANDBY",
        },
        "silo2_vision": {
            "name": "SILO 2 (VISION)",
            "health_pct": min(95, 50 + snap.get("aerial", {}).get("coverage_pct", 50)),
            "status": "PROCESSING" if snap.get("aerial", {}).get("coverage_pct", 0) < 80 else "ONLINE",
        },
        "silo3_infra": {
            "name": "SILO 3 (INFRA)",
            "health_pct": infra_pct,
            "status": "DEGRADED" if infra_pct < 60 else "ONLINE",
        },
    }

    return {
        "snapshot": snap,
        "silo_health": silo_health,
        "infrastructure": infra_health,
        "hospitals": hospital_summary,
        "resources": resource_summary,
        "environment": env_status,
        "command": command_summary,
        "damage_stats": damage_stats,
        "system_status": app_state.get_system_status(),
        "active_alerts": app_state.get_active_alerts(),
        "simulation_speed": app_state.speed,
        "uptime_seconds": round(app_state.get_uptime_seconds(), 1),
        "recent_reports_count": len(app_state._recent_reports),
        "pending_events_count": len(app_state._pending_events),
    }


@router.get("/dashboard/history")
async def get_dashboard_history(n: int = 50):
    """Return last N snapshots for chart backfill."""
    return {"snapshots": app_state.get_snapshot_history(n)}


def _avg_infra_health(health: dict) -> float:
    """Compute average health across all infrastructure networks."""
    if not health:
        return 50.0
    pcts = [v.get("health_pct", 50) for v in health.values()]
    return round(sum(pcts) / max(len(pcts), 1), 1)
