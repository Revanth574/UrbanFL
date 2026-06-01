"""System router — DEFCON, FL round, epsilon, alerts, telemetry."""

from fastapi import APIRouter
from api.state import app_state

router = APIRouter()


@router.get("/system/status")
async def system_status():
    """Dynamic system telemetry — DEFCON, FL round, privacy epsilon, etc."""
    return app_state.get_system_status()


@router.get("/system/alerts")
async def system_alerts():
    """Active system-wide alerts (infra, env, threat surges)."""
    return {"alerts": app_state.get_active_alerts()}


@router.get("/system/silo-health-history")
async def silo_health_history():
    """Last 30 silo health readings for mini-charts."""
    return app_state.get_silo_health_history()
