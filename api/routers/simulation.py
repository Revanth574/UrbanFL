"""Simulation control router — pause, speed, reset."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from api.state import app_state

router = APIRouter()


class SpeedRequest(BaseModel):
    speed: float = Field(ge=0, le=10, default=1.0, description="Simulation speed multiplier (0=paused, 1=normal, 2/5/10=fast)")


@router.get("/simulation/status")
async def simulation_status():
    """Current simulation state with event pipeline telemetry."""
    return {
        "speed": app_state.speed,
        "paused": app_state.speed == 0,
        "running": app_state._running,
        "uptime_seconds": round(app_state.get_uptime_seconds(), 1),
        "snapshots_generated": len(app_state._snapshot_history),
        "ws_clients": len(app_state._ws_clients),
        "tick_count": app_state._tick_count,
        "events_remaining": len(app_state._pending_events),
        "events_recycled": app_state._recycle_count,
        "total_original_events": len(app_state._original_events),
    }


@router.post("/simulation/speed")
async def set_speed(req: SpeedRequest):
    """Set simulation speed (0=paused, 1=normal, 2/5=fast forward)."""
    app_state.speed = req.speed
    return {
        "speed": app_state.speed,
        "paused": app_state.speed == 0,
    }


@router.post("/simulation/pause")
async def pause_simulation():
    """Pause the simulation."""
    app_state.speed = 0
    return {"speed": 0, "paused": True}


@router.post("/simulation/resume")
async def resume_simulation():
    """Resume at normal speed."""
    app_state.speed = 1.0
    return {"speed": 1.0, "paused": False}
