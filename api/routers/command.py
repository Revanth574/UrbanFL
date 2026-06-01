"""Command center router — SITREP, missions, agencies, decisions."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from api.state import app_state

router = APIRouter()


class MissionCreate(BaseModel):
    mission_type: str = "rescue"
    zone_id: str = "zone_2_2"
    objective: str = ""
    priority: int = 1
    agencies: Optional[List[str]] = None


@router.get("/command/sitrep")
async def get_sitrep():
    """Generate latest SITREP."""
    cc = app_state.command_center
    if not cc:
        return {"error": "Command center not initialised"}

    snap = app_state.get_latest_snapshot()
    return cc.generate_sitrep(situation_data=snap)


@router.get("/command/missions")
async def get_missions():
    """Active missions."""
    cc = app_state.command_center
    if not cc:
        return {"missions": []}

    active = cc.get_active_missions()
    all_missions = list(cc._missions.values())
    return {
        "active": [m.to_dict() for m in active],
        "all": [m.to_dict() for m in all_missions],
        "total": len(all_missions),
        "active_count": len(active),
    }


@router.post("/command/missions")
async def create_mission(req: MissionCreate):
    """Create and start a new mission."""
    cc = app_state.command_center
    if not cc:
        return {"error": "Command center not initialised"}

    mission = cc.create_mission(
        mission_type=req.mission_type,
        zone_id=req.zone_id,
        objective=req.objective,
        priority=req.priority,
        agencies=req.agencies,
    )
    cc.start_mission(mission.mission_id)
    return {"mission": mission.to_dict(), "status": "created_and_started"}


@router.get("/command/agencies")
async def get_agencies():
    """Registered agencies and resources."""
    cc = app_state.command_center
    if not cc:
        return {"agencies": []}

    return {
        "agencies": [
            {
                "id": a.agency_id,
                "name": a.name,
                "type": a.agency_type.value,
                "resources": a.resources_available,
                "active": a.active,
            }
            for a in cc._agencies.values()
        ],
        "total": len(cc._agencies),
    }


@router.get("/command/decisions")
async def get_decisions():
    """Decision audit log."""
    cc = app_state.command_center
    if not cc:
        return {"decisions": []}

    return {
        "decisions": [
            {
                "id": d.decision_id,
                "decision": d.decision,
                "rationale": d.rationale,
                "made_by": d.made_by,
                "timestamp": d.timestamp,
                "missions": d.related_missions,
            }
            for d in cc._decisions
        ],
        "total": len(cc._decisions),
    }


@router.get("/command/hospitals")
async def get_hospitals():
    """Hospital network status."""
    hn = app_state.hospital_network
    if not hn:
        return {"error": "Hospital network not initialised"}
    return hn.network_summary()


@router.get("/command/resources")
async def get_resources():
    """Emergency resource status."""
    rm = app_state.resource_manager
    if not rm:
        return {"error": "Resource manager not initialised"}
    return rm.summary()
