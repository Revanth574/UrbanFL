"""Analytics router — risk forecasts, scenario simulation, casualties."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from api.state import app_state

router = APIRouter()


class ScenarioRequest(BaseModel):
    type: str = "earthquake"
    lat: float = 41.05
    lon: float = 29.0
    magnitude: Optional[float] = 6.5
    population: int = 50000
    hours_since: Optional[float] = 0.5
    wind_speed_mps: Optional[float] = 8.0
    wind_direction_deg: Optional[float] = 225.0
    rainfall_mm_hr: Optional[float] = 40.0
    damage_pct: Optional[float] = Field(default=0.3, ge=0, le=1)


@router.get("/analytics/risk-forecast")
async def risk_forecast():
    """Current risk forecasts — aftershock, fire, flood."""
    sim = app_state.scenario_sim
    if not sim:
        return {"error": "Scenario simulator not initialised"}

    aftershock = sim.aftershock.forecast(hours_since=0.5, mainshock_mag=6.5)
    fire = sim.fire.forecast(fire_lat=41.05, fire_lon=29.0)
    flood = sim.flood.predict_flood_extent(center_lat=41.05, center_lon=29.0, hours_ahead=2.0)

    return {
        "aftershock": aftershock,
        "fire": fire,
        "flood": flood,
    }


@router.post("/analytics/scenario")
async def run_scenario(req: ScenarioRequest):
    """Run a what-if scenario simulation."""
    sim = app_state.scenario_sim
    if not sim:
        return {"error": "Scenario simulator not initialised"}

    result = sim.run_scenario(req.model_dump())
    return result


@router.get("/analytics/casualty-estimate")
async def casualty_estimate():
    """Casualty estimates per zone from current damage data."""
    monitor = app_state.monitor
    if not monitor:
        return {"zones": []}

    import random
    zones_out = []
    total_cas = 0
    total_trap = 0
    for z in monitor.zones:
        threat = getattr(z, 'threat_level', 0.0)
        pop = getattr(z, 'population', 50000)
        pop_norm = min(1.0, pop / 500000.0)
        rng = random.Random(hash(z.zone_id) % 1000)

        cas = int(threat * pop_norm * 80 * rng.uniform(0.5, 1.5))
        trap = int(threat * pop_norm * 40 * rng.uniform(0.3, 1.2))

        crit = "P3"
        score = threat * 0.7 + pop_norm * 0.3
        if score > 0.8: crit = "P1"
        elif score > 0.6: crit = "P2"
        elif score > 0.4: crit = "P3"
        else: crit = "P4"

        if cas > 0:
            zones_out.append({
                "zone_id": z.zone_id,
                "estimated_casualties": cas,
                "estimated_trapped": trap,
                "criticality": crit,
            })
            total_cas += cas
            total_trap += trap

    return {
        "zones": zones_out,
        "total_casualties": total_cas,
        "total_trapped": total_trap,
    }
