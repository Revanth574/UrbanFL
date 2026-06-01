"""Dispatch router — units, incidents, routes, hospitals, activity log."""

import time
import math
import random
import hashlib
from fastapi import APIRouter
from api.state import app_state

router = APIRouter()

EUR_BB = {"lat_min": 40.96, "lat_max": 41.12, "lon_min": 28.82, "lon_max": 29.10}

UNIT_ICONS = {
    "fire_engine": {"label": "FIRE", "icon": "🚒", "color": "#ff003c"},
    "ambulance": {"label": "EMS", "icon": "🚑", "color": "#00ff88"},
    "police": {"label": "POLICE", "icon": "🚓", "color": "#00f3ff"},
}

# ── Seeded incident generation for deterministic but varied data ─────
INCIDENT_TEMPLATES = [
    {"type": "structural_collapse", "severity": 0.85, "casualties": 12},
    {"type": "fire_outbreak", "severity": 0.72, "casualties": 5},
    {"type": "medical_emergency", "severity": 0.65, "casualties": 3},
    {"type": "gas_leak", "severity": 0.78, "casualties": 8},
    {"type": "road_accident", "severity": 0.55, "casualties": 4},
    {"type": "flood_rescue", "severity": 0.80, "casualties": 15},
    {"type": "building_fire", "severity": 0.90, "casualties": 20},
    {"type": "hazmat_spill", "severity": 0.68, "casualties": 6},
]

# Predetermined incident locations in European Istanbul
INCIDENT_LOCATIONS = [
    (41.015, 28.975, "Fatih"),
    (41.030, 28.880, "Bakırköy"),
    (41.045, 28.950, "Beyoğlu"),
    (41.008, 29.030, "Kadıköy"),
    (40.990, 28.870, "Küçükçekmece"),
    (41.060, 28.920, "Şişli"),
    (41.020, 28.840, "Bahçelievler"),
    (41.035, 29.010, "Üsküdar"),
]


def _get_active_assignments():
    """Get assignments from the dispatcher, or return empty list."""
    dispatcher = app_state.emergency_dispatcher
    if not dispatcher:
        return []
    return getattr(dispatcher, '_assignments', [])


def _generate_incidents(tick):
    """Generate deterministic incidents based on current tick."""
    seed = tick // 6  # Change every ~30s (6 ticks × 5s)
    rng = random.Random(seed)
    count = rng.randint(3, 6)  # 3-6 active incidents

    incidents = []
    for i in range(count):
        template = INCIDENT_TEMPLATES[i % len(INCIDENT_TEMPLATES)]
        loc = INCIDENT_LOCATIONS[i % len(INCIDENT_LOCATIONS)]
        # Add small jitter
        lat = loc[0] + rng.uniform(-0.005, 0.005)
        lon = loc[1] + rng.uniform(-0.005, 0.005)
        severity = min(1.0, template["severity"] + rng.uniform(-0.1, 0.1))

        incidents.append({
            "id": f"INC-{seed:04d}-{i+1:02d}",
            "type": template["type"],
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "severity": round(severity, 2),
            "priority": "P1" if severity > 0.7 else "P2" if severity > 0.4 else "P3",
            "status": "active",
            "reported_at": time.strftime("%H:%M:%S", time.localtime(time.time() - rng.randint(60, 600))),
            "units_assigned": rng.randint(1, 3),
            "casualties_est": template["casualties"] + rng.randint(-2, 5),
            "district": loc[2],
        })
    return incidents


def _generate_routes(unit_lat, unit_lon, inc_lat, inc_lon, rng):
    """Generate a realistic curved route between two points."""
    steps = rng.randint(12, 25)
    route = []
    for s in range(steps + 1):
        t = s / steps
        # Add curve via midpoint offset
        mid_lat = (unit_lat + inc_lat) / 2 + rng.uniform(-0.008, 0.008)
        mid_lon = (unit_lon + inc_lon) / 2 + rng.uniform(-0.008, 0.008)
        # Quadratic bezier
        lat = (1-t)**2 * unit_lat + 2*(1-t)*t * mid_lat + t**2 * inc_lat
        lon = (1-t)**2 * unit_lon + 2*(1-t)*t * mid_lon + t**2 * inc_lon
        route.append({"lat": round(lat, 6), "lon": round(lon, 6)})
    return route


@router.get("/dispatch/units")
async def dispatch_units():
    """Active dispatch units with locations and assignments."""
    dispatcher = app_state.emergency_dispatcher
    if not dispatcher:
        return {"units": [], "total": 0}

    units_list = []
    graph = app_state.real_loader.get_road_graph() if app_state.real_loader else None
    tick = app_state._tick_count

    vehicles_iter = dispatcher._vehicles if hasattr(dispatcher, '_vehicles') else (
        list(dispatcher.vehicles.values()) if hasattr(dispatcher, 'vehicles') else []
    )

    # Check if we have real assignments
    real_assignments = _get_active_assignments()

    # Generate synthetic incidents for assignment if no real ones exist
    if not real_assignments:
        incidents = _generate_incidents(tick)
    else:
        incidents = []

    rng = random.Random(tick // 6)  # Stable seed

    for idx, v in enumerate(vehicles_iter):
        v_type = getattr(v, 'vehicle_type', getattr(v, 'type', None))
        v_id = getattr(v, 'vehicle_id', getattr(v, 'id', 'UNKNOWN'))
        v_type_str = v_type.name.lower() if hasattr(v_type, 'name') else str(v_type)
        ui_cfg = UNIT_ICONS.get(v_type_str, {"label": "UNIT", "icon": "⛑️", "color": "#ffb000"})

        # Extract coordinates
        lat = getattr(v, 'lat', 0.0)
        lon = getattr(v, 'lon', 0.0)
        if (lat == 0.0 or lat == 41.0) and (lon == 0.0 or lon == 29.0):
            nearest = getattr(v, 'nearest_node', None)
            if graph and nearest and nearest in graph.nodes:
                nd = graph.nodes[nearest]
                lat = nd.get('y', 41.0)
                lon = nd.get('x', 29.0)

        status = "available"
        mission_id = None
        eta = None
        route_waypoints = []

        # Check real assignments first
        if real_assignments:
            assignment = next((a for a in real_assignments if a.get("vehicle_id") == v_id), None)
            if assignment:
                mission_id = assignment.get("incident_id")
                status = "en_route"
                route = assignment.get("route", {})
                raw_path = route.get("path", []) if isinstance(route, dict) else []
                duration_s = route.get("duration_s", 0) if isinstance(route, dict) else 0
                distance_m = route.get("distance_m", 0) if isinstance(route, dict) else 0

                for pt in raw_path:
                    if isinstance(pt, dict):
                        route_waypoints.append({"lat": pt.get("lat", 41.0), "lon": pt.get("lon", 29.0)})
                    elif graph and pt in graph.nodes:
                        nd = graph.nodes[pt]
                        route_waypoints.append({"lat": nd.get('y', 41.0), "lon": nd.get('x', 29.0)})

                if len(route_waypoints) > 80:
                    step = max(1, len(route_waypoints) // 80)
                    route_waypoints = route_waypoints[::step]

                if duration_s > 0:
                    eta = round(duration_s / 60.0, 1)
                elif distance_m > 0:
                    speed_mps = getattr(v, 'speed_mps', 15.0)
                    eta = round(distance_m / speed_mps / 60.0, 1) if speed_mps > 0 else None

        # Synthetic assignment for units that should be dispatched
        elif incidents and idx < len(incidents) + len(incidents) // 2:
            inc_idx = idx % len(incidents)
            if idx < len(incidents):
                # Primary responder
                inc = incidents[inc_idx]
                mission_id = inc["id"]
                status = "en_route" if rng.random() > 0.3 else "on_scene"
                route_waypoints = _generate_routes(lat, lon, inc["lat"], inc["lon"], rng)

                # Simulate partial progress along route
                progress = rng.uniform(0.2, 0.9)
                progress_idx = int(len(route_waypoints) * progress)
                if progress_idx < len(route_waypoints):
                    lat = route_waypoints[progress_idx]["lat"]
                    lon = route_waypoints[progress_idx]["lon"]

                dist_km = math.sqrt((lat - inc["lat"])**2 + (lon - inc["lon"])**2) * 111
                speed_kmh = getattr(v, 'speed_kmh', 35.0) if hasattr(v, 'speed_kmh') else 35.0
                eta = round(dist_km / max(speed_kmh, 1) * 60, 1)

        units_list.append({
            "id": v_id,
            "callsign": f"{ui_cfg['label']}-{v_id.split('_')[-1]}",
            "type": v_type_str,
            "icon": ui_cfg["icon"],
            "color": ui_cfg["color"],
            "lat": lat,
            "lon": lon,
            "status": status,
            "assigned_incident": mission_id,
            "speed_kmh": round(getattr(v, 'speed_mps', 15.0) * 3.6, 1),
            "eta_minutes": eta,
            "route": route_waypoints,
        })

    return {"units": units_list, "total": len(units_list)}


@router.get("/dispatch/incidents")
async def dispatch_incidents():
    """Active incidents with geospatial data."""
    # Check real assignments first
    real_assignments = _get_active_assignments()

    if real_assignments:
        dispatcher = app_state.emergency_dispatcher
        graph = app_state.real_loader.get_road_graph() if app_state.real_loader else None
        incidents = []
        seen = set()
        for a in real_assignments:
            inc_id = a.get("incident_id")
            if inc_id in seen:
                continue
            seen.add(inc_id)
            lat = a.get("lat", 41.0)
            lon = a.get("lon", 29.0)
            if lat == 41.0 and lon == 29.0:
                route = a.get("route", {})
                path = route.get("path", []) if isinstance(route, dict) else []
                if path and isinstance(path[-1], dict):
                    lat = path[-1].get("lat", 41.0)
                    lon = path[-1].get("lon", 29.0)
                elif graph and path and path[-1] in graph.nodes:
                    nd = graph.nodes[path[-1]]
                    lat = nd.get('y', 41.0)
                    lon = nd.get('x', 29.0)

            severity = a.get("severity", 0.5)
            incidents.append({
                "id": inc_id,
                "type": a.get("incident_type", "unknown"),
                "lat": lat, "lon": lon,
                "severity": round(severity, 2),
                "priority": "P1" if severity > 0.7 else "P2" if severity > 0.4 else "P3",
                "status": "active",
                "reported_at": time.strftime("%H:%M:%S", time.localtime(a.get("assigned_at", time.time()))),
                "units_assigned": sum(1 for x in real_assignments if x.get("incident_id") == inc_id),
                "casualties_est": a.get("casualties_est", 5),
            })
        return {"incidents": incidents, "total": len(incidents)}

    # Synthetic incidents
    tick = app_state._tick_count
    incidents = _generate_incidents(tick)
    return {"incidents": incidents, "total": len(incidents)}


@router.get("/dispatch/hospitals")
async def dispatch_hospitals():
    """Hospital network status with per-hospital details."""
    hn = app_state.hospital_network
    if not hn:
        return {"hospitals": [], "total": 0}

    summary = hn.network_summary()
    hospitals = []

    if hasattr(hn, 'hospitals'):
        for h in hn.hospitals.values() if isinstance(hn.hospitals, dict) else hn.hospitals:
            h_id = getattr(h, 'hospital_id', getattr(h, 'id', 'H-?'))
            hospitals.append({
                "id": h_id,
                "name": getattr(h, 'name', h_id),
                "lat": getattr(h, 'lat', 41.0),
                "lon": getattr(h, 'lon', 29.0),
                "beds": getattr(h, 'total_beds', getattr(h, 'beds', 100)),
                "utilized": getattr(h, 'occupied_beds', getattr(h, 'utilized', 50)),
                "er": getattr(h, 'er_patients', 0),
                "er_capacity": getattr(h, 'er_capacity', 15),
                "status": "FULL" if getattr(h, 'occupied_beds', 0) / max(getattr(h, 'total_beds', 1), 1) > 0.95 else "ACCEPTING",
            })

    return {"hospitals": hospitals, "total": len(hospitals), "summary": summary}


@router.get("/dispatch/activity-log")
async def dispatch_activity_log():
    """Last 20 dispatch actions (assignments, arrivals, completions)."""
    real_log = list(app_state._dispatch_log)[:20]

    if real_log:
        return {"log": real_log, "total": len(real_log)}

    # Generate synthetic activity log from current incidents
    tick = app_state._tick_count
    incidents = _generate_incidents(tick)
    rng = random.Random(tick // 6)

    dispatcher = app_state.emergency_dispatcher
    vehicles = []
    if dispatcher:
        vehicles = dispatcher._vehicles if hasattr(dispatcher, '_vehicles') else (
            list(dispatcher.vehicles.values()) if hasattr(dispatcher, 'vehicles') else []
        )

    log = []
    actions = ["DISPATCH", "EN_ROUTE", "ON_SCENE", "DISPATCH", "EN_ROUTE"]
    now = time.time()

    for i, inc in enumerate(incidents):
        for j in range(rng.randint(1, 2)):
            v_idx = (i * 2 + j) % max(len(vehicles), 1)
            v_id = getattr(vehicles[v_idx], 'vehicle_id', f"UNIT_{v_idx}") if v_idx < len(vehicles) else f"UNIT_{v_idx}"
            action = actions[(i + j) % len(actions)]
            t = now - rng.randint(10, 300)
            log.append({
                "time": time.strftime("%H:%M:%S", time.localtime(t)),
                "action": action,
                "vehicle": v_id,
                "incident": inc["id"],
                "type": inc["type"].replace("_", " ").upper(),
                "severity": inc["severity"],
                "lat": inc["lat"],
                "lon": inc["lon"],
            })

    log.sort(key=lambda x: x["time"], reverse=True)
    return {"log": log[:20], "total": len(log)}


@router.post("/dispatch/run")
async def run_pipeline():
    """Trigger an FL round advancement."""
    engine = app_state.orchestrator
    if not engine:
        return {"error": "Orchestrator not initialised"}

    try:
        engine.advance_round()
        health = engine.get_system_health()
        return {
            "success": True,
            "round": health.get("round", 0),
            "epsilon": health.get("epsilon", 0),
            "divergence": health.get("divergence", 0),
            "hetero_index": health.get("hetero_index", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/dispatch/routes")
async def dispatch_routes():
    """Pre-cooked OSRM route geometries (summary — full data via static file)."""
    return {
        "note": "Full route data available via /osrmPrecooked.js static file",
        "active_routes": len(app_state.emergency_dispatcher._assignments) if (
            app_state.emergency_dispatcher and hasattr(app_state.emergency_dispatcher, '_assignments')
        ) else 0,
        "bounding_box": EUR_BB,
    }
