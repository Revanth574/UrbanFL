"""
UrbanCityOS — Application State Container
==========================================
Thread-safe singleton holding all backend module instances.
Provides a self-sustaining background simulation loop with:
  - Event recycling (never stalls)
  - Dynamic DEFCON / FL round / privacy epsilon
  - Infrastructure & environment ticking
  - Auto-dispatch on critical events
  - Dispatch activity log
"""

import asyncio
import copy
import logging
import math
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import os
from pathlib import Path

logger = logging.getLogger("urbanfl.api.state")

# Resolve project root (two levels up from api/state.py)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


class AppState:
    """
    Central application state. Holds all backend module singletons.
    Initialised once during FastAPI lifespan, shared across all requests.
    """

    def __init__(self):
        # --- Core modules (initialised in boot()) ---
        self.orchestrator = None
        self.command_center = None
        self.monitor = None
        self.event_bus = None
        self.scenario_sim = None
        self.damage_assessor = None
        self.zone_scorer = None
        self.triage_ranker = None
        self.infra_grid = None
        self.hospital_network = None
        self.hospital_feed = None
        self.env_monitor = None
        self.resource_manager = None
        self.data_processor = None

        # --- Real Data Pipeline ---
        self.real_loader = None
        self.emergency_dispatcher = None
        self._pending_events = []
        self._original_events: list = []        # Immutable copy for recycling
        self._recycle_count: int = 0
        self._recent_reports: deque = deque(maxlen=60)

        # --- Dispatch Activity Log ---
        self._dispatch_log: deque = deque(maxlen=100)

        # --- Playback Mode ---
        self._active_disaster_name = "GUATEMALA-VOLCANO"
        self._disaster_playback_index = 0
        self._last_playback_rotation = 0.0
        self._playback_cycle_seconds = 60.0

        # --- Dynamic System Telemetry ---
        self._defcon_level: int = 3             # 1-5 (1 = max threat)
        self._fl_round: int = 0
        self._fl_round_timer: float = 0.0
        self._privacy_epsilon: float = 2.0
        self._active_threat_count: int = 0
        self._silo_health_history: dict = {     # Last 30 values per silo
            "silo1": deque(maxlen=30),
            "silo2": deque(maxlen=30),
            "silo3": deque(maxlen=30),
        }
        self._active_alerts: list = []          # Dynamic infra/env alerts

        # --- Simulation control ---
        self._speed_multiplier: float = 1.0
        self._sim_task: Optional[asyncio.Task] = None
        self._running = False
        self._tick_count: int = 0

        # --- Snapshot history (ring buffer) ---
        self._snapshot_history: deque = deque(maxlen=500)
        self._lock = threading.Lock()

        # --- Connection tracking ---
        self._ws_clients: set = set()
        self._boot_time: float = 0.0
        self._bbox: dict = {}

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self):
        """Initialise all backend modules. Called once at app startup."""
        logger.info("=" * 60)
        logger.info("  UrbanCityOS — Booting REAL DATA backend modules...")
        logger.info("=" * 60)

        self._boot_time = time.time()

        # 1. Load Real Data
        from src.data.real_data_loader import RealDataLoader
        self.real_loader = RealDataLoader(verbose=False)
        self.real_loader.load_all()
        bbox_raw = self.real_loader.get_istanbul_bbox()
        if isinstance(bbox_raw, tuple):
            bbox = {"lon_min": bbox_raw[0], "lat_min": bbox_raw[1], "lon_max": bbox_raw[2], "lat_max": bbox_raw[3]}
        else:
            bbox = bbox_raw
        self._bbox = bbox
        logger.info("  Real data loaded.")

        # Event bus
        from src.event_bus import EventBus, setup_default_subscriptions, Event, EventType
        self.event_bus = EventBus(max_queue_size=5000)
        setup_default_subscriptions(self.event_bus)

        # Orchestrator
        from src.orchestrator import OrchestratorEngine
        self.orchestrator = OrchestratorEngine.create_default()

        # Command Center
        from src.coordination.command_center import CommandCenter
        self.command_center = CommandCenter.create_default("ISTANBUL_EARTHQUAKE")

        # Situation Monitor
        from src.dashboard.situation_monitor import RealTimeMonitor
        pop_data = self.real_loader.get_population_data()
        districts = pop_data.get("districts", []) if isinstance(pop_data, dict) else []
        self.monitor = RealTimeMonitor(
            bus=self.event_bus,
            grid_rows=7, grid_cols=7,
            bbox=bbox,
            districts=districts
        )

        # Damage Assessment (xBD)
        from src.dashboard.damage_assessment import DamageAssessor
        self.damage_assessor = DamageAssessor()
        self.damage_assessor.load_real_data(max_samples=2000)

        # Zone Criticality
        from src.dashboard.zone_criticality import ZoneCriticalityScorer, TriageRanker
        self.zone_scorer = ZoneCriticalityScorer()
        self.triage_ranker = TriageRanker(self.zone_scorer)

        # Analytics
        from src.analytics.risk_forecaster import ScenarioSimulator
        self.scenario_sim = ScenarioSimulator()

        # Infrastructure Grid (Silo 3)
        from src.infrastructure.grid_monitor import InfrastructureGridMonitor
        self.infra_grid = InfrastructureGridMonitor(bbox)
        # Load real Istanbul infrastructure
        infra_path = os.path.join(PROJECT_ROOT, "data", "real", "istanbul_infrastructure.json")
        self.infra_grid.load_real_data(infra_path)
        self.infra_grid.simulate_damage(
            center_lat=(bbox["lat_min"] + bbox["lat_max"]) / 2,
            center_lon=(bbox["lon_min"] + bbox["lon_max"]) / 2,
            radius_km=5.0, severity=0.7
        )

        # Hospital Network
        from src.medical.hospital_network import HospitalNetwork, SimulatedHospitalFeed
        hospital_data = self.real_loader.get_hospitals()
        hospitals_list = hospital_data.get("hospitals", []) if isinstance(hospital_data, dict) else hospital_data
        self.hospital_network = HospitalNetwork.create_from_data(hospitals_list)
        self.hospital_feed = SimulatedHospitalFeed(self.hospital_network)

        # Environmental Monitor
        from src.environment.env_monitor import EnvironmentalMonitor
        self.env_monitor = EnvironmentalMonitor(bbox=bbox)
        # Load real Istanbul air quality
        aqi_path = os.path.join(PROJECT_ROOT, "data", "real", "istanbul_air_quality.json")
        self.env_monitor.load_real_data(aqi_path)

        # Resource Manager (Shelters)
        from src.resources.resource_manager import ResourceManager
        shelter_data = self.real_loader.get_shelters()
        shelters_list = shelter_data.get("shelters", []) if isinstance(shelter_data, dict) else shelter_data
        self.resource_manager = ResourceManager.create_from_data(shelters_list)

        # Command Data Processor
        from src.command.data_processor import CommandDataProcessor
        self.data_processor = CommandDataProcessor()

        # Emergency Router & Dispatcher
        from src.silo3.emergency_router import EmergencyRouter, EmergencyDispatcher, EmergencyVehicle, EmergencyVehicleType
        road_graph = self.real_loader.get_road_graph()
        router = EmergencyRouter(road_graph)
        self.emergency_dispatcher = EmergencyDispatcher(router)
        rng = np.random.RandomState(42)
        graph_nodes = list(road_graph.nodes)
        if graph_nodes:
            vehicles = []
            for i in range(12):
                node = rng.choice(graph_nodes)
                v = EmergencyVehicle(f"AMB_{i}", EmergencyVehicleType.AMBULANCE, node, 35.0)
                v.lat = road_graph.nodes[node].get('y', 41.0)
                v.lon = road_graph.nodes[node].get('x', 29.0)
                vehicles.append(v)
            for i in range(8):
                node = rng.choice(graph_nodes)
                v = EmergencyVehicle(f"FIRE_{i}", EmergencyVehicleType.FIRE_ENGINE, node, 25.0)
                v.lat = road_graph.nodes[node].get('y', 41.0)
                v.lon = road_graph.nodes[node].get('x', 29.0)
                vehicles.append(v)
            for i in range(8):
                node = rng.choice(graph_nodes)
                v = EmergencyVehicle(f"POLICE_{i}", EmergencyVehicleType.POLICE, node, 40.0)
                v.lat = road_graph.nodes[node].get('y', 41.0)
                v.lon = road_graph.nodes[node].get('x', 29.0)
                vehicles.append(v)
            self.emergency_dispatcher.register_vehicles(vehicles)

        # Buffer initial crisis reports and earthquakes for drip feed
        self._build_event_queue()

        # Generate initial snapshot
        snap = self.monitor.generate_snapshot()
        self._snapshot_history.append(snap.to_dict())

        # Initial FL round timer
        self._fl_round_timer = time.time()

        logger.info("  UrbanCityOS — All REAL modules ONLINE ✓")

    def _build_event_queue(self):
        """Build the pending events queue from real data."""
        from src.event_bus import Event, EventType

        events = []

        eq_data = self.real_loader.get_earthquakes()
        earthquakes = eq_data.get("earthquakes", eq_data) if isinstance(eq_data, dict) else eq_data
        for eq in earthquakes[:5]:
            events.append(Event(
                priority=min(1.0, eq["magnitude"] / 7.0),
                timestamp=eq.get("time_ms", time.time() * 1000) / 1000,
                event_type=EventType.ALERT_VERIFIED,
                source_silo=0,
                data={
                    "lat": eq["latitude"], "lon": eq["longitude"],
                    "type": "earthquake", "magnitude": eq["magnitude"],
                    "place": eq["place"], "depth_km": eq["depth_km"],
                }
            ))

        crisis_data = self.real_loader.get_crisis_reports()
        crisis_reports = crisis_data.get("reports", crisis_data) if isinstance(crisis_data, dict) else crisis_data
        for iteration, rep in enumerate(crisis_reports[:200]):
            report = {
                "lat": rep.get("latitude", 41.0 + np.random.uniform(-0.1, 0.1)),
                "lon": rep.get("longitude", 29.0 + np.random.uniform(-0.1, 0.1)),
                "text": rep.get("text", "")[:200],
                "category": rep.get("category", "unknown"),
                "source_type": rep.get("source_type", "twitter"),
                "credibility": rep.get("credibility", 0.6),
            }
            events.append(Event(
                priority=rep.get("severity", 0.5),
                timestamp=rep.get("timestamp", time.time()),
                event_type=EventType.CITIZEN_REPORT,
                source_silo=1,
                data=report,
            ))

        # Store originals for recycling (deep copy)
        self._original_events = [copy.deepcopy(e) for e in events]

        # Reverse so we pop from end
        events.reverse()
        self._pending_events = events

    def _recycle_events(self):
        """Recycle original events with new timestamps and slight coordinate jitter."""
        self._recycle_count += 1
        logger.info(f"  ♻ Recycling events (cycle #{self._recycle_count})")

        recycled = []
        for e in self._original_events:
            ev = copy.deepcopy(e)
            ev.timestamp = time.time()
            # Add slight coordinate jitter so markers don't stack
            if "lat" in ev.data:
                ev.data["lat"] += np.random.uniform(-0.005, 0.005)
            if "lon" in ev.data:
                ev.data["lon"] += np.random.uniform(-0.005, 0.005)
            # Clamp to bbox
            ev.data["lat"] = max(self._bbox.get("lat_min", 40.9),
                                 min(self._bbox.get("lat_max", 41.2), ev.data.get("lat", 41.0)))
            ev.data["lon"] = max(self._bbox.get("lon_min", 28.8),
                                 min(self._bbox.get("lon_max", 29.2), ev.data.get("lon", 29.0)))
            recycled.append(ev)

        recycled.reverse()
        self._pending_events = recycled

    # ------------------------------------------------------------------
    # Simulation Loop
    # ------------------------------------------------------------------

    async def start_simulation(self):
        """Start the background simulation loop."""
        self._running = True
        self._sim_task = asyncio.create_task(self._simulation_loop())
        logger.info("  Simulation loop STARTED (5s tick, drip-feed mode)")

    async def stop_simulation(self):
        """Stop the background simulation loop."""
        self._running = False
        if self._sim_task:
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass

    async def _simulation_loop(self):
        """Background loop: tick simulation, generate snapshots, push to WS."""
        while self._running:
            interval = 5.0 / max(self._speed_multiplier, 0.01)
            if self._speed_multiplier == 0:
                await asyncio.sleep(1.0)
                continue

            try:
                self._tick_count += 1

                # DRIP FEED: Pop 1-3 events and publish them
                with self._lock:
                    # Recycle if exhausted
                    if not self._pending_events:
                        self._recycle_events()

                    num_to_pop = np.random.randint(1, 4)
                    for _ in range(num_to_pop):
                        if self._pending_events:
                            ev = self._pending_events.pop()
                            ev.timestamp = time.time()
                            self.event_bus.publish(ev)

                            from src.event_bus import EventType
                            if ev.event_type == EventType.CITIZEN_REPORT:
                                d = ev.data
                                self._recent_reports.appendleft({
                                    "id": f"rpt_{self._tick_count}_{len(self._pending_events)}",
                                    "time": time.strftime("%H:%M:%S"),
                                    "text": d.get("text", ""),
                                    "severity": "CRITICAL" if ev.priority > 0.7 else "HIGH" if ev.priority > 0.4 else "LOW",
                                    "credibility": round(d.get("credibility", 0.5), 3),
                                    "verified": ev.priority > 0.75,
                                    "location": f"Sector {int(np.random.randint(1, 10))}",
                                    "source": d.get("source_type", "CrisisNLP"),
                                    "type": d.get("category", "earthquake"),
                                    "lat": round(d.get("lat", 41.0), 5),
                                    "lon": round(d.get("lon", 29.0), 5),
                                    "nlpConfidence": round(0.5 + np.random.random() * 0.5, 3),
                                    "envMatch": bool(np.random.random() > 0.5),
                                })

                            # AUTO-DISPATCH on critical events
                            if ev.priority > 0.4 and ev.data.get("lat") and ev.data.get("lon"):
                                self._try_auto_dispatch(ev)

                # --- Playback Mode Rotation ---
                if self.damage_assessor:
                    now = time.time()
                    if now - self._last_playback_rotation >= self._playback_cycle_seconds:
                        disasters = self.damage_assessor.get_disasters()
                        if disasters:
                            self._disaster_playback_index = (self._disaster_playback_index + 1) % len(disasters)
                            self._active_disaster_name = disasters[self._disaster_playback_index]
                            self._last_playback_rotation = now
                            logger.info(f"PLAYBACK ROTATION: Switching to region {self._active_disaster_name}")

                # --- Federated Learning Rounds ---
                now = time.time()
                if self.orchestrator and now - self._fl_round_timer >= 30.0:
                    self.orchestrator.execute_federated_round()
                    self._fl_round = self.orchestrator.current_round
                    self._privacy_epsilon = self.orchestrator.privacy_budget
                    self._fl_round_timer = now

                # Tick hospital and dispatcher simulation
                self.hospital_feed.tick(disaster_active=True, severity=0.5)
                if self.emergency_dispatcher:
                    self.emergency_dispatcher.tick(interval)

                # Force periodic dispatch if none have happened (every 3 ticks)
                if self.emergency_dispatcher and self._tick_count % 3 == 0:
                    if not hasattr(self.emergency_dispatcher, '_assignments') or len(self.emergency_dispatcher._assignments) < 8:
                        self._force_periodic_dispatch()

                # Tick infrastructure (stochastic health drift)
                self._tick_infrastructure()

                # Tick environment
                self._tick_environment()

                # Update dynamic telemetry
                self._update_system_telemetry()

                # Generate snapshot
                snap = self.monitor.generate_snapshot()
                snap_dict = snap.to_dict()

                with self._lock:
                    self._snapshot_history.append(snap_dict)

                # Build broadcast payload with system status
                broadcast_data = {
                    "snapshot": snap_dict,
                    "system_status": self.get_system_status(),
                    "recent_reports": list(self._recent_reports)[:3],
                    "dispatch_summary": self._get_dispatch_summary(),
                }

                # Push to WebSocket
                await self._broadcast_snapshot(broadcast_data)

            except Exception as e:
                logger.error(f"Simulation tick error: {e}", exc_info=True)

            await asyncio.sleep(interval)

    def get_system_status(self) -> dict:
        """Central status metrics for dashboard top-bar and health checks."""
        health = self.orchestrator.get_system_health() if self.orchestrator else {}
        return {
            "defcon": self._defcon_level,
            "fl_round": self._fl_round,
            "epsilon": round(self._privacy_epsilon, 3),
            "threats": self._active_threat_count,
            "hetero_index": round(health.get("hetero_index", 0.0), 3),
            "divergence": round(health.get("divergence", 0.0), 3),
            "active_disaster": self._active_disaster_name,
            "uptime_seconds": round(self.get_uptime_seconds(), 1),
            "events_remaining": len(self._pending_events),
            "events_recycled": self._recycle_count,
            "simulation_speed": self._speed_multiplier,
            "paused": self._speed_multiplier == 0,
            "ws_clients": len(self._ws_clients),
            "tick_count": self._tick_count,
            "total_reports_ingested": self._tick_count * 2,
        }

    def _try_auto_dispatch(self, event):
        """Attempt to auto-dispatch a vehicle to a critical event."""
        try:
            dispatcher = self.emergency_dispatcher
            if not dispatcher:
                return

            graph = self.real_loader.get_road_graph() if self.real_loader else None
            if not graph:
                return

            # Find closest node to event location
            import osmnx as ox
            try:
                dest_node = ox.nearest_nodes(graph, event.data["lon"], event.data["lat"])
            except Exception:
                return

            # Try to dispatch the nearest available vehicle
            from src.silo3.emergency_router import EmergencyVehicleType
            import random

            available = []
            vehicles_iter = dispatcher._vehicles if hasattr(dispatcher, '_vehicles') else (
                list(dispatcher.vehicles.values()) if hasattr(dispatcher, 'vehicles') else []
            )

            for v in vehicles_iter:
                status_str = str(getattr(v, 'status', 'AVAILABLE')).upper()
                if 'AVAILABLE' in status_str or 'IDLE' in status_str:
                    available.append(v)

            if available:
                chosen = random.choice(available[:5])  # pick from first 5 available
                v_id = getattr(chosen, 'vehicle_id', getattr(chosen, 'id', 'UNKNOWN'))
                origin_node = getattr(chosen, 'nearest_node', None)

                if origin_node and dest_node:
                    try:
                        route = dispatcher.router.find_route(origin_node, dest_node)
                        if route:
                            incident_id = f"INC_{self._tick_count}"
                            if not hasattr(dispatcher, '_assignments'):
                                dispatcher._assignments = []
                            dispatcher._assignments.append({
                                "vehicle_id": v_id,
                                "incident_id": incident_id,
                                "incident_type": event.data.get("type", event.data.get("category", "crisis")),
                                "route": route,
                                "assigned_at": time.time(),
                                "severity": event.priority,
                                "casualties_est": max(1, int(event.priority * 20)),
                                "lat": event.data.get("lat"),
                                "lon": event.data.get("lon"),
                            })

                            # Update vehicle status
                            if hasattr(chosen, 'status'):
                                from src.silo3.emergency_router import VehicleStatus
                                chosen.status = VehicleStatus.DISPATCHED

                            if hasattr(chosen, 'current_mission'):
                                chosen.current_mission = incident_id

                            # Log the dispatch
                            self._dispatch_log.appendleft({
                                "time": time.strftime("%H:%M:%S"),
                                "action": "DISPATCH",
                                "vehicle": v_id,
                                "incident": incident_id,
                                "type": event.data.get("type", event.data.get("category", "crisis")),
                                "severity": round(event.priority, 2),
                                "lat": round(event.data.get("lat", 0), 5),
                                "lon": round(event.data.get("lon", 0), 5),
                            })
                            logger.info(f"  🚨 Auto-dispatched {v_id} → {incident_id}")
                    except Exception:
                        pass  # Route not found — skip
        except Exception as e:
            logger.warning(f"Auto-dispatch failed: {e}")

    def _force_periodic_dispatch(self):
        """Force a dispatch to ensure the map always shows active units and incidents."""
        try:
            import random, time
            import osmnx as ox
            from src.silo3.emergency_router import VehicleStatus

            dispatcher = self.emergency_dispatcher
            if not dispatcher:
                return

            graph = self.real_loader.get_road_graph() if self.real_loader else None
            if not graph:
                return

            # Pick a random Istanbul location for the incident
            incident_lat = 41.0 + random.uniform(-0.05, 0.05)
            incident_lon = 28.95 + random.uniform(-0.1, 0.1)
            incident_types = ["structural_collapse", "fire_outbreak", "medical_emergency", "gas_leak", "road_accident", "flood_rescue"]
            inc_type = random.choice(incident_types)
            severity = round(random.uniform(0.4, 0.95), 2)

            try:
                dest_node = ox.nearest_nodes(graph, incident_lon, incident_lat)
            except Exception:
                return

            # Find available vehicles
            vehicles_iter = dispatcher._vehicles if hasattr(dispatcher, '_vehicles') else (
                list(dispatcher.vehicles.values()) if hasattr(dispatcher, 'vehicles') else []
            )
            available = [v for v in vehicles_iter if 'AVAILABLE' in str(getattr(v, 'status', 'AVAILABLE')).upper()]
            if not available:
                return

            chosen = random.choice(available)
            v_id = getattr(chosen, 'vehicle_id', getattr(chosen, 'id', 'UNKNOWN'))
            origin_node = getattr(chosen, 'nearest_node', None)

            if not origin_node or not dest_node:
                return

            try:
                route = dispatcher.router.find_route(origin_node, dest_node)
                if not route or not route.get("path"):
                    return
            except Exception as e:
                logger.warning(f"Forced dispatch route failed: {e}")
                return

            incident_id = f"INC_{self._tick_count}_{random.randint(100,999)}"
            if not hasattr(dispatcher, '_assignments'):
                dispatcher._assignments = []

            dispatcher._assignments.append({
                "vehicle_id": v_id,
                "incident_id": incident_id,
                "incident_type": inc_type,
                "route": route,
                "assigned_at": time.time(),
                "severity": severity,
                "casualties_est": max(1, int(severity * 15)),
                "lat": incident_lat,
                "lon": incident_lon,
            })

            # Update vehicle status
            if hasattr(chosen, 'status'):
                chosen.status = VehicleStatus.DISPATCHED
            if hasattr(chosen, 'current_mission'):
                chosen.current_mission = incident_id

            self._dispatch_log.appendleft({
                "time": time.strftime("%H:%M:%S"),
                "action": "DISPATCH",
                "vehicle": v_id,
                "incident": incident_id,
                "type": inc_type.replace("_", " ").upper(),
                "severity": severity,
                "lat": round(incident_lat, 5),
                "lon": round(incident_lon, 5),
            })
            logger.info(f"  🚨 Forced dispatch: {v_id} → {incident_id} ({inc_type})")
        except Exception as e:
            logger.warning(f"Forced dispatch failed: {e}")

    def _tick_infrastructure(self):
        """Stochastic drift of infrastructure node health + status sync."""
        if not self.infra_grid:
            return
        try:
            from src.infrastructure.grid_monitor import NodeStatus
            for node in self.infra_grid.nodes.values():
                if hasattr(node, 'health_pct'):
                    # Small random walk, biased slightly toward recovery
                    drift = np.random.normal(0.1, 1.5)
                    node.health_pct = max(0, min(100, node.health_pct + drift))
                    # Sync status with health
                    if node.health_pct < 20:
                        node.status = NodeStatus.FAILED
                    elif node.health_pct < 60:
                        node.status = NodeStatus.DEGRADED
                    else:
                        node.status = NodeStatus.ONLINE
        except Exception:
            pass

    def _tick_environment(self):
        """Stochastic drift of environmental readings."""
        if not self.env_monitor:
            return
        try:
            if hasattr(self.env_monitor, '_readings'):
                for key, reading in self.env_monitor._readings.items():
                    if isinstance(reading, (int, float)):
                        self.env_monitor._readings[key] = reading + np.random.normal(0, reading * 0.02)
        except Exception:
            pass

    def _update_system_telemetry(self):
        """Update DEFCON, FL round, epsilon, alerts, silo health history."""
        now = time.time()

        # Count active threats
        critical_reports = sum(1 for r in self._recent_reports if r.get("severity") == "CRITICAL")
        verified_critical = sum(1 for r in self._recent_reports if r.get("severity") == "CRITICAL" and r.get("verified"))
        self._active_threat_count = critical_reports

        # DEFCON calculation
        if verified_critical >= 5:
            self._defcon_level = 1
        elif critical_reports >= 8:
            self._defcon_level = 2
        elif critical_reports >= 3:
            self._defcon_level = 3
        elif critical_reports >= 1:
            self._defcon_level = 4
        else:
            self._defcon_level = 5

        # FL round — increment every ~60 seconds of simulated time
        if now - self._fl_round_timer > (60.0 / max(self._speed_multiplier, 0.1)):
            self._fl_round += 1
            self._fl_round_timer = now
            # Epsilon decreases slightly with each round
            self._privacy_epsilon = max(0.1, self._privacy_epsilon - 0.015)

        # Silo health history
        snap = self._snapshot_history[-1] if self._snapshot_history else {}
        infra_h = self.infra_grid.network_health() if self.infra_grid else {}
        pcts = [v.get("health_pct", 50) for v in infra_h.values()]
        infra_pct = round(sum(pcts) / max(len(pcts), 1), 1) if pcts else 50

        self._silo_health_history["silo1"].append({
            "time": time.strftime("%H:%M:%S"),
            "pct": min(98, 70 + (snap.get("alerts", {}).get("mean_credibility", 0.7) * 40)),
        })
        self._silo_health_history["silo2"].append({
            "time": time.strftime("%H:%M:%S"),
            "pct": min(95, 50 + snap.get("aerial", {}).get("coverage_pct", 50)),
        })
        self._silo_health_history["silo3"].append({
            "time": time.strftime("%H:%M:%S"),
            "pct": infra_pct,
        })

        # Dynamic alerts
        alerts = []
        if infra_pct < 60:
            failed = sum(1 for n in self.infra_grid.nodes.values()
                         if hasattr(n, 'status') and str(n.status.value) == "failed") if self.infra_grid else 0
            alerts.append({
                "id": "infra_degraded",
                "severity": "CRITICAL" if infra_pct < 40 else "WARNING",
                "title": f"INFRA DEGRADED — {failed} NODES FAILED",
                "message": f"Grid health at {infra_pct}%. Cascading failure risk elevated. Visual confirmation required.",
                "source": "SILO 3",
            })

        env_overlay = self.env_monitor.compute_risk_overlay() if self.env_monitor else {}
        if env_overlay.get("max_risk", 0) > 0.6:
            alerts.append({
                "id": "env_risk_high",
                "severity": "WARNING",
                "title": "ENVIRONMENTAL RISK ELEVATED",
                "message": f"{env_overlay.get('high_risk_cells', 0)} high-risk cells detected. AQI deteriorating.",
                "source": "ENV MONITOR",
            })

        if self._active_threat_count > 5:
            alerts.append({
                "id": "threat_surge",
                "severity": "CRITICAL",
                "title": f"THREAT SURGE — {self._active_threat_count} ACTIVE",
                "message": "Multiple critical incidents reported simultaneously. Escalation protocol engaged.",
                "source": "SILO 1 NLP",
            })

        self._active_alerts = alerts



    def _get_dispatch_summary(self) -> dict:
        """Quick summary of dispatch state for WS broadcast."""
        dispatcher = self.emergency_dispatcher
        if not dispatcher:
            return {"total_units": 0, "available": 0, "dispatched": 0, "incidents": 0}

        vehicles_iter = dispatcher._vehicles if hasattr(dispatcher, '_vehicles') else (
            list(dispatcher.vehicles.values()) if hasattr(dispatcher, 'vehicles') else []
        )

        total = 0
        available = 0
        dispatched = 0
        for v in vehicles_iter:
            total += 1
            s = str(getattr(v, 'status', 'AVAILABLE')).upper()
            if 'AVAILABLE' in s or 'IDLE' in s:
                available += 1
            elif 'DISPATCHED' in s or 'EN_ROUTE' in s:
                dispatched += 1

        incidents = len(dispatcher._assignments) if hasattr(dispatcher, '_assignments') else 0

        return {
            "total_units": total,
            "available": available,
            "dispatched": dispatched,
            "on_scene": total - available - dispatched,
            "incidents": incidents,
        }

    async def _broadcast_snapshot(self, payload: dict):
        import json
        dead = set()
        msg = json.dumps({"type": "tick", "data": payload})
        for ws in self._ws_clients.copy():
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed_multiplier

    @speed.setter
    def speed(self, value: float):
        self._speed_multiplier = max(0, min(10, value))

    def get_latest_snapshot(self) -> Optional[dict]:
        with self._lock:
            return self._snapshot_history[-1] if self._snapshot_history else None

    def get_snapshot_history(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._snapshot_history)[-n:]

    def get_uptime_seconds(self) -> float:
        return time.time() - self._boot_time if self._boot_time else 0


    def get_active_alerts(self) -> list:
        """Dynamic system-wide alerts."""
        return self._active_alerts

    def get_silo_health_history(self) -> dict:
        """Last 30 silo health values for mini charts."""
        return {k: list(v) for k, v in self._silo_health_history.items()}

    def register_ws(self, ws):
        self._ws_clients.add(ws)

    def unregister_ws(self, ws):
        self._ws_clients.discard(ws)


app_state = AppState()
