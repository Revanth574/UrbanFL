import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

from src.event_bus import EventBus, Event, EventType

logger = logging.getLogger("urbanfl.dashboard.monitor")

@dataclass
class Zone:
    zone_id: str
    name: str
    center_lat: float
    center_lon: float
    population: int
    area_km2: float
    threat_level: float = 0.0

@dataclass
class Snapshot:
    threat_grid: List[List[float]] = field(default_factory=list)
    alerts: Dict[str, Any] = field(default_factory=dict)
    aerial: Dict[str, Any] = field(default_factory=dict)
    zones: Dict[str, Any] = field(default_factory=dict)
    active_incidents: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "threat_grid": self.threat_grid,
            "alerts": self.alerts,
            "aerial": self.aerial,
            "zones": self.zones,
            "active_incidents": self.active_incidents,
            "timestamp": self.timestamp
        }

class RealTimeMonitor:
    """
    Maintains a high-performance grid representation of European Istanbul, 
    updating via rapid asynchronous events from the EventBus without blocking.
    """
    def __init__(self, bus: EventBus, grid_rows: int = 7, grid_cols: int = 7, bbox: Dict[str, float] = None, districts: List[Dict] = None):
        self.bus = bus
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # Use default Istanbul bbox if none provided
        self.bbox = bbox or {"lon_min": 28.5, "lat_min": 40.9, "lon_max": 29.05, "lat_max": 41.25}
        
        # Threat matrix: initializes at ~0 severity
        self._threat_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Rolling stats
        self._total_reports = 0
        self._recent_credibility: List[float] = []
        self._active_incidents_count = 0
        
        # Initialize zones from real districts
        self.zones: List[Zone] = []
        if districts:
            for d in districts:
                self.zones.append(Zone(
                    zone_id=d["name"].replace(" ", "_").upper(),
                    name=d["name"],
                    center_lat=d["lat"],
                    center_lon=d["lon"],
                    population=d["population"],
                    area_km2=d["area_km2"]
                ))
        else:
            # Fallback zones if no districts provided
            for i in range(5):
                self.zones.append(Zone(
                    zone_id=f"ZONE_{i+1}",
                    name=f"Sector {i+1}",
                    center_lat=self.bbox["lat_min"] + (self.bbox["lat_max"] - self.bbox["lat_min"]) * np.random.random(),
                    center_lon=self.bbox["lon_min"] + (self.bbox["lon_max"] - self.bbox["lon_min"]) * np.random.random(),
                    population=int(50000 + np.random.random() * 100000),
                    area_km2=10.0
                ))
        
        # Wire into event bus
        self.bus.subscribe(EventType.CITIZEN_REPORT, self._on_citizen_report)
        self.bus.subscribe(EventType.ALERT_VERIFIED, self._on_alert_verified)
        
        logger.info(f"RealTimeMonitor initialized with {grid_rows}x{grid_cols} matrix and {len(self.zones)} zones.")

    def _map_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert lat/lon to grid indices within the configured bounding box."""
        lat_pct = (lat - self.bbox["lat_min"]) / (self.bbox["lat_max"] - self.bbox["lat_min"])
        lon_pct = (lon - self.bbox["lon_min"]) / (self.bbox["lon_max"] - self.bbox["lon_min"])
        
        # Clamp to bounds
        lat_pct = max(0.0, min(0.9999, lat_pct))
        lon_pct = max(0.0, min(0.9999, lon_pct))
        
        row = int((1.0 - lat_pct) * self.grid_rows)  # Flip lat for array indexing (north is up)
        col = int(lon_pct * self.grid_cols)
        
        # Absolute safety bounds
        row = max(0, min(self.grid_rows - 1, row))
        col = max(0, min(self.grid_cols - 1, col))
        return (row, col)

    def _on_citizen_report(self, event: Event):
        """Update threat matrix based on unverified citizen reports."""
        lat = event.data.get("lat")
        lon = event.data.get("lon")
        credibility = event.data.get("credibility", 0.5)
        
        self._total_reports += 1
        self._recent_credibility.append(credibility)
        if len(self._recent_credibility) > 100:
            self._recent_credibility.pop(0)
            
        if lat is not None and lon is not None:
            r, c = self._map_to_grid(lat, lon)
            # Impact is softened because it's unverified
            impact = event.priority * credibility * 0.1
            self._threat_grid[r, c] = min(1.0, self._threat_grid[r, c] + impact)

    def _on_alert_verified(self, event: Event):
        """Massive localized threat increase for verified system alerts."""
        lat = event.data.get("lat")
        lon = event.data.get("lon")
        self._active_incidents_count += 1
        
        if lat is not None and lon is not None:
            r, c = self._map_to_grid(lat, lon)
            impact = event.priority * 0.4
            self._threat_grid[r, c] = min(1.0, self._threat_grid[r, c] + impact)
            
            # Splash damage to surrounding cells (Gaussian drop-off approximation)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        self._threat_grid[nr, nc] = min(1.0, self._threat_grid[nr, nc] + (impact * 0.25))

    def generate_snapshot(self) -> Snapshot:
        """Produce a pure dictionary representation of the current simulation epoch."""
        # Decay threat matrix slightly over time to simulate resolution
        self._threat_grid *= 0.99
        
        # Update zones with current threat levels from grid
        critical = 0
        severe = 0
        for z in self.zones:
            r, c = self._map_to_grid(z.center_lat, z.center_lon)
            z.threat_level = float(self._threat_grid[r, c])
            if z.threat_level > 0.7: critical += 1
            elif z.threat_level > 0.4: severe += 1

        mean_cred = float(np.mean(self._recent_credibility)) if self._recent_credibility else 0.5
        
        snap = Snapshot(
            threat_grid=self._threat_grid.tolist(),
            alerts={
                "mean_credibility": mean_cred,
                "total_processing": self._total_reports,
                "total": self._total_reports, # Alias for Dashboard.jsx
                "escalated": self._active_incidents_count,
                "per_minute": self._total_reports / 5.0, # Dummy rate
            },
            aerial={
                "coverage_pct": min(100, 20 + self._active_incidents_count * 2.5),
                "drones_active": min(50, 5 + self._active_incidents_count)
            },
            zones={
                "total": len(self.zones),
                "critical": critical,
                "severe": severe,
                "moderate": len(self.zones) - critical - severe,
            },
            active_incidents=self._active_incidents_count
        )
        return snap

