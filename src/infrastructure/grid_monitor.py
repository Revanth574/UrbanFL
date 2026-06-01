import logging
import random
from enum import Enum
import networkx as nx

logger = logging.getLogger("urbanfl.infrastructure")

class NodeStatus(Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    FAILED = "failed"

class InfrastructureNode:
    def __init__(self, node_id: str, lat: float, lon: float):
        self.node_id = node_id
        self.lat = lat
        self.lon = lon
        self.health_pct = 100.0
        self.status = NodeStatus.ONLINE
        self.type = "substation"

    def to_dict(self):
        return {
            "id": self.node_id,
            "node_id": self.node_id,
            "lat": self.lat,
            "lon": self.lon,
            "health_pct": round(self.health_pct, 1),
            "status": self.status.value,
            "type": getattr(self, 'type', 'general'),
            "node_type": getattr(self, 'type', 'general'),
        }

class InfrastructureGridMonitor:
    """Monitors the health of key infrastructure nodes (power supply, comms)."""
    def __init__(self, bbox: dict):
        self.bbox = bbox
        self.nodes = {}

    def load_real_data(self, json_path: str):
        """Loads real Istanbul power infrastructure from JSON."""
        import json
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            nodes_added = 0
            power_nodes = data.get("infrastructure", {}).get("power", [])
            for p in power_nodes:
                nid = str(p.get("id"))
                lat, lon = p.get("latitude"), p.get("longitude")
                if lat and lon:
                    node = InfrastructureNode(nid, lat, lon)
                    node.type = p.get("type", "substation")
                    self.nodes[nid] = node
                    nodes_added += 1
            
            logger.info(f"Initialized {nodes_added} real infrastructure nodes from {json_path}.")
        except Exception as e:
            logger.error(f"Failed to load real infrastructure: {e}")

    @classmethod
    def create_synthetic(cls, nodes_per_network: int, bbox: dict) -> "InfrastructureGridMonitor":
        monitor = cls(bbox)
        # Populate synthetic nodes across the bounding box
        lon_span = bbox["lon_max"] - bbox["lon_min"]
        lat_span = bbox["lat_max"] - bbox["lat_min"]
        
        for i in range(nodes_per_network):
            lat = bbox["lat_min"] + random.random() * lat_span
            lon = bbox["lon_min"] + random.random() * lon_span
            node = InfrastructureNode(f"infra_node_{i}", lat, lon)
            monitor.nodes[node.node_id] = node
            
        logger.info(f"Initialized {nodes_per_network} infrastructure nodes.")
        return monitor

    def simulate_damage(self, center_lat: float, center_lon: float, radius_km: float, severity: float):
        """Reduces health of nodes near the epicentre."""
        # Roughly 1 deg lat/lon ~ 111 km
        deg_radius = radius_km / 111.0 
        
        for node in self.nodes.values():
            dist = ((node.lat - center_lat)**2 + (node.lon - center_lon)**2)**0.5
            if dist <= deg_radius:
                # Severity falloff
                impact = severity * (1.0 - (dist / deg_radius))
                node.health_pct = max(0.0, node.health_pct - (impact * 100))

    def network_health(self) -> dict:
        return {nid: {"health_pct": node.health_pct, "status": node.status.value} for nid, node in self.nodes.items()}

    def predict_cascades(self) -> list:
        """Simulates potential failure cascades based on network proximity."""
        cascades = []
        failed_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.FAILED or n.health_pct < 20]
        
        for failed in failed_nodes:
            # Find neighbors (synthetic for now, based on distance)
            for other in self.nodes.values():
                if other.node_id == failed.node_id: continue
                dist = ((other.lat - failed.lat)**2 + (other.lon - failed.lon)**2)**0.5
                if dist < 0.05: # ~5km
                    from dataclasses import dataclass
                    @dataclass
                    class Cascade:
                        source_id: str
                        target_id: str
                        probability: float
                        def to_dict(self):
                            return {"source_id": self.source_id, "target_id": self.target_id, "probability": self.probability}
                    
                    cascades.append(Cascade(failed.node_id, other.node_id, 0.6 * (1.1 - dist/0.05)))
        return cascades

