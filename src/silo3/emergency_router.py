import logging
import time
from enum import Enum, auto
import networkx as nx
import math

logger = logging.getLogger("urbanfl.silo3.routing")

class EmergencyVehicleType(Enum):
    AMBULANCE = auto()
    FIRE_ENGINE = auto()
    POLICE = auto()

class VehicleStatus(Enum):
    AVAILABLE = auto()
    DISPATCHED = auto()
    EN_ROUTE = auto()
    ON_SCENE = auto()

class EmergencyVehicle:
    def __init__(self, vehicle_id: str, vtype: EmergencyVehicleType, nearest_node: int, speed_kmh: float):
        self.vehicle_id = vehicle_id
        self.id = vehicle_id  # alias
        self.type = vtype
        self.nearest_node = nearest_node
        self.speed_kmh = speed_kmh
        self.status = VehicleStatus.AVAILABLE
        self.current_mission = None
        self.lat = 0.0
        self.lon = 0.0
        self.speed_mps = speed_kmh / 3.6


class EmergencyRouter:
    """Road-constrained routing replacing straight-line pathing."""
    def __init__(self, road_graph: nx.MultiDiGraph = None):
        self.graph = road_graph or nx.MultiDiGraph()
        logger.info(f"EmergencyRouter initialized with {len(self.graph.nodes)} nodes.")

    def find_route(self, origin_node: int, dest_node: int) -> dict:
        """Finds road path using OSRM API with networkx fallback."""
        import requests
        
        # 1. Try OSRM API for high-fidelity road routing
        try:
            # Get coordinates for the nodes
            u_node = self.graph.nodes[origin_node]
            v_node = self.graph.nodes[dest_node]
            orig_lon, orig_lat = u_node['x'], u_node['y']
            dest_lon, dest_lat = v_node['x'], v_node['y']
            
            url = f"https://router.project-osrm.org/route/v1/driving/{orig_lon},{orig_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    geom = route["geometry"]["coordinates"]
                    # Convert [[lon, lat], ...] to [{"lat":, "lon":}, ...]
                    path = [{"lat": lat, "lon": lon} for lon, lat in geom]
                    return {
                        "path": path,
                        "distance_m": route.get("distance", 0.0),
                        "duration_s": route.get("duration", 0.0)
                    }
        except Exception as e:
            logger.debug(f"OSRM Routing failed, falling back to networkx: {e}")

        # 2. Local Fallback via networkx
        if not self.graph or origin_node not in self.graph or dest_node not in self.graph: 
            return {"path": [], "distance_m": 0.0}
            
        try:
            path = nx.shortest_path(self.graph, origin_node, dest_node, weight='length')
            distance_m = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.graph[u][v]
                dist = edge_data[0].get('length', 0) if 0 in edge_data else edge_data.get('length', 0)
                distance_m += float(dist)

            route_coords = []
            for n in path:
                if 'y' in self.graph.nodes[n] and 'x' in self.graph.nodes[n]:
                    route_coords.append({"lat": self.graph.nodes[n]['y'], "lon": self.graph.nodes[n]['x']})
                    
            return {"path": route_coords, "distance_m": distance_m}
        except nx.NetworkXNoPath:
            logger.warning(f"No path between {origin_node} and {dest_node}")
            return {"path": [], "distance_m": 0.0}
        except Exception as e:
            logger.debug(f"Local routing fallback error: {e}")
            return {"path": [], "distance_m": 0.0}

class EmergencyDispatcher:
    """Manages the dispatch queue and fleet status."""
    def __init__(self, router: EmergencyRouter):
        self.router = router
        self.vehicles: dict[str, EmergencyVehicle] = {}
        self._vehicles: list[EmergencyVehicle] = [] # List alias for state.py loops
        self._assignments = []

    def register_vehicles(self, vehicles: list[EmergencyVehicle]):
        for v in vehicles:
            self.vehicles[v.vehicle_id] = v
            self._vehicles.append(v)
        logger.info(f"Registered {len(vehicles)} vehicles.")

    def tick(self, delta_time_s: float):
        """Advances all active vehicles along their routes."""
        for assignment in self._assignments:
            v_id = assignment.get("vehicle_id")
            v = self.vehicles.get(v_id)
            if not v: continue
            
            route = assignment.get("route", {})
            path = route.get("path", [])
            if not path: continue
            
            # Progress simulation: moving along points in the path
            # We use index-based progression for simplicity in the 'industry-scale' demo
            # to ensure smooth visual updates.
            if "current_index" not in assignment:
                assignment["current_index"] = 0
                
            # Move speed_kmh * delta_time / dist_between_points
            # Approximation: Advance 1 path point per tick if speed > 0
            if v.status in [VehicleStatus.DISPATCHED, VehicleStatus.EN_ROUTE]:
                if assignment["current_index"] < len(path) - 1:
                    assignment["current_index"] += 1
                    curr_pt = path[assignment["current_index"]]
                    v.nearest_node = None # Coordinate takes precedence
                    v.lat = curr_pt["lat"]
                    v.lon = curr_pt["lon"]
                    v.status = VehicleStatus.EN_ROUTE
                else:
                    v.status = VehicleStatus.ON_SCENE
                    assignment["on_scene_at"] = time.time()

