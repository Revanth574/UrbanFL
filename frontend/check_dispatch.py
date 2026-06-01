"""Debug the dispatch directly."""
import sys
sys.path.insert(0, ".")

import osmnx as ox
import networkx as nx
from src.silo3.emergency_router import EmergencyRouter, EmergencyDispatcher, EmergencyVehicle, EmergencyVehicleType, VehicleStatus

# Load the road graph
import json
from pathlib import Path
road_path = Path("data/real/istanbul_roads.graphml")
if road_path.exists():
    graph = ox.load_graphml(str(road_path))
    print(f"Graph: {len(graph.nodes)} nodes")
else:
    print("No graphml found")
    sys.exit(1)

# Pick a random incident location
import random
incident_lat = 41.0 + random.uniform(-0.05, 0.05)
incident_lon = 28.95 + random.uniform(-0.1, 0.1)
print(f"Incident at: {incident_lat:.4f}, {incident_lon:.4f}")

# Find nearest node
dest_node = ox.nearest_nodes(graph, incident_lon, incident_lat)
print(f"Dest node: {dest_node}")

# Pick origin node
nodes_list = list(graph.nodes)
origin_node = nodes_list[random.randint(0, min(100, len(nodes_list)-1))]
print(f"Origin node: {origin_node}")

# Try routing
router = EmergencyRouter(graph)
try:
    route = router.find_route(origin_node, dest_node)
    path = route.get("path", [])
    print(f"Route: {len(path)} waypoints, distance: {route.get('distance_m',0):.0f}m")
    if path:
        print(f"  First: {path[0]}")
        print(f"  Last: {path[-1]}")
except Exception as e:
    print(f"Route failed: {e}")
    import traceback
    traceback.print_exc()
