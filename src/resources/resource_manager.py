import random

class ResourceNode:
    def __init__(self, id, name, lat, lon):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.supplies_inventory = random.randint(100, 1000)
        self.capacity = random.randint(500, 2000)
        self.occupied = int(self.capacity * random.uniform(0.1, 0.4))

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "supplies": self.supplies_inventory,
            "occupied": self.occupied,
            "capacity": self.capacity
        }

class ResourceManager:
    def __init__(self):
        self.nodes: dict[int, ResourceNode] = {}

    @classmethod
    def create_from_data(cls, shelter_data: list):
        mgr = cls()
        # Limit to 100 for performance
        for s in shelter_data[:100]:
            mgr.nodes[s["id"]] = ResourceNode(
                id=s["id"],
                name=s.get("name", "Shelter"),
                lat=s["latitude"],
                lon=s["longitude"]
            )
        return mgr

    # Compatibility shim
    @classmethod
    def create_default(cls, n_nodes: int, bbox: dict):
        return cls()
    
    def summary(self):
        active = sum(1 for n in self.nodes.values() if n.occupied > 0)
        total_occupied = sum(n.occupied for n in self.nodes.values())
        return {
            "active_centers": active,
            "total_occupied": total_occupied,
            "status": "OPERATIONAL" if active > 0 else "IDLE"
        }
