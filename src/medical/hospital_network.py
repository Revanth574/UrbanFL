import random

class Hospital:
    def __init__(self, id, name, lat, lon, emergency=False):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.has_emergency = emergency
        self.total_beds = random.randint(50, 300)
        self.utilized_beds = int(self.total_beds * random.uniform(0.4, 0.7))
        self.er_capacity = random.randint(10, 40)
        self.er_utilized = int(self.er_capacity * random.uniform(0.2, 0.5))

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "beds": self.total_beds,
            "utilized": self.utilized_beds,
            "er": self.er_utilized,
            "er_capacity": self.er_capacity,
            "status": "FULL" if self.utilized_beds >= self.total_beds else "NORMAL"
        }

class HospitalNetwork:
    def __init__(self):
        self.hospitals: dict[int, Hospital] = {}

    @classmethod
    def create_from_data(cls, hospital_data: list):
        net = cls()
        for h in hospital_data:
            net.hospitals[h["id"]] = Hospital(
                id=h["id"],
                name=h.get("name", "Unknown"),
                lat=h["latitude"],
                lon=h["longitude"],
                emergency=(h.get("emergency") == "yes")
            )
        return net

    # Compatibility shim
    @classmethod
    def create_default_network(cls, count: int, bbox: dict):
        return cls()

    def network_summary(self):
        total_beds = sum(h.total_beds for h in self.hospitals.values())
        utilized = sum(h.utilized_beds for h in self.hospitals.values())
        return {
            "total_beds": total_beds,
            "occupied_beds": utilized,
            "available_beds": total_beds - utilized,
            "overall_status": "NORMAL" if (utilized / total_beds) < 0.8 else "CRITICAL"
        }

class SimulatedHospitalFeed:
    def __init__(self, network: HospitalNetwork):
        self.network = network
        
    def tick(self, disaster_active: bool, severity: float):
        """Simulate dynamic utilization changes during an event."""
        for h in self.network.hospitals.values():
            if disaster_active:
                # Inflow based on severity
                change = int(h.total_beds * severity * 0.05 * random.uniform(0, 1))
                h.utilized_beds = min(h.total_beds, h.utilized_beds + change)
                h.er_utilized = min(h.er_capacity, h.er_utilized + (change // 2))
            else:
                # Gradual discharge
                if random.random() > 0.7:
                    h.utilized_beds = max(int(h.total_beds * 0.4), h.utilized_beds - 1)
