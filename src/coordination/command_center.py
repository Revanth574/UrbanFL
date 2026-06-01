class CommandCenter:
    def __init__(self):
        self._missions = {}
        self._agencies = {}
        self._decisions = []

    @classmethod
    def create_default(cls, name: str):
        return cls()
        
    def summary(self):
        return {"active_missions": 0, "agencies": 0}
        
    def generate_sitrep(self, situation_data):
        return {"status": "NORMAL"}
        
    def get_active_missions(self):
        return []
        
    def create_mission(self, **kwargs):
        class Mission:
            def __init__(self):
                self.mission_id = "M-1"
            def to_dict(self):
                return {"mission_id": self.mission_id}
        return Mission()
        
    def start_mission(self, mid):
        pass
