import logging
from enum import Enum

class Priority(Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"

class Assessment:
    def __init__(self, zone):
        self.zone_id = zone.zone_id
        # Calculate criticality score (0-1)
        # Weight population (density normalized) and local threat
        pop_weight = min(1.0, zone.population / 500000.0) 
        score = (zone.threat_level * 0.7) + (pop_weight * 0.3)
        
        if score > 0.8: self.criticality = Priority.P1
        elif score > 0.6: self.criticality = Priority.P2
        elif score > 0.4: self.criticality = Priority.P3
        elif score > 0.2: self.criticality = Priority.P4
        else: self.criticality = Priority.P5

    def to_dict(self):
        return {"zone_id": self.zone_id, "criticality": self.criticality.value}

class ZoneCriticalityScorer:
    def score_zone(self, zone):
        return Assessment(zone)

class TriageRanker:
    def __init__(self, scorer=None):
        self.scorer = scorer or ZoneCriticalityScorer()

    def rank_zones(self, zones):
        assessments = [self.scorer.score_zone(z) for z in zones]
        # Sort by priority (P1 first)
        return sorted(assessments, key=lambda x: x.criticality.value)

    def get_deployment_order(self, assessments):
        # Assessments are already sorted
        return [{"zone_id": a.zone_id, "order": i+1} for i, a in enumerate(assessments)]
